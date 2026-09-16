"""Adapters between tau2 simulations and EverOS memory.

Ingestion: a finished tau2 call becomes one EverOS session (uniform operation
ledger). Retrieval: agent skills/cases become a `<learned_experience>` prompt
block. Both directions are pure functions over plain dicts, easy to audit.
"""

from __future__ import annotations

import json
from typing import Any

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task

from voicedemo.config import AGENT_ID

MESSAGE_STEP_MS = 15_000
MAX_MESSAGES = 480  # EverOS /add caps at 500 per call
# Initialization actions that describe who/where the customer is, not a fault.
_SETUP_FUNCS = {"set_user_info", "set_user_location"}


def _tool_calls_dto(tool_calls: list[Any], *, name_prefix: str = "") -> list[dict]:
    return [
        {
            "id": tc.id or f"call_{i}",
            "type": "function",
            "function": {
                "name": f"{name_prefix}{tc.name}",
                "arguments": json.dumps(tc.arguments, default=str),
            },
        }
        for i, tc in enumerate(tool_calls)
    ]


def build_memorize_messages(
    sim: SimulationRun,
    *,
    customer_id: str,
    base_ts_ms: int,
    qa_note: str,
) -> list[dict[str, Any]]:
    """Render a tau2 trajectory as an EverOS memorize payload.

    Representation choice (documented in PLAN.md): guided device actions —
    tool calls the customer performs on their own phone at the agent's
    direction — are recorded as assistant-attributed tool rounds with a
    ``device_`` name prefix. In a production voice stack these are operations
    the agent drives (device diagnostics API / guided flow), and EverOS's
    case extractor gates on assistant tool rounds, so the uniform ledger is
    both faithful and extractable.
    """
    out: list[dict[str, Any]] = []

    def emit(msg: dict[str, Any]) -> None:
        msg["timestamp"] = base_ts_ms + len(out) * MESSAGE_STEP_MS
        out.append(msg)

    for m in sim.messages or []:
        role = getattr(m, "role", None)
        if role == "assistant":
            content = m.content if isinstance(m.content, str) else ""
            if m.tool_calls:
                emit(
                    {
                        "sender_id": AGENT_ID,
                        "role": "assistant",
                        "content": content,
                        "tool_calls": _tool_calls_dto(m.tool_calls),
                    }
                )
            elif content:
                emit({"sender_id": AGENT_ID, "role": "assistant", "content": content})
        elif role == "user":
            content = m.content if isinstance(m.content, str) else ""
            if getattr(m, "tool_calls", None):
                emit(
                    {
                        "sender_id": AGENT_ID,
                        "role": "assistant",
                        "content": content
                        or "(guides the customer through a device action)",
                        "tool_calls": _tool_calls_dto(
                            m.tool_calls, name_prefix="device_"
                        ),
                    }
                )
            elif content:
                emit({"sender_id": customer_id, "role": "user", "content": content})
        elif role == "tool":
            emit(
                {
                    "sender_id": AGENT_ID,
                    "role": "tool",
                    "content": str(m.content) if m.content is not None else "",
                    "tool_call_id": m.id,
                }
            )
        # system messages (if any) are skipped: policy is static context

    emit({"sender_id": AGENT_ID, "role": "assistant", "content": qa_note})

    if len(out) > MAX_MESSAGES:
        # Keep the head (problem statement) and tail (resolution + QA note).
        head, tail = out[:40], out[-(MAX_MESSAGES - 40):]
        out = head + tail
        for i, msg in enumerate(out):
            msg["timestamp"] = base_ts_ms + i * MESSAGE_STEP_MS
    return out


def describe_action(action: Any) -> str:
    args = action.arguments if isinstance(action.arguments, dict) else {}
    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{action.requestor}.{action.name}({arg_str})"


def build_qa_note(sim: SimulationRun, task: Task, *, resolved: bool, transferred: bool) -> str:
    """Factual post-call note appended to the call log before memorization.

    Successful calls get a plain confirmation (no ground truth). Failed or
    transferred calls get the tier-2 specialist's ticket-closure note built
    from the task's ground truth — the same correction signal a real call
    center records when an escalation is resolved by a human.
    """
    if resolved:
        return (
            "Post-call QA note: issue confirmed resolved during the call; "
            "environment checks passed."
        )

    lines = ["Tier-2 specialist resolution note (ticket closed after escalation):"]

    faults = [
        a
        for a in (task.initial_state.initialization_actions or [])
        if a.func_name not in _SETUP_FUNCS
    ]
    if faults:
        lines.append("Root state found on inspection:")
        for a in faults:
            args = ", ".join(f"{k}={v!r}" for k, v in (a.arguments or {}).items())
            lines.append(f"- {a.env_type}.{a.func_name}({args})")

    ref_actions = getattr(task.evaluation_criteria, "actions", None) or []
    if ref_actions:
        lines.append("Resolution steps that fixed it:")
        for a in ref_actions:
            lines.append(f"- {describe_action(a)}")

    assertions = getattr(task.evaluation_criteria, "env_assertions", None) or []
    if assertions:
        lines.append("Verified end state:")
        for a in assertions:
            args = ", ".join(f"{k}={v!r}" for k, v in (a.arguments or {}).items())
            lines.append(f"- {a.env_type}.{a.func_name}({args})")

    if transferred:
        lines.append(
            "Handling note: the call was transferred to a human agent before "
            "resolution; the steps above are what ultimately resolved it."
        )
    else:
        lines.append(
            "Handling note: the call ended unresolved; the steps above are "
            "what ultimately resolved it."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Retrieval → prompt block
# ---------------------------------------------------------------------------

_MAX_SKILL_CHARS = 1_200
_MAX_CASE_APPROACH_CHARS = 500
_MAX_BLOCK_CHARS = 7_000


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_experience_block(search_response: dict[str, Any]) -> str | None:
    """Render an EverOS agent search response into prompt text (or None)."""
    data = search_response.get("data", {})
    skills = data.get("agent_skills") or []
    cases = data.get("agent_cases") or []
    if not skills and not cases:
        return None

    parts: list[str] = []
    if skills:
        parts.append("## Learned procedures (distilled from your past calls)")
        for s in skills:
            header = f"### {s.get('name', 'skill')}"
            conf = s.get("confidence")
            if conf is not None:
                header += f" (confidence {float(conf):.2f})"
            body = "\n".join(
                x for x in (s.get("description", ""), s.get("content", "")) if x
            )
            parts.append(f"{header}\n{_clip(body, _MAX_SKILL_CHARS)}")
    if cases:
        parts.append("## Similar past calls")
        for c in cases:
            quality = c.get("quality_score")
            quality_s = f"{float(quality):.1f}" if quality is not None else "n/a"
            parts.append(
                "- Intent: {intent}\n  Approach: {approach}\n  Key insight: "
                "{insight}\n  Outcome quality: {q}".format(
                    intent=_clip(str(c.get("task_intent", "")), 200),
                    approach=_clip(
                        str(c.get("approach", "")), _MAX_CASE_APPROACH_CHARS
                    ),
                    insight=_clip(str(c.get("key_insight", "")), 200),
                    q=quality_s,
                )
            )
    return _clip("\n\n".join(parts), _MAX_BLOCK_CHARS)
