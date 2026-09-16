"""Per-call metrics over a tau2 SimulationRun + the voice handle-time model."""

from __future__ import annotations

import re
from typing import Any

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task

from voicedemo.config import (
    CARRIER_TOOL_SECONDS,
    DEVICE_ACTION_SECONDS,
    TRANSFER_PENALTY_SECONDS,
    WORDS_PER_SECOND,
)

TRANSFER_TOOL = "transfer_to_human_agents"

# Carrier-side (agent) write tools in the telecom domain.
CARRIER_WRITE_TOOLS = {
    "suspend_line",
    "resume_line",
    "send_payment_request",
    "enable_roaming",
    "disable_roaming",
    "refuel_data",
    "set_data_usage",
}
# Device-side (user) actions that modify state, by name pattern.
_DEVICE_WRITE_RE = re.compile(
    r"^(toggle_|turn_|set_|reset_|reseat_|reboot_|connect_|disconnect_"
    r"|grant_|make_)"
)


def is_fix_action(name: str, requestor: str) -> bool:
    if requestor == "assistant":
        return name in CARRIER_WRITE_TOOLS
    return bool(_DEVICE_WRITE_RE.match(name))


def call_metrics(sim: SimulationRun, task: Task) -> dict[str, Any]:
    reward = sim.reward_info.reward if sim.reward_info else 0.0
    success = reward is not None and reward >= 0.999

    agent_tool_calls = 0
    device_actions = 0
    agent_words = 0
    user_words = 0
    transferred = False
    fix_actions: list[tuple[str, str]] = []  # (requestor, name)

    for m in sim.messages or []:
        role = getattr(m, "role", None)
        content = getattr(m, "content", None)
        tool_calls = getattr(m, "tool_calls", None) or []
        if role == "assistant":
            if isinstance(content, str):
                agent_words += len(content.split())
            for tc in tool_calls:
                agent_tool_calls += 1
                if tc.name == TRANSFER_TOOL:
                    transferred = True
                if is_fix_action(tc.name, "assistant"):
                    fix_actions.append(("assistant", tc.name))
        elif role == "user":
            if isinstance(content, str):
                user_words += len(content.split())
            for tc in tool_calls:
                device_actions += 1
                if is_fix_action(tc.name, "user"):
                    fix_actions.append(("user", tc.name))

    ref_actions = getattr(task.evaluation_criteria, "actions", None) or []
    ref_names = {(a.requestor, a.name) for a in ref_actions}
    unnecessary_fix_actions = sum(1 for fa in fix_actions if fa not in ref_names)

    speech_s = (agent_words + user_words) / WORDS_PER_SECOND
    est_handle_time_s = (
        speech_s
        + agent_tool_calls * CARRIER_TOOL_SECONDS
        + device_actions * DEVICE_ACTION_SECONDS
        + (TRANSFER_PENALTY_SECONDS if transferred else 0.0)
    )

    return {
        "task_id": task.id,
        "reward": reward,
        "success": success,
        "termination_reason": str(sim.termination_reason),
        "agent_tool_calls": agent_tool_calls,
        "device_actions": device_actions,
        "agent_words": agent_words,
        "user_words": user_words,
        "transferred": transferred,
        "fix_actions": len(fix_actions),
        "unnecessary_fix_actions": unnecessary_fix_actions,
        "est_handle_time_s": round(est_handle_time_s, 1),
        "agent_cost": sim.agent_cost,
        "user_cost": sim.user_cost,
        "sim_duration_s": sim.duration,
        "num_messages": len(sim.messages or []),
    }


def fmt_mmss(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s:02d}s"
