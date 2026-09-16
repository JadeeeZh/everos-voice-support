"""Render side-by-side showcase transcripts from saved eval simulations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from voicedemo.config import RunConfig
from voicedemo.metrics import fmt_mmss
from voicedemo.report import load_jsonl


def _slug(task_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", task_id).strip("-")[:70]


def render_transcript(sim: dict[str, Any], *, max_result_chars: int = 160) -> str:
    lines: list[str] = []
    for m in sim.get("messages") or []:
        role = m.get("role")
        content = m.get("content")
        tool_calls = m.get("tool_calls") or []
        if role == "assistant":
            if isinstance(content, str) and content.strip():
                lines.append(f"**Agent:** {content.strip()}")
            for tc in tool_calls:
                args = json.dumps(tc.get("arguments", {}), default=str)
                lines.append(f"> agent action: `{tc.get('name')}({args})`")
        elif role == "user":
            if isinstance(content, str) and content.strip():
                lines.append(f"**Customer:** {content.strip()}")
            for tc in tool_calls:
                args = json.dumps(tc.get("arguments", {}), default=str)
                lines.append(f"> customer device action: `{tc.get('name')}({args})`")
        elif role == "tool":
            result = str(content or "").replace("\n", " ")
            if len(result) > max_result_chars:
                result = result[: max_result_chars - 1] + "…"
            lines.append(f"> result: `{result}`")
    return "\n\n".join(lines)


def _sim_path(cfg: RunConfig, arm: str, idx: int) -> Path:
    return cfg.eval_dir / "sims" / f"{arm}-{idx:03d}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    cfg = RunConfig(run_name=args.run)

    base = {r["task_id"]: r for r in load_jsonl(cfg.eval_dir / "baseline.jsonl")}
    ever = {r["task_id"]: r for r in load_jsonl(cfg.eval_dir / "everos.jsonl")}
    common = [t for t in ever if t in base]

    def contrast(task_id: str) -> tuple[int, float]:
        b, e = base[task_id], ever[task_id]
        success_gain = int(e["success"]) - int(b["success"])
        time_gain = b["est_handle_time_s"] - e["est_handle_time_s"]
        return (success_gain, time_gain)

    ranked = sorted(common, key=contrast, reverse=True)[: args.top]
    out_dir = cfg.report_dir / "showcase"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, task_id in enumerate(ranked, 1):
        b, e = base[task_id], ever[task_id]
        b_sim = json.loads(_sim_path(cfg, "baseline", b["idx"]).read_text())
        e_sim = json.loads(_sim_path(cfg, "everos", e["idx"]).read_text())
        retrieval_path = cfg.eval_dir / "sims" / f"everos-{e['idx']:03d}.retrieval.json"
        retrieval = (
            json.loads(retrieval_path.read_text()) if retrieval_path.exists() else {}
        )

        def outcome(r: dict[str, Any]) -> str:
            status = "resolved" if r["success"] else (
                "transferred to human" if r["transferred"] else "unresolved"
            )
            return (
                f"{status}, est. {fmt_mmss(r['est_handle_time_s'])}, "
                f"{r['agent_tool_calls']} agent tool calls, "
                f"{r['device_actions']} guided device actions"
            )

        md = [
            f"# Showcase {rank}: `{task_id}`",
            "",
            f"- **Standard agent:** {outcome(b)}",
            f"- **EverOS agent:** {outcome(e)}",
            "",
            "## Experience retrieved by the EverOS agent",
            "",
            "```",
            (retrieval.get("block") or "(none)").strip(),
            "```",
            "",
            "## Standard agent call",
            "",
            render_transcript(b_sim),
            "",
            "## EverOS agent call",
            "",
            render_transcript(e_sim),
            "",
        ]
        path = out_dir / f"{rank:02d}-{_slug(task_id)}.md"
        path.write_text("\n".join(md))
        print(f"[transcripts] wrote {path}")


if __name__ == "__main__":
    main()
