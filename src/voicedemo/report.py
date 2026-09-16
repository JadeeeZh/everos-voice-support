"""Aggregate learning + evaluation results into the demo report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from voicedemo.config import AGENT_ID, APP_ID, RunConfig
from voicedemo.metrics import fmt_mmss


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    return {
        "n": len(records),
        "resolution_rate": _mean([1.0 if r["success"] else 0.0 for r in records]),
        "transfer_rate": _mean([1.0 if r["transferred"] else 0.0 for r in records]),
        "mean_est_handle_time_s": _mean([r["est_handle_time_s"] for r in records]),
        "median_est_handle_time_s": statistics.median(
            [r["est_handle_time_s"] for r in records]
        ),
        "mean_agent_tool_calls": _mean([r["agent_tool_calls"] for r in records]),
        "mean_device_actions": _mean([r["device_actions"] for r in records]),
        "mean_fix_actions": _mean([r["fix_actions"] for r in records]),
        "mean_unnecessary_fix_actions": _mean(
            [r["unnecessary_fix_actions"] for r in records]
        ),
        "total_llm_cost_usd": sum(
            (r.get("agent_cost") or 0) + (r.get("user_cost") or 0) for r in records
        ),
    }


def learning_curve(records: list[dict[str, Any]], window: int = 10) -> list[dict]:
    curve = []
    for i in range(len(records)):
        lo = max(0, i - window + 1)
        chunk = records[lo : i + 1]
        curve.append(
            {
                "call": i + 1,
                "rolling_resolution_rate": _mean(
                    [1.0 if r["success"] else 0.0 for r in chunk]
                ),
                "rolling_est_handle_time_s": _mean(
                    [r["est_handle_time_s"] for r in chunk]
                ),
                "experience_used": records[i].get("experience_used", False),
            }
        )
    return curve


def learned_memory_inventory(cfg: RunConfig) -> dict[str, Any]:
    """What EverOS actually built on disk (markdown is the source of truth)."""
    agent_dir = cfg.everos_root / APP_ID / cfg.run_name / "agents" / AGENT_ID
    skills = []
    skills_dir = agent_dir / "skills"
    if skills_dir.exists():
        for d in sorted(skills_dir.iterdir()):
            skill_md = d / "SKILL.md"
            if skill_md.exists():
                first_lines = skill_md.read_text().splitlines()
                title = next(
                    (ln.lstrip("# ").strip() for ln in first_lines if ln.startswith("#")),
                    d.name,
                )
                skills.append({"name": d.name, "title": title, "path": str(skill_md)})
    cases_dir = agent_dir / ".cases"
    n_cases = 0
    if cases_dir.exists():
        for f in cases_dir.glob("*.md"):
            n_cases += sum(1 for ln in f.read_text().splitlines() if ln.startswith("## "))
    users_dir = cfg.everos_root / APP_ID / cfg.run_name / "users"
    n_customers = len(list(users_dir.iterdir())) if users_dir.exists() else 0
    return {"skills": skills, "agent_cases": n_cases, "customers_seen": n_customers}


def render_headline(base: dict, ever: dict) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:.0f}%"

    rows = [
        ("Resolution rate", pct(base.get("resolution_rate", 0)), pct(ever.get("resolution_rate", 0))),
        (
            "Est. handle time (mean)",
            fmt_mmss(base.get("mean_est_handle_time_s", 0)),
            fmt_mmss(ever.get("mean_est_handle_time_s", 0)),
        ),
        (
            "Agent tool calls (mean)",
            f"{base.get('mean_agent_tool_calls', 0):.1f}",
            f"{ever.get('mean_agent_tool_calls', 0):.1f}",
        ),
        (
            "Guided device actions (mean)",
            f"{base.get('mean_device_actions', 0):.1f}",
            f"{ever.get('mean_device_actions', 0):.1f}",
        ),
        (
            "Unnecessary fix actions (mean)",
            f"{base.get('mean_unnecessary_fix_actions', 0):.2f}",
            f"{ever.get('mean_unnecessary_fix_actions', 0):.2f}",
        ),
        ("Human transfer rate", pct(base.get("transfer_rate", 0)), pct(ever.get("transfer_rate", 0))),
    ]
    lines = [
        "| | Standard agent | After learning with EverOS |",
        "|---|---:|---:|",
    ]
    for name, b, e in rows:
        lines.append(f"| {name} | {b} | **{e}** |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    cfg = RunConfig(run_name=args.run)
    cfg.ensure_dirs()

    learn_records = load_jsonl(cfg.learn_dir / "results.jsonl")
    base_records = load_jsonl(cfg.eval_dir / "baseline.jsonl")
    ever_records = load_jsonl(cfg.eval_dir / "everos.jsonl")

    base_agg = aggregate(base_records)
    ever_agg = aggregate(ever_records)
    inventory = learned_memory_inventory(cfg)
    curve = learning_curve(learn_records)

    report = {
        "run": args.run,
        "learning": {
            "aggregate": aggregate(learn_records),
            "curve": curve,
        },
        "evaluation": {"baseline": base_agg, "everos": ever_agg},
        "memory_inventory": inventory,
    }
    (cfg.report_dir / "report.json").write_text(json.dumps(report, indent=2))

    md = ["# Voice Support Demo — Results", ""]
    md.append(f"Run: `{args.run}`")
    md.append("")
    md.append("## Held-out evaluation (new customers, same model both arms)")
    md.append("")
    md.append(render_headline(base_agg, ever_agg))
    md.append("")
    md.append(
        f"Baseline n={base_agg.get('n', 0)}, EverOS n={ever_agg.get('n', 0)}. "
        "Handle time is modeled from event counts (see PLAN.md)."
    )
    md.append("")
    md.append("## What EverOS learned during the learning phase")
    md.append("")
    md.append(
        f"- Learning calls handled: {len(learn_records)} "
        f"(resolution rate {100 * aggregate(learn_records).get('resolution_rate', 0):.0f}%)"
    )
    md.append(f"- Agent cases extracted: {inventory['agent_cases']}")
    md.append(f"- Distinct customers seen: {inventory['customers_seen']}")
    md.append(f"- Skills consolidated: {len(inventory['skills'])}")
    for s in inventory["skills"]:
        md.append(f"  - `{s['name']}`")
    md.append("")
    md.append("## Cost")
    md.append("")
    md.append(
        f"- Learning LLM cost: ${aggregate(learn_records).get('total_llm_cost_usd', 0):.2f}; "
        f"eval baseline: ${base_agg.get('total_llm_cost_usd', 0):.2f}; "
        f"eval EverOS arm: ${ever_agg.get('total_llm_cost_usd', 0):.2f}"
    )
    (cfg.report_dir / "report.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\n[report] written to {cfg.report_dir}")


if __name__ == "__main__":
    main()
