"""Evaluation phase: held-out test calls, baseline vs EverOS arms.

Both arms use the same LLM, policy, tools, seed, and user simulator. The
EverOS arm additionally performs one memory retrieval on the customer's
first utterance. Nothing is memorized during evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from voicedemo.config import AGENT_ID, RunConfig, apply_provider_env
from voicedemo.everos_client import EverOSClient, EverOSServer, dump_json
from voicedemo.experience import render_experience_block
from voicedemo.metrics import call_metrics, fmt_mmss
from voicedemo.simulate import load_split, run_call

ARMS = ("baseline", "everos")


def read_done(path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["task_id"]
        for line in path.read_text().splitlines()
        if line.strip()
    }


def run_arm(cfg: RunConfig, arm: str, tasks, provider) -> None:
    out_path = cfg.eval_dir / f"{arm}.jsonl"
    done = read_done(out_path)
    print(f"[eval:{arm}] {len(tasks)} tasks, {len(done)} already done")
    for idx, task in enumerate(tasks):
        if task.id in done:
            continue
        t0 = time.time()
        sim, agent = run_call(
            cfg, task, experience_provider=provider if arm == "everos" else None
        )
        retrieval_error = getattr(agent, "retrieval_error", None)
        if arm == "everos" and retrieval_error:
            # A degraded retrieval silently turns this arm into the baseline
            # and skews the headline comparison. Fail fast; the record is not
            # written, so a resume retries this task.
            raise RuntimeError(
                f"EverOS retrieval failed during eval of {task.id}: "
                f"{retrieval_error} — aborting the arm instead of recording "
                "a silently-degraded call."
            )
        met = call_metrics(sim, task)
        retrievals = getattr(agent, "retrievals", [])
        record = {
            "idx": idx,
            "arm": arm,
            **met,
            "retrieval_query": getattr(agent, "retrieval_query", None),
            "experience_used": bool(getattr(agent, "experience_block", None)),
            "midcall_retrieval_used": any(
                r["stage"] == "midcall" and r["block"] for r in retrievals
            ),
            "wall_seconds": round(time.time() - t0, 1),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        dump_json(cfg.eval_dir / "sims" / f"{arm}-{idx:03d}.json", sim.model_dump())
        if arm == "everos":
            dump_json(
                cfg.eval_dir / "sims" / f"{arm}-{idx:03d}.retrieval.json",
                {
                    "query": getattr(agent, "retrieval_query", None),
                    "block": getattr(agent, "experience_block", None),
                    "retrievals": retrievals,
                },
            )
        status = "RESOLVED" if met["success"] else (
            "TRANSFERRED" if met["transferred"] else "FAILED"
        )
        print(
            f"[eval:{arm} {idx + 1}/{len(tasks)}] {status:11s} "
            f"reward={met['reward']:.2f} est={fmt_mmss(met['est_handle_time_s'])} "
            f"{task.id[:60]}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arms", default="both", choices=["both", "baseline", "everos"])
    ap.add_argument(
        "--midcall-turn",
        type=int,
        default=None,
        metavar="N",
        help="everos arm: second retrieval on the Nth customer utterance, "
        "querying with the findings accumulated so far (fixed schedule; "
        "off when omitted — v1 behavior)",
    )
    args = ap.parse_args()

    cfg = RunConfig(run_name=args.run, midcall_retrieval_user_turn=args.midcall_turn)
    cfg.ensure_dirs()
    apply_provider_env()
    tasks = load_split(args.split, limit=args.limit)
    arms = list(ARMS) if args.arms == "both" else [args.arms]

    if "everos" in arms:
        with EverOSServer(cfg, os.environ.copy()):
            client = EverOSClient(cfg.everos_base_url, project_id=cfg.run_name)

            def provider(query: str) -> str | None:
                resp = client.search_agent(
                    AGENT_ID,
                    query,
                    method=cfg.search_method,
                    top_k=cfg.top_k,
                    enable_llm_rerank=cfg.enable_llm_rerank,
                )
                return render_experience_block(resp)

            for arm in arms:
                run_arm(cfg, arm, tasks, provider if arm == "everos" else None)
            client.close()
    else:
        for arm in arms:
            run_arm(cfg, arm, tasks, None)
    print("[eval] done")


if __name__ == "__main__":
    main()
