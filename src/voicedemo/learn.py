"""Learning phase: handle train-split calls sequentially, memorizing each one.

After every call the full call log (+ a factual QA / tier-2 resolution note)
is written to EverOS and the pipeline is allowed to drain, so each later call
retrieves experience distilled from all earlier ones.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from tau2.data_model.simulation import SimulationRun

from voicedemo.config import AGENT_ID, RUNS_DIR, RunConfig, apply_provider_env
from voicedemo.everos_client import (
    EverOSClient,
    EverOSServer,
    dump_json,
    wait_extraction_idle,
)
from voicedemo.experience import (
    build_memorize_messages,
    build_qa_note,
    render_experience_block,
)
from voicedemo.metrics import call_metrics, fmt_mmss
from voicedemo.simulate import load_split, run_call


def read_done_task_ids(path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["task_id"]
        for line in path.read_text().splitlines()
        if line.strip()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run name (also EverOS project_id)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--no-experience",
        action="store_true",
        help="ablation: learn without retrieval during the learning calls",
    )
    ap.add_argument(
        "--replay-from",
        default=None,
        metavar="SRC_RUN",
        help="memorize the call logs recorded under runs/<SRC_RUN>/learn/sims "
        "instead of simulating new calls (no agent/user LLM cost; extraction "
        "still runs). Gives identical learning input across runs, so only "
        "the memory pipeline configuration differs.",
    )
    ap.add_argument(
        "--skill-prompt",
        default="stock",
        choices=["stock", "v2"],
        help="agent-skill consolidation prompt in the EverOS server "
        "(v2 = contrastive-insight preservation; see PLAN.md)",
    )
    args = ap.parse_args()

    cfg = RunConfig(run_name=args.run, skill_prompt=args.skill_prompt)
    cfg.ensure_dirs()
    apply_provider_env()

    tasks = load_split(args.split, limit=args.limit)
    results_path = cfg.learn_dir / "results.jsonl"
    done = read_done_task_ids(results_path)
    print(f"[learn] {len(tasks)} tasks in split '{args.split}', {len(done)} already done")

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

        exp_provider = None if args.no_experience else provider

        # On resume, let any extraction still in flight from the previous
        # process drain before the next call retrieves against the index.
        if done:
            wait_extraction_idle(cfg.everos_root, timeout_s=cfg.extraction_wait_s)

        for idx, task in enumerate(tasks):
            if task.id in done:
                continue
            t0 = time.time()
            if args.replay_from:
                src = RUNS_DIR / args.replay_from / "learn" / "sims" / f"{idx:03d}.json"
                sim = SimulationRun.model_validate(json.loads(src.read_text()))
                if sim.task_id != task.id:
                    raise RuntimeError(
                        f"replay mismatch at idx={idx}: source sim is for "
                        f"{sim.task_id!r}, split task is {task.id!r}"
                    )
                agent = None
            else:
                sim, agent = run_call(cfg, task, experience_provider=exp_provider)
            met = call_metrics(sim, task)

            customer_id = f"cust-{args.split}-{idx:03d}"
            session_id = f"call-{args.split}-{idx:03d}"
            qa_note = build_qa_note(
                sim, task, resolved=met["success"], transferred=met["transferred"]
            )
            messages = build_memorize_messages(
                sim,
                customer_id=customer_id,
                base_ts_ms=int(time.time() * 1000),
                qa_note=qa_note,
            )
            add_resp = client.add(session_id, messages)
            flush_resp = client.flush(session_id)
            t_sim = time.time() - t0

            # Persist the record immediately after memorize: a crash during
            # the (long) extraction wait must not cause a resume to re-run
            # and re-memorize this call under the same session_id.
            record = {
                "idx": idx,
                "task_id": task.id,
                "session_id": session_id,
                "customer_id": customer_id,
                **met,
                "memorize_add": add_resp.get("data", add_resp),
                "memorize_flush": flush_resp.get("data", flush_resp),
                "retrieval_query": getattr(agent, "retrieval_query", None),
                "experience_used": bool(getattr(agent, "experience_block", None)),
                "retrieval_error": getattr(agent, "retrieval_error", None),
                "sim_seconds": round(t_sim, 1),
                "replayed_from": args.replay_from,
            }
            with results_path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")

            wait_extraction_idle(cfg.everos_root, timeout_s=cfg.extraction_wait_s)
            t_total = time.time() - t0
            dump_json(cfg.learn_dir / "sims" / f"{idx:03d}.json", sim.model_dump())
            if agent is not None:
                dump_json(
                    cfg.learn_dir / "sims" / f"{idx:03d}.retrieval.json",
                    {
                        "query": getattr(agent, "retrieval_query", None),
                        "block": getattr(agent, "experience_block", None),
                    },
                )
            status = "RESOLVED" if met["success"] else (
                "TRANSFERRED" if met["transferred"] else "FAILED"
            )
            phase = "replay" if args.replay_from else "learn"
            print(
                f"[{phase} {idx + 1}/{len(tasks)}] {status:11s} "
                f"reward={met['reward']:.2f} est={fmt_mmss(met['est_handle_time_s'])} "
                f"exp={'y' if record['experience_used'] else 'n'} "
                f"wall={fmt_mmss(t_total)} {task.id[:60]}"
            )

        client.close()
    print("[learn] done")


if __name__ == "__main__":
    main()
