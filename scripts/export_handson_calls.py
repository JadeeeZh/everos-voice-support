"""Export a curated starter pack of call logs as POST-ready memorize payloads.

Reads recorded learning-phase simulations from ``runs/v1/learn/sims`` and
renders each through the demo's own adapter (``build_memorize_messages`` +
``build_qa_note``), so the files under ``handson/calls/`` are byte-compatible
with what the measured experiment fed to ``POST /api/v2/memory/add``.

Selection covers two fault families (mobile-data, MMS) with three or more
cases each — enough for skill clustering to consolidate — and includes
failed/transferred calls whose payload carries the tier-2 correction note.

Run from the voice-support project: ``uv run python scripts/export_handson_calls.py``
"""

from __future__ import annotations

import json
from pathlib import Path

from voicedemo.config import DEMO_DIR
from voicedemo.experience import build_memorize_messages, build_qa_note
from voicedemo.metrics import call_metrics
from voicedemo.simulate import load_split

from tau2.data_model.simulation import SimulationRun

# (source train idx) → curated order. Two families, mixed outcomes.
SELECTION = [0, 1, 2, 3, 4, 5, 9, 11, 41, 43, 44, 49]

APP_ID = "voice-support"
PROJECT_ID = "handson"
BASE_TS_MS = 1_756_000_000_000  # fixed epoch base: deterministic output files
DAY_MS = 86_400_000

OUT_DIR = DEMO_DIR / "handson" / "calls"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_split("train")
    manifest: list[str] = [
        "| file | family | injected faults | outcome | correction note |",
        "|---|---|---|---|---|",
    ]
    for seq, src_idx in enumerate(SELECTION, start=1):
        sim_path = DEMO_DIR / "runs" / "v1" / "learn" / "sims" / f"{src_idx:03d}.json"
        sim = SimulationRun.model_validate(json.loads(sim_path.read_text()))
        task = tasks[src_idx]
        assert sim.task_id == task.id, (sim.task_id, task.id)
        met = call_metrics(sim, task)
        qa_note = build_qa_note(
            sim, task, resolved=met["success"], transferred=met["transferred"]
        )
        messages = build_memorize_messages(
            sim,
            customer_id=f"cust-{seq:03d}",
            base_ts_ms=BASE_TS_MS + (seq - 1) * DAY_MS,
            qa_note=qa_note,
        )
        payload = {
            "session_id": f"call-{seq:03d}",
            "app_id": APP_ID,
            "project_id": PROJECT_ID,
            "messages": messages,
        }
        out = OUT_DIR / f"call-{seq:03d}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
        family = task.id.split("]")[0].strip("[")
        faults = task.id.split("]")[1].split("[")[0]
        outcome = (
            "resolved" if met["success"]
            else ("transferred" if met["transferred"] else "failed")
        )
        corrected = "yes" if not met["success"] or met["transferred"] else "—"
        manifest.append(
            f"| `calls/{out.name}` | {family} | `{faults}` | {outcome} | {corrected} |"
        )
        print(f"{out.name}: {family:18s} {outcome:11s} msgs={len(messages)}")
    (DEMO_DIR / "handson" / "MANIFEST.md").write_text(
        "# Starter-pack call data\n\n"
        "Twelve real support calls from the measured experiment's learning\n"
        "split, rendered as POST-ready `/api/v2/memory/add` payloads. Failed\n"
        "or escalated calls end with the tier-2 specialist's factual\n"
        "close-out note — the correction signal a real call center records.\n\n"
        + "\n".join(manifest)
        + "\n"
    )
    print(f"\nwrote {len(SELECTION)} payloads + MANIFEST.md under handson/")


if __name__ == "__main__":
    main()
