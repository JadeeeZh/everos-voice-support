# EverOS Voice Support Demo

> Most voice agents start every call from the same playbook.
> EverOS lets them learn from every call.

A telecom customer-support agent, evaluated on [τ²-bench](https://github.com/sierra-research/tau2-bench)'s
telecom domain, that writes every finished call into [EverOS](https://github.com/EverMind-AI/EverOS)
and retrieves distilled operational experience when a brand-new customer calls.

This is **not** "remember the caller": every evaluation customer is a
first-time caller. What transfers between calls is the agent's *experience* —
EverOS consolidates individual calls (episodes → agent cases) into
generalized skills owned by the agent, and those skills make call #75 with a
never-seen customer faster and more likely to resolve than call #1.

```
Customer A experience ┐
Customer B experience ├─► EverOS ─► generalized operational experience
Customer C correction ┘              │
                                     ▼
                     brand-new Customer D gets better service
```

## How it works

1. **Learning phase** (`voicedemo.learn`) — the agent handles the τ² telecom
   `train` split (74 calls) sequentially. After each call, the full call log
   goes to EverOS (`/api/v2/memory/add` + `/flush`). Failed or escalated
   calls also get the tier-2 specialist's factual ticket-closure note (root
   state + resolution steps from the benchmark's ground truth) — the same
   correction signal a real call center records. EverOS then extracts
   episodes, per-call agent cases, and — by clustering cases across calls —
   consolidated agent skills. All automatic, no manual triggers.
2. **Evaluation phase** (`voicedemo.evaluate`) — the held-out `test` split
   (40 calls, new customers), two arms with the *same* model, policy, tools,
   and seed:
   - **baseline** — stock τ²-bench `LLMAgent`.
   - **everos** — identical agent plus one memory search on the customer's
     first utterance, rendered into a `<learned_experience>` prompt block.
3. **Report** (`voicedemo.report`) — resolution rate, estimated handle time,
   tool calls, guided device actions, unnecessary fix actions, transfer
   rate; plus the learning curve and the skill inventory EverOS built.
   `voicedemo.transcripts` renders side-by-side showcase calls.

## Run it

Requirements: `uv`, an OpenRouter API key (one key drives everything:
the sim agent + user simulator via litellm, EverOS extraction LLM, and
EverOS embeddings via OpenRouter's `/embeddings` endpoint).

EverOS itself needs no separate setup: if this directory sits inside an
EverOS source checkout the server runs from that tree; a standalone clone
gets the released CLI via `make setup` (`uv tool install everos`). Point
`VOICEDEMO_EVEROS_REPO=/path/to/EverOS` at a checkout to override.

```bash
make setup                          # vendor tau2-bench (pinned) + uv sync
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
make smoke                          # 3 learning + 2 eval calls end-to-end
make learn RUN=v1                   # 74 learning calls (sequential)
make evaluate RUN=v1                # 40 test calls x 2 arms
make report RUN=v1
uv run python -m voicedemo.transcripts --run v1
```

Outputs land in `runs/<run>/` (gitignored): the isolated EverOS memory root
(`everos-root/` — inspect `agents/support-agent/skills/*/SKILL.md` to read
what the agent learned), per-call simulation JSON, JSONL results, and the
final report.

## Honesty notes

- Both arms share model, temperature, policy, max steps, seed, and user
  simulator; the EverOS arm's only extra input is memory built from train
  calls. Test tasks are τ²-bench's own held-out split.
- Success is judged by the benchmark's environment assertions, not by us.
- "Estimated handle time" is modeled from event counts (~150 wpm speech,
  6 s per system lookup, 25 s per guided device action, 240 s transfer
  penalty) — the event counts themselves are raw benchmark facts.
- "Unnecessary fix actions" = state-changing actions not present in the
  task's reference resolution (matched by requestor + tool name).
- Guided device actions are recorded in the EverOS call log as
  assistant-attributed `device_*` tool rounds; see `experience.py` for the
  rationale.

## Layout

See [PLAN.md](PLAN.md) for the full experiment design and the EverOS
integration contract this code relies on.
