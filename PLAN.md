# Voice Support Demo — Plan (authoritative)

**Claim under demonstration**

> Most voice agents start every call from the same playbook. EverOS lets them
> learn from every call — so a brand-new customer gets better service.

Learn **across calls**, not just remember callers. This is NOT a CRM-memory
demo: evaluation customers are all first-time callers the agent has never seen.

## Scenario

Telecom carrier support (τ²-bench `telecom` domain): customers call about
no-service / mobile-data / MMS failures. The environment is dual-control — the
agent has carrier-side tools (account lookup, line status, roaming, refuel) and
guides the customer through on-device actions (check status bar, toggle
roaming, reseat SIM, ...). Task success is judged by the benchmark's own
environment assertions (e.g. "speed test returns excellent"), not by us.

## Experiment design

Data: τ²-bench telecom ships `split_tasks.json` with disjoint `train` (74) and
`test` (40) splits over ~2.3k generated tasks; underlying fault families:
airplane mode, bad network preference, data mode off, data saver, data usage
exceeded, roaming (abroad) combos, VPN, APN, SIM issues, overdue-bill
suspensions.

1. **Learning phase** — the EverOS-backed agent handles the `train` calls
   sequentially. After each call the full call log is written to EverOS
   (`/api/v2/memory/add` + `/flush`). For calls that fail or transfer to a
   human, a factual post-call resolution note (what was actually broken +
   the fix, from the task's ground truth) is appended — exactly the ticket
   close-out note a tier-2 human leaves in a real call center.
   EverOS then extracts, automatically: Episode (per customer), AgentCase
   (per call: task_intent / approach / key_insight / quality_score), and —
   via LLM clustering across calls — AgentSkill (generalized procedure,
   owner-scoped to the *agent*, not any customer).
2. **Evaluation phase** — held-out `test` calls, brand-new customers, two arms
   with the *same* agent LLM, same policy, same user simulator:
   - **Baseline**: stock τ²-bench `LLMAgent`.
   - **EverOS**: identical agent + one retrieval on the customer's first
     utterance (`agent_id` search → agent_skills + agent_cases) rendered into
     a `<learned_experience>` system-prompt block. Nothing else differs.
3. **Report** — per-call and aggregate metrics, plus side-by-side showcase
   transcripts for the strongest contrast pairs.

## Metrics

| Metric | Source |
|---|---|
| Resolution rate | τ² reward (product of env assertions / action checks) |
| Estimated handle time | explicit voice-time model over events (see below) |
| Agent tool calls | assistant-side tool calls in trajectory |
| Guided device actions | user-side tool calls in trajectory |
| Human transfers | `transfer_to_human_agents` called |
| Unnecessary fix actions | write/fix actions not in the task's reference resolution |

Handle-time model (documented assumption, not measured audio): speech at
~150 wpm for both parties, +6 s per carrier-system lookup, +25 s per guided
device action, +240 s penalty on human transfer. Reported as "estimated
handle time"; the event counts it derives from are raw benchmark facts.

## Honesty constraints

- Both arms use the same LLM, temperature, policy, max steps, and seeds.
- The EverOS arm gets no task ground truth: its only extra input is memory
  retrieved from *train* calls; eval tasks are held out and customers are new.
- Ground-truth resolution notes are only fed for failed/transferred learning
  calls, mirroring human escalation outcomes — and only during learning.
- Report failures faithfully, including calls where the EverOS arm does worse.

## Architecture

```
voice-support/
├── PLAN.md / README.md
├── pyproject.toml            # own uv project (py312): tau2 (vendored), httpx
├── Makefile                  # setup / smoke / learn / evaluate / report
├── scripts/setup.sh          # clone tau2-bench into vendor/ (pinned SHA), uv sync
├── vendor/tau2-bench/        # gitignored; pinned a2c0247 (main, 2026-08-18)
├── src/voicedemo/
│   ├── config.py             # RunConfig, paths, models, time model constants
│   ├── everos_client.py      # HTTP v2 client + server subprocess manager + readiness
│   ├── experience.py         # retrieval → prompt block; call log → memorize payload
│   ├── agent.py              # ExperienceAgent (LLMAgent + first-utterance retrieval)
│   ├── simulate.py           # power-user tau2 path: env/user/orchestrator per call
│   ├── metrics.py            # per-call metrics + handle-time model + aggregates
│   ├── learn.py              # CLI: learning phase
│   ├── evaluate.py           # CLI: eval phase (both arms)
│   ├── report.py             # CLI: aggregate report + headline table
│   └── transcripts.py        # side-by-side showcase transcript rendering
└── runs/<run>/               # gitignored: everos-root/, learn/, eval/, report/
```

EverOS integration facts the code relies on (verified against source):

- Server-mode only: cascade (LanceDB projection) and OME (case/skill
  extraction) live in the FastAPI lifespan → run `everos server start` with
  `EVEROS_ROOT=<run>/everos-root`, provider config injected via `EVEROS_*`
  env parsed from `~/.everos/everos.toml` (no secrets on disk here).
- Memorize payload: `{session_id, app_id, project_id, messages[≤500]}`,
  roles user/assistant/tool, unix-ms timestamps, OpenAI-shaped tool_calls;
  `sender_id` = customer id on user turns, `support-agent` on agent turns
  (agent memory owner). App/partition: `app_id=voice-support`,
  `project_id=<run name>`.
- AgentCase extraction gates: ≥3 assistant tool-call rounds per call log,
  trajectory ends with an assistant text message, quality ≥0.2 to cluster.
  The call-log adapter therefore renders guided device actions as
  assistant-attributed tool rounds (`device_*`) and always closes with the
  agent's wrap-up + factual QA note.
- Skills form automatically on flush (embedding configured); readiness =
  `/health` cascade pending 0 twice + OME idle; per-session lock is serial.
- Retrieval: `POST /api/v2/memory/search {agent_id, query, method=hybrid,
  top_k}` → `agent_skills` (name/description/content/confidence) +
  `agent_cases` (task_intent/approach/key_insight/quality_score).

## Scale & cost

Smoke: 3 learning + 2 eval calls. Full v1: 74 learning calls, 40 test × 2
arms + learning-arm cost. Agent + user sim LLM: `openrouter/openai/gpt-4.1`
via the OpenRouter key already configured for EverOS. Measure actual cost at
smoke and extrapolate before the full run.

## v1 result & diagnosis (2026-08-25)

v1 headline was a null result: resolution 50% vs 50% (n=40/arm); paired
2 everos-only wins / 2 losses / 18 both-succeed / 18 both-fail. Deep dive
over the 18 both-fail pairs: 17 involve `user_abroad_*` faults and in all
17 at least one arm left a roaming leg unfixed — the agent conflates the
carrier-side "Roaming Enabled" line flag with the device-side data-roaming
toggle and never asks whether the customer is traveling. The lesson exists
in the extracted agent cases (KeyInsights are exactly right), but skill
consolidation flattened it: the abroad-mobile-data skill contains zero
roaming steps, and the MMS skill codified the trap ("If roaming enabled →
proceed"). Positive control proving injection works: the skill's explicit
refuel steps cut missed `refuel_data` from 6 tasks (baseline) to 2 (everos).

## v2 levers (implemented 2026-08-25, run pending funding)

Only the memory pipeline changes; benchmark, models, policy, seeds, and the
baseline arm are untouched.

1. **Consolidation prompt v2** — `patches/skill_prompt_v2.py` derives the
   v2 prompt from the installed everalgo success-path prompt via anchored,
   domain-agnostic edits (contrastive-insight hard rules; pitfall cap 4→6
   with failure-derived priority; "preserve, never summarize away"). The
   derivation fails hard if everalgo's text drifts. Injected only into the
   EverOS server process via `patches/sitecustomize.py` on PYTHONPATH,
   gated by `VOICEDEMO_SKILL_PROMPT=v2` (everalgo re-exports its prompt
   constants precisely for startup patching). Zero telecom vocabulary in
   the added rules — the fix must generalize, not leak task knowledge.
   If v2 validates, upstream the improved prompt as the everalgo default.
2. **Mid-call retrieval** — `--midcall-turn N`: one additional retrieval on
   the Nth customer utterance (fixed schedule, never conditioned on call
   progress), querying with the opening symptom + accumulated grounded
   findings; appended to the experience block. First-utterance queries are
   too vague to reach the fault-specific cases.
3. **Replay learning** — `voicedemo.learn --replay-from v1` memorizes the
   recorded v1 call logs instead of simulating new calls: identical
   learning input across runs, so v1→v2 attribution is purely the memory
   pipeline. No agent/user LLM cost; extraction still runs (~$3).

v2 recipe (`make relearn SRC=v1 RUN=v2` → `make reuse-baseline SRC=v1
RUN=v2` → `make evaluate-everos RUN=v2` → `make report RUN=v2`); baseline
is copied from v1 verbatim (same model/seed/tasks). Estimated cost ≈ $9
(replay extraction ~$3 + everos arm ~$6).

## v2 result (2026-08-27)

Headline: resolution **62% vs 50%** (n=40/arm); every secondary metric
improved (handle time 13m57s vs 15m11s, transfers 55% vs 62%, unnecessary
fixes 2.12 vs 2.40). Paired vs baseline: **6 everos-only wins, 1 loss**,
19 both-succeed, 14 both-fail (discordant 6:1 — one-sided binomial
p≈0.063; directionally strong at n=40, not conventionally significant —
report as such). Paired vs the v1 everos arm: **5 wins, 0 losses** —
strict dominance on identical tasks/seed. Mechanism check: 5 of the 6
wins are abroad/roaming tasks, the exact diagnosed failure mode; the
abroad slice went 3/21 (baseline and v1 alike) → 7/21. Mid-call
retrieval fired in 28/40 calls. Consolidation produced 2 broader skills
(v1: 5 narrower ones); the roaming lesson survived all 74 updates
(9–12 mentions per skill).

Known remaining gaps (honest): the abroad skill still contains one
"if roaming enabled → proceed" branch — carrier-side vs device-side
disambiguation is present in pitfalls but not yet forced into the
decision branch — and the single loss (idx 0, mobile_data abroad) missed
exactly `u:toggle_roaming`; mobile_data_issue remains weak (2/9 vs 3/9
baseline). Actual v2 spend ≈ $11 (smoke $0.4, replay extraction ~$3.5,
everos arm $6.8). The v1 learning-cost line in v2's report ($11.69) is
carried over from the replayed v1 sims by construction.

Next per the v2 plan: upstream the v2 consolidation prompt to everalgo
as the default.

## Positioning (decided 2026-08-25)

The measured experiment stays on τ²-bench telecom. Vertical marketing
(healthcare, banking, ...) may re-skin the *narrative* — "any support line
learns from every call" — but every quantitative claim cites the telecom
runs. No domain-specific evaluation environments built by us: third-party
env-assertion judging is this demo's credibility backbone. Illustrative
vertical demos, if ever needed, are separate assets clearly labeled as
scripted, with no numbers.

## Out of scope for v1 (kept open)

- τ³ full-duplex audio runs (needs OpenAI Realtime + ElevenLabs + Deepgram
  keys; no Anthropic realtime provider). Transcripts are rendered so they
  can be TTS'd into the website's side-by-side call recordings later.
- Customer-profile personalization (returning callers) — deliberately
  excluded so the result isolates cross-call generalization.
