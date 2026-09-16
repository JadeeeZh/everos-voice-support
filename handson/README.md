# Hands-on: give a support agent a memory, from scratch

You play a developer adopting EverOS for a voice-support line. Nothing is
pre-wired: you install the released package, create a memory root, configure
providers, feed twelve real support calls one HTTP request at a time, watch
episodes → cases → skills form on disk, and run the retrieval a live agent
would issue when the next customer calls.

Every command below was executed verbatim against `everos 1.3.1` from PyPI.
Budget: ~20 minutes, well under $1 of OpenRouter usage (extraction runs
`gpt-4.1-mini`).

You need: [`uv`](https://docs.astral.sh/uv/), `curl`, an OpenRouter API key.

---

## 0. The identity model — what you set, what falls out

The question every new user asks first: *do I have to register a project,
an agent, a user?* No. Two ids you **choose**, everything else is **derived
from the messages you send**:

| id | who sets it | where | meaning |
|---|---|---|---|
| `app_id` | you | every request body | your application namespace (here: `voice-support`) |
| `project_id` | you | every request body | partition inside the app — env, tenant, experiment (here: `handson`) |
| `session_id` | you | every request body | one conversation (here: one call, `call-001`…) |
| `sender_id` on a `user` message | you, per message | inside `messages[]` | becomes a **user memory owner** — `users/<id>/` appears |
| `sender_id` on an `assistant` message | you, per message | inside `messages[]` | becomes the **agent memory owner** — `agents/<id>/` appears |
| `agent_id` at search time | you | search request | must equal the assistant `sender_id` you memorized under |

There is no registration step. The first call you memorize materializes the
whole tree:

```
<root>/voice-support/handson/          ← app_id / project_id
├── users/cust-001/
│   ├── episodes/episode-*.md          ← what happened, from the customer's side
│   └── .atomic_facts/atomic_fact-*.md ← durable facts about this customer
└── agents/support-agent/
    ├── .cases/agent_case-*.md         ← one operational "case file" per call
    └── skills/skill_*/SKILL.md        ← consolidated across calls, by clustering
```

Markdown is the source of truth; SQLite holds state; LanceDB holds the
rebuildable index. All three live under the root you choose next.

## 1. Install

```bash
mkdir everos-handson && cd everos-handson
uv venv .venv
VIRTUAL_ENV=$PWD/.venv uv pip install everos
.venv/bin/everos --help
```

## 2. Create a memory root

Use a local root so the experiment is disposable and nothing touches
`~/.everos`:

```bash
.venv/bin/everos init --root ./everos-root
```

This writes two files: `everos-root/everos.toml` (server + provider config,
heavily commented) and `everos-root/ome.toml` (background-extraction
scheduler). Read them — they are the whole configuration surface.

## 3. Configure providers — one OpenRouter key drives everything

Three things must be true: an extraction LLM, an embedding endpoint, and —
deliberately — **no rerank provider** (retrieval will pass
`enable_llm_rerank` instead, so no second vendor is needed).

**Option A — edit `everos-root/everos.toml`** (simplest):

```toml
[llm]                                   # extraction LLM — defaults already point at OpenRouter
model    = "openai/gpt-4.1-mini"
api_key  = "sk-or-..."                  # ← your key
base_url = "https://openrouter.ai/api/v1"

[embedding]                             # ship default is DeepInfra — switch it to OpenRouter
model      = "openai/text-embedding-3-small"
api_key    = "sk-or-..."                # ← same key
base_url   = "https://openrouter.ai/api/v1"
dimensions = 1024                       # ← REQUIRED: MRL-truncate to EverOS's vector width
```

**Option B — environment variables** (no secrets on disk); every TOML field
maps to `EVEROS_<SECTION>__<KEY>`:

```bash
export EVEROS_LLM__API_KEY="sk-or-..."
export EVEROS_EMBEDDING__MODEL="openai/text-embedding-3-small"
export EVEROS_EMBEDDING__BASE_URL="https://openrouter.ai/api/v1"
export EVEROS_EMBEDDING__API_KEY="sk-or-..."
export EVEROS_EMBEDDING__DIMENSIONS=1024
```

The one setup trap: forget `dimensions = 1024` and indexing fails with a
vector-width mismatch once the first memory lands.

## 4. Start the server

```bash
.venv/bin/everos server start --root ./everos-root --port 8600 > server.log 2>&1 &
sleep 5 && curl -s http://127.0.0.1:8600/health | python3 -m json.tool
```

`/health` is worth reading once: `capabilities` should show `llm: true` and
`embed: true`; `rerank: false` is expected and fine.

## 5. The call data

Twelve real support calls live in [`calls/`](calls/) — see
[`MANIFEST.md`](MANIFEST.md) for what each contains. They come from the
measured benchmark's learning phase (τ²-bench telecom, gpt-4.1 agent and
customer simulator) and are already in the exact `/api/v2/memory/add` body
shape:

```jsonc
{
 "session_id": "call-001",
 "app_id": "voice-support",
 "project_id": "handson",
 "messages": [
  {"sender_id": "support-agent", "role": "assistant", "content": "Hi! How can I help you today?", "timestamp": 1756000000000},
  {"sender_id": "cust-001",      "role": "user",      "content": "Hi! I'm having issues with my mobile data...", "timestamp": 1756000015000},
  {"sender_id": "support-agent", "role": "assistant", "content": "", "tool_calls": [{"id": "…", "type": "function", "function": {"name": "get_customer_by_phone", "arguments": "{…}"}}], "timestamp": …},
  {"sender_id": "support-agent", "role": "tool", "tool_call_id": "…", "content": "{…result…}", "timestamp": …}
  // … full call; timestamps are unix-ms; tool_calls are OpenAI-shaped
 ]
}
```

Two details that matter for learning quality:

- Guided device steps (things the customer does on their phone at the
  agent's direction) are recorded as assistant-attributed `device_*` tool
  rounds — the agent drove them, so they belong to its case file.
- Calls that failed or were escalated end with a factual tier-2 close-out
  note (root cause + fix). That is the correction signal a real call center
  records, and it is where the hardest lessons come from. `grep -l
  "Tier-2 specialist" calls/*.json` shows which.

## 6. Feed the first call

Copy the `calls/` directory next to your root, then:

```bash
curl -s -X POST http://127.0.0.1:8600/api/v2/memory/add \
  -H 'Content-Type: application/json' -d @calls/call-001.json
# → {"data": {"message_count": 41, "status": "accumulated"}}

curl -s -X POST http://127.0.0.1:8600/api/v2/memory/flush \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"call-001","app_id":"voice-support","project_id":"handson","messages":[]}'
# → {"data": {"status": "extracted"}}
```

`add` buffers the transcript under the session; `flush` closes the session
and hands it to extraction. In production you stream turns with repeated
`add` calls during the call and `flush` at hang-up.

## 7. Watch it digest

Extraction is asynchronous. Watch it happen:

```bash
tail -f server.log | grep -E "agent_skills_extracted|skill_cluster"
```

Within a minute or two, list what appeared:

```bash
find everos-root/voice-support -type f
```

After one call you already have the customer's episode + atomic facts, the
agent's first case, and — since even one case seeds a cluster — a first
`skills/skill_*/SKILL.md`. Open them; they are meant to be read:

```bash
cat everos-root/voice-support/handson/users/cust-001/episodes/*.md
cat everos-root/voice-support/handson/agents/support-agent/.cases/*.md
```

## 8. Feed the rest

```bash
for f in calls/call-0{02..12}.json; do
  sid=$(python3 -c "import json;print(json.load(open('$f'))['session_id'])")
  curl -s -X POST http://127.0.0.1:8600/api/v2/memory/add \
    -H 'Content-Type: application/json' -d @"$f" > /dev/null
  curl -s -X POST http://127.0.0.1:8600/api/v2/memory/flush \
    -H 'Content-Type: application/json' \
    -d "{\"session_id\":\"$sid\",\"app_id\":\"voice-support\",\"project_id\":\"handson\",\"messages\":[]}"
  echo " $sid fed"
done
```

Give the background pipeline a few minutes (the `tail -f` from step 7 shows
each case being clustered and consolidated).

## 9. Read what it learned

```bash
ls everos-root/voice-support/handson/agents/support-agent/skills/
grep -i -c roam everos-root/voice-support/handson/agents/support-agent/skills/*/SKILL.md
```

On our verified run the twelve calls consolidated into
`skill_Diagnose_and_fix_mobile_data_issues_abroad` with ~10 roaming
mentions — including the lesson the failed calls paid for: when the
customer is abroad, carrier-side roaming and the phone's own data-roaming
toggle are *two different switches*. Nobody wrote that skill; the calls did.

## 10. Retrieve like a live agent

When the next customer's first sentence arrives, the agent issues one
search:

```bash
curl -s -X POST http://127.0.0.1:8600/api/v2/memory/search \
  -H 'Content-Type: application/json' -d '{
    "agent_id": "support-agent",
    "app_id": "voice-support",
    "project_id": "handson",
    "query": "Customer cannot use mobile data on their phone while traveling",
    "method": "hybrid",
    "top_k": 3,
    "enable_llm_rerank": true
  }' | python3 -m json.tool | head -60
```

You get back `agent_skills` (name / description / content / confidence) and
`agent_cases` (task_intent / approach / key_insight) — render them into a
system-prompt block and the agent walks into the call carrying its own
experience. The exact rendering the benchmark used is
[`../src/voicedemo/experience.py`](../src/voicedemo/experience.py)
(`render_experience_block` + the `<learned_experience>` wrapper in
[`agent.py`](../src/voicedemo/agent.py)).

## 11. Where to go from here

- **Point it at your own calls** — anything that fits the message schema in
  step 5 works; the ids in step 0 are the whole contract.
- **Run the measured benchmark** — `make smoke` in
  [`voice-support/`](../README.md) runs the full learn → evaluate → report
  loop this data came from (50% → 62% held-out resolution; see
  [`PLAN.md`](../PLAN.md)).
- **Reset and replay** — the root is disposable: stop the server,
  `rm -rf everos-root`, `everos init`, feed again.

### Troubleshooting

| symptom | cause / fix |
|---|---|
| indexing error mentioning vector width | `[embedding] dimensions = 1024` missing |
| HTTP 422 on search | you asked for cross-encoder rerank without a provider — keep `"enable_llm_rerank": true` |
| `/health` shows `llm: false` | key not picked up — env var name or toml section typo |
| nothing appears under the root | you `add`-ed but never `flush`-ed the session |
| port already in use | pass another `--port`; the root, not the port, owns the data |
