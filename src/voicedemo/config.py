"""Run configuration, paths, and the voice handle-time model."""

from __future__ import annotations

import os
import shutil
import tomllib
import zlib
from dataclasses import dataclass, field
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[2]
# EverOS source checkout to run the server from. Defaults to the demo's parent
# directory (its original home inside the EverOS repo); a standalone clone
# falls back to an ``everos`` executable on PATH — see everos_command().
EVEROS_REPO = Path(os.environ.get("VOICEDEMO_EVEROS_REPO", DEMO_DIR.parent))
RUNS_DIR = DEMO_DIR / "runs"
VENDOR_TAU2 = DEMO_DIR / "vendor" / "tau2-bench"
PATCHES_DIR = DEMO_DIR / "patches"

DOMAIN = "telecom"
APP_ID = "voice-support"
AGENT_ID = "support-agent"

# Voice handle-time model (documented assumption; see PLAN.md).
# Event counts are raw benchmark facts; seconds are the model.
WORDS_PER_SECOND = 2.5  # ~150 wpm conversational speech
CARRIER_TOOL_SECONDS = 6.0  # agent-side system lookup/write
DEVICE_ACTION_SECONDS = 25.0  # explain + customer performs a device step
TRANSFER_PENALTY_SECONDS = 240.0  # requeue + re-explain to a human agent


def everos_command() -> list[str]:
    """Command prefix that invokes the ``everos`` CLI.

    Inside an EverOS source checkout the server runs from that tree via
    ``uv run --project``; anywhere else the released CLI on PATH is used
    (``uv tool install everos``, which ``scripts/setup.sh`` does for you).
    """
    pyproject = EVEROS_REPO / "pyproject.toml"
    if pyproject.exists() and 'name = "everos"' in pyproject.read_text():
        return ["uv", "run", "--project", str(EVEROS_REPO), "everos"]
    exe = shutil.which("everos")
    if exe:
        return [exe]
    raise RuntimeError(
        "EverOS not found: run this demo inside an EverOS checkout, set "
        "VOICEDEMO_EVEROS_REPO=/path/to/EverOS, or `uv tool install everos`."
    )


@dataclass
class RunConfig:
    run_name: str
    agent_llm: str = "openrouter/openai/gpt-4.1"
    user_llm: str = "openrouter/openai/gpt-4.1"
    agent_llm_args: dict = field(default_factory=lambda: {"temperature": 0.0})
    user_llm_args: dict = field(default_factory=dict)
    max_steps: int = 100
    max_errors: int = 10
    seed: int = 42
    everos_host: str = "127.0.0.1"
    # 0 = derive a stable per-run port, so a stale server from a *different*
    # run (different memory root) is never silently adopted on a shared port.
    everos_port: int = 0
    search_method: str = "hybrid"
    enable_llm_rerank: bool = True  # no cross-encoder rerank provider needed
    top_k: int = 5
    extraction_wait_s: int = 600
    # v2 levers (PLAN.md): consolidation prompt override ("stock" | "v2") and
    # a second retrieval on the Nth customer utterance (None = off, v1 behavior).
    skill_prompt: str = "stock"
    midcall_retrieval_user_turn: int | None = None

    def __post_init__(self) -> None:
        if self.everos_port == 0:
            self.everos_port = 8100 + zlib.crc32(self.run_name.encode()) % 800

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / self.run_name

    @property
    def everos_root(self) -> Path:
        return self.run_dir / "everos-root"

    @property
    def learn_dir(self) -> Path:
        return self.run_dir / "learn"

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / "eval"

    @property
    def report_dir(self) -> Path:
        return self.run_dir / "report"

    @property
    def everos_base_url(self) -> str:
        return f"http://{self.everos_host}:{self.everos_port}"

    def ensure_dirs(self) -> None:
        for d in (
            self.run_dir,
            self.everos_root,
            self.learn_dir / "sims",
            self.eval_dir / "sims",
            self.report_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip("'\"")
    return out


def load_provider_env() -> dict[str, str]:
    """Derive provider env vars for the demo from one OpenRouter key.

    Key lookup order: process env OPENROUTER_API_KEY > voice-support/.env >
    ~/.everos/everos.toml [llm].api_key. The same key drives (a) litellm for
    the tau2 agent + user simulator, (b) the EverOS server's extraction LLM,
    and (c) EverOS embeddings via OpenRouter's OpenAI-compatible
    /embeddings endpoint (text-embedding-3-small, MRL-truncated to 1024 to
    match EverOS's default vector dimension). Rerank stays unconfigured:
    retrieval uses hybrid search with enable_llm_rerank=true, which the
    search manager exempts from the cross-encoder rerank requirement.
    Existing process env always wins; no secrets are written to disk here.
    """
    env: dict[str, str] = {}
    toml_cfg: dict = {}
    cfg_path = Path("~/.everos/everos.toml").expanduser()
    if cfg_path.exists():
        toml_cfg = tomllib.loads(cfg_path.read_text())

    key = (
        os.environ.get("OPENROUTER_API_KEY")
        or _read_dotenv(DEMO_DIR / ".env").get("OPENROUTER_API_KEY")
        or str(toml_cfg.get("llm", {}).get("api_key") or "")
    )
    if key and "OPENROUTER_API_KEY" not in os.environ:
        env["OPENROUTER_API_KEY"] = key

    openrouter = "https://openrouter.ai/api/v1"
    llm = toml_cfg.get("llm", {})
    defaults = {
        "EVEROS_LLM__MODEL": str(llm.get("model") or "openai/gpt-4.1-mini"),
        "EVEROS_LLM__BASE_URL": str(llm.get("base_url") or openrouter),
        "EVEROS_LLM__API_KEY": key,
        "EVEROS_EMBEDDING__MODEL": "openai/text-embedding-3-small",
        "EVEROS_EMBEDDING__BASE_URL": openrouter,
        "EVEROS_EMBEDDING__API_KEY": key,
        "EVEROS_EMBEDDING__DIMENSIONS": "1024",
    }
    for env_key, val in defaults.items():
        if val and env_key not in os.environ:
            env[env_key] = val
    return env


def apply_provider_env() -> dict[str, str]:
    """Merge derived provider env into this process; return the full mapping."""
    derived = load_provider_env()
    os.environ.update(derived)
    if "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found in env or ~/.everos/everos.toml [llm]; "
            "the tau2 agent/user simulator cannot run."
        )
    # tau2 loads TAU2_DATA_DIR for non-editable installs; ours is editable but
    # setting it explicitly keeps behavior independent of install mode.
    os.environ.setdefault("TAU2_DATA_DIR", str(VENDOR_TAU2 / "data"))
    return dict(os.environ)
