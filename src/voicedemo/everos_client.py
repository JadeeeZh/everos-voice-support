"""EverOS HTTP client (v2 API) + server subprocess manager + readiness polling."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from voicedemo.config import APP_ID, PATCHES_DIR, RunConfig, everos_command


class EverOSClient:
    """Thin client over the EverOS v2 memory API."""

    def __init__(self, base_url: str, app_id: str = APP_ID, project_id: str = "default"):
        self.base_url = base_url.rstrip("/")
        self.app_id = app_id
        self.project_id = project_id
        self._http = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(600.0))

    def close(self) -> None:
        self._http.close()

    def health(self) -> dict[str, Any]:
        r = self._http.get("/health")
        r.raise_for_status()
        return r.json()

    def add(self, session_id: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        r = self._http.post(
            "/api/v2/memory/add",
            json={
                "session_id": session_id,
                "app_id": self.app_id,
                "project_id": self.project_id,
                "messages": messages,
            },
        )
        r.raise_for_status()
        return r.json()

    def flush(self, session_id: str) -> dict[str, Any]:
        r = self._http.post(
            "/api/v2/memory/flush",
            json={
                "session_id": session_id,
                "app_id": self.app_id,
                "project_id": self.project_id,
                "messages": [],
            },
        )
        r.raise_for_status()
        return r.json()

    def search_agent(
        self,
        agent_id: str,
        query: str,
        *,
        method: str = "hybrid",
        top_k: int = 5,
        enable_llm_rerank: bool = True,
    ) -> dict[str, Any]:
        r = self._http.post(
            "/api/v2/memory/search",
            json={
                "agent_id": agent_id,
                "app_id": self.app_id,
                "project_id": self.project_id,
                "query": query,
                "method": method,
                "top_k": top_k,
                "enable_llm_rerank": enable_llm_rerank,
            },
        )
        r.raise_for_status()
        return r.json()

    def ome_trigger(self, name: str, *, force: bool = True, timeout: float = 120.0) -> dict[str, Any]:
        r = self._http.post(
            "/api/v2/ome/trigger",
            json={"name": name, "force": force, "timeout": timeout},
        )
        r.raise_for_status()
        return r.json()


def wait_extraction_idle(
    everos_root: Path,
    *,
    timeout_s: int = 600,
    poll_interval_s: float = 2.0,
    stable_polls: int = 3,
    initial_delay_s: float = 3.0,
) -> dict[str, int]:
    """Block until the cascade queue and OME runs have drained.

    Polls the server's SQLite state read-only (same approach as
    benchmarks/run.py in the EverOS repo). The OME event chain re-enqueues
    work (case -> cluster -> skill), so we require several consecutive
    quiet polls before declaring idle, and wait a beat first so the run
    dispatched by the flush has a chance to appear in run_record at all.
    """
    time.sleep(initial_delay_s)
    system_db = everos_root / ".index" / "sqlite" / "system.db"
    ome_db = everos_root / ".index" / "sqlite" / "ome.db"
    deadline = time.time() + timeout_s
    quiet = 0
    last = {"cascade_pending": -1, "ome_running": -1, "ome_failed": 0}
    while time.time() < deadline:
        cascade_pending = _poll_cascade_pending(system_db)
        ome_running, ome_failed = _poll_ome(ome_db)
        last = {
            "cascade_pending": cascade_pending,
            "ome_running": ome_running,
            "ome_failed": ome_failed,
        }
        if cascade_pending == 0 and ome_running == 0:
            quiet += 1
            if quiet >= stable_polls:
                return last
        else:
            quiet = 0
        time.sleep(poll_interval_s)
    raise TimeoutError(f"extraction did not go idle within {timeout_s}s: {last}")


def _poll_cascade_pending(system_db: Path) -> int:
    if not system_db.exists():
        return 0
    conn = sqlite3.connect(f"file:{system_db}?mode=ro", uri=True)
    try:
        (pending,) = conn.execute(
            "SELECT COUNT(*) FROM md_change_state "
            "WHERE status IN ('pending','processing')"
        ).fetchone()
    finally:
        conn.close()
    return int(pending or 0)


def _poll_ome(ome_db: Path) -> tuple[int, int]:
    if not ome_db.exists():
        return 0, 0
    conn = sqlite3.connect(f"file:{ome_db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM run_record GROUP BY status"
        ).fetchall()
    finally:
        conn.close()
    running = sum(c for s, c in rows if s == "running")
    failed = sum(c for s, c in rows if s in ("failed", "dead_letter", "crashed"))
    return running, failed


class EverOSServer:
    """Manage a dedicated EverOS server subprocess for one demo run.

    The server runs from an EverOS source checkout when one is present
    (`uv run --project <repo>`) and from the released ``everos`` CLI
    otherwise (see ``config.everos_command``), with the memory root isolated
    under the run directory and provider config injected via EVEROS_* env
    (no secrets written to disk here).
    """

    def __init__(self, cfg: RunConfig, provider_env: dict[str, str]):
        self.cfg = cfg
        self.env = dict(provider_env)
        if cfg.skill_prompt != "stock":
            # Prompt override rides in via sitecustomize on PYTHONPATH; the
            # shim raises at server startup if the patch cannot apply, so a
            # requested override can never silently fall back to stock.
            self.env["VOICEDEMO_SKILL_PROMPT"] = cfg.skill_prompt
            prev = self.env.get("PYTHONPATH")
            self.env["PYTHONPATH"] = (
                f"{PATCHES_DIR}{os.pathsep}{prev}" if prev else str(PATCHES_DIR)
            )
        self.proc: subprocess.Popen | None = None
        self.log_path = cfg.run_dir / "everos-server.log"

    def init_root(self) -> None:
        toml_path = self.cfg.everos_root / "everos.toml"
        if toml_path.exists():
            return
        subprocess.run(
            [*everos_command(), "init", "--root", str(self.cfg.everos_root)],
            check=True,
            capture_output=True,
            env=self.env,
        )

    def start(self, *, ready_timeout_s: float = 180.0) -> None:
        # Ports are derived per run name (config.py), so a healthy server on
        # this port is this run's own (same memory root) — safe to adopt.
        if self._is_healthy():
            return
        self.init_root()
        log = open(self.log_path, "a")
        try:
            self.proc = subprocess.Popen(
                [
                    *everos_command(), "server", "start",
                    "--root", str(self.cfg.everos_root),
                    "--host", self.cfg.everos_host,
                    "--port", str(self.cfg.everos_port),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=self.env,
            )
        finally:
            log.close()  # the child holds its own dup of the fd
        try:
            deadline = time.time() + ready_timeout_s
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"EverOS server exited early (see {self.log_path})"
                    )
                if self._is_healthy():
                    return
                time.sleep(1.0)
            raise TimeoutError(
                f"EverOS server not healthy after {ready_timeout_s}s "
                f"(see {self.log_path})"
            )
        except BaseException:
            # Never leave the just-spawned server orphaned (also covers
            # KeyboardInterrupt during the readiness wait).
            self.stop()
            raise

    def _is_healthy(self) -> bool:
        try:
            r = httpx.get(f"{self.cfg.everos_base_url}/health", timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __enter__(self) -> EverOSServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
