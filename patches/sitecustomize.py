"""Env-gated prompt override, loaded via PYTHONPATH into the EverOS server.

The demo prepends this directory to PYTHONPATH when launching the EverOS
server with ``RunConfig.skill_prompt == "v2"`` (see everos_client.py), so the
interpreter imports this module at startup. everalgo re-exports its prompt
constants precisely for this startup patch — see the ``__all__`` comment in
``everalgo.agent_memory.skill``.

Inert unless ``VOICEDEMO_SKILL_PROMPT=v2`` is set. On any failure it raises,
killing the server at startup: silently running the stock prompt when v2 was
requested would corrupt the experiment.
"""

import os
import sys

if os.environ.get("VOICEDEMO_SKILL_PROMPT") == "v2":
    import everalgo.agent_memory.skill as _skill
    from skill_prompt_v2 import derive

    _skill.AGENT_SKILL_SUCCESS_EXTRACT_PROMPT = derive(
        _skill.AGENT_SKILL_SUCCESS_EXTRACT_PROMPT
    )
    print("[voicedemo] skill_success consolidation prompt: v2 override ACTIVE",
          file=sys.stderr)
