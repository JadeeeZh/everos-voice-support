"""ExperienceAgent: the stock tau2 LLMAgent + EverOS retrieval.

The difference from the baseline agent is memory retrieval rendered into a
`<learned_experience>` system-prompt block. Same model, same policy, same
tools, same loop. Two retrieval points:

- first customer utterance (always, v1 behavior), and
- optionally a second, fixed-schedule retrieval on the Nth customer
  utterance (``midcall_retrieval_user_turn``), querying with the symptoms
  and findings accumulated so far. The schedule is fixed per run — never
  conditioned on how the call is going — so the everos arm cannot
  cherry-pick when to consult memory.
"""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)

EXPERIENCE_WRAPPER = """<learned_experience>
Operational experience distilled from your own previous support calls (across
many different customers). Use it to prioritize likely root causes and avoid
steps that historically did not help for this kind of issue. It is advisory:
always verify with live diagnostics and never violate the policy above.

{body}
</learned_experience>"""

MIDCALL_HEADER = "## Updated recall (matched against findings so far)"

# Query budget for the mid-call retrieval: opening symptom + latest findings.
_MIDCALL_HEAD_CHARS = 300
_MIDCALL_TAIL_CHARS = 900

ExperienceProvider = Callable[[str], "str | None"]


class ExperienceAgent(LLMAgent[LLMAgentState]):
    """LLMAgent that injects retrieved experience during the call."""

    def __init__(
        self,
        tools,
        domain_policy: str,
        llm: str,
        llm_args: dict | None = None,
        experience_provider: ExperienceProvider | None = None,
        midcall_retrieval_user_turn: int | None = None,
    ):
        super().__init__(
            tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args
        )
        self._provider = experience_provider
        self._midcall_turn = midcall_retrieval_user_turn
        self._utterances: list[str] = []
        self._midcall_attempted = False
        self.retrievals: list[dict] = []
        self.retrieval_query: str | None = None
        self.experience_block: str | None = None
        self.retrieval_error: str | None = None

    @property
    def system_prompt(self) -> str:
        base = super().system_prompt
        if self.experience_block:
            base = f"{base}\n\n{EXPERIENCE_WRAPPER.format(body=self.experience_block)}"
        return base

    def _retrieve(self, stage: str, query: str) -> str | None:
        """One provider call; never breaks the live call, but records errors
        (evaluate.py fails the arm on any recorded error — a degraded
        retrieval must not silently turn this arm into the baseline)."""
        block: str | None = None
        try:
            block = self._provider(query)  # type: ignore[misc]
        except Exception as exc:
            logger.warning(f"experience retrieval ({stage}) failed: {exc}")
            self.retrieval_error = repr(exc)
        self.retrievals.append(
            {
                "stage": stage,
                "user_turn": len(self._utterances),
                "query": query,
                "block": block,
            }
        )
        return block

    def _midcall_query(self) -> str:
        head = self._utterances[0][:_MIDCALL_HEAD_CHARS]
        tail = " ".join(self._utterances[1:])[-_MIDCALL_TAIL_CHARS:]
        return f"{head} {tail}".strip()

    def generate_next_message(
        self, message, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        if (
            self._provider is not None
            and isinstance(message, UserMessage)
            and isinstance(message.content, str)
            and message.content.strip()
        ):
            self._utterances.append(message.content.strip())
            updated = False
            if len(self._utterances) == 1:
                self.retrieval_query = self._utterances[0]
                self.experience_block = self._retrieve("first", self.retrieval_query)
                updated = self.experience_block is not None
            elif (
                self._midcall_turn is not None
                and not self._midcall_attempted
                and len(self._utterances) >= self._midcall_turn
            ):
                self._midcall_attempted = True
                block = self._retrieve("midcall", self._midcall_query())
                if block:
                    base = self.experience_block or ""
                    self.experience_block = (
                        f"{base}\n\n{MIDCALL_HEADER}\n\n{block}" if base else block
                    )
                    updated = True
            if updated:
                state.system_messages = [
                    SystemMessage(role="system", content=self.system_prompt)
                ]
        return super().generate_next_message(message, state)
