"""Derive the v2 agent-skill consolidation prompt from the stock everalgo one.

v1 diagnosis (see PLAN.md "v2 levers"): the stock success-path prompt
consolidates the *majority* pattern of a cluster and caps pitfalls, so the
minority lessons carried by failed-then-corrected calls get averaged away —
in v1 the consolidated skill even codified the exact trap as a decision
branch ("if <check> passes -> proceed").

v2 is the stock prompt plus surgical, domain-agnostic additions that force
contrastive (failure-derived) insights to survive consolidation. Deriving at
runtime from the installed constant (instead of freezing a copy) keeps the
diff honest and auditable: run this module to print the derived prompt.

Every edit is anchored to an exact substring of the stock prompt and fails
hard if the anchor is missing or ambiguous, so an everalgo upgrade can never
silently produce a half-patched prompt.
"""

from __future__ import annotations

_GOOD_SKILL_ANCHOR = (
    "- A FEW well-chosen examples that illustrate distinct branches "
    "— not an exhaustive catalog"
)
_GOOD_SKILL_ADDITION = (
    "- Preserves hard-won corrections: the checks that distinguish look-alike "
    "situations, learned from cases that initially failed or were escalated"
)

_FIELD_REQ_ANCHOR = "**Field-level requirements:**"
_CONTRASTIVE_RULES = """**Contrastive insight preservation (HARD RULES):**
- When a case's key_insight or close-out note reveals a root cause that the
  standard checks missed (the attempt failed or was escalated before the true
  cause was found), the distinguishing check MUST appear as a numbered Step or
  a Decision branch — not only as a Pitfall — and MUST survive every later
  condensation or update.
- Never write a Decision branch of the form "if <check> passes → proceed" when
  any case shows <check> passing while the problem persisted. Name the extra
  check that separates those look-alike situations.
- When similar symptoms were resolved by different fixes across cases, state
  which observation or question tells the variants apart — listing both fixes
  without the discriminator is not acceptable.
- When one capability is governed by multiple independent controls (for
  example a server-side setting and a client-side setting), verifying one
  control does NOT verify the others: direct each control to be checked
  independently.

"""

_PITFALL_CAP_ANCHOR = (
    "  - **Max 4 pitfalls.** When adding a new one beyond 4, "
    "replace the most generic existing pitfall."
)
_PITFALL_CAP_V2 = (
    "  - **Max 6 pitfalls.** When adding a new one beyond 6, replace the most "
    "generic existing pitfall. A pitfall derived from a failed or escalated "
    "case MUST NOT be evicted in favor of generic advice."
)

_PRESERVE_ANCHOR = (
    "**preserve existing verified content unless the new case "
    "directly contradicts it**."
)
_PRESERVE_V2 = _PRESERVE_ANCHOR + (
    " Contrastive checks and failure-derived pitfalls are load-bearing: "
    "preserve or strengthen them, never summarize them away."
)


def _replace_once(text: str, anchor: str, replacement: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"skill prompt v2: anchor found {count} times (expected 1): "
            f"{anchor[:60]!r} — everalgo prompt text changed; re-verify the diff."
        )
    return text.replace(anchor, replacement)


def derive(stock_prompt: str) -> str:
    """Return the v2 prompt: stock text + anchored contrastive-insight rules."""
    out = stock_prompt
    out = _replace_once(
        out, _GOOD_SKILL_ANCHOR, _GOOD_SKILL_ANCHOR + "\n" + _GOOD_SKILL_ADDITION
    )
    out = _replace_once(
        out, _FIELD_REQ_ANCHOR, _CONTRASTIVE_RULES + _FIELD_REQ_ANCHOR
    )
    out = _replace_once(out, _PITFALL_CAP_ANCHOR, _PITFALL_CAP_V2)
    out = _replace_once(out, _PRESERVE_ANCHOR, _PRESERVE_V2)
    for placeholder in ("{new_case_json}", "{existing_skills_json}"):
        if placeholder not in out:
            raise RuntimeError(f"skill prompt v2: lost placeholder {placeholder}")
    return out


if __name__ == "__main__":
    from everalgo.agent_memory.skill import AGENT_SKILL_SUCCESS_EXTRACT_PROMPT

    print(derive(AGENT_SKILL_SUCCESS_EXTRACT_PROMPT))
