"""Coach agent — openai/gpt-oss-20b via Groq+instructor, with Ollama fallback.

Reads a Scorecard (rubric item names + scores + evidence TurnRef ids) and
writes a learning plan. Never given the raw transcript, the persona system
prompt, or evals/RUBRIC.md's full wording — per ARCHITECTURE.md's "who sees
what" table, the Coach sees only the Scorecard.

Every plan item must cite one of RUBRIC_ITEMS. A plan item citing anything
else is unrepresentable in the type — LearningPlan's validator rejects it,
same discipline as role.scorecard.Scorecard's evidence requirement.

Uses its own Groq client instance, separate from PersonaAgent's and
EvaluatorAgent's — same non-self-scoring principle (ROLE-008).
"""

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import instructor
from groq import Groq, RateLimitError
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

from role.scorecard import RUBRIC_ITEMS, Scorecard

if TYPE_CHECKING:
    from role.logger import SessionLogger

GROQ_MODEL = "openai/gpt-oss-20b"
# Verified live against Groq's model list before writing this module — no
# casing fixup needed, same family as openai/gpt-oss-120b.
GROQ_API_MODEL_ID = "openai/gpt-oss-20b"
OLLAMA_MODEL = "hf.co/openai/gpt-oss-20b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MAX_RETRIES = 3

PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "coach_v1.md"

SESSION_ID_PLACEHOLDER = "{{SESSION_ID}}"
SCORED_ITEMS_PLACEHOLDER = "{{SCORED_ITEMS}}"


class PlanItem(BaseModel):
    """One learning-plan recommendation, tied to one scored rubric item."""

    rubric_item: str
    recommendation: str


def _check_known_rubric_items(items: list[PlanItem]) -> None:
    unknown = sorted({item.rubric_item for item in items} - set(RUBRIC_ITEMS))
    if unknown:
        raise ValueError(
            f"learning plan cites unknown rubric item(s) not in RUBRIC_ITEMS: {unknown}"
        )


class _PlanItems(BaseModel):
    """What the model actually has to produce — session_id is a known
    bookkeeping value the caller already has (same trick as role.evaluator's
    _RubricScores)."""

    items: list[PlanItem] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_items(self) -> "_PlanItems":
        _check_known_rubric_items(self.items)
        return self


class LearningPlan(BaseModel):
    """A session's learning plan; every item must cite a real rubric item —
    an item citing an unscored claim is unrepresentable in this type."""

    session_id: str
    items: list[PlanItem] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_items(self) -> "LearningPlan":
        _check_known_rubric_items(self.items)
        return self


class CoachAgent:
    def __init__(self, prompt_template: "str | None" = None) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set — run `make doctor`")
        self._client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.JSON)
        self._prompt_template = prompt_template or PROMPT_PATH.read_text(encoding="utf-8")

    def _render_scored_items(self, scorecard: Scorecard) -> str:
        lines = []
        for name in RUBRIC_ITEMS:
            item = scorecard.items[name]
            turn_ids = ", ".join(ref.turn_id for ref in item.evidence)
            lines.append(f"- {name}: score {item.score}/2 (evidence: {turn_ids})")
        return "\n".join(lines)

    def _render_prompt(self, scorecard: Scorecard) -> str:
        text = self._prompt_template.replace(SESSION_ID_PLACEHOLDER, scorecard.session_id)
        return text.replace(SCORED_ITEMS_PLACEHOLDER, self._render_scored_items(scorecard))

    def plan(self, scorecard: Scorecard, logger: "SessionLogger | None" = None) -> LearningPlan:
        prompt = self._render_prompt(scorecard)
        messages = [{"role": "user", "content": prompt}]

        model_used = GROQ_MODEL
        prompt_tokens = completion_tokens = None

        t0 = time.perf_counter()
        try:
            result, completion = self._client.chat.completions.create_with_completion(
                model=GROQ_API_MODEL_ID,
                messages=messages,
                response_model=_PlanItems,
                max_retries=MAX_RETRIES,
                temperature=0.0,
            )
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
        except RateLimitError as exc:
            print(
                f"[arm-policy] Groq 429/cap — coach falling back to Ollama"
                f" ({OLLAMA_MODEL}): {exc}"
            )
            ollama_client = instructor.from_openai(
                OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama"), mode=instructor.Mode.JSON
            )
            result, _ = ollama_client.chat.completions.create_with_completion(
                model=OLLAMA_MODEL,
                messages=messages,
                response_model=_PlanItems,
                max_retries=MAX_RETRIES,
                temperature=0.0,
            )
            model_used = OLLAMA_MODEL
        elapsed = time.perf_counter() - t0

        plan = LearningPlan(session_id=scorecard.session_id, items=result.items)

        if logger is not None:
            logger.log(
                role="coach",
                content=plan.model_dump_json(),
                model=model_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                seconds=elapsed,
            )
        return plan


def render_markdown(plan: LearningPlan) -> str:
    """Render a LearningPlan as the evals/plans/{session_id}.md file ARCHITECTURE.md
    §5 describes."""
    lines = [f"# Learning Plan — {plan.session_id}", ""]
    for item in plan.items:
        lines.append(f"## {item.rubric_item}")
        lines.append(item.recommendation)
        lines.append("")
    return "\n".join(lines)
