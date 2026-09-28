"""Evaluator agent — Qwen/Qwen3.8-27B via Groq+instructor, with Ollama fallback.

Uses its own, separate Groq client instance from role.persona.PersonaAgent's —
never import or share that client object. Scoring a session with the same
model that played the persona inflates agreement scores (self-preference
bias, ROLE-008) — a distinct model, prompt, and client are the whole point.

Arm policy mirrors ROLE-012: Groq 429/daily-cap falls back to Ollama, with a
printed [arm-policy] warning, same as persona.py.
"""

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import instructor
from groq import Groq, RateLimitError
from openai import OpenAI
from pydantic import BaseModel, model_validator

from role import logger as _logger_mod
from role.scorecard import RUBRIC_ITEMS, Scorecard, ScoreItem

if TYPE_CHECKING:
    from role.logger import SessionLogger

GROQ_MODEL = "Qwen/Qwen3.8-27B"
# Verified live against Groq's model list (client.models.list()) — Groq hosts
# `qwen/qwen3.8-27b`, not the ticket's original "Qwen/Qwen3.6-27B" or
# ARCHITECTURE.md's older "Qwen/Qwen3-27B" (both 404 model_not_found).
GROQ_API_MODEL_ID = "qwen/qwen3.8-27b"
# NOTE: this GGUF's actual `ollama pull` availability is unverified — persona.py's
# own Ollama fallback already uses a DIFFERENT model than its Groq primary for
# exactly this reason (no guaranteed matching local quant). Swap if needed.
OLLAMA_MODEL = "hf.co/Qwen/Qwen3.8-27B"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MAX_RETRIES = 3
RUBRIC_VERSION = "evals/RUBRIC.md"

RUBRIC_PATH = Path(__file__).parent.parent.parent / "evals" / "RUBRIC.md"
PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "evaluator_v1.md"

RUBRIC_PLACEHOLDER = "{{RUBRIC}}"
TRANSCRIPT_PLACEHOLDER = "{{TRANSCRIPT}}"
SESSION_ID_PLACEHOLDER = "{{SESSION_ID}}"


class _RubricScores(BaseModel):
    """What the model actually has to judge — session_id/rubric_version are
    known bookkeeping values the caller already has; asking the model to also
    invent them would waste retries on trivial string mismatches instead of
    on real scoring quality.

    Carries the same missing/extra rubric-key check as Scorecard: without it,
    a dict[str, ScoreItem] validates fine even when the model returns the
    wrong keys or too few of them, so instructor never sees a validation
    failure to retry on — the incompleteness silently surfaces later instead,
    when Scorecard's own check finally runs after the LLM call.
    """

    items: dict[str, ScoreItem]

    @model_validator(mode="after")
    def _check_items(self) -> "_RubricScores":
        missing = set(RUBRIC_ITEMS) - set(self.items)
        if missing:
            raise ValueError(f"items is missing required rubric keys: {sorted(missing)}")
        extra = set(self.items) - set(RUBRIC_ITEMS)
        if extra:
            raise ValueError(f"items has unknown keys not in the rubric: {sorted(extra)}")
        return self


class EvaluatorAgent:
    def __init__(self, prompt_template: "str | None" = None) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set — run `make doctor`")
        # Mode.JSON, not Mode.TOOLS: Qwen/Qwen3.8-27B's Groq tool-calling was
        # unreliable on this schema (garbled function-call output, "Failed to
        # call a function" 400s) — direct JSON-object generation held up
        # instead. See notes/phase-1b-evaluator-findings.md.
        self._client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.JSON)
        self._prompt_template = prompt_template or PROMPT_PATH.read_text(encoding="utf-8")
        self._rubric_text = RUBRIC_PATH.read_text(encoding="utf-8")

    def _read_transcript(self, session_id: str) -> str:
        """Session log -> turn_id-tagged text.

        Exposes only role/content/turn_id per turn — never model, tokens,
        cost, or controller_state (the Evaluator does not see cost/latency
        metadata or scenario difficulty, ARCHITECTURE.md §3).
        """
        path = _logger_mod.SESSIONS_DIR / f"{session_id}.jsonl"
        records = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not records:
            raise ValueError(f"sessions/{session_id}.jsonl has no turns to evaluate")
        return "\n".join(f"[{r['turn_id']}] {r['role']}: {r['content']}" for r in records)

    def _render_prompt(self, session_id: str, transcript_text: str) -> str:
        text = self._prompt_template.replace(RUBRIC_PLACEHOLDER, self._rubric_text)
        text = text.replace(TRANSCRIPT_PLACEHOLDER, transcript_text)
        return text.replace(SESSION_ID_PLACEHOLDER, session_id)

    def evaluate(self, session_id: str, logger: "SessionLogger | None" = None) -> Scorecard:
        prompt = self._render_prompt(session_id, self._read_transcript(session_id))
        messages = [{"role": "user", "content": prompt}]

        model_used = GROQ_MODEL
        prompt_tokens = completion_tokens = None

        t0 = time.perf_counter()
        try:
            result, completion = self._client.chat.completions.create_with_completion(
                model=GROQ_API_MODEL_ID,
                messages=messages,
                response_model=_RubricScores,
                max_retries=MAX_RETRIES,
                temperature=0.0,
            )
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
        except RateLimitError as exc:
            print(
                f"[arm-policy] Groq 429/cap — evaluator falling back to Ollama"
                f" ({OLLAMA_MODEL}): {exc}"
            )
            ollama_client = instructor.from_openai(
                OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama"), mode=instructor.Mode.JSON
            )
            result, _ = ollama_client.chat.completions.create_with_completion(
                model=OLLAMA_MODEL,
                messages=messages,
                response_model=_RubricScores,
                max_retries=MAX_RETRIES,
                temperature=0.0,
            )
            model_used = OLLAMA_MODEL
        elapsed = time.perf_counter() - t0

        scorecard = Scorecard(
            session_id=session_id,
            rubric_version=RUBRIC_VERSION,
            items=result.items,
        )

        if logger is not None:
            logger.log(
                role="evaluator",
                content=scorecard.model_dump_json(),
                model=model_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                seconds=elapsed,
            )
        return scorecard
