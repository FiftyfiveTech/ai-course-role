"""Scorecard schema: every score must cite >=1 TurnRef — an uncited score cannot be constructed."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

RUBRIC_ITEMS: tuple[str, ...] = (
    "question_coverage",
    "listening_ratio",
    "factual_accuracy",
    "policy_adherence",
    "objection_handling",
    "clarity",
)


class TurnRef(BaseModel):
    """A citation into a session transcript: one turn_id from sessions/<id>.jsonl."""

    turn_id: str = Field(pattern=r"^[^:]+:\d+$")  # matches logger.py's f"{session_id}:{turn_index}"


class ScoreItem(BaseModel):
    """One rubric item's score; evidence is mandatory — an empty list is rejected."""

    score: Literal[0, 1, 2]
    evidence: list[TurnRef] = Field(min_length=1)


class Scorecard(BaseModel):
    """A session's rubric scores; total/max are derived, never trusted from input."""

    session_id: str
    rubric_version: str
    items: dict[str, ScoreItem]

    @model_validator(mode="after")
    def _check_items(self) -> "Scorecard":
        missing = set(RUBRIC_ITEMS) - set(self.items)
        if missing:
            raise ValueError(f"Scorecard missing rubric items: {sorted(missing)}")
        extra = set(self.items) - set(RUBRIC_ITEMS)
        if extra:
            raise ValueError(f"Scorecard has unknown rubric items: {sorted(extra)}")
        return self

    @property
    def total(self) -> int:
        return sum(item.score for item in self.items.values())

    @property
    def max(self) -> int:
        return len(self.items) * 2
