"""Unit tests for ROLE-22: an uncited score is unrepresentable in the Scorecard schema."""

import pydantic
import pytest

from role.scorecard import RUBRIC_ITEMS, ScoreItem, Scorecard, TurnRef


def _valid_items(score: int = 2) -> dict:
    return {name: {"score": score, "evidence": [{"turn_id": "sess1:1"}]} for name in RUBRIC_ITEMS}


def test_score_item_rejects_empty_evidence():
    with pytest.raises(pydantic.ValidationError) as excinfo:
        ScoreItem(score=2, evidence=[])
    assert excinfo.value.errors()[0]["loc"] == ("evidence",)


def test_score_item_accepts_single_evidence():
    item = ScoreItem(score=1, evidence=[{"turn_id": "sess1:3"}])
    assert item.evidence[0].turn_id == "sess1:3"


def test_scorecard_rejects_uncited_score():
    items = _valid_items()
    items["clarity"]["evidence"] = []
    with pytest.raises(pydantic.ValidationError):
        Scorecard(session_id="sess1", rubric_version="evals/RUBRIC.md", items=items)


def test_scorecard_valid_round_trip():
    original = Scorecard(session_id="sess1", rubric_version="evals/RUBRIC.md", items=_valid_items())
    rebuilt = Scorecard.model_validate(original.model_dump(mode="json"))
    assert rebuilt == original
    assert rebuilt.total == 12
    assert rebuilt.max == 12


def test_scorecard_rejects_missing_rubric_item():
    items = _valid_items()
    del items["clarity"]
    with pytest.raises(pydantic.ValidationError):
        Scorecard(session_id="sess1", rubric_version="evals/RUBRIC.md", items=items)


def test_scorecard_rejects_bad_turn_id_format():
    with pytest.raises(pydantic.ValidationError):
        TurnRef(turn_id="not-a-turn-id")


def test_scorecard_rejects_out_of_range_score():
    with pytest.raises(pydantic.ValidationError):
        ScoreItem(score=3, evidence=[{"turn_id": "sess1:1"}])
