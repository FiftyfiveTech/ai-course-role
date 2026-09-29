"""Unit tests for ROLE-030 render_scorecard_html: every score must show the
real transcript text of every turn it cites, not just the turn_id."""

import pytest

from role.coach import LearningPlan, PlanItem
from role.scorecard_html import render_scorecard_html
from role.scorecard import RUBRIC_ITEMS, Scorecard, ScoreItem


def _write_session_log(tmp_path, session_id: str) -> None:
    lines = [
        {"turn_id": f"{session_id}:1", "session_id": session_id, "turn_index": 1,
         "role": "persona", "content": "My order hasn't shipped yet."},
        {"turn_id": f"{session_id}:2", "session_id": session_id, "turn_index": 2,
         "role": "trainee", "content": "I hear you — let me pull up your order right away."},
    ]
    import json

    (tmp_path / f"{session_id}.jsonl").write_text(
        "\n".join(json.dumps(ln) for ln in lines) + "\n", encoding="utf-8"
    )


def _scorecard(session_id: str) -> Scorecard:
    return Scorecard(
        session_id=session_id,
        rubric_version="evals/RUBRIC.md",
        items={
            name: ScoreItem(score=2 if name == "listening_ratio" else 0, evidence=[{"turn_id": f"{session_id}:2"}])
            for name in RUBRIC_ITEMS
        },
    )


def _plan(session_id: str) -> LearningPlan:
    return LearningPlan(
        session_id=session_id,
        items=[PlanItem(rubric_item="clarity", recommendation="Be more explicit about next steps.")],
    )


def test_render_includes_every_rubric_item_and_score(tmp_path):
    _write_session_log(tmp_path, "s1")
    html = render_scorecard_html(_scorecard("s1"), _plan("s1"), sessions_dir=tmp_path)

    for name in RUBRIC_ITEMS:
        assert name in html
    assert "2/2" in html  # listening_ratio's score
    assert "0/2" in html  # every other item's score


def test_render_includes_real_cited_turn_text_not_just_id(tmp_path):
    _write_session_log(tmp_path, "s1")
    html = render_scorecard_html(_scorecard("s1"), _plan("s1"), sessions_dir=tmp_path)

    assert "s1:2" in html
    assert "I hear you" in html  # the actual cited turn's content, not just its id


def test_render_includes_learning_plan_content(tmp_path):
    _write_session_log(tmp_path, "s1")
    html = render_scorecard_html(_scorecard("s1"), _plan("s1"), sessions_dir=tmp_path)

    assert "clarity" in html
    assert "Be more explicit about next steps." in html


def test_render_raises_on_dangling_turn_id(tmp_path):
    _write_session_log(tmp_path, "s1")
    scorecard = _scorecard("s1")
    scorecard.items["clarity"].evidence[0].turn_id = "s1:99"  # dangling

    with pytest.raises(ValueError, match="not found"):
        render_scorecard_html(scorecard, _plan("s1"), sessions_dir=tmp_path)


def test_render_raises_on_session_id_mismatch(tmp_path):
    _write_session_log(tmp_path, "s1")
    with pytest.raises(ValueError, match="session_id"):
        render_scorecard_html(_scorecard("s1"), _plan("other-session"), sessions_dir=tmp_path)
