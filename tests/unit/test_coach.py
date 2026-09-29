"""Unit tests for ROLE-027 CoachAgent: arm policy, rendering, and the core
guarantee — a learning plan citing a rubric item outside RUBRIC_ITEMS is
unrepresentable in the type.

No live API calls — instructor.from_groq/from_openai are mocked at the module
boundary so nothing here touches the network or the Groq daily quota.
"""
import logging
from unittest.mock import MagicMock, patch

import pytest
from groq import RateLimitError
from pydantic import ValidationError

from role.coach import (
    GROQ_API_MODEL_ID,
    GROQ_MODEL,
    OLLAMA_MODEL,
    CoachAgent,
    LearningPlan,
    PlanItem,
    render_markdown,
)
from role.logger import SessionLogger
from role.scorecard import RUBRIC_ITEMS, Scorecard, ScoreItem


def _scorecard(session_id: str = "test-session") -> Scorecard:
    return Scorecard(
        session_id=session_id,
        rubric_version="evals/RUBRIC.md",
        items={
            name: ScoreItem(score=0, evidence=[{"turn_id": f"{session_id}:1"}])
            for name in RUBRIC_ITEMS
        },
    )


def _plan_items_result(rubric_item: str = RUBRIC_ITEMS[0]) -> MagicMock:
    result = MagicMock()
    result.items = [PlanItem(rubric_item=rubric_item, recommendation="do better")]
    return result


def _groq_completion() -> MagicMock:
    completion = MagicMock()
    completion.usage.prompt_tokens = 80
    completion.usage.completion_tokens = 30
    return completion


def _rate_limit_error() -> RateLimitError:
    return RateLimitError.__new__(RateLimitError)


def _make_agent() -> CoachAgent:
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        with patch("role.coach.instructor.from_groq") as mock_from_groq:
            agent = CoachAgent()
            agent._mock_from_groq = mock_from_groq  # stash for assertions
    return agent


# ---------------------------------------------------------------------------
# the core guarantee: an unscored/unknown rubric item is unrepresentable
# ---------------------------------------------------------------------------


def test_learning_plan_rejects_unknown_rubric_item():
    with pytest.raises(ValidationError, match="unknown rubric item"):
        LearningPlan(
            session_id="s1",
            items=[PlanItem(rubric_item="typing_speed", recommendation="type faster")],
        )


def test_learning_plan_accepts_known_rubric_item():
    plan = LearningPlan(
        session_id="s1",
        items=[PlanItem(rubric_item="clarity", recommendation="be clearer")],
    )
    assert plan.items[0].rubric_item == "clarity"


def test_learning_plan_rejects_empty_items():
    with pytest.raises(ValidationError):
        LearningPlan(session_id="s1", items=[])


def test_coach_plan_raises_when_model_cites_unknown_rubric_item():
    agent = _make_agent()
    bad_result = MagicMock()
    bad_result.items = [PlanItem(rubric_item="typing_speed", recommendation="type faster")]
    agent._client.chat.completions.create_with_completion.return_value = (
        bad_result,
        _groq_completion(),
    )

    with pytest.raises(ValidationError, match="unknown rubric item"):
        agent.plan(_scorecard())


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_missing_groq_api_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            CoachAgent()


# ---------------------------------------------------------------------------
# plan(): Groq happy path
# ---------------------------------------------------------------------------


def test_plan_uses_groq_when_no_error():
    agent = _make_agent()
    agent._client.chat.completions.create_with_completion.return_value = (
        _plan_items_result(),
        _groq_completion(),
    )

    plan = agent.plan(_scorecard("hh-80002"))

    assert plan.session_id == "hh-80002"
    assert plan.items[0].rubric_item in RUBRIC_ITEMS
    agent._client.chat.completions.create_with_completion.assert_called_once()
    _, kwargs = agent._client.chat.completions.create_with_completion.call_args
    assert kwargs["model"] == GROQ_API_MODEL_ID


def test_plan_logs_coach_turn_with_cost_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("role.logger.SESSIONS_DIR", tmp_path)
    agent = _make_agent()
    agent._client.chat.completions.create_with_completion.return_value = (
        _plan_items_result(),
        _groq_completion(),
    )
    logger = SessionLogger(session_id="test-session")

    agent.plan(_scorecard(), logger=logger)

    lines = (tmp_path / "test-session.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    import json as _json

    record = _json.loads(lines[0])
    assert record["role"] == "coach"
    assert record["model"] == GROQ_MODEL
    assert record["prompt_tokens"] == 80
    assert record["completion_tokens"] == 30


# ---------------------------------------------------------------------------
# plan(): Groq 429 -> Ollama fallback
# ---------------------------------------------------------------------------


def test_plan_falls_back_to_ollama_on_429(caplog):
    agent = _make_agent()
    agent._client.chat.completions.create_with_completion.side_effect = _rate_limit_error()

    with caplog.at_level(logging.WARNING, logger="role.coach"):
        with patch("role.coach.instructor.from_openai") as mock_from_openai:
            mock_from_openai.return_value.chat.completions.create_with_completion.return_value = (
                _plan_items_result(),
                None,
            )
            plan = agent.plan(_scorecard())

    assert plan.items[0].rubric_item in RUBRIC_ITEMS
    assert "arm-policy" in caplog.text
    assert "Ollama" in caplog.text


def test_plan_ollama_fallback_logs_ollama_model(tmp_path, monkeypatch):
    monkeypatch.setattr("role.logger.SESSIONS_DIR", tmp_path)
    agent = _make_agent()
    agent._client.chat.completions.create_with_completion.side_effect = _rate_limit_error()
    logger = SessionLogger(session_id="test-session")

    with patch("role.coach.instructor.from_openai") as mock_from_openai:
        mock_from_openai.return_value.chat.completions.create_with_completion.return_value = (
            _plan_items_result(),
            None,
        )
        agent.plan(_scorecard(), logger=logger)

    lines = (tmp_path / "test-session.jsonl").read_text(encoding="utf-8").strip().splitlines()
    import json as _json

    record = _json.loads(lines[0])
    assert record["model"] == OLLAMA_MODEL
    assert record["prompt_tokens"] is None


# ---------------------------------------------------------------------------
# prompt rendering — never leaks the transcript, only names/scores/turn_ids
# ---------------------------------------------------------------------------


def test_render_prompt_contains_scores_and_evidence_not_transcript():
    agent = _make_agent()
    scorecard = _scorecard("hh-80002")

    prompt = agent._render_prompt(scorecard)

    assert "hh-80002" in prompt
    assert "clarity: score 0/2" in prompt
    assert "hh-80002:1" in prompt


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------


def test_render_markdown_includes_session_and_items():
    plan = LearningPlan(
        session_id="s1",
        items=[PlanItem(rubric_item="clarity", recommendation="be clearer")],
    )
    md = render_markdown(plan)
    assert "# Learning Plan — s1" in md
    assert "## clarity" in md
    assert "be clearer" in md
