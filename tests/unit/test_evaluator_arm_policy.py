"""Unit tests for ROLE-023 EvaluatorAgent: arm policy and transcript rendering.

No live API calls — instructor.from_groq/from_openai are mocked at the module
boundary so nothing here touches the network or the Groq daily quota.
"""
from unittest.mock import MagicMock, patch

import pytest
from groq import RateLimitError

from role.evaluator import GROQ_API_MODEL_ID, GROQ_MODEL, OLLAMA_MODEL, EvaluatorAgent
from role.logger import SessionLogger
from role.scorecard import RUBRIC_ITEMS, ScoreItem


def _valid_items(session_id: str) -> dict:
    return {
        item: ScoreItem(score=2, evidence=[{"turn_id": f"{session_id}:1"}])
        for item in RUBRIC_ITEMS
    }


def _rubric_scores_result(session_id: str) -> MagicMock:
    result = MagicMock()
    result.items = _valid_items(session_id)
    return result


def _groq_completion() -> MagicMock:
    completion = MagicMock()
    completion.usage.prompt_tokens = 100
    completion.usage.completion_tokens = 40
    return completion


def _rate_limit_error() -> RateLimitError:
    return RateLimitError.__new__(RateLimitError)


def _make_agent() -> EvaluatorAgent:
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        with patch("role.evaluator.instructor.from_groq") as mock_from_groq:
            agent = EvaluatorAgent()
            agent._mock_from_groq = mock_from_groq  # stash for assertions
    return agent


def _seed_session(tmp_path, monkeypatch, session_id: str = "test-session") -> str:
    monkeypatch.setattr("role.logger.SESSIONS_DIR", tmp_path)
    logger = SessionLogger(session_id=session_id)
    logger.log(role="persona", content="My order 123 hasn't arrived.")
    logger.log(role="trainee", content="I'm sorry to hear that — let me look into order 123.")
    return session_id


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_missing_groq_api_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            EvaluatorAgent()


# ---------------------------------------------------------------------------
# evaluate(): Groq happy path
# ---------------------------------------------------------------------------

def test_evaluate_uses_groq_when_no_error(tmp_path, monkeypatch):
    session_id = _seed_session(tmp_path, monkeypatch)
    agent = _make_agent()
    agent._client.chat.completions.create_with_completion.return_value = (
        _rubric_scores_result(session_id),
        _groq_completion(),
    )

    scorecard = agent.evaluate(session_id)

    assert scorecard.session_id == session_id
    assert scorecard.rubric_version == "evals/RUBRIC.md"
    assert set(scorecard.items) == set(RUBRIC_ITEMS)
    agent._client.chat.completions.create_with_completion.assert_called_once()
    _, kwargs = agent._client.chat.completions.create_with_completion.call_args
    assert kwargs["model"] == GROQ_API_MODEL_ID


def test_evaluate_logs_evaluator_turn_with_cost_fields(tmp_path, monkeypatch):
    session_id = _seed_session(tmp_path, monkeypatch)
    agent = _make_agent()
    agent._client.chat.completions.create_with_completion.return_value = (
        _rubric_scores_result(session_id),
        _groq_completion(),
    )
    logger = SessionLogger(session_id=session_id)

    agent.evaluate(session_id, logger=logger)

    lines = (tmp_path / f"{session_id}.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # persona + trainee (seeded) + evaluator
    import json as _json
    evaluator_record = _json.loads(lines[-1])
    assert evaluator_record["role"] == "evaluator"
    assert evaluator_record["model"] == GROQ_MODEL
    assert evaluator_record["prompt_tokens"] == 100
    assert evaluator_record["completion_tokens"] == 40


# ---------------------------------------------------------------------------
# evaluate(): Groq 429 -> Ollama fallback
# ---------------------------------------------------------------------------

def test_evaluate_falls_back_to_ollama_on_429(tmp_path, monkeypatch, capsys):
    session_id = _seed_session(tmp_path, monkeypatch)
    agent = _make_agent()
    agent._client.chat.completions.create_with_completion.side_effect = _rate_limit_error()

    with patch("role.evaluator.instructor.from_openai") as mock_from_openai:
        mock_from_openai.return_value.chat.completions.create_with_completion.return_value = (
            _rubric_scores_result(session_id),
            None,
        )
        scorecard = agent.evaluate(session_id)

    assert scorecard.session_id == session_id
    captured = capsys.readouterr()
    assert "arm-policy" in captured.out
    assert "Ollama" in captured.out


def test_evaluate_ollama_fallback_logs_ollama_model(tmp_path, monkeypatch):
    session_id = _seed_session(tmp_path, monkeypatch)
    agent = _make_agent()
    agent._client.chat.completions.create_with_completion.side_effect = _rate_limit_error()
    logger = SessionLogger(session_id=session_id)

    with patch("role.evaluator.instructor.from_openai") as mock_from_openai:
        mock_from_openai.return_value.chat.completions.create_with_completion.return_value = (
            _rubric_scores_result(session_id),
            None,
        )
        agent.evaluate(session_id, logger=logger)

    lines = (tmp_path / f"{session_id}.jsonl").read_text(encoding="utf-8").strip().splitlines()
    import json as _json
    evaluator_record = _json.loads(lines[-1])
    assert evaluator_record["model"] == OLLAMA_MODEL
    assert evaluator_record["prompt_tokens"] is None


# ---------------------------------------------------------------------------
# transcript rendering
# ---------------------------------------------------------------------------

def test_read_transcript_tags_turn_ids_and_roles(tmp_path, monkeypatch):
    session_id = _seed_session(tmp_path, monkeypatch)
    agent = _make_agent()

    text = agent._read_transcript(session_id)

    assert f"[{session_id}:1] persona:" in text
    assert f"[{session_id}:2] trainee:" in text


def test_read_transcript_empty_session_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("role.logger.SESSIONS_DIR", tmp_path)
    (tmp_path / "empty-session.jsonl").write_text("", encoding="utf-8")
    agent = _make_agent()

    with pytest.raises(ValueError, match="no turns to evaluate"):
        agent._read_transcript("empty-session")
