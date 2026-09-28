"""Unit tests for ROLE-025 BoundaryChecker: arm policy and case-list shape.

No live API calls — instructor.from_groq/from_openai are mocked at the module
boundary so nothing here touches the network or the Groq daily quota.
"""
from unittest.mock import MagicMock, patch

import pytest
from groq import RateLimitError

from role.boundary_check import (
    BOUNDARY_CASES,
    GROQ_API_MODEL_ID,
    GROQ_MODEL,
    OLLAMA_MODEL,
    BoundaryChecker,
    BoundaryVerdict,
)
from role.logger import SessionLogger


def _verdict_result(verdict: str = "COMPLIANT") -> BoundaryVerdict:
    return BoundaryVerdict(verdict=verdict, rationale="because")


def _groq_completion() -> MagicMock:
    completion = MagicMock()
    completion.usage.prompt_tokens = 50
    completion.usage.completion_tokens = 20
    return completion


def _rate_limit_error() -> RateLimitError:
    return RateLimitError.__new__(RateLimitError)


def _make_checker() -> BoundaryChecker:
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        with patch("role.boundary_check.instructor.from_groq") as mock_from_groq:
            checker = BoundaryChecker()
            checker._mock_from_groq = mock_from_groq  # stash for assertions
    return checker


# ---------------------------------------------------------------------------
# case list
# ---------------------------------------------------------------------------


def test_boundary_cases_has_ten_cases():
    assert len(BOUNDARY_CASES) == 10
    assert len(set(BOUNDARY_CASES)) == 10  # no duplicates


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_missing_groq_api_key_raises():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            BoundaryChecker()


# ---------------------------------------------------------------------------
# classify(): Groq happy path
# ---------------------------------------------------------------------------


def test_classify_uses_groq_when_no_error():
    checker = _make_checker()
    checker._client.chat.completions.create_with_completion.return_value = (
        _verdict_result("COMPLIANT"),
        _groq_completion(),
    )

    verdict = checker.classify("Are you happy?", "Let's focus on your order.")

    assert verdict.verdict == "COMPLIANT"
    checker._client.chat.completions.create_with_completion.assert_called_once()
    _, kwargs = checker._client.chat.completions.create_with_completion.call_args
    assert kwargs["model"] == GROQ_API_MODEL_ID


def test_classify_logs_safeguard_turn_with_cost_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("role.logger.SESSIONS_DIR", tmp_path)
    checker = _make_checker()
    checker._client.chat.completions.create_with_completion.return_value = (
        _verdict_result("VIOLATION"),
        _groq_completion(),
    )
    logger = SessionLogger(session_id="test-session")

    checker.classify("Are you happy?", "I feel great today!", logger=logger)

    lines = (tmp_path / "test-session.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    import json as _json

    record = _json.loads(lines[0])
    assert record["role"] == "safeguard"
    assert record["model"] == GROQ_MODEL
    assert record["prompt_tokens"] == 50
    assert record["completion_tokens"] == 20
    assert "VIOLATION" in record["content"]


# ---------------------------------------------------------------------------
# classify(): Groq 429 -> Ollama fallback
# ---------------------------------------------------------------------------


def test_classify_falls_back_to_ollama_on_429(capsys):
    checker = _make_checker()
    checker._client.chat.completions.create_with_completion.side_effect = _rate_limit_error()

    with patch("role.boundary_check.instructor.from_openai") as mock_from_openai:
        mock_from_openai.return_value.chat.completions.create_with_completion.return_value = (
            _verdict_result("COMPLIANT"),
            None,
        )
        verdict = checker.classify("Are you happy?", "Let's focus on your order.")

    assert verdict.verdict == "COMPLIANT"
    captured = capsys.readouterr()
    assert "arm-policy" in captured.out
    assert "Ollama" in captured.out


def test_classify_ollama_fallback_logs_ollama_model(tmp_path, monkeypatch):
    monkeypatch.setattr("role.logger.SESSIONS_DIR", tmp_path)
    checker = _make_checker()
    checker._client.chat.completions.create_with_completion.side_effect = _rate_limit_error()
    logger = SessionLogger(session_id="test-session")

    with patch("role.boundary_check.instructor.from_openai") as mock_from_openai:
        mock_from_openai.return_value.chat.completions.create_with_completion.return_value = (
            _verdict_result("COMPLIANT"),
            None,
        )
        checker.classify("Are you happy?", "Let's focus on your order.", logger=logger)

    lines = (tmp_path / "test-session.jsonl").read_text(encoding="utf-8").strip().splitlines()
    import json as _json

    record = _json.loads(lines[0])
    assert record["model"] == OLLAMA_MODEL
    assert record["prompt_tokens"] is None
