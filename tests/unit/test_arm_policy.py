"""Unit tests for ROLE-012 arm policy: Groq→Ollama fallback and NON-REPRODUCIBLE."""
from unittest.mock import MagicMock, patch

import pytest
from groq import RateLimitError

from role.persona import OLLAMA_MODEL, GROQ_MODEL, PersonaAgent


SYSTEM_PROMPT = "You are a customer named Alex."


def _make_agent() -> PersonaAgent:
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        with patch("role.persona.Groq"):
            return PersonaAgent(SYSTEM_PROMPT)


def _groq_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = text
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


def _rate_limit_error() -> RateLimitError:
    return RateLimitError.__new__(RateLimitError)


# ---------------------------------------------------------------------------
# Fallback: Groq 429 → Ollama
# ---------------------------------------------------------------------------

def test_reply_falls_back_to_ollama_on_429(capsys):
    agent = _make_agent()
    agent._client.chat.completions.create.side_effect = _rate_limit_error()

    with patch("role.persona._ollama_chat", return_value=("Ollama reply", 0.5)) as mock_ollama:
        result = agent.reply("hello")

    assert result == "Ollama reply"
    mock_ollama.assert_called_once()
    captured = capsys.readouterr()
    assert "arm-policy" in captured.out
    assert "Ollama" in captured.out


def test_reply_uses_groq_when_no_error():
    agent = _make_agent()
    agent._client.chat.completions.create.return_value = _groq_response("Groq reply")

    with patch("role.persona._ollama_chat") as mock_ollama:
        result = agent.reply("hello")

    assert result == "Groq reply"
    mock_ollama.assert_not_called()


# ---------------------------------------------------------------------------
# repeat(): NON-REPRODUCIBLE when runs disagree
# ---------------------------------------------------------------------------

def test_repeat_returns_text_when_all_agree():
    agent = _make_agent()
    agent._client.chat.completions.create.return_value = _groq_response("same reply")

    result = agent.repeat("what do you think?", n=3)

    assert result == "same reply"


def test_repeat_returns_none_and_prints_non_reproducible_when_disagree(capsys):
    agent = _make_agent()
    replies = ["reply A", "reply B", "reply A"]
    agent._client.chat.completions.create.side_effect = [
        _groq_response(r) for r in replies
    ]

    result = agent.repeat("what do you think?", n=3)

    assert result is None
    captured = capsys.readouterr()
    assert "NON-REPRODUCIBLE" in captured.out


def test_repeat_falls_back_to_ollama_on_429(capsys):
    agent = _make_agent()
    agent._client.chat.completions.create.side_effect = _rate_limit_error()

    with patch("role.persona._ollama_chat", return_value=("ollama reply", 0.3)):
        result = agent.repeat("hello", n=3)

    # All 3 ollama replies are identical → reproducible
    assert result == "ollama reply"
