"""End-to-end test for ROLE-018: controller state flows into every session-log turn.

Mocks builtins.input (scripted trainee lines) and role.persona.Groq (canned
replies) — no live inference, exercises the full role.session.run() loop.
"""
import json
from unittest.mock import MagicMock, patch

import role.session as session_mod
from role.scenario import load, SCENARIOS_DIR


def _groq_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = text
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


TRAINEE_LINES = [
    "Hi, I need help with my order.",
    "Can you repeat that?",                               # escalate trigger
    "One more time, say that again?",                     # escalate, capped for easy
    "Sorry, I understand, let me fix that right away.",    # de-escalate trigger
]

PERSONA_REPLIES = ["ok1", "ok2", "ok3", "ok4", "ok5"]  # opening + 4 replies


def test_every_turn_carries_controller_state(tmp_path, monkeypatch):
    monkeypatch.setattr("role.logger.SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("role.session.SESSIONS_DIR", tmp_path)
    scenario = load(SCENARIOS_DIR / "order-change.yaml")  # difficulty: easy

    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        with patch("role.persona.Groq") as mock_groq_cls:
            mock_groq_cls.return_value.chat.completions.create.side_effect = [
                _groq_response(text) for text in PERSONA_REPLIES
            ]
            with patch("builtins.input", side_effect=TRAINEE_LINES):
                session_mod.run(n_turns=9, scenario=scenario)

    log_files = list(tmp_path.glob("*.jsonl"))
    assert len(log_files) == 1
    records = [json.loads(ln) for ln in log_files[0].read_text().splitlines() if ln.strip()]

    assert len(records) == 9  # 1 opening + 4 trainee/persona exchanges
    assert all(r["controller_state"] is not None for r in records)  # the "Done when"
    assert all(r["controller_state"]["difficulty"] == "easy" for r in records)

    moods = [r["controller_state"]["mood"] for r in records]
    assert moods[0] == "cooperative"       # opening line, before any trainee input
    assert "frustrated" in moods           # mood actually changed, not static
    assert moods[-1] == "cooperative"      # the de-escalate line brought it back down
    assert "escalated" not in moods        # easy difficulty ceiling holds end-to-end
