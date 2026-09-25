"""Unit tests for ROLE-017 persona-drift harness — no live inference.

Mocks the Groq client (same pattern as test_arm_policy.py) so the harness's
turn-counting and per-turn drift-flag logic can be verified without calling
Groq or Ollama.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import drift_harness  # noqa: E402


def _groq_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = text
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


# 1 opening + 10 replies (one per drift_harness.SCRIPTED_REPLIES trainee line).
# Three are scripted to hit a known drift pattern; the rest stay clean.
PERSONA_REPLIES = [
    "Hi, thanks for picking up. I'm calling about order BT-78432 — "
    "I want to switch my SoundCore 2 for a SoundCore 3.",
    "Great, yes — order BT-78432, upgrading from the SoundCore 2 to the SoundCore 3.",
    "That's right, still processing on your end too.",
    "Wait, I thought the SoundCore 3 was £89?",           # FACT_DRIFT(price_target)
    "Okay, thanks for clarifying.",
    "Sounds good.",
    "Actually, never mind, forget about it.",              # GOAL_DRIFT
    "Alright, let's pick this back up.",
    "This is absolutely ridiculous, you're useless!",      # DIFF_DRIFT
    "Sorry about that, let's continue.",
    "Yes, confirmed, thanks.",
]


def test_drift_harness_flags_without_live_inference(capsys):
    assert len(PERSONA_REPLIES) == len(drift_harness.SCRIPTED_REPLIES) + 1

    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        with patch("role.persona.Groq") as mock_groq_cls:
            mock_groq_cls.return_value.chat.completions.create.side_effect = [
                _groq_response(text) for text in PERSONA_REPLIES
            ]
            result = drift_harness.run_drift_harness()
            call_count = mock_groq_cls.return_value.chat.completions.create.call_count

    drift_harness._print_report(result)

    # Every persona turn came from the mock — zero live inference calls.
    assert call_count == len(PERSONA_REPLIES)

    assert result["total_turns"] >= drift_harness.N_TURNS
    assert result["persona_turns"] == len(PERSONA_REPLIES)

    persona_entries = [e for e in result["drift_log"] if e["role"] == "persona"]
    assert len(persona_entries) == len(PERSONA_REPLIES)
    assert all("flags" in e for e in persona_entries)  # per-turn drift flag present

    flags_by_turn = {e["turn_index"]: e["flags"] for e in persona_entries}
    assert flags_by_turn[0] == []  # opening line is clean
    assert any("FACT_DRIFT" in f for f in flags_by_turn[3])
    assert any("GOAL_DRIFT" in f for f in flags_by_turn[6])
    assert any("DIFF_DRIFT" in f for f in flags_by_turn[8])
    assert result["drifted_turns"] == 3

    log_path = Path(result["log_path"])
    assert log_path.exists()
    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == result["total_turns"]
