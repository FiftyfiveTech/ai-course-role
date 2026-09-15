"""Unit tests for ROLE-010.

Cases:
  1. Session resumes from log — turn ids are stable across two SessionLogger instances.
  2. Empty transcript raises — a session file that exists but has no records is corrupt.
  3. Malformed scenario raises — missing required fields produce a ValueError with a
     useful message that names the bad fields.

No API calls, no file system side-effects outside tmp_path.
"""

import json
import pytest

from role.logger import SessionLogger
from role.scenario import load


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_turns(logger: SessionLogger, n: int) -> list[str]:
    """Append n trainee turns; return the list of turn_ids."""
    return [
        logger.log(role="trainee", content=f"turn {i}")
        for i in range(1, n + 1)
    ]


# ── test 1: resume keeps turn ids stable ─────────────────────────────────────

def test_resume_continues_turn_ids(tmp_path, monkeypatch):
    """A resumed session continues from where the previous one left off."""
    monkeypatch.setattr("role.logger.SESSIONS_DIR", tmp_path)

    lg1 = SessionLogger()
    sid = lg1.session_id
    ids_first = _write_turns(lg1, 3)

    # ids must be <sid>:1, <sid>:2, <sid>:3
    assert ids_first == [f"{sid}:1", f"{sid}:2", f"{sid}:3"]

    # resume with same session_id
    lg2 = SessionLogger(sid)
    ids_second = _write_turns(lg2, 2)

    # must continue at 4, 5 — not restart at 1
    assert ids_second == [f"{sid}:4", f"{sid}:5"]

    # verify the file has 5 lines with monotonically increasing turn_index
    lines = (tmp_path / f"{sid}.jsonl").read_text().strip().splitlines()
    assert len(lines) == 5
    indices = [json.loads(ln)["turn_index"] for ln in lines]
    assert indices == [1, 2, 3, 4, 5]


# ── test 2: empty transcript raises ──────────────────────────────────────────

def test_empty_transcript_raises(tmp_path, monkeypatch):
    """Resuming a session whose file exists but is empty raises ValueError."""
    monkeypatch.setattr("role.logger.SESSIONS_DIR", tmp_path)

    # create an empty session file (simulates a corrupt/interrupted write)
    sid = "deadbeef1234"
    (tmp_path / f"{sid}.jsonl").write_text("")

    with pytest.raises(ValueError, match="no turn records"):
        SessionLogger(sid)


# ── test 3: malformed scenario raises ────────────────────────────────────────

def test_malformed_scenario_missing_fields_raises(tmp_path):
    """A scenario YAML missing required top-level fields raises ValueError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: incomplete\ntitle: No other fields\n")

    with pytest.raises(ValueError, match="missing required fields"):
        load(bad)


def test_malformed_scenario_bad_difficulty_raises(tmp_path):
    """A scenario YAML with an invalid difficulty value raises ValueError."""
    bad = tmp_path / "bad_diff.yaml"
    bad.write_text(
        "id: x\n"
        "title: Bad difficulty\n"
        "difficulty: legendary\n"
        "goal: something\n"
        "persona:\n"
        "  name: Bob\n"
        "  policy: persona_v1.md\n"
        "  opening: hello\n"
    )

    with pytest.raises(ValueError, match="difficulty"):
        load(bad)


def test_malformed_scenario_missing_persona_fields_raises(tmp_path):
    """A scenario YAML with an incomplete persona block raises ValueError."""
    bad = tmp_path / "bad_persona.yaml"
    bad.write_text(
        "id: x\n"
        "title: Incomplete persona\n"
        "difficulty: easy\n"
        "goal: something\n"
        "persona:\n"
        "  name: Bob\n"
    )

    with pytest.raises(ValueError, match="missing required fields"):
        load(bad)
