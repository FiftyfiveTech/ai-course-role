"""Unit tests for ROLE-023's seed adapter: raw eval seed -> sessions/<id>.jsonl.

Pure data transformation — no API calls, no live model, all fast/free/deterministic.
"""
import json

import pytest

from role.seed_adapter import (
    SEEDS_DEV_DIR,
    _extract_bitext_turns,
    _extract_hh_turns,
    _extract_soda_turns,
    _extract_turns,
    seed_to_session,
)


def _seed_raw(seed_id: str) -> dict:
    return json.loads((SEEDS_DEV_DIR / f"{seed_id}.json").read_text(encoding="utf-8"))["raw"]


# Real content already committed under evals/seeds/dev — exercised directly so
# these tests fail if the real seed files ever change shape.
BITEXT_RAW = _seed_raw("bitext-1000")
HH_RAW = _seed_raw("hh-80002")
SODA_RAW = _seed_raw("soda-0100")


# ── bitext ────────────────────────────────────────────────────────────────────

def test_extract_bitext_turns_agent_scored():
    turns = _extract_bitext_turns(BITEXT_RAW, "Agent (response field)")
    assert turns == [
        ("persona", BITEXT_RAW["instruction"]),
        ("trainee", BITEXT_RAW["response"]),
    ]


def test_extract_bitext_turns_unrecognized_speaker_raises():
    with pytest.raises(ValueError, match="scored_speaker"):
        _extract_bitext_turns(BITEXT_RAW, "Nobody")


# ── hh-rlhf ───────────────────────────────────────────────────────────────────

def test_extract_hh_turns_assistant_scored():
    turns = _extract_hh_turns(HH_RAW, "Assistant (chosen branch)")

    assert len(turns) == 6
    assert [role for role, _ in turns] == [
        "persona", "trainee", "persona", "trainee", "persona", "trainee",
    ]
    assert turns[0][1] == "How can I reset the SMC on my Macbook Air?"
    assert turns[-1][1] == "I don’t understand what you’re asking."


def test_extract_hh_turns_no_markers_raises():
    with pytest.raises(ValueError, match="no Human"):
        _extract_hh_turns({"chosen": "just plain text, no turn markers"}, "Assistant (chosen branch)")


# ── soda ──────────────────────────────────────────────────────────────────────

def test_extract_soda_turns_second_speaker_scored():
    turns = _extract_soda_turns(SODA_RAW, "Friend")

    assert len(turns) == len(SODA_RAW["dialogue"])
    assert [role for role, _ in turns] == [
        "persona", "trainee", "persona", "trainee", "persona", "trainee", "persona",
    ]
    assert turns[0][1] == SODA_RAW["dialogue"][0]


# ── dispatch ──────────────────────────────────────────────────────────────────

def test_extract_turns_dispatches_by_raw_shape():
    assert _extract_turns(SODA_RAW, "Friend") == _extract_soda_turns(SODA_RAW, "Friend")
    assert _extract_turns(HH_RAW, "Assistant (chosen branch)") == _extract_hh_turns(
        HH_RAW, "Assistant (chosen branch)"
    )
    assert _extract_turns(BITEXT_RAW, "Agent (response field)") == _extract_bitext_turns(
        BITEXT_RAW, "Agent (response field)"
    )


def test_extract_turns_unrecognized_shape_raises():
    with pytest.raises(ValueError, match="unrecognized raw seed shape"):
        _extract_turns({"foo": "bar"}, "whoever")


# ── seed_to_session: idempotency, real files, tmp sessions dir ──────────────

def test_seed_to_session_writes_transcript_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr("role.logger.SESSIONS_DIR", tmp_path)

    session_id = seed_to_session("bitext-1000")
    assert session_id == "bitext-1000"

    session_path = tmp_path / "bitext-1000.jsonl"
    lines = session_path.read_text().strip().splitlines()
    assert len(lines) == 2

    records = [json.loads(ln) for ln in lines]
    assert records[0]["role"] == "persona"
    assert records[1]["role"] == "trainee"
    assert records[0]["turn_id"] == "bitext-1000:1"
    assert records[1]["turn_id"] == "bitext-1000:2"

    # second call must not append a duplicate copy of the same turns
    seed_to_session("bitext-1000")
    assert len(session_path.read_text().strip().splitlines()) == 2
