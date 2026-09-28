"""ROLE-023: raw eval seed -> sessions/<seed_id>.jsonl.

Pure reformatting, zero model calls, zero cost. Converts a seed's raw dialogue
(evals/seeds/<split>/<seed_id>.json) into a session transcript via the existing
SessionLogger, so turn_ids come out in the exact "<session_id>:<turn_index>"
format TurnRef expects — the evaluator can then cite real, checkable evidence
against seeds that never went through a live roleplay session.

Reads a seed's paired evals/<split>/<seed_id>.score.json ONLY for its
`scored_speaker` field, which says which side of the raw dialogue is "the
trainee" (rubric-scored) vs "the persona/customer" side. Never reads or writes
that file's `items`/`total`/`max` — those are hand-scored gold labels owned by
a different ticket (ROLE-024's agreement scorer) and must never be touched here.

seeds_dir/scores_dir default to the dev split but are parameters, not
constants baked into the logic, so ROLE-024 can reuse this unchanged for a
heldout pass under supervisor control.
"""

import json
import re
from pathlib import Path

from role import logger as _logger_mod
from role.logger import SessionLogger

EVALS_DIR = Path(__file__).parent.parent.parent / "evals"
SEEDS_DEV_DIR = EVALS_DIR / "seeds" / "dev"
SCORES_DEV_DIR = EVALS_DIR / "dev"

_HH_TURN_RE = re.compile(r"\n\n(Human|Assistant): ")


def _extract_bitext_turns(raw: dict, scored_speaker: str) -> list[tuple[str, str]]:
    if scored_speaker.startswith("Agent"):
        return [("persona", raw["instruction"]), ("trainee", raw["response"])]
    if scored_speaker.startswith("Customer"):
        return [("trainee", raw["instruction"]), ("persona", raw["response"])]
    raise ValueError(f"bitext: unrecognized scored_speaker {scored_speaker!r}")


def _extract_hh_turns(raw: dict, scored_speaker: str) -> list[tuple[str, str]]:
    # re.split with one capturing group always alternates [text, speaker, text, speaker, ...],
    # so after dropping the leading (pre-first-marker) text, what's left is guaranteed
    # even-length by construction — the only real failure mode is zero markers at all.
    parts = _HH_TURN_RE.split(raw["chosen"])[1:]
    if not parts:
        raise ValueError("hh-rlhf: no Human:/Assistant: turns found in raw['chosen']")

    speakers = parts[0::2]
    texts = [t.strip() for t in parts[1::2]]

    trainee_speaker = "Assistant" if scored_speaker.startswith("Assistant") else "Human"
    other_speaker = "Human" if trainee_speaker == "Assistant" else "Assistant"
    role_for = {trainee_speaker: "trainee", other_speaker: "persona"}

    return [(role_for[speaker], text) for speaker, text in zip(speakers, texts)]


def _extract_soda_turns(raw: dict, scored_speaker: str) -> list[tuple[str, str]]:
    return [
        ("trainee" if speaker == scored_speaker else "persona", line)
        for line, speaker in zip(raw["dialogue"], raw["speakers"])
    ]


def _extract_turns(raw: dict, scored_speaker: str) -> list[tuple[str, str]]:
    if "dialogue" in raw and "speakers" in raw:
        return _extract_soda_turns(raw, scored_speaker)
    if "chosen" in raw:
        return _extract_hh_turns(raw, scored_speaker)
    if "instruction" in raw and "response" in raw:
        return _extract_bitext_turns(raw, scored_speaker)
    raise ValueError(f"unrecognized raw seed shape, keys={sorted(raw)}")


def seed_to_session(
    seed_id: str,
    seeds_dir: Path = SEEDS_DEV_DIR,
    scores_dir: Path = SCORES_DEV_DIR,
) -> str:
    """Convert one seed into sessions/<seed_id>.jsonl; return the session id.

    Idempotent: if the session file already exists, return immediately without
    appending a second copy of the same turns.
    """
    session_path = _logger_mod.SESSIONS_DIR / f"{seed_id}.jsonl"
    if session_path.exists():
        return seed_id

    seed = json.loads((seeds_dir / f"{seed_id}.json").read_text(encoding="utf-8"))
    score_meta = json.loads((scores_dir / f"{seed_id}.score.json").read_text(encoding="utf-8"))
    scored_speaker = score_meta["scored_speaker"]

    turns = _extract_turns(seed["raw"], scored_speaker)

    log = SessionLogger(session_id=seed_id)
    for role, content in turns:
        log.log(role=role, content=content)
    return log.session_id
