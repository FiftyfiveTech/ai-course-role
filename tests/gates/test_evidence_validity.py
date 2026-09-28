#!/usr/bin/env python3
"""test_evidence_validity.py — ROLE-026 gate: every Scorecard evidence TurnRef
must resolve to a real turn id in its session's log — a dangling ref is a
DEFECT, not a warning.

Checks evals/scores/dev/*.json (real Scorecard output, ROLE-023) against
sessions/<session_id>.jsonl (ROLE-006). Never touches evals/heldout/ or
evals/dev/ (hand scores predate the TurnRef schema, see role.agreement).

Pass conditions:
1. evals/scores/dev/ contains at least one *.json file. An empty directory
   produces a VACUOUS PASS warning and the test is skipped — a vacuous pass
   is not a real pass.
2. Every evidence turn_id, for every rubric item, in every scorecard, appears
   in that scorecard's own sessions/<session_id>.jsonl.
3. Every scorecard's session_id resolves to an existing session log at all —
   a missing log is itself a DEFECT, not silently skipped.

Run standalone : uv run python tests/gates/test_evidence_validity.py
Run via pytest : uv run pytest tests/gates/test_evidence_validity.py -s
"""

import json
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from role.scorecard import Scorecard  # noqa: E402

SCORES_DEV_DIR = REPO_ROOT / "evals" / "scores" / "dev"
SESSIONS_DIR = REPO_ROOT / "sessions"


def _load_scorecards(scores_dir: Path) -> list[Scorecard]:
    return [
        Scorecard.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(scores_dir.glob("*.json"))
    ]


def _session_turn_ids(sessions_dir: Path, session_id: str) -> "set[str] | None":
    """Return the set of real turn_ids in sessions/<session_id>.jsonl, or None
    if that log does not exist at all."""
    path = sessions_dir / f"{session_id}.jsonl"
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {json.loads(ln)["turn_id"] for ln in lines}


def find_dangling_refs(scorecards: list[Scorecard], sessions_dir: Path) -> list[dict]:
    """Return one dict per defect: a missing session log, or an evidence
    turn_id that does not appear in its session's log."""
    defects: list[dict] = []
    for scorecard in scorecards:
        turn_ids = _session_turn_ids(sessions_dir, scorecard.session_id)
        if turn_ids is None:
            defects.append(
                {
                    "session_id": scorecard.session_id,
                    "item": None,
                    "turn_id": None,
                    "reason": f"sessions/{scorecard.session_id}.jsonl not found",
                }
            )
            continue
        for item_name, item in scorecard.items.items():
            for ref in item.evidence:
                if ref.turn_id not in turn_ids:
                    defects.append(
                        {
                            "session_id": scorecard.session_id,
                            "item": item_name,
                            "turn_id": ref.turn_id,
                            "reason": "turn_id not found in session log",
                        }
                    )
    return defects


def run_check(
    scores_dir: Path = SCORES_DEV_DIR, sessions_dir: Path = SESSIONS_DIR
) -> dict:
    scorecards = _load_scorecards(scores_dir)
    vacuous = len(scorecards) == 0
    defects = [] if vacuous else find_dangling_refs(scorecards, sessions_dir)
    return {
        "scorecard_count": len(scorecards),
        "defect_count": len(defects),
        "defects": defects,
        "vacuous": vacuous,
    }


def _print_report(r: dict) -> None:
    print()
    print(f"{'scorecards checked':>20}: {r['scorecard_count']}")
    print(f"{'dangling refs':>20}: {r['defect_count']}")
    print()
    if r["vacuous"]:
        print(f"VACUOUS PASS — {SCORES_DEV_DIR} is empty")
        print("This is NOT a real pass. Fill evals/scores/dev/ before trusting gate results.")
    elif r["defect_count"] == 0:
        print("PASS — every evidence turn_id resolves to a real turn in its session log")
    else:
        print("FAIL — dangling evidence refs:")
        for d in r["defects"]:
            print(
                f"  session={d['session_id']}  item={d['item']}  "
                f"turn_id={d['turn_id']}  reason={d['reason']}"
            )
    print()


def test_evidence_validity() -> None:
    """pytest entry-point: fail on any dangling ref, skip with warning on vacuous."""
    result = run_check()
    _print_report(result)

    if result["vacuous"]:
        warnings.warn(
            f"VACUOUS PASS: {SCORES_DEV_DIR} is empty — evidence-validity check "
            "cannot run.",
            stacklevel=2,
        )
        import pytest

        pytest.skip(f"VACUOUS PASS — {SCORES_DEV_DIR} is empty")

    assert result["defect_count"] == 0, (
        f"{result['defect_count']} dangling evidence ref(s):\n"
        + "\n".join(
            f"  session={d['session_id']} item={d['item']} turn_id={d['turn_id']} "
            f"({d['reason']})"
            for d in result["defects"]
        )
    )


def test_detects_planted_dangling_ref(tmp_path) -> None:
    """Proof the checker actually catches a defect, not a vacuous always-pass.

    Plants one real turn_id and one dangling turn_id across the rubric items
    of a single synthetic scorecard, and asserts find_dangling_refs reports
    exactly the dangling one.
    """
    from role.scorecard import RUBRIC_ITEMS

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "sess1.jsonl").write_text(
        json.dumps({"turn_id": "sess1:1", "session_id": "sess1", "turn_index": 1}) + "\n",
        encoding="utf-8",
    )

    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    items = {
        name: {"score": 1, "evidence": [{"turn_id": "sess1:1"}]} for name in RUBRIC_ITEMS
    }
    items["clarity"] = {"score": 1, "evidence": [{"turn_id": "sess1:99"}]}  # planted dangling ref
    body = {"session_id": "sess1", "rubric_version": "evals/RUBRIC.md", "items": items}
    (scores_dir / "sess1.json").write_text(json.dumps(body), encoding="utf-8")

    result = run_check(scores_dir=scores_dir, sessions_dir=sessions_dir)

    assert result["defect_count"] == 1
    assert result["defects"][0] == {
        "session_id": "sess1",
        "item": "clarity",
        "turn_id": "sess1:99",
        "reason": "turn_id not found in session log",
    }


if __name__ == "__main__":
    result = run_check()
    _print_report(result)

    if result["vacuous"]:
        print("GATE: VACUOUS PASS (not a real pass)")
        sys.exit(2)

    print(f"GATE: {'PASS' if result['defect_count'] == 0 else 'FAIL'}")
    sys.exit(0 if result["defect_count"] == 0 else 1)
