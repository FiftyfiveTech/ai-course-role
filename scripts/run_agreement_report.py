#!/usr/bin/env python3
"""scripts/run_agreement_report.py — ROLE-024 report.

Computes exact-agreement, +/-1-band agreement, and Cohen's kappa per rubric
item between evals/dev/*.score.json (hand/gold, ROLE-015) and
evals/scores/dev/*.json (machine, from ROLE-023's run_evaluator_dev_gate.py).

This is a read-only report, not a pass/fail gate: ROLE-024's "Done when" asks
for the numbers to be computed and unit-tested against a hand fixture, not
for a numeric bar to clear. See role.agreement's docstring for the same
framing. Never touches evals/heldout/ (sealed, Evaluator-only).

Run standalone : uv run python scripts/run_agreement_report.py
Run via pytest : uv run pytest scripts/run_agreement_report.py -s
"""

import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from role.agreement import (  # noqa: E402
    EVALS_DEV_DIR,
    SCORES_DEV_DIR,
    load_hand_scores,
    load_machine_scores,
    pair_and_score,
)
from role.scorecard import RUBRIC_ITEMS  # noqa: E402


def run_report() -> dict:
    hand = load_hand_scores(EVALS_DEV_DIR)
    machine = load_machine_scores(SCORES_DEV_DIR)
    if not hand or not machine:
        return {"vacuous": True, "hand_n": len(hand), "machine_n": len(machine), "items": {}}
    items = pair_and_score(hand, machine)
    return {"vacuous": False, "hand_n": len(hand), "machine_n": len(machine), "items": items}


def _print_report(r: dict) -> None:
    print()
    if r["vacuous"]:
        print(f"VACUOUS — hand={r['hand_n']} machine={r['machine_n']} (need both non-empty)")
        print("This is NOT a real report. Fill the missing split before trusting these numbers.")
        print()
        return
    print(f"{'item':<20}  {'n':>3}  {'exact%':>8}  {'+/-1-band%':>11}  {'kappa':>8}")
    print("-" * 62)
    for item in RUBRIC_ITEMS:
        a = r["items"][item]
        kappa_str = f"{a.kappa:.3f}" if a.kappa is not None else "undef"
        print(
            f"{item:<20}  {a.n:>3}  {a.exact_agreement * 100:>7.1f}%  "
            f"{a.band_agreement * 100:>10.1f}%  {kappa_str:>8}"
        )
    print("-" * 62)
    print()


def test_agreement_report() -> None:
    """pytest entry-point: report the numbers; skip with a warning if either
    split is empty (a vacuous report is not a real report)."""
    result = run_report()
    _print_report(result)
    if result["vacuous"]:
        warnings.warn(
            "VACUOUS: evals/dev or evals/scores/dev is empty — agreement report "
            "cannot run.",
            stacklevel=2,
        )
        import pytest

        pytest.skip("VACUOUS — no scores to compare")
    assert set(result["items"]) == set(RUBRIC_ITEMS)


if __name__ == "__main__":
    result = run_report()
    _print_report(result)
    if result["vacuous"]:
        print("REPORT: VACUOUS (not a real report)")
        sys.exit(2)
    print("REPORT: OK")
    sys.exit(0)
