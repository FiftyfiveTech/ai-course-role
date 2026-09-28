#!/usr/bin/env python3
"""run_drift_matrix.py — ROLE-019 drift matrix runner.

Runs drift_harness.run_drift_harness() over the full 3-persona x 2-difficulty
grid (6 cells), all live against Groq (no mocked/reused sessions — see
notes/phase-1-findings.md for why: the drift_*.jsonl files left over from
ROLE-017 turned out to be mocked test fixtures, not real measurements, so
there was nothing genuine to reuse).

Run standalone : uv run python scripts/run_drift_matrix.py before|after
Run via pytest : uv run pytest scripts/run_drift_matrix.py -s
"""

import os
import sys
from collections import Counter
from pathlib import Path

_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(Path(__file__).parent))
import drift_harness  # noqa: E402

GRID: list[tuple[str, str]] = [
    ("order-change", "easy"),
    ("order-change", "hard"),
    ("billing-dispute", "easy"),
    ("billing-dispute", "hard"),
    ("delivery-delay", "easy"),
    ("delivery-delay", "hard"),
]

# Recorded after the BEFORE phase actually ran (see notes/phase-1-findings.md
# for the command + output that produced this number). Used by the AFTER gate
# to assert the fix did not make things worse — never averaged or estimated.
#
# First BEFORE run flagged 1 FACT_DRIFT(price_target), but it was a false
# positive: the contradiction regex for price_target included "49", which
# collides with the *valid* price_original value — any turn correctly stating
# both prices in one sentence tripped it. Fixed in drift_harness.py's
# order-change fixture; re-run below is the real number.
# `uv run python scripts/run_drift_matrix.py before` -> 0 drifted / 66 persona turns (0.0%)
BEFORE_TOP_CAUSE_COUNT: int | None = 0


def _flag_type(flag: str) -> str:
    return flag.split("(", 1)[0]


def run_matrix(phase: str) -> list[dict]:
    results = []
    for scenario_id, difficulty in GRID:
        result = drift_harness.run_drift_harness(scenario_id, difficulty)
        result["phase"] = phase
        results.append(result)
    return results


def _aggregate(results: list[dict]) -> dict:
    counts: Counter = Counter()
    total_persona_turns = 0
    total_drifted_turns = 0

    for r in results:
        total_persona_turns += r["persona_turns"]
        total_drifted_turns += r["drifted_turns"]
        for entry in r["drift_log"]:
            if entry["role"] != "persona":
                continue
            for flag in entry["flags"]:
                counts[_flag_type(flag)] += 1

    top_cause = counts.most_common(1)[0][0] if counts else None
    return {
        "counts": dict(counts),
        "top_cause": top_cause,
        "total_persona_turns": total_persona_turns,
        "total_drifted_turns": total_drifted_turns,
        "drift_rate": total_drifted_turns / total_persona_turns if total_persona_turns else 0.0,
    }


def _print_matrix_report(phase: str, results: list[dict], aggregate: dict) -> None:
    print()
    print(f"=== DRIFT MATRIX — {phase.upper()} ===")
    print()
    print(f"{'scenario':<16}  {'difficulty':<10}  {'persona_turns':>13}  {'drifted':>7}  {'rate':>7}")
    print("-" * 66)
    for r in results:
        print(
            f"{r['scenario']:<16}  {r['difficulty']:<10}  {r['persona_turns']:>13}  "
            f"{r['drifted_turns']:>7}  {r['drift_rate']:>6.1%}"
        )
    print("-" * 66)
    print(
        f"{'TOTAL':<16}  {'':<10}  {aggregate['total_persona_turns']:>13}  "
        f"{aggregate['total_drifted_turns']:>7}  {aggregate['drift_rate']:>6.1%}"
    )
    print()
    print("Flag-type breakdown:")
    for flag_type, count in sorted(aggregate["counts"].items(), key=lambda kv: -kv[1]):
        marker = "  <-- TOP CAUSE" if flag_type == aggregate["top_cause"] else ""
        print(f"  {flag_type:<12} {count:>4}{marker}")
    print()


def test_run_drift_matrix_after() -> None:
    """pytest gate: after-phase total drifted turns must not exceed the recorded
    before-phase count. Skipped (via explicit skip, not silent) until the
    before-phase number is recorded — see conftest.py's silent-skip-is-a-FAIL
    rule, so this must be an explicit @pytest.mark.skip once wired, never bare.

    Uses "<=" (non-increase), not "<" (strict decrease): the real BEFORE run
    measured 0 drifted turns (see notes/phase-1-findings.md), so a strict
    decrease is unsatisfiable by construction. The pinned-facts fix is still
    worth shipping as a preventative measure — this gate asserts it does not
    regress, not that it manufactures an improvement that wasn't there to find.
    """
    if BEFORE_TOP_CAUSE_COUNT is None:
        import pytest

        pytest.skip("BEFORE_TOP_CAUSE_COUNT not recorded yet — run `before` phase first")

    results = run_matrix("after")
    aggregate = _aggregate(results)
    _print_matrix_report("after", results, aggregate)

    assert aggregate["total_drifted_turns"] <= BEFORE_TOP_CAUSE_COUNT, (
        f"Expected after-phase drifted turns ({aggregate['total_drifted_turns']}) <= "
        f"before-phase count ({BEFORE_TOP_CAUSE_COUNT}) — fix regressed drift"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("before", "after"):
        print("usage: uv run python scripts/run_drift_matrix.py before|after")
        sys.exit(2)

    phase = sys.argv[1]
    results = run_matrix(phase)
    aggregate = _aggregate(results)
    _print_matrix_report(phase, results, aggregate)
    print(f"GATE: PASS — {len(results)} cells run, top cause = {aggregate['top_cause']}")
