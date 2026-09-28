#!/usr/bin/env python3
"""scripts/run_coach_dev_gate.py — ROLE-027 gate.

Runs the coach over all 8 real evals/scores/dev/*.json Scorecards (machine
output, ROLE-023), validates each LearningPlan, writes
evals/plans/dev/<seed_id>.md, and reports N/8 schema-valid. Never touches
evals/heldout/ or the raw session transcripts — the coach reads only the
Scorecard.

Run standalone : uv run python scripts/run_coach_dev_gate.py
Run via pytest : uv run pytest scripts/run_coach_dev_gate.py -s
"""

import os
import sys
from pathlib import Path

_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from role.coach import CoachAgent, render_markdown  # noqa: E402
from role.scorecard import Scorecard  # noqa: E402

SCORES_DEV_DIR = Path(__file__).parent.parent / "evals" / "scores" / "dev"
PLANS_OUT_DIR = Path(__file__).parent.parent / "evals" / "plans" / "dev"
SCORECARD_SEED_IDS = sorted(p.stem for p in SCORES_DEV_DIR.glob("*.json"))


def run_gate() -> dict:
    PLANS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    coach = CoachAgent()
    results = []

    for seed_id in SCORECARD_SEED_IDS:
        try:
            scorecard = Scorecard.model_validate_json(
                (SCORES_DEV_DIR / f"{seed_id}.json").read_text(encoding="utf-8")
            )
            plan = coach.plan(scorecard)
            out_path = PLANS_OUT_DIR / f"{seed_id}.md"
            out_path.write_text(render_markdown(plan), encoding="utf-8")
            results.append({"seed_id": seed_id, "ok": True, "error": None})
        except Exception as exc:  # schema/IO failure counts against the gate too
            results.append({"seed_id": seed_id, "ok": False, "error": str(exc)})

    return {
        "results": results,
        "n_ok": sum(r["ok"] for r in results),
        "n_total": len(results),
    }


def _print_report(r: dict) -> None:
    print()
    print(f"{'seed_id':<16}  {'status':<10}  detail")
    print("-" * 70)
    for entry in r["results"]:
        status = "PASS" if entry["ok"] else "FAIL"
        detail = (entry["error"] or "")[:200]
        print(f"{entry['seed_id']:<16}  {status:<10}  {detail}")
    print("-" * 70)
    print(f"{r['n_ok']}/{r['n_total']} schema-valid")
    print()


def test_coach_dev_gate() -> None:
    result = run_gate()
    _print_report(result)
    assert result["n_ok"] == result["n_total"] == 8, (
        f"Expected 8/8 dev learning plans schema-valid, got {result['n_ok']}/{result['n_total']}"
    )


if __name__ == "__main__":
    result = run_gate()
    _print_report(result)
    passed = result["n_ok"] == result["n_total"] == 8
    print(f"GATE: {'PASS' if passed else 'FAIL'} — {result['n_ok']}/8 schema-valid")
    sys.exit(0 if passed else 1)
