#!/usr/bin/env python3
"""scripts/run_evaluator_dev_gate.py — ROLE-023 gate.

Runs the evaluator over all 8 evals/dev seeds (seed_adapter -> EvaluatorAgent),
validates each Scorecard, writes evals/scores/dev/<seed_id>.json, and reports
N/8 schema-valid. Never touches evals/heldout/ or evals/seeds/heldout/ — only
evals/dev/*.score.json seed ids are ever read.

Run standalone : uv run python scripts/run_evaluator_dev_gate.py
Run via pytest : uv run pytest scripts/run_evaluator_dev_gate.py -s
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
    # Windows consoles default to cp1252; a model's raw (possibly garbled) output
    # embedded in an error message can contain characters that encoding can't
    # represent — replace rather than crash the report itself.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from instructor.core import InstructorRetryException  # noqa: E402

from role.evaluator import EvaluatorAgent  # noqa: E402
from role.logger import SessionLogger  # noqa: E402
from role.seed_adapter import SCORES_DEV_DIR, seed_to_session  # noqa: E402

SCORES_OUT_DIR = Path(__file__).parent.parent / "evals" / "scores" / "dev"
DEV_SEED_IDS = sorted(p.stem.removesuffix(".score") for p in SCORES_DEV_DIR.glob("*.score.json"))


def run_gate() -> dict:
    SCORES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluator = EvaluatorAgent()
    results = []

    for seed_id in DEV_SEED_IDS:
        try:
            session_id = seed_to_session(seed_id)
            session_logger = SessionLogger(session_id=session_id)
            scorecard = evaluator.evaluate(session_id, logger=session_logger)
            out_path = SCORES_OUT_DIR / f"{seed_id}.json"
            out_path.write_text(scorecard.model_dump_json(indent=2), encoding="utf-8")
            results.append({"seed_id": seed_id, "ok": True, "error": None})
        except InstructorRetryException as exc:
            results.append({"seed_id": seed_id, "ok": False, "error": f"retries exhausted: {exc}"})
        except Exception as exc:  # any other schema/IO failure counts against the gate too
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


def test_evaluator_dev_gate() -> None:
    result = run_gate()
    _print_report(result)
    assert result["n_ok"] == result["n_total"] == 8, (
        f"Expected 8/8 dev scorecards schema-valid, got {result['n_ok']}/{result['n_total']}"
    )


if __name__ == "__main__":
    result = run_gate()
    _print_report(result)
    passed = result["n_ok"] == result["n_total"] == 8
    print(f"GATE: {'PASS' if passed else 'FAIL'} — {result['n_ok']}/8 schema-valid")
    sys.exit(0 if passed else 1)
