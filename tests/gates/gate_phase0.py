#!/usr/bin/env python3
"""gate_phase0.py — Phase 0 exit gate.

Pass conditions (all must be true):
1. order-change scenario loads without error.
2. A 4-turn scripted session completes end-to-end (2 persona + 2 trainee turns).
3. Every persona turn in the session log carries non-None values for:
   model, prompt_tokens, completion_tokens, seconds, cost_usd.
4. Prints a PASS table with $/session.

Run standalone : uv run python tests/gates/gate_phase0.py
Run via pytest : uv run pytest tests/gates/gate_phase0.py -s
"""

import json
import os
import sys
from pathlib import Path

# Load .env when run via pytest (Makefile sources it for demo/doctor targets)
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from role.logger import SessionLogger, SESSIONS_DIR  # noqa: E402
from role.persona import PersonaAgent  # noqa: E402
from role.scenario import load, SCENARIOS_DIR  # noqa: E402

SCENARIO_FILE = SCENARIOS_DIR / "order-change.yaml"
SCRIPTED_REPLIES = [
    "Hi there, I placed an order this morning and I need to change the size.",
    "I ordered a medium but I'd like to upgrade to a large — is that possible?",
]
REQUIRED_PERSONA_FIELDS = {"model", "prompt_tokens", "completion_tokens", "seconds", "cost_usd"}
EXPECTED_MODEL = "openai/gpt-oss-120b"


def _run_scripted_session() -> dict:
    """Run 4 turns (2 persona + 2 trainee) non-interactively; return summary."""
    scenario = load(SCENARIO_FILE)
    logger = SessionLogger()
    persona = PersonaAgent(scenario.policy_path.read_text())

    # Persona opens (turn 1)
    persona.reply(scenario.opening, logger=logger)

    # Two scripted trainee + persona exchanges
    for reply_text in SCRIPTED_REPLIES:
        logger.log(role="trainee", content=reply_text)
        persona.reply(reply_text, logger=logger)

    records = [
        json.loads(ln)
        for ln in (SESSIONS_DIR / f"{logger.session_id}.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    persona_records = [r for r in records if r["role"] == "persona"]
    total_cost = sum(
        r["cost_usd"] for r in persona_records if r["cost_usd"] is not None
    )

    missing_fields: list[str] = []
    for r in persona_records:
        for field in REQUIRED_PERSONA_FIELDS:
            if r.get(field) is None:
                missing_fields.append(f"{r['turn_id']}: {field} is None")

    return {
        "scenario": scenario.id,
        "total_turns": len(records),
        "persona_turns": len(persona_records),
        "model": persona_records[0]["model"] if persona_records else None,
        "cost_usd": total_cost,
        "session_id": logger.session_id,
        "log_ok": len(missing_fields) == 0,
        "missing_fields": missing_fields,
    }


def _print_table(r: dict) -> None:
    cost_str = f"${r['cost_usd']:.4f}"
    log_str = "PASS" if r["log_ok"] else "FAIL"
    model_str = r["model"] or "—"
    print()
    print(f"{'scenario':<22} {'turns':>5} {'$/session':>10} {'model':<28} {'log_ok':>6}")
    print("-" * 76)
    print(
        f"{r['scenario']:<22} {r['total_turns']:>5} {cost_str:>10} "
        f"{model_str:<28} {log_str:>6}"
    )
    print()
    for msg in r["missing_fields"]:
        print(f"  MISSING: {msg}")


def test_gate_phase0() -> None:
    """pytest entry-point: run the gate and assert PASS."""
    result = _run_scripted_session()
    _print_table(result)

    assert result["persona_turns"] >= 2, (
        f"Expected ≥2 persona turns, got {result['persona_turns']}"
    )
    assert result["model"] == EXPECTED_MODEL, (
        f"Expected model {EXPECTED_MODEL!r}, got {result['model']!r}"
    )
    assert result["log_ok"], (
        f"Session log missing required fields:\n" + "\n".join(result["missing_fields"])
    )


if __name__ == "__main__":
    result = _run_scripted_session()
    _print_table(result)
    passed = (
        result["log_ok"]
        and result["persona_turns"] >= 2
        and result["model"] == EXPECTED_MODEL
    )
    print(f"GATE: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)
