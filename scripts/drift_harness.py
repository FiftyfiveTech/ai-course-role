#!/usr/bin/env python3
"""drift_harness.py — persona-drift detection over ≥20 turns.

Runs a scenario non-interactively using scripted trainee replies, then checks
each persona turn for consistency against the scenario's ground-truth facts.

Drift flags (per turn):
  FACT_DRIFT   — persona contradicts a known fact (order #, model name, price)
  GOAL_DRIFT   — persona stops pursuing its stated goal (order change) and
                 accepts an unrelated resolution or goes off-topic
  DIFF_DRIFT   — persona's emotional tone / difficulty deviates from the
                 scenario's difficulty level (easy → should stay cooperative
                 once the rep engages; should not escalate to hostility)

Each flag is heuristic: keyword-based, so it catches obvious contradictions
without requiring a second LLM call.

Output: prints a per-turn table and writes drift_log.jsonl to sessions/.

Run standalone : uv run python scripts/drift_harness.py
Run via pytest : uv run pytest scripts/drift_harness.py -s
"""

import json
import os
import re
import sys
from pathlib import Path

# Load .env when run via pytest
_env_file = Path(__file__).parent.parent / ".env"
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
N_TURNS = 20  # total utterances (persona + trainee interleaved)

# Ground-truth facts from persona_v1.md — these must never be contradicted
FACTS: dict[str, list[str]] = {
    "order_number": ["BT-78432", "BT78432"],
    "model_original": ["SoundCore 2", "soundcore 2"],
    "model_target": ["SoundCore 3", "soundcore 3"],
    "price_original": ["£49", "49"],
    "price_target": ["£69", "69"],
}

# Contradiction patterns: if persona says one of these, it's a fact drift
FACT_CONTRADICTIONS: dict[str, list[str]] = {
    "order_number": [r"\bBT-(?!78432\b)\d+\b"],   # any order # that isn't BT-78432
    "model_original": [r"\bSoundCore\s+(?:1|4|Pro)\b"],
    "model_target": [r"\bSoundCore\s+(?:1|4|Pro)\b"],
    "price_original": [r"£(?:39|59|79)\b"],
    "price_target": [r"£(?:39|49|79|89)\b"],
}

# Goal drift: persona accepts resolution that doesn't address the order change
GOAL_DRIFT_PATTERNS = [
    r"\bforget\s+(?:about\s+)?(?:it|the\s+order)\b",
    r"\bdon'?t\s+(?:bother|worry\s+about\s+it)\b",
    r"\bjust\s+cancel\s+everything\b",
    r"\bnever\s+mind\b",
    r"\bi(?:'?ll)?\s+(?:just\s+)?(?:keep|take)\s+the\s+soundcore\s+2\b",
]

# Difficulty drift for easy scenario: persona should not become hostile
# Hard negative signals (should stay polite after rep engages)
DIFF_DRIFT_PATTERNS = [
    r"\byou(?:'?re|\s+are)\s+(?:useless|incompetent|an?\s+idiot)\b",
    r"\bthis\s+is\s+(?:absolutely\s+)?ridiculous\b",
    r"\bi(?:'?m)?\s+(?:going\s+to\s+)?(?:sue|report)\s+you\b",
    r"\bscrew\s+(?:this|you)\b",
]


def _check_turn(text: str, turn_index: int) -> list[str]:
    """Return list of drift flags for one persona turn."""
    flags: list[str] = []
    lower = text.lower()

    # Fact drift: contradiction present
    for fact_key, patterns in FACT_CONTRADICTIONS.items():
        for pattern in patterns:
            if re.search(pattern, lower, re.IGNORECASE):
                flags.append(f"FACT_DRIFT({fact_key})")

    # Goal drift
    for pattern in GOAL_DRIFT_PATTERNS:
        if re.search(pattern, lower):
            p = repr(pattern)[:40]
            flags.append(f"GOAL_DRIFT({p})")

    # Difficulty drift
    for pattern in DIFF_DRIFT_PATTERNS:
        if re.search(pattern, lower):
            p = repr(pattern)[:40]
            flags.append(f"DIFF_DRIFT({p})")

    return flags


# Scripted trainee replies — enough to fill 20 turns (10 exchanges)
SCRIPTED_REPLIES = [
    "Hi, I need to change my order from a SoundCore 2 to a SoundCore 3.",
    "The order number is BT-78432. It's still showing as processing.",
    "Yes, I'd like the SoundCore 3 which is £69. Can you make that change?",
    "How long will that take to process?",
    "Will I receive a confirmation email once it's updated?",
    "And the price difference — will that be charged to the same card?",
    "Great. Is there anything else you need from me to complete the change?",
    "OK. And if the order has already shipped, what are my options?",
    "Can I return it for a refund and reorder the SoundCore 3?",
    "Perfect, thank you. Just to confirm — the change to SoundCore 3 is now done?",
]


def run_drift_harness() -> dict:
    scenario = load(SCENARIO_FILE)
    logger = SessionLogger()
    persona = PersonaAgent(scenario.policy_path.read_text())

    drift_log: list[dict] = []
    persona_turn_index = 0

    # Turn 1: persona opens
    opening_reply = persona.reply(scenario.opening, logger=logger)
    flags = _check_turn(opening_reply, persona_turn_index)
    drift_log.append({
        "turn_index": persona_turn_index,
        "role": "persona",
        "text": opening_reply,
        "flags": flags,
    })
    persona_turn_index += 1

    # Remaining turns: trainee + persona interleaved
    for trainee_text in SCRIPTED_REPLIES:
        logger.log(role="trainee", content=trainee_text)
        drift_log.append({
            "turn_index": None,
            "role": "trainee",
            "text": trainee_text,
            "flags": [],
        })

        persona_reply = persona.reply(trainee_text, logger=logger)
        flags = _check_turn(persona_reply, persona_turn_index)
        drift_log.append({
            "turn_index": persona_turn_index,
            "role": "persona",
            "text": persona_reply,
            "flags": flags,
        })
        persona_turn_index += 1

    total_turns = len(drift_log)
    persona_turns = [e for e in drift_log if e["role"] == "persona"]
    drifted_turns = [e for e in persona_turns if e["flags"]]

    # Write drift log
    log_path = SESSIONS_DIR / f"drift_{logger.session_id}.jsonl"
    with log_path.open("w") as fh:
        for entry in drift_log:
            fh.write(json.dumps(entry) + "\n")

    return {
        "session_id": logger.session_id,
        "scenario": scenario.id,
        "difficulty": scenario.difficulty,
        "total_turns": total_turns,
        "persona_turns": len(persona_turns),
        "drifted_turns": len(drifted_turns),
        "drift_rate": len(drifted_turns) / len(persona_turns) if persona_turns else 0.0,
        "drift_log": drift_log,
        "log_path": str(log_path),
    }


def _print_report(r: dict) -> None:
    print()
    print(f"scenario  : {r['scenario']}  [{r['difficulty']}]")
    print(f"session   : {r['session_id']}")
    print(f"turns     : {r['total_turns']} total  |  {r['persona_turns']} persona")
    print(f"drifted   : {r['drifted_turns']} / {r['persona_turns']} persona turns")
    print(f"drift_rate: {r['drift_rate']:.2%}")
    print()
    print(f"{'#':>3}  {'role':<8}  {'flags':<50}  text[:60]")
    print("-" * 110)
    idx = 0
    for entry in r["drift_log"]:
        if entry["role"] != "persona":
            continue
        flag_str = ", ".join(entry["flags"]) or "—"
        text_preview = entry["text"][:60].replace("\n", " ")
        marker = "<<DRIFT" if entry["flags"] else ""
        print(f"{idx:>3}  {'persona':<8}  {flag_str:<50}  {text_preview}  {marker}")
        idx += 1
    print()
    print(f"drift log : {r['log_path']}")
    print()


def test_drift_harness() -> None:
    """pytest entry-point: run harness, assert ≥20 turns logged."""
    result = run_drift_harness()
    _print_report(result)

    assert result["total_turns"] >= N_TURNS, (
        f"Expected ≥{N_TURNS} turns, got {result['total_turns']}"
    )
    assert result["persona_turns"] >= N_TURNS // 2, (
        f"Expected ≥{N_TURNS // 2} persona turns, got {result['persona_turns']}"
    )
    # Report drift rate — harness does not assert a threshold (Phase 1 measure only)
    print(
        f"DRIFT RATE: {result['drift_rate']:.2%}  "
        f"({result['drifted_turns']}/{result['persona_turns']} persona turns flagged)"
    )


if __name__ == "__main__":
    result = run_drift_harness()
    _print_report(result)
    passed = result["total_turns"] >= N_TURNS
    print(f"GATE: {'PASS' if passed else 'FAIL'} — {result['total_turns']} turns logged")
    sys.exit(0 if passed else 1)
