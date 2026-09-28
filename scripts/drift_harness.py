#!/usr/bin/env python3
"""drift_harness.py — persona-drift detection over >=20 turns.

Runs a scenario non-interactively using scripted trainee replies, then checks
each persona turn for consistency against the scenario's ground-truth facts.
Generalized (ROLE-019) to run any (scenario, difficulty) pair: difficulty is
passed to the Controller (ROLE-018) as an independent variable from the
scenario's own YAML difficulty, and each scenario carries its own fixture of
facts / contradiction patterns / scripted replies.

Drift flags (per turn):
  FACT_DRIFT   — persona contradicts a known fact (order #, model name, price)
  GOAL_DRIFT   — persona stops pursuing its stated goal and accepts an
                 unrelated resolution or goes off-topic
  DIFF_DRIFT   — persona uses hostile/abusive language, regardless of
                 difficulty or controller mood (escalated means pacing and
                 directness, never insults or threats — see Controller.render())

Each flag is heuristic: keyword-based, so it catches obvious contradictions
without requiring a second LLM call.

Output: prints a per-turn table and writes drift_{session_id}.jsonl to sessions/.

Run standalone : uv run python scripts/drift_harness.py
Run via pytest : uv run pytest scripts/drift_harness.py -s
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Load .env when run via pytest
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from role.controller import Controller  # noqa: E402
from role.logger import SessionLogger, SESSIONS_DIR  # noqa: E402
from role.persona import PersonaAgent  # noqa: E402
from role.scenario import load, SCENARIOS_DIR  # noqa: E402

N_TURNS = 20  # total utterances (persona + trainee interleaved)

# Hostile/abusive language — wrong at every difficulty, so this list is shared
# across all fixtures rather than gated by the Controller's mood ceiling.
DIFF_DRIFT_PATTERNS = [
    r"\byou(?:'?re|\s+are)\s+(?:useless|incompetent|an?\s+idiot)\b",
    r"\bthis\s+is\s+(?:absolutely\s+)?ridiculous\b",
    r"\bi(?:'?m)?\s+(?:going\s+to\s+)?(?:sue|report)\s+you\b",
    r"\bscrew\s+(?:this|you)\b",
]


@dataclass(frozen=True)
class PersonaFacts:
    """Ground-truth facts for one persona — the single source of truth used both

    to detect FACT_DRIFT and to render the {{PINNED_FACTS}} block (ROLE-019's
    fix), so "facts checked" and "facts pinned" can never drift apart.
    """

    facts: dict[str, str]
    contradictions: dict[str, list[str]]

    def render_pinned_block(self) -> str:
        return "\n".join(f"- {key.replace('_', ' ')}: {value}" for key, value in self.facts.items())


@dataclass(frozen=True)
class ScenarioFixture:
    facts: PersonaFacts
    goal_drift_patterns: list[str]
    scripted_replies: list[str]


FIXTURES: dict[str, ScenarioFixture] = {
    "order-change": ScenarioFixture(
        facts=PersonaFacts(
            facts={
                "order_number": "BT-78432",
                "model_original": "SoundCore 2",
                "model_target": "SoundCore 3",
                "price_original": "£49",
                "price_target": "£69",
            },
            contradictions={
                "order_number": [r"\bBT-(?!78432\b)\d+\b"],
                "model_original": [r"\bSoundCore\s+(?:1|4|Pro)\b"],
                "model_target": [r"\bSoundCore\s+(?:1|4|Pro)\b"],
                "price_original": [r"£(?:39|59|79)\b"],
                # excludes 49 (£49 is the valid price_original — a persona correctly
                # stating both prices in one sentence must not false-flag as drift)
                "price_target": [r"£(?:39|59|79|89)\b"],
            },
        ),
        goal_drift_patterns=[
            r"\bforget\s+(?:about\s+)?(?:it|the\s+order)\b",
            r"\bdon'?t\s+(?:bother|worry\s+about\s+it)\b",
            r"\bjust\s+cancel\s+everything\b",
            r"\bnever\s+mind\b",
            r"\bi(?:'?ll)?\s+(?:just\s+)?(?:keep|take)\s+the\s+soundcore\s+2\b",
        ],
        scripted_replies=[
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
        ],
    ),
    "billing-dispute": ScenarioFixture(
        facts=PersonaFacts(
            facts={
                "account_number": "STR-22991",
                "monthly_charge": "£12.99",
                "charge_dates": "the 3rd and the 4th",
                "tenure": "18 months",
            },
            contradictions={
                "account_number": [r"\bSTR-(?!22991\b)\d+\b"],
                "monthly_charge": [r"£(?:9\.99|14\.99|19\.99)\b"],
                "charge_dates": [r"\b(?:1st|2nd|5th|6th|7th)\s+and\s+the\s+(?:2nd|3rd|6th|7th|8th)\b"],
                "tenure": [r"\b(?:6|12|24)\s+months\b"],
            },
        ),
        goal_drift_patterns=[
            r"\bforget\s+(?:about\s+)?(?:it|the\s+refund)\b",
            r"\bdon'?t\s+worry\s+about\s+(?:it|the\s+refund)\b",
            r"\bi(?:'?ll)?\s+just\s+pay\s+it\b",
            r"\bnever\s+mind\b",
            r"\bkeep\s+the\s+charge\b",
        ],
        scripted_replies=[
            "Hi, my account number is STR-22991. I was charged £12.99 twice last month.",
            "The charges were on the 3rd and the 4th — same amount both times.",
            "I've been a customer for 18 months and never had this problem before.",
            "Can you confirm you can see both charges on your end?",
            "How long will the refund take once you process it?",
            "Will it go back to the same card I paid with?",
            "Is there anything else you need from me to process this?",
            "Just to be clear, this is for the duplicate charge only, not my regular subscription.",
            "OK, and I'll get an email confirmation once it's done?",
            "Perfect, thank you for sorting this out.",
        ],
    ),
    "delivery-delay": ScenarioFixture(
        facts=PersonaFacts(
            facts={
                "order_number": "FJ-10293",
                "product": "formal jacket",
                "event": "job interview",
                "event_time": "tomorrow at 10am",
            },
            contradictions={
                "order_number": [r"\bFJ-(?!10293\b)\d+\b"],
                "product": [r"\b(?:shoes|shirt|suit|jeans)\b"],
                "event_time": [r"\b(?:9am|11am|noon)\b"],
            },
        ),
        goal_drift_patterns=[
            r"\bnever\s+mind\b",
            r"\bit'?s\s+fine,?\s+i'?ll\s+just\s+wait\b",
            r"\bdon'?t\s+worry\s+about\s+it\b",
            r"\bforget\s+(?:about\s+)?(?:it|the\s+delivery)\b",
        ],
        scripted_replies=[
            "Hi, my order number is FJ-10293 — a formal jacket that was supposed to arrive three days ago.",
            "The tracking has said 'Out for delivery' for two days without updating.",
            "I need it for a job interview tomorrow at 10am.",
            "Can you check exactly where the package is right now?",
            "Will it arrive today, yes or no?",
            "If not, what are my options — expedited reship, store pickup, or refund?",
            "I only paid for standard delivery since expedited wasn't offered at checkout.",
            "Is there a store near me that could have this in stock instead?",
            "OK, what's the timeline if I go with the option you suggested?",
            "Thank you — please send me the confirmation once it's arranged.",
        ],
    ),
}


def _check_turn(text: str, turn_index: int, fixture: ScenarioFixture) -> list[str]:
    """Return list of drift flags for one persona turn."""
    flags: list[str] = []
    lower = text.lower()

    for fact_key, patterns in fixture.facts.contradictions.items():
        for pattern in patterns:
            if re.search(pattern, lower, re.IGNORECASE):
                flags.append(f"FACT_DRIFT({fact_key})")

    for pattern in fixture.goal_drift_patterns:
        if re.search(pattern, lower):
            p = repr(pattern)[:40]
            flags.append(f"GOAL_DRIFT({p})")

    for pattern in DIFF_DRIFT_PATTERNS:
        if re.search(pattern, lower):
            p = repr(pattern)[:40]
            flags.append(f"DIFF_DRIFT({p})")

    return flags


def run_drift_harness(scenario_id: str = "order-change", difficulty: str = "easy") -> dict:
    if scenario_id not in FIXTURES:
        raise ValueError(f"No drift fixture for scenario {scenario_id!r}")

    scenario = load(SCENARIOS_DIR / f"{scenario_id}.yaml")
    fixture = FIXTURES[scenario_id]
    controller = Controller(difficulty)
    logger = SessionLogger()
    persona = PersonaAgent(scenario.policy_path.read_text())
    persona.set_facts(fixture.facts.render_pinned_block())

    drift_log: list[dict] = []
    persona_turn_index = 0

    # Turn 1: persona opens
    state = controller.initial_state()
    persona.set_state(state.render())
    opening_reply = persona.reply(scenario.opening, logger=logger, controller_state=state.to_dict())
    flags = _check_turn(opening_reply, persona_turn_index, fixture)
    drift_log.append({
        "turn_index": persona_turn_index,
        "role": "persona",
        "text": opening_reply,
        "flags": flags,
    })
    persona_turn_index += 1

    # Remaining turns: trainee + persona interleaved
    for trainee_text in fixture.scripted_replies:
        logger.log(role="trainee", content=trainee_text)
        drift_log.append({
            "turn_index": None,
            "role": "trainee",
            "text": trainee_text,
            "flags": [],
        })

        state = controller.step(trainee_text)
        persona.set_state(state.render())
        persona_reply = persona.reply(trainee_text, logger=logger, controller_state=state.to_dict())
        flags = _check_turn(persona_reply, persona_turn_index, fixture)
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
        "difficulty": difficulty,
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
    """pytest entry-point: run harness, assert >=20 turns logged."""
    result = run_drift_harness()
    _print_report(result)

    assert result["total_turns"] >= N_TURNS, (
        f"Expected >={N_TURNS} turns, got {result['total_turns']}"
    )
    assert result["persona_turns"] >= N_TURNS // 2, (
        f"Expected >={N_TURNS // 2} persona turns, got {result['persona_turns']}"
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
