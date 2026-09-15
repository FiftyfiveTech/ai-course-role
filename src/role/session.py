"""Text-only turn loop for one roleplay session.

10 turns = 5 customer utterances + 5 trainee utterances, interleaved.
The persona always opens first.
"""

import json
import time
from pathlib import Path
from typing import Optional

from role.logger import SessionLogger, SESSIONS_DIR
from role.persona import PersonaAgent
from role.scenario import Scenario, load, SCENARIOS_DIR

TOTAL_TURNS = 10  # persona opens + (trainee + persona) × 4 + trainee closes = 10


def _print_cost_meter(session_id: str, n_turns: int, wall_secs: float) -> None:
    """Print one summary line: cost, turns, model time, wall time."""
    path = SESSIONS_DIR / f"{session_id}.jsonl"
    records = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    total_cost = sum(r["cost_usd"] for r in records if r["cost_usd"] is not None)
    model_secs = sum(r["seconds"] for r in records if r["seconds"] is not None)
    print(
        f"cost ${total_cost:.4f} | turns {n_turns} | "
        f"model {model_secs:.1f}s | wall {wall_secs:.1f}s"
    )


def run(
    n_turns: int = TOTAL_TURNS,
    session_id: Optional[str] = None,
    scenario: Optional[Scenario] = None,
) -> None:
    """Run an interactive session. n_turns counts each utterance (persona + trainee).

    Pass *session_id* to resume an existing session (turn ids continue from
    where the previous run left off).  A new session id is generated otherwise.
    Pass *scenario* to select a scenario; defaults to order-change.
    """
    if scenario is None:
        scenario = load(SCENARIOS_DIR / "order-change.yaml")

    wall_start = time.perf_counter()
    logger = SessionLogger(session_id)
    system_prompt = scenario.policy_path.read_text()
    persona = PersonaAgent(system_prompt)

    print()
    print("=" * 62)
    print("  ROLE — Roleplay & Skills Coach")
    print(f"  Scenario : {scenario.title}  [{scenario.difficulty}]")
    print(f"  Goal     : {scenario.goal}")
    print(f"  Persona  : {scenario.persona_name} — openai/gpt-oss-120b via Groq")
    print(f"  Session  : {n_turns} turns  |  id: {logger.session_id}")
    print("=" * 62)
    print()

    # Persona opens (turn 1)
    opening = persona.reply(scenario.opening, logger=logger)
    print(f"Alex : {opening}")
    print()

    persona_turns = 1
    trainee_turns = 0

    while persona_turns + trainee_turns < n_turns:
        # Trainee speaks
        try:
            line = input("You  : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[session interrupted]")
            return
        if not line:
            line = "[no response]"
        logger.log(role="trainee", content=line)
        trainee_turns += 1

        if persona_turns + trainee_turns >= n_turns:
            break

        # Persona replies
        reply = persona.reply(line, logger=logger)
        print(f"\nAlex : {reply}\n")
        persona_turns += 1

    wall_secs = time.perf_counter() - wall_start
    _print_cost_meter(logger.session_id, persona_turns + trainee_turns, wall_secs)

    print()
    print("=" * 62)
    print(f"  Session complete — {persona_turns} persona turns, {trainee_turns} trainee turns.")
    print("=" * 62)
