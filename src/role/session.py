"""Text-only turn loop for one roleplay session.

10 turns = 5 customer utterances + 5 trainee utterances, interleaved.
The persona always opens first.
"""

import json
import time
from pathlib import Path
from typing import Optional

from role.controller import Controller
from role.logger import SessionLogger, SESSIONS_DIR
from role.persona import PersonaAgent
from role.scenario import Scenario, load, SCENARIOS_DIR

TOTAL_TURNS = 10  # persona opens + (trainee + persona) × 4 + trainee closes = 10
DOCS_SCORECARDS_DIR = Path(__file__).parent.parent.parent / "docs" / "scorecards"


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
    trainee_lines: Optional[list[str]] = None,
) -> Optional[str]:
    """Run an interactive session. n_turns counts each utterance (persona + trainee).

    Pass *session_id* to resume an existing session (turn ids continue from
    where the previous run left off).  A new session id is generated otherwise.
    Pass *scenario* to select a scenario; defaults to order-change.
    Pass *trainee_lines* to script the trainee's side instead of reading from
    stdin (ROLE-030: --scripted demo mode / the automated gate) — raises
    IndexError if the session needs more lines than were given, same as a
    scripted conversation running out of script.

    Returns the session id on a completed session, or None if the trainee
    interrupted it (EOF/Ctrl-C) — an interrupted session has no closing turn
    and should not be handed to the Evaluator.
    """
    if scenario is None:
        scenario = load(SCENARIOS_DIR / "order-change.yaml")

    wall_start = time.perf_counter()
    logger = SessionLogger(session_id)
    system_prompt = scenario.policy_path.read_text()
    persona = PersonaAgent(system_prompt)
    controller = Controller(scenario.difficulty)

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
    state = controller.initial_state()
    persona.set_state(state.render())
    opening = persona.reply(scenario.opening, logger=logger, controller_state=state.to_dict())
    print(f"Alex : {opening}")
    print()

    persona_turns = 1
    trainee_turns = 0

    while persona_turns + trainee_turns < n_turns:
        # Trainee speaks
        if trainee_lines is not None:
            line = trainee_lines[trainee_turns].strip()
        else:
            try:
                line = input("You  : ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[session interrupted]")
                return None
        if not line:
            line = "[no response]"
        state = controller.step(line)
        logger.log(role="trainee", content=line, controller_state=state.to_dict())
        trainee_turns += 1

        if persona_turns + trainee_turns >= n_turns:
            break

        # Persona replies
        persona.set_state(state.render())
        reply = persona.reply(line, logger=logger, controller_state=state.to_dict())
        print(f"\nAlex : {reply}\n")
        persona_turns += 1

    wall_secs = time.perf_counter() - wall_start
    _print_cost_meter(logger.session_id, persona_turns + trainee_turns, wall_secs)

    print()
    print("=" * 62)
    print(f"  Session complete — {persona_turns} persona turns, {trainee_turns} trainee turns.")
    print("=" * 62)

    return logger.session_id


def demo(
    n_turns: int = TOTAL_TURNS,
    scenario: Optional[Scenario] = None,
    trainee_lines: Optional[list[str]] = None,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """ROLE-030: scenario in -> session + evidence-linked scorecard + learning
    plan out. Chains run() -> EvaluatorAgent -> CoachAgent -> a static HTML
    scorecard, written under *out_dir* (defaults to
    role.session.DOCS_SCORECARDS_DIR, resolved at call time so tests can
    monkeypatch it — same convention as SESSIONS_DIR elsewhere in this module).

    Returns the scorecard's path, or None if the session was interrupted
    before completion (nothing to evaluate).
    """
    # Imported lazily: EvaluatorAgent/CoachAgent require GROQ_API_KEY at
    # construction time, and run()'s own import already handles that failure
    # mode for the persona — no need to pay that cost for callers who only
    # want the bare interactive session (role.session.run() alone).
    from role.coach import CoachAgent
    from role.evaluator import EvaluatorAgent
    from role.scorecard_html import render_scorecard_html

    if out_dir is None:
        out_dir = DOCS_SCORECARDS_DIR

    session_id = run(n_turns=n_turns, scenario=scenario, trainee_lines=trainee_lines)
    if session_id is None:
        return None

    scorecard = EvaluatorAgent().evaluate(session_id)
    plan = CoachAgent().plan(scorecard)
    html_text = render_scorecard_html(scorecard, plan)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}.html"
    out_path.write_text(html_text, encoding="utf-8")

    print()
    print(f"Scorecard : {scorecard.total}/{scorecard.max}")
    print(f"Written   : {out_path}")

    return out_path
