"""Text-only turn loop for one roleplay session.

10 turns = 5 customer utterances + 5 trainee utterances, interleaved.
The persona always opens first.
"""

from pathlib import Path

from role.persona import PersonaAgent

REPO_ROOT = Path(__file__).parent.parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"
TOTAL_TURNS = 10  # persona opens + (trainee + persona) × 4 + trainee closes = 10


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text()


def run(n_turns: int = TOTAL_TURNS) -> None:
    """Run an interactive session. n_turns counts each utterance (persona + trainee)."""
    system_prompt = _load_prompt("persona_v1.md")
    persona = PersonaAgent(system_prompt)

    print()
    print("=" * 62)
    print("  ROLE — Roleplay & Skills Coach")
    print("  Scenario : Order-change request")
    print("  Persona  : Alex, a customer — openai/gpt-oss-120b via Groq")
    print(f"  Session  : {n_turns} turns")
    print("=" * 62)
    print()

    # Persona opens (turn 1)
    opening = persona.reply(
        "The call just connected. Greet the customer service rep and state your problem."
    )
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
        trainee_turns += 1

        if persona_turns + trainee_turns >= n_turns:
            break

        # Persona replies
        reply = persona.reply(line)
        print(f"\nAlex : {reply}\n")
        persona_turns += 1

    print()
    print("=" * 62)
    print(f"  Session complete — {persona_turns} persona turns, {trainee_turns} trainee turns.")
    print("=" * 62)
