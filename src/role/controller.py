"""Scenario controller: difficulty and mood as explicit state.

The persona used to decide for itself, from prose in its own system prompt,
when to get more cooperative or more frustrated. That decision now lives here
as code — the same regex/keyword-heuristic style scripts/drift_harness.py
uses for drift flags, so no extra model inference is spent classifying tone.
The persona is only ever told its current mood; it never escalates or
de-escalates on its own.
"""

import re
from dataclasses import dataclass

MOODS = ["cooperative", "frustrated", "escalated"]

# How high mood may climb for a given scenario difficulty — easy scenarios
# never reach "escalated" (mirrors drift_harness.py's DIFF_DRIFT assumption
# that an easy persona should not turn hostile).
DIFFICULTY_CEILING = {
    "easy": "frustrated",
    "medium": "escalated",
    "hard": "escalated",
}

# Ported from the escalation triggers that used to be prose in each persona
# prompt's "How you behave" section (repeat/vague-stall -> escalate).
ESCALATE_PATTERNS = [
    r"\brepeat\b",
    r"\bsay that again\b",
    r"\bone more time\b",
    r"\bwhat was\b",
    r"\blet me check\b",
    r"\bnot sure\b",
    r"\bi'?ll get back to you\b",
]

# (acknowledges/commits/fixes -> de-escalate).
DEESCALATE_PATTERNS = [
    r"\bsorry\b",
    r"\bapologi[sz]e\b",
    r"\bunderstand\b",
    r"\blet me (fix|sort|take care)\b",
    r"\bconfirmed\b",
    r"\bright away\b",
]


@dataclass(frozen=True)
class ControllerState:
    difficulty: str
    mood: str
    controller_turn: int

    def render(self) -> str:
        """A behavioral directive for the persona prompt — never an emotion claim."""
        return (
            f"Controller state — difficulty: {self.difficulty}, mood: {self.mood}, "
            f"turn: {self.controller_turn}. Express this only through pacing, "
            f'directness, and word choice. Never say "I feel X".'
        )

    def to_dict(self) -> dict:
        return {
            "difficulty": self.difficulty,
            "mood": self.mood,
            "controller_turn": self.controller_turn,
        }


class Controller:
    """Tracks one session's difficulty (static) and mood (mutable, code-decided)."""

    def __init__(self, difficulty: str) -> None:
        if difficulty not in DIFFICULTY_CEILING:
            raise ValueError(f"Controller: unknown difficulty {difficulty!r}")
        self._difficulty = difficulty
        self._mood_index = 0
        self._turn = 0

    def initial_state(self) -> ControllerState:
        """State for the persona's opening line, before any trainee input."""
        return ControllerState(self._difficulty, MOODS[self._mood_index], self._turn)

    def step(self, trainee_text: str) -> ControllerState:
        """Advance one turn; update mood from the trainee's latest line."""
        self._turn += 1
        lower = trainee_text.lower()
        ceiling = MOODS.index(DIFFICULTY_CEILING[self._difficulty])

        if any(re.search(p, lower) for p in ESCALATE_PATTERNS):
            self._mood_index = min(self._mood_index + 1, ceiling)
        elif any(re.search(p, lower) for p in DEESCALATE_PATTERNS):
            self._mood_index = max(self._mood_index - 1, 0)

        return ControllerState(self._difficulty, MOODS[self._mood_index], self._turn)
