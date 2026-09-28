"""Session logger: every turn persisted to sessions/<id>.jsonl.

Each record carries a stable turn_id (`<session_id>:<turn_index>`) that the
scorecard will use to cite specific transcript segments.  Resuming a session
continues from the last written turn_index so ids never change.

Cost is recorded at the published Groq rate for the model.  The project runs
on the free tier so cost_usd is always 0.0 — update _RATES if that changes.
"""

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

SESSIONS_DIR = Path(__file__).parent.parent.parent / "sessions"

# Published Groq rates (USD per 1 M tokens).  Free-tier → 0.0.
# Update both values if the project moves to a paid tier.
_RATES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.0, 0.0),  # (prompt, completion)
}
_DEFAULT_RATE = (0.0, 0.0)


def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_rate, completion_rate = _RATES.get(model, _DEFAULT_RATE)
    return (prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1_000_000


@dataclass
class TurnRecord:
    turn_id: str
    session_id: str
    turn_index: int
    role: str  # "persona" | "trainee"
    model: Optional[str]
    content: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    seconds: Optional[float]
    cost_usd: Optional[float]
    controller_state: Optional[dict] = None


class SessionLogger:
    """Appends one JSONL line per turn to sessions/<session_id>.jsonl."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        SESSIONS_DIR.mkdir(exist_ok=True)
        self.session_id: str = session_id if session_id else uuid.uuid4().hex[:12]
        self._path = SESSIONS_DIR / f"{self.session_id}.jsonl"
        self._turn_index: int = self._resume_index()

    def _resume_index(self) -> int:
        """Return the next turn_index (1-based), continuing after any prior turns.

        Raises ValueError if the file exists but contains no valid turn records —
        an empty transcript is a corrupt state, not a fresh session.
        """
        if not self._path.exists():
            return 1
        lines = [ln.strip() for ln in self._path.read_text().splitlines() if ln.strip()]
        if not lines:
            raise ValueError(
                f"Session file {self._path} exists but contains no turn records. "
                "Delete it to start a new session or fix the corrupt file."
            )
        return json.loads(lines[-1])["turn_index"] + 1

    def log(
        self,
        role: str,
        content: str,
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        seconds: Optional[float] = None,
        controller_state: Optional[dict] = None,
    ) -> str:
        """Write one turn record; return the stable turn_id."""
        if prompt_tokens is not None and completion_tokens is not None:
            total: Optional[int] = prompt_tokens + completion_tokens
            cost: Optional[float] = _cost(model or "", prompt_tokens, completion_tokens)
        else:
            total = None
            cost = None

        record = TurnRecord(
            turn_id=f"{self.session_id}:{self._turn_index}",
            session_id=self.session_id,
            turn_index=self._turn_index,
            role=role,
            model=model,
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            seconds=round(seconds, 3) if seconds is not None else None,
            cost_usd=cost,
            controller_state=controller_state,
        )
        with self._path.open("a") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")

        turn_id = record.turn_id
        self._turn_index += 1
        return turn_id
