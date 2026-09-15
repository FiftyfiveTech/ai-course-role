"""Persona agent — openai/gpt-oss-120b via Groq.

Maintains conversation history and returns the persona's reply for each turn.
Raises on any API error — callers decide whether to fall back to Ollama.
"""

import os
import time
from typing import TYPE_CHECKING

from groq import Groq

if TYPE_CHECKING:
    from role.logger import SessionLogger

MODEL = "openai/gpt-oss-120b"
MAX_TOKENS = 300


class PersonaAgent:
    def __init__(self, system_prompt: str) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set — run `make doctor`")
        self._client = Groq(api_key=api_key)
        self._history: list[dict] = [{"role": "system", "content": system_prompt}]

    def reply(self, user_message: str, logger: "SessionLogger | None" = None) -> str:
        """Append user_message to history, call the model, return the reply.

        If *logger* is provided each persona turn is persisted with the HF
        model id, token counts, wall-clock seconds, and cost.
        """
        self._history.append({"role": "user", "content": user_message})
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=MODEL,
            messages=self._history,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )
        elapsed = time.perf_counter() - t0
        text = response.choices[0].message.content.strip()
        self._history.append({"role": "assistant", "content": text})
        if logger is not None:
            usage = response.usage
            logger.log(
                role="persona",
                content=text,
                model=MODEL,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                seconds=elapsed,
            )
        return text

    @property
    def history(self) -> list[dict]:
        return list(self._history)
