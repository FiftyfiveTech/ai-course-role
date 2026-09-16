"""Persona agent — openai/gpt-oss-120b via Groq, with Ollama fallback.

Arm policy
----------
* If Groq returns 429 (rate-limit) or a daily-cap error the call falls back
  to Ollama (model: hf.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF).
  A warning is printed so the operator knows which arm is serving.
* ``repeat(n=3)`` runs the same prompt three times and returns the text if all
  three agree (stripped).  If any reply differs it prints NON-REPRODUCIBLE and
  returns None — results are never averaged.
"""

import json as _json
import os
import time
import urllib.request
from typing import TYPE_CHECKING

from groq import Groq, RateLimitError

if TYPE_CHECKING:
    from role.logger import SessionLogger

GROQ_MODEL = "openai/gpt-oss-120b"
OLLAMA_MODEL = "hf.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF"
OLLAMA_URL = "http://localhost:11434/api/chat"
MAX_TOKENS = 300


def _ollama_chat(messages: list[dict], max_tokens: int = MAX_TOKENS) -> tuple[str, float]:
    """Call the local Ollama server; return (text, elapsed_seconds).

    Raises RuntimeError if Ollama is unreachable or returns an error.
    """
    payload = _json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = _json.loads(resp.read())
    except Exception as exc:
        raise RuntimeError(f"Ollama unreachable at {OLLAMA_URL}: {exc}") from exc
    elapsed = time.perf_counter() - t0
    return data["message"]["content"].strip(), elapsed


class PersonaAgent:
    def __init__(self, system_prompt: str) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set — run `make doctor`")
        self._client = Groq(api_key=api_key)
        self._history: list[dict] = [{"role": "system", "content": system_prompt}]

    def _call_groq(self, messages: list[dict]) -> tuple[str, float, object]:
        """Return (text, elapsed, usage) from Groq."""
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
        )
        elapsed = time.perf_counter() - t0
        return response.choices[0].message.content.strip(), elapsed, response.usage

    def reply(self, user_message: str, logger: "SessionLogger | None" = None) -> str:
        """Append user_message to history, call the model, return the reply.

        Groq is tried first.  On 429 or daily-cap error the call falls back to
        Ollama and prints a warning.  If *logger* is provided each persona turn
        is persisted with the HF model id, token counts, wall-clock seconds,
        and cost.
        """
        self._history.append({"role": "user", "content": user_message})

        model_used = GROQ_MODEL
        prompt_tokens = completion_tokens = None

        try:
            text, elapsed, usage = self._call_groq(self._history)
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
        except RateLimitError as exc:
            print(
                f"[arm-policy] Groq 429/cap — falling back to Ollama"
                f" ({OLLAMA_MODEL}): {exc}"
            )
            text, elapsed = _ollama_chat(self._history)
            model_used = OLLAMA_MODEL

        self._history.append({"role": "assistant", "content": text})

        if logger is not None:
            logger.log(
                role="persona",
                content=text,
                model=model_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                seconds=elapsed,
            )
        return text

    def repeat(self, user_message: str, n: int = 3) -> "str | None":
        """Run the same prompt *n* times; return text if all agree, else None.

        Prints NON-REPRODUCIBLE when any reply differs.  Results are never
        averaged — a disagreeing run is a signal, not noise to smooth over.
        """
        messages = self._history + [{"role": "user", "content": user_message}]
        replies: list[str] = []
        for i in range(n):
            try:
                text, _, _ = self._call_groq(messages)
            except RateLimitError as exc:
                print(
                    f"[arm-policy] Groq 429/cap on repeat {i + 1}"
                    f" — falling back to Ollama: {exc}"
                )
                text, _ = _ollama_chat(messages)
            replies.append(text)

        if len(set(replies)) == 1:
            return replies[0]

        print(
            f"NON-REPRODUCIBLE — {n} runs returned {len(set(replies))} distinct replies:\n"
            + "\n---\n".join(f"[{i + 1}] {r}" for i, r in enumerate(replies))
        )
        return None

    @property
    def history(self) -> list[dict]:
        return list(self._history)
