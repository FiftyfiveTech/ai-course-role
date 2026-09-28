# ROLE-023 — Evaluator agent findings

## Model id: two doc mismatches, resolved against Groq's live model list

The ticket named `Qwen/Qwen3.6-27B`; ARCHITECTURE.md's earlier draft named
`Qwen/Qwen3-27B`. Neither exists on Groq — a live call to both 404'd with
`model_not_found`. `client.models.list()` shows Groq only hosts
`qwen/qwen3.8-27b`. Per user decision, the code and ARCHITECTURE.md now use
`Qwen/Qwen3.8-27B` (HF-style id, used for logging/cost-rate lookups) with
`qwen/qwen3.8-27b` as the literal Groq API model string (`GROQ_API_MODEL_ID`
in `src/role/evaluator.py`) — Groq lowercases its model slugs, unlike
`openai/gpt-oss-120b` where the two happen to coincide.

## Attempt 1 (failed) — Mode.TOOLS, dict[str, ScoreItem] with no completeness check

Command: single-seed smoke test (`seed_to_session("bitext-1000")` +
`EvaluatorAgent().evaluate(...)`) then the full 8-seed
`uv run python scripts/run_evaluator_dev_gate.py`.

Result: 3/8 failed outright with `groq.NotFoundError`-adjacent `tool_use_failed`
400s — the model's `failed_generation` was degenerate repeated whitespace/
box-drawing characters, never a valid function call. A 4th (`hh-80019`)
returned a *schema-valid* `_RubricScores` (no error from instructor) but with
an empty/wrong-keyed `items` — `Scorecard`'s own missing-keys check caught it
only *after* the LLM call, too late for instructor's retry loop to act on.
Root cause of the second failure: `_RubricScores` (the slim response model
built to avoid asking the LLM for known bookkeeping fields, see
`src/role/evaluator.py`) validated fine on any `dict[str, ScoreItem]`
regardless of which keys were present — the actual rubric-completeness
requirement lived only on `Scorecard`, one validation stage too late.

Also crashed the report printer itself: an error string containing the
model's garbled unicode output raised `UnicodeEncodeError` on the Windows
console's default cp1252 encoding.

## Attempt 2 (fixed) — Mode.JSON + completeness validator on the LLM-facing model

Two changes, both in `src/role/evaluator.py`:
1. `instructor.from_groq(..., mode=instructor.Mode.JSON)` instead of
   `Mode.TOOLS` — direct JSON-object generation, not Groq's function-calling
   path, which this Qwen build handled unreliably on this schema.
2. Added the same missing/extra-rubric-key `@model_validator` that
   `Scorecard` has to `_RubricScores` too, so instructor's own retry loop
   sees (and can react to) an incomplete `items` dict instead of silently
   accepting it.

Also fixed `scripts/run_evaluator_dev_gate.py`'s report printer:
`sys.stdout.reconfigure(encoding="utf-8", errors="replace")` plus truncating
the displayed error detail to 200 chars, so a future failure's raw model
output can never crash the gate script itself.

Command: `uv run python scripts/run_evaluator_dev_gate.py`
```
seed_id           status      detail
----------------------------------------------------------------------
bitext-1000       PASS
bitext-5000       PASS
hh-80002          PASS
hh-80012          PASS
hh-80019          PASS
soda-0100         PASS
soda-0500         PASS
soda-1000         PASS
----------------------------------------------------------------------
8/8 schema-valid

GATE: PASS — 8/8 schema-valid
```

`evals/scores/dev/*.json` now holds all 8 real, schema-valid `Scorecard`
objects with genuine `TurnRef` evidence citing the seed-adapter-produced
`sessions/<seed_id>.jsonl` transcripts. `evals/dev/*.score.json` (hand-scored
gold labels) and `evals/heldout/` are untouched (`git status`/`git diff`
confirm zero changes there); `tests/gates/test_no_leakage.py` still passes.

## What this does not claim

This is a schema-validity gate, not an agreement measurement — whether these
8 agent-produced scores actually *agree* with the hand-scored gold labels is
ROLE-024's job (Cohen's κ, exact/±1-band agreement), not this ticket's.
