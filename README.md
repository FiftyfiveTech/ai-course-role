# ai-course-template

Starting scaffold for the FiftyFive AI engineering course. One repo per track, grown phase by
phase by hand.

## Use it

```bash
gh repo create FiftyfiveTech/ai-course-<track> --template FiftyfiveTech/ai-course-template --public --clone
cd ai-course-<track>
make setup
```

Then, in order: replace `<TRACK>` in `CLAUDE.md` and `pyproject.toml`, protect `main`, add the
other person as a collaborator with push access.

## Layout

| Path | Holds |
|---|---|
| `src/` | The system. Small modules, one job each. |
| `prompts/` | Versioned prompt files (`extract_v1.md`, `extract_v2.md`, …). Never inline a prompt in code. |
| `schemas/` | Pydantic models. Structured output is validated, not parsed by hand. |
| `evals/dev/` | **Builder** tunes here. 15 cases. |
| `evals/heldout/` | **Evaluator** only. Sealed Wednesday, tagged `heldout-v1`. The Builder never reads it. |
| `tests/gates/` | One script per phase gate. It prints the number; the number decides. |
| `STANDUP.md` | Daily log. Two minutes, append-only. |

## Deliberately missing

Two files are absent because they are Week 0 tasks, not scaffolding:

- `tests/gates/test_no_leakage.py` — asserts `evals/dev ∩ evals/heldout = ∅` by content hash
  (task **0.7**). Until it exists, the blind-labelling rule is unenforced.
- `src/telemetry.py` — the shared cost/latency logger every model call goes through (task **0.8**).

Write them. Do not import them from somewhere else.

## Rules that live in this repo

`CLAUDE.md` carries the full contract. The short version:

- Models and datasets are named by **Hugging Face repo id**. The provider is only where it runs.
- **Zero spend.** A paid call is a STOP-and-ask, never a judgement call.
- A phase is done when its gate **prints the number**, not when the code looks right.
- Every PR is reviewed by the other person. `main` is protected; self-merges are the one thing
  the Friday retro always checks.
- Tasks come from the Odoo board via the `odoo-board` MCP server, not from this README.

## heldout-v1 seal

12 conversations hand-scored against `evals/RUBRIC.md` by ritika@fiftyfivetech.io on 2026-09-15.

sha256 of `evals/heldout/` score files (concatenated, sorted by filename):

```
13196da5bc65e79b10c6a341032f7d101eb267c07f7c3ef7230d4acb47323241
```

Scores summary (total / 12 per conversation):

| Seed | Scored Speaker | Total |
|------|---------------|-------|
| bitext-10000 | Agent | 11 |
| bitext-12000 | Agent | 10 |
| bitext-3000 | Agent | 9 |
| bitext-7000 | Agent | 10 |
| hh-80007 | Assistant (chosen) | 9 |
| hh-80014 | Assistant (chosen) | 10 |
| hh-80015 | Assistant (chosen) | 8 |
| hh-80023 | Assistant (chosen) | 11 |
| soda-0502 | Instructor | 9 |
| soda-1500 | Friend | 10 |
| soda-2000 | Friend | 10 |
| soda-2500 | Jayci | 11 |
