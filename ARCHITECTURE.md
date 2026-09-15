# ROLE — Roleplay & Skills Coach: Architecture

**Status:** Draft — pending sign-off by Ritika, Vimal, and supervisor
**Last updated:** 2026-09-15

---

## 1. Purpose

The system lets a trainee practise a sales or support conversation against a simulated customer
persona. After the session, an independent evaluator agent scores the transcript against a
published rubric, and a coach agent turns those scores into a targeted learning plan.

No agent is allowed to claim personality, emotion, or mental state. The persona plays a role;
it does not pretend to be sentient.

---

## 2. Agent Topology

```
┌─────────────────────────────────────────────────────────────┐
│                        USER (Trainee)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │ text (CLI / voice leg optional)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      CONTROLLER                              │
│  - loads scenario YAML                                       │
│  - manages turn loop                                         │
│  - enforces consent gate (see §4)                            │
│  - writes every turn to session log                          │
│  - routes to Persona or ends session                         │
└──────────┬──────────────────────────────┬───────────────────┘
           │ persona prompt + history      │ full session log
           ▼                              ▼
┌──────────────────────┐      ┌──────────────────────────────┐
│       PERSONA         │      │           OBSERVER            │
│  model: openai/       │      │  read-only; watches for       │
│  gpt-oss-120b         │      │  drift, policy breaches,      │
│  via Groq             │      │  or cost thresholds;          │
│                       │      │  flags → Controller           │
│  sees:                │      │                               │
│  • system prompt      │      │  sees:                        │
│  • scenario config    │      │  • full session log           │
│  • conversation hist  │      │  • scenario config            │
│                       │      │  • does NOT see rubric        │
│  does NOT see:        │      └──────────────────────────────┘
│  • rubric             │
│  • evaluator output   │      (after session ends)
│  • coach output       │               │
└──────────────────────┘               ▼
                            ┌──────────────────────────────┐
                            │          EVALUATOR            │
                            │  model: Qwen/Qwen3-27B        │
                            │  via Groq / Ollama fallback   │
                            │  structured output via        │
                            │  instructor + Pydantic        │
                            │                               │
                            │  sees:                        │
                            │  • session log (turn ids)     │
                            │  • evals/RUBRIC.md            │
                            │                               │
                            │  does NOT see:                │
                            │  • persona system prompt      │
                            │  • scenario difficulty vars   │
                            │  • coach output               │
                            │                               │
                            │  outputs: Scorecard           │
                            │  (each score cites ≥1 TurnRef)│
                            └──────────────┬───────────────┘
                                           │ Scorecard only
                                           ▼
                            ┌──────────────────────────────┐
                            │            COACH              │
                            │  model: openai/gpt-oss-20b   │
                            │                               │
                            │  sees:                        │
                            │  • Scorecard (scores +        │
                            │    evidence references)       │
                            │                               │
                            │  does NOT see:                │
                            │  • raw transcript             │
                            │  • persona system prompt      │
                            │                               │
                            │  outputs: learning plan       │
                            │  grounded in rubric items     │
                            └──────────────────────────────┘
```

---

## 3. Who Sees What

| Data | Controller | Persona | Observer | Evaluator | Coach |
|------|:---:|:---:|:---:|:---:|:---:|
| Scenario config (goal, difficulty, opening) | ✅ | ✅ | ✅ | — | — |
| Persona system prompt | ✅ | ✅ | — | — | — |
| Conversation history (live) | ✅ | ✅ | — | — | — |
| Full session log (post-turn) | ✅ | — | ✅ | ✅ | — |
| evals/RUBRIC.md | — | — | — | ✅ | — |
| Scorecard (scores + TurnRefs) | — | — | — | — | ✅ |
| Raw transcript | — | — | — | ✅ | — |
| Cost / latency metadata | ✅ | — | ✅ | — | — |

**Key invariant:** The Evaluator and Coach use a **different model** from the Persona.
Self-scoring by the same model inflates agreement scores (self-preference bias).

---

## 4. Consent Gate

Before the session starts the Controller prints a plain-language disclosure and requires
explicit confirmation:

```
This session will be recorded and scored.
Your transcript will be read by an evaluator model and a coach model.
No human will read your transcript without your permission.
Type YES to continue or NO to exit.
```

- `NO` → session exits cleanly, nothing is logged.
- `YES` → session log is created; the session id is printed so the trainee can request deletion.
- The gate runs **before** any model call. No tokens are consumed on refusal.

---

## 5. Data Flow and Persistence

```
turn N
  │
  ├─► Controller appends to session_log/{session_id}.jsonl
  │     fields: turn_id, role, text, model, tokens_in, tokens_out,
  │             latency_s, cost_usd, timestamp_utc
  │
  └─► Observer reads the appended line (read-only)

end of session
  │
  ├─► Evaluator reads session_log/{session_id}.jsonl
  │     outputs: evals/scores/{session_id}.json  (Scorecard schema)
  │
  └─► Coach reads evals/scores/{session_id}.json
        outputs: evals/plans/{session_id}.md
```

Session logs live in `session_log/`. They are **never** in `evals/heldout/` — that
directory is sealed and belongs to the Evaluator role only.

---

## 6. Arm Policy (model routing)

| Arm | Primary | Fallback | Trigger |
|-----|---------|----------|---------|
| Persona | Groq → `openai/gpt-oss-120b` | Ollama → `hf.co/openai/gpt-oss-120b` | Groq 429 or daily cap |
| Evaluator | Groq → `Qwen/Qwen3-27B` | Ollama → `hf.co/Qwen/Qwen3-27B` | Groq 429 or daily cap |
| Coach | Groq → `openai/gpt-oss-20b` | Ollama → `hf.co/openai/gpt-oss-20b` | Groq 429 or daily cap |

Every call goes through the shared cost/latency logger regardless of arm.
A run that disagrees across n=3 repeats is reported **NON-REPRODUCIBLE** — never averaged.

---

## 7. Boundaries and Safety

- The Persona prompt explicitly forbids claims of personality, emotion, or mental state.
- `openai/gpt-oss-safeguard-20b` runs a boundary check on 10 adversarial cases before any
  held-out evaluation (ROLE-025).
- The Observer flags boundary violations to the Controller mid-session; the Controller can
  terminate the session and log the violation without exposing it to the Evaluator.

---

## 8. Sign-off

| Name | Role | Signed |
|------|------|--------|
| Ritika | Builder | ☐ |
| Vimal | Builder | ☐ |
| Supervisor | Supervisor | ☐ |
| | | |

**No code is written until all three sign-off boxes are checked.**
