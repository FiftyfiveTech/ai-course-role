"""FastAPI web front end for the roleplay session.

Reuses the exact same objects session.run()/demo() drive — Controller,
PersonaAgent, SessionLogger, EvaluatorAgent, CoachAgent, render_scorecard_html
— replacing the blocking input() loop with one HTTP request per turn. No
agent-layer module changes; see ARCHITECTURE.md for who-sees-what, which this
does not alter.

In-progress sessions live only in the _LIVE registry below — Controller's
mood/turn state and PersonaAgent's message history are plain Python objects
with no on-disk resume path (only SessionLogger's turn index resumes from the
session log). A server restart mid-conversation loses that session; this is
an accepted limitation for a single-dev local tool, not a production service.
Completed sessions need no registry entry — they're re-scored from
sessions/<id>.jsonl on demand, same as `scripts/roleplay.py --render` today.
"""

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from role.coach import CoachAgent
from role.controller import Controller
from role.evaluator import EvaluatorAgent
from role.logger import SessionLogger, SESSIONS_DIR
from role.logging_config import configure_logging
from role.persona import PersonaAgent
from role.scenario import Scenario, load_all, SCENARIOS_DIR
from role.scorecard_html import render_report_fragment, render_scorecard_html
from role.session import DOCS_SCORECARDS_DIR, TOTAL_TURNS

configure_logging()
log = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="ROLE — Roleplay & Skills Coach")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Safe/free no-op without OTEL_EXPORTER_OTLP_ENDPOINT set (role.tracing) — one
# span per request either way, and route handlers' persona/evaluator/coach
# spans nest under it (anyio's threadpool dispatch propagates contextvars).
FastAPIInstrumentor().instrument_app(app)


@dataclass
class LiveSession:
    """One in-progress session's live agent objects plus the transcript
    the templates render — mirrors the turn bookkeeping role.session.run()
    does with its own local persona_turns/trainee_turns counters."""

    scenario: Scenario
    controller: Controller
    persona: PersonaAgent
    logger: SessionLogger
    transcript: list[dict] = field(default_factory=list)
    persona_turns: int = 0
    trainee_turns: int = 0

    @property
    def done(self) -> bool:
        return self.persona_turns + self.trainee_turns >= TOTAL_TURNS


_LIVE: dict[str, LiveSession] = {}
_LIVE_LOCK = threading.Lock()  # guards _LIVE membership only, held briefly


def _claim_live(session_id: str) -> Optional[LiveSession]:
    """Atomically pop *session_id* out of the registry to claim exclusive
    processing of one turn. Route handlers here are plain `def`s, so
    Starlette runs each request on its own threadpool thread — a duplicate
    submit for the same session_id (easy to trigger: this route's final turn
    takes several seconds, running live Evaluator/Coach calls, with no UI
    feedback in between) would otherwise race a plain dict lookup + delete.
    Returns None if there's nothing to claim (already completed, or another
    request is already processing this session's turn)."""
    with _LIVE_LOCK:
        return _LIVE.pop(session_id, None)


def _release_live(session_id: str, live: LiveSession) -> None:
    """Put a claimed, still-in-progress session back so its next turn can
    be processed. Not called when this turn completed the session."""
    with _LIVE_LOCK:
        _LIVE[session_id] = live


def _scenario_by_id(scenario_id: str) -> Scenario:
    for scenario in load_all(SCENARIOS_DIR):
        if scenario.id == scenario_id:
            return scenario
    raise ValueError(f"webapp: unknown scenario id {scenario_id!r}")


def _latest_session_id() -> Optional[str]:
    if not SESSIONS_DIR.exists():
        return None
    logs = sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0].stem if logs else None


def _score_and_render(session_id: str) -> str:
    """Evaluate + plan + persist the static artifact; return the report
    fragment — the same call scripts/roleplay.py's --render path makes."""
    scorecard = EvaluatorAgent().evaluate(session_id)
    plan = CoachAgent().plan(scorecard)
    html_text = render_scorecard_html(scorecard, plan)
    DOCS_SCORECARDS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_SCORECARDS_DIR / f"{session_id}.html").write_text(html_text, encoding="utf-8")
    log.info("session scored: session_id=%s score=%d/%d", session_id, scorecard.total, scorecard.max)
    return render_report_fragment(scorecard, plan)


@app.get("/")
def index(request: Request):
    scenarios = load_all(SCENARIOS_DIR)
    return templates.TemplateResponse(
        request,
        "index.html",
        {"scenarios": scenarios, "latest_session_id": _latest_session_id(), "active": "new"},
    )


@app.post("/session")
def create_session(scenario_id: str = Form(...)):
    scenario = _scenario_by_id(scenario_id)
    logger = SessionLogger()
    system_prompt = scenario.policy_path.read_text(encoding="utf-8")
    persona = PersonaAgent(system_prompt)
    controller = Controller(scenario.difficulty)

    live = LiveSession(scenario=scenario, controller=controller, persona=persona, logger=logger)

    state = controller.initial_state()
    persona.set_state(state.render())
    opening = persona.reply(scenario.opening, logger=logger, controller_state=state.to_dict())
    live.transcript.append({"role": "persona", "text": opening})
    live.persona_turns = 1

    _release_live(logger.session_id, live)
    log.info("session created: scenario=%s session_id=%s", scenario.id, logger.session_id)
    return RedirectResponse(f"/session/{logger.session_id}", status_code=303)


@app.get("/session/{session_id}")
def view_session(request: Request, session_id: str):
    live = _LIVE.get(session_id)
    if live is None:
        # Already completed (or unknown) — its report lives in history now.
        return RedirectResponse(f"/history/{session_id}", status_code=303)
    return templates.TemplateResponse(
        request,
        "session.html",
        {
            "session_id": session_id,
            "scenario": live.scenario,
            "transcript": live.transcript,
            "done": False,
        },
    )


@app.post("/session/{session_id}/turn")
def post_turn(request: Request, session_id: str, line: str = Form(...)):
    live = _claim_live(session_id)
    if live is None:
        # Nothing to claim — already completed, or another request is mid-
        # flight processing this same session's turn (e.g. a duplicate
        # submit). Its report lives in history once scored.
        return RedirectResponse(f"/history/{session_id}", status_code=303)
    line = line.strip() or "[no response]"

    state = live.controller.step(line)
    live.logger.log(role="trainee", content=line, controller_state=state.to_dict())
    live.trainee_turns += 1
    live.transcript.append({"role": "trainee", "text": line})

    report_html = None
    if live.done:
        report_html = _score_and_render(session_id)
        # Already popped by _claim_live() above — nothing to remove.
    else:
        live.persona.set_state(state.render())
        reply = live.persona.reply(line, logger=live.logger, controller_state=state.to_dict())
        live.transcript.append({"role": "persona", "text": reply})
        live.persona_turns += 1
        _release_live(session_id, live)

    log.info(
        "turn processed: session_id=%s persona_turns=%d trainee_turns=%d done=%s",
        session_id, live.persona_turns, live.trainee_turns, report_html is not None,
    )

    return templates.TemplateResponse(
        request,
        "session.html",
        {
            "session_id": session_id,
            "scenario": live.scenario,
            "transcript": live.transcript,
            "done": report_html is not None,
            "report_html": report_html,
        },
    )


@app.get("/history")
def history(request: Request):
    logs = sorted(SESSIONS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if SESSIONS_DIR.exists() else []
    sessions = [
        {
            "session_id": p.stem,
            "in_progress": p.stem in _LIVE,
            "scored": (DOCS_SCORECARDS_DIR / f"{p.stem}.html").exists(),
        }
        for p in logs
    ]
    return templates.TemplateResponse(
        request, "history.html", {"sessions": sessions, "active": "history"}
    )


@app.get("/history/{session_id}")
def history_detail(request: Request, session_id: str):
    report_html = _score_and_render(session_id)
    return templates.TemplateResponse(
        request,
        "history_detail.html",
        {"session_id": session_id, "report_html": report_html, "active": "history"},
    )
