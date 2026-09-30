"""Static HTML scorecard renderer (ROLE-030).

Every score shows the real transcript text of every turn it cites — not just
the turn_id — so the evidence-linked claim is visible, not just asserted.
Reuses docs/role-day1.html's dark CSS variables so the demo output reads as
the same product family as the concept-primer page.
"""

import html
import json
from pathlib import Path
from typing import Optional

from role.coach import LearningPlan
from role.logger import SESSIONS_DIR
from role.scorecard import RUBRIC_ITEMS, Scorecard

THEME_CSS_PATH = Path(__file__).parent / "static" / "theme.css"
_STYLE = THEME_CSS_PATH.read_text(encoding="utf-8")


def _load_turn_text(sessions_dir: Path, turn_id: str) -> str:
    """Return "role: content" for turn_id, read from sessions/<session_id>.jsonl.

    Raises if the turn cannot be found — ROLE-026's evidence-validity gate
    already guarantees real scorecards never cite a dangling ref, so a miss
    here is a real defect, not a case to render around silently.
    """
    session_id, _, _ = turn_id.rpartition(":")
    path = sessions_dir / f"{session_id}.jsonl"
    if not path.exists():
        raise ValueError(f"scorecard_html: session log not found for turn_id {turn_id!r}: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record["turn_id"] == turn_id:
            return f"{record['role']}: {record['content']}"
    raise ValueError(f"scorecard_html: turn_id {turn_id!r} not found in {path}")


def render_report_fragment(
    scorecard: Scorecard, plan: LearningPlan, sessions_dir: Optional[Path] = None
) -> str:
    """Render the results component: rubric item cards (each with its cited
    transcript turns' real text) followed by the Coach's learning plan cards.

    No <html>/<head>/<body> shell — this fragment is meant to be embedded,
    either standalone inside render_scorecard_html()'s page shell below, or
    directly into a webapp page (session-complete view, history detail view)
    so every surface renders the same markup instead of a reimplementation.

    *sessions_dir* defaults to the current role.scorecard_html.SESSIONS_DIR at
    call time (not at import time) — resolved here rather than as a bound
    default parameter so tests can monkeypatch it, same convention as
    role.session's own SESSIONS_DIR usage.
    """
    if sessions_dir is None:
        sessions_dir = SESSIONS_DIR

    if scorecard.session_id != plan.session_id:
        raise ValueError(
            f"scorecard_html: scorecard.session_id {scorecard.session_id!r} != "
            f"plan.session_id {plan.session_id!r}"
        )

    session_id = scorecard.session_id
    score_emoji = {0: "❌", 1: "⚠️", 2: "✅"}
    items_html = []
    for name in RUBRIC_ITEMS:
        item = scorecard.items[name]
        evidence_html = "".join(
            f'<div class="evidence">'
            f'<div class="turn-id">{html.escape(ref.turn_id)}</div>'
            f'<div class="turn-text">{html.escape(_load_turn_text(sessions_dir, ref.turn_id))}</div>'
            f"</div>"
            for ref in item.evidence
        )
        items_html.append(
            f'<div class="item">'
            f'<div class="item-head">'
            f'<span class="item-name">{html.escape(name)}</span>'
            f'<span class="score s{item.score}">{score_emoji[item.score]} {item.score}/2</span>'
            f"</div>"
            f"{evidence_html}"
            f"</div>"
        )

    plan_html = "".join(
        f'<div class="plan-item">'
        f'<div class="rubric-item">💡 {html.escape(p.rubric_item)}</div>'
        f"<div>{html.escape(p.recommendation)}</div>"
        f"</div>"
        for p in plan.items
    )

    pct = round(100 * scorecard.total / scorecard.max) if scorecard.max else 0
    headline = "🏆" if pct >= 80 else "👍" if pct >= 50 else "📚"

    return f"""<header>
    <div class="label">ROLE · Scorecard</div>
    <h1>{headline} Session {html.escape(session_id)}</h1>
    <p>{scorecard.total} / {scorecard.max} — rubric version {html.escape(scorecard.rubric_version)}</p>
    <div class="score-bar"><div class="score-bar-fill" style="width: {pct}%"></div></div>
  </header>

  <h2>Rubric scores</h2>
  {"".join(items_html)}

  <h2>Learning plan</h2>
  {plan_html}
"""


def render_scorecard_html(
    scorecard: Scorecard, plan: LearningPlan, sessions_dir: Optional[Path] = None
) -> str:
    """Render a self-contained HTML page wrapping render_report_fragment() in
    the standalone page shell — every score with its cited transcript turns'
    real text, followed by the Coach's learning plan.
    """
    session_id = scorecard.session_id
    fragment = render_report_fragment(scorecard, plan, sessions_dir=sessions_dir)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ROLE Scorecard — {html.escape(session_id)}</title>
  <style>{_STYLE}</style>
</head>
<body>
<div class="page">
  {fragment}
  <footer>ROLE — Roleplay &amp; Skills Coach · every score above cites the real transcript turn it is grounded in</footer>
</div>
</body>
</html>
"""
