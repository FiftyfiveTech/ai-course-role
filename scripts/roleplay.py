#!/usr/bin/env python3
"""scripts/roleplay.py — ROLE-030: the roleplay CLI + session endpoint.

Default: runs one interactive session end-to-end — scenario in, session +
evidence-linked scorecard + learning plan out — writing
docs/scorecards/<session_id>.html.

  uv run python scripts/roleplay.py                        interactive demo
  uv run python scripts/roleplay.py --scenario billing-dispute
  uv run python scripts/roleplay.py --scripted             no typing, canned trainee lines
  uv run python scripts/roleplay.py --render <session_id>  re-score an existing session log
"""

import argparse
import os
import sys
from pathlib import Path

_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from role.coach import CoachAgent  # noqa: E402
from role.evaluator import EvaluatorAgent  # noqa: E402
from role.scenario import load, SCENARIOS_DIR  # noqa: E402
from role.scorecard_html import render_scorecard_html  # noqa: E402
from role.session import DOCS_SCORECARDS_DIR, demo  # noqa: E402

# Generic 5-line trainee script for --scripted — no typing required, works
# reasonably across any scenario's opening.
SCRIPTED_TRAINEE_LINES = [
    "Hi, thanks for calling — I understand you have a question, let me help with that.",
    "Can you tell me a bit more about what happened, and your order details?",
    "I understand — let me see what options we have here.",
    "I can go ahead and make that change for you, is that okay?",
    "You're all set — you'll get a confirmation shortly. Anything else I can help with?",
]


def _print_view_instructions(path: Path) -> None:
    print()
    print("View it:")
    print(f"  open directly : {path}")
    print("  or serve it   : python3 -m http.server 8000 --directory docs")
    print(f"                  then open http://localhost:8000/scorecards/{path.name}")
    print()


def _render_existing(session_id: str) -> Path:
    """The 'session endpoint': (re)score an already-logged session and
    (re)write its HTML scorecard, without re-running the persona."""
    scorecard = EvaluatorAgent().evaluate(session_id)
    plan = CoachAgent().plan(scorecard)
    html_text = render_scorecard_html(scorecard, plan)
    DOCS_SCORECARDS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_SCORECARDS_DIR / f"{session_id}.html"
    out_path.write_text(html_text, encoding="utf-8")
    print(f"Scorecard : {scorecard.total}/{scorecard.max}")
    print(f"Written   : {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="ROLE roleplay CLI + session endpoint")
    parser.add_argument(
        "--scenario", default="order-change", help="scenario id (scenarios/*.yaml stem)"
    )
    parser.add_argument(
        "--scripted", action="store_true", help="no typing — 5 canned trainee lines"
    )
    parser.add_argument(
        "--render", metavar="SESSION_ID", help="re-score an existing session log by id"
    )
    args = parser.parse_args()

    if args.render:
        out_path = _render_existing(args.render)
        _print_view_instructions(out_path)
        return 0

    scenario = load(SCENARIOS_DIR / f"{args.scenario}.yaml")
    trainee_lines = SCRIPTED_TRAINEE_LINES if args.scripted else None

    out_path = demo(scenario=scenario, trainee_lines=trainee_lines)
    if out_path is None:
        print("[session interrupted — nothing to score]")
        return 1

    _print_view_instructions(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
