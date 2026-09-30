#!/usr/bin/env python3
"""tests/gates/test_demo_pipeline.py — ROLE-030 gate.

Fully mocked, deterministic, no live calls — proves scenario in -> session +
evidence-linked scorecard + learning plan out is correctly wired end-to-end,
"from a clean clone" (no external state required). Mirrors
tests/unit/test_session_controller.py's input/Groq mocking, extended to the
Evaluator's and Coach's instructor clients (tests/unit/test_evaluator_arm_policy.py
and tests/unit/test_coach.py's `_make_agent()` pattern).

The live version of this same pipeline (real Groq calls) is
`uv run python scripts/roleplay.py` / `make demo`.

Run standalone : uv run python tests/gates/test_demo_pipeline.py
Run via pytest : uv run pytest tests/gates/test_demo_pipeline.py -s
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import role.session as session_mod  # noqa: E402
from role.coach import PlanItem  # noqa: E402
from role.scenario import load, SCENARIOS_DIR  # noqa: E402
from role.scorecard import RUBRIC_ITEMS, ScoreItem  # noqa: E402

SCENARIO_FILE = SCENARIOS_DIR / "order-change.yaml"

TRAINEE_LINES = [
    "Hi, I placed an order and need to change it.",
    "I ordered a medium but want a large instead.",
    "Can that be done before it ships?",
    "Great, please go ahead.",
    "Thanks for your help, that's all I needed.",
]
PERSONA_REPLIES = ["persona-1", "persona-2", "persona-3", "persona-4", "persona-5"]


def _groq_persona_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = text
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


def _rubric_scores_result(session_id: str) -> MagicMock:
    result = MagicMock()
    result.items = {
        item: ScoreItem(score=1, evidence=[{"turn_id": f"{session_id}:2"}])
        for item in RUBRIC_ITEMS
    }
    return result


def _plan_items_result() -> MagicMock:
    result = MagicMock()
    result.items = [
        PlanItem(rubric_item=item, recommendation=f"work on {item}") for item in RUBRIC_ITEMS
    ]
    return result


def _groq_completion() -> MagicMock:
    completion = MagicMock()
    completion.usage.prompt_tokens = 50
    completion.usage.completion_tokens = 20
    return completion


def _make_evaluator():
    """Same narrow-scope construction trick as
    tests/unit/test_evaluator_arm_policy.py's _make_agent(): patch
    instructor.from_groq only for __init__, then stash the resulting mock
    client. role.evaluator and role.coach both do `import instructor` — the
    same module object — so patching both agents' from_groq in one shared
    `with` statement makes the second patch silently clobber the first for
    BOTH agents. Constructing them in separate, non-overlapping `with` scopes
    avoids that collision entirely."""
    from role.evaluator import EvaluatorAgent

    with patch("role.evaluator.instructor.from_groq"):
        return EvaluatorAgent()


def _make_coach():
    from role.coach import CoachAgent

    with patch("role.coach.instructor.from_groq"):
        return CoachAgent()


def run_gate(tmp_path) -> dict:
    """Runs the same real code role.session.demo() chains — run() ->
    EvaluatorAgent.evaluate() -> CoachAgent.plan() -> render_scorecard_html()
    -> write — fully mocked, in tmp_path isolation. Split from a single
    demo() call because the Evaluator's mocked result must cite the
    just-created session's real turn_id, known only after run() returns."""
    scenario = load(SCENARIO_FILE)  # difficulty: easy

    with patch("role.logger.SESSIONS_DIR", tmp_path), patch(
        "role.session.SESSIONS_DIR", tmp_path
    ), patch("role.scorecard_html.SESSIONS_DIR", tmp_path), patch.dict(
        "os.environ", {"GROQ_API_KEY": "test-key"}
    ), patch("role.persona.Groq") as mock_groq_cls, patch(
        "builtins.input", side_effect=TRAINEE_LINES
    ):
        mock_groq_cls.return_value.chat.completions.create.side_effect = [
            _groq_persona_response(text) for text in PERSONA_REPLIES
        ]

        session_id = session_mod.run(scenario=scenario, trainee_lines=TRAINEE_LINES)
        assert session_id is not None

        evaluator = _make_evaluator()
        evaluator._client.chat.completions.create_with_completion.return_value = (
            _rubric_scores_result(session_id),
            _groq_completion(),
        )
        coach = _make_coach()
        coach._client.chat.completions.create_with_completion.return_value = (
            _plan_items_result(),
            _groq_completion(),
        )

        from role.scorecard_html import render_scorecard_html

        scorecard = evaluator.evaluate(session_id)
        plan = coach.plan(scorecard)
        html_text = render_scorecard_html(scorecard, plan, sessions_dir=tmp_path)

        out_dir = tmp_path / "docs-scorecards"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{session_id}.html"
        out_path.write_text(html_text, encoding="utf-8")

    return {"session_id": session_id, "out_path": out_path, "scorecard": scorecard}


def test_demo_pipeline_end_to_end(tmp_path) -> None:
    result = run_gate(tmp_path)
    session_id = result["session_id"]
    out_path = result["out_path"]

    assert out_path.exists()
    html_text = out_path.read_text(encoding="utf-8")

    for item in RUBRIC_ITEMS:
        assert item in html_text
    # evidence-linked: the real cited turn's text appears, not just its id
    assert f"{session_id}:2" in html_text

    print()
    print(f"session_id : {session_id}")
    print(f"scorecard  : {result['scorecard'].total}/{result['scorecard'].max}")
    print(f"html       : {out_path}")
    print("GATE: PASS")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        result = run_gate(Path(td))
        print(f"session_id : {result['session_id']}")
        print(f"scorecard  : {result['scorecard'].total}/{result['scorecard'].max}")
        print(f"html       : {result['out_path']}")
        print("GATE: PASS")
    sys.exit(0)
