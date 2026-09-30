#!/usr/bin/env python3
"""tests/gates/test_webapp_pipeline.py — webapp ticket gate.

Fully mocked, deterministic, no live calls — proves the FastAPI webapp drives
the exact same pipeline role.session.demo() does (Controller, PersonaAgent,
EvaluatorAgent, CoachAgent, render_scorecard_html) end to end through HTTP
requests instead of session.run()'s blocking input() loop, and that the
report renders inline the moment a session completes plus identically again
in history.

Mirrors tests/gates/test_demo_pipeline.py's mocking approach, with one
difference: role.evaluator.EvaluatorAgent and role.coach.CoachAgent are both
constructed fresh inside webapp routes (once when a session completes, again
on every /history/{id} view) rather than once by the test itself, so
instructor.from_groq is patched at its one canonical location
(`instructor.from_groq` — role.evaluator and role.coach both `import
instructor`, the same module object) with a single side_effect that
discriminates evaluator vs coach by the `model=` kwarg each one calls
create_with_completion with, instead of test_demo_pipeline.py's
separate-non-overlapping-with-blocks trick (which only works when the agent
is constructed once, up front, by the test itself).

The live version of this same pipeline (real Groq calls) is `make web`.

Run standalone : uv run python tests/gates/test_webapp_pipeline.py
Run via pytest : uv run pytest tests/gates/test_webapp_pipeline.py -s
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from role.coach import GROQ_API_MODEL_ID as COACH_MODEL_ID, PlanItem  # noqa: E402
from role.evaluator import GROQ_API_MODEL_ID as EVALUATOR_MODEL_ID  # noqa: E402
from role.scorecard import RUBRIC_ITEMS, ScoreItem  # noqa: E402

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


def _make_shared_instructor_client(session_id: str) -> MagicMock:
    """One client, shared by every EvaluatorAgent/CoachAgent the webapp
    constructs, discriminated by the `model=` kwarg each one calls
    create_with_completion with."""
    client = MagicMock()

    def _side_effect(*args, model=None, **kwargs):
        if model == EVALUATOR_MODEL_ID:
            return _rubric_scores_result(session_id), _groq_completion()
        if model == COACH_MODEL_ID:
            return _plan_items_result(), _groq_completion()
        raise AssertionError(f"unexpected model kwarg: {model!r}")

    client.chat.completions.create_with_completion.side_effect = _side_effect
    return client


def run_gate(tmp_path) -> dict:
    """Drives the webapp through FastAPI's TestClient: pick a scenario, run
    5 trainee turns, confirm the report renders inline the moment the
    session completes, confirm docs/scorecards/<id>.html was written (same
    artifact scripts/roleplay.py produces), confirm /history lists the
    session and /history/{id} renders the same report."""
    import role.webapp as webapp_mod

    docs_scorecards_dir = tmp_path / "docs-scorecards"

    with patch("role.logger.SESSIONS_DIR", tmp_path), patch(
        "role.webapp.SESSIONS_DIR", tmp_path
    ), patch("role.scorecard_html.SESSIONS_DIR", tmp_path), patch(
        "role.webapp.DOCS_SCORECARDS_DIR", docs_scorecards_dir
    ), patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}), patch(
        "role.persona.Groq"
    ) as mock_groq_cls:
        mock_groq_cls.return_value.chat.completions.create.side_effect = [
            _groq_persona_response(text) for text in PERSONA_REPLIES
        ]

        client = TestClient(webapp_mod.app)

        create_resp = client.post(
            "/session", data={"scenario_id": "order-change"}, follow_redirects=False
        )
        assert create_resp.status_code == 303
        session_id = create_resp.headers["location"].rsplit("/", 1)[-1]

        with patch(
            "instructor.from_groq", return_value=_make_shared_instructor_client(session_id)
        ):
            last_resp = None
            for line in TRAINEE_LINES:
                last_resp = client.post(f"/session/{session_id}/turn", data={"line": line})
                assert last_resp.status_code == 200

            history_resp = client.get("/history")
            detail_resp = client.get(f"/history/{session_id}")

    return {
        "session_id": session_id,
        "inline_html": last_resp.text,
        "history_html": history_resp.text,
        "detail_html": detail_resp.text,
        "scorecard_path": docs_scorecards_dir / f"{session_id}.html",
    }


def test_webapp_pipeline_end_to_end(tmp_path) -> None:
    result = run_gate(tmp_path)
    session_id = result["session_id"]

    for item in RUBRIC_ITEMS:
        assert item in result["inline_html"]
    assert f"{session_id}:2" in result["inline_html"]  # evidence-linked, not just a score
    assert result["scorecard_path"].exists()  # docs/scorecards/<id>.html still gets written

    assert session_id in result["history_html"]
    for item in RUBRIC_ITEMS:
        assert item in result["detail_html"]
    assert f"{session_id}:2" in result["detail_html"]

    print()
    print(f"session_id     : {session_id}")
    print("inline report  : rubric items + evidence present")
    print("scorecard      : " + str(result["scorecard_path"]))
    print("history list   : session listed")
    print("history detail : renders the same report")
    print("GATE: PASS")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        result = run_gate(Path(td))
        print(f"session_id : {result['session_id']}")
        print("GATE: PASS")
    sys.exit(0)
