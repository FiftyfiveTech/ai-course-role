"""Unit tests for ROLE-018 controller-state plumbing and ROLE-019 pinned-facts
plumbing on PersonaAgent."""
from unittest.mock import MagicMock, patch

from role.persona import PersonaAgent, PINNED_FACTS_PLACEHOLDER, STATE_PLACEHOLDER


TEMPLATE = f"You are Alex.\n\n## Current state\n{STATE_PLACEHOLDER}\n"
FACTS_TEMPLATE = (
    f"You are Alex.\n\n## Current state\n{STATE_PLACEHOLDER}\n"
    f"\n## Pinned facts\n{PINNED_FACTS_PLACEHOLDER}\n"
)


def _make_agent(system_prompt: str) -> PersonaAgent:
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        with patch("role.persona.Groq"):
            return PersonaAgent(system_prompt)


def _groq_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = text
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


def test_placeholder_blanked_at_construction_when_never_set():
    agent = _make_agent(TEMPLATE)
    assert STATE_PLACEHOLDER not in agent.history[0]["content"]


def test_template_without_placeholder_is_untouched():
    agent = _make_agent("You are a customer named Alex.")
    assert agent.history[0]["content"] == "You are a customer named Alex."


def test_set_state_renders_block_into_system_prompt():
    agent = _make_agent(TEMPLATE)
    agent.set_state("mood: frustrated")
    assert "mood: frustrated" in agent.history[0]["content"]
    assert STATE_PLACEHOLDER not in agent.history[0]["content"]


def test_set_state_does_not_accumulate_across_calls():
    agent = _make_agent(TEMPLATE)
    agent.set_state("mood: frustrated")
    agent.set_state("mood: escalated")
    content = agent.history[0]["content"]
    assert "escalated" in content
    assert "frustrated" not in content


def test_reply_sends_updated_state_to_groq():
    agent = _make_agent(TEMPLATE)
    agent.set_state("mood: escalated")
    agent._client.chat.completions.create.return_value = _groq_response("reply")

    agent.reply("hello")

    sent_messages = agent._client.chat.completions.create.call_args.kwargs["messages"]
    assert "mood: escalated" in sent_messages[0]["content"]


def test_reply_forwards_controller_state_to_logger():
    agent = _make_agent(TEMPLATE)
    agent._client.chat.completions.create.return_value = _groq_response("reply")
    logger = MagicMock()

    agent.reply("hello", logger=logger, controller_state={"mood": "frustrated"})

    logger.log.assert_called_once()
    assert logger.log.call_args.kwargs["controller_state"] == {"mood": "frustrated"}


def test_reply_without_controller_state_forwards_none():
    agent = _make_agent("You are a customer named Alex.")
    agent._client.chat.completions.create.return_value = _groq_response("reply")
    logger = MagicMock()

    agent.reply("hello", logger=logger)

    assert logger.log.call_args.kwargs["controller_state"] is None


def test_set_facts_renders_block_into_system_prompt():
    agent = _make_agent(FACTS_TEMPLATE)
    agent.set_facts("- order number: BT-78432")
    assert "order number: BT-78432" in agent.history[0]["content"]
    assert PINNED_FACTS_PLACEHOLDER not in agent.history[0]["content"]


def test_set_facts_does_not_accumulate_across_calls():
    agent = _make_agent(FACTS_TEMPLATE)
    agent.set_facts("- price: £49")
    agent.set_facts("- price: £69")
    content = agent.history[0]["content"]
    assert "£69" in content
    assert "£49" not in content


def test_set_state_and_set_facts_do_not_clobber_each_other():
    agent = _make_agent(FACTS_TEMPLATE)
    agent.set_state("mood: frustrated")
    agent.set_facts("- order number: BT-78432")
    content = agent.history[0]["content"]
    assert "mood: frustrated" in content
    assert "order number: BT-78432" in content

    agent.set_state("mood: escalated")
    content = agent.history[0]["content"]
    assert "mood: escalated" in content
    assert "order number: BT-78432" in content  # facts survive a later set_state call
