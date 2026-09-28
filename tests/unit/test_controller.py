"""Unit tests for ROLE-018 Controller — pure logic, no mocks needed."""
import pytest

from role.controller import Controller, ControllerState, MOODS


def test_initial_state_is_cooperative_turn_zero():
    state = Controller("easy").initial_state()
    assert state == ControllerState("easy", "cooperative", 0)


def test_step_escalates_on_repeat_trigger():
    controller = Controller("medium")
    state = controller.step("Can you repeat that for me?")
    assert state.mood == "frustrated"
    assert state.controller_turn == 1


def test_step_deescalates_on_cooperative_trigger():
    controller = Controller("medium")
    controller.step("Can you repeat that?")           # -> frustrated
    state = controller.step("Sorry about that, let me fix it right away.")
    assert state.mood == "cooperative"


def test_easy_difficulty_never_reaches_escalated():
    controller = Controller("easy")
    for _ in range(5):
        state = controller.step("Say that again? What was that, one more time?")
    assert state.mood == "frustrated"
    assert state.mood != "escalated"


def test_medium_and_hard_can_reach_escalated():
    for difficulty in ("medium", "hard"):
        controller = Controller(difficulty)
        controller.step("Can you repeat that?")
        state = controller.step("One more time, say that again.")
        assert state.mood == "escalated"


def test_mood_never_drops_below_cooperative():
    controller = Controller("hard")
    state = controller.step("Sorry, I understand, let me fix that right away.")
    assert state.mood == "cooperative"


def test_turn_counter_increments_each_step():
    controller = Controller("easy")
    turns = [controller.step("hello").controller_turn for _ in range(3)]
    assert turns == [1, 2, 3]


def test_neutral_text_does_not_change_mood():
    controller = Controller("easy")
    state = controller.step("How long will that take to process?")
    assert state.mood == "cooperative"


def test_invalid_difficulty_raises():
    with pytest.raises(ValueError, match="unknown difficulty"):
        Controller("legendary")


def test_render_never_claims_a_feeling():
    state = ControllerState("easy", "frustrated", 3)
    text = state.render()
    assert "Never say" in text and "I feel" in text  # prohibits the claim, doesn't make it
    assert "frustrated" in text
    assert "easy" in text


def test_to_dict_round_trips_fields():
    state = ControllerState("hard", "escalated", 7)
    assert state.to_dict() == {"difficulty": "hard", "mood": "escalated", "controller_turn": 7}


def test_moods_are_ordered_cooperative_first():
    assert MOODS[0] == "cooperative"
