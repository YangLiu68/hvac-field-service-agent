from app.agent.planner import choose_next_action, new_state, update_from_text
from types import SimpleNamespace

from app.main import (
    _is_greeting_message,
    _is_meta_message,
    _looks_like_field_observation,
    _message_channel,
    _is_valid_pending_observation,
    _is_low_information_message,
    _casual_reply,
)


def test_no_cooling_graph_starts_with_safe_thermostat_check():
    state = new_state("no_cooling")
    update_from_text(state, "The AC is blowing warm air.")
    action = choose_next_action(state, "no_cooling")
    assert action["measurement_type"] == "thermostat_call"
    assert action["score"] > 0.9


def test_graph_advances_one_observation_at_a_time():
    state = new_state("no_cooling")
    update_from_text(state, "The thermostat is in cool mode and calling for cooling.")
    action = choose_next_action(state, "no_cooling")
    assert action["measurement_type"] == "indoor_blower"
    update_from_text(state, "The indoor blower is running.")
    assert choose_next_action(state, "no_cooling")["measurement_type"] == "outdoor_unit_operation"


def test_chat_text_cannot_be_recorded_as_evaporator_observation():
    pending = SimpleNamespace(measurement_type="evaporator_ice", unit=None)
    assert _is_meta_message("do not have any questions") is True
    assert _is_valid_pending_observation("do not have any questions", pending) is False
    assert _is_valid_pending_observation("No ice on the coil", pending) is True


def test_message_router_separates_conversation_from_field_evidence():
    assert _is_greeting_message("Good morning") is True
    assert _message_channel("What is HVAC?") == "chat"
    assert _message_channel("What can you do?") == "chat"
    assert _message_channel("The filter is clean and airflow is not restricted.") == "guided"
    assert _message_channel("Supply air is 78F and return air is 80F.") == "guided"


def test_meta_message_does_not_look_like_field_evidence():
    assert _looks_like_field_observation("do not have any questions") is False
    assert _message_channel("?") == "chat"
    assert _is_low_information_message("nothing") is True
    assert _casual_reply("what is h v a c") is not None
    assert _message_channel("shut it") == "chat"
