from banbu.turn.model import ProactiveTrigger, Turn


def test_reactive_turn_input_returns_utterance() -> None:
    turn = Turn.from_reactive("开灯", home_id="home_1", user_id="user_1")
    assert turn.input == "开灯"


def test_reactive_turn_input_empty_utterance() -> None:
    turn = Turn.from_reactive("  x  ", home_id="home_1", user_id="user_1")
    assert turn.input == "x"


def test_proactive_turn_input_returns_structured_dict() -> None:
    trigger = ProactiveTrigger(
        scene_id="entry_auto_light_v1",
        home_id="home_1",
        facts={"door_sensor_1": {"contact": False}},
        source_event_ids=["evt_abc123"],
    )
    turn = Turn.from_proactive(trigger)
    inp = turn.input
    assert isinstance(inp, dict)
    assert inp["scene_id"] == "entry_auto_light_v1"
    assert inp["trigger_id"] == trigger.trigger_id
    assert inp["facts"] == {"door_sensor_1": {"contact": False}}
    assert inp["source_event_ids"] == ["evt_abc123"]


def test_proactive_turn_input_no_trigger_returns_empty_dict() -> None:
    turn = Turn(
        turn_id="turn_x",
        thread_type="proactive",
        conversation_id="home_1_scene_1",
        home_id="home_1",
        scene_id="scene_1",
    )
    assert turn.input == {}
