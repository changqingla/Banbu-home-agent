from banbu.state.feedback import FeedbackEntry, FeedbackStore


def test_feedback_store_returns_recent_entries_for_scene_only() -> None:
    store = FeedbackStore(max_entries_per_scene=2)

    first = FeedbackEntry(
        home_id="home_default",
        scene_id="entry_auto_light_v1",
        trigger_id="trg_1",
        outcome="success",
        summary="executed 1 action",
    )
    second = FeedbackEntry(
        home_id="home_default",
        scene_id="entry_auto_light_v1",
        trigger_id="trg_2",
        outcome="skipped",
        summary="agent returned no executable actions",
    )
    third = FeedbackEntry(
        home_id="home_default",
        scene_id="entry_auto_light_v1",
        trigger_id="trg_3",
        outcome="agent_error",
        summary="llm timeout",
    )
    other_scene = FeedbackEntry(
        home_id="home_default",
        scene_id="other_scene",
        trigger_id="trg_4",
        outcome="failure",
        summary="execute failed",
    )

    store.add(first)
    store.add(second)
    store.add(other_scene)
    store.add(third)

    assert store.recent("home_default", "entry_auto_light_v1") == [second, third]
    assert store.recent("home_default", "other_scene") == [other_scene]
    assert store.recent("missing_home", "entry_auto_light_v1") == []
