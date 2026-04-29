from banbu.agent.loop import AgentResult
from banbu.main import _feedback_outcome


def test_feedback_outcome_distinguishes_success_failure_skipped_and_agent_error() -> None:
    assert _feedback_outcome(
        AgentResult(iterations=1, executed=[{"ok": True}]),
        [{"ok": True}],
    ) == ("success", "executed 1 action(s)")

    assert _feedback_outcome(
        AgentResult(iterations=1, executed=[]),
        [{"ok": False, "error": "device offline"}],
    ) == ("failure", "one or more execute_plan calls failed")

    assert _feedback_outcome(
        AgentResult(iterations=1, executed=[]),
        [],
    ) == ("skipped", "agent returned no executable actions")

    assert _feedback_outcome(
        AgentResult(iterations=1, executed=[], error="llm timeout"),
        [{"ok": False}],
    ) == ("agent_error", "llm timeout")
