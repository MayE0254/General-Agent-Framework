from app.orchestrator import OrchestratorState
from app.orchestrator.shared_state import (
    append_tool_result,
    record_agent_output,
    set_last_tool_results,
    set_review_context,
)


def test_orchestrator_state_normalizes_shared_state_protocol() -> None:
    state = OrchestratorState(
        request_id="req-shared-1",
        initial_agent_id="planner_agent",
        shared_state={"custom": {"x": 1}, "agent_outputs": []},
    )

    assert state.shared_state["custom"] == {"x": 1}
    assert state.shared_state["agent_outputs"] == {}
    assert state.shared_state["tool_results"] == {}
    assert state.shared_state["last_tool_results"] == []
    assert state.shared_state["review_context"] == {}


def test_shared_state_helpers_update_standard_sections() -> None:
    state = OrchestratorState(
        request_id="req-shared-2",
        initial_agent_id="planner_agent",
    )

    record_agent_output(
        state.shared_state,
        agent_id="planner_agent",
        output={"plan_steps": ["a"]},
    )
    append_tool_result(
        state.shared_state,
        agent_id="planner_agent",
        result_payload={"tool_name": "knowledge_search", "status": "success"},
    )
    set_last_tool_results(
        state.shared_state,
        [{"tool_name": "knowledge_search", "status": "success"}],
    )
    set_review_context(
        state.shared_state,
        {"source_review_id": "review-1", "approved_by": "tester"},
    )

    assert state.shared_state["agent_outputs"]["planner_agent"] == {
        "plan_steps": ["a"]
    }
    assert state.shared_state["tool_results"]["planner_agent"][0]["tool_name"] == (
        "knowledge_search"
    )
    assert state.shared_state["last_tool_results"][0]["status"] == "success"
    assert state.shared_state["review_context"]["source_review_id"] == "review-1"
