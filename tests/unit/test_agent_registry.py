from app.agents.base.schemas import AgentKind
from app.agents.registry import AgentRegistry, create_default_registry
from app.agents.system import PlannerAgent
from app.tools import create_default_tool_registry


def test_registry_registers_unique_agent() -> None:
    registry = AgentRegistry()

    registry.register(PlannerAgent)

    assert registry.exists("planner_agent")
    assert registry.get_metadata("planner_agent").agent_name == "Planner Agent"


def test_default_registry_contains_core_system_agents() -> None:
    registry = create_default_registry()

    metadata_items = registry.list_metadata(agent_kind=AgentKind.SYSTEM)

    assert len(metadata_items) == 5
    assert metadata_items[0].agent_id == "executor_agent"
    assert metadata_items[-1].agent_id == "router_agent"


def test_registry_creates_agent_instance() -> None:
    registry = create_default_registry()

    agent = registry.create("knowledge_agent")

    assert agent.metadata.agent_id == "knowledge_agent"
    assert agent.can_use_tool("milvus_search") is True


def test_default_tool_registry_contains_core_system_tools() -> None:
    registry = create_default_tool_registry()

    assert registry.registered_tool_names() == [
        "document_parser",
        "http_request",
        "knowledge_search",
        "milvus_search",
        "pgvector_search",
    ]
