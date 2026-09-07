import asyncio

from app.agents.base.schemas import AgentContext, AgentExecutionStatus, LLMProfile
from app.agents.system.builtin_agents import PlannerAgent, ReviewerAgent
from app.core.settings import LLMSettings
from app.llm import LLMService


def _fake_llm_service(content: str) -> LLMService:
    async def _fake_completion(kwargs: dict) -> object:
        class _Message:
            content: str = ""

        class _Choice:
            message: object = None

        class _Usage:
            prompt_tokens = 3
            completion_tokens = 2
            total_tokens = 5

        class _Raw:
            model: str = ""
            choices: list = []
            usage: object = None

        raw = _Raw()
        raw.model = kwargs["model"]
        message = _Message()
        message.content = content
        raw.choices = [_Choice()]
        raw.choices[0].message = message
        raw.usage = _Usage()
        return raw

    return LLMService(
        LLMSettings(
            api_key="sk-real",
            model="gpt-test",
            providers={
                "deepseek": {
                    "api_base": "https://api.deepseek.com",
                    "api_key": "sk-deepseek",
                    "model": "deepseek/deepseek-chat",
                }
            },
        ),
        completion_fn=_fake_completion,
    )


def _context(**overrides) -> AgentContext:
    base = {
        "request_id": "req-planner-reviewer",
        "input_text": "draft a quarterly report",
        "structured_input": {},
    }
    base.update(overrides)
    return AgentContext(**base)


# ---------- PlannerAgent ----------


def test_planner_uses_rule_when_no_llm_service() -> None:
    agent = PlannerAgent()
    result = asyncio.run(
        agent.run(
            _context(
                structured_input={
                    "plan_steps": ["gather data", "write report"],
                    "planner_tool_query": "quarterly metrics",
                }
            )
        )
    )

    assert result.output["planning_source"] == "rule"
    assert result.output["plan_steps"] == ["gather data", "write report"]
    assert [tc.tool_name for tc in result.tool_calls] == ["knowledge_search"]


def test_planner_uses_llm_when_configured() -> None:
    agent = PlannerAgent()
    result = asyncio.run(
        agent.run(
            _context(
                llm_service=_fake_llm_service(
                    '{"plan_steps": ["step 1", "step 2"], "tool_query": "metrics"}'
                )
            )
        )
    )

    assert result.output["planning_source"] == "llm"
    assert result.output["planning_model"] == "deepseek/deepseek-chat"
    assert result.output["plan_steps"] == ["step 1", "step 2"]
    assert [tc.tool_name for tc in result.tool_calls] == ["knowledge_search"]
    assert result.tool_calls[0].arguments == {"query": "metrics"}


def test_planner_llm_without_tool_query_emits_no_tool_call() -> None:
    agent = PlannerAgent()
    result = asyncio.run(
        agent.run(
            _context(
                llm_service=_fake_llm_service(
                    '{"plan_steps": ["just plan"], "tool_query": ""}'
                )
            )
        )
    )

    assert result.output["planning_source"] == "llm"
    assert result.tool_calls == []


def test_planner_falls_back_when_llm_output_is_invalid() -> None:
    agent = PlannerAgent()
    result = asyncio.run(
        agent.run(
            _context(
                llm_service=_fake_llm_service("this is not json"),
                structured_input={
                    "plan_steps": ["fallback step"],
                    "planner_tool_query": "fallback query",
                },
            )
        )
    )

    assert result.output["planning_source"] == "rule"
    assert result.output["plan_steps"] == ["fallback step"]
    assert [tc.tool_name for tc in result.tool_calls] == ["knowledge_search"]


def test_planner_falls_back_when_llm_returns_empty_steps() -> None:
    agent = PlannerAgent()
    result = asyncio.run(
        agent.run(
            _context(llm_service=_fake_llm_service('{"plan_steps": []}'))
        )
    )

    assert result.output["planning_source"] == "rule"
    assert result.next_agent_id == "router_agent"


def test_planner_request_profile_overrides_agent_model_and_provider() -> None:
    agent = PlannerAgent()
    captured: dict = {}

    async def _capture(kwargs: dict) -> object:
        captured.update(kwargs)

        class _Message:
            content: str = '{"plan_steps": ["step 1"], "tool_query": ""}'

        class _Choice:
            message: object = None

        class _Usage:
            prompt_tokens = 3
            completion_tokens = 2
            total_tokens = 5

        class _Raw:
            model: str = ""
            choices: list = []
            usage: object = None

        raw = _Raw()
        raw.model = kwargs["model"]
        raw.choices = [_Choice()]
        raw.choices[0].message = _Message()
        raw.usage = _Usage()
        return raw

    service = LLMService(
        LLMSettings(
            api_key="sk-real",
            model="default-model",
            providers={
                "deepseek": {
                    "api_base": "https://api.deepseek.com",
                    "api_key": "sk-deepseek",
                    "model": "deepseek/deepseek-chat",
                }
            },
        ),
        completion_fn=_capture,
    )
    result = asyncio.run(
        agent.run(
            _context(
                llm_service=service,
                llm_profile=LLMProfile(
                    provider="deepseek",
                    model="deepseek/deepseek-chat",
                    temperature=0.3,
                ),
            )
        )
    )

    assert result.output["planning_source"] == "llm"
    assert result.output["planning_model"] == "deepseek/deepseek-chat"
    assert captured["model"] == "deepseek/deepseek-chat"
    assert captured["api_key"] == "sk-deepseek"
    assert captured["temperature"] == 0.3


# ---------- ReviewerAgent ----------


def test_reviewer_uses_rule_when_no_llm_service() -> None:
    agent = ReviewerAgent()
    result = asyncio.run(agent.run(_context()))

    assert result.output["review_source"] == "rule"
    assert result.status == AgentExecutionStatus.SUCCESS
    assert result.requires_human is False


def test_reviewer_rule_honors_requires_human_flag() -> None:
    agent = ReviewerAgent()
    result = asyncio.run(
        agent.run(_context(structured_input={"requires_human": True}))
    )

    assert result.output["review_source"] == "rule"
    assert result.status == AgentExecutionStatus.NEEDS_HUMAN_REVIEW
    assert result.requires_human is True


def test_reviewer_uses_llm_when_configured_and_passes() -> None:
    agent = ReviewerAgent()
    result = asyncio.run(
        agent.run(
            _context(
                llm_service=_fake_llm_service(
                    '{"requires_human": false, "feedback": "looks good"}'
                ),
                shared_state={"agent_outputs": {"executor_agent": {"ok": True}}},
            )
        )
    )

    assert result.output["review_source"] == "llm"
    assert result.output["review_model"] == "deepseek/deepseek-chat"
    assert result.output["review_feedback"] == "looks good"
    assert result.status == AgentExecutionStatus.SUCCESS
    assert result.requires_human is False


def test_reviewer_llm_requests_human_review() -> None:
    agent = ReviewerAgent()
    result = asyncio.run(
        agent.run(
            _context(
                llm_service=_fake_llm_service(
                    '{"requires_human": true, "feedback": "ambiguous output"}'
                )
            )
        )
    )

    assert result.output["review_source"] == "llm"
    assert result.status == AgentExecutionStatus.NEEDS_HUMAN_REVIEW
    assert result.requires_human is True


def test_reviewer_explicit_flag_overrides_llm_passes() -> None:
    """A structured requires_human flag must not be downgraded by the LLM."""
    agent = ReviewerAgent()
    result = asyncio.run(
        agent.run(
            _context(
                llm_service=_fake_llm_service(
                    '{"requires_human": false, "feedback": "looks fine"}'
                ),
                structured_input={"requires_human": True},
            )
        )
    )

    assert result.output["review_source"] == "llm"
    assert result.status == AgentExecutionStatus.NEEDS_HUMAN_REVIEW
    assert result.requires_human is True


def test_reviewer_falls_back_when_llm_output_is_invalid() -> None:
    agent = ReviewerAgent()
    result = asyncio.run(
        agent.run(
            _context(
                llm_service=_fake_llm_service("not json"),
                structured_input={"requires_human": True},
            )
        )
    )

    assert result.output["review_source"] == "rule"
    assert result.status == AgentExecutionStatus.NEEDS_HUMAN_REVIEW


# ---------- ExecutorAgent ----------


def test_executor_uses_llm_when_configured() -> None:
    from app.agents.system.builtin_agents import ExecutorAgent

    agent = ExecutorAgent()
    result = asyncio.run(
        agent.run(
            _context(
                llm_service=_fake_llm_service('{"result": "REST and gRPC differ in X."}'),
                shared_state={
                    "agent_outputs": {"knowledge_agent": {"knowledge_refs": ["r1"]}}
                },
            )
        )
    )

    assert result.output["execution_source"] == "llm"
    assert result.output["execution_model"] == "deepseek/deepseek-chat"
    assert result.output["result"] == "REST and gRPC differ in X."
    assert result.output["knowledge_refs"] == ["r1"]
    assert result.next_agent_id == "reviewer_agent"


def test_executor_falls_back_when_llm_output_is_invalid() -> None:
    from app.agents.system.builtin_agents import ExecutorAgent

    agent = ExecutorAgent()
    result = asyncio.run(
        agent.run(
            _context(llm_service=_fake_llm_service("not json"))
        )
    )

    assert result.output["execution_source"] == "rule"
    assert result.next_agent_id == "reviewer_agent"


def test_executor_falls_back_when_llm_result_is_empty() -> None:
    from app.agents.system.builtin_agents import ExecutorAgent

    agent = ExecutorAgent()
    result = asyncio.run(
        agent.run(
            _context(llm_service=_fake_llm_service('{"result": ""}'))
        )
    )

    assert result.output["execution_source"] == "rule"
