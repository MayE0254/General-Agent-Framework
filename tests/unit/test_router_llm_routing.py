import asyncio
import pytest

from app.agents.base.schemas import AgentContext
from app.agents.system.builtin_agents import RouterAgent
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
        "request_id": "req-router-llm",
        "input_text": "summarize the retrieved documents",
        "structured_input": {"requested_agent": "executor_agent"},
    }
    base.update(overrides)
    return AgentContext(**base)


def test_router_uses_rule_when_no_llm_service() -> None:
    agent = RouterAgent()
    result = asyncio.run(agent.run(_context()))

    assert result.next_agent_id == "executor_agent"
    assert result.output["routing_source"] == "rule"
    assert result.output["knowledge_hits"] == 0


def test_router_uses_llm_when_configured() -> None:
    agent = RouterAgent()
    result = asyncio.run(
        agent.run(
            _context(
                llm_service=_fake_llm_service(
                    '{"next_agent": "knowledge_agent"}'
                )
            )
        )
    )

    assert result.next_agent_id == "knowledge_agent"
    assert result.output["routing_source"] == "llm"
    assert result.output["routing_model"] == "deepseek/deepseek-chat"


def test_router_falls_back_when_llm_output_is_invalid() -> None:
    agent = RouterAgent()
    result = asyncio.run(
        agent.run(
            _context(
                llm_service=_fake_llm_service("not valid json at all"),
            )
        )
    )

    assert result.next_agent_id == "executor_agent"
    assert result.output["routing_source"] == "rule"


def test_router_falls_back_when_llm_selects_unknown_agent() -> None:
    agent = RouterAgent()
    result = asyncio.run(
        agent.run(
            _context(
                llm_service=_fake_llm_service(
                    '{"next_agent": "not_a_real_agent"}'
                )
            )
        )
    )

    assert result.next_agent_id == "executor_agent"
    assert result.output["routing_source"] == "rule"


def test_router_llm_source_carries_knowledge_hits() -> None:
    agent = RouterAgent()
    context = _context(
        llm_service=_fake_llm_service('{"next_agent": "knowledge_agent"}'),
        shared_state={
            "tool_results": {
                "planner_agent": [
                    {
                        "tool_name": "knowledge_search",
                        "content": {"hits": 2, "query": "enterprise platform"},
                    }
                ]
            }
        },
    )
    result = asyncio.run(agent.run(context))

    assert result.output["knowledge_hits"] == 2
    assert result.output["routing_source"] == "llm"
