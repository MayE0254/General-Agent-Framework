from __future__ import annotations

import json

from app.agents.base.agent import BaseAgent
from app.agents.base.schemas import (
    AgentContext,
    AgentExecutionStatus,
    AgentKind,
    AgentMetadata,
    AgentResult,
    LLMProfile,
    PendingToolCall,
)
from app.agents.registry.routing import domain_route_entries
from app.llm import LLMMessage, LLMRequest, LLMRole

_BASE_ROUTABLE_AGENTS = (
    "executor_agent: execute the main task",
    "knowledge_agent: consume retrieval context before executing",
    "reviewer_agent: review outputs or finish the workflow",
)


class PlannerAgent(BaseAgent):
    @classmethod
    def build_metadata(cls) -> AgentMetadata:
        return AgentMetadata(
            agent_id="planner_agent",
            agent_name="Planner Agent",
            agent_role="task_planning",
            agent_kind=AgentKind.SYSTEM,
            description="Breaks down a request into structured execution steps.",
            allowed_tools=["knowledge_search"],
            llm_profile=LLMProfile(provider="deepseek", temperature=0.1),
            tags=["core", "planner"],
        )

    async def _run(self, context: AgentContext) -> AgentResult:
        # Prefer an LLM-driven plan when a configured service is injected;
        # fall back to deterministic rules on any failure.
        llm_plan = await self._try_llm_plan(context)
        if llm_plan is not None:
            plan_steps, tool_query, llm_model = llm_plan
            tool_calls: list[PendingToolCall] = []
            if tool_query:
                tool_calls.append(
                    PendingToolCall(
                        tool_name="knowledge_search",
                        arguments={"query": tool_query},
                    )
                )
            return AgentResult(
                agent_id=self.metadata.agent_id,
                status=AgentExecutionStatus.SUCCESS,
                summary=f"LLM generated {len(plan_steps)} planning step(s).",
                output={
                    "plan_steps": plan_steps,
                    "shared_state_keys": sorted(context.shared_state.keys()),
                    "planning_source": "llm",
                    "planning_model": llm_model,
                },
                messages=["PlannerAgent generated a plan via LLM."],
                tool_calls=tool_calls,
                next_agent_id="router_agent",
            )

        summary = "Generated an initial execution plan."
        if context.input_text:
            summary = f"Generated an initial execution plan for: {context.input_text}"
        tool_calls = []
        planner_tool_query = context.structured_input.get("planner_tool_query")
        if planner_tool_query:
            tool_calls.append(
                PendingToolCall(
                    tool_name="knowledge_search",
                    arguments={"query": planner_tool_query},
                )
            )
        return AgentResult(
            agent_id=self.metadata.agent_id,
            status=AgentExecutionStatus.SUCCESS,
            summary=summary,
            output={
                "plan_steps": context.structured_input.get("plan_steps", []),
                "shared_state_keys": sorted(context.shared_state.keys()),
                "planning_source": "rule",
            },
            messages=["PlannerAgent placeholder executed successfully."],
            tool_calls=tool_calls,
            next_agent_id="router_agent",
        )

    async def _try_llm_plan(
        self,
        context: AgentContext,
    ) -> tuple[list[str], str | None, str] | None:
        llm_service = getattr(context, "llm_service", None)
        if llm_service is None or not getattr(llm_service, "is_configured", False):
            return None

        profile = context.llm_profile or self.metadata.llm_profile
        model = profile.model or llm_service.model
        system_prompt = (
            "You are a task planner. Break the user request into a concise "
            "list of execution steps. Answer with a JSON object of the form "
            '{"plan_steps": ["step1", "step2", ...], "tool_query": "..."}.\n'
            'Set "tool_query" to a knowledge-base query when retrieval is '
            'needed; use "" otherwise.'
        )
        user_content = f"Task: {context.input_text or '(empty)'}\n"
        user_content += f"Structured input: {context.structured_input}\n"
        memory = getattr(context, "memory", None) or {}
        session_history = memory.get("session_history") or []
        recalled_memories = memory.get("recalled_memories") or []
        if session_history:
            user_content += (
                "Conversation history (earlier turns in this session):\n"
                + "\n".join(
                    f"- {item.get('role', '?')}: {item.get('content', '')}"
                    for item in session_history
                )
                + "\n"
            )
        if recalled_memories:
            user_content += (
                "Known facts and preferences:\n"
                + "\n".join(
                    f"- ({item.get('memory_type', 'fact')}) "
                    f"{item.get('content', '')}"
                    for item in recalled_memories
                )
                + "\n"
            )
        request = LLMRequest(
            model=model,
            provider=profile.provider,
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
                LLMMessage(role=LLMRole.USER, content=user_content),
            ],
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            request_id=context.request_id,
            session_id=context.session_id,
            workflow_id=context.workflow_id,
            trace_id=context.trace_id,
            agent_id=self.metadata.agent_id,
        )
        try:
            response = await llm_service.complete(request)
            payload = json.loads(response.content)
            raw_steps = payload.get("plan_steps")
            plan_steps = (
                [str(step).strip() for step in raw_steps if str(step).strip()]
                if isinstance(raw_steps, list)
                else []
            )
            tool_query_raw = payload.get("tool_query")
            tool_query = str(tool_query_raw).strip() if tool_query_raw else None
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            return None
        except Exception:  # pragma: no cover - defensive external boundary
            return None

        if not plan_steps:
            return None
        return plan_steps, tool_query, response.model


class RouterAgent(BaseAgent):
    @classmethod
    def build_metadata(cls) -> AgentMetadata:
        return AgentMetadata(
            agent_id="router_agent",
            agent_name="Router Agent",
            agent_role="task_routing",
            agent_kind=AgentKind.SYSTEM,
            description="Selects the next agent based on workflow state, tool results, and intent.",
            allowed_tools=[],
            llm_profile=LLMProfile(provider="deepseek", temperature=0.0),
            tags=["core", "router"],
        )

    async def _run(self, context: AgentContext) -> AgentResult:
        requested_agent = context.structured_input.get("requested_agent", "executor_agent")

        # Consume shared tool results: if planner retrieved knowledge hits,
        # route to the knowledge agent so the retrieval context is materialized.
        tool_results = context.shared_state.get("tool_results", {})
        planner_results = tool_results.get("planner_agent", [])
        knowledge_hits = sum(
            item.get("content", {}).get("hits", 0)
            for item in planner_results
            if item.get("tool_name") == "knowledge_search"
        )

        # Prefer an LLM-driven routing decision when a configured service is
        # injected; fall back to deterministic rules on any failure.
        llm_route = await self._try_llm_route(context, knowledge_hits=knowledge_hits)
        if llm_route is not None:
            next_agent_id, summary, llm_model = llm_route
            return AgentResult(
                agent_id=self.metadata.agent_id,
                status=AgentExecutionStatus.SUCCESS,
                summary=summary,
                output={
                    "selected_agent": next_agent_id,
                    "knowledge_hits": knowledge_hits,
                    "routing_source": "llm",
                    "routing_model": llm_model,
                },
                messages=["RouterAgent routed via LLM decision."],
                next_agent_id=next_agent_id,
            )

        if knowledge_hits > 0:
            next_agent_id = "knowledge_agent"
            summary = (
                f"Routed to knowledge_agent to consume {knowledge_hits} "
                "retrieval hit(s)."
            )
        else:
            next_agent_id = requested_agent
            summary = f"Routed the task to {next_agent_id}."

        return AgentResult(
            agent_id=self.metadata.agent_id,
            status=AgentExecutionStatus.SUCCESS,
            summary=summary,
            output={
                "selected_agent": next_agent_id,
                "knowledge_hits": knowledge_hits,
                "routing_source": "rule",
            },
            messages=["RouterAgent consumed shared tool results."],
            next_agent_id=next_agent_id,
        )

    async def _try_llm_route(
        self,
        context: AgentContext,
        *,
        knowledge_hits: int,
    ) -> tuple[str, str, str] | None:
        llm_service = getattr(context, "llm_service", None)
        if llm_service is None or not getattr(llm_service, "is_configured", False):
            return None

        profile = context.llm_profile or self.metadata.llm_profile
        model = profile.model or llm_service.model
        # Base system agents plus every registered DOMAIN agent (kept in
        # sync by the runtime stack) are offered as routing candidates.
        domain_entries = domain_route_entries()
        candidates = list(_BASE_ROUTABLE_AGENTS) + [
            f"{agent_id}: {description}"
            for agent_id, description in sorted(domain_entries.items())
        ]
        system_prompt = (
            "You are a workflow router. Given the task context, choose the "
            "single best next agent. Answer with a JSON object of the form "
            '{"next_agent": "<agent_id>"}.\n'
            "Available agents:\n"
            + "\n".join(f"- {line}" for line in candidates)
        )
        user_content = f"Task: {context.input_text or '(empty)'}\n"
        user_content += (
            f"Retrieval hits available: {knowledge_hits}\n"
            f"Structured input: {context.structured_input}"
        )
        request = LLMRequest(
            model=model,
            provider=profile.provider,
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
                LLMMessage(role=LLMRole.USER, content=user_content),
            ],
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            request_id=context.request_id,
            session_id=context.session_id,
            workflow_id=context.workflow_id,
            trace_id=context.trace_id,
            agent_id=self.metadata.agent_id,
        )
        try:
            response = await llm_service.complete(request)
            payload = json.loads(response.content)
            next_agent = str(payload.get("next_agent", "")).strip()
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            return None
        except Exception:  # pragma: no cover - defensive external boundary
            return None

        if not next_agent or next_agent not in {
            "executor_agent",
            "knowledge_agent",
            "reviewer_agent",
            *domain_entries,
        }:
            return None

        summary = f"LLM routed the task to {next_agent}."
        return next_agent, summary, response.model


class KnowledgeAgent(BaseAgent):
    @classmethod
    def build_metadata(cls) -> AgentMetadata:
        return AgentMetadata(
            agent_id="knowledge_agent",
            agent_name="Knowledge Agent",
            agent_role="knowledge_retrieval",
            agent_kind=AgentKind.SYSTEM,
            description="Consumes tool results to build retrieval context for downstream agents.",
            allowed_tools=["pgvector_search", "milvus_search"],
            llm_profile=LLMProfile(provider="deepseek", temperature=0.0),
            tags=["core", "knowledge"],
        )

    async def _run(self, context: AgentContext) -> AgentResult:
        # Consume retrieval results produced earlier by planner tool calls and
        # materialize them into knowledge_refs for downstream agents.
        tool_results = context.shared_state.get("tool_results", {})
        planner_results = tool_results.get("planner_agent", [])
        search_items = [
            item
            for item in planner_results
            if item.get("tool_name") == "knowledge_search"
        ]
        total_hits = sum(
            item.get("content", {}).get("hits", 0) for item in search_items
        )

        consumed_refs: list[str] = list(context.knowledge_refs)
        for item in search_items:
            query = item.get("content", {}).get("query")
            ref = f"knowledge_search:{query}" if query else "knowledge_search"
            if ref not in consumed_refs:
                consumed_refs.append(ref)

        # Surface citation-ready hits so downstream agents and audit can
        # attribute outputs to concrete knowledge documents.
        retrieved_hits: list[dict] = []
        retrieval_backends: set[str] = set()
        for item in search_items:
            content = item.get("content", {})
            if content.get("backend"):
                retrieval_backends.add(str(content["backend"]))
            for hit in content.get("results", []) or []:
                if isinstance(hit, dict) and hit not in retrieved_hits:
                    retrieved_hits.append(hit)

        return AgentResult(
            agent_id=self.metadata.agent_id,
            status=AgentExecutionStatus.SUCCESS,
            summary=(
                f"Prepared retrieval context with {total_hits} hit(s) "
                f"from {len(search_items)} tool result(s)."
            ),
            output={
                "knowledge_refs": consumed_refs,
                "retrieved_hits": retrieved_hits,
                "retrieval_backends": (
                    sorted(retrieval_backends) if retrieval_backends
                    else ["pgvector", "milvus"]
                ),
                "consumed_tool_results": len(search_items),
                "total_hits": total_hits,
            },
            messages=["KnowledgeAgent consumed tool results."],
            next_agent_id="executor_agent",
        )


class ExecutorAgent(BaseAgent):
    @classmethod
    def build_metadata(cls) -> AgentMetadata:
        return AgentMetadata(
            agent_id="executor_agent",
            agent_name="Executor Agent",
            agent_role="task_execution",
            agent_kind=AgentKind.SYSTEM,
            description="Executes the main task with bounded tool permissions.",
            allowed_tools=["http_request", "document_parser"],
            llm_profile=LLMProfile(provider="deepseek", temperature=0.2),
            tags=["core", "executor"],
        )

    async def _run(self, context: AgentContext) -> AgentResult:
        # Consume retrieval refs materialized by the knowledge agent (exposed
        # through shared agent_outputs) and expose them in the execution output.
        agent_outputs = context.shared_state.get("agent_outputs", {})
        knowledge_output = agent_outputs.get("knowledge_agent", {})
        knowledge_refs = knowledge_output.get("knowledge_refs", [])

        # Prefer an LLM-produced answer; fall back to deterministic rules on
        # any failure so the workflow can still reach the reviewer.
        llm_execution = await self._try_llm_execute(context, knowledge_refs)
        if llm_execution is not None:
            result_text, summary, llm_model = llm_execution
            return AgentResult(
                agent_id=self.metadata.agent_id,
                status=AgentExecutionStatus.SUCCESS,
                summary=summary,
                output={
                    "execution_source": "llm",
                    "execution_model": llm_model,
                    "result": result_text,
                    "input_text": context.input_text,
                    "structured_input": context.structured_input,
                    "knowledge_refs": knowledge_refs,
                },
                messages=["ExecutorAgent executed the task via LLM."],
                next_agent_id="reviewer_agent",
            )

        return AgentResult(
            agent_id=self.metadata.agent_id,
            status=AgentExecutionStatus.SUCCESS,
            summary="Executed the current task with shared knowledge context.",
            output={
                "execution_source": "rule",
                "input_text": context.input_text,
                "structured_input": context.structured_input,
                "knowledge_refs": knowledge_refs,
            },
            messages=["ExecutorAgent consumed shared knowledge context."],
            next_agent_id="reviewer_agent",
        )

    async def _try_llm_execute(
        self,
        context: AgentContext,
        knowledge_refs: list[str],
    ) -> tuple[str, str, str] | None:
        llm_service = getattr(context, "llm_service", None)
        if llm_service is None or not getattr(llm_service, "is_configured", False):
            return None

        profile = context.llm_profile or self.metadata.llm_profile
        model = profile.model or llm_service.model
        system_prompt = (
            "You are a task executor. Produce the final answer to the given "
            "task directly, without meta commentary. Answer with a JSON "
            'object of the form {"result": "<final answer>"}.'
        )
        user_content = f"Task: {context.input_text or '(empty)'}\n"
        user_content += f"Structured input: {context.structured_input}\n"
        if knowledge_refs:
            user_content += (
                "Retrieval context: "
                f"{json.dumps(knowledge_refs, ensure_ascii=False)}\n"
            )
        user_content += "Provide only the result."
        request = LLMRequest(
            model=model,
            provider=profile.provider,
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
                LLMMessage(role=LLMRole.USER, content=user_content),
            ],
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            request_id=context.request_id,
            session_id=context.session_id,
            workflow_id=context.workflow_id,
            trace_id=context.trace_id,
            agent_id=self.metadata.agent_id,
        )
        try:
            response = await llm_service.complete(request)
            payload = json.loads(response.content)
            result_raw = payload.get("result")
            result = str(result_raw).strip() if result_raw else ""
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            return None
        except Exception:  # pragma: no cover - defensive external boundary
            return None

        if not result:
            return None
        summary = result if len(result) <= 120 else f"{result[:117]}..."
        return result, summary, response.model


class ReviewerAgent(BaseAgent):
    @classmethod
    def build_metadata(cls) -> AgentMetadata:
        return AgentMetadata(
            agent_id="reviewer_agent",
            agent_name="Reviewer Agent",
            agent_role="quality_review",
            agent_kind=AgentKind.SYSTEM,
            description="Reviews agent outputs and decides whether human review is needed.",
            allowed_tools=[],
            llm_profile=LLMProfile(provider="deepseek", temperature=0.0),
            tags=["core", "reviewer"],
        )

    async def _run(self, context: AgentContext) -> AgentResult:
        # Prefer an LLM-driven review when a configured service is injected;
        # fall back to deterministic rules on any failure.
        llm_review = await self._try_llm_review(context)
        if llm_review is not None:
            llm_needs_human, feedback, llm_model = llm_review
            # An explicit requires_human flag in the structured input is a
            # hard override: the requester asked for human review, so the
            # reviewer must not silently downgrade it to a plain success.
            needs_human = llm_needs_human or bool(
                context.structured_input.get("requires_human")
            )
            status = (
                AgentExecutionStatus.NEEDS_HUMAN_REVIEW
                if needs_human
                else AgentExecutionStatus.SUCCESS
            )
            return AgentResult(
                agent_id=self.metadata.agent_id,
                status=status,
                summary="LLM reviewed the current result set.",
                output={
                    "review_passed": not needs_human,
                    "review_source": "llm",
                    "review_model": llm_model,
                    "review_feedback": feedback,
                },
                messages=["ReviewerAgent reviewed results via LLM."],
                requires_human=needs_human,
            )

        needs_human = bool(context.structured_input.get("requires_human"))
        status = (
            AgentExecutionStatus.NEEDS_HUMAN_REVIEW
            if needs_human
            else AgentExecutionStatus.SUCCESS
        )
        return AgentResult(
            agent_id=self.metadata.agent_id,
            status=status,
            summary="Reviewed the current result set.",
            output={
                "review_passed": not needs_human,
                "review_source": "rule",
            },
            messages=["ReviewerAgent placeholder executed successfully."],
            requires_human=needs_human,
        )

    async def _try_llm_review(
        self,
        context: AgentContext,
    ) -> tuple[bool, str, str] | None:
        llm_service = getattr(context, "llm_service", None)
        if llm_service is None or not getattr(llm_service, "is_configured", False):
            return None

        profile = context.llm_profile or self.metadata.llm_profile
        model = profile.model or llm_service.model
        agent_outputs = context.shared_state.get("agent_outputs", {})
        system_prompt = (
            "You are a quality reviewer. Given the task and the execution "
            "outputs, decide whether a human review is required. Answer with "
            "a JSON object of the form "
            '{"requires_human": false, "feedback": "..."}.'
        )
        user_content = f"Task: {context.input_text or '(empty)'}\n"
        user_content += (
            "Agent outputs: "
            f"{json.dumps(agent_outputs, ensure_ascii=False)}\n"
            f"Structured input: {context.structured_input}"
        )
        request = LLMRequest(
            model=model,
            provider=profile.provider,
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
                LLMMessage(role=LLMRole.USER, content=user_content),
            ],
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            request_id=context.request_id,
            session_id=context.session_id,
            workflow_id=context.workflow_id,
            trace_id=context.trace_id,
            agent_id=self.metadata.agent_id,
        )
        try:
            response = await llm_service.complete(request)
            payload = json.loads(response.content)
            needs_human = bool(payload.get("requires_human", False))
            feedback_raw = payload.get("feedback")
            feedback = str(feedback_raw).strip() if feedback_raw else ""
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            return None
        except Exception:  # pragma: no cover - defensive external boundary
            return None

        return needs_human, feedback, response.model
