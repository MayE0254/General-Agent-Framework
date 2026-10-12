from __future__ import annotations

import re
import time
from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy import func, select

from app.agents.registry import AgentRegistry
from app.api.dependencies import (
    AuthenticatedUser,
    admin_required,
    get_agent_registry,
    get_app_resources,
    get_app_settings,
    get_database_session_factory,
    get_tool_registry,
)
from app.core import Settings
from app.infra import AppResources, probe_redis_connection
from app.models import (
    AgentRunLog,
    HumanReviewTask,
    LLMCallLog,
    MemoryEntryRecord,
    TaskRunRecord,
    ToolCallLog,
    WorkflowRunRecord,
)
from app.services.worker_supervisor import (
    WorkerWatchdog,
    spawn_detached_worker,
    summarize_worker_health,
)
from app.tasks.celery_app import celery_app
from app.tools.registry import ToolRegistry

router = APIRouter(tags=["system"])

_URL_CREDENTIALS = re.compile(r"^(?P<scheme>[a-z0-9+.-]+://)(?P<auth>[^@/]+)@")




def _mask_url(url: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        auth = match.group("auth")
        if ":" in auth:
            username, _password = auth.split(":", 1)
            return f"{match.group('scheme')}{username}:***@"
        return f"{match.group('scheme')}***@"

    return _URL_CREDENTIALS.sub(_replace, url or "")


def _normalize_worker_snapshot(
    watchdog: WorkerWatchdog | None = None,
) -> dict[str, object]:
    """Combine Celery's worker inspect with the independent liveness view.

    ``inspect`` cannot see a wedged worker: a worker stuck inside a task may
    stop answering entirely, which is indistinguishable from "no worker
    configured". The heartbeat registry and the broker backlog are therefore
    always reported alongside it, and they drive the final ``status``.
    """
    health = summarize_worker_health()
    stale_ids = set(health.stale_worker_ids)

    try:
        inspector = celery_app.control.inspect(timeout=3.0)
        ping = inspector.ping() or {}
        stats = inspector.stats() or {}
        active_queues = inspector.active_queues() or {}
        inspect_error = ""
    except Exception as exc:  # pragma: no cover - defensive against broker issues
        ping, stats, active_queues = {}, {}, {}
        inspect_error = str(exc)

    worker_names = sorted(set(ping) | set(stats) | set(active_queues))
    workers: list[dict[str, object]] = []
    for worker_name in worker_names:
        worker_stats = stats.get(worker_name, {}) or {}
        pool_info = worker_stats.get("pool", {}) or {}
        queue_info = active_queues.get(worker_name, []) or []
        workers.append(
            {
                "worker_name": worker_name,
                "online": worker_name in ping or worker_name in stats,
                "pool": pool_info.get("implementation", ""),
                "max_concurrency": pool_info.get(
                    "max-concurrency",
                    pool_info.get("max_concurrency", ""),
                ),
                "queues": [item.get("name", "") for item in queue_info],
                "ping": ping.get(worker_name, {}),
            }
        )

    if health.stall_detected:
        status = "stalled"
        message = (
            "队列有积压或未确认消息，但没有活跃 worker 心跳：任务不会被消费。"
        )
    elif workers:
        status = "ready"
        message = f"Detected {len(workers)} online worker(s)."
    elif health.live_worker_ids:
        status = "ready"
        message = "inspect 未应答，但 worker 心跳正常（可能正在执行长任务）。"
    else:
        status = "offline"
        message = "No Celery workers responded to inspect ping."

    troubleshooting = [
        "若任务长时间 pending，先看这里是否至少有 1 个 online worker",
        "若 worker 在线但队列不含 multiagent，检查 task_default_queue",
        "若 worker 在线但任务不消费，检查 Redis broker_db/result_db 是否和 API 一致",
        "status=stalled 表示队列积压且无心跳；看门狗会自动补 worker，也可 POST /system/workers/recover",
        "broker.unacked>0 且 oldest_unacked_age_seconds 超过 visibility_timeout，说明消息被取走后未确认",
    ]
    if inspect_error:
        troubleshooting.insert(0, f"inspect 调用失败：{inspect_error}")

    return {
        "status": status,
        "message": message,
        "workers": workers,
        "inspect_error": inspect_error,
        "live_worker_ids": health.live_worker_ids,
        "stale_worker_ids": health.stale_worker_ids,
        "heartbeats": [
            {
                **item.model_dump(mode="json"),
                "age_seconds": round(item.age_seconds(), 1),
                "stale": item.worker_id in stale_ids,
            }
            for item in health.heartbeats
        ],
        "broker": health.backlog.model_dump(mode="json"),
        "stall_detected": health.stall_detected,
        "watchdog": watchdog.state.model_dump(mode="json") if watchdog else None,
        "troubleshooting": troubleshooting,
    }


@router.get("/system/resources", response_class=JSONResponse)
def get_system_resources(
    settings: Settings = Depends(get_app_settings),
    resources: AppResources = Depends(get_app_resources),
    _: AuthenticatedUser = Depends(admin_required),
) -> JSONResponse:
    redis_probe = probe_redis_connection(settings)
    redis_status = "ready" if redis_probe["reachable"] else "degraded"
    redis_message = (
        f"Redis probe succeeded in {redis_probe['latency_ms']} ms."
        if redis_probe["reachable"]
        else (
            "Redis probe failed: "
            f"{redis_probe['error_type']}: {redis_probe['error_message']}"
        )
    )
    payload = {
        "database": {
            "status": resources.database.status.value,
            "message": resources.database.message,
            "config": {
                "masked_url": resources.database.config.get("masked_url", ""),
                "sqlalchemy_available": resources.database.config.get(
                    "sqlalchemy_available", False
                ),
            },
        },
        "redis": {
            "status": redis_status,
            "message": redis_message,
            "config": {
                "url": _mask_url(resources.redis.config.get("url", "")),
                "broker_url": _mask_url(
                    resources.redis.config.get("broker_url", "")
                ),
                "result_backend": _mask_url(
                    resources.redis.config.get("result_backend", "")
                ),
                "task_default_queue": resources.redis.config.get(
                    "task_default_queue", ""
                ),
                "worker_pool": resources.redis.config.get("worker_pool", ""),
                "connect_timeout_seconds": resources.redis.config.get(
                    "connect_timeout_seconds", ""
                ),
                "socket_timeout_seconds": resources.redis.config.get(
                    "socket_timeout_seconds", ""
                ),
                "reachable": redis_probe["reachable"],
                "latency_ms": redis_probe["latency_ms"],
                "probe_error": (
                    f"{redis_probe['error_type']}: {redis_probe['error_message']}"
                    if redis_probe["error_type"]
                    else ""
                ),
            },
        },
        "llm": {
            "status": resources.llm.status.value,
            "message": resources.llm.message,
        },
    }
    return JSONResponse(
        content=payload,
        media_type="application/json; charset=utf-8",
    )


@router.get("/system/workers", response_class=JSONResponse)
def get_system_workers(
    request: Request,
    _: AuthenticatedUser = Depends(admin_required),
) -> JSONResponse:
    watchdog = getattr(request.app.state, "worker_watchdog", None)
    return JSONResponse(
        content=_normalize_worker_snapshot(watchdog),
        media_type="application/json; charset=utf-8",
    )


@router.post("/system/workers/recover", response_class=JSONResponse)
def recover_system_workers(
    _: AuthenticatedUser = Depends(admin_required),
) -> JSONResponse:
    """Spawn a fresh Celery worker unless a live consumer is already healthy.

    Deliberately keyed on heartbeats rather than ``inspect ping``: a wedged
    worker is exactly the case that needs recovering, and it may keep answering
    pings while holding a message unacked forever.
    """
    health = summarize_worker_health()
    if health.live_worker_ids and not health.backlog.has_backlog:
        return JSONResponse(
            content={
                "recovered": False,
                "status": health.status,
                "message": (
                    f"Worker 心跳正常（{', '.join(health.live_worker_ids)}），无需恢复。"
                ),
                "workers": health.live_worker_ids,
                "health": health.model_dump(mode="json"),
            },
            media_type="application/json; charset=utf-8",
        )

    spawned_pid = spawn_detached_worker()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        time.sleep(3)
        confirmed = summarize_worker_health()
        if confirmed.live_worker_ids:
            return JSONResponse(
                content={
                    "recovered": True,
                    "status": "ready",
                    "message": (
                        f"已拉起新 worker 进程 (pid={spawned_pid}) "
                        "并确认其心跳上线。"
                    ),
                    "pid": spawned_pid,
                    "workers": confirmed.live_worker_ids,
                    "snapshot": _normalize_worker_snapshot(),
                },
                media_type="application/json; charset=utf-8",
            )
    return JSONResponse(
        status_code=502,
        content={
            "recovered": False,
            "status": "offline",
            "message": (
                f"已尝试拉起 worker (pid={spawned_pid})，但 30 秒内未探测到心跳，"
                "请查看 logs/worker.err.log。"
            ),
            "pid": spawned_pid,
        },
        media_type="application/json; charset=utf-8",
    )


_ACCEPTANCE_SCOPE_PRIORITY = "user -> project -> global"


def _acceptance_count(session, model, *conditions) -> int:
    stmt = select(func.count()).select_from(model)
    if conditions:
        stmt = stmt.where(*conditions)
    return int(session.scalar(stmt) or 0)


def _acceptance_group_counts(session, column) -> dict[str, int]:
    rows = session.execute(select(column, func.count()).group_by(column)).all()
    return {str(key): int(value) for key, value in rows if key is not None}


def _acceptance_feature(
    *,
    phase: str,
    feature_id: str,
    title: str,
    status: str,
    evidence: list[str],
    api: str | None = None,
) -> dict[str, object]:
    return {
        "phase": phase,
        "id": feature_id,
        "title": title,
        "status": status,
        "evidence": evidence,
        "api": api,
    }


@router.get("/system/acceptance", response_class=JSONResponse)
def get_acceptance_overview(
    settings: Settings = Depends(get_app_settings),
    resources: AppResources = Depends(get_app_resources),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    tool_registry: ToolRegistry = Depends(get_tool_registry),
    session_factory=Depends(get_database_session_factory),
    _: AuthenticatedUser = Depends(admin_required),
) -> JSONResponse:
    """Aggregate live evidence of every P0/P1 deliverable for acceptance."""
    agent_metadata = agent_registry.list_metadata()
    tool_names = tool_registry.registered_tool_names()
    knowledge_settings = settings.knowledge

    with session_factory() as session:
        task_runs_total = _acceptance_count(session, TaskRunRecord)
        task_runs_by_status = _acceptance_group_counts(
            session, TaskRunRecord.status
        )
        agent_runs_total = _acceptance_count(session, AgentRunLog)
        tool_calls_total = _acceptance_count(session, ToolCallLog)
        tool_calls_by_status = _acceptance_group_counts(
            session, ToolCallLog.status
        )
        llm_calls_total = _acceptance_count(session, LLMCallLog)
        workflow_runs_total = _acceptance_count(session, WorkflowRunRecord)
        workflow_runs_by_status = _acceptance_group_counts(
            session, WorkflowRunRecord.status
        )
        reviews_total = _acceptance_count(session, HumanReviewTask)
        reviews_by_status = _acceptance_group_counts(
            session, HumanReviewTask.status
        )
        reviews_with_continuation = _acceptance_count(
            session,
            HumanReviewTask,
            HumanReviewTask.continuation_request_id.is_not(None),
        )
        chained_reviews = _acceptance_count(
            session,
            HumanReviewTask,
            HumanReviewTask.source_review_id.is_not(None),
        )
        memories_total = _acceptance_count(session, MemoryEntryRecord)
        memories_by_scope = _acceptance_group_counts(
            session, MemoryEntryRecord.scope
        )
        memories_by_source = _acceptance_group_counts(
            session, MemoryEntryRecord.source
        )

        # error_category / retryable live in the metrics JSON payload; scan a
        # bounded window of recent tool calls instead of the whole table.
        recent_rows = session.execute(
            select(
                ToolCallLog.tool_name,
                ToolCallLog.status,
                ToolCallLog.metrics,
                ToolCallLog.content,
            )
            .order_by(ToolCallLog.created_at.desc())
            .limit(500)
        ).all()

    error_category_counter: Counter[str] = Counter()
    retryable_tool_calls = 0
    knowledge_hits = 0
    for tool_name, _status, metrics, content in recent_rows:
        metrics = metrics or {}
        category = metrics.get("error_category")
        if category:
            error_category_counter[str(category)] += 1
        if metrics.get("retryable"):
            retryable_tool_calls += 1
        if tool_name == "knowledge_search":
            knowledge_hits += len((content or {}).get("results") or [])

    knowledge_search_calls = sum(
        1 for row in recent_rows if row.tool_name == "knowledge_search"
    )

    stats: dict[str, object] = {
        "task_runs": task_runs_total,
        "task_runs_by_status": task_runs_by_status,
        "agent_runs": agent_runs_total,
        "tool_calls": tool_calls_total,
        "tool_calls_by_status": tool_calls_by_status,
        "tool_error_categories": dict(error_category_counter),
        "retryable_tool_calls": retryable_tool_calls,
        "llm_calls": llm_calls_total,
        "workflow_runs": workflow_runs_total,
        "workflow_runs_by_status": workflow_runs_by_status,
        "reviews": reviews_total,
        "reviews_by_status": reviews_by_status,
        "reviews_with_continuation": reviews_with_continuation,
        "chained_reviews": chained_reviews,
        "memories": memories_total,
        "memories_by_scope": memories_by_scope,
        "memories_by_source": memories_by_source,
        "knowledge_search_calls": knowledge_search_calls,
        "knowledge_hits": knowledge_hits,
    }

    agent_ids = [item.agent_id for item in agent_metadata]
    database_ok = resources.database.status.value == "ready"
    redis_probe = probe_redis_connection(settings)

    p0 = [
        _acceptance_feature(
            phase="P0",
            feature_id="agent-registry",
            title="Agent 注册中心",
            status="ready" if agent_ids else "empty",
            evidence=[
                f"{len(agent_ids)} 个 agent：{', '.join(agent_ids) or '-'}",
                "统一 metadata / allowed_tools / llm_profile 声明",
            ],
            api="GET /agents",
        ),
        _acceptance_feature(
            phase="P0",
            feature_id="tool-registry",
            title="工具注册中心与配置注入",
            status="ready" if tool_names else "empty",
            evidence=[
                f"{len(tool_names)} 个工具：{', '.join(tool_names) or '-'}",
                "settings 配置块按工具绑定（http / knowledge 等）",
            ],
            api="ToolRegistry.register(config=...)",
        ),
        _acceptance_feature(
            phase="P0",
            feature_id="orchestrator",
            title="双后端编排器",
            status="ready",
            evidence=[
                f"backend={settings.orchestrator.backend}, "
                f"max_graph_steps={settings.orchestrator.max_graph_steps}",
                "SimpleOrchestrator 顺序链 / LangGraphOrchestrator 图执行",
            ],
            api="POST /workflows/run",
        ),
        _acceptance_feature(
            phase="P0",
            feature_id="llm-service",
            title="LLM 服务（含降级模板）",
            status=resources.llm.status.value,
            evidence=[
                f"provider={settings.llm.provider}, model={settings.llm.model}",
                resources.llm.message,
            ],
        ),
        _acceptance_feature(
            phase="P0",
            feature_id="celery-async",
            title="Celery 异步任务管线",
            status=(
                "ready"
                if database_ok and redis_probe["reachable"]
                else "degraded"
            ),
            evidence=[
                f"database={resources.database.status.value}, "
                f"redis={'reachable' if redis_probe['reachable'] else 'unreachable'}",
                f"queue={settings.celery.task_default_queue}",
            ],
            api="POST /workflows/run/async",
        ),
        _acceptance_feature(
            phase="P0",
            feature_id="runtime-audit",
            title="Runtime 审计落库",
            status="ready" if task_runs_total else "empty",
            evidence=[
                f"task_runs={task_runs_total}, agent_runs={agent_runs_total}",
                f"tool_calls={tool_calls_total}, llm_calls={llm_calls_total}",
            ],
            api="GET /runtime/tasks/{request_id}",
        ),
        _acceptance_feature(
            phase="P0",
            feature_id="human-review",
            title="人工审批链路",
            status="ready" if reviews_total else "empty",
            evidence=[
                f"审批任务 {reviews_total} 条："
                + ", ".join(f"{k}={v}" for k, v in reviews_by_status.items()),
            ],
            api="POST /runtime/reviews/{id}/approve",
        ),
        _acceptance_feature(
            phase="P0",
            feature_id="memory-service",
            title="分层记忆服务",
            status="ready" if memories_total else "empty",
            evidence=[
                f"记忆 {memories_total} 条，scope 分布：{memories_by_scope or '-'}",
            ],
            api="GET /runtime/memories",
        ),
    ]

    p1 = [
        _acceptance_feature(
            phase="P1",
            feature_id="tool-error-semantics",
            title="工具错误语义标准化",
            status="ready" if tool_calls_total else "empty",
            evidence=[
                f"error_category 分布：{dict(error_category_counter) or '-'}",
                f"retryable 调用 {retryable_tool_calls} 次；"
                f"timeout={tool_calls_by_status.get('timeout', 0)}, "
                f"failed={tool_calls_by_status.get('failed', 0)}",
            ],
            api="GET /runtime/tool-calls?error_category=...",
        ),
        _acceptance_feature(
            phase="P1",
            feature_id="runtime-query-filters",
            title="Runtime 查询筛选",
            status="ready",
            evidence=[
                "tool-calls 支持 agent/tool_name/status/error_category/"
                "retryable 组合筛选",
                f"当前可查询审计：tool_calls={tool_calls_total}, "
                f"agent_runs={agent_runs_total}",
            ],
            api="GET /runtime/tool-calls",
        ),
        _acceptance_feature(
            phase="P1",
            feature_id="review-continuation-linkage",
            title="审批续跑联查",
            status=(
                "ready"
                if reviews_with_continuation or chained_reviews
                else "empty"
            ),
            evidence=[
                f"已回写续跑关系 {reviews_with_continuation} 条，"
                f"链式审批（source_review_id）{chained_reviews} 条",
                "运行详情 metadata.source_review_id 与控制台双向跳转",
            ],
            api="GET /runtime/reviews/{id}",
        ),
        _acceptance_feature(
            phase="P1",
            feature_id="memory-scope-rules",
            title="Memory scope / recall / merge 收敛",
            status="ready",
            evidence=[
                f"scope 枚举 user/project/global，分布：{memories_by_scope or '-'}",
                f"source 分布：{memories_by_source or '-'}",
                f"召回优先级 {_ACCEPTANCE_SCOPE_PRIORITY}；"
                "合并阈值 0.85，候选上限 200",
            ],
            api="POST /runtime/memories",
        ),
        _acceptance_feature(
            phase="P1",
            feature_id="knowledge-abstraction",
            title="Knowledge 接入抽象层",
            status=(
                "ready"
                if knowledge_settings.documents or knowledge_search_calls
                else "empty"
            ),
            evidence=[
                f"retriever={knowledge_settings.retriever}, "
                f"top_k={knowledge_settings.top_k}, "
                f"min_score={knowledge_settings.min_score}",
                f"已配置文档 {len(knowledge_settings.documents)} 篇；"
                f"知识检索调用 {knowledge_search_calls} 次，命中 {knowledge_hits} 条",
                "refs 由 _sync_knowledge_refs 回写审计",
            ],
            api="GET /runtime/tool-calls?tool_name=knowledge_search",
        ),
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "p0": p0,
        "p1": p1,
        "runtime": {
            "orchestrator_backend": settings.orchestrator.backend,
            "llm_provider": settings.llm.provider,
            "llm_model": settings.llm.model,
            "knowledge": {
                "retriever": knowledge_settings.retriever,
                "top_k": knowledge_settings.top_k,
                "min_score": knowledge_settings.min_score,
                "documents": len(knowledge_settings.documents),
            },
            "agents": agent_ids,
            "tools": list(tool_names),
        },
    }
    return JSONResponse(
        content=payload,
        media_type="application/json; charset=utf-8",
    )


@router.get("/system/dashboard", include_in_schema=False)
def get_system_dashboard() -> RedirectResponse:
    # 旧内嵌 HTML 看板已下线，前端控制台迁至 /console（Vue 3 + Element Plus 静态资源）。
    return RedirectResponse(url="/console/", status_code=307)
