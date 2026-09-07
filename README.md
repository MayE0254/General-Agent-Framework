# MultiAgent — 通用多 Agent 运行框架

一个可直接落地的通用 Agent 基建：FastAPI 服务 + LangGraph 编排 + Celery 异步任务，
内置 Agent/工具注册中心、运行审计、人工审批回路、会话记忆与运行看板。
业务场景以**可选插件**形式挂载（本仓库默认不含任何业务插件）。

## 功能特性

- **多 Agent 编排**：router / planner / executor / reviewer / knowledge 等内置角色，
  LangGraph 图式编排，支持按领域路由（domain routing）
- **LLM 多模型路由**：LiteLLM 统一接入（默认 DeepSeek），按请求选择 provider，
  内置 token 成本估算与 LLM 调用审计
- **工具协议**：注册式工具中心，内置 HTTP 请求工具（SSRF 防护、方法/大小白名单）
- **异步执行**：Celery worker 执行工作流，beat 承载定时任务，任务与工作流全程可查
- **运行审计与看板**：workflow run / agent run / tool call / LLM call 全量落库，
  内置可视化看板（执行路径、耗时、失败原因、成本）
- **人工审批回路**：关键步骤可挂起等待人工审批，审批后续跑
- **会话记忆**：会话历史注入 + 长期记忆抽取/召回/去重合并
- **业务插件化**：`app/business` 下的业务包是可选插件——存在即接入，缺失即纯基建

## 目录结构

```
app/
├── agents/          # Agent 协议、注册中心、内置角色
├── api/             # FastAPI 应用、路由、中间件、生命周期
├── business/        # 业务插件挂载点（可选，见下文「业务插件」）
├── core/            # 配置加载（TOML + ${VAR} 环境变量展开）
├── infra/           # PostgreSQL / Redis / pgvector / Milvus 资源管理
├── knowledge/       # 知识库抽象（内存实现 + pgvector/Milvus 后端）
├── llm/             # LLM 服务、多 provider、定价与成本
├── models/          # SQLAlchemy ORM 模型
├── observability/   # 结构化日志、OpenTelemetry/Langfuse 追踪
├── orchestrator/    # LangGraph 编排 runner
├── repositories/    # 数据访问层
├── schemas/         # API 请求/响应模型
├── services/        # 业务服务层
├── tasks/           # Celery app 与工作流任务
└── tools/           # 工具协议、注册中心、执行器
configs/             # application.toml 主配置
migrations/          # Alembic 数据库迁移
tests/               # pytest 单元/集成测试
docs/                # 开发过程文档
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
```

### 2. 配置

复制/编辑 `configs/application.toml`（非敏感连接参数直接改），
并在环境中提供被 `${VAR}` 引用的敏感变量：

```powershell
setx POSTGRES_PASSWORD "<你的数据库密码>"
setx REDIS_PASSWORD   "<你的Redis密码>"
setx DEEPSEEK_API_KEY "<你的DeepSeek Key>"
```

环境变量缺失会在启动时 fail-fast 报错，不会静默降级。

### 3. 数据库迁移

```bash
.venv\Scripts\alembic upgrade head
```

### 4. 启动三个进程

```bash
# API 服务（默认 127.0.0.1:8000）
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Celery worker
.venv\Scripts\celery -A app.tasks.celery_app worker -l info

# Celery beat（定时任务）
.venv\Scripts\celery -A app.tasks.celery_app beat -l info
```

没有数据库时框架以 preview 模式运行（无持久化，编排照常）；
没有 Redis 时同步执行工作流，异步接口返回明确错误。

### 5. 运行看板

浏览器打开：

- Agent 总览：`http://127.0.0.1:8000/api/v1/agents`
- 运行看板（执行路径/工具调用/LLM 成本/审批）：`http://127.0.0.1:8000/api/v1/system/dashboard`

## 业务插件

`app/business/` 是业务插件的挂载点，框架对插件做**惰性发现**（见
`app/business/__init__.py`）：扫描其子包，逐个尝试导入，失败的包仅记录
warning 并跳过——因此本仓库不含任何业务包时依然是完整可运行的纯基建。

一个业务插件只需遵守两条约定：

1. **包入口**：在业务包 `__init__.py` 中暴露

   ```python
   def register_business(*, agent_registry, tool_registry) -> None:
       """把本业务的工具与 DOMAIN Agent 注册进框架 registries."""
   ```

2. **定时任务（可选）**：在业务包 `tasks.py` 中暴露

   ```python
   def beat_entries() -> dict[str, dict]:
       """返回 Celery beat 调度条目，会被合并进框架 beat_schedule."""
   ```

   任务函数请使用 `celery.shared_task` 装饰（而非绑定具体 app 实例），
   避免与框架 `app.tasks.celery_app` 的模块级构建互相等待。

框架侧的接入点：

- API/worker 进程启动时调用 `register_business_agents()` 挂载 Agent 与工具
- Celery app 构建时通过 `business_task_modules()` 收集任务的 `include` 模块、
  `business_beat_entries()` 合并 beat 调度

业务插件通常适配一个独立仓库中的业务系统，建议保持插件包薄（工具封装 + DOMAIN Agent + bridge），
业务逻辑留在原仓库——这样插件可以被 .gitignore 排除而不影响框架迭代。

## 测试

```bash
.venv\Scripts\python -m pytest -q
```

业务相关断言使用 `pytest.importorskip("app.business.<pkg>")`，
在不含业务包的 checkout 上自动跳过。

## 技术栈

FastAPI · LangGraph · LiteLLM · Celery · SQLAlchemy/Alembic · PostgreSQL(+pgvector) ·
Redis · Milvus · structlog · OpenTelemetry/Langfuse · pytest
