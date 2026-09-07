from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import func, select

from app.agents.registry import AgentRegistry
from app.api.dependencies import (
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
from app.tasks.celery_app import celery_app
from app.tools.registry import ToolRegistry

router = APIRouter(tags=["system"])

_URL_CREDENTIALS = re.compile(r"^(?P<scheme>[a-z0-9+.-]+://)(?P<auth>[^@/]+)@")


def _dashboard_html(api_prefix: str) -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MultiAgent Runtime Dashboard</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #0b1020;
      --panel: #151b2f;
      --panel-soft: #1c2440;
      --text: #e7ecff;
      --muted: #9fb0e1;
      --ok: #36c97c;
      --warn: #ffb84d;
      --bad: #ff6f7d;
      --line: #2d375c;
      --accent: #74a4ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", sans-serif;
      background: linear-gradient(180deg, #0b1020, #111933 35%, #0f1530);
      color: var(--text);
    }
    .wrap { max-width: 1280px; margin: 0 auto; padding: 24px; }
    h1, h2, h3 { margin: 0 0 12px; }
    p { color: var(--muted); line-height: 1.6; }
    .topbar, .panel {
      background: rgba(21, 27, 47, 0.92);
      border: 1px solid var(--line);
      border-radius: 16px;
      backdrop-filter: blur(10px);
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
    }
    .topbar { padding: 20px 22px; margin-bottom: 18px; }
    .grid { display: grid; gap: 16px; }
    .grid.cards { grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
    .grid.cols { grid-template-columns: minmax(0, 1.2fr) minmax(360px, 0.8fr); align-items: start; }
    .grid.dual { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .panel { padding: 18px; }
    .status {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 4px 10px; border-radius: 999px; font-size: 12px;
      font-weight: 600; background: var(--panel-soft);
    }
    .status.ready, .status.completed, .status.success { color: var(--ok); }
    .status.pending, .status.running, .status.started { color: var(--warn); }
    .status.failed, .status.degraded, .status.rejected, .status.error { color: var(--bad); }
    .metric { font-size: 28px; font-weight: 700; margin: 10px 0 8px; }
    .label { color: var(--muted); font-size: 13px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
    .kv { margin-top: 10px; color: var(--muted); font-size: 13px; line-height: 1.7; }
    .toolbar { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
    input, button, textarea, select {
      border-radius: 10px; border: 1px solid var(--line);
      background: #0f1630; color: var(--text); padding: 10px 12px;
    }
    input, textarea, select { min-width: 0; width: 100%; }
    button {
      cursor: pointer; background: linear-gradient(180deg, #6d9cff, #4d78d8);
      border: none; font-weight: 600;
    }
    button.secondary { background: var(--panel-soft); border: 1px solid var(--line); }
    button.warn { background: linear-gradient(180deg, #ffba55, #db8b2e); color: #1e170a; }
    button.bad { background: linear-gradient(180deg, #ff7d89, #dc5160); }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
    tr:last-child td { border-bottom: 0; }
    .mono {
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: 12px; word-break: break-all;
    }
    pre {
      margin: 0; white-space: pre-wrap; word-break: break-word;
      background: #0b1124; border: 1px solid var(--line);
      border-radius: 12px; padding: 14px; min-height: 120px;
      max-height: 420px; overflow: auto;
    }
    .hint { font-size: 13px; color: var(--muted); margin-top: 8px; }
    .toolbar.compact { margin-bottom: 0; }
    .clickable-row { cursor: pointer; }
    .clickable-row:hover { background: rgba(116, 164, 255, 0.08); }
    .mini-actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .mini-actions button:disabled { opacity: 0.45; cursor: not-allowed; }
    .timeline {
      display: grid; gap: 10px; margin-top: 10px;
    }
    .timeline-item {
      border: 1px solid var(--line); border-radius: 12px; padding: 12px;
      background: rgba(11, 17, 36, 0.9);
    }
    .timeline-item h4 { margin: 0 0 8px; font-size: 14px; }
    .timeline-meta { color: var(--muted); font-size: 12px; line-height: 1.6; }
    .banner {
      padding: 12px 14px; border-radius: 12px; border: 1px solid var(--line);
      background: rgba(116, 164, 255, 0.08); color: var(--text); min-height: 46px;
    }
    .stack { display: grid; gap: 12px; }
    .field { display: grid; gap: 6px; }
    .field label { color: var(--muted); font-size: 13px; }
    .pill {
      display: inline-flex; padding: 4px 8px; border-radius: 999px;
      background: rgba(116, 164, 255, 0.12); color: var(--muted); font-size: 12px;
      border: 1px solid var(--line);
    }
    .section-title { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
    .tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
    .tab {
      background: var(--panel-soft); border: 1px solid var(--line);
      color: var(--muted); padding: 8px 16px; border-radius: 10px;
    }
    .tab.active {
      background: linear-gradient(180deg, #6d9cff, #4d78d8);
      color: #ffffff; border-color: transparent;
    }
    .status.empty { color: var(--warn); }
    .check-row {
      border: 1px solid var(--line); border-radius: 12px;
      padding: 12px; background: rgba(11, 17, 36, 0.9);
    }
    .check-row h4 { margin: 0; font-size: 14px; }
    .check-evidence { margin-top: 8px; color: var(--muted); font-size: 12.5px; line-height: 1.7; }
    .phase-pill {
      display: inline-flex; padding: 2px 8px; border-radius: 999px;
      font-size: 11px; font-weight: 700; letter-spacing: 0.04em;
      margin-right: 6px; vertical-align: middle;
    }
    .phase-pill.p0 {
      background: rgba(116, 164, 255, 0.16); color: var(--accent);
      border: 1px solid rgba(116, 164, 255, 0.35);
    }
    .phase-pill.p1 {
      background: rgba(54, 201, 124, 0.14); color: var(--ok);
      border: 1px solid rgba(54, 201, 124, 0.32);
    }
    @media (max-width: 980px) { .grid.cols { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="topbar">
      <div class="row" style="justify-content: space-between;">
        <div>
          <h1>MultiAgent Runtime Console</h1>
          <p>可操作的运行时控制台：看资源接线、提交异步任务、追踪工作流细节，并直接处理人工审批。</p>
        </div>
        <div class="row">
          <a class="pill mono" href="/docs" target="_blank" rel="noreferrer">OpenAPI</a>
          <span class="status" id="refresh-status">等待首次刷新</span>
          <button id="refresh-button" type="button">立即刷新</button>
        </div>
      </div>
      <div class="tabs">
        <button class="tab active" data-tab="console" type="button">运行控制台</button>
        <button class="tab" data-tab="acceptance" type="button">P0 / P1 验收看板</button>
      </div>
    </section>

    <div id="tab-console">
    <section class="panel" style="margin-bottom: 16px;">
      <div class="section-title">
        <h2>审批事件</h2>
        <span class="pill mono">/runtime/reviews/events</span>
      </div>
      <div class="banner mono" id="review-event-banner">等待审批事件...</div>
    </section>

    <section class="grid cards" id="resource-cards"></section>

    <section class="grid cols" style="margin-top: 16px;">
      <div class="grid stack">
        <article class="panel">
          <div class="section-title">
            <h2>提交异步工作流</h2>
            <span class="pill mono">POST /workflows/run/async</span>
          </div>
          <div class="grid dual">
            <div class="field">
              <label for="submit-request-id">request_id</label>
              <input id="submit-request-id" placeholder="例如 runtime-console-001" />
            </div>
            <div class="field">
              <label for="submit-agent-id">initial_agent_id</label>
              <select id="submit-agent-id">
                <option value="">加载 agent 中...</option>
              </select>
            </div>
          </div>
          <div class="grid dual" style="margin-top: 12px;">
            <div class="field">
              <label for="submit-session-id">session_id</label>
              <input id="submit-session-id" placeholder="可选" />
            </div>
            <div class="field">
              <label for="submit-user-id">user_id</label>
              <input id="submit-user-id" placeholder="可选" />
            </div>
          </div>
          <div class="field" style="margin-top: 12px;">
            <label for="submit-input-text">input_text</label>
            <textarea id="submit-input-text" rows="4" placeholder="输入工作流要处理的任务"></textarea>
          </div>
          <div class="field" style="margin-top: 12px;">
            <label for="submit-structured-input">structured_input (JSON)</label>
            <textarea id="submit-structured-input" rows="4" placeholder='例如 {"requires_human": true}'>{}</textarea>
          </div>
          <div class="mini-actions" style="margin-top: 12px;">
            <button id="submit-async-button" type="button">提交异步任务</button>
            <button class="secondary" id="seed-request-id-button" type="button">生成 request_id</button>
          </div>
          <pre id="submit-output" style="margin-top: 12px;">等待提交...</pre>
        </article>

        <article class="panel">
          <div class="section-title">
            <h2>Worker 在线状态</h2>
            <span class="pill mono">GET /system/workers</span>
          </div>
          <div class="mini-actions">
            <button class="secondary" id="refresh-workers-button" type="button">检查 Worker</button>
          </div>
          <div class="timeline" id="worker-summary"></div>
          <pre id="worker-status-output" style="margin-top: 12px;">等待检查...</pre>
        </article>

        <article class="panel">
          <div class="section-title">
            <h2>最近异步提交</h2>
            <span class="pill mono">GET /workflows/submissions</span>
          </div>
          <p>这里展示的是后端任务中心里的真实台账，不再依赖浏览器本地缓存。</p>
          <table>
            <thead>
              <tr><th>Request</th><th>Task</th><th>Status</th><th>Submitted</th></tr>
            </thead>
            <tbody id="submission-body"></tbody>
          </table>
        </article>

        <article class="panel">
          <div class="section-title">
            <h2>最近工作流</h2>
            <span class="pill mono">点击行加载右侧详情</span>
          </div>
          <p>这里只显示已经被 worker 真正执行并写入审计表的记录；若还在 pending，请先看上面的“最近异步提交”。</p>
          <table>
            <thead>
              <tr><th>Request</th><th>Status</th><th>Path</th><th>Updated</th></tr>
            </thead>
            <tbody id="workflow-body"></tbody>
          </table>
        </article>

        <article class="panel">
          <div class="section-title">
            <h2>最近审批任务</h2>
            <span class="pill mono">点击行加载审批操作台</span>
          </div>
          <table>
            <thead><tr><th>Review</th><th>Status</th><th>Agent</th><th>Continuation</th><th>Created</th></tr></thead>
            <tbody id="review-body"></tbody>
          </table>
        </article>
      </div>

      <div class="grid stack">
        <article class="panel">
          <div class="section-title">
            <h2>运行详情</h2>
            <span class="pill mono">右侧联动显示</span>
          </div>
          <div class="toolbar compact">
            <input id="request-id-input" placeholder="输入 request_id，或点击左侧工作流 / 提交记录自动带入" />
            <button id="request-id-button" type="button">查询详情</button>
          </div>
          <div class="banner" id="detail-banner">等待选择 request_id...</div>
          <div class="timeline" id="detail-summary"></div>
          <div class="grid dual" style="margin-top: 12px;">
            <article class="panel" style="padding: 14px;">
              <h3>Agent Timeline</h3>
              <div class="timeline" id="agent-runs"></div>
            </article>
            <article class="panel" style="padding: 14px;">
              <h3>Session Messages</h3>
              <pre id="session-messages-output">等待加载...</pre>
            </article>
          </div>
          <div class="grid dual" style="margin-top: 12px;">
              <article class="panel" style="padding: 14px;">
                <h3>Tool Calls</h3>
                <div class="mini-actions" style="margin-bottom: 10px;">
                  <label class="mono" style="display: inline-flex; gap: 6px; align-items: center;">
                    <input id="tool-calls-failed-only" type="checkbox" />
                    仅失败
                  </label>
                  <label class="mono" style="display: inline-flex; gap: 6px; align-items: center;">
                    <input id="tool-calls-retryable-only" type="checkbox" />
                    仅可重试
                  </label>
                </div>
                <div class="timeline" id="tool-calls-output">等待加载...</div>
              </article>
            <article class="panel" style="padding: 14px;">
              <h3>LLM Calls</h3>
              <pre id="llm-calls-output">等待加载...</pre>
            </article>
          </div>
        </article>

        <article class="panel">
          <div class="section-title">
            <h2>异步任务状态</h2>
            <span class="pill mono">GET /workflows/tasks/{task_id}</span>
          </div>
          <div class="toolbar">
            <input id="task-id-input" placeholder="输入 Celery task_id，或由提交结果 / 提交记录自动带入" />
            <button id="task-id-button" type="button">查询</button>
          </div>
          <div class="mini-actions">
            <button class="secondary" id="poll-task-button" type="button">轮询直到结束</button>
          </div>
          <pre id="task-status-output">等待查询...</pre>
        </article>

        <article class="panel">
          <div class="section-title">
            <h2>审批操作台</h2>
            <span class="pill mono">approve / reject</span>
          </div>
          <div class="grid dual">
            <div class="field">
              <label for="review-id-input">review_id</label>
              <input id="review-id-input" placeholder="点击左侧审批记录自动带入" />
            </div>
            <div class="field">
              <label for="review-decided-by">decided_by</label>
              <input id="review-decided-by" placeholder="例如 admin" />
            </div>
          </div>
          <div class="field" style="margin-top: 12px;">
            <label for="review-note">note</label>
            <textarea id="review-note" rows="3" placeholder="审批备注"></textarea>
          </div>
          <div class="grid dual" style="margin-top: 12px;">
            <div class="field">
              <label for="review-continuation-agent">continuation.agent_id</label>
              <input id="review-continuation-agent" placeholder="例如 executor_agent" />
            </div>
            <div class="field">
              <label for="review-continuation-request-id">continuation.request_id</label>
              <input id="review-continuation-request-id" placeholder="可选，留空则后端生成" />
            </div>
          </div>
          <div class="field" style="margin-top: 12px;">
            <label for="review-continuation-input">continuation.input_text</label>
            <textarea id="review-continuation-input" rows="3" placeholder="批准后继续执行的输入"></textarea>
          </div>
          <div class="field" style="margin-top: 12px;">
            <label for="review-continuation-structured">continuation.structured_input (JSON)</label>
            <textarea id="review-continuation-structured" rows="3" placeholder="{}">{}</textarea>
          </div>
          <div class="status" id="review-status-pill" style="margin-top: 12px;">尚未加载审批详情</div>
          <div class="mini-actions" style="margin-top: 12px;">
            <button class="warn" id="review-approve-button" type="button">批准</button>
            <button class="bad" id="review-reject-button" type="button">驳回</button>
          </div>
          <div class="timeline" id="review-relation" style="margin-top: 12px;"></div>
          <pre id="review-action-output" style="margin-top: 12px;">等待审批操作...</pre>
        </article>
      </div>
    </section>
    </div>

    <div id="tab-acceptance" hidden>
      <section class="panel" style="margin-bottom: 16px;">
        <div class="section-title">
          <h2>验收总览</h2>
          <div class="row">
            <span class="pill mono">GET /system/acceptance</span>
            <button class="secondary" id="acceptance-refresh-button" type="button">刷新验收数据</button>
          </div>
        </div>
        <p>所有数据来自真实数据库、注册中心与配置：P0 验证底座基建，P1 验证治理与规则收敛。「empty」表示功能已就绪但库内暂无对应数据，可用下方“能力演示”一键生成证据。</p>
        <section class="grid cards" id="acceptance-stats"></section>
      </section>

      <section class="grid cols">
        <div class="grid stack">
          <article class="panel">
            <div class="section-title">
              <h2>P0 底座能力</h2>
              <span class="pill mono">基建验收</span>
            </div>
            <div class="stack" id="p0-checklist">加载中...</div>
          </article>
        </div>
        <div class="grid stack">
          <article class="panel">
            <div class="section-title">
              <h2>P1 治理与规则收敛</h2>
              <span class="pill mono">增量验收</span>
            </div>
            <div class="stack" id="p1-checklist">加载中...</div>
          </article>
          <article class="panel">
            <div class="section-title">
              <h2>能力演示</h2>
              <span class="pill mono">一键生成证据</span>
            </div>
            <div class="mini-actions">
              <button id="demo-knowledge-button" type="button">同步运行：知识检索链路</button>
              <button class="warn" id="demo-review-button" type="button">同步运行：人工审批链</button>
              <button class="secondary" id="demo-memory-button" type="button">Memory：三 scope 写入</button>
            </div>
            <p class="hint">演示全部走同步接口，不依赖 Redis / Worker。审批链演示成功后会出现“跳转审批”按钮，到运行控制台批准即可触发续跑并回写关系。</p>
            <div class="mini-actions" id="acceptance-demo-actions" style="margin-top: 10px;"></div>
            <pre id="acceptance-demo-output" style="margin-top: 12px;">等待演示...</pre>
          </article>
        </div>
      </section>
    </div>
  </div>

  <script>
    const apiPrefix = "__API_PREFIX__";
    let eventSource = null;
    let currentToolCalls = [];

    function formatStatus(value) {
      const normalized = String(value || "unknown").toLowerCase();
      return '<span class="status ' + normalized + '">' + normalized + '</span>';
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function short(value, limit = 48) {
      const text = String(value ?? "");
      return text.length > limit ? text.slice(0, limit) + "..." : text;
    }

    function toJsonBlock(value) {
      return JSON.stringify(value, null, 2);
    }

    async function fetchJson(path) {
      const response = await fetch(path, { headers: { "Accept": "application/json" } });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(path + " -> " + response.status + " " + text);
      }
      return await response.json();
    }

    function setRefreshStatus(text, kind = "") {
      const node = document.getElementById("refresh-status");
      node.className = "status " + kind;
      node.textContent = text;
    }

    function setPre(id, value) {
      document.getElementById(id).textContent = typeof value === "string" ? value : toJsonBlock(value);
    }

    function setBanner(text) {
      document.getElementById("detail-banner").textContent = text;
    }

    function maybeParseJson(raw, label) {
      const text = String(raw || "").trim();
      if (!text) {
        return {};
      }
      try {
        return JSON.parse(text);
      } catch (error) {
        throw new Error(label + " 不是合法 JSON: " + error.message);
      }
    }

    function generateRequestId(prefix = "runtime-console") {
      const stamp = new Date().toISOString().replaceAll(":", "").replaceAll(".", "").replace("T", "-").replace("Z", "");
      return `${prefix}-${stamp}`;
    }

    async function postJson(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Accept": "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const text = await response.text();
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch (_error) {}
      if (!response.ok) {
        throw new Error(path + " -> " + response.status + " " + (data ? JSON.stringify(data) : text));
      }
      return data;
    }

    async function loadAgents() {
      const select = document.getElementById("submit-agent-id");
      try {
        const agents = await fetchJson(`${apiPrefix}/agents`);
        select.innerHTML = '<option value="">请选择 agent</option>' + agents
          .map((item) => `<option value="${escapeHtml(item.agent_id)}">${escapeHtml(item.agent_id)} | ${escapeHtml(item.agent_name)}</option>`)
          .join("");
      } catch (error) {
        select.innerHTML = '<option value="">加载失败</option>';
        setPre("submit-output", String(error));
      }
    }

    function renderWorkerStatus(snapshot) {
      const workers = snapshot.workers || [];
      const summary = document.getElementById("worker-summary");
      summary.innerHTML = workers.length ? workers.map((item) => `
        <div class="timeline-item">
          <div class="section-title">
            <h4>${escapeHtml(item.worker_name)}</h4>
            ${formatStatus(item.online ? "ready" : "offline")}
          </div>
          <div class="timeline-meta">
            <div><strong>Pool:</strong> ${escapeHtml(item.pool || "-")}</div>
            <div><strong>Concurrency:</strong> ${escapeHtml(item.max_concurrency || "-")}</div>
            <div><strong>Queues:</strong> ${escapeHtml((item.queues || []).join(", ") || "-")}</div>
          </div>
        </div>
      `).join("") : '<div class="timeline-item"><div class="timeline-meta">当前没有 worker 响应 inspect ping。若任务长期 pending，优先检查 worker 进程、队列名与 Redis 配置。</div></div>';
      setPre("worker-status-output", snapshot);
    }

    function renderRecentSubmissions(items) {
      const body = document.getElementById("submission-body");
      body.innerHTML = items.length ? items.map((item) => `
        <tr class="clickable-row" data-task-id="${escapeHtml(item.task_id)}" data-request-id="${escapeHtml(item.request_id || "")}">
          <td class="mono">${escapeHtml(short(item.request_id || "-"))}</td>
          <td class="mono">${escapeHtml(short(item.task_id || "-"))}</td>
          <td>${formatStatus(item.status || "pending")}</td>
          <td>${escapeHtml(item.created_at || "-")}</td>
        </tr>
      `).join("") : '<tr><td colspan="4" class="label">暂无提交记录</td></tr>';
      body.querySelectorAll("tr[data-task-id]").forEach((row) => {
        row.addEventListener("click", async () => {
          const taskId = row.getAttribute("data-task-id") || "";
          const requestId = row.getAttribute("data-request-id") || "";
          document.getElementById("task-id-input").value = taskId;
          if (requestId) {
            document.getElementById("request-id-input").value = requestId;
            setBanner(`已选中 request_id=${requestId}，右侧可继续查询详情。`);
          }
          await lookupTaskId();
        });
      });
    }

    async function refreshWorkerStatus() {
      try {
        const snapshot = await fetchJson(`${apiPrefix}/system/workers`);
        renderWorkerStatus(snapshot);
      } catch (error) {
        setPre("worker-status-output", String(error));
      }
    }

    async function refreshRecentSubmissions() {
      try {
        const payload = await fetchJson(`${apiPrefix}/workflows/submissions?limit=10`);
        renderRecentSubmissions(payload.items || []);
      } catch (error) {
        document.getElementById("submission-body").innerHTML =
          '<tr><td colspan="4" class="label">' + escapeHtml(String(error)) + '</td></tr>';
      }
    }

    function renderResources(data) {
      const cards = [
        {
          title: "Database",
          status: data.database.status,
          metric: data.database.config.masked_url || "-",
          hint: data.database.message,
          extra: [
            ["SQLAlchemy", String(data.database.config.sqlalchemy_available)],
          ],
        },
        {
          title: "Redis / Celery",
          status: data.redis.status,
          metric: data.redis.config.task_default_queue || "-",
          hint: data.redis.message,
          extra: [
            ["Reachable", String(data.redis.config.reachable)],
            ["LatencyMs", String(data.redis.config.latency_ms ?? "-")],
            ["ProbeError", data.redis.config.probe_error || "-"],
            ["URL", data.redis.config.url || "-"],
            ["Broker", data.redis.config.broker_url || "-"],
            ["Backend", data.redis.config.result_backend || "-"],
            ["ConnectTimeout", String(data.redis.config.connect_timeout_seconds ?? "-")],
            ["SocketTimeout", String(data.redis.config.socket_timeout_seconds ?? "-")],
            ["WorkerPool", data.redis.config.worker_pool || "-"],
          ],
        },
        {
          title: "LLM",
          status: data.llm.status,
          metric: data.llm.status,
          hint: data.llm.message,
          extra: [],
        },
      ];
      document.getElementById("resource-cards").innerHTML = cards.map((card) => `
        <article class="panel">
          <div class="row" style="justify-content: space-between;">
            <h3>${escapeHtml(card.title)}</h3>
            ${formatStatus(card.status)}
          </div>
          <div class="metric mono">${escapeHtml(card.metric)}</div>
          <div class="label">${escapeHtml(card.hint || "")}</div>
          <div class="kv">
            ${card.extra.map(([k, v]) => `<div><strong>${escapeHtml(k)}:</strong> <span class="mono">${escapeHtml(v)}</span></div>`).join("")}
          </div>
        </article>
      `).join("");
    }

    function renderWorkflows(items) {
      const body = document.getElementById("workflow-body");
      body.innerHTML = items.length ? items.map((item) => `
        <tr class="clickable-row" data-request-id="${escapeHtml(item.request_id)}">
          <td class="mono">${escapeHtml(short(item.request_id))}</td>
          <td>${formatStatus(item.status)}</td>
          <td>${escapeHtml(short((item.execution_path || []).join(" -> "), 64))}</td>
          <td>${escapeHtml(item.updated_at || "-")}</td>
        </tr>
      `).join("") : '<tr><td colspan="4" class="label">暂无数据</td></tr>';
      body.querySelectorAll("tr[data-request-id]").forEach((row) => {
        row.addEventListener("click", () => {
          const requestId = row.getAttribute("data-request-id");
          document.getElementById("request-id-input").value = requestId || "";
          setBanner(`已从最近工作流加载 request_id=${requestId}`);
          loadRequestDetail();
        });
      });
    }

    function renderReviews(items) {
      const body = document.getElementById("review-body");
      body.innerHTML = items.length ? items.map((item) => `
        <tr class="clickable-row" data-review-id="${escapeHtml(item.id)}">
          <td class="mono">${escapeHtml(short(item.id))}</td>
          <td>${formatStatus(item.status)}</td>
          <td>${escapeHtml(item.agent_id || "-")}</td>
          <td class="mono">${item.continuation_request_id ? escapeHtml(short(item.continuation_request_id)) : "-"}</td>
          <td>${escapeHtml(item.created_at || "-")}</td>
        </tr>
      `).join("") : '<tr><td colspan="5" class="label">暂无审批任务</td></tr>';
      body.querySelectorAll("tr[data-review-id]").forEach((row) => {
        row.addEventListener("click", async () => {
          const reviewId = row.getAttribute("data-review-id");
          document.getElementById("review-id-input").value = reviewId || "";
          setBanner(`已选中审批任务 ${reviewId}，可在右侧执行 approve / reject。`);
          await loadReviewDetail();
        });
      });
    }

    function renderDetailSummary(detail, workflowDetail = null) {
      const taskRun = detail.task_run || {};
      const aggregates = detail.aggregates || {};
      const workflow = workflowDetail ? workflowDetail.workflow_run || {} : {};
      const metadata = taskRun.metadata_payload || workflow.metadata_payload || {};
      const sourceReviewId = metadata.source_review_id || "-";
      const cards = [
        ["request_id", taskRun.request_id],
        ["status", taskRun.status],
        ["session_id", taskRun.session_id || "-"],
        ["workflow_id", taskRun.workflow_id || "-"],
        ["initial_agent", taskRun.initial_agent_id || "-"],
        ["final_agent", taskRun.final_agent_id || "-"],
        ["agent_runs", aggregates.agent_runs ?? detail.agent_runs?.length ?? 0],
        ["tool_calls", aggregates.tool_calls ?? detail.tool_calls?.length ?? 0],
        ["llm_calls", aggregates.llm_calls ?? detail.llm_calls?.length ?? 0],
        ["workflow_path", (workflow.execution_path || []).join(" -> ") || "-"],
        ["knowledge_refs", (metadata.knowledge_refs || []).join("; ") || "-"],
        ["source_review", sourceReviewId],
      ];
      document.getElementById("detail-summary").innerHTML = cards.map(([label, value]) => `
        <div class="timeline-item">
          <h4>${escapeHtml(label)}</h4>
          <div class="timeline-meta mono">${label === "source_review" && sourceReviewId !== "-" ? `<span class="clickable-row" data-source-review-id="${escapeHtml(sourceReviewId)}">${escapeHtml(sourceReviewId)}</span>` : escapeHtml(value)}</div>
        </div>
      `).join("");
      document.querySelectorAll("#detail-summary .clickable-row[data-source-review-id]").forEach((el) => {
        el.addEventListener("click", async () => {
          const reviewId = el.getAttribute("data-source-review-id") || "";
          document.getElementById("review-id-input").value = reviewId;
          setBanner(`已从运行详情跳转到来源审批 ${reviewId}。`);
          await loadReviewDetail();
        });
      });
    }

    function renderAgentRuns(items) {
      const node = document.getElementById("agent-runs");
      node.innerHTML = items.length ? items.map((item) => `
        <div class="timeline-item">
          <div class="section-title">
            <h4>#${escapeHtml(item.step_index)} ${escapeHtml(item.agent_id)}</h4>
            ${formatStatus(item.status)}
          </div>
          <div class="timeline-meta">
            <div><strong>Summary:</strong> ${escapeHtml(item.summary || "-")}</div>
            <div><strong>Next:</strong> ${escapeHtml(item.next_agent_id || "-")}</div>
            <div><strong>Started:</strong> ${escapeHtml(item.started_at_runtime || "-")}</div>
            <div><strong>Completed:</strong> ${escapeHtml(item.completed_at_runtime || "-")}</div>
          </div>
          <pre>${escapeHtml(toJsonBlock(item.output || {}))}</pre>
        </div>
      `).join("") : '<div class="label">暂无 Agent 执行记录</div>';
    }

    function renderListAsJson(id, items, emptyText) {
      if (!items || !items.length) {
        setPre(id, emptyText);
        return;
      }
      setPre(id, items);
    }

    function renderToolCalls(items = currentToolCalls) {
      currentToolCalls = Array.isArray(items) ? items : [];
      const container = document.getElementById("tool-calls-output");
      if (!currentToolCalls.length) {
        container.innerHTML = '<div class="label">暂无 Tool 调用</div>';
        return;
      }
      const failedOnly = document.getElementById("tool-calls-failed-only").checked;
      const retryableOnly = document.getElementById("tool-calls-retryable-only").checked;
      const filtered = currentToolCalls.filter((item) => {
        if (failedOnly && item.status === "success") {
          return false;
        }
        if (retryableOnly && !item.retryable) {
          return false;
        }
        return true;
      });
      if (!filtered.length) {
        container.innerHTML = '<div class="label">当前筛选条件下没有 Tool 调用</div>';
        return;
      }
      container.innerHTML = filtered.map((item) => `
        <div class="timeline-item">
          <h4>${escapeHtml(item.tool_name || "-")}</h4>
          <div class="timeline-meta mono">
            status=${escapeHtml(item.status || "-")} |
            error_category=${escapeHtml(item.error_category || "-")} |
            retryable=${escapeHtml(String(Boolean(item.retryable)))}
          </div>
          <div class="label" style="margin: 6px 0 8px 0;">${escapeHtml(item.message || "-")}</div>
          <div class="mono" style="margin-bottom: 8px;">error_detail=${escapeHtml(item.error_detail || "-")}</div>
          <pre>${escapeHtml(toJsonBlock({
            agent_id: item.agent_id,
            arguments: item.arguments || {},
            errors: item.errors || [],
            content: item.content || {},
            metrics: item.metrics || {},
          }))}</pre>
        </div>
      `).join("");
    }

    async function loadSessionMessages(sessionId) {
      if (!sessionId) {
        setPre("session-messages-output", "当前 request 没有关联 session_id");
        return;
      }
      try {
        const payload = await fetchJson(`${apiPrefix}/runtime/sessions/${encodeURIComponent(sessionId)}/messages?limit=20`);
        setPre("session-messages-output", payload.items || []);
      } catch (error) {
        setPre("session-messages-output", String(error));
      }
    }

    async function loadRequestDetail() {
      const requestId = document.getElementById("request-id-input").value.trim();
      if (!requestId) {
        setBanner("请输入 request_id");
          currentToolCalls = [];
          document.getElementById("tool-calls-output").innerHTML = '<div class="label">请输入 request_id</div>';
        return;
      }
      setBanner(`正在加载 request_id=${requestId} 的运行详情...`);
        currentToolCalls = [];
        document.getElementById("tool-calls-output").innerHTML = '<div class="label">查询中...</div>';
      setPre("llm-calls-output", "查询中...");
      setPre("session-messages-output", "查询中...");
      document.getElementById("agent-runs").innerHTML = '<div class="label">查询中...</div>';
      try {
        const [taskDetail, workflowDetail] = await Promise.all([
          fetchJson(`${apiPrefix}/runtime/tasks/${encodeURIComponent(requestId)}`),
          fetchJson(`${apiPrefix}/runtime/workflow-runs/${encodeURIComponent(requestId)}`).catch(() => null),
        ]);
        renderDetailSummary(taskDetail, workflowDetail);
        renderAgentRuns(taskDetail.agent_runs || []);
          renderToolCalls(taskDetail.tool_calls || []);
        renderListAsJson("llm-calls-output", taskDetail.llm_calls || [], "暂无 LLM 调用");
        await loadSessionMessages(taskDetail.task_run?.session_id);
        setBanner(`已加载 request_id=${requestId} 的完整运行详情。`);
      } catch (error) {
          currentToolCalls = [];
          document.getElementById("tool-calls-output").innerHTML = `<div class="label">${escapeHtml(String(error))}</div>`;
        setPre("llm-calls-output", String(error));
        setPre("session-messages-output", String(error));
        document.getElementById("detail-summary").innerHTML = "";
        document.getElementById("agent-runs").innerHTML = '<div class="label">加载失败</div>';
        setBanner(`request_id=${requestId} 详情暂不可用。若任务仍为 pending，请先检查 Worker 状态并轮询 task_id。`);
      }
    }

    async function submitAsyncWorkflow() {
      try {
        const requestId = document.getElementById("submit-request-id").value.trim();
        const agentId = document.getElementById("submit-agent-id").value.trim();
        const inputText = document.getElementById("submit-input-text").value.trim();
        if (!requestId) {
          throw new Error("request_id 不能为空");
        }
        if (!agentId) {
          throw new Error("请选择 initial_agent_id");
        }
        if (!inputText) {
          throw new Error("input_text 不能为空");
        }
        const payload = {
          request_id: requestId,
          initial_agent_id: agentId,
          input_text: inputText,
          structured_input: maybeParseJson(
            document.getElementById("submit-structured-input").value,
            "structured_input",
          ),
        };
        const sessionId = document.getElementById("submit-session-id").value.trim();
        const userId = document.getElementById("submit-user-id").value.trim();
        if (sessionId) {
          payload.session_id = sessionId;
        }
        if (userId) {
          payload.user_id = userId;
        }
        setPre("submit-output", "提交中...");
        const result = await postJson(`${apiPrefix}/workflows/run/async`, payload);
        setPre("submit-output", result);
        if (result?.task_id) {
          document.getElementById("task-id-input").value = result.task_id;
          document.getElementById("request-id-input").value = requestId;
          setBanner(`异步任务已提交：request_id=${requestId}，现在可先看 task_id 状态；若仍 pending，请检查 Worker 在线状态。`);
          await lookupTaskId();
        }
        await refreshDashboard();
      } catch (error) {
        setPre("submit-output", String(error));
      }
    }

    async function lookupTaskId() {
      const taskId = document.getElementById("task-id-input").value.trim();
      const output = document.getElementById("task-status-output");
      if (!taskId) {
        output.textContent = "请输入 task_id";
        return;
      }
      output.textContent = "查询中...";
      try {
        const result = await fetchJson(`${apiPrefix}/workflows/tasks/${encodeURIComponent(taskId)}`);
        output.textContent = toJsonBlock(result);
        const requestId = result?.request_id || result?.result?.request_id;
        if (requestId) {
          document.getElementById("request-id-input").value = requestId;
          setBanner(`task_id=${taskId} 已返回 request_id=${requestId}，右侧已开始加载详情。`);
          await loadRequestDetail();
        } else if (result.status === "pending" || result.status === "started") {
          setBanner(`task_id=${taskId} 当前为 ${result.status}。若长时间不变，请检查左侧 Worker 在线状态与队列配置。`);
        }
      } catch (error) {
        output.textContent = String(error);
      }
    }

    async function pollTaskUntilDone() {
      const taskId = document.getElementById("task-id-input").value.trim();
      if (!taskId) {
        setPre("task-status-output", "请输入 task_id");
        return;
      }
      for (let i = 0; i < 30; i += 1) {
        await lookupTaskId();
        const raw = document.getElementById("task-status-output").textContent || "";
        if (raw.includes('"status": "success"') || raw.includes('"status": "failure"')) {
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    }

    function renderReviewRelation(result) {
      const node = document.getElementById("review-relation");
      if (!node) {
        return;
      }
      const rows = [
        ["来源审批", result.source_review_id || "-", "review", result.source_review_id],
        ["续跑 request", result.continuation_request_id || "-", "request", result.continuation_request_id],
        ["续跑状态", result.continuation_status || "-", null, null],
        ["续跑 terminal_reason", result.continuation_terminal_reason || "-", null, null],
        ["续跑时间", result.continuation_run_at || "-", null, null],
      ];
      node.innerHTML = rows.map(([label, value, linkType, linkId]) => `
        <div class="timeline-item">
          <h4>${escapeHtml(label)}</h4>
          <div class="timeline-meta mono">${linkId ? `<span class="link-like clickable-row" data-link-type="${linkType}" data-link-id="${escapeHtml(linkId)}">${escapeHtml(linkId)}</span>` : escapeHtml(value)}</div>
        </div>
      `).join("");
      node.querySelectorAll(".clickable-row[data-link-id]").forEach((el) => {
        el.addEventListener("click", async () => {
          const linkId = el.getAttribute("data-link-id") || "";
          if (el.getAttribute("data-link-type") === "review") {
            document.getElementById("review-id-input").value = linkId;
            await loadReviewDetail();
          } else {
            document.getElementById("request-id-input").value = linkId;
            setBanner(`已从审批关系跳转到 request_id=${linkId}。`);
            await loadRequestDetail();
          }
        });
      });
    }

    async function loadReviewDetail() {
      const reviewId = document.getElementById("review-id-input").value.trim();
      if (!reviewId) {
        setPre("review-action-output", "请输入 review_id");
        return;
      }
      try {
        const result = await fetchJson(`${apiPrefix}/runtime/reviews/${encodeURIComponent(reviewId)}`);
        setPre("review-action-output", result);
        renderReviewRelation(result);
        applyReviewStatus(result);
        document.getElementById("review-note").value = result.decision_note || "";
        document.getElementById("review-continuation-agent").value = result.agent_id || "";
        document.getElementById("review-continuation-input").value = result.payload?.input_text || "";
        document.getElementById("review-continuation-structured").value = toJsonBlock(result.payload?.structured_input || {});
      } catch (error) {
        setPre("review-action-output", String(error));
      }
    }

    function applyReviewStatus(result) {
      // 终态（approved/rejected）时禁用操作按钮，避免重复提交触发 409。
      const statusText = String(result.status || "-");
      const terminal = statusText === "approved" || statusText === "rejected";
      const pill = document.getElementById("review-status-pill");
      pill.className = "status " + (terminal ? "completed" : "pending");
      pill.textContent = terminal
        ? `当前状态：${statusText}（终态，无需重复操作）`
        : `当前状态：${statusText}（可审批）`;
      document.getElementById("review-approve-button").disabled = terminal;
      document.getElementById("review-reject-button").disabled = terminal;
    }

    async function actOnReview(action) {
      try {
        const reviewId = document.getElementById("review-id-input").value.trim();
        if (!reviewId) {
          throw new Error("review_id 不能为空");
        }
        const note = document.getElementById("review-note").value;
        const decidedBy = document.getElementById("review-decided-by").value.trim();
        const payload = { note };
        if (decidedBy) {
          payload.decided_by = decidedBy;
        }
        if (action === "approve") {
          const continuationAgent = document.getElementById("review-continuation-agent").value.trim();
          const continuationInput = document.getElementById("review-continuation-input").value.trim();
          const continuationStructured = maybeParseJson(
            document.getElementById("review-continuation-structured").value,
            "continuation.structured_input",
          );
          const continuationRequestId = document.getElementById("review-continuation-request-id").value.trim();
          if (continuationAgent && continuationInput) {
            payload.continuation = {
              agent_id: continuationAgent,
              input_text: continuationInput,
              structured_input: continuationStructured,
            };
            if (continuationRequestId) {
              payload.continuation.request_id = continuationRequestId;
            }
          }
        }
        setPre("review-action-output", "提交中...");
        const result = await postJson(`${apiPrefix}/runtime/reviews/${encodeURIComponent(reviewId)}/${action}`, payload);
        setPre("review-action-output", result);
        setBanner(`审批已${action === "approve" ? "批准" : "驳回"}：${reviewId}`);
        await loadReviewDetail();
        if (result?.continuation_run?.request_id) {
          document.getElementById("request-id-input").value = result.continuation_run.request_id;
          await loadRequestDetail();
        }
        await refreshDashboard();
      } catch (error) {
        const message = String(error);
        if (message.includes("409")) {
          // 状态冲突说明任务已处理，禁用按钮并提示，而不是让用户反复点。
          document.getElementById("review-approve-button").disabled = true;
          document.getElementById("review-reject-button").disabled = true;
          setPre("review-action-output", "该审批任务已是终态（409 状态冲突），无需重复操作。可重新查询查看最新状态。原始错误：" + message);
        } else {
          setPre("review-action-output", message);
        }
      }
    }

    function setupReviewEvents() {
      const banner = document.getElementById("review-event-banner");
      if (eventSource) {
        eventSource.close();
      }
      eventSource = new EventSource(`${apiPrefix}/runtime/reviews/events`);
      eventSource.onmessage = (event) => {
        banner.textContent = event.data || "收到审批事件";
      };
      eventSource.onerror = () => {
        banner.textContent = "审批 SSE 已断开，5 秒后会在页面刷新时继续拉取列表。";
      };
    }

    function switchTab(name) {
      document.querySelectorAll(".tab").forEach((tab) => {
        tab.classList.toggle("active", tab.getAttribute("data-tab") === name);
      });
      document.getElementById("tab-console").hidden = name !== "console";
      document.getElementById("tab-acceptance").hidden = name !== "acceptance";
      if (name === "acceptance") {
        loadAcceptance();
      }
    }

    function formatObj(value) {
      const entries = Object.entries(value || {});
      return entries.length ? entries.map(([k, v]) => `${k}=${v}`).join(", ") : "-";
    }

    function renderAcceptanceChecklist(items) {
      return items.map((item) => `
        <div class="check-row">
          <div class="section-title">
            <h4><span class="phase-pill ${escapeHtml(item.phase.toLowerCase())}">${escapeHtml(item.phase)}</span>${escapeHtml(item.title)}</h4>
            ${formatStatus(item.status)}
          </div>
          <div class="check-evidence">
            ${(item.evidence || []).map((line) => `<div class="mono">${escapeHtml(line)}</div>`).join("")}
            ${item.api ? `<div class="hint mono">API: ${escapeHtml(item.api)}</div>` : ""}
          </div>
        </div>
      `).join("");
    }

    async function loadAcceptance() {
      const p0Node = document.getElementById("p0-checklist");
      const p1Node = document.getElementById("p1-checklist");
      try {
        const data = await fetchJson(`${apiPrefix}/system/acceptance`);
        const stats = data.stats || {};
        const cards = [
          ["工作流", stats.workflow_runs, formatObj(stats.workflow_runs_by_status)],
          ["Agent 执行", stats.agent_runs, `task_runs=${stats.task_runs}`],
          ["工具调用", stats.tool_calls, formatObj(stats.tool_calls_by_status)],
          ["LLM 调用", stats.llm_calls, `${data.runtime?.llm_provider || "-"} / ${data.runtime?.llm_model || "-"}`],
          ["审批任务", stats.reviews, `续跑 ${stats.reviews_with_continuation || 0} / 链式 ${stats.chained_reviews || 0}`],
          ["记忆条目", stats.memories, formatObj(stats.memories_by_scope)],
        ];
        document.getElementById("acceptance-stats").innerHTML = cards.map(([title, value, hint]) => `
          <article class="panel">
            <div class="label">${escapeHtml(title)}</div>
            <div class="metric">${escapeHtml(String(value ?? "-"))}</div>
            <div class="kv mono">${escapeHtml(hint || "")}</div>
          </article>
        `).join("");
        p0Node.innerHTML = renderAcceptanceChecklist(data.p0 || []);
        p1Node.innerHTML = renderAcceptanceChecklist(data.p1 || []);
      } catch (error) {
        p0Node.innerHTML = `<div class="timeline-item"><div class="label">${escapeHtml(String(error))}</div></div>`;
        p1Node.innerHTML = "";
      }
    }

    async function runSyncDemo(payload) {
      const output = document.getElementById("acceptance-demo-output");
      output.textContent = "同步运行中...";
      const result = await postJson(`${apiPrefix}/workflows/run`, payload);
      let knowledgeCalls = [];
      try {
        const calls = await fetchJson(
          `${apiPrefix}/runtime/tool-calls?request_id=${encodeURIComponent(payload.request_id)}&limit=20`
        );
        knowledgeCalls = (calls.items || []).filter((item) => item.tool_name === "knowledge_search");
      } catch (_error) {}
      const summary = {
        request_id: result.request_id,
        status: result.status,
        terminal_reason: result.terminal_reason,
        execution_path: (result.execution_path || []).join(" -> "),
        review_id: result.review_id || "-",
        knowledge: knowledgeCalls.map((item) => ({
          backend: item.content?.backend || "-",
          hits: item.content?.hits ?? 0,
          results: (item.content?.results || []).map((hit) => hit.id),
          error_category: item.error_category || "-",
        })),
      };
      output.textContent = "=== 摘要 ===\\n" + toJsonBlock(summary) + "\\n\\n=== 完整响应 ===\\n" + toJsonBlock(result);
      document.getElementById("request-id-input").value = payload.request_id;
      return result;
    }

    async function demoKnowledgeRun() {
      const output = document.getElementById("acceptance-demo-output");
      try {
        await runSyncDemo({
          request_id: generateRequestId("acceptance-knowledge"),
          initial_agent_id: "planner_agent",
          input_text: "检索平台知识库并总结平台能力",
          structured_input: { planner_tool_query: "multi-agent platform" },
        });
        await loadAcceptance();
      } catch (error) {
        output.textContent = String(error);
      }
    }

    async function demoReviewChain() {
      const output = document.getElementById("acceptance-demo-output");
      const actions = document.getElementById("acceptance-demo-actions");
      actions.innerHTML = "";
      try {
        const result = await runSyncDemo({
          request_id: generateRequestId("acceptance-review"),
          initial_agent_id: "planner_agent",
          input_text: "演示人工审批链路：运行结束后等待人工决策",
          structured_input: { requires_human: true },
        });
        await loadAcceptance();
        if (result.review_id) {
          const btn = document.createElement("button");
          btn.className = "warn";
          btn.type = "button";
          btn.textContent = `跳转审批 ${short(result.review_id, 18)}`;
          btn.addEventListener("click", async () => {
            switchTab("console");
            document.getElementById("review-id-input").value = result.review_id || "";
            await loadReviewDetail();
          });
          actions.appendChild(btn);
        } else {
          // 模板字面量允许真实换行，避免 Python 字符串转义歧义。
          output.textContent += `
（本次运行未产生 review_id：请确认 structured_input.requires_human=true 且链路包含 reviewer_agent。）`;
        }
      } catch (error) {
        output.textContent = String(error);
      }
    }

    async function demoMemoryScopes() {
      const output = document.getElementById("acceptance-demo-output");
      output.textContent = "写入三条 scope 记忆...";
      try {
        const entries = [
          {
            memory_type: "preference",
            content: "验收演示：用户偏好简洁的中文回复",
            scope: "user",
            scope_key: "acceptance-user",
            source: "manual",
            tags: ["acceptance-demo"],
          },
          {
            memory_type: "fact",
            content: "验收演示：项目当前处于 P1 收口阶段",
            scope: "project",
            scope_key: "acceptance-project",
            source: "manual",
            tags: ["acceptance-demo"],
          },
          {
            memory_type: "fact",
            content: "验收演示：全局公告——底层基建已具备审计与治理能力",
            scope: "global",
            // API 层 scope_key 必填；服务端会把 global scope 重写为固定 key。
            scope_key: "global",
            source: "manual",
            tags: ["acceptance-demo"],
          },
        ];
        const created = [];
        for (const entry of entries) {
          created.push(await postJson(`${apiPrefix}/runtime/memories`, entry));
        }
        const globalList = await fetchJson(`${apiPrefix}/runtime/memories?scope=global&limit=5`);
        output.textContent = toJsonBlock({
          created: created.map((item) => ({
            id: item.id,
            scope: item.scope,
            scope_key: item.scope_key,
            source: item.source,
            content: item.content,
          })),
          global_scope_key_fixed: created[2]?.scope_key,
          global_recall_total: globalList.pagination?.total,
          recall_priority: "user -> project -> global（global 兜底参与召回）",
        });
        await loadAcceptance();
      } catch (error) {
        output.textContent = String(error);
      }
    }

    async function refreshDashboard() {
      try {
        setRefreshStatus("刷新中...", "running");
        const [resources, workflows, reviews] = await Promise.all([
          fetchJson(`${apiPrefix}/system/resources`),
          fetchJson(`${apiPrefix}/runtime/workflow-runs?limit=10`),
          fetchJson(`${apiPrefix}/runtime/reviews?limit=10`),
        ]);
        renderResources(resources);
        renderWorkflows(workflows.items || []);
        renderReviews(reviews.items || []);
        await refreshWorkerStatus();
        await refreshRecentSubmissions();
        setRefreshStatus("已刷新", "success");
      } catch (error) {
        setRefreshStatus("刷新失败", "failed");
        document.getElementById("resource-cards").innerHTML = `
          <article class="panel">
            <h3>加载失败</h3>
            <pre>${escapeHtml(String(error))}</pre>
          </article>
        `;
      }
    }

    document.getElementById("refresh-button").addEventListener("click", refreshDashboard);
    document.getElementById("refresh-workers-button").addEventListener("click", refreshWorkerStatus);
    document.getElementById("submit-async-button").addEventListener("click", submitAsyncWorkflow);
    document.getElementById("seed-request-id-button").addEventListener("click", () => {
      document.getElementById("submit-request-id").value = generateRequestId();
    });
    document.getElementById("task-id-button").addEventListener("click", lookupTaskId);
    document.getElementById("poll-task-button").addEventListener("click", pollTaskUntilDone);
    document.getElementById("request-id-button").addEventListener("click", loadRequestDetail);
    document.getElementById("tool-calls-failed-only").addEventListener("change", () => renderToolCalls());
    document.getElementById("tool-calls-retryable-only").addEventListener("change", () => renderToolCalls());
    document.getElementById("review-approve-button").addEventListener("click", () => actOnReview("approve"));
    document.getElementById("review-reject-button").addEventListener("click", () => actOnReview("reject"));
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => switchTab(tab.getAttribute("data-tab")));
    });
    document.getElementById("acceptance-refresh-button").addEventListener("click", loadAcceptance);
    document.getElementById("demo-knowledge-button").addEventListener("click", demoKnowledgeRun);
    document.getElementById("demo-review-button").addEventListener("click", demoReviewChain);
    document.getElementById("demo-memory-button").addEventListener("click", demoMemoryScopes);
    refreshDashboard();
    loadAgents();
    setupReviewEvents();
    setInterval(refreshDashboard, 5000);
  </script>
</body>
</html>
""".replace("__API_PREFIX__", api_prefix)


def _mask_url(url: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        auth = match.group("auth")
        if ":" in auth:
            username, _password = auth.split(":", 1)
            return f"{match.group('scheme')}{username}:***@"
        return f"{match.group('scheme')}***@"

    return _URL_CREDENTIALS.sub(_replace, url or "")


def _normalize_worker_snapshot() -> dict[str, object]:
    try:
        inspector = celery_app.control.inspect(timeout=3.0)
        ping = inspector.ping() or {}
        stats = inspector.stats() or {}
        active_queues = inspector.active_queues() or {}
    except Exception as exc:  # pragma: no cover - defensive against broker issues
        return {
            "status": "error",
            "message": f"Worker inspection failed: {exc}",
            "workers": [],
            "troubleshooting": [
                "确认 Celery worker 进程正在运行",
                "确认 worker 使用的 broker_url 与 API 配置一致",
                "确认 Redis 端口、ACL 用户和密码可从当前主机访问",
            ],
        }

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

    if workers:
        message = f"Detected {len(workers)} online worker(s)."
        status = "ready"
    else:
        message = "No Celery workers responded to inspect ping."
        status = "offline"

    return {
        "status": status,
        "message": message,
        "workers": workers,
        "troubleshooting": [
            "若任务长时间 pending，先看这里是否至少有 1 个 online worker",
            "若 worker 在线但队列不含 multiagent，检查 task_default_queue",
            "若 worker 在线但任务不消费，检查 Redis broker_db/result_db 是否和 API 一致",
        ],
    }


@router.get("/system/resources", response_class=JSONResponse)
def get_system_resources(
    settings: Settings = Depends(get_app_settings),
    resources: AppResources = Depends(get_app_resources),
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
def get_system_workers() -> JSONResponse:
    return JSONResponse(
        content=_normalize_worker_snapshot(),
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


@router.get("/system/dashboard", response_class=HTMLResponse)
def get_system_dashboard(
    settings: Settings = Depends(get_app_settings),
) -> HTMLResponse:
    return HTMLResponse(_dashboard_html(settings.api.prefix))
