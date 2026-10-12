// API 封装：与后端 /api/v1 约定的所有端点集中在这里，页面组件只调用语义化函数。

import { store } from "./store.js";

const apiPrefix = "/api/v1";

// 会话过期统一处理：清空登录态并跳转登录页（登录接口本身除外）。
function handleAuthExpired() {
  store.currentUser = null;
  if (!String(window.location.hash || "").startsWith("#/login")) {
    window.location.hash = "#/login";
  }
}

export async function fetchJson(path, options = {}) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    if (response.status === 401 && !options.skipAuthRedirect) handleAuthExpired();
    const text = await response.text();
    throw new Error(`${path} -> ${response.status} ${text}`);
  }
  return await response.json();
}

export async function postJson(path, payload, options = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(payload ?? {}),
  });
  if (!response.ok) {
    if (response.status === 401 && !options.skipAuthRedirect) handleAuthExpired();
    const text = await response.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { /* 保留原始文本用于报错 */ }
    throw new Error(`${path} -> ${response.status} ${data ? JSON.stringify(data) : text}`);
  }
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

export async function patchJson(path, payload, options = {}) {
  const response = await fetch(path, {
    method: "PATCH",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(payload ?? {}),
  });
  if (!response.ok) {
    if (response.status === 401 && !options.skipAuthRedirect) handleAuthExpired();
    const text = await response.text();
    throw new Error(`${path} -> ${response.status} ${text}`);
  }
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

export async function deleteJson(path, options = {}) {
  const response = await fetch(path, { method: "DELETE", headers: { Accept: "application/json" } });
  if (!response.ok) {
    if (response.status === 401 && !options.skipAuthRedirect) handleAuthExpired();
    const text = await response.text();
    throw new Error(`${path} -> ${response.status} ${text}`);
  }
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

// ---------- 资源 / 系统 ----------
export const getResources = () => fetchJson(`${apiPrefix}/system/resources`);
export const getWorkers = () => fetchJson(`${apiPrefix}/system/workers`);
export const recoverWorkers = () => postJson(`${apiPrefix}/system/workers/recover`);
export const getAcceptance = () => fetchJson(`${apiPrefix}/system/acceptance`);

// ---------- Agent / 工作流 ----------
export const getAgents = () => fetchJson(`${apiPrefix}/agents`);
export const getAutomationAgents = () => fetchJson(`${apiPrefix}/automation/agents`);
export const getSubmissions = (limit = 10) =>
  fetchJson(`${apiPrefix}/workflows/submissions?limit=${limit}`);
export const getWorkflowRuns = (limit = 10) =>
  fetchJson(`${apiPrefix}/runtime/workflow-runs?limit=${limit}`);
export const getTaskStatus = (taskId) =>
  fetchJson(`${apiPrefix}/workflows/tasks/${encodeURIComponent(taskId)}`);
export const submitAsyncWorkflow = (payload) =>
  postJson(`${apiPrefix}/workflows/run/async`, payload);
export const runSyncWorkflow = (payload) => postJson(`${apiPrefix}/workflows/run`, payload);

// ---------- 运行详情 ----------
export const getTaskDetail = (requestId) =>
  fetchJson(`${apiPrefix}/runtime/tasks/${encodeURIComponent(requestId)}`);
export const getWorkflowDetail = (requestId) =>
  fetchJson(`${apiPrefix}/runtime/workflow-runs/${encodeURIComponent(requestId)}`);
export const getSessionMessages = (sessionId, limit = 20) =>
  fetchJson(`${apiPrefix}/runtime/sessions/${encodeURIComponent(sessionId)}/messages?limit=${limit}`);
export const queryToolCalls = (requestId, limit = 20) =>
  fetchJson(`${apiPrefix}/runtime/tool-calls?request_id=${encodeURIComponent(requestId)}&limit=${limit}`);

// ---------- 人工审批 ----------
export const getReviews = (limit = 10, userId = "") => {
  const params = new URLSearchParams({ limit: String(limit) });
  if (userId) params.set("user_id", userId);
  return fetchJson(`${apiPrefix}/runtime/reviews?${params.toString()}`);
};
export const getReviewDetail = (reviewId) =>
  fetchJson(`${apiPrefix}/runtime/reviews/${encodeURIComponent(reviewId)}`);
export const decideReview = (reviewId, action, payload) =>
  postJson(`${apiPrefix}/runtime/reviews/${encodeURIComponent(reviewId)}/${action}`, payload);
export const reviewEventsUrl = () => `${apiPrefix}/runtime/reviews/events`;

// ---------- 登录 / 会话 ----------
export const apiLogin = (username, password) =>
  postJson(`${apiPrefix}/auth/login`, { username, password }, { skipAuthRedirect: true });
export const apiLogout = () => postJson(`${apiPrefix}/auth/logout`, {});
export const apiMe = () => fetchJson(`${apiPrefix}/auth/me`, { skipAuthRedirect: true });

// ---------- 成员管理（仅管理员） ----------
export const getUsers = () => fetchJson(`${apiPrefix}/users`);
export const createUser = (payload) => postJson(`${apiPrefix}/users`, payload);
export const updateUser = (userId, payload) =>
  patchJson(`${apiPrefix}/users/${encodeURIComponent(userId)}`, payload);
export const deleteUser = (userId) =>
  deleteJson(`${apiPrefix}/users/${encodeURIComponent(userId)}`);

// ---------- 记忆（验收演示） ----------
export const createMemory = (payload) => postJson(`${apiPrefix}/runtime/memories`, payload);
export const listMemories = (query) => fetchJson(`${apiPrefix}/runtime/memories?${new URLSearchParams(query)}`);

// ---------- 模型池 ----------
// 成员只读且只能看到启用项（后端强制），管理员可增删改并看到全部。
export const getModels = (enabledOnly = false) =>
  fetchJson(`${apiPrefix}/models?enabled_only=${enabledOnly ? "true" : "false"}&limit=200`);
export const createModel = (payload) => postJson(`${apiPrefix}/models`, payload);
export const updateModel = (entryId, payload) =>
  patchJson(`${apiPrefix}/models/${encodeURIComponent(entryId)}`, payload);
export const deleteModel = (entryId) =>
  deleteJson(`${apiPrefix}/models/${encodeURIComponent(entryId)}`);

// ---------- Chat 会话 ----------
export const getChatSessions = (limit = 100) =>
  fetchJson(`${apiPrefix}/chat/sessions?limit=${limit}`);
export const createChatSession = (payload) =>
  postJson(`${apiPrefix}/chat/sessions`, payload || {});
export const deleteChatSession = (sessionId) =>
  deleteJson(`${apiPrefix}/chat/sessions/${encodeURIComponent(sessionId)}`);
export const getChatMessages = (sessionId, limit = 200) =>
  fetchJson(
    `${apiPrefix}/chat/sessions/${encodeURIComponent(sessionId)}/messages?limit=${limit}`,
  );
export const chatTurnsPath = () => `${apiPrefix}/chat/turns`;

// 一轮对话走 SSE（打字机效果）。EventSource 只支持 GET，因此这里用 fetch + 流式读取，
// 逐帧解析 `data: {json}\n\n`。返回的每个事件都是后端 ChatService 产出的结构体。
export async function streamChatTurn(payload, { onEvent, signal } = {}) {
  const response = await fetch(chatTurnsPath(), {
    method: "POST",
    headers: { Accept: "text/event-stream", "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) {
    if (response.status === 401) handleAuthExpired();
    const text = await response.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { /* 保留原始文本用于报错 */ }
    const detail = data && data.detail ? data.detail : text;
    throw new Error(`${response.status} ${detail || "对话请求失败"}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const line = frame.split("\n").find((item) => item.startsWith("data:"));
      const raw = line ? line.slice(5).trim() : "";
      if (raw) {
        try {
          onEvent?.(JSON.parse(raw));
        } catch {
          // 单帧解析失败不应中断整轮对话。
        }
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}

// ---------- 用量统计 ----------
export const getLlmUsage = (query = {}) => {
  const params = new URLSearchParams();
  Object.entries(query).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") params.set(key, value);
  });
  const suffix = params.toString();
  return fetchJson(`${apiPrefix}/runtime/usage/llm${suffix ? `?${suffix}` : ""}`);
};

// ---------- 通用工具 ----------
export function maybeParseJson(raw, label) {
  const text = String(raw ?? "").trim();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`${label} 不是合法 JSON: ${error.message}`);
  }
}

export function generateRequestId(prefix = "runtime-console") {
  const stamp = new Date()
    .toISOString()
    .replaceAll(":", "")
    .replaceAll(".", "")
    .replace("T", "-")
    .replace("Z", "");
  return `${prefix}-${stamp}`;
}
