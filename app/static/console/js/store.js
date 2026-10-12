// 全局响应式状态：主题、审批 SSE 事件、最近一次全局刷新时间。

const { reactive } = window.Vue;

export const store = reactive({
  theme: document.documentElement.classList.contains("dark") ? "dark" : "light",

  // 当前登录用户（{id, username, role, display_name}）；null 表示未登录。
  // authReady 表示已完成一次 /auth/me 会话探测，路由守卫依赖它避免误跳登录页。
  currentUser: null,
  authReady: false,

  // 审批 SSE 最新事件（总览页与审批中心页共享）
  reviewEvent: null,
  reviewEventAt: null,
  reviewEventConnected: false,

  // 顶栏刷新状态；refreshTick 供各页面 watch 实现"立即刷新"
  refreshing: false,
  lastRefreshedAt: null,
  refreshTick: 0,
});

export function toggleTheme() {
  const dark = store.theme !== "dark";
  store.theme = dark ? "dark" : "light";
  document.documentElement.classList.toggle("dark", dark);
  localStorage.setItem("console-theme", store.theme);
}

export function setReviewEvent(raw) {
  store.reviewEvent = raw;
  store.reviewEventAt = new Date().toLocaleTimeString();
}

// EventSource 单例：路由切换不重复建连。
let eventSource = null;

// 与 api.js 的 API_BASE 保持一致（/api/v1 前缀 + 审批事件流端点）
const REVIEW_EVENTS_URL = "/api/v1/runtime/reviews/events";

export function connectReviewEvents() {
  if (eventSource) return;
  eventSource = new EventSource(REVIEW_EVENTS_URL);
  eventSource.onopen = () => {
    store.reviewEventConnected = true;
  };
  eventSource.onmessage = (event) => {
    store.reviewEventConnected = true;
    setReviewEvent(event.data || "收到审批事件");
  };
  eventSource.onerror = () => {
    store.reviewEventConnected = false;
  };
}
