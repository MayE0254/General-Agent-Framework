// 控制台入口：布局（侧边栏 + 顶栏 + 内容区）、路由（含登录守卫与角色管控）、
// 主题切换、SSE 连接。菜单与页面访问按角色区分：admin 全量，user 仅自己的工作流/审批。

const { createApp, computed, watch } = window.Vue;
const { createRouter, createWebHashHistory } = window.VueRouter;

import { connectReviewEvents, store, toggleTheme } from "./store.js";
import { installComponents } from "./components.js";
import { apiMe, apiLogout } from "./api.js";

import LoginPage from "./pages/login.js";
import OverviewPage from "./pages/overview.js";
import ChatPage from "./pages/chat.js";
import ModelsPage from "./pages/models.js";
import UsagePage from "./pages/usage.js";
import WorkflowsPage from "./pages/workflows.js";
import RunDetailPage from "./pages/run-detail.js";
import ReviewsPage from "./pages/reviews.js";
import UsersPage from "./pages/users.js";
import AutomationPage from "./pages/automation.js";
import SystemPage from "./pages/system.js";
import AcceptancePage from "./pages/acceptance.js";

const routes = [
  { path: "/login", name: "login", component: LoginPage, meta: { title: "登录", public: true } },
  { path: "/", name: "overview", component: OverviewPage, meta: { title: "总览" } },
  { path: "/chat", name: "chat", component: ChatPage, meta: { title: "对话" } },
  { path: "/workflows", name: "workflows", component: WorkflowsPage, meta: { title: "工作流" } },
  { path: "/runs/:requestId?", name: "run-detail", component: RunDetailPage, meta: { title: "运行详情" } },
  { path: "/reviews", name: "reviews", component: ReviewsPage, meta: { title: "审批中心" } },
  { path: "/automation", name: "automation", component: AutomationPage, meta: { title: "自动化 Agent" } },
  { path: "/users", name: "users", component: UsersPage, meta: { title: "成员管理", roles: ["admin"] } },
  { path: "/system", name: "system", component: SystemPage, meta: { title: "系统状态", roles: ["admin"] } },
  { path: "/acceptance", name: "acceptance", component: AcceptancePage, meta: { title: "验收看板", roles: ["admin"] } },
  { path: "/models", name: "models", component: ModelsPage, meta: { title: "模型池", roles: ["admin"] } },
  { path: "/usage", name: "usage", component: UsagePage, meta: { title: "用量统计" } },
  { path: "/:pathMatch(.*)*", redirect: "/" },
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

// 登录守卫：启动时先探测一次 /auth/me 恢复会话；未登录跳登录页，角色不符回首页。
router.beforeEach(async (to) => {
  if (!store.authReady) {
    try {
      store.currentUser = await apiMe();
    } catch {
      store.currentUser = null;
    }
    store.authReady = true;
  }
  const user = store.currentUser;
  if (to.meta.public) return user ? { path: "/" } : true;
  if (!user) return { path: "/login" };
  if (to.meta.roles && !to.meta.roles.includes(user.role)) return { path: "/" };
  return true;
});

const App = {
  name: "ConsoleApp",
  computed: {
    pageTitle() {
      return this.$route.meta.title || "MultiAgent";
    },
    activeMenu() {
      // 菜单 index 使用 path（而非 name）：el-menu router 模式会执行 router.push(index)，
      // 用 name 会在 /runs/xxx 下被相对解析成 /runs/overview，导致误触发 404 查询。
      const path = String(this.$route.path || "/");
      if (path.startsWith("/runs")) return "/workflows";
      return path;
    },
    themeIcon() {
      return this.store.theme === "dark" ? "Sunny" : "Moon";
    },
    isAdmin() {
      return this.store.currentUser?.role === "admin";
    },
    userLabel() {
      const user = this.store.currentUser;
      if (!user) return "";
      return user.display_name || user.username;
    },
    // 按角色渲染菜单：admin 全量；普通用户仅自己的工作流与审批。
    menuItems() {
      if (this.isAdmin) {
        return [
          { index: "/", icon: "Odometer", label: "总览" },
          { index: "/chat", icon: "ChatDotRound", label: "对话" },
          { index: "/workflows", icon: "VideoPlay", label: "工作流" },
          { index: "/reviews", icon: "Checked", label: "审批中心" },
          { index: "/users", icon: "User", label: "成员管理" },
          { index: "/automation", icon: "Timer", label: "自动化 Agent" },
          { index: "/models", icon: "Coin", label: "模型池" },
          { index: "/usage", icon: "DataLine", label: "用量统计" },
          { index: "/system", icon: "Monitor", label: "系统状态" },
          { index: "/acceptance", icon: "Stamp", label: "验收看板" },
        ];
      }
      return [
        { index: "/", icon: "Odometer", label: "总览" },
        { index: "/chat", icon: "ChatDotRound", label: "对话" },
        { index: "/workflows", icon: "VideoPlay", label: "我的工作流" },
        { index: "/reviews", icon: "Checked", label: "我的审批" },
        { index: "/automation", icon: "Timer", label: "自动化 Agent" },
        { index: "/usage", icon: "DataLine", label: "用量统计" },
      ];
    },
  },
  setup() {
    // 登录成功后才建立审批 SSE 连接（EventSource 同源自动携带会话 cookie）。
    watch(
      () => Boolean(store.authReady && store.currentUser),
      (signedIn) => {
        if (signedIn) connectReviewEvents();
      },
      { immediate: true },
    );

    async function logout() {
      try {
        await apiLogout();
      } catch {
        // 后端登出失败也照常清理前端登录态。
      }
      store.currentUser = null;
      window.location.hash = "#/login";
    }

    return { store, toggleTheme, logout };
  },
  template: `
    <el-container class="console-layout">
      <aside class="console-sidebar">
        <div class="console-brand">
          <el-icon :size="26" color="var(--el-color-primary)"><Platform></Platform></el-icon>
          <div>
            <div class="brand-name">MultiAgent</div>
            <div class="brand-sub">运行中台控制台</div>
          </div>
        </div>
        <el-menu :default-active="activeMenu" router>
          <el-menu-item v-for="item in menuItems" :key="item.index" :index="item.index">
            <el-icon><component :is="item.icon"></component></el-icon><span>{{ item.label }}</span>
          </el-menu-item>
        </el-menu>
        <div class="console-sidebar-footer">
          <span class="hint-text">内部平台</span>
          <el-tooltip :content="store.theme === 'dark' ? '切换到亮色' : '切换到暗色'" placement="bottom">
            <el-button text circle @click="toggleTheme">
              <el-icon><component :is="themeIcon"></component></el-icon>
            </el-button>
          </el-tooltip>
        </div>
      </aside>

      <el-container class="console-main">
        <header class="console-topbar">
          <span class="page-title">{{ pageTitle }}</span>
          <div style="display: flex; align-items: center; gap: 12px;">
            <el-tag
              :type="store.reviewEventConnected ? 'success' : 'info'"
              size="small"
              effect="plain"
            >{{ store.reviewEventConnected ? "SSE 已连接" : "SSE 未连接" }}</el-tag>
            <span class="hint-text" v-if="store.lastRefreshedAt">
              刷新于 {{ store.lastRefreshedAt }}
            </span>
            <el-button size="small" :loading="store.refreshing" @click="store.refreshTick += 1">
              立即刷新
            </el-button>
            <a href="/docs" target="_blank" rel="noreferrer">
              <el-button size="small" text>OpenAPI</el-button>
            </a>
            <el-divider direction="vertical"></el-divider>
            <el-tag :type="isAdmin ? 'warning' : 'info'" size="small" effect="plain">
              {{ isAdmin ? "管理员" : "用户" }}
            </el-tag>
            <span class="hint-text">{{ userLabel }}</span>
            <el-button size="small" text @click="logout">退出登录</el-button>
          </div>
        </header>
        <main class="console-content">
          <router-view></router-view>
        </main>
      </el-container>
    </el-container>
  `,
};

const app = createApp(App);
app.use(router);
app.use(window.ElementPlus, { locale: window.ElementPlusLocaleZhCn });
for (const [name, comp] of Object.entries(window.ElementPlusIconsVue)) {
  app.component(name, comp);
}
installComponents(app);
app.mount("#app");
