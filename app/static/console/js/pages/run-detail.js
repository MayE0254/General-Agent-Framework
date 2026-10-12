// 运行详情页：request_id → 摘要 + Agent Timeline / Tool Calls / LLM Calls / Session Messages。

const { ref, computed, watch } = window.Vue;
const { useRoute, useRouter } = window.VueRouter;

import { getTaskDetail, getWorkflowDetail, getSessionMessages, getSubmissions } from "../api.js";

export default {
  name: "RunDetailPage",
  setup() {
    const route = useRoute();
    const router = useRouter();

    const requestIdInput = ref("");
    const loading = ref(false);
    const errorText = ref("");
    const taskDetail = ref(null);
    const workflowDetail = ref(null);
    const sessionMessages = ref(null);
    const activeTab = ref("agents");
    // Friendly hint instead of a raw 404 while a task is still running.
    const runningHint = ref({ type: "info", text: "" });

    const RUNNING_SUBMISSION_STATES = ["pending", "started", "running", "retry"];

    async function resolveRunningHint(requestId) {
      try {
        const subs = await getSubmissions(50);
        const item = (subs.items || []).find(
          (s) => (s.request_id || "") === requestId
        );
        if (!item) return { type: "info", text: "" };
        const status = String(item.status || "").toLowerCase();
        if (RUNNING_SUBMISSION_STATES.includes(status)) {
          return {
            type: "info",
            text: "任务正在运行中，运行详情将在结束后生成，请稍后刷新查看。",
          };
        }
        if (status === "failure") {
          return {
            type: "warning",
            text: `任务执行失败：${item.message || "可在工作流页查看台账与错误信息"}`,
          };
        }
        if (status === "success") {
          return {
            type: "info",
            text: "任务已结束，运行详情生成中，请稍后刷新查看。",
          };
        }
        return { type: "info", text: `任务状态：${item.status}` };
      } catch {
        return { type: "info", text: "" };
      }
    }

    // Tool Calls 筛选
    const toolFilters = ref({ failedOnly: false, retryableOnly: false });

    const summaryRows = computed(() => {
      const detail = taskDetail.value;
      if (!detail) return [];
      const taskRun = detail.task_run || {};
      const aggregates = detail.aggregates || {};
      const workflow = workflowDetail.value?.workflow_run || {};
      const metadata = taskRun.metadata_payload || workflow.metadata_payload || {};
      return [
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
        ["source_review", metadata.source_review_id || "-"],
      ];
    });

    const sourceReviewId = computed(() => {
      const taskRun = taskDetail.value?.task_run || {};
      const workflow = workflowDetail.value?.workflow_run || {};
      const metadata = taskRun.metadata_payload || workflow.metadata_payload || {};
      return metadata.source_review_id || "";
    });

    const filteredToolCalls = computed(() => {
      const items = taskDetail.value?.tool_calls || [];
      return items.filter((item) => {
        if (toolFilters.value.failedOnly && item.status === "success") return false;
        if (toolFilters.value.retryableOnly && !item.retryable) return false;
        return true;
      });
    });

    async function loadDetail() {
      const requestId = String(route.params.requestId || "").trim();
      requestIdInput.value = requestId;
      taskDetail.value = null;
      workflowDetail.value = null;
      sessionMessages.value = null;
      errorText.value = "";
      if (!requestId) return;
      loading.value = true;
      try {
        const [task, workflow] = await Promise.all([
          getTaskDetail(requestId).catch(() => null),
          getWorkflowDetail(requestId).catch(() => null),
        ]);
        taskDetail.value = task;
        workflowDetail.value = workflow;
        if (task) {
          runningHint.value = { type: "info", text: "" };
        } else {
          // No audit records yet: the task is likely still running. Show a
          // friendly hint (submission ledger status) instead of a 404 error.
          const hint = await resolveRunningHint(requestId);
          if (hint.text) {
            runningHint.value = hint;
          } else {
            errorText.value = `request_id '${requestId}' 暂无运行详情（任务尚未提交或记录不存在）。`;
          }
        }
        const sessionId = task?.task_run?.session_id;
        if (sessionId) {
          sessionMessages.value = await getSessionMessages(sessionId).catch((e) => ({
            __error: String(e.message || e),
          }));
        }
      } catch (error) {
        errorText.value = String(error.message || error);
      } finally {
        loading.value = false;
      }
    }

    watch(() => route.params.requestId, loadDetail, { immediate: true });

    function lookup() {
      const id = requestIdInput.value.trim();
      if (!id) return;
      router.push(`/runs/${encodeURIComponent(id)}`);
      // 同路由参数不变时手动触发
      if (String(route.params.requestId || "") === id) loadDetail();
    }

    function goReview(reviewId) {
      if (reviewId) router.push({ path: "/reviews", query: { reviewId } });
    }

    return {
      requestIdInput, loading, errorText, runningHint,
      taskDetail, sessionMessages, activeTab, toolFilters,
      summaryRows, sourceReviewId, filteredToolCalls,
      lookup, goReview,
    };
  },
  template: `
    <div class="page-inner">
      <el-card shadow="never" class="block-card">
        <template #header>
          <div class="card-header">
            <span class="title">运行详情<span class="api-pill">GET /runtime/tasks/{request_id}</span></span>
          </div>
        </template>
        <div style="display: flex; gap: 8px; max-width: 720px;">
          <el-input v-model="requestIdInput" placeholder="request_id" @keyup.enter="lookup"></el-input>
          <el-button type="primary" :loading="loading" @click="lookup">查询详情</el-button>
        </div>
        <el-alert v-if="errorText" type="error" :closable="false" style="margin-top: 12px;">
          {{ errorText }}
        </el-alert>
        <el-alert v-else-if="runningHint.text" :type="runningHint.type" :closable="false" style="margin-top: 12px;">
          {{ runningHint.text }}
        </el-alert>
        <el-alert v-else-if="!requestIdInput" type="info" :closable="false" style="margin-top: 12px;">
          从"工作流"页点击任意一行，或直接输入 request_id 查询完整执行链路。
        </el-alert>
      </el-card>

      <template v-if="taskDetail">
        <el-card shadow="never" class="block-card">
          <template #header>
            <div class="card-header">
              <span class="title">执行摘要</span>
              <el-button v-if="sourceReviewId" size="small" text type="primary" @click="goReview(sourceReviewId)">
                来源审批：{{ sourceReviewId }}
              </el-button>
            </div>
          </template>
          <el-descriptions :column="3" border size="small">
            <el-descriptions-item v-for="([k, v], i) in summaryRows" :key="i" :label="k">
              <span class="mono" style="word-break: break-all;">{{ v }}</span>
            </el-descriptions-item>
          </el-descriptions>
        </el-card>

        <el-card shadow="never" class="block-card">
          <el-tabs v-model="activeTab">
            <el-tab-pane label="Agent Timeline" name="agents">
              <div v-if="(taskDetail.agent_runs || []).length === 0" class="hint-text">暂无 Agent 执行记录</div>
              <el-timeline v-else>
                <el-timeline-item
                  v-for="item in taskDetail.agent_runs"
                  :key="item.id || item.step_index"
                  :type="(item.status || '') === 'success' ? 'success' : ((item.status || '') === 'failed' ? 'danger' : 'warning')"
                >
                  <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <strong>#{{ item.step_index }} {{ item.agent_id }}</strong>
                    <status-tag :value="item.status"></status-tag>
                  </div>
                  <div class="hint-text" style="margin-top: 4px;">
                    Summary: {{ item.summary || "-" }} ｜ Next: {{ item.next_agent_id || "-" }}
                  </div>
                  <div class="hint-text mono" style="word-break: break-all;">
                    {{ item.started_at_runtime || "-" }} → {{ item.completed_at_runtime || "-" }}
                  </div>
                  <json-block :value="item.output || {}" style="margin-top: 8px;"></json-block>
                </el-timeline-item>
              </el-timeline>
            </el-tab-pane>

            <el-tab-pane :label="'Tool Calls (' + (taskDetail.tool_calls || []).length + ')'" name="tools">
              <div style="margin-bottom: 10px; display: flex; gap: 16px;">
                <el-checkbox v-model="toolFilters.failedOnly">仅失败</el-checkbox>
                <el-checkbox v-model="toolFilters.retryableOnly">仅可重试</el-checkbox>
              </div>
              <div v-if="filteredToolCalls.length === 0" class="hint-text">当前筛选条件下没有 Tool 调用</div>
              <el-collapse v-else>
                <el-collapse-item v-for="(item, i) in filteredToolCalls" :key="i" :name="String(i)">
                  <template #title>
                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; min-width: 0;">
                      <strong class="mono">{{ item.tool_name || "-" }}</strong>
                      <status-tag :value="item.status"></status-tag>
                      <span class="hint-text mono">error_category={{ item.error_category || "-" }} ｜ retryable={{ !!item.retryable }}</span>
                    </div>
                  </template>
                  <div class="hint-text" style="margin-bottom: 6px;">{{ item.message || "-" }}</div>
                  <div class="hint-text mono" style="margin-bottom: 6px; word-break: break-all;">error_detail={{ item.error_detail || "-" }}</div>
                  <json-block :value="{
                    agent_id: item.agent_id,
                    arguments: item.arguments || {},
                    errors: item.errors || [],
                    content: item.content || {},
                    metrics: item.metrics || {},
                  }"></json-block>
                </el-collapse-item>
              </el-collapse>
            </el-tab-pane>

            <el-tab-pane :label="'LLM Calls (' + (taskDetail.llm_calls || []).length + ')'" name="llm">
              <json-block :value="taskDetail.llm_calls || []" empty-text="暂无 LLM 调用"></json-block>
            </el-tab-pane>

            <el-tab-pane label="Session Messages" name="messages">
              <div v-if="!sessionMessages" class="hint-text">当前 request 没有关联 session_id</div>
              <div v-else-if="sessionMessages.__error" class="hint-text" style="color: var(--el-color-danger);">
                {{ sessionMessages.__error }}
              </div>
              <json-block v-else :value="sessionMessages.items || sessionMessages" empty-text="暂无消息"></json-block>
            </el-tab-pane>
          </el-tabs>
        </el-card>
      </template>
    </div>
  `,
};
