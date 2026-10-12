// 工作流页：提交异步任务 + 提交台账 + 最近执行 + Celery 任务查询/轮询。

const { ref, reactive, onMounted } = window.Vue;
const { useRouter } = window.VueRouter;
const { ElMessage } = window.ElementPlus;

import {
  getAgents,
  getSubmissions,
  getWorkflowRuns,
  getTaskStatus,
  submitAsyncWorkflow,
  generateRequestId,
  maybeParseJson,
} from "../api.js";
import { autoRefresh } from "../hooks.js";

export default {
  name: "WorkflowsPage",
  setup() {
    const router = useRouter();

    // ----- 提交表单（user_id 由后端强制取当前登录用户，前端不提供输入） -----
    const agents = ref([]);
    const form = reactive({
      requestId: "",
      agentId: "",
      sessionId: "",
      inputText: "",
      structuredInput: "{}",
    });
    const submitting = ref(false);

    // ----- 列表 -----
    const submissions = ref([]);
    const workflows = ref([]);

    // ----- 任务查询 -----
    const taskForm = reactive({ taskId: "" });
    const taskResult = ref(null);
    const taskError = ref("");
    const polling = ref(false);

    async function loadLists() {
      const [sub, wf] = await Promise.all([
        getSubmissions(10).catch(() => ({ items: [] })),
        getWorkflowRuns(10).catch(() => ({ items: [] })),
      ]);
      submissions.value = sub.items || [];
      workflows.value = wf.items || [];
    }

    onMounted(async () => {
      try {
        agents.value = await getAgents();
      } catch (error) {
        ElMessage.error(`Agent 列表加载失败：${error.message}`);
      }
    });

    autoRefresh(loadLists);

    async function submitWorkflow() {
      if (!form.requestId.trim()) return ElMessage.warning("request_id 不能为空");
      if (!form.agentId) return ElMessage.warning("请选择 initial_agent_id");
      if (!form.inputText.trim()) return ElMessage.warning("input_text 不能为空");
      let structured;
      try {
        structured = maybeParseJson(form.structuredInput, "structured_input");
      } catch (error) {
        return ElMessage.error(error.message);
      }
      const payload = {
        request_id: form.requestId.trim(),
        initial_agent_id: form.agentId,
        input_text: form.inputText.trim(),
        structured_input: structured,
      };
      if (form.sessionId.trim()) payload.session_id = form.sessionId.trim();

      submitting.value = true;
      try {
        const result = await submitAsyncWorkflow(payload);
        ElMessage.success(`异步任务已提交：${form.requestId}，状态见右侧台账`);
        // Stay on this page: the submission ledger on the right reflects the
        // task status via auto-refresh. Navigate to the detail page only on
        // explicit user action (row click / task lookup / polling).
        if (result?.task_id) {
          taskForm.taskId = result.task_id;
        }
        await loadLists();
      } catch (error) {
        ElMessage.error(`提交失败：${error.message}`);
      } finally {
        submitting.value = false;
      }
    }

    async function lookupTask() {
      const taskId = taskForm.taskId.trim();
      if (!taskId) return ElMessage.warning("请输入 task_id");
      taskError.value = "";
      try {
        taskResult.value = await getTaskStatus(taskId);
        const requestId = taskResult.value?.request_id || taskResult.value?.result?.request_id;
        if (requestId) {
          router.push(`/runs/${encodeURIComponent(requestId)}`);
        }
      } catch (error) {
        taskError.value = String(error.message || error);
      }
    }

    async function pollTask() {
      const taskId = taskForm.taskId.trim();
      if (!taskId) return ElMessage.warning("请输入 task_id");
      polling.value = true;
      try {
        for (let i = 0; i < 30; i += 1) {
          try {
            taskResult.value = await getTaskStatus(taskId);
          } catch (error) {
            taskError.value = String(error.message || error);
            break;
          }
          const status = String(taskResult.value?.status || "");
          if (["success", "failure"].includes(status)) {
            const requestId = taskResult.value?.request_id || taskResult.value?.result?.request_id;
            if (requestId) router.push(`/runs/${encodeURIComponent(requestId)}`);
            break;
          }
          await new Promise((resolve) => setTimeout(resolve, 2000));
        }
      } finally {
        polling.value = false;
      }
    }

    function onSubmissionRow(row) {
      if (row.task_id) taskForm.taskId = row.task_id;
      if (row.request_id) router.push(`/runs/${encodeURIComponent(row.request_id)}`);
    }
    function onWorkflowRow(row) {
      router.push(`/runs/${encodeURIComponent(row.request_id)}`);
    }

    return {
      agents, form, submitting,
      submissions, workflows,
      taskForm, taskResult, taskError, polling,
      submitWorkflow, lookupTask, pollTask,
      onSubmissionRow, onWorkflowRow,
      generateRequestId,
    };
  },
  template: `
    <div class="page-inner">
      <div class="two-col">
        <div>
          <el-card shadow="never" class="block-card">
            <template #header>
              <div class="card-header">
                <span class="title">提交异步工作流<span class="api-pill">POST /workflows/run/async</span></span>
              </div>
            </template>
            <el-form label-position="top" @submit.prevent>
              <div class="two-col" style="gap: 12px;">
                <el-form-item label="request_id" required>
                  <el-input v-model="form.requestId" placeholder="例如 runtime-console-001">
                    <template #append>
                      <el-button @click="form.requestId = generateRequestId()">生成</el-button>
                    </template>
                  </el-input>
                </el-form-item>
                <el-form-item label="initial_agent_id" required>
                  <el-select v-model="form.agentId" placeholder="选择 agent" style="width: 100%;">
                    <el-option v-for="a in agents" :key="a.agent_id"
                      :label="a.agent_id + ' | ' + a.agent_name" :value="a.agent_id"></el-option>
                  </el-select>
                </el-form-item>
              </div>
              <el-form-item label="session_id"><el-input v-model="form.sessionId" placeholder="可选"></el-input></el-form-item>
              <el-form-item label="input_text" required>
                <el-input v-model="form.inputText" type="textarea" :rows="4" placeholder="输入工作流要处理的任务"></el-input>
              </el-form-item>
              <el-form-item label="structured_input (JSON)">
                <el-input v-model="form.structuredInput" type="textarea" :rows="4"></el-input>
              </el-form-item>
              <el-button type="primary" :loading="submitting" @click="submitWorkflow">提交异步任务</el-button>
            </el-form>
          </el-card>

          <el-card shadow="never" class="block-card">
            <template #header>
              <div class="card-header">
                <span class="title">异步任务状态<span class="api-pill">GET /workflows/tasks/{task_id}</span></span>
              </div>
            </template>
            <div style="display: flex; gap: 8px;">
              <el-input v-model="taskForm.taskId" placeholder="Celery task_id，提交后自动带入"></el-input>
              <el-button @click="lookupTask">查询</el-button>
              <el-button :loading="polling" @click="pollTask">轮询直到结束</el-button>
            </div>
            <div v-if="taskError" class="hint-text" style="margin-top: 10px; color: var(--el-color-danger);">{{ taskError }}</div>
            <json-block v-if="taskResult" :value="taskResult" style="margin-top: 10px;"></json-block>
          </el-card>
        </div>

        <div>
          <el-card shadow="never" class="block-card">
            <template #header>
              <div class="card-header">
                <span class="title">最近异步提交<span class="api-pill">GET /workflows/submissions</span></span>
                <el-button text size="small" @click="loadLists()">刷新</el-button>
              </div>
            </template>
            <el-table :data="submissions" size="small" class="clickable-table" @row-click="onSubmissionRow">
              <el-table-column label="Request" min-width="170">
                <template #default="{ row }"><span class="mono">{{ row.request_id || "-" }}</span></template>
              </el-table-column>
              <el-table-column label="Task" min-width="150">
                <template #default="{ row }"><span class="mono hint-text">{{ row.task_id || "-" }}</span></template>
              </el-table-column>
              <el-table-column label="Status" width="120">
                <template #default="{ row }"><status-tag :value="row.status"></status-tag></template>
              </el-table-column>
              <el-table-column label="Submitted" width="165">
                <template #default="{ row }"><span class="hint-text">{{ row.created_at || "-" }}</span></template>
              </el-table-column>
              <template #empty>暂无提交记录</template>
            </el-table>
            <div class="hint-text" style="margin-top: 8px;">
              台账为后端任务中心真实数据；点击行可查询任务并跳转运行详情。
            </div>
          </el-card>

          <el-card shadow="never" class="block-card">
            <template #header>
              <div class="card-header">
                <span class="title">最近工作流<span class="api-pill">GET /runtime/workflow-runs</span></span>
                <el-button text size="small" @click="loadLists()">刷新</el-button>
              </div>
            </template>
            <el-table :data="workflows" size="small" class="clickable-table" @row-click="onWorkflowRow">
              <el-table-column label="Request" min-width="180">
                <template #default="{ row }"><span class="mono">{{ row.request_id }}</span></template>
              </el-table-column>
              <el-table-column label="Status" width="140">
                <template #default="{ row }"><status-tag :value="row.status"></status-tag></template>
              </el-table-column>
              <el-table-column label="Path" min-width="200">
                <template #default="{ row }">
                  <span class="hint-text mono">{{ (row.execution_path || []).join(" -> ") || "-" }}</span>
                </template>
              </el-table-column>
              <el-table-column label="Updated" width="165">
                <template #default="{ row }"><span class="hint-text">{{ row.updated_at || "-" }}</span></template>
              </el-table-column>
              <template #empty>暂无数据</template>
            </el-table>
          </el-card>
        </div>
      </div>
    </div>
  `,
};
