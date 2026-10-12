// 对话页（管理员与成员共用）：左侧会话列表 + 模型选择，右侧多轮对话。
//
// 一轮对话固定走 POST /chat/turns 的 SSE 流：普通问答以 delta 帧流式回填（打字机效果），
// 识别为垂类业务时后端只回一帧 task，这里再轮询任务并把最终结果从会话记录里拉回来。
// 页面本身不认识任何垂类 agent，新增业务无需改动此处。

const { ref, computed, onMounted, onUnmounted, nextTick } = window.Vue;
const { ElMessage, ElMessageBox } = window.ElementPlus;

import {
  getChatSessions,
  deleteChatSession,
  getChatMessages,
  getModels,
  getTaskStatus,
  streamChatTurn,
} from "../api.js";

const TERMINAL_TASK_STATUSES = ["success", "failure", "revoked"];

export default {
  name: "ChatPage",
  setup() {
    const sessions = ref([]);
    const currentSessionId = ref("");
    const messages = ref([]);
    const models = ref([]);
    const selectedModelId = ref("");
    const input = ref("");
    const sending = ref(false);
    const routing = ref(false);
    const stage = ref("");
    const loading = ref(false);
    const bodyRef = ref(null);

    // 每个异步任务的轮询句柄，组件卸载时必须清掉，否则会一直打后端。
    const taskTimers = new Map();

    const selectedModel = computed(
      () => models.value.find((item) => item.id === selectedModelId.value) || null,
    );
    const canSend = computed(() => !sending.value && Boolean(input.value.trim()));

    function notifyScroll() {
      nextTick(() => {
        const el = bodyRef.value;
        if (el) el.scrollTop = el.scrollHeight;
      });
    }

    async function loadModels() {
      const payload = await getModels(true);
      models.value = payload.items || [];
      if (!selectedModelId.value) {
        const preferred =
          models.value.find((item) => item.is_default) || models.value[0];
        selectedModelId.value = preferred ? preferred.id : "";
      }
    }

    async function loadSessions() {
      const payload = await getChatSessions();
      sessions.value = payload.items || [];
    }

    function toViewModel(record) {
      return {
        id: record.id,
        role: record.role,
        content: record.content,
        taskId: record.metadata_payload?.task_id || "",
        agentId: record.metadata_payload?.agent_id || "",
        mode: record.metadata_payload?.mode || "",
        citations: record.metadata_payload?.citations || [],
        contexts: record.metadata_payload?.contexts || [],
        insufficientEvidence: record.metadata_payload?.insufficient_evidence || false,
      };
    }

    async function reloadTranscript() {
      if (!currentSessionId.value) return;
      const payload = await getChatMessages(currentSessionId.value);
      messages.value = (payload.items || []).map(toViewModel);
      notifyScroll();
    }

    async function openSession(sessionId) {
      if (sending.value) return;
      currentSessionId.value = sessionId;
      loading.value = true;
      try {
        await reloadTranscript();
      } catch (error) {
        ElMessage.error(`会话内容加载失败：${error.message}`);
      } finally {
        loading.value = false;
      }
    }

    function newSession() {
      if (sending.value) return;
      stopAllTaskTimers();
      currentSessionId.value = "";
      messages.value = [];
      input.value = "";
    }

    async function removeSession(sessionId) {
      if (sending.value) return;
      try {
        await ElMessageBox.confirm(
          "确定删除该会话？会话记录将被清除，运行审计仍会保留。",
          "删除会话",
          { type: "warning", confirmButtonText: "删除", cancelButtonText: "取消" },
        );
      } catch {
        return;
      }
      try {
        await deleteChatSession(sessionId);
        stopTaskTimer(sessionId);
        if (currentSessionId.value === sessionId) newSession();
        await loadSessions();
        ElMessage.success("会话已删除");
      } catch (error) {
        ElMessage.error(`删除失败：${error.message}`);
      }
    }

    // ---------- 异步任务轮询 ----------
    function stopTaskTimer(taskId) {
      const timer = taskTimers.get(taskId);
      if (timer) {
        clearInterval(timer);
        taskTimers.delete(taskId);
      }
    }

    function stopAllTaskTimers() {
      taskTimers.forEach((timer) => clearInterval(timer));
      taskTimers.clear();
    }

    function trackTask(message, taskId) {
      const tick = async () => {
        try {
          const status = await getTaskStatus(taskId);
          message.taskStatus = status.status;
          message.workflowStatus = status.workflow_status || message.workflowStatus;
          message.taskNote = status.message || "";
          if (status.result) {
            message.taskSummary = status.result.final_result?.summary || "";
            message.taskErrors = status.result.errors || [];
          }
          const normalized = String(status.status || "").toLowerCase();
          if (TERMINAL_TASK_STATUSES.includes(normalized)) {
            stopTaskTimer(taskId);
            // 任务终态后以服务端会话记录为准：专员的最终答复由编排器写回同一会话。
            if (!sending.value) await reloadTranscript();
          }
        } catch (error) {
          stopTaskTimer(taskId);
          message.taskStatus = "failure";
          message.taskNote = `任务状态查询失败：${error.message}`;
        }
      };
      taskTimers.set(taskId, setInterval(tick, 3000));
      tick();
    }

    // ---------- 发送一轮对话 ----------
    async function send() {
      const content = input.value.trim();
      if (!content || sending.value) return;
      if (!selectedModelId.value) {
        ElMessage.warning("请先在模型池中选择一个模型");
        return;
      }

      messages.value.push({ id: `local-user-${Date.now()}`, role: "user", content });
      messages.value.push({
        id: `local-assistant-${Date.now()}`,
        role: "assistant",
        content: "",
        streaming: true,
      });
      const reply = messages.value[messages.value.length - 1];
      input.value = "";
      sending.value = true;
      routing.value = true;
      stage.value = "routing";
      notifyScroll();

      try {
        await streamChatTurn(
          {
            content,
            session_id: currentSessionId.value || null,
            model_entry_id: selectedModelId.value,
          },
          {
            onEvent: (event) => {
              if (event.session_id) currentSessionId.value = event.session_id;
              if (event.type === "status") {
                routing.value = true;
                stage.value = event.stage || "routing";
              } else if (event.type === "delta") {
                routing.value = false;
                reply.content += event.content || "";
                notifyScroll();
              } else if (event.type === "task") {
                routing.value = false;
                reply.taskId = event.task_id;
                reply.agentId = event.agent_id;
                reply.taskStatus = event.status || "pending";
                reply.content = "";
                reply.streaming = false;
                trackTask(reply, event.task_id);
              } else if (event.type === "evidence") {
                reply.mode = event.mode;
                reply.citations = event.citations || [];
                reply.contexts = event.contexts || [];
                reply.insufficientEvidence = Boolean(event.insufficient_evidence);
              } else if (event.type === "done") {
                routing.value = false;
                reply.streaming = false;
                if (event.mode === "task") reply.role = "assistant";
              } else if (event.type === "error") {
                routing.value = false;
                reply.streaming = false;
                reply.error = event.message || "对话失败";
              }
            },
          },
        );
      } catch (error) {
        reply.streaming = false;
        reply.error = error.message;
        ElMessage.error(`对话失败：${error.message}`);
      } finally {
        sending.value = false;
        routing.value = false;
        notifyScroll();
        // 首轮会新建会话（标题取自首条消息），因此每轮结束后刷新左侧列表。
        loadSessions().catch(() => {});
      }
    }

    function sessionLabel(item) {
      return item.metadata_payload?.title || `会话 ${item.session_id.slice(0, 8)}`;
    }

    function statusTagType(status) {
      const normalized = String(status || "").toLowerCase();
      if (normalized === "success") return "success";
      if (normalized === "failure") return "danger";
      return "warning";
    }

    onMounted(async () => {
      try {
        await loadModels();
      } catch (error) {
        ElMessage.error(`模型池加载失败：${error.message}`);
      }
      try {
        await loadSessions();
        if (sessions.value.length) await openSession(sessions.value[0].session_id);
      } catch (error) {
        ElMessage.error(`会话列表加载失败：${error.message}`);
      }
    });

    onUnmounted(stopAllTaskTimers);

    return {
      sessions, currentSessionId, messages, models, selectedModelId, selectedModel,
      input, sending, routing, stage, loading, bodyRef, canSend,
      loadSessions, openSession, newSession, removeSession, send, sessionLabel,
      statusTagType,
    };
  },
  template: `
    <div class="page-inner" style="display: flex; gap: 16px; align-items: stretch;">
      <el-card shadow="never" class="block-card" style="width: 280px; flex: none;">
        <template #header>
          <div class="card-header">
            <span class="title">会话<span class="api-pill">GET /chat/sessions</span></span>
            <el-button type="primary" size="small" :disabled="sending" @click="newSession">新建</el-button>
          </div>
        </template>
        <div style="display: flex; flex-direction: column; gap: 6px;">
          <div
            v-for="item in sessions"
            :key="item.session_id"
            @click="openSession(item.session_id)"
            :style="{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              gap: '6px', padding: '8px 10px', borderRadius: '6px', cursor: 'pointer',
              background: item.session_id === currentSessionId ? 'var(--el-fill-color)' : 'transparent',
            }"
          >
            <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
              {{ sessionLabel(item) }}
            </span>
            <el-button
              text size="small" type="danger"
              :disabled="sending"
              @click.stop="removeSession(item.session_id)"
            >删除</el-button>
          </div>
          <div v-if="!sessions.length" class="hint-text">暂无会话，点击“新建”开始对话</div>
        </div>
      </el-card>

      <el-card shadow="never" class="block-card" style="flex: 1; min-width: 0;">
        <template #header>
          <div class="card-header">
            <span class="title">
              对话
              <span class="api-pill">POST /chat/turns (SSE)</span>
            </span>
            <el-select
              v-model="selectedModelId"
              :disabled="sending"
              size="small"
              style="width: 260px;"
              placeholder="选择模型"
            >
              <el-option
                v-for="item in models"
                :key="item.id"
                :label="item.is_default ? item.name + '（默认）' : item.name"
                :value="item.id"
              >
                <span>{{ item.name }}</span>
                <span class="hint-text" style="margin-left: 8px;">{{ item.model }}</span>
              </el-option>
            </el-select>
          </div>
        </template>

        <div
          ref="bodyRef"
          v-loading="loading"
          style="height: 420px; overflow-y: auto; padding-right: 6px; display: flex; flex-direction: column; gap: 12px;"
        >
          <div v-if="!messages.length" class="hint-text">
            直接提问即可。事实性问题会先检索财税法规知识库，再由所选模型依据条款回答；
            证据不足时会明确说明。若请求属于某个垂类业务（例如生成公众号文章），
            系统会自行判断并交给对应的 Agent 异步执行，完成后结果会自动回填到这里。
          </div>

          <div
            v-for="message in messages"
            :key="message.id"
            :style="{
              display: 'flex',
              justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
            }"
          >
            <div
              :style="{
                maxWidth: '78%', padding: '8px 12px', borderRadius: '8px',
                whiteSpace: 'pre-wrap', wordBreak: 'break-word', lineHeight: '1.6',
                background: message.role === 'user'
                  ? 'var(--el-color-primary-light-9)'
                  : 'var(--el-fill-color-light)',
                border: message.error ? '1px solid var(--el-color-danger)' : 'none',
              }"
            >
              <template v-if="message.taskId">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                  <el-tag size="small" type="info" effect="plain">垂类任务</el-tag>
                  <el-tag size="small" :type="statusTagType(message.taskStatus)">
                    {{ message.taskStatus || "pending" }}
                  </el-tag>
                  <span class="hint-text">Agent：{{ message.agentId }}</span>
                </div>
                <div>任务已提交，正在由 {{ message.agentId }} 异步执行。</div>
                <div v-if="message.taskNote" class="hint-text">{{ message.taskNote }}</div>
                <div v-if="message.taskSummary" style="margin-top: 6px;">
                  <div class="hint-text">执行结果</div>
                  <div>{{ message.taskSummary }}</div>
                </div>
                <div v-if="message.taskErrors && message.taskErrors.length" style="margin-top: 6px; color: var(--el-color-danger);">
                  {{ message.taskErrors.join("；") }}
                </div>
                <div class="hint-text" style="margin-top: 4px;">
                  任务号 <span class="mono">{{ message.taskId }}</span>
                </div>
              </template>
              <template v-else>
                <div v-if="message.mode === 'rag'" style="margin-bottom: 6px;">
                  <el-tag size="small" effect="plain">单跳知识库回答</el-tag>
                  <el-tag v-if="message.insufficientEvidence" size="small" type="warning" style="margin-left: 6px;">证据不足</el-tag>
                </div>
                <span>{{ message.content }}</span>
                <span v-if="message.streaming && !message.content">▍</span>
                <details v-if="message.citations && message.citations.length" style="margin-top: 10px; white-space: normal;">
                  <summary style="cursor: pointer;">查看本次依据（{{ message.citations.length }} 条）</summary>
                  <div v-for="citation in message.citations" :key="citation.index" style="margin-top: 10px;">
                    <strong>[{{ citation.index }}] {{ citation.title }} {{ citation.section }}</strong>
                    <el-tag v-if="citation.supplementary" size="small" type="info" style="margin-left: 6px;">补充条款</el-tag>
                    <div class="hint-text">{{ citation.source }}</div>
                    <div style="white-space: pre-wrap;">{{ message.contexts[citation.index - 1] }}</div>
                  </div>
                </details>
              </template>
              <div v-if="message.error" style="margin-top: 4px; color: var(--el-color-danger);">
                {{ message.error }}
              </div>
            </div>
          </div>

          <div v-if="routing" class="hint-text">{{ stage === "retrieving" ? "正在检索法规并生成有依据的回答；首次加载本地模型可能较慢…" : "正在判断请求类型…" }}</div>
        </div>

        <div style="margin-top: 12px; display: flex; gap: 8px; align-items: flex-end;">
          <el-input
            v-model="input"
            type="textarea"
            :rows="3"
            resize="none"
            placeholder="输入消息，Enter 发送 / Shift+Enter 换行"
            @keydown.enter.exact.prevent="send"
          ></el-input>
          <el-button type="primary" :loading="sending" :disabled="!canSend" @click="send">
            发送
          </el-button>
        </div>
        <div class="hint-text" style="margin-top: 6px;">
          当前模型：{{ selectedModel ? selectedModel.name + "（" + selectedModel.model + "）" : "未选择" }}
        </div>
      </el-card>
    </div>
  `,
};
