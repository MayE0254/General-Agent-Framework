// P0/P1 验收看板页：平台能力统计 + P0/P1 checklist + 能力演示（知识库 / 审批链 / 记忆 scope）。

const { ref, computed } = window.Vue;
const { useRouter } = window.VueRouter;
const { ElMessage } = window.ElementPlus;

import { getAcceptance, runSyncWorkflow, queryToolCalls, createMemory, listMemories, generateRequestId } from "../api.js";
import { autoRefresh } from "../hooks.js";
import { short } from "../components.js";

export default {
  name: "AcceptancePage",
  setup() {
    const router = useRouter();
    const data = ref(null);
    const loading = ref(false);

    const statsCards = computed(() => {
      const s = data.value?.stats || {};
      const fmt = (obj) =>
        obj && Object.keys(obj).length
          ? Object.entries(obj).map(([k, v]) => `${k}=${v}`).join(" / ")
          : "-";
      return [
        { title: "工作流", value: s.workflow_runs, hint: fmt(s.workflow_runs_by_status) },
        { title: "Agent 执行", value: s.agent_runs, hint: `task_runs=${s.task_runs ?? "-"}` },
        { title: "工具调用", value: s.tool_calls, hint: fmt(s.tool_calls_by_status) },
        { title: "LLM 调用", value: s.llm_calls, hint: `${data.value?.runtime?.llm_provider || "-"} / ${data.value?.runtime?.llm_model || "-"}` },
        { title: "审批任务", value: s.reviews, hint: `续跑 ${s.reviews_with_continuation ?? 0} / 链式 ${s.chained_reviews ?? 0}` },
        { title: "记忆条目", value: s.memories, hint: fmt(s.memories_by_scope) },
      ];
    });

    const p0 = computed(() => data.value?.p0 || []);
    const p1 = computed(() => data.value?.p1 || []);

    async function refresh() {
      loading.value = true;
      try {
        data.value = await getAcceptance();
      } catch (error) {
        ElMessage.error(`验收数据加载失败：${error.message}`);
      } finally {
        loading.value = false;
      }
    }

    autoRefresh(refresh);

    // ---------- 能力演示 ----------
    const demoOutput = ref("");
    const demoRunning = ref(false);
    const demoJumpReviewId = ref("");

    async function runSyncDemo(payload) {
      const result = await runSyncWorkflow(payload);
      let knowledgeCalls = [];
      try {
        const calls = await queryToolCalls(payload.request_id, 20);
        knowledgeCalls = (calls.items || []).filter((item) => item.tool_name === "knowledge_search");
      } catch { /* 查询失败不影响演示主流程 */ }
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
      demoOutput.value =
        `=== 摘要 ===\n${JSON.stringify(summary, null, 2)}\n\n=== 完整响应 ===\n${JSON.stringify(result, null, 2)}`;
      await refresh();
      return result;
    }

    async function demoKnowledgeRun() {
      demoRunning.value = true;
      demoJumpReviewId.value = "";
      try {
        await runSyncDemo({
          request_id: generateRequestId("acceptance-knowledge"),
          initial_agent_id: "planner_agent",
          input_text: "检索平台知识库并总结平台能力",
          structured_input: { planner_tool_query: "multi-agent platform" },
        });
      } catch (error) {
        demoOutput.value = String(error);
      } finally {
        demoRunning.value = false;
      }
    }

    async function demoReviewChain() {
      demoRunning.value = true;
      demoJumpReviewId.value = "";
      try {
        const result = await runSyncDemo({
          request_id: generateRequestId("acceptance-review"),
          initial_agent_id: "planner_agent",
          input_text: "演示人工审批链路：运行结束后等待人工决策",
          structured_input: { requires_human: true },
        });
        if (result.review_id) {
          demoJumpReviewId.value = result.review_id;
        } else {
          demoOutput.value +=
            "\n\n（本次运行未产生 review_id：请确认 structured_input.requires_human=true 且链路包含 reviewer_agent。）";
        }
      } catch (error) {
        demoOutput.value = String(error);
      } finally {
        demoRunning.value = false;
      }
    }

    async function demoMemoryScopes() {
      demoRunning.value = true;
      demoJumpReviewId.value = "";
      demoOutput.value = "写入三条 scope 记忆...";
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
          created.push(await createMemory(entry));
        }
        const globalList = await listMemories({ scope: "global", limit: 5 });
        demoOutput.value = JSON.stringify(
          {
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
          },
          null,
          2,
        );
        await refresh();
      } catch (error) {
        demoOutput.value = String(error);
      } finally {
        demoRunning.value = false;
      }
    }

    function openRun(requestId) {
      router.push(`/runs/${encodeURIComponent(requestId)}`);
    }
    function openReview(reviewId) {
      router.push({ path: "/reviews", query: { reviewId } });
    }

    return {
      data, loading, refresh, statsCards, p0, p1,
      demoOutput, demoRunning, demoJumpReviewId,
      demoKnowledgeRun, demoReviewChain, demoMemoryScopes,
      openRun, openReview, short,
    };
  },
  template: `
    <div class="page-inner" v-loading="loading">
      <div class="three-col">
        <el-card shadow="never" class="block-card stat-card" v-for="card in statsCards" :key="card.title">
          <div class="stat-value mono">{{ card.value ?? "-" }}</div>
          <div class="stat-title">{{ card.title }}</div>
          <div class="hint-text mono">{{ card.hint }}</div>
        </el-card>
      </div>

      <el-divider content-position="left">能力演示</el-divider>
      <el-card shadow="never" class="block-card" style="margin-bottom: 16px;">
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;">
          <el-button type="primary" :loading="demoRunning" @click="demoKnowledgeRun">知识库检索运行</el-button>
          <el-button type="warning" :loading="demoRunning" @click="demoReviewChain">审批链路演示</el-button>
          <el-button :loading="demoRunning" @click="demoMemoryScopes">记忆 scope 演示</el-button>
          <el-button v-if="demoJumpReviewId" type="success" @click="openReview(demoJumpReviewId)">
            跳转审批 {{ short(demoJumpReviewId, 18) }}
          </el-button>
        </div>
        <json-block :value="demoOutput" empty-text="点击上方按钮运行演示，输出会显示在这里"></json-block>
      </el-card>

      <el-divider content-position="left">P0 Checklist</el-divider>
      <div class="check-grid">
        <el-card v-for="item in p0" :key="item.title" shadow="never" class="block-card check-card">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong><el-tag size="small" type="info" style="margin-right: 6px;">{{ item.phase }}</el-tag>{{ item.title }}</strong>
            <status-tag :value="item.status"></status-tag>
          </div>
          <div class="check-evidence mono">
            <div v-for="(line, i) in item.evidence || []" :key="i">{{ line }}</div>
            <div v-if="item.api" class="hint-text">API: {{ item.api }}</div>
          </div>
        </el-card>
      </div>

      <el-divider content-position="left">P1 Checklist</el-divider>
      <div class="check-grid">
        <el-card v-for="item in p1" :key="item.title" shadow="never" class="block-card check-card">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong><el-tag size="small" type="warning" style="margin-right: 6px;">{{ item.phase }}</el-tag>{{ item.title }}</strong>
            <status-tag :value="item.status"></status-tag>
          </div>
          <div class="check-evidence mono">
            <div v-for="(line, i) in item.evidence || []" :key="i">{{ line }}</div>
            <div v-if="item.api" class="hint-text">API: {{ item.api }}</div>
          </div>
        </el-card>
      </div>
    </div>
  `,
};
