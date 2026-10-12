// 审批中心页：左侧审批列表（可按状态筛选），右侧详情 + 批准/驳回操作台。
// 批准可选携带 continuation（续跑 agent），驳回仅记录决策不触发补偿动作。

const { ref, computed, watch, onMounted } = window.Vue;
const { useRoute, useRouter } = window.VueRouter;
const { ElMessage } = window.ElementPlus;

import { getReviews, getReviewDetail, decideReview, getUsers, maybeParseJson } from "../api.js";
import { store } from "../store.js";
import { autoRefresh } from "../hooks.js";
import { JsonBlock as _JsonBlock } from "../components.js"; // 全局已注册，导入仅为保持依赖显式

export default {
  name: "ReviewsPage",
  setup() {
    const route = useRoute();
    const router = useRouter();

    const isAdmin = computed(() => store.currentUser?.role === "admin");

    // ---------- 列表 ----------
    const filter = ref("pending");
    const items = ref([]);
    const listLoading = ref(false);

    // 管理员视图：按成员筛选 + 成员 id -> 显示名映射；普通用户后端强制只返回自己的审批。
    const members = ref([]);
    const memberFilter = ref("");
    const userMap = computed(() => {
      const map = {};
      for (const m of members.value) map[m.id] = m.display_name || m.username;
      return map;
    });
    const memberName = (userId) => userMap.value[userId] || userId || "-";

    async function loadMembers() {
      if (!isAdmin.value) return;
      try {
        const payload = await getUsers();
        members.value = payload.items || [];
      } catch {
        // 映射加载失败时降级显示原始 user_id，不阻塞审批列表。
      }
    }

    const filteredItems = computed(() => {
      if (filter.value === "all") return items.value;
      return items.value.filter((item) => String(item.status || "").toLowerCase() === filter.value);
    });

    async function loadList() {
      listLoading.value = true;
      try {
        const payload = await getReviews(50, isAdmin.value ? memberFilter.value : "");
        items.value = payload.items || [];
      } catch (error) {
        ElMessage.error(`审批列表加载失败：${error.message}`);
      } finally {
        listLoading.value = false;
      }
    }

    // ---------- 详情 ----------
    const detail = ref(null);
    const detailLoading = ref(false);
    const selectedId = ref("");

    // 审批操作表单（操作人固定为当前登录用户，由后端强制）
    const note = ref("");
    // 续跑配置（仅批准时生效）
    const contCollapse = ref([]);
    const showContinuation = ref(false);
    const contAgent = ref("");
    const contInput = ref("");
    const contStructured = ref("{}");
    const contRequestId = ref("");

    const terminal = computed(() => {
      const status = String(detail.value?.status || "").toLowerCase();
      return status === "approved" || status === "rejected";
    });

    const relationRows = computed(() => {
      const d = detail.value || {};
      return [
        ["来源审批", d.source_review_id],
        ["续跑 request", d.continuation_request_id],
        ["续跑状态", d.continuation_status],
        ["续跑 terminal_reason", d.continuation_terminal_reason],
        ["续跑时间", d.continuation_run_at],
      ];
    });

    async function loadDetail(reviewId) {
      if (!reviewId) return;
      selectedId.value = reviewId;
      detailLoading.value = true;
      try {
        const result = await getReviewDetail(reviewId);
        detail.value = result;
        note.value = result.decision_note || "";
        contAgent.value = result.agent_id || "";
        contInput.value = result.payload?.input_text || "";
        contStructured.value = JSON.stringify(result.payload?.structured_input ?? {}, null, 2);
        contRequestId.value = "";
      } catch (error) {
        ElMessage.error(`审批详情加载失败：${error.message}`);
      } finally {
        detailLoading.value = false;
      }
    }

    function selectRow(row) {
      router.replace({ query: { reviewId: row.id } });
      loadDetail(row.id);
    }

    // ---------- 批准 / 驳回 ----------
    const acting = ref(false);

    async function act(action) {
      if (!selectedId.value) return ElMessage.warning("请先选择一条审批");
      if (acting.value) return;
      acting.value = true;
      try {
        const payload = { note: note.value };
        if (action === "approve" && showContinuation.value && contAgent.value.trim() && contInput.value.trim()) {
          const continuation = {
            agent_id: contAgent.value.trim(),
            input_text: contInput.value.trim(),
            structured_input: maybeParseJson(contStructured.value, "续跑 structured_input"),
          };
          if (contRequestId.value.trim()) continuation.request_id = contRequestId.value.trim();
          payload.continuation = continuation;
        }
        const result = await decideReview(selectedId.value, action, payload);
        ElMessage.success(`审批已${action === "approve" ? "批准" : "驳回"}：${selectedId.value}`);
        await loadDetail(selectedId.value);
        await loadList();
        if (result?.continuation_run?.request_id) {
          router.push(`/runs/${encodeURIComponent(result.continuation_run.request_id)}`);
        }
      } catch (error) {
        if (String(error.message).includes("409")) {
          ElMessage.warning("该审批已是终态（409 冲突），无需重复操作，已刷新最新状态。");
          await loadDetail(selectedId.value);
          await loadList();
        } else {
          ElMessage.error(`操作失败：${error.message}`);
        }
      } finally {
        acting.value = false;
      }
    }

    // ---------- 联动 ----------
    watch(filter, () => {
      // 切筛选后若当前选中项不在视图内，不强制切换，仅保留详情。
    });

    // 管理员切换成员筛选时重新加载列表
    watch(memberFilter, () => {
      loadList();
    });

    // SSE 收到新审批事件时自动刷新列表
    watch(
      () => store.reviewEventAt,
      () => {
        if (store.reviewEventConnected) loadList();
      },
    );

    onMounted(async () => {
      await loadMembers();
      await loadList();
      const qid = route.query.reviewId;
      if (qid) {
        loadDetail(String(qid));
      } else if (filteredItems.value.length) {
        loadDetail(filteredItems.value[0].id);
      }
    });

    autoRefresh(loadList);

    function openRun(requestId) {
      router.push(`/runs/${encodeURIComponent(requestId)}`);
    }

    // 关系区点击跳转：来源审批 → 就地加载；续跑 request → 运行详情页。
    function onRelationClick(label, value) {
      if (label === "来源审批") loadDetail(value);
      else if (label === "续跑 request") openRun(value);
    }

    return {
      store, isAdmin, members, memberFilter, memberName,
      filter, filteredItems, listLoading, loadList,
      detail, detailLoading, selectedId, loadDetail, selectRow,
      note, showContinuation, contAgent, contInput, contStructured, contRequestId,
      terminal, relationRows, acting, act, openRun, onRelationClick, contCollapse,
    };
  },
  template: `
    <div class="page-inner">
      <el-alert
        v-if="store.reviewEvent"
        :type="store.reviewEventConnected ? 'warning' : 'info'"
        :closable="false"
        style="margin-bottom: 16px;"
      >
        <span class="mono">
          审批事件（{{ store.reviewEventAt }}）：{{ store.reviewEvent }}
          <el-tag v-if="!store.reviewEventConnected" size="small" type="info" style="margin-left:8px;">SSE 已断开</el-tag>
        </span>
      </el-alert>

      <div class="two-col" style="align-items: flex-start;">
        <el-card shadow="never" class="block-card">
          <template #header>
            <div class="card-header">
              <span class="title">审批任务<span class="api-pill">GET /runtime/reviews</span></span>
            <div style="display: flex; align-items: center; gap: 8px;">
              <el-select
                v-if="isAdmin"
                v-model="memberFilter"
                size="small"
                clearable
                placeholder="全部成员"
                style="width: 150px;"
              >
                <el-option v-for="m in members" :key="m.id"
                  :label="m.display_name || m.username" :value="m.id"></el-option>
              </el-select>
              <el-radio-group v-model="filter" size="small">
                <el-radio-button value="pending">待处理</el-radio-button>
                <el-radio-button value="approved">已批准</el-radio-button>
                <el-radio-button value="rejected">已驳回</el-radio-button>
                <el-radio-button value="all">全部</el-radio-button>
              </el-radio-group>
            </div>
            </div>
          </template>
          <el-table
            :data="filteredItems"
            v-loading="listLoading"
            size="small"
            highlight-current-row
            class="clickable-table"
            @row-click="selectRow"
          >
            <el-table-column label="Review ID" min-width="180">
              <template #default="{ row }"><span class="mono">{{ row.id }}</span></template>
            </el-table-column>
            <el-table-column v-if="isAdmin" label="提交人" width="110">
              <template #default="{ row }">{{ memberName(row.user_id) }}</template>
            </el-table-column>
            <el-table-column label="状态" width="130">
              <template #default="{ row }"><status-tag :value="row.status"></status-tag></template>
            </el-table-column>
            <el-table-column label="Agent" width="150">
              <template #default="{ row }">{{ row.agent_id || "-" }}</template>
            </el-table-column>
            <el-table-column label="创建时间" width="170">
              <template #default="{ row }"><span class="hint-text">{{ row.created_at || "-" }}</span></template>
            </el-table-column>
            <template #empty>当前筛选下暂无审批任务</template>
          </el-table>
        </el-card>

        <el-card shadow="never" class="block-card" v-loading="detailLoading">
          <template #header>
            <div class="card-header">
              <span class="title">
                审批操作台
                <span class="api-pill">GET/POST /runtime/reviews/{id}</span>
              </span>
              <status-tag v-if="detail" :value="detail.status"></status-tag>
            </div>
          </template>

          <el-empty v-if="!detail" description="从左侧选择一条审批任务"></el-empty>

          <template v-else>
            <el-descriptions :column="2" size="small" border style="margin-bottom: 16px;">
              <el-descriptions-item label="review_id" :span="2">
                <span class="mono">{{ detail.id }}</span>
              </el-descriptions-item>
              <el-descriptions-item label="agent_id">{{ detail.agent_id || "-" }}</el-descriptions-item>
              <el-descriptions-item label="状态">
                <status-tag :value="detail.status"></status-tag>
              </el-descriptions-item>
              <el-descriptions-item v-if="isAdmin" label="提交人">{{ memberName(detail.user_id) }}</el-descriptions-item>
              <el-descriptions-item label="decided_by">{{ detail.decided_by || "-" }}</el-descriptions-item>
              <el-descriptions-item label="decided_at">{{ detail.decided_at || "-" }}</el-descriptions-item>
            </el-descriptions>

            <el-divider content-position="left">审批关系</el-divider>
            <div class="kv-list" style="margin-bottom: 16px;">
              <div v-for="([label, value], i) in relationRows" :key="i">
                <strong>{{ label }}:</strong>
                <span v-if="value && (label === '来源审批' || label === '续跑 request')" class="mono link-text" @click="onRelationClick(label, value)">{{ value }}</span>
                <span v-else class="mono">-</span>
              </div>
            </div>

            <template v-if="!isAdmin">
              <el-divider content-position="left">决策</el-divider>
              <el-form label-width="140px" size="small">
                <el-form-item label="决策备注 note">
                  <el-input v-model="note" type="textarea" :rows="2" placeholder="批准/驳回理由（写入 decision_note）"></el-input>
                </el-form-item>
                <el-form-item label="操作人">
                  <span class="hint-text">{{ store.currentUser?.display_name || store.currentUser?.username || "-" }}（当前登录用户）</span>
                </el-form-item>
              </el-form>

              <el-collapse style="margin: 8px 0 16px;" v-model="contCollapse">
                <el-collapse-item name="cont">
                  <template #title>
                    续跑配置（批准时生效，可选）
                    <el-checkbox v-model="showContinuation" style="margin-left: 8px;" @click.stop>启用</el-checkbox>
                  </template>
                  <el-form label-width="140px" size="small">
                    <el-form-item label="agent_id">
                      <el-input v-model="contAgent" placeholder="续跑起始 agent"></el-input>
                    </el-form-item>
                    <el-form-item label="input_text">
                      <el-input v-model="contInput" type="textarea" :rows="2"></el-input>
                    </el-form-item>
                    <el-form-item label="structured_input">
                      <el-input v-model="contStructured" type="textarea" :rows="4" class="mono"></el-input>
                    </el-form-item>
                    <el-form-item label="request_id">
                      <el-input v-model="contRequestId" placeholder="自定义续跑 request_id（可选）"></el-input>
                    </el-form-item>
                  </el-form>
                </el-collapse-item>
              </el-collapse>

              <div style="display: flex; gap: 8px;">
                <el-button
                  type="success"
                  :disabled="terminal"
                  :loading="acting"
                  @click="act('approve')"
                >批准{{ showContinuation ? "并续跑" : "" }}</el-button>
                <el-button type="danger" :disabled="terminal" :loading="acting" @click="act('reject')">驳回</el-button>
                <el-button text :disabled="terminal" @click="loadDetail(selectedId)">刷新状态</el-button>
              </div>
              <div v-if="terminal" class="hint-text" style="margin-top: 8px;">
                当前为终态（{{ detail.status }}），按钮已禁用，避免重复提交触发 409。
              </div>
            </template>
            <template v-else>
              <el-divider content-position="left">审批结果（管理员只读）</el-divider>
              <el-alert type="info" :closable="false" style="margin-bottom: 8px;">
                管理员仅查看审批通过/驳回的结果与详情；批准与驳回由提交人本人操作。
              </el-alert>
              <el-button text :disabled="terminal" @click="loadDetail(selectedId)">刷新状态</el-button>
            </template>

            <el-divider content-position="left">原始 payload</el-divider>
            <json-block :value="detail.payload"></json-block>
          </template>
        </el-card>
      </div>
    </div>
  `,
};
