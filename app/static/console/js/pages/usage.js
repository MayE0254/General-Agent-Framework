// 用量统计页：按用户 / 按模型 / 按用户+模型 汇总 token 与费用。
//
// 成员只能看到自己的数据（后端强制以当前登录用户覆盖 user_id），
// 因此筛选项仅对管理员展示；数据全部来自 llm_call_logs 审计表。

const { ref, computed, onMounted } = window.Vue;
const { ElMessage } = window.ElementPlus;

import { getLlmUsage, getUsers } from "../api.js";
import { store } from "../store.js";
import { autoRefresh } from "../hooks.js";

function formatNumber(value) {
  return Number(value || 0).toLocaleString("zh-CN");
}

function formatCost(value) {
  const amount = Number(value || 0);
  return `$${amount.toFixed(4)}`;
}

export default {
  name: "UsagePage",
  setup() {
    const loading = ref(false);
    const report = ref(null);
    const members = ref([]);

    // 日期区间用 el-date-picker 的 datetimerange，value-format 直接产出
    // 后端可解析的 ISO 字符串（不做时区换算，按本地时间记账）。
    const filters = ref({ userId: "", model: "", range: null });

    const isAdmin = computed(() => store.currentUser?.role === "admin");

    const totals = computed(() => report.value?.totals || null);
    const byUser = computed(() => report.value?.by_user || []);
    const byModel = computed(() => report.value?.by_model || []);
    const byUserModel = computed(() => report.value?.by_user_model || []);

    const cards = computed(() => {
      const t = totals.value;
      if (!t) return [];
      return [
        { label: "总 tokens", value: formatNumber(t.total_tokens), hint: `输入 ${formatNumber(t.prompt_tokens)} / 输出 ${formatNumber(t.completion_tokens)}` },
        { label: "调用次数", value: formatNumber(t.call_count), hint: `失败 ${formatNumber(t.failed_call_count)} 次` },
        { label: "累计费用", value: formatCost(t.cost_usd), hint: "按模型池配置的单价折算" },
        { label: "覆盖用户", value: formatNumber(byUser.value.length), hint: `涉及模型 ${byModel.value.length} 个` },
      ];
    });

    function buildQuery() {
      const query = {};
      if (isAdmin.value) {
        if (filters.value.userId) query.user_id = filters.value.userId;
      }
      if (isAdmin.value && filters.value.model) query.model = filters.value.model;
      const range = filters.value.range;
      if (range && range.length === 2) {
        query.created_from = range[0];
        query.created_to = range[1];
      }
      return query;
    }

    async function loadReport() {
      loading.value = true;
      try {
        report.value = await getLlmUsage(buildQuery());
      } catch (error) {
        ElMessage.error(`用量统计加载失败：${error.message}`);
      } finally {
        loading.value = false;
      }
    }

    function resetFilters() {
      filters.value = { userId: "", model: "", range: null };
      loadReport();
    }

    onMounted(async () => {
      if (isAdmin.value) {
        // 成员下拉仅管理员可用；失败不影响统计主体。
        try {
          const payload = await getUsers();
          members.value = payload.items || payload || [];
        } catch {
          members.value = [];
        }
      }
      await loadReport();
    });
    autoRefresh(loadReport, 15000);

    return {
      isAdmin, loading, filters, members, cards,
      totals, byUser, byModel, byUserModel,
      loadReport, resetFilters, formatNumber, formatCost,
    };
  },
  template: `
    <div class="page-inner">
      <el-card shadow="never" class="block-card">
        <template #header>
          <div class="card-header">
            <span class="title">用量统计<span class="api-pill">GET /runtime/usage/llm</span></span>
            <div style="display: flex; gap: 8px;">
              <el-button text size="small" @click="loadReport()">刷新</el-button>
            </div>
          </div>
        </template>
        <div v-if="isAdmin" style="display: flex; gap: 12px; flex-wrap: wrap; align-items: center;">
          <el-select v-model="filters.userId" filterable clearable placeholder="全部用户" style="width: 200px;">
            <el-option
              v-for="member in members"
              :key="member.id"
              :label="member.display_name || member.username"
              :value="member.id"
            ></el-option>
          </el-select>
          <el-input v-model="filters.model" clearable placeholder="模型名（可留空）" style="width: 240px;"></el-input>
          <el-date-picker
            v-model="filters.range"
            type="datetimerange"
            value-format="YYYY-MM-DDTHH:mm:ss"
            range-separator="至"
            start-placeholder="开始时间"
            end-placeholder="结束时间"
          ></el-date-picker>
          <el-button type="primary" size="small" @click="loadReport()">查询</el-button>
          <el-button size="small" @click="resetFilters">重置</el-button>
        </div>
        <div v-else class="hint-text">当前为用户视图，仅展示自己的调用记录。</div>
      </el-card>

      <div class="three-col" style="margin-top: 16px;">
        <el-card v-for="card in cards" :key="card.label" shadow="never" class="block-card">
          <div class="hint-text">{{ card.label }}</div>
          <div style="font-size: 22px; font-weight: 600; margin: 6px 0;">{{ card.value }}</div>
          <div class="hint-text">{{ card.hint }}</div>
        </el-card>
      </div>

      <el-card shadow="never" class="block-card" style="margin-top: 16px;">
        <template #header>
          <div class="card-header"><span class="title">按用户</span></div>
        </template>
        <el-table :data="byUser" v-loading="loading" size="small">
          <el-table-column label="用户" min-width="200">
            <template #default="{ row }">
              <span class="mono">{{ row.key || "（未记录用户）" }}</span>
            </template>
          </el-table-column>
          <el-table-column label="调用次数" width="100">
            <template #default="{ row }">{{ formatNumber(row.call_count) }}</template>
          </el-table-column>
          <el-table-column label="失败次数" width="100">
            <template #default="{ row }">{{ formatNumber(row.failed_call_count) }}</template>
          </el-table-column>
          <el-table-column label="输入 tokens" width="130">
            <template #default="{ row }">{{ formatNumber(row.prompt_tokens) }}</template>
          </el-table-column>
          <el-table-column label="输出 tokens" width="130">
            <template #default="{ row }">{{ formatNumber(row.completion_tokens) }}</template>
          </el-table-column>
          <el-table-column label="总 tokens" width="130">
            <template #default="{ row }">{{ formatNumber(row.total_tokens) }}</template>
          </el-table-column>
          <el-table-column label="费用" width="120">
            <template #default="{ row }">{{ formatCost(row.cost_usd) }}</template>
          </el-table-column>
          <template #empty>暂无调用记录</template>
        </el-table>
      </el-card>

      <el-card shadow="never" class="block-card" style="margin-top: 16px;">
        <template #header>
          <div class="card-header"><span class="title">按模型</span></div>
        </template>
        <el-table :data="byModel" v-loading="loading" size="small">
          <el-table-column label="模型" min-width="240">
            <template #default="{ row }"><span class="mono">{{ row.key || "（未记录模型）" }}</span></template>
          </el-table-column>
          <el-table-column label="调用次数" width="100">
            <template #default="{ row }">{{ formatNumber(row.call_count) }}</template>
          </el-table-column>
          <el-table-column label="失败次数" width="100">
            <template #default="{ row }">{{ formatNumber(row.failed_call_count) }}</template>
          </el-table-column>
          <el-table-column label="输入 tokens" width="130">
            <template #default="{ row }">{{ formatNumber(row.prompt_tokens) }}</template>
          </el-table-column>
          <el-table-column label="输出 tokens" width="130">
            <template #default="{ row }">{{ formatNumber(row.completion_tokens) }}</template>
          </el-table-column>
          <el-table-column label="总 tokens" width="130">
            <template #default="{ row }">{{ formatNumber(row.total_tokens) }}</template>
          </el-table-column>
          <el-table-column label="费用" width="120">
            <template #default="{ row }">{{ formatCost(row.cost_usd) }}</template>
          </el-table-column>
          <template #empty>暂无调用记录</template>
        </el-table>
      </el-card>

      <el-card shadow="never" class="block-card" style="margin-top: 16px;">
        <template #header>
          <div class="card-header">
            <span class="title">按用户 + 模型</span>
            <span class="hint-text">回答「谁在哪个模型上花了多少」</span>
          </div>
        </template>
        <el-table :data="byUserModel" v-loading="loading" size="small">
          <el-table-column label="用户" min-width="180">
            <template #default="{ row }"><span class="mono">{{ row.user_id || "（未记录用户）" }}</span></template>
          </el-table-column>
          <el-table-column label="模型" min-width="200">
            <template #default="{ row }"><span class="mono">{{ row.model || "（未记录模型）" }}</span></template>
          </el-table-column>
          <el-table-column label="调用次数" width="100">
            <template #default="{ row }">{{ formatNumber(row.call_count) }}</template>
          </el-table-column>
          <el-table-column label="失败次数" width="100">
            <template #default="{ row }">{{ formatNumber(row.failed_call_count) }}</template>
          </el-table-column>
          <el-table-column label="输入 tokens" width="130">
            <template #default="{ row }">{{ formatNumber(row.prompt_tokens) }}</template>
          </el-table-column>
          <el-table-column label="输出 tokens" width="130">
            <template #default="{ row }">{{ formatNumber(row.completion_tokens) }}</template>
          </el-table-column>
          <el-table-column label="总 tokens" width="130">
            <template #default="{ row }">{{ formatNumber(row.total_tokens) }}</template>
          </el-table-column>
          <el-table-column label="费用" width="120">
            <template #default="{ row }">{{ formatCost(row.cost_usd) }}</template>
          </el-table-column>
          <template #empty>暂无调用记录</template>
        </el-table>
      </el-card>
    </div>
  `,
};
