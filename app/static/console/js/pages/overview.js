// 总览页：资源接线状态 + 审批事件横幅 + 最近工作流 / 待审批速览。

const { ref, computed } = window.Vue;
const { useRouter } = window.VueRouter;

import { getResources, getWorkflowRuns, getReviews } from "../api.js";
import { store } from "../store.js";
import { autoRefresh } from "../hooks.js";

export default {
  name: "OverviewPage",
  setup() {
    const router = useRouter();
    const resources = ref(null);
    const workflows = ref([]);
    const reviews = ref([]);

    const resourceCards = computed(() => {
      const d = resources.value;
      if (!d) return [];
      return [
        {
          title: "Database",
          status: d.database?.status,
          metric: d.database?.config?.masked_url || "-",
          hint: d.database?.message,
          details: [["SQLAlchemy", String(d.database?.config?.sqlalchemy_available)]],
        },
        {
          title: "Redis / Celery",
          status: d.redis?.status,
          metric: d.redis?.config?.task_default_queue || "-",
          hint: d.redis?.message,
          details: [
            ["Reachable", String(d.redis?.config?.reachable)],
            ["LatencyMs", String(d.redis?.config?.latency_ms ?? "-")],
            ["Broker", d.redis?.config?.broker_url || "-"],
          ],
        },
        {
          title: "LLM",
          status: d.llm?.status,
          metric: d.llm?.status || "-",
          hint: d.llm?.message,
          details: [],
        },
      ];
    });

    async function refresh() {
      const [res, wf, rv] = await Promise.all([
        // /system/resources 仅管理员可用；普通用户 403 时置 null，隐藏资源卡片区。
        getResources().catch(() => null),
        getWorkflowRuns(8).catch(() => ({ items: [] })),
        getReviews(8).catch(() => ({ items: [] })),
      ]);
      resources.value = res;
      workflows.value = wf.items || [];
      reviews.value = rv.items || [];
    }

    autoRefresh(refresh);

    function openRun(requestId) {
      router.push(`/runs/${encodeURIComponent(requestId)}`);
    }
    function openReview(reviewId) {
      router.push({ path: "/reviews", query: { reviewId } });
    }

    return { store, resourceCards, workflows, reviews, openRun, openReview };
  },
  template: `
    <div class="page-inner">
      <el-alert
        v-if="store.reviewEvent"
        type="warning"
        :closable="false"
        style="margin-bottom: 16px;"
      >
        <span class="mono">审批事件（{{ store.reviewEventAt }}）：{{ store.reviewEvent }}</span>
      </el-alert>

      <div class="three-col" v-if="resourceCards.length">
        <resource-card v-for="card in resourceCards" :key="card.title" v-bind="card"></resource-card>
      </div>

      <div class="two-col" style="margin-top: 16px;">
        <el-card shadow="never" class="block-card">
          <template #header>
            <div class="card-header">
              <span class="title">最近工作流<span class="api-pill">GET /runtime/workflow-runs</span></span>
              <el-button text size="small" @click="$router.push('/workflows')">全部</el-button>
            </div>
          </template>
          <el-table :data="workflows" size="small" class="clickable-table" @row-click="(row) => openRun(row.request_id)">
            <el-table-column label="Request" min-width="180">
              <template #default="{ row }"><span class="mono">{{ row.request_id }}</span></template>
            </el-table-column>
            <el-table-column label="Status" width="150">
              <template #default="{ row }"><status-tag :value="row.status"></status-tag></template>
            </el-table-column>
            <el-table-column label="Updated" width="170">
              <template #default="{ row }"><span class="hint-text">{{ row.updated_at || "-" }}</span></template>
            </el-table-column>
            <template #empty>暂无数据</template>
          </el-table>
        </el-card>

        <el-card shadow="never" class="block-card">
          <template #header>
            <div class="card-header">
              <span class="title">待处理审批<span class="api-pill">GET /runtime/reviews</span></span>
              <el-button text size="small" @click="$router.push('/reviews')">审批中心</el-button>
            </div>
          </template>
          <el-table :data="reviews" size="small" class="clickable-table" @row-click="(row) => openReview(row.id)">
            <el-table-column label="Review" min-width="170">
              <template #default="{ row }"><span class="mono">{{ row.id }}</span></template>
            </el-table-column>
            <el-table-column label="Status" width="130">
              <template #default="{ row }"><status-tag :value="row.status"></status-tag></template>
            </el-table-column>
            <el-table-column label="Agent" width="150">
              <template #default="{ row }">{{ row.agent_id || "-" }}</template>
            </el-table-column>
            <template #empty>暂无审批任务</template>
          </el-table>
        </el-card>
      </div>
    </div>
  `,
};
