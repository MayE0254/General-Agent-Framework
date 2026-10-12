// 系统状态页：Celery Worker 在线状态 + 资源接线全量细节（Database / Redis / LLM）。

const { ref, computed } = window.Vue;
const { ElMessage } = window.ElementPlus;

import { getResources, getWorkers, recoverWorkers } from "../api.js";
import { autoRefresh } from "../hooks.js";

export default {
  name: "SystemPage",
  setup() {
    const resources = ref(null);
    const snapshot = ref(null);
    const loading = ref(false);
    const recovering = ref(false);

    const workers = computed(() => snapshot.value?.workers || []);
    const heartbeats = computed(() => snapshot.value?.heartbeats || []);
    const broker = computed(() => snapshot.value?.broker || null);
    const watchdog = computed(() => snapshot.value?.watchdog || null);
    // 看板顶部状态以心跳+积压为准：inspect 对“卡死但仍占着消息”的 worker 是盲区。
    const consumerStatus = computed(() => snapshot.value?.status || "unknown");
    const consumerMessage = computed(() => snapshot.value?.message || "");
    // statusTagType 不认识 stalled，这里显式映射，避免把“停滞”显示成中性色。
    const consumerTagType = computed(() => {
      if (consumerStatus.value === "ready") return "success";
      if (consumerStatus.value === "stalled" || consumerStatus.value === "offline") return "danger";
      return "info";
    });

    function formatAge(seconds) {
      if (seconds === null || seconds === undefined) return "-";
      const total = Math.max(0, Math.round(seconds));
      if (total < 60) return `${total}s`;
      if (total < 3600) return `${Math.floor(total / 60)}m${total % 60}s`;
      return `${Math.floor(total / 3600)}h${Math.floor((total % 3600) / 60)}m`;
    }

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
            ["ProbeError", d.redis?.config?.probe_error || "-"],
            ["URL", d.redis?.config?.url || "-"],
            ["Broker", d.redis?.config?.broker_url || "-"],
            ["Backend", d.redis?.config?.result_backend || "-"],
            ["ConnectTimeout", String(d.redis?.config?.connect_timeout_seconds ?? "-")],
            ["SocketTimeout", String(d.redis?.config?.socket_timeout_seconds ?? "-")],
            ["WorkerPool", d.redis?.config?.worker_pool || "-"],
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
      loading.value = true;
      try {
        const [res, workerSnapshot] = await Promise.all([
          getResources(),
          getWorkers().catch(() => ({ workers: [] })),
        ]);
        resources.value = res;
        snapshot.value = workerSnapshot;
      } catch (error) {
        ElMessage.error(`系统状态加载失败：${error.message}`);
      } finally {
        loading.value = false;
      }
    }

    // 系统状态变化低频：自动轮询放宽到 1 小时；顶栏/页内刷新键仍可随时手动刷新。
    autoRefresh(refresh, 60 * 60 * 1000);

    // 手动恢复：探测 worker，离线时后端自动拉起新进程并等待确认上线。
    async function recover() {
      recovering.value = true;
      try {
        const result = await recoverWorkers();
        if (result?.status === "ready") {
          ElMessage.success(result.message || "Worker 已在线");
        } else {
          ElMessage.warning(result?.message || "恢复流程已执行，请检查日志");
        }
      } catch (error) {
        ElMessage.error(`恢复失败：${error.message}`);
      } finally {
        recovering.value = false;
        await refresh();
      }
    }

    return {
      resourceCards,
      workers,
      heartbeats,
      broker,
      watchdog,
      consumerStatus,
      consumerMessage,
      consumerTagType,
      formatAge,
      loading,
      recovering,
      refresh,
      recover,
    };
  },
  template: `
    <div class="page-inner" v-loading="loading">
      <el-card shadow="never" class="block-card" style="margin-bottom: 16px;">
        <template #header>
          <div class="card-header">
            <span class="title">Celery Worker<span class="api-pill">GET /system/workers</span></span>
            <div style="display: flex; align-items: center; gap: 8px;">
              <el-tag :type="consumerTagType" size="small" effect="light" round>
                {{ consumerStatus }}
              </el-tag>
              <span v-if="broker" class="mono" style="font-size: 12px; color: #909399;">
                ready={{ broker.ready }} / unacked={{ broker.unacked }}
              </span>
              <el-button size="small" type="warning" plain :loading="recovering" @click="recover">手动恢复</el-button>
              <el-button size="small" text :loading="loading" @click="refresh">刷新</el-button>
            </div>
          </div>
        </template>

        <el-alert
          v-if="consumerStatus === 'stalled' || consumerStatus === 'offline'"
          type="error"
          :closable="false"
          style="margin-bottom: 12px;"
          title="无活跃消费者"
          :description="consumerMessage"
        ></el-alert>
        <el-alert
          v-else
          type="success"
          :closable="false"
          style="margin-bottom: 12px;"
          :title="consumerMessage"
        ></el-alert>

        <div v-if="workers.length" class="worker-grid">
          <div v-for="item in workers" :key="item.worker_name" class="worker-item">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <strong class="mono">{{ item.worker_name }}</strong>
              <status-tag :value="item.online ? 'ready' : 'offline'"></status-tag>
            </div>
            <div class="kv-list">
              <div><strong>Pool:</strong> <span class="mono">{{ item.pool || "-" }}</span></div>
              <div><strong>Concurrency:</strong> <span class="mono">{{ item.max_concurrency ?? "-" }}</span></div>
              <div><strong>Queues:</strong> <span class="mono">{{ (item.queues || []).join(", ") || "-" }}</span></div>
            </div>
          </div>
        </div>
        <el-alert v-else type="warning" :closable="false">
          当前没有 worker 响应 inspect ping。若任务长期 pending，优先检查 worker 进程、队列名与 Redis 配置。
        </el-alert>

        <el-divider content-position="left">心跳（独立于 inspect 的存活信号）</el-divider>
        <el-table v-if="heartbeats.length" :data="heartbeats" size="small" border>
          <el-table-column prop="worker_id" label="Worker" min-width="180"></el-table-column>
          <el-table-column prop="pid" label="PID" width="90"></el-table-column>
          <el-table-column prop="pool" label="Pool" min-width="120"></el-table-column>
          <el-table-column label="最近心跳" width="120">
            <template #default="{ row }">{{ formatAge(row.age_seconds) }}</template>
          </el-table-column>
          <el-table-column label="状态" width="100">
            <template #default="{ row }">
              <status-tag :value="row.stale ? 'stale' : 'ready'"></status-tag>
            </template>
          </el-table-column>
        </el-table>
        <el-alert v-else type="info" :closable="false">
          没有读到任何 worker 心跳。心跳由 worker 进程内的独立线程写入，任务卡死时仍会继续刷新；
          完全没有心跳意味着没有进程在消费队列。
        </el-alert>

        <el-divider content-position="left">Broker 积压</el-divider>
        <div v-if="broker" class="kv-list">
          <div><strong>Reachable:</strong> <span class="mono">{{ String(broker.reachable) }}</span></div>
          <div><strong>Queue:</strong> <span class="mono">{{ broker.queue_name || "-" }}</span></div>
          <div><strong>Ready:</strong> <span class="mono">{{ broker.ready ?? "-" }}</span></div>
          <div><strong>Unacked:</strong> <span class="mono">{{ broker.unacked ?? "-" }}</span></div>
          <div><strong>最久未确认:</strong> <span class="mono">{{ formatAge(broker.oldest_unacked_age_seconds) }}</span></div>
          <div v-if="broker.error"><strong>Error:</strong> <span class="mono">{{ broker.error }}</span></div>
        </div>

        <el-divider content-position="left">看门狗</el-divider>
        <div v-if="watchdog" class="kv-list">
          <div><strong>运行中:</strong> <span class="mono">{{ String(watchdog.running) }}</span></div>
          <div><strong>检查次数:</strong> <span class="mono">{{ watchdog.checks ?? 0 }}</span></div>
          <div><strong>自动恢复次数:</strong> <span class="mono">{{ watchdog.recoveries ?? 0 }}</span></div>
          <div><strong>已拉起 PID:</strong> <span class="mono">{{ (watchdog.spawned_pids || []).join(", ") || "-" }}</span></div>
          <div><strong>最近动作:</strong> <span class="mono">{{ watchdog.last_action || "-" }}</span></div>
          <div><strong>最近消息:</strong> <span class="mono">{{ watchdog.last_message || "-" }}</span></div>
        </div>
        <el-alert v-else type="info" :closable="false">
          看门狗未启动（它随 API 进程生命周期运行）。
        </el-alert>
      </el-card>

      <el-divider content-position="left">资源接线</el-divider>
      <div class="three-col" v-if="resourceCards.length">
        <resource-card v-for="card in resourceCards" :key="card.title" v-bind="card"></resource-card>
      </div>
    </div>
  `,
};
