// 自动化 Agent 页：定时（beat 驱动）自动化 Agent 只读清单，双角色可见。

const { ref } = window.Vue;
const { ElMessage } = window.ElementPlus;

import { getAutomationAgents } from "../api.js";
import { autoRefresh } from "../hooks.js";

export default {
  name: "AutomationPage",
  setup() {
    const agents = ref([]);
    const loading = ref(false);

    async function refresh() {
      loading.value = true;
      try {
        agents.value = (await getAutomationAgents()) || [];
      } catch (error) {
        ElMessage.error(`自动化 Agent 加载失败：${error.message}`);
      } finally {
        loading.value = false;
      }
    }

    // 清单变化极低频：自动轮询放宽到 1 小时，页内刷新键可随时手动刷新。
    autoRefresh(refresh, 60 * 60 * 1000);

    return { agents, loading, refresh };
  },
  template: `
    <div class="page-inner" v-loading="loading">
      <el-card shadow="never" class="block-card">
        <template #header>
          <div class="card-header">
            <span class="title">自动化 Agent<span class="api-pill">GET /automation/agents</span></span>
            <div style="display: flex; align-items: center; gap: 8px;">
              <el-tag type="info" size="small" effect="light" round>{{ agents.length }} 个已配置</el-tag>
              <el-button size="small" text :loading="loading" @click="refresh">刷新</el-button>
            </div>
          </div>
        </template>

        <el-alert
          type="info"
          :closable="false"
          style="margin-bottom: 12px;"
          title="以下为系统定时调度的自动化 Agent（非人员使用的交互 Agent），由 Celery beat 按计划自动执行，此处只读展示。"
        ></el-alert>

        <el-table :data="agents" stripe>
          <el-table-column prop="name" label="名称" min-width="170">
            <template #default="{ row }">
              <strong>{{ row.name }}</strong>
            </template>
          </el-table-column>
          <el-table-column prop="id" label="ID" min-width="180">
            <template #default="{ row }">
              <span class="mono">{{ row.id }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="description" label="描述" min-width="320"></el-table-column>
          <el-table-column prop="schedule" label="定时规则" min-width="170"></el-table-column>
          <el-table-column prop="business" label="归属业务" min-width="110">
            <template #default="{ row }">
              <span class="mono">{{ row.business }}</span>
            </template>
          </el-table-column>
          <el-table-column label="状态" width="90">
            <template #default="{ row }">
              <el-tag :type="row.enabled ? 'success' : 'info'" size="small" effect="light">
                {{ row.enabled ? "已启用" : "已停用" }}
              </el-tag>
            </template>
          </el-table-column>
        </el-table>
      </el-card>
    </div>
  `,
};
