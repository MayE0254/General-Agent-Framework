// 公共展示组件：状态标签、JSON 块、卡片头、资源卡。

const { h } = window.Vue;

/** 工作流/任务/审批状态 → Element Plus tag 类型映射 */
export function statusTagType(value) {
  const normalized = String(value ?? "unknown").toLowerCase();
  if (["success", "completed", "approved", "ready"].includes(normalized)) return "success";
  if (["failed", "failure", "rejected", "offline", "error"].includes(normalized)) return "danger";
  if (["pending", "started", "running", "needs_human_review"].includes(normalized)) return "warning";
  return "info";
}

export const StatusTag = {
  name: "StatusTag",
  props: { value: { type: [String, Boolean], default: "unknown" } },
  computed: {
    normalized() {
      return String(this.value ?? "unknown").toLowerCase();
    },
  },
  template: `
    <el-tag :type="statusTagType(normalized)" size="small" effect="light" round>
      {{ normalized }}
    </el-tag>
  `,
  setup() {
    return { statusTagType };
  },
};

/** JSON 展示块：统一格式化 + 溢出滚动 */
export const JsonBlock = {
  name: "JsonBlock",
  props: {
    value: { type: [Object, Array, String, Number, null], default: () => ({}) },
    emptyText: { type: String, default: "暂无数据" },
  },
  computed: {
    pretty() {
      if (this.value === null || this.value === undefined) return this.emptyText;
      if (typeof this.value === "string") return this.value || this.emptyText;
      const text = JSON.stringify(this.value, null, 2);
      return text === "[]" || text === "{}" ? this.emptyText : text;
    },
  },
  template: `<pre class="json-pre">{{ pretty }}</pre>`,
};

/** 卡片头：标题 + API 说明 + 右侧操作插槽 */
export const CardHeader = {
  name: "CardHeader",
  props: { title: String, api: String },
  template: `
    <div class="card-header">
      <div class="title">
        <span>{{ title }}</span>
        <span v-if="api" class="api-pill">{{ api }}</span>
      </div>
      <div><slot name="extra"></slot></div>
    </div>
  `,
};

/** 赎源状态卡（Database / Redis / LLM） */
export const ResourceCard = {
  name: "ResourceCard",
  components: { StatusTag },
  props: {
    title: String,
    status: String,
    metric: String,
    hint: String,
    details: { type: Array, default: () => [] }, // [ [label, value], ... ]
  },
  template: `
    <el-card shadow="never" class="block-card">
      <template #header>
        <div class="card-header">
          <span class="title">{{ title }}</span>
          <status-tag :value="status"></status-tag>
        </div>
      </template>
      <div class="resource-metric mono">{{ metric || "-" }}</div>
      <div class="hint-text" style="margin-bottom: 8px;">{{ hint || "" }}</div>
      <div class="kv-list" v-if="details && details.length">
        <div v-for="([k, v], i) in details" :key="i">
          <strong>{{ k }}:</strong> <span class="mono">{{ v }}</span>
        </div>
      </div>
    </el-card>
  `,
};

/** 便捷跳转链接（mono 样式的可点击 id） */
export const LinkText = {
  name: "LinkText",
  props: { value: { type: String, default: "" } },
  emits: ["click"],
  template: `
    <span class="mono" style="cursor: pointer; color: var(--el-color-primary);"
      @click="$emit('click', value)">{{ value }}</span>
  `,
};

export function short(value, limit = 48) {
  const text = String(value ?? "");
  return text.length > limit ? `${text.slice(0, limit)}...` : text;
}

export function installComponents(app) {
  for (const comp of [StatusTag, JsonBlock, CardHeader, ResourceCard, LinkText]) {
    app.component(comp.name, comp);
  }
}

export { h };
