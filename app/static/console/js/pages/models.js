// 模型池管理页（仅管理员）：配置成员与管理员在对话页可选的模型。
//
// 密钥不落库：provider_key 只是 configs/application.toml 里 [llm.providers] 的名称，
// api_base / api_key 由服务端配置持有，因此列表可以直接展示。

const { ref, reactive, computed, onMounted } = window.Vue;
const { ElMessage, ElMessageBox } = window.ElementPlus;

import { getModels, createModel, updateModel, deleteModel } from "../api.js";
import { autoRefresh } from "../hooks.js";

const EMPTY_FORM = {
  name: "",
  providerKey: "",
  model: "",
  description: "",
  temperature: 0.2,
  maxTokens: null,
  inputPerMillion: 0,
  outputPerMillion: 0,
  enabled: true,
  isDefault: false,
  sortOrder: 0,
};

export default {
  name: "ModelsPage",
  setup() {
    const items = ref([]);
    const listLoading = ref(false);

    const providerKeys = computed(() =>
      [...new Set(items.value.map((item) => item.provider_key))].sort(),
    );

    async function loadList() {
      listLoading.value = true;
      try {
        const payload = await getModels(false);
        items.value = payload.items || [];
      } catch (error) {
        ElMessage.error(`模型池加载失败：${error.message}`);
      } finally {
        listLoading.value = false;
      }
    }

    onMounted(loadList);
    autoRefresh(loadList, 10000);

    // ---------- 新建 / 编辑 ----------
    const dialogVisible = ref(false);
    const saving = ref(false);
    const editingId = ref("");
    const form = reactive({ ...EMPTY_FORM });

    function resetForm() {
      Object.assign(form, EMPTY_FORM);
    }

    function openCreate() {
      editingId.value = "";
      resetForm();
      dialogVisible.value = true;
    }

    function openEdit(row) {
      editingId.value = row.id;
      Object.assign(form, {
        name: row.name,
        providerKey: row.provider_key,
        model: row.model,
        description: row.description || "",
        temperature: row.temperature,
        maxTokens: row.max_tokens,
        inputPerMillion: row.input_per_million,
        outputPerMillion: row.output_per_million,
        enabled: Boolean(row.enabled),
        isDefault: Boolean(row.is_default),
        sortOrder: row.sort_order,
      });
      dialogVisible.value = true;
    }

    function toPayload() {
      return {
        name: form.name.trim(),
        provider_key: form.providerKey.trim(),
        model: form.model.trim(),
        description: form.description.trim(),
        temperature: Number(form.temperature) || 0,
        // 空值显式提交为 null，用于清空此前设置的 token 上限。
        max_tokens: form.maxTokens === null || form.maxTokens === "" ? null : Number(form.maxTokens),
        input_per_million: Number(form.inputPerMillion) || 0,
        output_per_million: Number(form.outputPerMillion) || 0,
        enabled: Boolean(form.enabled),
        is_default: Boolean(form.isDefault),
        sort_order: Number(form.sortOrder) || 0,
      };
    }

    async function submit() {
      const payload = toPayload();
      if (!payload.name) return ElMessage.warning("请填写模型显示名");
      if (!payload.provider_key) return ElMessage.warning("请填写 provider_key");
      if (!payload.model) return ElMessage.warning("请填写模型名");
      saving.value = true;
      try {
        if (editingId.value) {
          await updateModel(editingId.value, payload);
          ElMessage.success(`模型 ${payload.name} 已更新`);
        } else {
          await createModel(payload);
          ElMessage.success(`模型 ${payload.name} 已创建`);
        }
        dialogVisible.value = false;
        await loadList();
      } catch (error) {
        ElMessage.error(`保存失败：${error.message}`);
      } finally {
        saving.value = false;
      }
    }

    async function toggleEnabled(row) {
      try {
        await updateModel(row.id, { enabled: !row.enabled });
        await loadList();
      } catch (error) {
        ElMessage.error(`状态更新失败：${error.message}`);
      }
    }

    async function makeDefault(row) {
      try {
        await updateModel(row.id, { is_default: true });
        ElMessage.success(`${row.name} 已设为默认模型`);
        await loadList();
      } catch (error) {
        ElMessage.error(`设置默认失败：${error.message}`);
      }
    }

    async function remove(row) {
      try {
        await ElMessageBox.confirm(
          `确定删除模型 ${row.name}？已使用该模型的会话不受影响，但之后无法再选择。`,
          "删除确认",
          { type: "warning", confirmButtonText: "删除", cancelButtonText: "取消" },
        );
      } catch {
        return;
      }
      try {
        await deleteModel(row.id);
        ElMessage.success(`模型 ${row.name} 已删除`);
        await loadList();
      } catch (error) {
        ElMessage.error(`删除失败：${error.message}`);
      }
    }

    return {
      items, listLoading, loadList, providerKeys,
      dialogVisible, saving, editingId, form,
      openCreate, openEdit, submit, toggleEnabled, makeDefault, remove,
    };
  },
  template: `
    <div class="page-inner">
      <el-card shadow="never" class="block-card">
        <template #header>
          <div class="card-header">
            <span class="title">模型池<span class="api-pill">GET /models</span></span>
            <div style="display: flex; gap: 8px;">
              <el-button text size="small" @click="loadList()">刷新</el-button>
              <el-button type="primary" size="small" @click="openCreate">新增模型</el-button>
            </div>
          </div>
        </template>
        <el-table :data="items" v-loading="listLoading" size="small">
          <el-table-column label="显示名" min-width="150">
            <template #default="{ row }">
              <span>{{ row.name }}</span>
              <el-tag v-if="row.is_default" size="small" type="success" effect="plain" style="margin-left: 6px;">默认</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="provider_key" width="130">
            <template #default="{ row }"><span class="mono">{{ row.provider_key }}</span></template>
          </el-table-column>
          <el-table-column label="模型" min-width="180">
            <template #default="{ row }"><span class="mono">{{ row.model }}</span></template>
          </el-table-column>
          <el-table-column label="温度" width="70">
            <template #default="{ row }">{{ row.temperature }}</template>
          </el-table-column>
          <el-table-column label="最大 tokens" width="110">
            <template #default="{ row }">{{ row.max_tokens === null ? "不限" : row.max_tokens }}</template>
          </el-table-column>
          <el-table-column label="价格/百万 tokens" width="170">
            <template #default="{ row }">
              <span class="hint-text">入 {{ row.input_per_million }} / 出 {{ row.output_per_million }}</span>
            </template>
          </el-table-column>
          <el-table-column label="排序" width="70">
            <template #default="{ row }">{{ row.sort_order }}</template>
          </el-table-column>
          <el-table-column label="状态" width="90">
            <template #default="{ row }">
              <el-tag :type="row.enabled ? 'success' : 'danger'" size="small">
                {{ row.enabled ? "启用" : "停用" }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="220" fixed="right">
            <template #default="{ row }">
              <el-button text size="small" type="primary" @click="openEdit(row)">编辑</el-button>
              <el-button text size="small" @click="toggleEnabled(row)">{{ row.enabled ? "停用" : "启用" }}</el-button>
              <el-button text size="small" :disabled="row.is_default" @click="makeDefault(row)">设为默认</el-button>
              <el-button text size="small" type="danger" @click="remove(row)">删除</el-button>
            </template>
          </el-table-column>
          <template #empty>暂无模型，请先新增</template>
        </el-table>
        <div class="hint-text" style="margin-top: 8px;">
          成员在对话页只能看到「启用」的模型；未显式选择时使用「默认」模型。
          密钥不落库：provider_key 对应 configs/application.toml 的 [llm.providers] 名称。
        </div>
      </el-card>

      <el-dialog
        v-model="dialogVisible"
        :title="editingId ? '编辑模型' : '新增模型'"
        width="560px"
      >
        <el-form label-position="top">
          <el-form-item label="显示名" required>
            <el-input v-model="form.name" placeholder="对话页模型选择器中的名称，需唯一"></el-input>
          </el-form-item>
          <el-form-item label="provider_key" required>
            <el-input v-model="form.providerKey" placeholder="[llm.providers] 中的名称，如 deepseek"></el-input>
            <div class="hint-text" v-if="providerKeys.length">
              已有：{{ providerKeys.join("、") }}
            </div>
          </el-form-item>
          <el-form-item label="模型名" required>
            <el-input v-model="form.model" placeholder="传给 provider 的模型标识，如 deepseek/deepseek-chat"></el-input>
          </el-form-item>
          <el-form-item label="说明">
            <el-input v-model="form.description" placeholder="可选，供成员理解适用场景"></el-input>
          </el-form-item>
          <el-form-item label="温度">
            <el-input-number v-model="form.temperature" :min="0" :max="2" :step="0.1" :precision="2"></el-input-number>
          </el-form-item>
          <el-form-item label="最大 tokens">
            <el-input-number v-model="form.maxTokens" :min="1" :step="128" placeholder="留空表示不限制"></el-input-number>
          </el-form-item>
          <el-form-item label="输入价格（每百万 tokens）">
            <el-input-number v-model="form.inputPerMillion" :min="0" :step="0.1" :precision="4"></el-input-number>
          </el-form-item>
          <el-form-item label="输出价格（每百万 tokens）">
            <el-input-number v-model="form.outputPerMillion" :min="0" :step="0.1" :precision="4"></el-input-number>
          </el-form-item>
          <el-form-item label="排序">
            <el-input-number v-model="form.sortOrder" :min="0" :step="1"></el-input-number>
          </el-form-item>
          <el-form-item label="启用">
            <el-switch v-model="form.enabled" active-text="启用" inactive-text="停用"></el-switch>
          </el-form-item>
          <el-form-item label="设为默认">
            <el-switch v-model="form.isDefault" active-text="默认" inactive-text="非默认"></el-switch>
          </el-form-item>
        </el-form>
        <template #footer>
          <el-button @click="dialogVisible = false">取消</el-button>
          <el-button type="primary" :loading="saving" @click="submit">保存</el-button>
        </template>
      </el-dialog>
    </div>
  `,
};
