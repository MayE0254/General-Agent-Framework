// 成员管理页（仅管理员）：成员列表 + 创建 / 编辑（角色、显示名、启停、重置密码）/ 删除。
// 后端约束：不能自降权、自禁用、自删除；此处按钮态与后端校验保持一致。

const { ref, reactive, computed, onMounted } = window.Vue;
const { ElMessage, ElMessageBox } = window.ElementPlus;

import { getUsers, createUser, updateUser, deleteUser } from "../api.js";
import { store } from "../store.js";
import { autoRefresh } from "../hooks.js";

export default {
  name: "UsersPage",
  setup() {
    const items = ref([]);
    const listLoading = ref(false);

    const selfId = computed(() => store.currentUser?.id || "");

    async function loadList() {
      listLoading.value = true;
      try {
        const payload = await getUsers();
        items.value = payload.items || [];
      } catch (error) {
        ElMessage.error(`成员列表加载失败：${error.message}`);
      } finally {
        listLoading.value = false;
      }
    }

    onMounted(loadList);
    autoRefresh(loadList);

    // ---------- 创建 ----------
    const createVisible = ref(false);
    const creating = ref(false);
    const createForm = reactive({ username: "", password: "", role: "user", displayName: "" });

    function openCreate() {
      createForm.username = "";
      createForm.password = "";
      createForm.role = "user";
      createForm.displayName = "";
      createVisible.value = true;
    }

    async function submitCreate() {
      const username = createForm.username.trim();
      if (username.length < 2) return ElMessage.warning("用户名至少 2 个字符");
      if (createForm.password.length < 6) return ElMessage.warning("密码至少 6 位");
      creating.value = true;
      try {
        const payload = { username, password: createForm.password, role: createForm.role };
        if (createForm.displayName.trim()) payload.display_name = createForm.displayName.trim();
        await createUser(payload);
        ElMessage.success(`成员 ${username} 已创建`);
        createVisible.value = false;
        await loadList();
      } catch (error) {
        ElMessage.error(`创建失败：${error.message}`);
      } finally {
        creating.value = false;
      }
    }

    // ---------- 编辑 ----------
    const editVisible = ref(false);
    const editing = ref(false);
    const editForm = reactive({ id: "", username: "", role: "user", displayName: "", isActive: true, newPassword: "" });
    const isSelf = computed(() => editForm.id && editForm.id === selfId.value);

    function openEdit(row) {
      editForm.id = row.id;
      editForm.username = row.username;
      editForm.role = row.role || "user";
      editForm.displayName = row.display_name || "";
      editForm.isActive = Boolean(row.is_active);
      editForm.newPassword = "";
      editVisible.value = true;
    }

    async function submitEdit() {
      // 自账号的 role/is_active 控件已禁用，值与原值相同，提交后端无副作用；
      // 后端另有自降权/自禁用/自删除的兜底校验。
      const payload = {
        display_name: editForm.displayName.trim() || null,
        role: editForm.role,
        is_active: editForm.isActive,
      };
      if (editForm.newPassword) {
        if (editForm.newPassword.length < 6) return ElMessage.warning("新密码至少 6 位");
        payload.password = editForm.newPassword;
      }
      editing.value = true;
      try {
        await updateUser(editForm.id, payload);
        ElMessage.success(`成员 ${editForm.username} 已更新`);
        editVisible.value = false;
        await loadList();
      } catch (error) {
        ElMessage.error(`更新失败：${error.message}`);
      } finally {
        editing.value = false;
      }
    }

    async function remove(row) {
      try {
        await ElMessageBox.confirm(
          `确定删除成员 ${row.username}？该操作不可恢复。`,
          "删除确认",
          { type: "warning", confirmButtonText: "删除", cancelButtonText: "取消" },
        );
      } catch {
        return;
      }
      try {
        await deleteUser(row.id);
        ElMessage.success(`成员 ${row.username} 已删除`);
        await loadList();
      } catch (error) {
        ElMessage.error(`删除失败：${error.message}`);
      }
    }

    function roleName(role) {
      return role === "admin" ? "管理员" : "普通用户";
    }

    return {
      items, listLoading, loadList, selfId,
      createVisible, creating, createForm, openCreate, submitCreate,
      editVisible, editing, editForm, isSelf, openEdit, submitEdit, remove, roleName,
    };
  },
  template: `
    <div class="page-inner">
      <el-card shadow="never" class="block-card">
        <template #header>
          <div class="card-header">
            <span class="title">成员列表<span class="api-pill">GET /users</span></span>
            <div style="display: flex; gap: 8px;">
              <el-button text size="small" @click="loadList()">刷新</el-button>
              <el-button type="primary" size="small" @click="openCreate">新建成员</el-button>
            </div>
          </div>
        </template>
        <el-table :data="items" v-loading="listLoading" size="small">
          <el-table-column label="用户名" min-width="140">
            <template #default="{ row }">
              <span class="mono">{{ row.username }}</span>
              <el-tag v-if="row.id === selfId" size="small" type="info" effect="plain" style="margin-left: 6px;">我</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="显示名" min-width="120">
            <template #default="{ row }">{{ row.display_name || "-" }}</template>
          </el-table-column>
          <el-table-column label="角色" width="110">
            <template #default="{ row }">
              <el-tag :type="row.role === 'admin' ? 'warning' : 'info'" size="small">{{ roleName(row.role) }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="状态" width="100">
            <template #default="{ row }">
              <el-tag :type="row.is_active ? 'success' : 'danger'" size="small">
                {{ row.is_active ? "启用" : "禁用" }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="创建时间" width="170">
            <template #default="{ row }"><span class="hint-text">{{ row.created_at || "-" }}</span></template>
          </el-table-column>
          <el-table-column label="操作" width="140" fixed="right">
            <template #default="{ row }">
              <el-button text size="small" type="primary" @click="openEdit(row)">编辑</el-button>
              <el-button text size="small" type="danger" :disabled="row.id === selfId" @click="remove(row)">删除</el-button>
            </template>
          </el-table-column>
          <template #empty>暂无成员</template>
        </el-table>
        <div class="hint-text" style="margin-top: 8px;">
          不能对自己执行删除；管理员账号不可自降权或自禁用。
        </div>
      </el-card>

      <el-dialog v-model="createVisible" title="新建成员" width="440px">
        <el-form label-position="top">
          <el-form-item label="用户名" required>
            <el-input v-model="createForm.username" placeholder="登录用户名（至少 2 个字符）"></el-input>
          </el-form-item>
          <el-form-item label="初始密码" required>
            <el-input v-model="createForm.password" type="password" show-password placeholder="至少 6 位"></el-input>
          </el-form-item>
          <el-form-item label="角色">
            <el-radio-group v-model="createForm.role">
              <el-radio value="user">普通用户</el-radio>
              <el-radio value="admin">管理员</el-radio>
            </el-radio-group>
          </el-form-item>
          <el-form-item label="显示名">
            <el-input v-model="createForm.displayName" placeholder="可选，用于界面展示"></el-input>
          </el-form-item>
        </el-form>
        <template #footer>
          <el-button @click="createVisible = false">取消</el-button>
          <el-button type="primary" :loading="creating" @click="submitCreate">创建</el-button>
        </template>
      </el-dialog>

      <el-dialog v-model="editVisible" :title="editForm.username ? '编辑成员 ' + editForm.username : '编辑成员'" width="440px">
        <el-form label-position="top">
          <el-form-item label="角色">
            <el-radio-group v-model="editForm.role" :disabled="isSelf">
              <el-radio value="user">普通用户</el-radio>
              <el-radio value="admin">管理员</el-radio>
            </el-radio-group>
          </el-form-item>
          <el-form-item label="启用状态">
            <el-switch v-model="editForm.isActive" :disabled="isSelf" active-text="启用" inactive-text="禁用"></el-switch>
          </el-form-item>
          <el-form-item label="显示名">
            <el-input v-model="editForm.displayName" placeholder="可选，用于界面展示"></el-input>
          </el-form-item>
          <el-form-item label="重置密码">
            <el-input v-model="editForm.newPassword" type="password" show-password placeholder="留空表示不修改"></el-input>
          </el-form-item>
        </el-form>
        <template #footer>
          <el-button @click="editVisible = false">取消</el-button>
          <el-button type="primary" :loading="editing" @click="submitEdit">保存</el-button>
        </template>
      </el-dialog>
    </div>
  `,
};
