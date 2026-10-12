// 登录页：用户名 + 密码；成功后写入 store.currentUser 并回跳。

const { ref } = window.Vue;
const { useRouter } = window.VueRouter;
const { ElMessage } = window.ElementPlus;

import { apiLogin } from "../api.js";
import { store } from "../store.js";

export default {
  name: "LoginPage",
  setup() {
    const router = useRouter();

    const form = ref({ username: "", password: "" });
    const loading = ref(false);

    async function submit() {
      const username = form.value.username.trim();
      const password = form.value.password;
      if (!username) return ElMessage.warning("请输入用户名");
      if (!password) return ElMessage.warning("请输入密码");
      loading.value = true;
      try {
        const user = await apiLogin(username, password);
        store.currentUser = user;
        store.authReady = true;
        ElMessage.success(`欢迎，${user.display_name || user.username}`);
        router.push("/");
      } catch (error) {
        const message = String(error.message || "");
        if (message.includes("401")) ElMessage.error("用户名或密码错误");
        else ElMessage.error(`登录失败：${message}`);
      } finally {
        loading.value = false;
      }
    }

    return { form, loading, submit };
  },
  template: `
    <div class="login-wrapper">
      <el-card shadow="always" class="login-card">
        <div class="login-brand">
          <el-icon :size="34" color="var(--el-color-primary)"><Platform></Platform></el-icon>
          <div class="login-title">MultiAgent 中台</div>
          <div class="login-sub">请登录以继续</div>
        </div>
        <el-form label-position="top" @submit.prevent="submit">
          <el-form-item label="用户名">
            <el-input v-model="form.username" placeholder="用户名" autofocus></el-input>
          </el-form-item>
          <el-form-item label="密码">
            <el-input v-model="form.password" type="password" show-password
              placeholder="密码" @keyup.enter="submit"></el-input>
          </el-form-item>
          <el-button type="primary" style="width: 100%;" :loading="loading" @click="submit">
            登 录
          </el-button>
        </el-form>
        <div class="hint-text" style="margin-top: 12px; text-align: center;">
          账号由管理员创建；如需开通请联系管理员。
        </div>
      </el-card>
    </div>
  `,
};
