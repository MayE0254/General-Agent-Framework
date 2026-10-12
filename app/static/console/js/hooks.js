// 页面级组合函数：统一处理自动刷新。
// - autoRefresh(fn, intervalMs)：挂载后立即执行一次，随后定时轮询；
//   页面不可见时暂停轮询，恢复可见时立刻补一次；
//   顶栏"立即刷新"（store.refreshTick 变化）也会触发。

const { watch, onUnmounted } = window.Vue;
import { store } from "./store.js";

export function autoRefresh(fn, intervalMs = 5000) {
  let timer = null;
  let running = false;

  async function run() {
    if (running) return; // 上一次还没完成时不叠加
    running = true;
    store.refreshing = true;
    try {
      await fn();
      store.lastRefreshedAt = new Date().toLocaleTimeString();
    } catch {
      // 页面内部自行处理错误展示
    } finally {
      running = false;
      store.refreshing = false;
    }
  }

  function start() {
    if (timer) return;
    timer = setInterval(() => {
      if (document.visibilityState === "visible") run();
    }, intervalMs);
  }

  function stop() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }

  function onVisibility() {
    if (document.visibilityState === "visible") {
      run();
      start();
    } else {
      stop();
    }
  }

  document.addEventListener("visibilitychange", onVisibility);
  const stopWatch = watch(
    () => store.refreshTick,
    () => run()
  );

  run();
  start();

  onUnmounted(() => {
    stop();
    document.removeEventListener("visibilitychange", onVisibility);
    stopWatch();
  });
}
