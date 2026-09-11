// Axios 封装：统一 baseURL + 错误处理
// =============================================================
// baseURL 用相对路径 "/api"，配合 vite.config.js 的 proxy：
//   开发时：浏览器请求 /api/* → Vite 代理 → http://127.0.0.1:8000/api/*
//   生产时：把 dist/ 部署到任意静态服务器，要求后端 /api 在同源（或改 CORS）
// 这样前后端解耦，部署时只关心后端地址。
// =============================================================
import axios from "axios";
import { message } from "antd";

const api = axios.create({
  baseURL: "/api",
  timeout: 600_000,        // 上传/embedding 可能较慢，给 10 分钟
  headers: { "Content-Type": "application/json" },
});

// 响应拦截：统一把后端的 4xx/5xx 错误吐成 message.error
// 业务层仍可读 err.response.data 拿到结构化错误
api.interceptors.response.use(
  (r) => r,
  (err) => {
    const status = err?.response?.status;
    const detail = err?.response?.data?.detail || err?.message || "请求失败";
    // 404 不弹全局错误（业务层会自己处理，比如 doc 不存在）
    if (status && status !== 404) {
      message.error(`[${status || "网络错误"}] ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
    }
    return Promise.reject(err);
  }
);

export default api;
