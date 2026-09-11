import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite 配置
// ============================================================
// dev server 默认 5173 端口；/api/* 反代到后端 8000 端口。
// 这样前端代码里 axios 写 baseURL="/api" 就能直接打后端，
// 既避开 CORS（虽然后端也开了 CORS *），又让前后端可独立部署。
// 后端启动方式：python scripts/api.py
// ============================================================
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: "127.0.0.1",
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,  // 生产构建不产 sourcemap，省体积
  },
});
