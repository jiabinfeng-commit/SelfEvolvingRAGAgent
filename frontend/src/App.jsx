// 顶层路由：Layout 包住所有页面
// =============================================================
// 路由：
//   /            → 重定向到 /knowledge
//   /knowledge   → 知识库管理页（上传 / 文档 / 切片）
//   /chat        → 问答页
//   /retrieval   → 检索测试页（只召回，不调 LLM）
//   /health      → 知识库体检页（阶段7）
//   /traces      → 链路追踪页（阶段8）
// =============================================================
import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import KnowledgeBase from "./components/KnowledgeBase.jsx";
import Chat from "./components/Chat.jsx";
import Retrieval from "./components/Retrieval.jsx";
import Health from "./components/Health.jsx";
import Traces from "./components/Traces.jsx";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Navigate to="/knowledge" replace />} />
        <Route path="/knowledge" element={<KnowledgeBase />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/retrieval" element={<Retrieval />} />
        <Route path="/health" element={<Health />} />
        <Route path="/traces" element={<Traces />} />
      </Route>
    </Routes>
  );
}
