// 顶栏 + 侧边栏 + 内容区
// 仿 RAGFlow：深蓝顶栏带 logo，左侧白底菜单，右侧内容。
import React, { useEffect, useState } from "react";
import { Layout as AntLayout, Menu, Tag, Space, Spin } from "antd";
import {
  DatabaseOutlined,
  MessageOutlined,
  FileSearchOutlined,
  ExperimentOutlined,
  NodeIndexOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import api from "../api.js";

const { Header, Sider, Content } = AntLayout;

// 菜单项：图标 + 文字 + path。RAGFlow 风格用 outline 图标。
// 上组=日常使用（知识库/问答/检索测试），下组=运维诊断（体检/追踪）。
const MENU_ITEMS = [
  { key: "/knowledge", icon: <DatabaseOutlined />, label: "知识库" },
  { key: "/chat",      icon: <MessageOutlined />,  label: "问答" },
  { key: "/retrieval", icon: <FileSearchOutlined />, label: "检索测试" },
  { type: "divider" },
  { key: "/health",    icon: <ExperimentOutlined />, label: "知识库体检" },
  { key: "/traces",    icon: <NodeIndexOutlined />, label: "链路追踪" },
];

export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const [llm, setLlm] = useState(null);   // { backend, model, embed_model }
  const [health, setHealth] = useState(null); // /api/health 的 status

  // 顶栏状态：拉一次 /api/llm-info + /api/health 展示后端连通性
  useEffect(() => {
    api.get("/llm-info").then((r) => setLlm(r.data)).catch(() => setLlm({ backend: "?", model: "?" }));
    api.get("/health")
      .then((r) => setHealth(r.data))
      .catch(() => setHealth({ status: "down" }));
  }, []);

  return (
    <AntLayout style={{ minHeight: "100vh" }}>
      <Header style={{ padding: 0, lineHeight: "48px", height: 48 }}>
        <div className="app-logo">
          <span className="logo-mark"><ThunderboltOutlined /></span>
          Self-Evolving RAG Agent
          <span style={{ marginLeft: "auto", marginRight: 20, fontSize: 12, fontWeight: 400, opacity: 0.85 }}>
            {llm ? (
              <Space size={6}>
                <Tag color={health?.status === "ok" ? "success" : "warning"} style={{ margin: 0 }}>
                  {health?.status === "ok" ? "● 在线" : "● 降级"}
                </Tag>
                <span>{llm.backend} / {llm.model}</span>
              </Space>
            ) : (
              <Spin size="small" />
            )}
          </span>
        </div>
      </Header>
      <AntLayout>
        <Sider width={200} className="app-sider" theme="light" style={{ borderRight: "1px solid #e8e8e8" }}>
          <Menu
            mode="inline"
            selectedKeys={[location.pathname]}
            items={MENU_ITEMS}
            onClick={({ key }) => navigate(key)}
            style={{ borderInlineEnd: "none" }}
          />
        </Sider>
        <Content className="app-content">
          <Outlet />
        </Content>
      </AntLayout>
    </AntLayout>
  );
}
