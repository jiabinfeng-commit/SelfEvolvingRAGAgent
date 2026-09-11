// 链路追踪页（阶段 8）
// 读 trace_log：每次开启 --trace 的问答都会留痕，用于复盘"为什么答成这样 / 成本花在哪"。
import React, { useEffect, useState, useCallback } from "react";
import { Card, Table, Button, Tag, Empty, Typography, Space } from "antd";
import { ReloadOutlined, NodeIndexOutlined } from "@ant-design/icons";
import api from "../api.js";

const { Text } = Typography;

export default function Traces() {
  const [traces, setTraces] = useState([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await api.get("/traces", { params: { limit: 50 } });
      setTraces(r.data.traces || []);
    } catch (e) {
      // 拦截器已弹错
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const columns = [
    { title: "时间", dataIndex: "created_at", width: 170 },
    { title: "问题", dataIndex: "question", ellipsis: true },
    { title: "检索", dataIndex: "retrieval", width: 80,
      render: (v) => <Tag color={v === "hybrid" ? "geekblue" : "blue"}>{v}</Tag> },
    { title: "反射", dataIndex: "reflected", width: 70,
      render: (v) => v ? <Tag color="cyan">✅</Tag> : <span style={{ color: "#bfbfbf" }}>—</span> },
    { title: "核查通过", dataIndex: "reflect_pass", width: 90,
      render: (v) => v === true ? <Tag color="green">✅</Tag> : v === false ? <Tag color="red">❌</Tag> : <span style={{ color: "#bfbfbf" }}>—</span> },
    { title: "耗时(s)", dataIndex: "latency_s", width: 90,
      render: (v) => (v != null ? Number(v).toFixed(2) : "—") },
    { title: "估算token", dataIndex: "est_tokens", width: 100 },
  ];

  const expandedRow = (r) => (
    <div style={{ padding: "4px 0" }}>
      <div style={{ marginBottom: 8 }}>
        <Text type="secondary">request_id：</Text><Text code>{r.request_id}</Text>
      </div>
      <div style={{ marginBottom: 8 }}>
        <Text type="secondary">召回块（{(r.recalled_ids || []).length} 个）：</Text>
        <div style={{ marginTop: 4 }}>
          {(r.recalled_ids || []).slice(0, 20).map((id, i) => (
            <Tag key={i} style={{ marginBottom: 4 }}>{id.slice(0, 8)}</Tag>
          ))}
        </div>
      </div>
      <div style={{ marginBottom: 8 }}>
        <Text type="secondary">召回分数：</Text>
        <Text>{(r.recalled_scores || []).slice(0, 20).map((x) => Number(x).toFixed(3)).join(", ")}</Text>
      </div>
      <div style={{ marginBottom: 8 }}>
        <Text type="secondary">Prompt（发给 LLM 的 user 部分，前 1500 字）：</Text>
        <pre style={{ background: "#fafafa", border: "1px solid #f0f0f0", borderRadius: 6, padding: 10, marginTop: 4, maxHeight: 240, overflow: "auto", fontSize: 12, whiteSpace: "pre-wrap" }}>
          {(r.prompt || "").slice(0, 1500) || "(空)"}
        </pre>
      </div>
      <div>
        <Text type="secondary">答案：</Text>
        <div style={{ background: "#f6ffed", border: "1px solid #d9f7be", borderRadius: 6, padding: 10, marginTop: 4, whiteSpace: "pre-wrap" }}>
          {r.answer || "(空)"}
        </div>
      </div>
    </div>
  );

  return (
    <Card
      title={<span><NodeIndexOutlined /> 链路追踪（阶段 8）</span>}
      extra={<Button onClick={load} icon={<ReloadOutlined />} loading={loading}>刷新</Button>}
    >
      <div style={{ color: "#8c8c8c", fontSize: 13, marginBottom: 12 }}>
        读取 trace_log：每次在「问答」页勾选「阶段8 链路追踪落库」后问答，这里就会留痕，
        用于复盘「为什么答成这样 / 成本花在哪 / 哪些问题常触发反射」。点击行可展开明细。
      </div>
      {traces.length === 0 && !loading ? (
        <Empty description="trace_log 为空。去「问答」页勾选「阶段8 链路追踪落库」后问答，这里就会出现记录。" />
      ) : (
        <Table
          rowKey="id"
          dataSource={traces}
          columns={columns}
          loading={loading}
          size="middle"
          expandable={{ expandedRowRender: expandedRow }}
          pagination={{ pageSize: 15, showSizeChanger: false }}
          scroll={{ x: 900 }}
        />
      )}
    </Card>
  );
}
