// 问答页：消息列表 + 输入框 + 设置面板（retrieval/top_k/各阶段开关）
// 调用 POST /api/ask；展示答案、状态徽标、召回块、阶段 6 过程。
import React, { useEffect, useState, useRef } from "react";
import {
  Input, Button, Card, Switch, Radio, Slider, Space, Spin, Tag, Empty, Collapse,
  Typography, Divider,
} from "antd";
import { SendOutlined, ThunderboltOutlined, RobotOutlined, UserOutlined } from "@ant-design/icons";
import api from "../api.js";

const { TextArea } = Input;
const { Text } = Typography;

// 单条消息的结构
//   { role: "user" | "assistant", content, ts, meta? }
//   assistant 的 meta = /api/ask 的完整返回
function MessageBubble({ msg }) {
  if (msg.role === "user") {
    return (
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 16 }}>
        <div style={{ maxWidth: "75%" }}>
          <div style={{ fontSize: 11, color: "#8c8c8c", textAlign: "right", marginBottom: 4 }}>
            <UserOutlined /> 你
          </div>
          <div style={{
            background: "#2a5298", color: "#fff",
            padding: "10px 14px", borderRadius: 10, whiteSpace: "pre-wrap", wordBreak: "break-word",
          }}>{msg.content}</div>
        </div>
      </div>
    );
  }
  // assistant
  const m = msg.meta || {};
  const badges = [];
  if (m.self_healed_by_agent) badges.push(<Tag key="h6" color="purple">✅ 阶段6 缺口自愈</Tag>);
  else if (m.self_healed)      badges.push(<Tag key="h5" color="green">✅ 阶段5 拒答自愈</Tag>);
  else if (m.refusal)         badges.push(<Tag key="rf" color="orange">⚠ 拒答</Tag>);
  if (m.reflected) badges.push(m.reflect_pass
    ? <Tag key="rp" color="cyan">🔎 反射通过</Tag>
    : <Tag key="rp" color="red">🔎 反射未过（已降级拒答）</Tag>);

  return (
    <div style={{ display: "flex", marginBottom: 16 }}>
      <div style={{ maxWidth: "85%" }}>
        <div style={{ fontSize: 11, color: "#8c8c8c", marginBottom: 4 }}>
          <RobotOutlined /> RAG Agent
        </div>
        <div style={{
          background: "#fff", border: "1px solid #f0f0f0",
          padding: "12px 16px", borderRadius: 10,
        }}>
          <div style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: 14, lineHeight: 1.7 }}>
            {msg.content}
          </div>
          {badges.length > 0 && <div style={{ marginTop: 10 }}>{badges}</div>}
          {m.request_id && (
            <Text type="secondary" style={{ fontSize: 11, display: "block", marginTop: 6 }}>
              request_id: {m.request_id} · 耗时 {m.latency_s || 0}s · 召回 { (m.retrieved || []).length } 块
            </Text>
          )}

          {/* 阶段 6 过程明细 */}
          {m.gap_detected && (
            <div style={{ marginTop: 10, padding: 10, background: "#faf5ff", borderRadius: 6, fontSize: 12 }}>
              <div><b>🕸 阶段6 缺口自愈过程</b></div>
              <div>缺口判定：{m.gap_reason}</div>
              {m.searched?.length > 0 && <div>调研来源：{m.searched.slice(0, 5).map((u, i) => <div key={i}>· <a href={u} target="_blank" rel="noreferrer">{u}</a></div>)}</div>}
              {m.quality_score != null && <div>质量门禁：{m.quality_score.toFixed(2)} {m.quality_pass ? "✅" : "❌"}</div>}
              <div>影子库：{m.ingested_chunks} 个 chunk (status=staging)</div>
            </div>
          )}

          {/* 召回块折叠 */}
          {(m.retrieved || []).length > 0 && (
            <Collapse
              ghost size="small" style={{ marginTop: 8 }}
              items={[{
                key: "ctx",
                label: <Text type="secondary" style={{ fontSize: 12 }}>📚 召回的 {m.retrieved.length} 个上下文块（点击展开）</Text>,
                children: m.retrieved.map((c, i) => (
                  <div key={i} className="retrieved-block">
                    <div style={{ color: "#595959", marginBottom: 4, fontSize: 12 }}>
                      [{i+1}] score={c.score?.toFixed(4) ?? "?"} · {c.doc_name}{c.heading ? ` / ${c.heading}` : ""}{c.unverified ? "  ⚠未验证" : ""}
                    </div>
                    <div>{c.content}</div>
                  </div>
                )),
              }]}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  // 检索设置
  const [retrieval, setRetrieval] = useState("vector");
  const [topK, setTopK] = useState(5);
  const [save, setSave] = useState(true);
  const [selfHeal, setSelfHeal] = useState(true);
  const [agentHeal, setAgentHeal] = useState(false);
  const [reflect, setReflect] = useState(false);
  const [trace, setTrace] = useState(false);

  // 新消息自动滚到底
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const send = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    const userMsg = { role: "user", content: q, ts: Date.now() };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    try {
      const r = await api.post("/ask", {
        question: q,
        top_k: topK,
        retrieval,
        self_heal: selfHeal,
        agent_heal: agentHeal,
        reflect,
        trace,
        save,
      });
      const meta = r.data;
      const ansText = meta.answer || "(无答案)";
      setMessages((prev) => [...prev, { role: "assistant", content: ansText, ts: Date.now(), meta }]);
    } catch (e) {
      setMessages((prev) => [...prev, {
        role: "assistant",
        content: `⚠ 调用失败：${e?.response?.data?.detail || e.message}`,
        ts: Date.now(),
      }]);
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="chat-container">
      {/* === 左侧：设置 === */}
      <Card title="⚙ 检索与自愈设置" style={{ width: 280, flexShrink: 0 }} bodyStyle={{ padding: 16 }}>
        <Space direction="vertical" size={14} style={{ width: "100%" }}>
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>检索模式</Text>
            <Radio.Group value={retrieval} onChange={(e) => setRetrieval(e.target.value)} style={{ marginTop: 4, display: "flex" }}>
              <Radio.Button value="vector" style={{ flex: 1, textAlign: "center" }}>向量</Radio.Button>
              <Radio.Button value="hybrid" style={{ flex: 1, textAlign: "center" }}>混合</Radio.Button>
            </Radio.Group>
          </div>
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>召回块数 top_k = {topK}</Text>
            <Slider min={1} max={10} value={topK} onChange={setTopK} />
          </div>
          <Divider style={{ margin: "4px 0" }} />
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <Text style={{ fontSize: 13 }}>阶段5 拒答自愈</Text>
            <Switch size="small" checked={selfHeal} onChange={setSelfHeal} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <Text style={{ fontSize: 13 }}>阶段6 缺口自愈（联网补库）</Text>
            <Switch size="small" checked={agentHeal} onChange={setAgentHeal} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <Text style={{ fontSize: 13 }}>阶段8 反射事实核查</Text>
            <Switch size="small" checked={reflect} onChange={setReflect} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <Text style={{ fontSize: 13 }}>阶段8 链路追踪落库</Text>
            <Switch size="small" checked={trace} onChange={setTrace} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <Text style={{ fontSize: 13 }}>记录到 qa_log</Text>
            <Switch size="small" checked={save} onChange={setSave} />
          </div>
          <Divider style={{ margin: "4px 0" }} />
          <Text type="secondary" style={{ fontSize: 11, lineHeight: 1.5 }}>
            提示：阶段 6 联网补库会触发外网请求；阶段 8 反射会多调一次 LLM；切回向量模式可省 BM25 重建。
          </Text>
        </Space>
      </Card>

      {/* === 右侧：对话 === */}
      <div className="chat-main">
        <div className="chat-messages" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="chat-empty">
              <Empty description="输入问题开始对话" />
            </div>
          ) : (
            messages.map((m, i) => <MessageBubble key={i} msg={m} />)
          )}
          {loading && (
            <div style={{ textAlign: "center", padding: 20, color: "#8c8c8c" }}>
              <Spin size="small" /> &nbsp; 检索 + 生成中…
            </div>
          )}
        </div>
        <div className="chat-input-bar">
          <Space.Compact style={{ width: "100%" }}>
            <TextArea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="输入你的问题（Enter 发送，Shift+Enter 换行）"
              autoSize={{ minRows: 1, maxRows: 6 }}
              disabled={loading}
              style={{ resize: "none" }}
            />
            <Button type="primary" icon={<ThunderboltOutlined />} onClick={send} loading={loading}
              disabled={!input.trim()} style={{ height: "auto" }}>
              发送
            </Button>
          </Space.Compact>
        </div>
      </div>
    </div>
  );
}
