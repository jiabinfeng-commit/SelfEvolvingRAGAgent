// 检索测试页（RAGFlow 的 "Retrieval testing"）
// 只召回、不调 LLM：输入 query，看 top_k 块和分数。比问答页更能定位"检索质量"。
import React, { useState } from "react";
import {
  Card, Input, Button, Radio, Slider, Space, Tag, Empty, Spin, Typography,
} from "antd";
import { SearchOutlined, FileSearchOutlined } from "@ant-design/icons";
import api from "../api.js";

const { TextArea } = Input;
const { Text } = Typography;

export default function Retrieval() {
  const [question, setQuestion] = useState("");
  const [retrieval, setRetrieval] = useState("vector");
  const [topK, setTopK] = useState(8);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const run = async () => {
    const q = question.trim();
    if (!q) return;
    setLoading(true);
    setResult(null);
    try {
      const r = await api.post("/retrieve", { question: q, top_k: topK, retrieval });
      setResult(r.data);
    } catch (e) {
      // 拦截器已弹错
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card title={<span><FileSearchOutlined /> 检索测试（只召回，不调 LLM）</span>}>
      <TextArea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="输入一个 query，看它召回了哪些块（例如：FastAPI 怎么做依赖注入？）"
        autoSize={{ minRows: 2, maxRows: 5 }}
        style={{ marginBottom: 12 }}
      />
      <Space size="large" wrap style={{ marginBottom: 12 }}>
        <div>
          <Text type="secondary" style={{ fontSize: 12, marginRight: 8 }}>检索模式</Text>
          <Radio.Group value={retrieval} onChange={(e) => setRetrieval(e.target.value)}>
            <Radio.Button value="vector">向量</Radio.Button>
            <Radio.Button value="hybrid">混合（向量+BM25）</Radio.Button>
          </Radio.Group>
        </div>
        <div style={{ width: 240 }}>
          <Text type="secondary" style={{ fontSize: 12 }}>top_k = {topK}</Text>
          <Slider min={1} max={30} value={topK} onChange={setTopK} />
        </div>
        <Button type="primary" icon={<SearchOutlined />} onClick={run} loading={loading}
          disabled={!question.trim()}>
          检索
        </Button>
      </Space>

      {loading && <div style={{ textAlign: "center", padding: 40 }}><Spin /> &nbsp;检索中…</div>}

      {!loading && result && (
        <>
          <div style={{ marginBottom: 12, color: "#595959", fontSize: 13 }}>
            命中 <b>{result.count}</b> 块（retrieval={result.retrieval}, top_k={result.top_k}）
          </div>
          {result.chunks.length === 0 ? (
            <Empty description="没有召回到任何块（语料库可能确实没有相关内容）" />
          ) : (
            result.chunks.map((c, i) => (
              <Card key={i} size="small" style={{ marginBottom: 10 }}
                title={
                  <Space>
                    <Tag color="blue">[{i + 1}]</Tag>
                    <Tag color="green">score {c.score}</Tag>
                    <span style={{ fontWeight: 400, color: "#595959", fontSize: 13 }}>
                      {c.doc_name}{c.heading ? ` / ${c.heading}` : ""} · 字符 {c.char_start}-{c.char_end}
                    </span>
                  </Space>
                }>
                <div style={{ whiteSpace: "pre-wrap", fontSize: 13 }}>{c.content}</div>
              </Card>
            ))
          )}
        </>
      )}

      {!loading && !result && (
        <Empty description="输入 query 后点「检索」" style={{ padding: 30 }} />
      )}
    </Card>
  );
}
