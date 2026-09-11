// 知识库体检页（阶段 7）
// 跑四个检测器（过时 / 矛盾 / 僵尸 / 重复），展示概览 + Markdown 健康报告 + 待审影子库。
// 只读不写：报告只给建议，删/改动作人工执行。
import React, { useState } from "react";
import {
  Card, Row, Col, Button, Checkbox, Slider, Space, Alert, Tag, Empty, Spin, Divider,
} from "antd";
import { ExperimentOutlined, FileTextOutlined } from "@ant-design/icons";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import api from "../api.js";

export default function Health() {
  const [skipConflict, setSkipConflict] = useState(false);
  const [dupThreshold, setDupThreshold] = useState(0.95);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);

  const run = async () => {
    setLoading(true);
    setReport(null);
    try {
      const r = await api.post("/health-check", {
        dup_threshold: dupThreshold,
        skip_conflict: skipConflict,
      });
      setReport(r.data);
    } catch (e) {
      // 拦截器已弹错
    } finally {
      setLoading(false);
    }
  };

  const s = report?.summary;
  const bk = s?.by_kind || {};

  return (
    <div>
      {/* 选项 + 运行 */}
      <Card
        title={<span><ExperimentOutlined /> 知识库体检（阶段 7）</span>}
        style={{ marginBottom: 16 }}
        extra={
          <Button type="primary" onClick={run} loading={loading}>运行体检</Button>
        }
      >
        <div style={{ color: "#8c8c8c", fontSize: 13, marginBottom: 12 }}>
          跑四个检测器：过时 / 矛盾 / 僵尸 / 重复。**只读不写** —— 报告只给建议，删除/修改动作需人工执行。
        </div>
        <Space size="large" wrap>
          <Checkbox checked={skipConflict} onChange={(e) => setSkipConflict(e.target.checked)}>
            跳过矛盾检测（省 LLM 调用）
          </Checkbox>
          <div style={{ width: 280 }}>
            <div style={{ fontSize: 12, color: "#8c8c8c" }}>重复阈值（余弦 ≥ 此值判重复）: {dupThreshold.toFixed(2)}</div>
            <Slider min={0.8} max={0.99} step={0.01} value={dupThreshold} onChange={setDupThreshold} />
          </div>
        </Space>
      </Card>

      {loading && (
        <Card><div style={{ textAlign: "center", padding: 40 }}><Spin /> &nbsp;体检中…（重编码全库 chunk + 可能的 LLM 矛盾判定，稍等）</div></Card>
      )}

      {!loading && !report && (
        <Card><Empty description="点击右上角「运行体检」开始" /></Card>
      )}

      {!loading && report && (
        <>
          {/* 概览统计 */}
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={6}><Card className="stat-card"><div className="stat-label">文档数</div><div className="stat-value">{s.documents}</div></Card></Col>
            <Col span={6}><Card className="stat-card"><div className="stat-label">切片数</div><div className="stat-value">{s.chunks}</div></Card></Col>
            <Col span={6}><Card className="stat-card"><div className="stat-label">发现问题</div><div className="stat-value" style={{ color: s.total_issues > 0 ? "#cf1322" : "#389e0d" }}>{s.total_issues}</div></Card></Col>
            <Col span={6}><Card className="stat-card"><div className="stat-label">待审影子库</div><div className="stat-value" style={{ color: report.staging_docs.length > 0 ? "#d48806" : "#389e0d" }}>{report.staging_docs.length}</div></Card></Col>
          </Row>

          <Card style={{ marginBottom: 16 }} size="small">
            <Space size="large" wrap>
              <span>过时 <Tag color="orange">{bk.stale || 0}</Tag></span>
              <span>矛盾 <Tag color="red">{bk.conflict || 0}</Tag></span>
              <span>僵尸 <Tag color="purple">{bk.zombie || 0}</Tag></span>
              <span>重复 <Tag color="blue">{bk.duplicate || 0}</Tag></span>
              <Divider type="vertical" />
              <span>高危 <Tag color="red">{(s.by_severity || {}).high || 0}</Tag></span>
              <span>中危 <Tag color="gold">{(s.by_severity || {}).medium || 0}</Tag></span>
              <span>低危 <Tag color="default">{(s.by_severity || {}).low || 0}</Tag></span>
            </Space>
          </Card>

          {report.skipped?.map((sk, i) => (
            <Alert key={i} type="warning" showIcon message={`跳过：${sk}`} style={{ marginBottom: 12 }} />
          ))}

          {/* 待审影子库 */}
          {report.staging_docs.length > 0 && (
            <Card title="🟡 待人工审核的影子库（staging → active）" size="small" style={{ marginBottom: 16 }}>
              {report.staging_docs.map((d, i) => (
                <div key={i} style={{ marginBottom: 4 }}>
                  <Tag color="orange">{d.doc_id}</Tag> {d.doc_name}
                  <span style={{ color: "#8c8c8c", fontSize: 12 }}>（写入于 {d.created_at}）</span>
                </div>
              ))}
              <div style={{ color: "#8c8c8c", fontSize: 12, marginTop: 8 }}>
                处置：人工确认正确后把 document.status 由 staging 改 active；错误则删除。未经审核不要直接转 active（防投毒最后一公里）。
              </div>
            </Card>
          )}

          {/* Markdown 报告 */}
          <Card title={<span><FileTextOutlined /> 健康报告（Markdown）</span>}>
            <div className="markdown-body" style={{ fontSize: 14, lineHeight: 1.8 }}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{report.markdown}</ReactMarkdown>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
