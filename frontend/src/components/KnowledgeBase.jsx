// 知识库管理页：统计 + 上传（异步解析进度条） + 文档表 + 切片抽屉
// 仿 RAGFlow "Files" 标签页。
import React, { useEffect, useState, useCallback, useRef } from "react";
import {
  Card, Row, Col, Table, Button, Upload, Space, Popconfirm, message, Progress,
  Tag, Drawer, Pagination, Empty, Tooltip, Select, Steps,
} from "antd";
import {
  InboxOutlined, DeleteOutlined, EyeOutlined, ReloadOutlined, CloudUploadOutlined,
} from "@ant-design/icons";
import api from "../api.js";

const { Dragger } = Upload;

// 入库阶段顺序（和后端 scripts/api.py 的 STAGE_WEIGHTS 一一对应），
// 用来把后端的 current_stage 字符串映射成 Steps 的高亮下标。
const STAGES = ["解析", "切片", "向量化", "入库"];
const stageIndex = (s) => {
  const i = STAGES.indexOf(s);
  return i < 0 ? 0 : i;   // "完成"/"失败"/"加载模型"等非阶段名一律回到 0
};

// 单文档详情抽屉（查看切片）
function ChunkDrawer({ docId, docName, open, onClose }) {
  const [chunks, setChunks] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const pageSize = 20;

  const load = useCallback(async () => {
    if (!docId) return;
    setLoading(true);
    try {
      const r = await api.get(`/documents/${docId}/chunks`, { params: { limit: pageSize, offset: (page - 1) * pageSize } });
      setChunks(r.data.chunks || []);
      setTotal(r.data.total || 0);
    } catch (e) {
      // 拦截器已弹错
    } finally {
      setLoading(false);
    }
  }, [docId, page]);

  useEffect(() => { if (open) { setPage(1); } }, [open, docId]);
  useEffect(() => { if (open) load(); }, [open, load]);

  return (
    <Drawer
      title={docName ? `切片详情 — ${docName}` : "切片详情"}
      open={open}
      onClose={onClose}
      width={720}
      destroyOnClose
    >
      {loading ? (
        <div style={{ textAlign: "center", padding: 40 }}>加载中…</div>
      ) : chunks.length === 0 ? (
        <Empty description="暂无切片" />
      ) : (
        <>
          {chunks.map((c) => (
            <Card
              key={c.chunk_id}
              size="small"
              style={{ marginBottom: 10 }}
              title={
                <Space>
                  <Tag color="blue">#{c.chunk_index}</Tag>
                  <span style={{ fontWeight: 400, color: "#595959" }}>
                    {c.heading || "(无标题)"} · 字符 {c.char_start}-{c.char_end}
                  </span>
                </Space>
              }
            >
              <div style={{ whiteSpace: "pre-wrap", fontSize: 13, color: "#1f2937" }}>
                {c.content}
              </div>
            </Card>
          ))}
          <div style={{ textAlign: "right", marginTop: 12 }}>
            <Pagination
              current={page}
              total={total}
              pageSize={pageSize}
              onChange={setPage}
              showSizeChanger={false}
            />
          </div>
        </>
      )}
    </Drawer>
  );
}

export default function KnowledgeBase() {
  // 总览统计
  const [stats, setStats] = useState({ documents: 0, chunks: 0, vectors: null, staging_docs: 0 });
  // 文档列表
  const [docs, setDocs] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  // 上传相关
  const [fileList, setFileList] = useState([]);     // 待上传（受控）
  const [strategy, setStrategy] = useState("recursive"); // 切片策略
  const [uploading, setUploading] = useState(false);
  // phase: idle(空闲) / uploading(字节上传中) / processing(服务端解析入库中) / done(收尾)
  const [phase, setPhase] = useState("idle");
  const [uploadPct, setUploadPct] = useState(0);     // ① 浏览器→服务端 的字节上传进度 0~100
  const [job, setJob] = useState(null);              // ② 服务端异步入库任务快照（轮询回来）
  const [uploadResults, setUploadResults] = useState(null); // 上传完的逐文件结果
  // 轮询定时器句柄（组件卸载/重新上传时要清掉，否则会泄漏 + setState 到已卸载组件）
  const pollTimer = useRef(null);
  // 切片抽屉
  const [drawer, setDrawer] = useState({ open: false, docId: null, docName: null });

  const refresh = useCallback(async () => {
    setLoadingDocs(true);
    try {
      const [s, d] = await Promise.all([api.get("/stats"), api.get("/documents")]);
      setStats({
        documents: s.data.documents,
        chunks: s.data.chunks,
        vectors: s.data.vectors,
        staging_docs: s.data.staging_docs || 0,
      });
      setDocs(d.data.documents || []);
    } catch (e) {
      // 拦截器已处理
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // 组件卸载时清掉轮询定时器
  useEffect(() => () => clearTimeout(pollTimer.current), []);

  // ===== 上传（两段式进度）=====
  // 第一段：浏览器 → 服务端 的字节上传进度（用 XHR，fetch 拿不到 upload.onprogress）。
  // 第二段：服务端 解析 → 切片 → 向量化 → 入库 的进度（后端异步任务，前端轮询 /api/jobs/{id}）。
  // 为什么分两段：几十个文件的 embedding 可能要几十秒~几分钟，
  //   如果上传接口同步跑完，前端只能卡在一个 100% 的进度条上干等。
  //   异步化后字节传完（通常很快）即返回 job_id，剩下的重活后台跑、前端实时看到阶段进度。
  const pollJob = (jobId) =>
    new Promise((resolve) => {
      const tick = async () => {
        let j = null;
        try {
          const r = await api.get(`/jobs/${jobId}`);
          j = r.data;
          setJob(j);
        } catch (e) {
          // 单次轮询失败不致命（比如瞬时网络抖动）：继续重试，直到任务结束或组件卸载
        }
        if (j && (j.status === "done" || j.status === "failed")) {
          resolve(j);
          return;
        }
        pollTimer.current = setTimeout(tick, 700);
      };
      tick();
    });

  const handleUpload = async () => {
    if (fileList.length === 0) {
      message.warning("请先拖入或选择文件");
      return;
    }
    setUploading(true);
    setPhase("uploading");
    setUploadPct(0);
    setJob(null);
    setUploadResults(null);
    const fd = new FormData();
    fd.append("strategy", strategy);   // 切片策略（recursive/fixed/structural）
    for (const f of fileList) {
      // f 是 AntD 的 file 对象（带 originFileObj），取原始 File
      const raw = f.originFileObj || f;
      fd.append("files", raw);
    }
    try {
      // —— 第一段：把字节传上去，拿回 job_id ——
      const data = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/documents/upload");
        xhr.upload.onprogress = (ev) => {
          if (ev.lengthComputable) setUploadPct(Math.round((ev.loaded / ev.total) * 100));
        };
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try { resolve(JSON.parse(xhr.responseText)); } catch { resolve({}); }
          } else {
            reject(new Error(`HTTP ${xhr.status}`));
          }
        };
        xhr.onerror = () => reject(new Error("网络错误"));
        xhr.send(fd);
      });

      // 兼容两种后端返回：
      //   - 新后端（异步）：{"job_id": "...", "status": "pending", ...} → 进第二段轮询
      //   - 老后端（同步）：{"results": [...]} → 直接当结果用
      if (!data.job_id) {
        setUploadResults(data.results || []);
        finishUpload(data.results || []);
        return;
      }

      // —— 第二段：轮询服务端入库进度 ——
      setPhase("processing");
      setJob({ status: "pending", progress: 0, current_stage: "排队中",
               current_file: "", done_files: 0, total_files: data.total_files,
               filenames: data.filenames || [] });
      const j = await pollJob(data.job_id);
      setUploadResults(j.results || []);
      setPhase("done");
      if (j.status === "failed") {
        message.error(`入库失败：${j.error || "未知错误"}`);
      } else {
        finishUpload(j.results || []);
      }
      setFileList([]);
      refresh();
    } catch (e) {
      message.error(`上传失败：${e.message || e}`);
      setPhase("idle");
    } finally {
      setUploading(false);
    }
  };

  // 统一的「跑完了」提示（同步/异步两条路复用）
  const finishUpload = (results) => {
    const ok = results.filter((r) => r.ok).length;
    const fail = results.length - ok;
    if (results.length === 0 || fail === 0) message.success(`入库完成：${ok} 个文件全部成功`);
    else message.warning(`入库完成：${ok} 成功 / ${fail} 失败，详见下方`);
  };

  // ===== 删除 =====
  const handleDelete = async (doc) => {
    try {
      await api.delete(`/documents/${doc.doc_id}`);
      message.success(`已删除：${doc.doc_name}`);
      refresh();
    } catch (e) { /* 拦截器已弹 */ }
  };

  // ===== 表格列 =====
  const columns = [
    {
      title: "文档名", dataIndex: "doc_name", ellipsis: true,
      render: (v, r) => (
        <Space>
          <span>{v}</span>
          <Tooltip title={r.doc_id}><Tag color="default" style={{ fontSize: 11 }}>{r.doc_id.slice(0, 8)}</Tag></Tooltip>
        </Space>
      ),
    },
    { title: "切片数", dataIndex: "chunk_count", width: 90, sorter: (a, b) => a.chunk_count - b.chunk_count, defaultSortOrder: "descend" },
    { title: "字符数", dataIndex: "char_count", width: 100,
      render: (v) => v != null ? v.toLocaleString() : "—" },
    {
      title: "状态", dataIndex: "status", width: 90,
      render: (s) => <span className={`status-badge status-${s}`}>{s}</span>,
    },
    {
      title: "来源", dataIndex: "source", width: 90,
      render: (s) => <Tag color={s === "agent_generated" ? "purple" : "blue"}>{s}</Tag>,
    },
    { title: "创建时间", dataIndex: "created_at", width: 170 },
    {
      title: "操作", width: 170, fixed: "right",
      render: (_, r) => (
        <Space>
          <Button size="small" icon={<EyeOutlined />}
            onClick={() => setDrawer({ open: true, docId: r.doc_id, docName: r.doc_name })}>
            切片
          </Button>
          <Popconfirm
            title="确认删除该文档？"
            description="将同时删除 PG 切片和 Milvus 向量，不可恢复。"
            okText="删除" cancelText="取消" okButtonProps={{ danger: true }}
            onConfirm={() => handleDelete(r)}
          >
            <Button size="small" danger icon={<DeleteOutlined />}>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      {/* === 统计卡 === */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}><Card className="stat-card"><div className="stat-label">文档数</div><div className="stat-value">{stats.documents}</div></Card></Col>
        <Col span={6}><Card className="stat-card"><div className="stat-label">切片数</div><div className="stat-value">{stats.chunks}</div></Card></Col>
        <Col span={6}><Card className="stat-card"><div className="stat-label">向量数</div><div className="stat-value">{stats.vectors ?? "—"}</div><div className="stat-extra">Milvus Lite</div></Card></Col>
        <Col span={6}><Card className="stat-card"><div className="stat-label">待审影子库</div><div className="stat-value">{stats.staging_docs}</div><div className="stat-extra">阶段 6 写入待人工转 active</div></Card></Col>
      </Row>

      {/* === 上传区 === */}
      <Card
        title="上传文件到知识库"
        extra={
          <Space>
            <span style={{ color: "#8c8c8c", fontSize: 13 }}>分块策略</span>
            <Select
              size="small"
              value={strategy}
              onChange={setStrategy}
              disabled={uploading}
              style={{ width: 168 }}
              options={[
                { value: "recursive", label: "递归字符（推荐）" },
                { value: "structural", label: "Markdown 标题" },
                { value: "fixed", label: "固定长度" },
              ]}
            />
            <Button onClick={refresh} icon={<ReloadOutlined />} disabled={uploading}>刷新</Button>
            <Button type="primary" onClick={handleUpload} loading={uploading}
              icon={<CloudUploadOutlined />} disabled={fileList.length === 0}>
              开始上传（{fileList.length}）
            </Button>
          </Space>
        }
        style={{ marginBottom: 16 }}
      >
        <Dragger
          multiple
          beforeUpload={() => false}                 // 阻止自动上传，由「开始上传」统一发
          fileList={fileList}
          onChange={({ fileList }) => setFileList(fileList.slice(-50))}  // 最多 50 个
          onRemove={(f) => setFileList((prev) => prev.filter(x => x.uid !== f.uid))}
          accept=".md,.markdown,.txt,.log,.csv,.json,.yaml,.yml,.html,.htm,.pdf,.docx"
          className="upload-dragger"
          style={{ padding: "4px 0" }}
        >
          <p className="ant-upload-drag-icon" style={{ marginBottom: 4 }}><InboxOutlined /></p>
          <p className="ant-upload-text">点击或拖拽文件到这里上传</p>
          <p className="ant-upload-hint" style={{ fontSize: 12 }}>
            支持 md / txt / html / pdf / docx / json / yaml / csv 等任意文本类文件；
            同一文件名重复上传会覆盖更新。
          </p>
        </Dragger>
        {/* 第一段：字节上传进度 */}
        {phase === "uploading" && (
          <div style={{ marginTop: 12 }}>
            <Progress percent={uploadPct} status="active" />
            <div style={{ color: "#8c8c8c", fontSize: 12 }}>正在把文件字节传到服务端…</div>
          </div>
        )}

        {/* 第二段：服务端异步入库进度（解析 → 切片 → 向量化 → 入库） */}
        {phase === "processing" && job && (
          <div style={{ marginTop: 12 }}>
            <Progress
              percent={Math.round(job.progress || 0)}
              status="active"
              strokeColor={{ from: "#108ee9", to: "#87d068" }}
            />
            <Steps
              size="small"
              current={stageIndex(job.current_stage)}
              style={{ marginTop: 10, marginBottom: 6 }}
              items={STAGES.map((s) => ({ title: s }))}
            />
            <div style={{ fontSize: 12, color: "#595959" }}>
              当前文件：<b>{job.current_file || "—"}</b>
              　阶段：<b>{job.current_stage}</b>
              　已完成 <b>{job.done_files}/{job.total_files}</b> 个文件
            </div>
          </div>
        )}

        {uploadResults && (
          <div style={{ marginTop: 12 }}>
            {uploadResults.filter(r => r.ok).length > 0 && (
              <div style={{ marginBottom: 6 }}>
                <Tag color="success">成功 {uploadResults.filter(r => r.ok).length}</Tag>
                {uploadResults.filter(r => r.ok).map((r, i) => (
                  <Tag key={i} color="blue" style={{ marginBottom: 4 }}>
                    {r.filename} · {r.chunks} 块 · {r.char_count} 字
                  </Tag>
                ))}
              </div>
            )}
            {uploadResults.filter(r => !r.ok).length > 0 && (
              <div>
                <Tag color="error">失败 {uploadResults.filter(r => !r.ok).length}</Tag>
                {uploadResults.filter(r => !r.ok).map((r, i) => (
                  <div key={i} style={{ color: "#cf1322", fontSize: 12, marginTop: 2 }}>
                    ✗ {r.filename}：{r.error}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Card>

      {/* === 文档表 === */}
      <Card title={`文档列表（${docs.length}）`} bodyStyle={{ padding: 0 }}>
        <Table
          rowKey="doc_id"
          dataSource={docs}
          columns={columns}
          loading={loadingDocs}
          pagination={{ pageSize: 20, showSizeChanger: false }}
          scroll={{ x: 900 }}
          size="middle"
        />
      </Card>

      <ChunkDrawer
        docId={drawer.docId}
        docName={drawer.docName}
        open={drawer.open}
        onClose={() => setDrawer({ open: false, docId: null, docName: null })}
      />
    </div>
  );
}
