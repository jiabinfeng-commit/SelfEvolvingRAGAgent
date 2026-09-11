# 阶段 10：前端 — 仿 RAGFlow 风格的 React 知识库管理 + 问答

之前所有阶段（1~9）都只有命令行和 Streamlit，**只能问答**，看不到知识库的全貌（有哪些文档、多少切片、能不能删、上传新文档）。本阶段把这两块短板一次性补齐：

- **后端** `scripts/api.py`：把知识库管理（上传 / 列表 / 切片查看 / 删除）暴露成 REST API
- **前端** `frontend/`：仿 RAGFlow 官网风格的 React 单页应用，知识库管理 + 问答对话

## 为什么仿 RAGFlow？

RAGFlow 是开源 RAG 引擎里前端做得最完整的一个：左侧导航 + 主区卡片 + 拖拽上传 + 文档表格 + 切片查看 + 检索测试。我们项目的核心场景和它重合（"管文档 → 切片入库 → 问答"），仿它的布局可以省掉大量交互设计成本，且观感专业。

技术栈也尽量对齐：RAGFlow 用 React + Ant Design，我们用 React 18 + Vite 5 + Ant Design 5。

## 架构

```
┌─────────────────────┐         ┌──────────────────────────┐
│  React Frontend     │  HTTP   │  FastAPI Backend          │
│  (frontend/, 5173)  │ ──────► │  (scripts/api.py, 8000)  │
│                     │ /api/*  │                          │
│  - Layout           │         │  - GET  /api/health       │
│  - KnowledgeBase    │         │  - GET  /api/stats        │
│  - Chat             │         │  - GET  /api/documents    │
│  - ChunkDrawer      │         │  - POST /api/documents/upload │
└─────────────────────┘         │  - GET  /api/documents/{id}/chunks │
        ▲                       │  - DELETE /api/documents/{id}     │
        │ Vite proxy /api →     │  - POST /api/ask                  │
        │  http://127.0.0.1:8000│  - GET  /api/traces               │
                                │  - GET  /api/llm-info             │
                                └────────────┬─────────────────────┘
                                             │
                                ┌────────────┴─────────────────────┐
                                │  core/* (现有 Python 核心)         │
                                │  - core.rag.generate_answer      │
                                │  - core.agent.run_self_heal      │
                                │  - core.health_check             │
                                │  - core.ingest.ingest_file       │
                                │  - core.storage.pg_store / vec   │
                                └──────────────────────────────────┘
```

## 后端：`scripts/api.py`

`serve.py` 是阶段 4 的最小服务（`/health` + `/ask`），保留不动；本阶段新增 `scripts/api.py` 作为**前端要打的全 API**，路径统一前缀 `/api/*`，区别于 serve.py。

### 接口表

| 方法 | 路径 | 作用 | 备注 |
|---|---|---|---|
| GET  | `/api/health` | 健康检查 + 存储条数 | PG 不可达也 200 + status=degraded |
| GET  | `/api/stats` | 总览（文档/块/向量/待审影子库数） | 首页统计卡用 |
| GET  | `/api/llm-info` | LLM 后端/模型名 | 顶栏状态展示，不暴露 key |
| GET  | `/api/documents` | 列出全部文档（带 chunk_count） | 按 created_at DESC |
| POST | `/api/documents/upload` | 批量上传文件（multipart field="files"） | 支持 md/txt/html/pdf/docx/...，可带 `strategy` 分块策略 |
| GET  | `/api/documents/{doc_id}` | 单文档详情 | 不存在返 404 |
| GET  | `/api/documents/{doc_id}/chunks` | 切片列表（分页） | `?limit=&offset=`，默认 50/0 |
| DELETE | `/api/documents/{doc_id}` | 删除文档 | PG 级联 + Milvus 同步删 |
| POST | `/api/ask` | 问答 | 含阶段 5/6/8 全部开关 |
| POST | `/api/retrieve` | **检索测试（只召回，不调 LLM）** | 看 top_k 块 + 分数，定位检索质量问题 |
| POST | `/api/health-check` | **阶段 7 知识库体检** | 返回 summary / markdown 报告 / 影子库；`skip_conflict` 可省 LLM |
| GET  | `/api/traces` | 最近 trace 记录 | 阶段 8 链路追踪页用 |

### 复用

所有接口都直接调 `core.*` 已有的函数（`generate_answer` / `run_self_heal` / `ingest_file` / `pg_store.*`），与 CLI / Streamlit / Swagger 完全等价，**没有重新实现 RAG 逻辑**。

## 前端：`frontend/`

### 目录

```
frontend/
├── package.json          依赖：react 18 / antd 5 / vite 5 / axios / react-router 6 / react-markdown
├── vite.config.js        dev server 5173 + /api 代理到 8000
├── index.html
├── README.md
└── src/
    ├── main.jsx          入口（ConfigProvider + Router）
    ├── App.jsx           路由（5 条）
    ├── api.js            axios 封装（baseURL=/api）
    ├── theme.js          AntD 主题（深蓝 #2a5298，仿 RAGFlow）
    ├── App.css           全局样式（含 Markdown 报告渲染样式）
    └── components/
        ├── Layout.jsx        顶栏 + 侧边栏（5 项菜单）+ Outlet
        ├── KnowledgeBase.jsx 知识库页：统计 + 上传（含分块策略）+ 文档表 + 切片 Drawer
        ├── Chat.jsx          问答页：消息列表 + 设置面板
        ├── Retrieval.jsx     检索测试页：只召回不调 LLM，看 top_k 块 + 分数
        ├── Health.jsx        体检页（阶段7）：四检测器概览 + Markdown 健康报告 + 影子库
        └── Traces.jsx        追踪页（阶段8）：trace 列表 + 展开看 prompt/答案/召回
```

侧边栏菜单分两组：日常使用（知识库 / 问答 / 检索测试）+ 运维诊断（知识库体检 / 链路追踪）。

### 知识库页（`/`、`/knowledge`）

仿 RAGFlow "Files" 标签页：

- **顶部统计卡**：文档数 / 切片数 / 向量数 / 待审影子库数
- **上传区**：AntD `Upload.Dragger`，支持拖拽 + 多选，支持 `.md/.txt/.html/.pdf/.docx/.json/.yaml/.csv` 等
  - `beforeUpload: () => false` 阻止自动上传（避免逐文件请求）
  - 右上角可**选分块策略**（递归字符 / Markdown 标题 / 固定长度），随 `FormData` 的 `strategy` 字段一起提交
  - "开始上传"按钮用 `XMLHttpRequest` 一次性 POST `FormData`，带**上传进度条**
  - 上传完按文件逐个展示成功/失败 + 切片数 + 字符数
- **文档表**：列 = 文档名（含 doc_id 短哈希）/ 切片数 / 字符数 / 状态徽标 / 来源徽标 / 创建时间 / 操作（查看切片、删除带确认）
- **切片 Drawer**：分页（每页 20），按 chunk_index 升序，每块展示编号、标题、字符位置、原文

### 问答页（`/chat`）

- **左侧设置面板**：检索模式（vector / hybrid）、top_k 滑块、阶段 5/6/8 各开关、save 开关
- **右侧对话区**：用户消息靠右蓝色气泡，助手消息靠左白底卡片
  - 答案下方展示**状态徽标**：阶段5自愈 / 阶段6自愈 / 拒答 / 反射通过/未过
  - 阶段 6 触发时展开过程明细：缺口判定 / 调研链接 / 质量门禁 / 影子库数
  - 召回块可折叠展开：每块显示分数、文档名、标题、未验证来源标签

### 检索测试页（`/retrieval`）

仿 RAGFlow 的 "Retrieval testing"：**只召回、不调 LLM**。

- 输入 query + 选检索模式（向量 / 混合）+ top_k 滑块（1~30）
- 结果按分数降序列出每个召回块：分数、文档名、标题、字符位置、原文
- 为什么单开一页？问答页只能看到"最终答案好/坏"，定位不到"检索本身好不好"。检索测试页把召回质量单独暴露出来。

### 知识库体检页（`/health`，阶段 7）

- 选项：跳过矛盾检测（省 LLM）、重复阈值滑块
- 点"运行体检" → `POST /api/health-check` → 跑四个检测器（过时 / 矛盾 / 僵尸 / 重复）
- 展示：4 个概览指标卡 + 分类计数（过时/矛盾/僵尸/重复 + 高/中/低危）+ **Markdown 健康报告**（`react-markdown` + `remark-gfm` 渲染，表格带边框）+ 待审影子库清单
- 强调"只读不写"：报告只是建议，删/改要人工执行

### 链路追踪页（`/traces`，阶段 8）

- 读 `trace_log`：表格列 = 时间 / 问题 / 检索模式 / 反射 / 核查通过 / 耗时 / 估算 token
- 点行展开看明细：request_id、召回块 ID 列表、召回分数、发给 LLM 的 prompt（前 1500 字）、答案
- 空态提示：去「问答」页勾选「阶段8 链路追踪落库」后问答，这里就有记录

### RAGFlow 风格细节

- 顶栏渐变深蓝（`#1e3a5f → #2a5298`），AntD 主色配 `#2a5298`
- 侧边栏白底、菜单选中态淡蓝
- 统计卡 + 上传区 + 文档表各为独立 Card，间距统一
- 状态徽标用 `status-active`（蓝）/ `status-staging`（橙）/ `status-archived`（灰）三色

## 运行

```bash
# 1) 后端（项目根）
python scripts/api.py
#   → http://127.0.0.1:8000   Swagger 文档在 /docs

# 2) 前端（新终端）
cd frontend
npm install      # 第一次
npm run dev      # → http://127.0.0.1:5173
```

前端通过 Vite proxy 把 `/api/*` 转给后端 8000，开发时**完全不需要 CORS**（虽然后端也开了 `*`）。

## 验证

| 项 | 结果 |
|---|---|
| 后端 `py_compile` | ✅ |
| 后端路由注册（12 个自定义 + 4 个 FastAPI 自动） | ✅ |
| 后端 headless 启动 + `/openapi.json` 200 + `/api/llm-info` 返配置 + CORS `*` | ✅ 2s |
| 后端 `POST /api/retrieve` 参数校验（坏 retrieval→400 / 缺参数→422） | ✅ |
| 前端 `npm install`（157 + 98 包 / npmmirror） | ✅ 1min + 4s |
| 前端 `npm run build`（3352 模块 → dist/） | ✅ 2.04s |
| 前端 dist/ 静态服务（index/JS/CSS 全 200）+ 5 页面字符串均在 bundle | ✅ |
| 端到端真问答 | ⚠ 需本机 PG 隧道 + DashScope key；逻辑与 CLI 共用 |

## 已知限制

- **v1 没做**：知识库（dataset）分组（需给 `document` 表加 `dataset_id` 字段 + 迁移）、文档解析的**异步进度条**（当前上传是同步返回，进度只反映"上传字节数"而非"解析/embedding 进度"）。分块策略已暴露在界面上（`strategy`）。
- **bundle 体积**：1.35MB（gzip 431KB），因为 AntD 全量打入；将来可 `manualChunks` 拆 antd 出去。
- **CORS**：`allow_origins=["*"]`，生产应收敛。
- **Streamlit UI** 保留不动，作为"快速问答"备选入口；前端是主入口。

## 收尾待办（需用户授权，Agent 未动）

- 提交上云：本阶段改动只落本地
- PG 历史明文密码 `rag_dev_2026`（3 个旧提交里还有）
