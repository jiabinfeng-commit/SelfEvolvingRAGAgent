# Frontend — Self-Evolving RAG Agent

仿 RAGFlow 风格的前端页面：知识库管理（上传 / 列表 / 切片查看 / 删除）+ 问答对话 + 检索测试 + 知识库体检（阶段7）+ 链路追踪（阶段8）。

## 技术栈
- React 18 + Vite 5
- Ant Design 5（UI 组件库，RAGFlow 同款）
- React Router 6
- Axios（HTTP 客户端）

## 运行

### 1. 启动后端
```bash
# 在项目根
python scripts/api.py
# 默认 http://127.0.0.1:8000
# Swagger 自带文档： http://127.0.0.1:8000/docs
```

### 2. 启动前端
```bash
cd frontend
npm install      # 第一次需要
npm run dev      # 默认 http://127.0.0.1:5173
```
浏览器自动打开 `http://127.0.0.1:5173`，会通过 Vite proxy 把 `/api/*` 转到后端 8000。

### 3. 生产构建
```bash
npm run build    # 产物在 frontend/dist/
npm run preview  # 预览构建结果
```

## 页面
| 路由 | 页面 | 说明 |
| --- | --- | --- |
| `/knowledge` | 知识库 | 统计卡 + 拖拽上传（可选分块策略）+ 文档表 + 切片抽屉 |
| `/chat` | 问答 | 对话 + 检索/自愈设置（阶段 5/6/8 开关） |
| `/retrieval` | 检索测试 | 只召回不调 LLM，看 top_k 块 + 分数 |
| `/health` | 知识库体检 | 阶段 7 四检测器 + Markdown 健康报告 + 影子库 |
| `/traces` | 链路追踪 | 阶段 8 trace 列表 + 展开明细 |

## 目录
```
frontend/
├── package.json
├── vite.config.js          # dev server + /api 代理
├── index.html
└── src/
    ├── main.jsx            # 入口
    ├── App.jsx             # 路由（5 条）
    ├── api.js              # axios 封装（/api 前缀）
    ├── theme.js            # AntD 主题（仿 RAGFlow 蓝）
    ├── App.css             # 全局样式（含 Markdown 报告样式）
    └── components/
        ├── Layout.jsx      # 顶栏 + 侧边栏（5 项菜单）+ 内容区
        ├── KnowledgeBase.jsx   # 知识库页
        ├── Chat.jsx        # 问答页
        ├── Retrieval.jsx   # 检索测试页
        ├── Health.jsx      # 体检页（阶段7）
        └── Traces.jsx      # 追踪页（阶段8）
```

## 后端 API 对照
前端调的全部接口都在 `scripts/api.py`（12 个），启动后看 `http://127.0.0.1:8000/docs`。
关键几个：`POST /api/documents/upload`（上传）、`DELETE /api/documents/{id}`（删）、`POST /api/ask`（问答）、`POST /api/retrieve`（检索测试）、`POST /api/health-check`（体检）、`GET /api/traces`（追踪）。
