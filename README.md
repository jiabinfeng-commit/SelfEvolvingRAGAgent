# Self-Evolving RAG Agent — 语料与评测集（起步包）

> 这是项目 **阶段 0 的地基**：真实语料 + 真实评测集。
> 没有这两样，后面所有的"优化提升"都是自嗨。

---

## 一、为什么选 FastAPI 官方中文文档

| 评估维度 | 说明 |
| --- | --- |
| 中文 | 官方中文翻译，不是机翻，术语规范 |
| 权威 | fastapi/fastapi 官方仓库，`docs/zh/docs` |
| 结构化 | `tutorial/`、`advanced/`、`deployment/` 分层清晰，有标题层级，适合结构化切片实验 |
| **边界清晰** | 语料只覆盖 FastAPI 框架知识 → "知识库里没有" 的问题非常好构造 |
| **有权威可爬站点** | `fastapi.tiangolo.com/zh/` —— 这是阶段 6 Agent 自愈时的理想爬取源 |
| 会过时 | 版本演进快，适合阶段 7 的"过时内容检测" |
| 与你的方向契合 | Python Web + AI 应用开发，和你 Java→Python/AI 的转型叙事一致 |

### 语料规模

| 项目 | 数值 |
| --- | --- |
| 文档数 | 41 篇 |
| 清洗前 | 210,742 字符 |
| **清洗后** | **179,228 字符**（去掉 15.0% 噪音） |
| 单篇最大 | `deployment__docker.md` 25KB |
| 预计切片后 | 约 350-500 个 chunk（按 512 字/块估算） |

### 清洗了什么（`scripts/clean_corpus.py`）

| 噪音类型 | 数量 | 说明 |
| --- | --- | --- |
| `{* ../../docs_src/xxx.py *}` | 206 处 | mkdocs 代码引用占位符，实际代码未包含在 md 里 |
| `/// tip \| 提示`、`//// tab \| Python 3.10+` | 若干 | mkdocs 标记行 |
| `<div class="termy">`、`<details>` | 若干 | HTML 终端模拟块、折叠块 |
| `{#first-steps}` | 若干 | 标题锚点 |

> ⚠️ **已知限制**：因为代码是占位符引用，清洗后**文档里没有完整代码示例**。
> 所以"某某功能怎么写代码"这类问题答不了——这恰恰是 Agent 自愈场景的绝佳素材（见第四节）。

---

## 二、评测集：`eval/questions.json`

**50 条，全部经过脚本校验，答案关键词确认存在于语料中。**

| 类型 | 数量 | 说明 |
| --- | --- | --- |
| `single_hop` | 36 | 答案在单篇文档内可直接找到 |
| `multi_hop` | 6 | 需综合 2 篇以上文档 |
| `unanswerable` | 8 | **语料中确实没有答案**，用于测拒答率 |

分布比例：单跳 72% / 多跳 12% / 拒答 16%

### 题目示例

```json
{
  "id": 12,
  "question": "在 FastAPI 中添加 ASGI 中间件推荐用什么方法？为什么？",
  "gold_answer": "推荐使用 app.add_middleware()，第一个参数是中间件类...",
  "gold_doc": "advanced__middleware.md",
  "answer_keywords": ["add_middleware", "第一个参数是中间件的类", "服务器错误"],
  "type": "single_hop",
  "difficulty": "medium"
}
```

### 校验方式

```bash
python eval/verify_questions.py
```

输出示例：
```
  ✓ #12 single_hop   全部 3 个关键词命中 advanced__middleware.md
  ✓ #44 拒答题（已人工确认，语料仅提及 ['AWS']）
  ...
  single_hop     通过 36/36
  multi_hop      通过 6/6
  unanswerable   通过 8/8
  ✅ 全部评测题校验通过
```

校验逻辑：
1. **可答题**：`gold_doc` 必须存在，且 `answer_keywords` 全部能在语料中命中
2. **拒答题**：检查问题核心词是否意外出现在语料中（防误标），已逐条人工确认

---

## 三、目录结构

```
rag-demo/
├── README.md                    ← 本文件
├── data/
│   ├── raw/                     ← 原始文档（41 篇，未清洗）
│   └── clean/                   ← 清洗后文档 ← 阶段 1 从这里读
├── eval/
│   ├── questions.json           ← 评测集（50 条）
│   └── verify_questions.py      ← 校验脚本
└── scripts/
    ├── fetch_docs.py            ← 重新下载语料
    └── clean_corpus.py          ← 清洗语料
```

---

## 四、★ 8 条拒答题 = Agent 自愈的完美触发用例

这是这个起步包最有价值的设计：**拒答题不是用来为难系统的，是用来演示自愈的。**

| ID | 拒答问题 | 自愈时该去哪爬 |
| --- | --- | --- |
| 43 | 如何集成 Celery 实现异步任务队列？ | fastapi.tiangolo.com 或 Celery 文档 |
| 44 | 如何把上传文件存储到 AWS S3？ | boto3 / AWS 官方文档 |
| 45 | 如何用 Strawberry 实现 GraphQL？ | strawberry.rocks |
| 46 | 如何为接口添加限流？ | slowapi / 相关文档 |
| 47 | 如何实现多租户架构？ | 需综合多篇资料 |
| 48 | 如何连接 MongoDB？ | motor / mongodb 官方 |
| 49 | 如何配置 Prometheus 监控？ | prometheus 官方 + starlette exporter |
| 50 | 如何做 A/B 测试？ | 需综合资料 |

**完整演示流程**（阶段 6 做出来后就是这样的效果）：

```
1. 问："FastAPI 如何集成 Celery？"
   → 检索：最高分 0.31，低于阈值 0.5
   → 判定：知识缺口 ✓

2. Agent 自主调研
   → 搜索 "FastAPI Celery integration"
   → 抓取正文、去广告

3. 生成结构化文档《FastAPI 集成 Celery 指南.md》

4. 质量门禁打分：0.82 ≥ 0.7 ✓

5. 写入影子库（status='staging'），不污染主库

6. 再问同一个问题
   → 现在能答对了，并标注来源为"Agent 生成，待验证"
```

> 这个 demo 演示效果极佳：**从"答不了"到"自己学会"的全过程可视化**，
> 而且是 Dify / RAGFlow 完全做不到的（它们是固定工作流，不会自己决定"要不要去学"）。

---

## 五、下一步

### 立即做：初始化 Git 仓库

```bash
cd /Users/fengjiabin/WorkBuddy/2026-09-09-14-25-42/rag-demo
git init
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
.venv/
data/raw/
data/milvus.db
.env
EOF
git add .
git commit -m "chore: 初始化语料与评测集（41篇文档 + 50条评测题）"
git tag v0.0-corpus
```

> `data/raw/` 不进 Git（可由 `fetch_docs.py` 重新生成）；`data/clean/` 建议进库，保证可复现。

### 然后进入阶段 1

对照 `SelfEvolving-RAG-Agent-实施路线图.md` 的阶段 1：
1. 解析：`data/clean/` 下全是 markdown，直接读即可（**这一步你比别人轻松，不用啃 PDF**）
2. 切片：实现 `FixedChunker` + `StructuralChunker` 两种
3. 双存储：Milvus Lite（向量）+ PostgreSQL（chunk 原文）

---

## 六、复用与扩展

**换语料**：改 `scripts/fetch_docs.py` 里的 `TARGETS` 列表即可。
想换主题（比如换成 LangChain 文档），只要保证三点：
1. 有明确的领域边界
2. 有权威的可爬站点
3. 能构造出"确实答不了"的问题

**扩评测集**：往 `eval/questions.json` 里加，然后跑 `verify_questions.py` 校验。
建议最终规模 80-100 条（现在 50 条够起步）。

---

## 七、运行入口（阶段 1~5 全链路）

语料与评测集只是地基。真正能跑的 RAG Agent 在 `core/` + `scripts/` 里，三个入口共用同一份问答逻辑（`core/rag.generate_answer`）：

| 入口 | 文件 | 用途 | 启动命令 |
| --- | --- | --- | --- |
| 命令行 | `scripts/ask.py` | 终端问答 / 调试 | `python scripts/ask.py "FastAPI 怎么做依赖注入？" [--retrieval hybrid] [--self-heal]` |
| HTTP 服务 | `scripts/serve.py` | 阶段4 服务化（FastAPI） | `python scripts/serve.py` → http://127.0.0.1:8000/docs |
| 页面 | `scripts/ui.py` | 阶段5 可视化（Streamlit） | `streamlit run scripts/ui.py` → http://localhost:8501 |

前置（PG 走 SSH 隧道连云上）：
```bash
bash scripts/tunnel_pg.sh          # 开隧道（.env 里 PG_HOST=localhost）
```

各阶段详解见 `docs/`：
- `docs/01-阶段1-切片与双存储.md`
- `docs/02-阶段2-RAG闭环.md`
- `docs/03-阶段3-评估闭环.md`
- `docs/04-阶段4-检索优化与自进化闭环.md`（算法层：BM25 混合检索）
- `docs/05-阶段5-Agent自愈.md`（逻辑层：拒答检测 + 重试）
- `docs/06-阶段4-服务化.md`（服务层：FastAPI）
- `docs/07-阶段5-streamlit页面.md`（UI 层：Streamlit）

---

*生成时间：2026-09-09（地基）｜ 运行入口更新：2026-09-11（阶段1~5）| 语料来源：github.com/fastapi/fastapi @master (docs/zh/docs)*
