# 阶段 6：Agent 自愈闭环（知识缺口自愈 + 影子库防投毒）

> **这是整个项目的灵魂，也是"Dify / RAGFlow 做不到"的核心差异点。**
> 前面 1~5 阶段都是"被动"的：你问、它答，答不上来就老实拒答。
> 阶段 6 让系统"主动"——发现知识缺口时自己联网查、自己整理成知识条目、自己过质量门禁、再写回知识库；下次再问同一个问题就能答上来。这就是项目名 "Self-Evolving"（自进化）真正落地的地方。

---

## 1. 一句话讲清它干了什么

```
用户问一个知识库里没有的问题
   → 阶段5 库内自救（换宽松 prompt 重试一次）
   → 还答不上来（确定是真缺口）
   → 联网搜索 + 抓正文（去广告/导航）
   → LLM 把爬到的内容整理成结构化知识条目
   → LLM 质量门禁打分（相关性/事实性/无冲突/信息密度）
   → 门禁通过 → 写进"影子库"（staging，不进主库）
   → 再用【全新】向量实例重新问答 → 这次能答上来了
```

Demo 命令（知识库里没有的问题）：

```bash
python scripts/heal_knowledge.py "FastAPI 的 BackgroundTasks 怎么用"
# 或限定只爬官方文档，更准也更安全：
python scripts/heal_knowledge.py "Flask 和 FastAPI 的区别" --site docs.python.org
# 或直接用 ask.py 的开关：
python scripts/ask.py "某个我库里没有的概念" --agent-heal
```

---

## 2. 和阶段 5 的关系（递进，不是重复）

路线图里"自愈"这个词出现了两次，容易混，这里钉死边界：

| | 阶段 5（core/self_heal.py） | 阶段 6（core/agent.py，本阶段） |
| --- | --- | --- |
| 触发 | 模型拒答 **且** 召回分 ≥ 0.5 | 阶段 5 也没救回来（真缺口） |
| 范围 | **只**在已有语料内部自救 | 出去**联网**调研，补语料 |
| 动作 | 换宽松 prompt + 扩大召回重试 | 搜索→抓正文→生成→门禁→写影子库 |
| 是否碰外部 | 否 | 是 |
| 本质 | 把"模型过度保守误拒"救回来 | 把"语料真的缺知识"补上 |

所以 `run_self_heal()` **第一步就调 `generate_answer(self_heal=True)`**，先把阶段 5 用上；只有阶段 5 也没救回来，才进入本阶段的联网调研分支。两者是"先库内、后库外"的递进。

---

## 3. 设计取舍：为什么【没有】用 LangGraph

路线图 6.1 书面建议用 LangGraph 画状态机。实际落地**没有引 LangGraph**，和阶段 2~5 一致保持"纯标准库、零重依赖"：

1. **流程是线性的、可穷举的**——检索→缺口判定→调研→生成→门禁→入库。一个 `while` 循环 + 状态字典就表达得清清楚楚，对 Java 背景的你更透明、可断点调试，不黑盒。
2. **死循环风险**用 `max_steps` 计数器就解决了（路线图风险清单专门点名这条），不需要为这点控制流引一整套图编排框架。
3. **可插拔**：以后真想可视化编排，把 `run_self_heal()` 内部换成 LangGraph 实现即可，对外接口（`question` 进、`结果 dict` 出）完全不变。

> 简历写法仍可写"Agent 状态机编排"；若面试官问为何不用 LangGraph，这正是体现工程判断力的点："功能该有的都有，但用更轻可控的方式实现，不为了显得高级而加重依赖。"

---

## 4. 模块 API（core/agent.py）

| 函数 | 职责 | 说明 |
| --- | --- | --- |
| `web_search(query, top_n, site, timeout)` | 联网搜索 | 纯 urllib 打 DuckDuckGo HTML 版，**免 Key、免付费**；失败返回 `( [], 原因 )` 优雅降级 |
| `fetch_page(url, timeout)` | 抓正文 | urllib GET + `clean_html` 去广告/导航 |
| `clean_html(html)` | HTML→纯文本 | 标准库 `HTMLParser`，跳过 `<script>/<style>`；如需更干净的"文章提取"可换 trafilatura，接口不变 |
| `gap_detected(answer, retrieved)` | 缺口判定 | 空召回 / 模型拒答 / 最高相似度<0.5 任一即判定有缺口 |
| `draft_document(question, crawled, llm)` | 生成知识条目 | 把爬到的正文整理成 Markdown，要求"只基于参考、标注来源" |
| `quality_gate(draft, question, llm, existing)` | 质量门禁 | LLM 四维度自评 JSON → 加权分 ≥ 0.7 才通过；**任何异常都判不通过** |
| `ingest_shadow(pg, vec, emb, doc_id, doc_name, content)` | 写影子库 | 双写 PG(status=staging, source=agent_generated) + 向量 |
| `run_self_heal(question, ...)` | 总编排 | 阶段 6 状态机入口，返回结构化结果 dict |

`run_self_heal` 返回 dict 关键字段：

```python
{
  "question": ..., "answer": ..., "refrieved": [...], "refusal": bool,
  "gap_detected": bool, "gap_reason": str,
  "searched": [urls],                 # 调研抓过的链接
  "draft": str,                       # 生成的草稿
  "quality_pass": bool, "quality_score": float, "quality_detail": {...},
  "ingested_chunks": int, "shadow_doc_id": str,
  "self_healed_by_agent": bool,       # 是否走了联网补库
  "steps": [str, ...],                # 可观测轨迹
}
```

> **"再问就能答"的实现坑点**：`ingest_shadow` 把新块写进同一个向量文件，但同一进程里已 `load` 的向量实例可能看不见刚 upsert 的数据。所以重新问答时传 `vec=None`，让 `generate_answer` 新建一个 `VecStore` 从文件重新 load——保证一定读到刚写入的 staging 块。

---

## 5. 防投毒设计（这比功能本身更值钱）

"自动把模型生成的内容写回知识库"是个高危动作——模型可能编造、可能抓到谣言。双重保险：

1. **影子库（staging）**：新内容先以 `status='staging'`、`source='agent_generated'` 入库，**不进主库（active）**。检索时能召回，但 `run_self_heal` 会用 `_annotate()` 给这些块打 `unverified=True` 标签，前端/CLI 据此提示"这条来自机器自爬、尚未人工审核"。**能召回，但明确告诉你它没被审过**——这是防投毒的"透明化"一环。
2. **LLM 质量门禁**：写库前四维度自评，加权分 < 0.7 一律不准入库。四项权重：`relevance 0.4 / factuality 0.3 / conflict 0.2 / info_density 0.1`。

> "转 active"留给阶段 7 体检 / 人工审核（人工确认、或被正确引用足够多次后），本阶段只负责"写 staging"，不越权。

---

## 6. 指标采集（简历硬通货）

批量模式会把结果汇总成 `heal_report.md`，直接给出两项数字：

```bash
python scripts/heal_knowledge.py --file questions.txt
```

- **自愈成功率** = Agent 自愈成功（写影子库且最终答出）的问题数 / 总问题数
- **脏数据拦截率** = 质量门禁不通过（被拦下）的草稿数 / 总草稿数

> 简历写法："设计影子库 + LLM 质量门禁双重校验，脏数据拦截率 XX%，解决自动入库导致的知识库投毒问题"

---

## 7. 自测方法

### 7.1 纯函数（不依赖 PG / 模型 / 外网之外）
`scripts/selftest_stage6.py` 做两层验证：
- **联网工具**：对真实互联网测 `web_search` / `fetch_page` / `clean_html`（沙箱已验证可联网）。
- **状态机（桩）**：用桩 LLM + 桩 pg/vec 跑完整闭环——注入"首轮拒答 + 门禁通过"，验证"缺口→调研→草稿→门禁通过→影子库入库→再问能答"全链路。

```bash
python scripts/selftest_stage6.py
```

### 7.2 端到端真问答（需你本机）
需要：
1. 阿里云 PG 隧道（`bash scripts/tunnel_pg.sh`）——沙箱无隧道；
2. DashScope API Key（在 `.env` 配好）——走 `config.LLM_BACKEND=openai`；
3. 本机能联网（爬取依赖外网，但**不需要 VPN**，只爬公开页面）。

跑一个知识库里确实没有的问题，观察它是否自动走完自愈闭环。

---

## 8. 已知限制 / 合规

- **搜索稳定性**：DuckDuckGo HTML 版无 Key、免付费，但某些网络环境会被限流；生产中若要稳定可换商业搜索 API（改 `web_search` 内部即可，接口不变）。
- **正文清洗**：自带 `clean_html` 已能去广告/导航大半，若要文章级提取可换 trafilatura（路线图书面建议），属可选升级。
- **合规**：只爬公开页面，README 注明用途为学习；不抓需登录/有版权的内容。
- **模型配置**：严格遵守"后续阶段不改模型配置"纪律——本阶段不碰 `LLM_MODEL` / key / 后端，全部走 `config`。
