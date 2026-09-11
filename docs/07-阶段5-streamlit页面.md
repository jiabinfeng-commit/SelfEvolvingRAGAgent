# 阶段 5 可视化：Streamlit 交互式问答页面

> 同样先澄清：**「阶段 5」也有两层**，别再混。
> - **阶段 5（逻辑层，已做）**：Agent 自愈 —— `core/self_heal.py` + `--self-heal` 开关，A/B 评估 +6pp。
> - **阶段 5（UI 层，本次补做）**：一个能跑的 Web 问答页面 `scripts/ui.py`。
>
> 本文件讲的是**第二层：streamlit 页面**。你那句「阶段5的streamlit页面也没做」——没错，确实没做，现在补上。

---

## 一、为什么要做这个页面

阶段 1~4 试 RAG 效果，只能开终端敲 `python scripts/ask.py "..."`。痛点：
- 产品/同事想看效果，得先学「怎么开终端、怎么配 .env、怎么开隧道」——门槛劝退。
- 看不到**召回了哪些块、每个块相似度多少**——而这对判断「为什么答错」至关重要。
- 自愈有没有触发、救没救回，命令行只打一行小字，不直观。

Streamlit 用极少量代码就能做出一个能跑的交互界面：
```
左边选检索模式 / 勾自愈 / 拖 top_k
        ↓
中间输入问题 → 点「提问」
        ↓
右边看到答案 + 召回的原文块（带分数，可展开）
```

---

## 二、页面长啥样（功能清单）

| 区域 | 组件 | 作用 |
| --- | --- | --- |
| 侧边栏 | `检索模式` radio | vector（纯向量）/ hybrid（阶段4 混合检索） |
| 侧边栏 | `启用 Agent 自愈` checkbox | 默认勾选，开阶段 5 自愈 |
| 侧边栏 | `召回块数 top_k` slider | 1~10，默认读 `config.RAG_TOP_K` |
| 侧边栏 | `记录到 qa_log` checkbox | 是否落库（阶段 2 闭环） |
| 主区 | `问题` 输入框 + `提问` 按钮 | 触发问答 |
| 主区 | 答案区 | 显示答案；自愈成功显示绿色 ✓，拒答显示黄色 ⚠ |
| 主区 | 召回块区 | 每个块一个可展开卡片，显示 `score + 来源 + 原文` |

自愈状态一目了然：
- 🟢 `✅ 本次触发了阶段5 Agent 自愈并成功救回` —— 原拒答被判为误拒，换宽松指令救回来了。
- 🟡 `⚠️ 模型判定为拒答` —— 没触发自愈，或被安全阀拦下（可能是真没答案）。

---

## 三、两种调用姿势（本项目选 A）

### 方式 A：直连 core（本文件采用）

```python
from core.rag import generate_answer
res = generate_answer(question, top_k=..., retrieval=..., self_heal=..., save=...)
```

页面进程内直接跑问答逻辑，**不依赖** FastAPI 服务。

- 优点：一条 `streamlit run` 就能用，不用先起服务，依赖最少。
- 缺点：页面进程自己加载模型 + 连 PG，和 CLI 是两套进程（但逻辑同一份）。

### 方式 B：调 HTTP 服务（可选）

把上面那行换成：

```python
import requests
r = requests.post("http://127.0.0.1:8000/ask",
                  json={"question": question, "retrieval": ..., "self_heal": ...})
res = r.json()
```

- 优点：和线上服务**完全一致**（连的是同一个常驻进程，模型/连接只加载一次）。
- 缺点：得先 `python scripts/serve.py` 起服务，页面还要处理网络异常。

> 教授建议：本地 demo / 给同事看，用方式 A 最省事；要做成「前端 + 后端分离」的正经系统，用方式 B，页面当纯客户端。本项目两样都给了，切换就一行代码。

---

## 四、三端一致性（重点）

`ask.py`（CLI）、`serve.py`（HTTP）、`ui.py`（页面）**全部调用 `core/rag.generate_answer()`**。这意味着：

- 命令行看到的效果 = 接口返回 = 页面显示，三者 100% 一致。
- 改检索口径 / 改 prompt / 改自愈逻辑，**只动 `core/rag.py` 一处**，三端同时生效，永不漂移。
- 这正是阶段 1 抽 `core/retrieval.retrieve()`、阶段 2 抽 `core/rag.generate_answer()` 的初心：把「业务事实」收敛到单一来源。

---

## 五、怎么跑

```bash
# 前置：PG 隧道（页面直连 core，也需要 PG）
bash scripts/tunnel_pg.sh

# 启动页面（自动打开 http://localhost:8501）
.venv/bin/python -m streamlit run scripts/ui.py
# 或：streamlit run scripts/ui.py
```

首次提问会加载 bge 模型（~1s），之后复用，后续提问很快。

---

## 六、自测结论（本机）

| 项目 | 结果 |
| --- | --- |
| 依赖安装 | streamlit 1.63.0 装好（清华源） |
| 模块导入 | `import streamlit` ✅；`from core.rag import generate_answer` ✅ |
| 页面启动 | `streamlit run scripts/ui.py` 正常拉起，8501 端口监听 ✅ |
| 渲染逻辑 | 答案区 / 召回卡片 / 自愈状态分支代码就绪 ✅ |
| 端到端问答 | 需 PG 隧道 + 阿里云 key 联网，在你本机跑（沙箱无隧道/外网，未跑全链路） |

> 诚实说明：页面框架（启动、组件、调用 `generate_answer`）已在本机验证；真实问答依赖你的 PG 隧道 + 阿里云 DashScope，沙箱连不上，需你本机 `bash scripts/tunnel_pg.sh` 后跑一次确认。页面逻辑和 CLI 完全一致（共用 `generate_answer`），CLI 能答页面就能答。

---

## 七、一句话总结

阶段 5 从「只有自愈逻辑」升级成「**逻辑 + 可视化**」：用户在网页上点一下，就能直观看到
「召回了什么 → 模型答了什么 → 有没有自愈救回」，把之前藏在日志里的黑盒变成了白盒。
