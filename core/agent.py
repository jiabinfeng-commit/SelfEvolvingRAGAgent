# -*- coding: utf-8 -*-
"""
core/agent.py —— 阶段 6：Agent 自愈闭环（知识缺口自愈 + 影子库防投毒）

====================================================================
这一段为什么是整个项目的"灵魂"
====================================================================
前面 1~5 阶段做的都是"被动"的：你问、它答，答不上来就老实拒答。
阶段 6 要让它"主动"——发现自己知识有缺口时，自己出去查、自己把查到的东西
整理成结构化文档、自己过一道质量门禁、再写回知识库。下次再问同一个问题，
它就能答上来了。这就是项目名 "Self-Evolving"（自进化）真正落地的地方，
也是 Dify / RAGFlow 这类"画死的工作流"做不到的核心差异点。

====================================================================
为什么这里【没有】用 LangGraph（路线图书面建议用）
====================================================================
路线图 6.1 写的是用 LangGraph 画状态机。但实际落地的取舍是——【不引 LangGraph】，
原因和阶段 2~5 保持一致（项目一贯保持依赖精简、纯标准库 urllib、零重依赖）：

1. 本阶段的流程是线性的、可穷举的：
       检索 → 缺口判定 →（有缺口）调研 → 生成 → 质量门禁 → 入库
   它不像"按需动态分支的复杂 Agent"，用一个简单的 while 循环 + 状态字典就表达得
   清清楚楚，对 Java 背景的你来说也更透明、可断点调试，不黑盒。
2. 路线图风险清单明确点名"Agent 死循环"风险。我们用一个 max_steps 计数器就解决了，
   不需要为这点控制流去引一整套图编排框架。
3. 以后若真想可视化编排，把 run_self_heal() 内部换成 LangGraph 实现即可，
   对外接口（输入 question、输出结果 dict）完全不变——这才是"可插拔"的正确姿势。

一句话：功能该有的都有，只是用更轻、更可控的方式实现，不为了"显得高级"而加重依赖。

====================================================================
和【阶段 5】的关系（重要，别混淆）
====================================================================
- 阶段 5 的自愈（core/self_heal.py + rag.generate_answer 的 self_heal 开关）：
  只在"已有语料内部"自救——见到拒答且召回分够高，就换宽松 prompt + 扩大召回重试一次。
  它【不碰外部世界】，本质是把"模型过度保守误拒"救回来。
- 阶段 6 的 Agent 自愈（本模块）：当语料里【确实没有答案】时，才出去联网调研，
  并把新知识写回语料。两者是"递进"关系：先 5（库内自救）→ 再 6（库外补库）。
  所以 run_self_heal 第一步就调 generate_answer(self_heal=True)，把阶段 5 用上；
  只有阶段 5 也没救回来，才进入本模块的联网调研分支。

====================================================================
防投毒设计（这是 6 比功能本身更值钱的地方）
====================================================================
"自动把模型生成的内容写回知识库"是个高危动作——模型可能编造、可能抓到谣言。
所以双重保险：
  1) 影子库（staging）：新内容先以 status='staging'、source='agent_generated' 入库，
     不进主库（active）。检索时它能召回，但会被打上"未验证来源"标签（见 _annotate），
     让人一眼看出现在这条答案来自"机器自己爬的、还没人审过"。
  2) LLM 质量门禁：写库前让模型从 相关性/事实性/无冲突/信息密度 四个维度自评打分，
     加权分 < 0.7（QUALITY_THRESHOLD）一律不准入库。
只有人工确认、或被正确引用足够多次后，才把 staging 转 active（本模块只负责"写 staging"，
"转 active"留给阶段 7 体检 / 人工审核，这里不越权）。

====================================================================
对外暴露
====================================================================
- web_search / fetch_page / clean_html        ：联网调研三件套（纯标准库 urllib）
- gap_detected(answer, retrieved)             ：缺口判定（规则，主）
- draft_document(question, crawled, llm)      ：把爬到的内容整理成知识条目
- quality_gate(draft, question, llm, existing)：LLM 质量门禁（返回 通过?/分数/明细）
- ingest_shadow(pg, vec, emb, ...)            ：写影子库（双存储：PG + 向量）
- run_self_heal(question, ...)                ：阶段 6 总编排（状态机入口）
"""
import os
import re
import json
import hashlib
import html
import socket
import urllib.parse
import urllib.request
import urllib.error
from html.parser import HTMLParser
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from core import config
from core.chunker import get_chunker
from core.embedder import get_embedder
from core.storage.pg_store import PGStore
from core.storage.vec_store import VecStore
from core import llm as llm_mod
from core.rag import generate_answer
from core.self_heal import is_refusal


# ================================================================
# 可调参数（集中放顶部，方便调参 / 写进简历的"拦截率/成功率"对比）
# ================================================================

# 缺口阈值：首轮召回最高相似度低于这个值，就认为"语料里大概率没有答案"。
# 与阶段 5 的 SELF_HEAL_SCORE_THRESHOLD=0.5 保持一致，口径统一。
GAP_THRESHOLD = 0.5

# 质量门禁通过线：四项加权分 >= 0.7 才允许写影子库（路线图 6.4 的硬指标）。
QUALITY_THRESHOLD = 0.7

# 质量门禁四项权重（加起来 = 1）。相关性最重要，无冲突性次之，信息密度权重最低。
QUALITY_WEIGHTS = {
    "relevance": 0.40,     # 是否真的回答了原问题
    "factuality": 0.30,    # 内容自洽、无明显错误、不编造
    "conflict": 0.20,      # 与已有知识库是否冲突（1=完全不冲突）
    "info_density": 0.10,  # 是否大段废话（1=信息密集）
}

# 喂给"生成/门禁"的爬取正文上限（字符）。控制 token 消耗，也避免塞进噪声。
MAX_CRAWL_CHARS = 6000

# 默认浏览器 UA：很多站点对 Python urllib 默认 UA 直接挡，伪装一下提高成功率。
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ---------- 联网可达性探测（阶段 6 联网调研的前置闸门） ----------
# 背景（实测踩过的坑）：国内网络访问 duckduckgo / google 这类境外站点时，TCP 握手
# 得不到任何响应 —— 连接停在 SYN_SENT，**既连不上也不被拒绝**。于是
# urllib 的 timeout 只能靠等待耗完，一轮 10s、max_steps 轮就是几十秒白等，
# 还会把整个 Web 服务的线程池拖住（前端表现为接口全部超时）。
# 所以进"联网调研循环"之前先做一次**带超时的 TCP 探测**：
#   不通 → 直接跳过调研、保持拒答，几秒内就返回明确原因。
# 探测目标可在 .env 里用 NET_PROBE_HOST / NET_PROBE_PORT 覆盖（换了搜索后端就改它们）。
NET_PROBE_HOST = os.getenv("NET_PROBE_HOST", "html.duckduckgo.com")
NET_PROBE_PORT = int(os.getenv("NET_PROBE_PORT", "443"))
NET_PROBE_TIMEOUT = float(os.getenv("NET_PROBE_TIMEOUT", "3"))


# ================================================================
# 联网调研三件套（纯标准库 urllib，不引 requests）
# ================================================================

@dataclass
class SearchHit:
    """一条搜索结果"""
    title: str
    url: str
    snippet: str = ""


def _strip_tags(s: str) -> str:
    """剥掉 HTML 标签 + 反转义实体 + 压平空白。给搜索结果标题/摘要用。"""
    s = re.sub(r"<[^>]+>", " ", s)        # 去标签
    s = html.unescape(s)                   # &amp; → &, &#39; → ' 等
    s = re.sub(r"\s+", " ", s).strip()     # 多空白压成一个空格
    return s


def _extract_ddg_url(href: str) -> Optional[str]:
    """
    从 DuckDuckGo 的跳转链接里解出真实 URL。

    DDG HTML 版的每条结果 href 不是直链，而是形如：
        /l/?uddg=https%3A%2F%2Fexample.com%2Fpage&amp;rut=...
    真实地址在 uddg= 这个参数里（URL 编码）。我们把它解出来即可。
    """
    if href.startswith("http://") or href.startswith("https://"):
        return href
    m = re.search(r"uddg=([^&]+)", href)
    if m:
        return urllib.parse.unquote(m.group(1))
    return None


def probe_net(host: str = None, port: int = None,
              timeout: float = None) -> Tuple[bool, str]:
    """
    快速探测外网是否可达。返回 (是否可达, 说明文字)。

    只做 **TCP 三次握手**，不发 HTTP 请求 —— 目的是判断"路通不通"，
    不是判断"对方站点好不好"。这样：
      · 更快：一次 connect 拿结果，不用等完整 HTTP 往返
      · 更准：即便对方返回 403/503，也说明网络是通的（那是另一回事）
      · 更省：不消耗对方配额，也不受对方限流影响

    为什么需要它：见文件上方 NET_PROBE_* 的注释 —— 境外站点在国内会卡在
    SYN_SENT，不探测的话只能在调研循环里按 timeout 硬等，白白拖垮请求。
    """
    h = host or NET_PROBE_HOST
    p = port or NET_PROBE_PORT
    t = timeout or NET_PROBE_TIMEOUT
    try:
        # create_connection 自带超时；连上后 with 退出即关闭，不留连接
        with socket.create_connection((h, p), timeout=t):
            return True, f"{h}:{p} 可达"
    except Exception as e:
        # 超时 / DNS 失败 / 连接被拒 都归为"不可达"，具体类型写进说明里便于排查
        return False, f"{h}:{p} 不可达（{type(e).__name__}: {e}）"


def web_search(query: str, top_n: int = 5, site: str = None,
               timeout: int = 10) -> Tuple[List[SearchHit], Optional[str]]:
    """
    通用网页搜索（无 API Key，走 DuckDuckGo HTML 版）。

    :param query: 搜索词
    :param top_n: 最多返回几条
    :param site:  限定站点域名，如 "docs.python.org"（拼成 "query site:xxx"）
    :param timeout: 网络超时（秒）
    :return: (SearchHit 列表, 错误信息或 None)。失败时返回空列表 + 原因，
             调用方据此优雅降级（不抛异常，避免自愈循环崩掉）。

    为什么用 DDG HTML 而不是某家搜索 API？
    - 免 Key、免付费、纯标准库就能打，符合项目"依赖精简"的纪律；
    - 牺牲点：稳定性不如商业 API，某些网络环境会被限流。所以这里失败时
      只返回空 + 原因，由上层决定是否换 query 重试。
    """
    q = f"{query} site:{site}" if site else query
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(q)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            page = r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return [], f"搜索请求失败: {e}"

    # 1) 抓标题链接（结果__a 是 DDG 标题锚点的固定 class）
    hits: List[SearchHit] = []
    for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"', page):
        href = m.group(1)
        seg = page[m.end(): m.end() + 400]          # 标题文本就在 href 之后一小段
        tm = re.search(r">(.*?)</a>", seg, re.S)
        title = _strip_tags(tm.group(1)) if tm else ""
        real = _extract_ddg_url(href)
        if real:
            hits.append(SearchHit(title=title, url=real))
        if len(hits) >= top_n:
            break

    # 2) 抓摘要（result__snippet 是 DDG 摘要的固定 class），按出现顺序配对
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
    snippets = [_strip_tags(s) for s in snippets]
    for i, h in enumerate(hits):
        h.snippet = snippets[i] if i < len(snippets) else ""

    if not hits:
        return [], "DDG 未返回结果（可能被限流或当前网络受限）"
    return hits, None


class _HtmlTextExtractor(HTMLParser):
    """
    把 HTML 正文抽成纯文本（仅用标准库，不引 trafilatura / readability）。

    做法和 core/chunker.py 里的 _HtmlTextExtractor 同源思路：
    遇 <script>/<style>/<noscript> 跳过其内容（导航栏/广告/统计代码大多在这几类里），
    遇块级标签补一个换行保留段落感。清洗后的文本已经能把导航/广告滤掉大半，
    后续若想要更干净的正文（trafilatura 那种文章提取），在 fetch_page 里换实现即可，
    对外接口不变。
    """
    _BLOCK = ("p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
              "tr", "br", "section", "article")

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1
            return
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            self.parts.append(data)


def clean_html(html_text: str) -> str:
    """HTML → 纯文本：抽正文、去脚本样式、压平多余空行与空格。"""
    ex = _HtmlTextExtractor()
    ex.feed(html_text or "")
    text = "".join(ex.parts)
    text = re.sub(r"\n{3,}", "\n\n", text)     # 3+ 空行压成 2 个
    text = re.sub(r"[ \t]{2,}", " ", text)      # 连续空格压成 1 个
    return text.strip()


def fetch_page(url: str, timeout: int = 10) -> Tuple[str, Optional[str]]:
    """
    抓取指定 URL 的正文（去广告/导航后的纯文本）。

    :return: (清洗后的正文, 错误信息或 None)。失败返回空串 + 原因。
    """
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            enc = r.headers.get_content_charset() or "utf-8"
            html_text = raw.decode(enc, errors="ignore")
    except Exception as e:
        return "", f"抓取失败: {e}"
    return clean_html(html_text), None


# ================================================================
# 缺口判定（阶段 6.2）
# ================================================================

def gap_detected(answer: str, retrieved: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    判定"语料里是否真的答不上来"（即存在知识缺口）。

    触发调研的三条规则（任一满足即认为有缺口）：
      ① 召回结果为空         —— 压根没召回到东西，肯定没答案
      ② 模型拒答             —— 阶段 5 库内自救都没救回来，说明不是"模型保守"而是"真没有"
      ③ 最高相似度 < 阈值    —— 召回到的东西和问题的相关度太低，不足为信

    设计上【保守】：默认只在"拒答 / 空召回"时才去联网，避免每次都爬、把噪声灌进知识库。
    注意和阶段 5 的衔接：本函数是在 generate_answer(self_heal=True) 之后调用的，
    也就是说"模型过度保守误拒"那种情况，阶段 5 已经用宽松 prompt 救回来了，
    能走到这里还拒答的，基本就是真缺口。

    :param answer:    首轮（含阶段5自愈）的最终答案
    :param retrieved: generate_answer 返回的召回列表，每项含 "score" 和 "doc_id"
    """
    if not retrieved:
        return True, "召回为空（语料中无相关内容）"
    top = retrieved[0]["score"]
    if is_refusal(answer):
        return True, "模型判定为拒答（阶段5 库内自救未解决 → 疑为真缺口）"
    if top < GAP_THRESHOLD:
        return True, f"最高相似度 {top:.3f} 低于阈值 {GAP_THRESHOLD}（召回信心不足）"
    return False, "已可作答（召回充分且模型给出了答案）"


# ================================================================
# 文档生成（阶段 6.3 调研 → 6.4 生成）
# ================================================================

_DRAFT_SYSTEM = (
    "你是一个严谨的知识库编辑。请根据【参考资料】为用户问题撰写一份"
    "简洁、准确、结构化的知识条目（Markdown 格式）。\n"
    "要求：\n"
    "1. 只基于参考资料，绝不编造参考资料里没有的内容；\n"
    "2. 直接回答问题，用分点/小标题组织，方便日后检索复用；\n"
    "3. 末尾用 '- 来源: <url 或 参考标题>' 标注出处；\n"
    "4. 使用中文。"
)

_DRAFT_USER_TMPL = (
    "用户问题：{question}\n\n"
    "参考资料（来自联网检索，已清洗掉广告/导航）：\n{crawled}\n\n"
    "请撰写知识条目："
)


def draft_document(question: str, crawled: str, llm) -> str:
    """
    把爬取并清洗后的正文，整理成一份可入库的知识条目（Markdown）。

    :param crawled: 清洗后的爬取正文（多个页面拼起来，已截断到 MAX_CRAWL_CHARS）
    :param llm:     任意 BaseLLM 实例（提供 generate(system, user)）
    :return:        生成的 Markdown 文本；调用失败返回空串（上层门禁会判不通过）
    """
    user = _DRAFT_USER_TMPL.format(question=question, crawled=crawled or "（无）")
    try:
        return (llm.generate(_DRAFT_SYSTEM, user) or "").strip()
    except Exception as e:
        # 生成失败 → 返回空，质量门禁会因内容空而判不通过，安全失败（不入库）
        return ""


# ================================================================
# 质量门禁（阶段 6.4，防投毒核心）
# ================================================================

_GATE_SYSTEM = (
    "你是知识库质量门禁。请评估下面这份【待入库草稿】能否作为知识库条目。"
    "从四个维度各打 0~1 分（1 最好）：\n"
    "- relevance    相关性  ：是否真正回答了原问题\n"
    "- factuality   事实性  ：内容自洽、无明显错误、不编造\n"
    "- conflict     无冲突性：与【已有知识库片段】相比是否冲突（1=完全不冲突）\n"
    "- info_density 信息密度：是否大段废话（1=信息密集）\n"
    "只输出一个 JSON，不要任何解释文字，格式：\n"
    '{"relevance":0.0,"factuality":0.0,"conflict":0.0,"info_density":0.0,"reason":"一句话理由"}'
)

_GATE_USER_TMPL = (
    "原问题：{question}\n\n"
    "已有知识库相关片段（用于比对冲突）：\n{existing}\n\n"
    "待评估草稿：\n{draft}"
)


def _parse_quality_json(text: str) -> Optional[dict]:
    """从 LLM 输出里抠出 JSON（兼容 ```json 代码块包裹 / 前后多余文字）。"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None


def _fmt_existing(retrieved: List[Dict[str, Any]], limit: int = 2) -> str:
    """把已有召回片段格式化成门禁比对用的文本（只取前几条，控长度）。"""
    if not retrieved:
        return "（无，无法比对冲突）"
    lines = []
    for i, c in enumerate(retrieved[:limit], 1):
        content = (c.get("content") or "")[:300]
        lines.append(f"[{i}] {c.get('doc_name','')}：{content}")
    return "\n".join(lines)


def quality_gate(draft: str, question: str, llm,
                 existing: List[Dict[str, Any]] = None) -> Tuple[bool, float, dict]:
    """
    LLM 质量门禁：评估草稿是否值得入库。

    :param draft:    待入库草稿（draft_document 的产物）
    :param question: 原问题（用于判断相关性）
    :param llm:      BaseLLM 实例
    :param existing: 已有知识库召回片段（用于比对冲突），可空
    :return: (是否通过, 加权总分, 明细dict)。

    失败闭环（安全优先）：
    - LLM 调用异常 → 不通过、分 0；
    - 返回的不是合法 JSON → 不通过、分 0（"解析不了就别信"）；
    - 加权分 < QUALITY_THRESHOLD → 不通过。
    任何异常都"判不通过"，绝不让脏数据溜进影子库。
    """
    user = _GATE_USER_TMPL.format(
        question=question,
        existing=_fmt_existing(existing),
        draft=draft or "（空）",
    )
    try:
        raw = llm.generate(_GATE_SYSTEM, user)
    except Exception as e:
        return False, 0.0, {"reason": f"门禁调用失败: {e}"}

    obj = _parse_quality_json(raw)
    if obj is None:
        return False, 0.0, {"reason": "门禁未返回合法 JSON，安全起见判不通过",
                            "raw": (raw or "")[:200]}

    # 取四项分数（缺项按 0 计），按权重求和
    scores = {k: float(obj.get(k, 0)) for k in QUALITY_WEIGHTS}
    score = sum(QUALITY_WEIGHTS[k] * scores[k] for k in QUALITY_WEIGHTS)
    passed = score >= QUALITY_THRESHOLD
    detail = {
        "scores": scores,
        "score": round(score, 3),
        "passed": passed,
        "reason": obj.get("reason", ""),
    }
    return passed, score, detail


# ================================================================
# 影子库入库（阶段 6.5，双存储：PG + 向量，status='staging'）
# ================================================================

def ingest_shadow(pg: PGStore, vec: VecStore, emb,
                  doc_id: str, doc_name: str, content: str) -> int:
    """
    把一份生成好的知识条目写入"影子库"（staging，非主库 active）。

    双写（和阶段 1 的 ingest 完全一致，只是文档状态不同）：
      1) PG：document 表 status='staging'、source='agent_generated'；chunk 表写切片
      2) 向量库：同一批 chunk 的向量 upsert 进去（这样下次检索能召回，但被打标签）

    :return: 实际写入的 chunk 数（0 表示内容为空、没写任何东西）
    """
    chunker = get_chunker("recursive")            # 和阶段 1 默认切片策略一致
    chunks = chunker.split(content, doc_id, doc_name)
    if not chunks:
        return 0

    texts = [c.content for c in chunks]
    vectors = emb.encode_docs(texts)              # 文档侧编码（bge normalize 后内积=余弦）

    # 写 PG：关键就在这两行的 status / source，区别于人传的 active/upload
    pg.upsert_document(doc_id, doc_name, char_count=len(content),
                       status="staging", source="agent_generated")
    pg.upsert_chunks([c.to_dict() for c in chunks])

    # 写向量库
    vec.upsert([c.chunk_id for c in chunks], vectors, [c.doc_id for c in chunks])
    return len(chunks)


# ================================================================
# 总编排：阶段 6 状态机（核心入口）
# ================================================================

def _research_query(question: str, step: int) -> str:
    """多轮调研时逐步换搜索词，增加命中多样性（避免每轮都搜同一句、卡死）。"""
    suffixes = ["", " 官方文档", " 教程", " 最佳实践"]
    suf = suffixes[step] if step < len(suffixes) else f" 第{step}次"
    return question + suf


def _annotate(res: Dict[str, Any], pg: PGStore) -> Dict[str, Any]:
    """
    给召回结果打"未验证来源"标签：如果某块来自影子库（status != 'active'），
    就标 unverified=True，前端/CLI 可据此提示用户"这条答案来自机器自爬、尚未人工审核"。
    这是防投毒的"透明化"一环——能召回，但明确告诉你它没被审过。
    """
    for c in res.get("retrieved", []):
        doc_id = c.get("doc_id")
        if doc_id and pg:
            doc = pg.get_document(doc_id)
            if doc:
                c["source"] = doc.get("source")
                c["doc_status"] = doc.get("status")
                c["unverified"] = (doc.get("status") != "active")
    return res


def run_self_heal(question: str, *, top_k: int = None, retrieval: str = "vector",
                  pg: PGStore = None, vec: VecStore = None, emb=None,
                  backend_llm=None, bm25=None, max_steps: int = 3,
                  site: str = None, save: bool = True,
                  search_fn=None, fetch_fn=None, reranker=None) -> Dict[str, Any]:
    """
    阶段 6 总入口：对一个问题跑完整"自愈闭环"。

    流程：
      1) 先用阶段 5（库内自愈）答一次；
      2) gap_detected 判定是否有真缺口；
      3) 有缺口 → 最多 max_steps 轮联网调研（搜索→抓正文→生成→门禁）；
         门禁通过 → ingest_shadow 写影子库 → 跳出；
      4) 入库后用【全新】向量实例重新问答（确保读到刚写入的 staging 块），
         此时同一问题应该能答上来了；
      5) 全程记录 steps，便于可观测 / 复盘。

    :param question:    用户问题
    :param top_k:       召回块数（None → config.RAG_TOP_K）
    :param retrieval:   vector / hybrid
    :param pg/vec/emb/backend_llm/bm25: 可注入已建好的实例（服务/CLI 常驻复用）；
                         为 None 时本函数惰性创建，并在 own_pg 时负责关闭 pg。
    :param max_steps:   调研重试上限（防死循环，路线图风险清单点名的要求）
    :param site:        限定调研站点域名（可选）
    :param save:        是否把问答写入 qa_log
    :param search_fn:   搜索后端（默认 web_search）。可注入别的实现（如商业搜索 API），
                       接口需为 (query, top_n, site, timeout) -> (List[SearchHit], err)
    :param fetch_fn:    抓取后端（默认 fetch_page）。接口需为 (url, timeout) -> (text, err)
    :return: 结构化 dict（供 ask.py / heal_knowledge.py / serve 渲染）

    关于"再问就能答"的实现细节（坑点）：
      ingest_shadow 把新块写进向量库后，Milvus Lite 同进程的同一个 client 即可检索到
      （growing segment 可见），所以重新问答时直接复用同一个 vec 实例，新写入的 staging
      块能被召回到。若将来换成"独立进程入库 + 独立进程检索"的部署，检索侧需显式 reload
      集合（详见 core/storage/vec_store.py 的 ensure_loaded 注释）。
    """
    top_k = top_k or config.RAG_TOP_K
    search_fn = search_fn or web_search      # 可注入搜索后端（如商业 API）
    fetch_fn = fetch_fn or fetch_page         # 可注入抓取后端
    own_pg = pg is None
    if own_pg:
        pg = PGStore()

    try:
        # 惰性创建其余依赖（CLI / 一次性调用场景）
        if emb is None:
            emb = get_embedder("bge")
        if vec is None:
            vec = VecStore(dim=emb.dim)
        if backend_llm is None:
            backend_llm = llm_mod.get_llm(config.LLM_BACKEND)

        steps: List[str] = []

        # —— 第 1 步：阶段 5 库内自愈首答 ——
        first = generate_answer(
            question, top_k=top_k, retrieval=retrieval, self_heal=True,
            save=save, emb=emb, vec=vec, pg=pg, backend_llm=backend_llm, bm25=bm25,
            reranker=reranker,
        )
        top_score = first["retrieved"][0]["score"] if first["retrieved"] else None
        steps.append(f"库内首答: refusal={first['refusal']}, top_score={top_score}")

        # —— 第 2 步：缺口判定 ——
        gap, reason = gap_detected(first["answer"], first["retrieved"])
        if not gap:
            # 没缺口：不需要联网，直接返回（self_healed_by_agent=False 表示没走调研）
            first["gap_detected"] = False
            first["gap_reason"] = reason
            first["self_healed_by_agent"] = False
            first["steps"] = steps
            return _annotate(first, pg)

        # —— 第 3 步：缺口存在 → 联网调研循环 ——
        steps.append(f"缺口判定: {reason} → 进入联网调研（最多 {max_steps} 轮）")
        searched: List[str] = []
        draft = ""
        passed = False
        score = 0.0
        detail: dict = {}

        # 进循环前先探一次网络（见 probe_net 的说明）。不通就一轮都不跑：
        # 境外站点在受限网络下会卡在 SYN_SENT，硬等 max_steps × timeout 会让
        # 请求拖到超时、还会占着服务线程。这里几秒内失败并给出明确原因，
        # 之后 draft 仍为空串、passed 仍为 False，自然走到下面的"保持拒答"兜底。
        reachable, probe_msg = probe_net()
        steps.append(f"联网探测: {probe_msg}")

        for step in range(max_steps if reachable else 0):
            q = _research_query(question, step)
            hits, err = search_fn(q, top_n=5, site=site)
            if err:
                steps.append(f"第{step + 1}轮: 搜索失败({err})")
                break
            searched.extend([h.url for h in hits])
            steps.append(f"第{step + 1}轮: 搜到 {len(hits)} 条结果")

            # 抓前 3 个页面的正文
            pages = []
            for h in hits[:3]:
                text, e = fetch_fn(h.url)
                if text:
                    pages.append(f"# {h.title}\n{h.url}\n{text}")
            crawled = "\n\n".join(pages)[:MAX_CRAWL_CHARS]

            # 生成草稿 + 质量门禁
            draft = draft_document(question, crawled, backend_llm)
            passed, score, detail = quality_gate(
                draft, question, backend_llm, existing=first["retrieved"])
            steps.append(
                f"  生成草稿({len(draft)}字) → 门禁 {score:.2f} "
                f"{'✅通过' if passed else '❌不通过'}: {detail.get('reason', '')}"
            )
            if passed:
                break   # 门禁通过，停止调研

        # —— 第 4 步：门禁通过 → 写影子库 → 重新问答 ——
        if passed:
            # doc_id 按问题哈希稳定生成：同一个问题多次自愈不会重复建文档（幂等）
            doc_id = "agent_" + hashlib.md5(question.encode("utf-8")).hexdigest()[:16]
            doc_name = f"[agent] {question[:40]}"
            n = ingest_shadow(pg, vec, emb, doc_id, doc_name, draft)
            steps.append(f"写影子库: {n} 个 chunk（status=staging, source=agent_generated）")

            # 重新问答：复用同一个 vec 实例（Milvus Lite 同进程 upsert 后即可检索到新块；
            # 若 vec 是服务单例也一样，因为就是同一个 client）。bm25=None 则按最新 PG 重建，
            # 保证新写入的 staging 块也能被关键词召回。
            final = generate_answer(
                question, top_k=top_k, retrieval=retrieval, self_heal=False,
                save=save, emb=emb, vec=vec, pg=pg, backend_llm=backend_llm, bm25=None,
                reranker=reranker,
            )
            final["gap_detected"] = True
            final["gap_reason"] = reason
            final["searched"] = searched
            final["draft"] = draft
            final["quality_pass"] = passed
            final["quality_score"] = score
            final["quality_detail"] = detail
            final["ingested_chunks"] = n
            final["shadow_doc_id"] = doc_id
            final["self_healed_by_agent"] = True
            final["steps"] = steps
            return _annotate(final, pg)

        # —— 兜底：调研失败，保持原拒答，诚实返回 ——
        first["gap_detected"] = True
        first["gap_reason"] = reason
        first["searched"] = searched
        first["draft"] = draft
        first["quality_pass"] = passed
        first["quality_score"] = score
        first["quality_detail"] = detail
        first["ingested_chunks"] = 0
        first["self_healed_by_agent"] = False
        first["steps"] = steps
        return _annotate(first, pg)
    finally:
        # 只有本函数自己创建的 pg 才负责关闭；注入的由调用方管理
        if own_pg:
            pg.close()
