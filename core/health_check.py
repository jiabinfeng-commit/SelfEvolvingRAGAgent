# -*- coding: utf-8 -*-
"""
core/health_check.py —— 阶段 7：知识库体检 Agent（自动找出"该修的内容"）

====================================================================
为什么要有"体检"这一步？
====================================================================
阶段 1~6 我们让系统能"答、能自愈、能补库"。但补得多了、跑得久了，知识库自己也会
积累垃圾：
  - 过时内容：写于 2023 年的"最新版配置"，现在早就变了
  - 矛盾内容：同一篇文档前后两句话打架，或不同文档说法冲突
  - 僵尸 chunk：入库了却从没被任何一次问答召回过（死内容 / 检索永远够不到）
  - 重复内容：同一段话被切出好几块，或不同文档抄了同一段

这些不会让系统"报错"，但会悄悄拉低答案质量。所以阶段 7 做一个"体检 Agent"：
跑一遍四个检测器，把问题列成一份人能直接照着处理的 Markdown 健康报告。

====================================================================
四个检测器（各管一类）
====================================================================
  ① detect_stale    过时内容   —— 正则扫"X年最新/当前版本" + 比对文档入库时间
  ② detect_conflicts 矛盾内容  —— 同文档内 chunk 两两交给 LLM 判"是否冲突"
  ③ detect_zombies  僵尸 chunk  —— 从 qa_log 反推"哪些块从没被召回过"
  ④ detect_duplicates 重复内容  —— 重编码全库 chunk，余弦相似度 > 0.95 的聚成簇

设计取舍（和项目一贯纪律一致）：
  - 全是"可注入 / 可跳过"的：矛盾检测要烧 LLM，llm=None 时自动跳过并注明；
    其余三个是纯规则 / 纯向量，不需要 LLM 也能跑。
  - 报告只读不写：体检只产出"建议"（合并/更新/删除），绝不自动改库——
    真要动数据（尤其删 chunk）必须人来拍板，避免 Agent 自己删库这种灾难。
  - 阶段 6 留下的"影子库(staging)"也在这里给出"待人工审核转正"清单，
    把"写 staging"和"转 active"两个动作彻底分开（防投毒的最后一公里）。

====================================================================
对外暴露
====================================================================
- HealthIssue 数据类（一条问题记录）
- detect_stale / detect_conflicts / detect_zombies / detect_duplicates
- run_health_check(pg, vec, emb, llm=None, ...) ：总入口，返回 {issues, summary, markdown}
"""
import re
import json
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple

import numpy as np

from core import config
from core.storage.pg_store import PGStore
from core.storage.vec_store import VecStore
from core.embedder import get_embedder


# ================================================================
# 可调参数（集中放顶部）
# ================================================================

# ① 过时检测：内容里提到的"年份"比（当前年 - 这个值）还早 → 疑为过时
# 例：现在是 2026，gap=2 → 提到 2023 及更早的"最新版"就报警
STALE_YEAR_GAP = 2

# ① 过时检测（按时间）：文档入库超过这么多天且内容里又出现时效词，判过时
STALE_DAYS = 365

# ④ 重复检测阈值：两 chunk 余弦相似度 >= 此值视为"重复候选"（bge 归一化后=内积）
DUP_THRESHOLD = 0.95

# ② 矛盾检测上限：两两比对可能爆炸（N 块 → N*(N-1)/2 对），这里封顶避免烧爆 LLM
MAX_CONFLICT_PAIRS = 200

# ③ 僵尸兜底：块入库不足这么多天时，不轻易判僵尸（可能只是还没机会被问到）
ZOMBIE_MIN_AGE_DAYS = 1


# ================================================================
# 问题记录结构
# ================================================================

@dataclass
class HealthIssue:
    """体检发现的一条问题。kind 决定它属于哪个检测器。"""
    issue_id: str
    kind: str                 # stale / conflict / zombie / duplicate
    severity: str             # high / medium / low
    chunk_ids: List[str]      # 涉及的 chunk（僵尸/重复可能是多个）
    doc_ids: List[str]        # 涉及的文档
    doc_names: List[str]      # 涉及的文档名（给人看）
    detail: str               # 现象描述
    suggestion: str           # 处置建议


# ================================================================
# ① 过时内容检测
# ================================================================

# 时效敏感词：命中这些 + 同时满足"提到旧年份 / 文档很老"才报警，避免误伤普通叙述
_STALE_PHRASES = [
    r"最新", r"当前版本", r"目前版本", r"截至\d{4}", r"现阶段", r"现已",
    r"今年", r"去年", r"新版本", r"最新版", r"最新发布",
]
_STALE_PHRASE_RE = re.compile("|".join(_STALE_PHRASES))
# 抓"4 位年份"：2020~当前年。注意只取年份数字，后面和当前年比。
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _mentioned_old_year(text: str, current_year: int) -> Optional[int]:
    """内容里提到的最早年份，若比（当前年 - gap）还早则返回该年份，否则 None。"""
    years = [int(m.group()) for m in _YEAR_RE.finditer(text or "")
             if 1990 <= int(m.group()) <= current_year]
    if not years:
        return None
    earliest = min(years)
    return earliest if earliest <= current_year - STALE_YEAR_GAP else None


def detect_stale(chunks_with_doc: List[Dict],
                 use_llm: bool = False, llm=None,
                 current_year: int = None) -> List[HealthIssue]:
    """
    过时内容检测。

    两条触发规则（任一满足即报警）：
      A) 内容里有"最新/当前版本"等时效词，且提到了一个偏老的年份（<= 当前年 - STALE_YEAR_GAP）
      B) 文档入库时间距今超过 STALE_DAYS（默认一年），且内容里出现时效词

    为什么用"时效词 + 老年份"双条件？因为单独抓"2023"会误伤（历史背景叙述也写年份），
    必须配合"最新/当前版本"这种明显自称时效性的措辞，才大概率是真的该更新了。

    :param chunks_with_doc: pg.get_all_chunks_with_doc() 的结果（含 doc_id/doc_name/created_at）
    :param use_llm: 是否再让 LLM 精筛（默认关，省钱；开的话对疑似过时的再让模型判一次）
    :param llm:     use_llm=True 时需要的 LLM 实例
    :param current_year: 当前年（默认取系统时间，便于测试注入）
    """
    cy = current_year or datetime.date.today().year
    issues: List[HealthIssue] = []
    # 按文档聚合：同一文档下只要有一个块触发，整篇标过时（避免同文档碎成多条）
    seen_doc = set()

    for c in chunks_with_doc:
        doc_id = c.get("doc_id")
        if doc_id in seen_doc:
            continue
        doc_name = c.get("doc_name", "?")
        content = c.get("content", "") or ""
        created = c.get("created_at")

        has_phrase = bool(_STALE_PHRASE_RE.search(content))
        if not has_phrase:
            continue  # 连时效词都没有，怎么都不算过时

        old_year = _mentioned_old_year(content, cy)
        old_by_time = False
        if created:
            try:
                age_days = (datetime.datetime.now() - created).days
                old_by_time = age_days >= STALE_DAYS
            except Exception:
                old_by_time = False

        # 规则 A：时效词 + 老年份
        if old_year is not None:
            seen_doc.add(doc_id)
            issues.append(HealthIssue(
                issue_id=f"stale_{doc_id}",
                kind="stale",
                severity="medium",
                chunk_ids=[c.get("chunk_id")],
                doc_ids=[doc_id],
                doc_names=[doc_name],
                detail=f"文档提到「{old_year}年」相关内容且含时效性表述（最新/当前版本等），"
                       f"可能已过时。",
                suggestion="核对当前实际版本/数据，更新该文档后重新入库。",
            ))
            continue
        # 规则 B：时效词 + 文档很老
        if old_by_time:
            seen_doc.add(doc_id)
            issues.append(HealthIssue(
                issue_id=f"stale_{doc_id}",
                kind="stale",
                severity="low",
                chunk_ids=[c.get("chunk_id")],
                doc_ids=[doc_id],
                doc_names=[doc_name],
                detail=f"文档入库已超 {STALE_DAYS} 天且含时效性表述，建议确认内容是否仍有效。",
                suggestion="人工确认是否仍准确；如仍有效可忽略，过期则更新重入库。",
            ))
    return issues


# ================================================================
# ② 矛盾内容检测（需 LLM）
# ================================================================

_CONFLICT_SYSTEM = (
    "你是知识库一致性审查员。下面给你两段来自知识库的【陈述】，请判断它们是否相互矛盾"
    "（即一段说 A、另一段说非 A，读者同时看到会困惑）。\n"
    "只输出一个 JSON，不要多余文字，格式：\n"
    '{"conflict": true/false, "reason": "一句话说明"}'
)

_CONFLICT_USER_TMPL = (
    "【陈述 1】\n{a}\n\n【陈述 2】\n{b}"
)


def _extract_json(text: str) -> Optional[dict]:
    """从 LLM 输出抠 JSON（兼容 ```json 包裹 / 前后废话）。与 agent.py 同源思路。"""
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


def _conflict_judge(a: str, b: str, llm) -> Tuple[bool, str]:
    """让 LLM 判两段是否矛盾。任何异常 → 不判矛盾（保守，宁漏勿错）。"""
    user = _CONFLICT_USER_TMPL.format(a=(a or "")[:600], b=(b or "")[:600])
    try:
        raw = llm.generate(_CONFLICT_SYSTEM, user)
    except Exception as e:
        return False, f"冲突判定调用失败: {e}"
    obj = _extract_json(raw)
    if not obj:
        return False, "未返回合法 JSON，保守判不冲突"
    return bool(obj.get("conflict")), obj.get("reason", "")


def detect_conflicts(chunks_with_doc: List[Dict], llm=None,
                     max_pairs: int = MAX_CONFLICT_PAIRS) -> List[HealthIssue]:
    """
    矛盾内容检测（需 LLM）。

    做法：先把 chunk 按 doc_name 分组（同一文档内部最容易出现前后矛盾；
    跨文档矛盾理论上也要查，但成本太高，先做文档内，够覆盖绝大多数场景）。
    组内两两交给 LLM 判冲突，命中即记一条 high 级问题。

    :param llm: 必须传 LLM 实例；为 None 时直接返回空并提示"跳过"。
    :param max_pairs: 最多比对多少对（防 N^2 爆炸 + 控成本）。
    """
    if llm is None:
        # 不烧 LLM 就跳过，但调用方应在报告里注明"矛盾检测未执行"
        return []

    # 按 doc_name 分组
    groups: Dict[str, List[Dict]] = {}
    for c in chunks_with_doc:
        groups.setdefault(c.get("doc_name", "?"), []).append(c)

    issues: List[HealthIssue] = []
    checked = 0
    for doc_name, group in groups.items():
        if len(group) < 2:
            continue
        # 组内两两比对
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if checked >= max_pairs:
                    return issues
                checked += 1
                a = group[i].get("content", "") or ""
                b = group[j].get("content", "") or ""
                conflict, reason = _conflict_judge(a, b, llm)
                if conflict:
                    issues.append(HealthIssue(
                        issue_id=f"conflict_{group[i].get('chunk_id')}_{group[j].get('chunk_id')}",
                        kind="conflict",
                        severity="high",
                        chunk_ids=[group[i].get("chunk_id"), group[j].get("chunk_id")],
                        doc_ids=[group[i].get("doc_id"), group[j].get("doc_id")],
                        doc_names=[doc_name, doc_name],
                        detail=f"同文档内两段内容疑似矛盾：{reason}",
                        suggestion="人工核对两段，保留正确说法，删除/修正矛盾段后重新入库。",
                    ))
    return issues


# ================================================================
# ③ 僵尸 chunk 检测（从 qa_log 反推，无需新表）
# ================================================================

def detect_zombies(pg: PGStore,
                    chunks_with_doc: List[Dict] = None) -> List[HealthIssue]:
    """
    僵尸 chunk 检测：找出"入库了却从没被任何一次问答召回过"的块。

    实现：召回过的 chunk_id 集合来自 qa_log.ctx_chunk_ids（见 pg.get_recalled_chunk_ids），
    全库 chunk 减去它，剩下的就是僵尸。空集合里若某块入库不足 ZOMBIE_MIN_AGE_DAYS 天，
    不轻易判僵尸（可能只是还没被问到）。

    :param pg: PGStore 实例（必须，要查 qa_log 与 chunk）
    :param chunks_with_doc: 可选，已取好的全库 chunk（避免重复查；不传则现场查）
    """
    recalled = pg.get_recalled_chunk_ids()
    all_chunks = chunks_with_doc if chunks_with_doc is not None else pg.get_all_chunks_with_doc()
    if not all_chunks:
        return []

    zombie_ids: List[str] = []
    zombie_names: List[str] = []
    for c in all_chunks:
        cid = c.get("chunk_id")
        if cid in recalled:
            continue
        # 入库太新的不轻易判僵尸
        created = c.get("created_at")
        if created:
            try:
                if (datetime.datetime.now() - created).days < ZOMBIE_MIN_AGE_DAYS:
                    continue
            except Exception:
                pass
        zombie_ids.append(cid)
        zombie_names.append(c.get("doc_name", "?"))

    if not zombie_ids:
        return []
    return [HealthIssue(
        issue_id="zombie_batch",
        kind="zombie",
        severity="low",
        chunk_ids=zombie_ids,
        doc_ids=list({c.get("doc_id") for c in all_chunks if c.get("chunk_id") in set(zombie_ids)}),
        doc_names=list(dict.fromkeys(zombie_names)),
        detail=f"共 {len(zombie_ids)} 个 chunk 从未被任何一次问答召回（疑似死内容或检索够不到）。",
        suggestion="确认无用后删除对应 chunk（及文档，若整篇都未被召回）；"
                   "若内容重要却召不回，检查切片/embedding/检索阈值。",
    )]


# ================================================================
# ④ 重复内容检测（重编码 + 余弦聚类）
# ================================================================

def _cosine_clusters(vectors: np.ndarray, threshold: float) -> List[List[int]]:
    """
    对归一化向量做相似度聚类：两两余弦 >= threshold 的并到同一簇（并查集）。

    :param vectors: (N, dim) 已 L2 归一化，点积即余弦
    :param threshold: 相似度阈值
    :return: 簇列表，每个簇是向量下标列表（长度 >1 才是重复簇）
    """
    n = vectors.shape[0]
    if n < 2:
        return []
    sim = vectors @ vectors.T            # (N,N) 余弦相似度矩阵
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 只扫上三角，避免重复比对
    for i in range(n):
        for j in range(i + 1, n):
            if float(sim[i, j]) >= threshold:
                union(i, j)
    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return [v for v in clusters.values() if len(v) > 1]


def detect_duplicates(chunks: List[Dict], emb,
                      threshold: float = DUP_THRESHOLD) -> List[HealthIssue]:
    """
    重复内容检测：把全库 chunk 重新编码，余弦相似度 >= 0.95 的聚成一簇，每簇报一条。

    为什么阈值定 0.95 这么高？bge 归一化后，同义改写通常 0.7~0.85，真正"几乎一字不差"
    才会到 0.95 以上。定太高可能漏（非精确重复不报），定太低会误报（同主题但不同内容）。
    0.95 是"几乎重复才报"的保守线，符合"宁可漏报也不乱删"的原则。

    :param chunks: 全库 chunk（含 chunk_id / doc_id / doc_name / content）
    :param emb:    Embedder 实例（提供 encode_docs）
    """
    if not chunks:
        return []
    texts = [(c.get("content") or "") for c in chunks]
    vectors = emb.encode_docs(texts)          # (N, dim) 已归一化
    # 空内容向量全 0，余弦算出来是 0/nan，不会误聚；但为稳妥过滤全 0
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-6
    if valid.sum() < 2:
        return []
    sub = vectors[valid]
    idx_map = [i for i, v in enumerate(valid) if v]   # 有效向量的原下标

    clusters = _cosine_clusters(sub, threshold)
    issues: List[HealthIssue] = []
    for k, cluster in enumerate(clusters):
        orig_idx = [idx_map[i] for i in cluster]
        cids = [chunks[i].get("chunk_id") for i in orig_idx]
        dids = [chunks[i].get("doc_id") for i in orig_idx]
        dnames = [chunks[i].get("doc_name", "?") for i in orig_idx]
        issues.append(HealthIssue(
            issue_id=f"dup_{k}",
            kind="duplicate",
            severity="medium",
            chunk_ids=cids,
            doc_ids=list(dict.fromkeys(dids)),
            doc_names=list(dict.fromkeys(dnames)),
            detail=f"发现 {len(cids)} 个高度相似的 chunk（余弦 >= {threshold}），疑似重复。",
            suggestion="合并为一条或删除冗余 chunk，减少检索噪声与 token 浪费。",
        ))
    return issues


# ================================================================
# 汇总 + 报告
# ================================================================

def _list_staging_docs(pg: PGStore) -> List[Dict]:
    """列出所有 staging 文档（阶段 6 写入、待人工审核转 active 的影子库内容）。"""
    # 这里不引 psycopg2（避免模块级依赖），用普通 cursor + 按列序手动转 dict。
    with pg.conn.cursor() as cur:
        cur.execute(
            "SELECT doc_id, doc_name, created_at FROM document WHERE status='staging' "
            "ORDER BY created_at DESC")
        rows = cur.fetchall()
    return [
        {"doc_id": r[0], "doc_name": r[1], "created_at": r[2]}
        for r in rows
    ]


def run_health_check(pg: PGStore = None, vec: VecStore = None, emb=None,
                     llm=None, *, top_k: int = None,
                     dup_threshold: float = DUP_THRESHOLD,
                     stale_days: int = STALE_DAYS,
                     max_conflict_pairs: int = MAX_CONFLICT_PAIRS,
                     use_llm_stale: bool = False,
                     report_path: str = None) -> Dict[str, Any]:
    """
    阶段 7 总入口：跑四个检测器，产出健康报告（dict，含 markdown）。

    :param pg/vec/emb/llm: 可注入已建好的实例（服务/CLI 常驻复用）；为 None 时惰性创建，
                           own_pg 时本函数负责关闭 pg。vec/emb 仅重复检测需要，
                           llm 仅矛盾检测需要（不传则跳过矛盾检测）。
    :param report_path:   若指定，把 markdown 报告写到这里（默认不写，只返回 dict）
    :return: {
        "issues":   [HealthIssue...],
        "summary":  {按 kind/severity 计数, 文档数, 块数},
        "markdown": str,           # 可直接落盘的 Markdown 健康报告
        "skipped":  [str...],      # 因缺依赖被跳过的检测器说明
    }
    """
    own_pg = pg is None
    if own_pg:
        pg = PGStore()
    skipped: List[str] = []
    try:
        if emb is None:
            emb = get_embedder("bge")
        # 全库 chunk（带文档信息），四个检测器共用这一份
        chunks = pg.get_all_chunks_with_doc()

        # —— ① 过时 ——
        stale = detect_stale(chunks, use_llm=use_llm_stale, llm=llm)
        # —— ② 矛盾（需 LLM）——
        if llm is not None:
            conflicts = detect_conflicts(chunks, llm=llm, max_pairs=max_conflict_pairs)
        else:
            conflicts = []
            skipped.append("矛盾检测：未提供 LLM 实例，已跳过（如需请传入 llm=...）")
        # —— ③ 僵尸 ——
        zombies = detect_zombies(pg, chunks_with_doc=chunks)
        # —— ④ 重复 ——
        duplicates = detect_duplicates(chunks, emb, threshold=dup_threshold)

        issues = stale + conflicts + zombies + duplicates

        # 概览统计
        by_kind = {}
        by_sev = {"high": 0, "medium": 0, "low": 0}
        for it in issues:
            by_kind[it.kind] = by_kind.get(it.kind, 0) + 1
            by_sev[it.severity] = by_sev.get(it.severity, 0) + 1

        summary = {
            "documents": pg.count().get("documents", 0),
            "chunks": len(chunks),
            "total_issues": len(issues),
            "by_kind": by_kind,
            "by_severity": by_sev,
        }

        # 待审影子库（阶段 6 写入、待转 active）
        staging = _list_staging_docs(pg)

        md = _to_markdown(summary, issues, staging, skipped)
        if report_path:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(md)

        return {
            "issues": issues,
            "summary": summary,
            "staging_docs": staging,
            "markdown": md,
            "skipped": skipped,
        }
    finally:
        if own_pg:
            pg.close()


def _to_markdown(summary: Dict, issues: List[HealthIssue],
                 staging: List[Dict], skipped: List[str]) -> str:
    """把体检结果渲染成一份人能照着处理的 Markdown 健康报告。"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("# 知识库体检报告")
    lines.append(f"\n> 生成时间：{now}\n")
    lines.append("## 概览")
    lines.append(f"- 文档数：**{summary['documents']}**　块数：**{summary['chunks']}**")
    lines.append(f"- 发现问题总数：**{summary['total_issues']}**")
    bk = summary["by_kind"]
    lines.append(
        f"- 分类计数：过时 {bk.get('stale',0)} / 矛盾 {bk.get('conflict',0)} / "
        f"僵尸 {bk.get('zombie',0)} / 重复 {bk.get('duplicate',0)}")
    bs = summary["by_severity"]
    lines.append(f"- 严重度：🔴 high {bs['high']} / 🟡 medium {bs['medium']} / 🟢 low {bs['low']}")

    # 跳过说明
    if skipped:
        lines.append("\n## 本次跳过的检测")
        for s in skipped:
            lines.append(f"- ⚠️ {s}")

    # 各类明细
    sections = [
        ("一、过时内容（stale）", "stale"),
        ("二、矛盾内容（conflict）", "conflict"),
        ("三、僵尸 chunk（zombie）", "zombie"),
        ("四、重复内容（duplicate）", "duplicate"),
    ]
    for title, kind in sections:
        lines.append(f"\n## {title}")
        sub = [it for it in issues if it.kind == kind]
        if not sub:
            lines.append("- ✅ 未发现此类问题")
            continue
        for it in sub:
            sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(it.severity, "⚪")
            lines.append(f"\n### {sev_icon} [{it.severity}] {it.issue_id}")
            lines.append(f"- 文档：{', '.join(it.doc_names) or '(未知)'}")
            lines.append(f"- 现象：{it.detail}")
            lines.append(f"- 建议：{it.suggestion}")
            if it.chunk_ids:
                shown = it.chunk_ids[:10]
                more = "" if len(it.chunk_ids) <= 10 else f" …等共 {len(it.chunk_ids)} 个"
                lines.append(f"- 涉及 chunk：`{', '.join(shown)}{more}`")

    # 待审影子库（阶段 6 闭环的"转 active"动作放在这里提示）
    lines.append("\n## 五、待人工审核的影子库（staging → active）")
    if not staging:
        lines.append("- ✅ 无待审影子内容")
    else:
        lines.append(f"- 共 {len(staging)} 篇由 Agent 自动生成、尚在影子库待审核：")
        for d in staging:
            lines.append(f"  - `{d['doc_id']}` {d['doc_name']}（写入于 {d['created_at']}）")
        lines.append("- 处置：人工确认内容正确后，将其 document.status 由 staging 改为 active；"
                     "错误则删除。未经审核不要直接转 active（防投毒最后一公里）。")

    # 处置建议汇总
    lines.append("\n## 六、处置建议汇总")
    lines.append("1. **矛盾(high)** 优先处理：人工核对后保留正确说法，删除/修正矛盾段。")
    lines.append("2. **重复(medium)** 合并或删冗余，减少检索噪声。")
    lines.append("3. **过时(medium/low)** 核对当前实际信息后更新重入库。")
    lines.append("4. **僵尸(low)** 确认无用后清理；重要却召不回则查检索链路。")
    lines.append("5. **影子库** 逐篇人工审核转正，绝不自动转 active。")
    lines.append("\n> 本报告只读不写：所列建议均需人工确认后再操作，体检 Agent 不会自动修改知识库。")

    return "\n".join(lines)
