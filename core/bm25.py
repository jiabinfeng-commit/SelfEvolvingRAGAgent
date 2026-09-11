# -*- coding: utf-8 -*-
"""
BM25 关键词检索（阶段 4：混合检索的「关键词」那一半）
===============================================================================
【零基础导读】建议按这个顺序读：
    1. 先读本段 docstring（搞懂"为什么需要它"和公式每项的含义）
    2. 再读 tokenize()        —— 怎么把一句话拆成"词"
    3. 再读 BM25.__init__     —— 需要缓存哪些统计量
    4. 再读 BM25.build()      —— 怎么"建索引"（一次性预处理）
    5. 最后读 BM25.search()   —— 怎么"打分排序"（每次查询时做）

===============================================================================
一、为什么需要 BM25？
===============================================================================
- 阶段 1~3 只有**向量语义检索**：意思相近就能召回，但对「专业术语 / 精确字符串」很迟钝。
  例如问 "Depends 怎么用"、"HTTPS 配置"，向量可能召回一堆语义相关但不含该词的块。
- BM25 是经典的**关键词精确匹配**算法（Elasticsearch / Lucene 的默认打分），
  正好补上语义检索的短板。两者融合 = 混合检索（hybrid retrieval）。

一句话类比：
    向量检索 = "这个人说话的意思像不像"（懂语义，但可能忽略原词）
    BM25     = "这个人有没有原封不动地说出这几个词"（死抠字面，但精准）

===============================================================================
二、BM25 公式逐项拆解
===============================================================================
对「查询里的某个词 t」和「某篇文档 d」：

    score(t, d) = IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

一篇文档 d 的总分 = 查询里**每个词**分别算一遍，然后相加。

┌────────────┬──────────────────────────────────────────────────────────────┐
│ 符号       │ 含义（通俗版）                                               │
├────────────┼──────────────────────────────────────────────────────────────┤
│ t          │ 查询里的一个"词"（本文件里也叫 token）                       │
│ d          │ 一篇文档（本项目里 = 一个 chunk）                            │
│ N          │ 全库文档总数（本项目 = 494 块）                              │
│ df(t)      │ "文档频率"：全库有多少篇文档**包含** t（文档中出现多次只算 1）│
│ IDF(t)     │ 逆文档频率 = 词 t 有多"稀有"。**越稀有，区分度越高，权重越大**│
│ tf         │ "词频"：t 在文档 d 里出现了几次                               │
│ dl         │ 文档 d 的长度（token 个数）                                  │
│ avgdl      │ 全库平均文档长度                                             │
│ k1 = 1.5   │ 控制 tf 的"饱和速度"（Lucene/ES 默认值，一般不用调）          │
│ b  = 0.75  │ 控制"长度归一化"的强度（Lucene/ES 默认值，一般不用调）        │
└────────────┴──────────────────────────────────────────────────────────────┘

IDF 公式：
    IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    直觉：df 越小（词越罕见）→ 分子越大 → IDF 越大 → 这个词越"金贵"。
    两个修正项的作用：
      · "+0.5" 在分子分母上 → 平滑（smoothing），避免 df=0 时除零/极端值
      · 外层的 "+1"        → 保证 IDF 永远 > 0
        （教科书版 BM25 的 IDF 在词太常见时会是负数，Lucene 加了这层 +1 来避免）

tf 部分 (tf * (k1 + 1)) / (tf + k1 * (...)) ：
    直觉：一个词出现越多，分越高，但**收益递减（饱和）**。
      · 出现 1 次 → 0 次是质变，加分明显
      · 出现 10 次 → 20 次几乎不加分（防止"堆关键词"刷分）
      · k1 越大，饱和越慢（越"贪心"地奖励高频词）

长度归一化 (1 - b + b * dl / avgdl) ：
    直觉：长文档天然词多、tf 更容易高，必须惩罚，否则长文档永远占便宜。
      · dl = avgdl（文档长度刚好是平均）→ 括号里 = 1，不奖不罚
      · dl > avgdl（偏长）            → 括号里 > 1 → 分母变大 → 分数被压低
      · b = 0 完全不归一化；b = 1 完全归一化；b = 0.75 表示 75% 力度

===============================================================================
三、一个具体例子（让上面的公式落地）
===============================================================================
假设 N = 494，avgdl = 100（每块平均 100 个 token）

  词 A = "depends"，只出现在 5 个块里 → df = 5
    IDF = ln(1 + (494 - 5 + 0.5) / (5 + 0.5))
        = ln(1 + 489.5 / 5.5) = ln(90) ≈ 4.50

  词 B = "使用"，出现在 400 个块里 → df = 400
    IDF = ln(1 + (494 - 400 + 0.5) / (400 + 0.5))
        = ln(1 + 94.5 / 400.5) ≈ ln(1.236) ≈ 0.21

  → 命中 "depends" 的权重是命中 "使用" 的约 21 倍。
    这就是 BM25 的聪明之处：不只看"有没有出现"，还看"这个词有多大区分度"。

===============================================================================
四、为什么自己写而不装 rank_bm25 / jieba？
===============================================================================
- 项目一贯保持依赖精简（阶段 1 就为省事用过标准库 urllib 替代 requests）。
- BM25 公式本身很短，纯 Python 三四十行就够；494 块的规模性能完全够用。
- 中文分词不用 jieba：**按字 bigram 切分**（"依赖注入" → "依赖"/"赖注"/"注入"），
  这是中文信息检索里最经典、无依赖且效果不错的做法。
  （详见下面 tokenize() 的注释）
"""
import re                            # 正则：切词用
import math                          # log()：算 IDF 用（注意 math.log 是**自然对数** ln）
from collections import Counter, defaultdict
# Counter      ：计数器，本质是 dict 子类。Counter(["a","a","b"]) → {"a": 2, "b": 1}
#                这里用来统计"每个词出现了几次"（tf）以及"有多少文档含某个词"（df）
# defaultdict  ：带默认值的 dict。defaultdict(list) 取值时若 key 不存在会自动建一个空 list，
#                省掉 "if key not in d: d[key] = []" 这种样板代码
from typing import List, Tuple       # 仅用于类型标注，方便 IDE 提示，运行时无影响


# ==============================================================================
# 第一部分：分词规则（3 个正则常量）
# ==============================================================================

# CJK 统一表意文字范围（中日韩汉字）。注意这里是**字符串**不是正则对象，
# 下面用它拼进 f-string 动态构造正则。
# \u4e00-\u9fff 是 Unicode 里汉字的码点区间（"一" 到 "鿿"）
_CJK = r"\u4e00-\u9fff"

# ASCII 词的正则（已编译成 pattern 对象，后面用 .findall() 直接匹配）
#   [a-z0-9]           ：必须由**字母或数字开头**（所以 ".gitignore" 开头的点是匹配不到的）
#   [a-z0-9_\-\.]*     ：后面可以跟任意个 字母/数字/下划线/连字符/点
#   注意：这里匹配的是**已经 lower() 过**的文本，所以只写小写 a-z 就够
# 效果：
#   "pydantic.BaseSettings" → 整体切成一个词 "pydantic.basesettings"
#                             （技术文档里这种点号连接的标识符要保留，不能从点处切开）
#   "bge-small-zh-v1.5"     → 整体保留 "bge-small-zh-v1.5"
#   "0.1"                   → 整体保留 "0.1"
_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9_\-\.]*")

# 中文片段的正则：连续的一段汉字
#   因为要先拿到"连续的汉字串"，再对每段做 bigram 滑动切分（见 tokenize）
# 效果："在 FastAPI 中添加" → findall 得到 ["在", "中添加"]（被英文/空格隔断）
_CJK_SEG = re.compile(f"[{_CJK}]+")


# ==============================================================================
# 第二部分：分词函数
# ==============================================================================
def tokenize(text: str) -> List[str]:
    """
    把文本切成词表（token 列表）。这是整个 BM25 的**第一步**，也是最影响效果的一步。

    参数：
        text: 原始文本（一个 chunk 的正文，或用户的查询串）

    返回：
        词表，例如 "FastAPI 依赖注入" → ["fastapi", "依赖", "赖注", "注入"]

    策略（3 步，无任何外部依赖）：
      1. 全部转小写（英文大小写不敏感：FastAPI 和 fastapi 视为同一个词）
      2. ASCII 部分：按正则切成「词」，如 "fastapi" / "pydantic.settings" / "0.1"
      3. CJK 部分：按字 bigram 滑动切分，如 "依赖注入" → ["依赖","赖注","注入"]
         单字片段直接作为一词（否则 "的" 这种单字会被丢掉，但影响很小）

    【为什么中文要用 bigram（二元组）？】
    - 中文没有空格，不像英文天然分好词。
    - 专业分词（jieba）需要装包 + 依赖词典，且对本项目 494 块的规模是过度设计。
    - bigram 是最经典的无词典方案：把 "依赖注入" 滑窗切成 依赖/赖注/注入。
      优点：不需要词典、对未登录词（专有名词）友好、实现极简。
      代价：会产生 "赖注" 这种无意义组合，让索引变大一点（可接受）。
    """
    # 空文本直接返回空列表。必要性：下面 findall 在 None 上会抛 TypeError，
    # 而调用方有 c.get("content") or "" 兜底，这里再兜一层更稳。
    if not text:
        return []

    # 第 1 步：统一转小写 → 让 "FastAPI" 和 "fastapi" 匹配到同一个 token
    text = text.lower()

    # 累加结果的列表。注意用它而不是 yield，是因为后面要 .extend() 批量追加
    tokens = []

    # 第 2 步：切出所有 ASCII / 数字词
    # findall 会把正则里**所有不重叠的匹配**找出来，返回字符串列表。
    # 多个捕获组时返回元组，这里没有捕获组，所以直接是字符串列表。
    for w in _ASCII_WORD.findall(text):
        tokens.append(w)

    # 第 3 步：处理中文——先按"连续汉字段"取出来，再对每段做滑动窗口
    for seg in _CJK_SEG.findall(text):
        # 特殊情况：整段只有 1 个字（如 "的"、"和"）。
        # 没有相邻字可以组合，直接把这个单字当成一个 token 收进去。
        if len(seg) == 1:
            tokens.append(seg)
        else:
            # 滑动窗口取 2 个字（bigram）：
            #   "依赖注入"（长度 4）→ i 取 0,1,2 → "依赖", "赖注", "注入"
            #   range(len(seg) - 1) 保证不会越界：最后一个 i = len-2，seg[i:i+2] 恰好取到末尾两字
            # extend 而不是 append：把生成器里的每个元素**逐个**加进 tokens，
            #   如果用 append，会把整个生成器对象当成一个元素塞进去（常见新手错误）
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))

    return tokens


# ==============================================================================
# 第三部分：BM25 主体
# ==============================================================================
class BM25:
    """
    极简 BM25 实现：建索引 → 查询 → 返回 top-n。

    典型用法（也是 scripts/api.py 里的实际用法）：

        bm = BM25()                              # 1. 造一个空实例
        bm.build([(chunk_id, "文本"), ...])       # 2. 灌入全库语料建索引（只需建一次）
        bm.search("查询文本", top_n=20)            # 3. 查询 → [(chunk_id, score), ...] 降序

    设计要点：把"建索引"（贵、只做一次）和"查询"（频繁）分离。
    build() 会预先算好 IDF，search() 时就不用重复计算了。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化。这里只**声明并清空**所有状态，不做任何计算。

        参数（两个都是 BM25 公式里的可调超参，直接用 Lucene/ES 的默认值即可）：
            k1: 控制词频 tf 的饱和速度。越大，高频词越"加分"。
                1.2~2.0 是常见范围，1.5 是 Lucene 默认。
            b : 控制文档长度归一化的强度。0 = 不归一化，1 = 完全归一化。
                0.75 是 Lucene 默认。
        """
        # ---- 两个超参，原样存下来，build/search 时要用 ----
        self.k1 = k1
        self.b = b

        # ---- 以下 7 个都是"建索引后的产物"，用平行数组（下标对齐）组织 ----
        # 平行数组的意思是：self.doc_ids[i]、self.tfs[i]、self.doc_len[i] 描述的是同一篇文档。
        # 这样查询时按下标 i 就能一次拿到某篇文档的全部信息，比套 dict 更快。

        # 第 i 篇文档的 id。本项目里传进来的是 chunk_id（如 "76b71ee98e6bcee4"）
        self.doc_ids = []

        # 第 i 篇文档的"词频表"：Counter，形如 {"依赖": 3, "注入": 2, "fastapi": 1}
        # 注意是**每个文档一份**，所以是 list[Counter]
        self.tfs = []

        # 第 i 篇文档的长度（token 个数，不是字符数！）。
        # 用途：公式里的 dl，做长度归一化
        self.doc_len = []

        # 文档频率 df：词 -> **出现在多少篇文档里**（同一文档里出现 10 次也只算 1 篇）
        # 注意与 tfs 的区别：tfs 是"某文档内出现几次"，df 是"多少文档包含它"。
        # 这是全库级别的统计量，所以只有一份
        self.df = Counter()

        # 【核心数据结构】倒排索引：词 -> [包含该词的文档下标, ...]
        # 例如 {"fastapi": [0, 3, 7], "依赖": [1, 2, 9, ...]}
        # 为什么叫"倒排"？正常思路是"文档 → 它有哪些词"（正向）；
        # 这里反过来存"词 → 哪些文档有它"，查询时就能**只遍历命中的文档**，
        # 而不是把全库 494 块挨个算 —— 语料变大时才不会退化成 O(N × Q)。
        # 用 defaultdict(list)：第一次遇到某个词时自动创建空 list，不用手写判断
        self.inverted = defaultdict(list)

        # 词 -> IDF 值。build() 里一次性算好，search() 直接查表，避免重复 log 计算
        self.idf = {}

        # N = 全库文档总数（就是 len(self.doc_ids)，单独存一份方便公式里用）
        self.N = 0

        # avgdl = 全库平均文档长度（average document length）= total_len / N
        # 公式里做长度归一化的基准。空语料时留 0.0，search() 里有专门分支处理
        self.avgdl = 0.0

        # 增量接口用：chunk_id -> 在平行数组里的下标。后加，不改上面任何逻辑。
        self._pos = {}

    def build(self, docs: List[Tuple[str, str]]):
        """
        建索引（预处理，整个生命周期只需调用一次）。

        参数：
            docs: [(doc_id, text), ...]
                  doc_id = chunk_id；text = chunk 正文。
                  scripts/api.py 里就是 [(c["chunk_id"], c.get("content") or "") for c in all_chunks]

        返回：
            self（返回自己是为了支持链式写法：BM25().build(...).search(...)；
                 不过 api.py 里是分开写的：STATE["bm25"] = BM25().build(...)）

        复杂度：O(总 token 数)。一次遍历搞定所有统计量。
        """
        # 累加全库 token 总数，最后除以 N 得到 avgdl。
        # 用独立变量而不是每次都 sum(self.doc_len)，省一次遍历
        total_len = 0

        # enumerate 同时拿到下标 idx 和元素本身。
        # idx 很关键：它就是倒排索引里存的"文档下标"，也是平行数组的下标。
        # 这里把元组解包成 (doc_id, text)，所以下面能直接用两个变量
        for idx, (doc_id, text) in enumerate(docs):

            # 1) 分词：把这篇文档的正文切成 token 列表
            toks = tokenize(text)

            # 2) 统计词频 tf：Counter 会把列表变成 {词: 出现次数}。
            #    注意 Counter 的 key 是**去重后**的词，这正是我们后面想要的
            tf = Counter(toks)

            # 3) 把这篇文档的信息按 idx 顺序填进 3 个平行数组
            self.doc_ids.append(doc_id)
            self.tfs.append(tf)                 # 这篇文档的词频表
            self.doc_len.append(len(toks))      # 这篇文档的长度

            # 4) 累加总长度（用 len(toks)，和 doc_len 存的是同一个数）
            total_len += len(toks)

            # 5) 更新全库统计量：df 和倒排索引
            #    关键点：遍历的是 `tf`（Counter 的 key，即**去重后的词**），
            #    所以每个词对这篇文档只 +1 —— 恰好符合 df 的定义（同文档出现多次只算 1）。
            #    如果写成遍历 toks（有重复），df 就会被重复计数，算错。
            for t in tf:
                self.df[t] += 1                 # 含词 t 的文档数 +1
                self.inverted[t].append(idx)    # 在 t 的倒排表里记下"这篇文档有它"
                # 因为 tf 无重复，同一个 idx 对同一个 t 只会 append 一次，倒排表不会有重复项

        # 6) 收尾：算出 N 和 avgdl
        self.N = len(self.doc_ids)
        # 空语料时 N=0，会除零，所以必须判一下（这也解释了 search() 里为什么有 avgdl 分支）
        self.avgdl = (total_len / self.N) if self.N else 0.0

        # 7) 预计算 IDF——建索引时算一次，查询时直接用（空间换时间）
        #    字典推导式：对 self.df 里的每个 (词, 文档频率) 算一个 IDF
        #    公式：IDF = ln(1 + (N - df + 0.5) / (df + 0.5))
        #      · math.log 是**自然对数** ln（不是 log10，也不是 log2）
        #      · df 越小 → 分子越大 → IDF 越大 → 词越稀有越重要
        #      · 外层 +1 保证 IDF 恒 > 0（教科书 BM25 在词极常见时 IDF 会变负）
        #      · 0.5 是平滑项，避免 df=0 时公式退化
        self.idf = {
            t: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for t, df in self.df.items()
        }

        # 返回自身，方便链式调用
        return self

    def search(self, query: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        查询打分：返回 [(doc_id, bm25_score), ...]，按分数降序，最多 top_n 条。

        参数：
            query: 查询文本（用户的问题）
            top_n: 最多返回多少条（默认 20；调用方通常还要和向量结果融合，所以多取一些）

        返回：
            [(chunk_id, 分数), ...]，分数是 float，越大越相关。
            注意：这里返回的是**原始 chunk_id**，不是内部下标（内部下标的转换在最后一步做）。

        整体思路（与"暴力遍历全库"的关键区别）：
            只遍历「查询词命中的文档」，而不是全库所有文档。
        """
        # 1) 查询也走同一套分词，保证和建索引时口径一致（非常重要！）
        #    外面套 set()：把查询里的重复词去重。
        #    为什么？BM25 标准做法是把查询当"词的集合"处理，
        #    否则 "fastapi fastapi fastapi" 会把分数刷成 3 倍。
        qtoks = set(tokenize(query))

        # 2) 累加器：文档下标 -> 累计得分。
        #    用 defaultdict(float)，第一次访问某个 idx 时自动初始化为 0.0，
        #    于是可以直接写 `scores[idx] += ...` 而不用先判断 key 在不在
        scores = defaultdict(float)

        # 3) 遍历查询里的每个词，把它的贡献加到所有命中它的文档上
        for t in qtoks:

            # 3.1) 这个词在全库一次都没出现过 → 对任何文档都没有贡献，直接跳过。
            #      in 判断走的是 dict 的 key 查找，O(1)。
            #      这行很关键：正因为大部分查询词可能不存在，跳过能省大量时间
            if t not in self.inverted:
                continue                      # 这个词全库没有，跳过

            # 3.2) 取出预计算好的 IDF（同一次查询里，同一个词的 IDF 对所有文档都一样，
            #      所以提到循环外取一次，避免在内层循环里重复查 dict）
            idf = self.idf[t]

            # 3.3) 倒排表说"这些文档含有 t"，只遍历它们 —— 这就是倒排索引的价值
            for idx in self.inverted[t]:

                # 该词在这篇文档里的出现次数。
                # 用 .get(t, 0) 而不是 self.tfs[idx][t]：虽然逻辑上一定存在，
                # 但 .get 更防御，也符合"取不到就当 0 次"的语义，不会抛 KeyError
                tf = self.tfs[idx].get(t, 0)

                # 该文档长度（公式里的 dl）
                dl = self.doc_len[idx]

                # ---- BM25 分母：控制「长文档不该因为词多就得分高」 ----
                # 展开看：tf + k1 * (1 - b + b * dl / avgdl)
                #   · dl/avgdl 是"相对长度"：等于 1 说明长度刚好平均
                #   · b 控制归一化力度（0.75）
                #   · 文档越长 → 括号越大 → 分母越大 → 分数被压低（惩罚长文档）
                #
                # 末尾的 `if self.avgdl else tf + self.k1` 是**空语料兜底**：
                #   avgdl == 0 时 dl/avgdl 会 ZeroDivisionError，
                #   退化成分母 = tf + k1（相当于不做长度归一化）。
                # 注意运算符优先级：条件表达式优先级最低，所以实际是
                #   denom = (tf + k1 * (1 - b + b * dl / avgdl)) if self.avgdl else (tf + k1)
                #   两边的分支各自是一个完整的表达式，不会出现只包裹一部分的情况。
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl \
                        else tf + self.k1

                # 4) 把该词的贡献累加到这篇文档上。
                #    公式：idf * (tf * (k1 + 1)) / denom
                #      · tf * (k1 + 1)：分子，保证 tf 越大分越高
                #      · denom         ：分母含饱和与长度惩罚，抑制无限增长
                #      · idf           ：乘上词的稀有度权重
                #    一篇文档被查询里多个词命中时，多次 += 累加 = 各词得分之和
                scores[idx] += idf * (tf * (self.k1 + 1)) / denom

        # 5) 排序：按分数**降序**，取前 top_n。
        #    sorted 返回新列表（不原地改）；key=lambda x: -x[1] 表示"按第 2 个元素（分数）的相反数"排，
        #    取负数是最常见的降序写法（等价于 reverse=True，但写法更短）。
        #    [:top_n] 是切片，只保留前 top_n 条
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]

        # 6) 把内部下标 idx 换回对外的 chunk_id。
        #    ranked 里是 [(idx, score), ...]，推导式里解包成 (idx, s) 再查平行数组。
        #    返回后调用方拿到的就是能去 PG 回表的真实 chunk_id 了
        return [(self.doc_ids[idx], s) for idx, s in ranked]

    def __len__(self):
        """
        让 len(bm) 返回库里的文档数 N。

        这是 Python 的魔术方法（dunder method）：实现了它，len() 内建函数就能作用于本对象。
        api.py 里用 len(STATE["bm25"]) 之类的地方就能直接拿到底库规模。
        """
        return self.N

    # ==================================================================
    # 增量更新接口（后加，不改上面 build() / search() 的原有逻辑与注释）
    # 目的：上传只增量加入新 chunk，删除只移除对应 chunk，
    #       不再每次都 STATE["bm25"] = None 触发"下一次查询全库重建"的性能坑。
    # 粒度：始终以 chunk_id 为单位（和 build() 的入参一致）。
    # 并发：方法本身不加锁，由调用方（scripts/api.py 的 BM25_LOCK）保证串行。
    # ==================================================================
    def _recompute_idf(self):
        """按当前 self.df / self.N 重算 IDF 表。O(词表大小)，比全库重建便宜得多。"""
        self.idf = {
            t: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for t, df in self.df.items()
        }

    def _reindex(self):
        """
        从现有的 tfs / doc_ids / doc_len 重新推导 inverted / df / idf / _pos。
        不重新分词（tfs 已是分好的词频表），复杂度 O(总 token 数)。
        删除场景调用：摘掉若干 chunk 后下标位移，必须整体重排一次。
        """
        self.inverted = defaultdict(list)
        self.df = Counter()
        for idx, tf in enumerate(self.tfs):
            for t in tf:
                self.df[t] += 1
                self.inverted[t].append(idx)
        self.N = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self._recompute_idf()
        self._pos = {d: i for i, d in enumerate(self.doc_ids)}

    def _ensure_pos(self):
        """懒初始化 chunk_id -> 下标 映射（不改动 build()，首次增量时补齐）。"""
        if not self._pos:
            self._pos = {d: i for i, d in enumerate(self.doc_ids)}

    def add_chunk(self, chunk_id: str, text: str):
        """
        增量加入一个 chunk（上传入库时调用）。常见路径（全新 chunk）复杂度 O(新块 token 数)。
        若 chunk_id 已存在，先按删除处理再重新加入（等价于"更新"），删除会触发一次 _reindex。
        """
        self._ensure_pos()
        if chunk_id in self._pos:
            self.remove_chunks([chunk_id])        # 已存在 → 先删后加（_reindex 会校正下标）
        toks = tokenize(text)
        tf = Counter(toks)
        idx = len(self.doc_ids)
        self.doc_ids.append(chunk_id)
        self.tfs.append(tf)
        self.doc_len.append(len(toks))
        self._pos[chunk_id] = idx
        for t in tf:
            self.df[t] += 1
            self.inverted[t].append(idx)
        self.N = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self._recompute_idf()

    def add_chunks(self, pairs):
        """批量增量加入：pairs = [(chunk_id, text), ...]"""
        for cid, text in pairs:
            self.add_chunk(cid, text)

    def remove_chunks(self, chunk_ids):
        """
        批量删除（一次删一整个文档的所有 chunk）。只重排一次，O(总块数)。
        删除是低频操作，全量重排可接受；上传（高频）走 add_chunk 才是真正增量。
        """
        drop = set(chunk_ids)
        if not drop:
            return
        self._ensure_pos()
        if not (drop & set(self.doc_ids)):
            return                              # 索引里根本没有这些块，无需动
        keep = [i for i, d in enumerate(self.doc_ids) if d not in drop]
        self.doc_ids = [self.doc_ids[i] for i in keep]
        self.tfs = [self.tfs[i] for i in keep]
        self.doc_len = [self.doc_len[i] for i in keep]
        self._reindex()

    def remove_chunk(self, chunk_id: str):
        """删除单个 chunk（按 chunk_id）。"""
        self.remove_chunks([chunk_id])
