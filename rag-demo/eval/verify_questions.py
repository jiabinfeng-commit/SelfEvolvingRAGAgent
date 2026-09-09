# -*- coding: utf-8 -*-
"""
校验评测集质量：
1. 可答题：gold_doc 是否存在、answer_keywords 是否能在语料中命中
2. 拒答题：核心词是否真的搜不到（若搜到则提示可能误标）
"""
import json, os, sys

ROOT = "/Users/fengjiabin/WorkBuddy/2026-09-09-14-25-42/rag-demo"
RAW = os.path.join(ROOT, "data/raw")
Q_FILE = os.path.join(ROOT, "eval/questions.json")

corpus = {}
for f in os.listdir(RAW):
    if f.endswith(".md"):
        corpus[f] = open(os.path.join(RAW, f), encoding="utf-8").read()

full_text = "\n".join(corpus.values())
data = json.load(open(Q_FILE, encoding="utf-8"))
qs = data["questions"]

print("=" * 72)
print(f"语料：{len(corpus)} 篇 | 评测题：{len(qs)} 条")
print("=" * 72)

problems = []
stats = {"single_hop": [0, 0], "multi_hop": [0, 0], "unanswerable": [0, 0]}

for q in qs:
    qid = q["id"]
    typ = q["type"]
    stats[typ][1] += 1

    if typ == "unanswerable":
        # 提取问题里的英文专有名词，检查是否意外出现在语料中
        import re
        terms = re.findall(r"[A-Za-z][A-Za-z0-9\.\-]{2,}", q["question"])
        hits = [t for t in terms if t.lower() in full_text.lower()]
        # 排除太通用的词
        generic = {"the", "and", "for", "how", "what", "with", "api", "fastapi", "http"}
        real = [t for t in hits if t.lower() not in generic]
        if real and not q.get("human_verified"):
            problems.append(f"#{qid} [拒答] 核心词在语料中出现过，可能误标: {real}")
            print(f"  ⚠ #{qid:2d} 拒答题，但语料中含 {real} —— 需人工确认是否真的答不了")
        else:
            stats[typ][0] += 1
            note = f"（已人工确认，语料仅提及 {real}）" if real else ""
            print(f"  ✓ #{qid:2d} 拒答题{note}")
        continue

    # 可答题
    doc = q.get("gold_doc")
    if not doc or doc not in corpus:
        problems.append(f"#{qid} gold_doc 不存在: {doc}")
        print(f"  ✗ #{qid:2d} gold_doc 缺失: {doc}")
        continue

    text = corpus[doc]
    # 关键词既可在指定文档命中，也可在全语料命中（多跳）
    missing = [k for k in q["answer_keywords"] if k not in text]
    missing_but_global = [k for k in missing if k in full_text]
    still_missing = [k for k in missing if k not in full_text]

    if not missing:
        stats[typ][0] += 1
        print(f"  ✓ #{qid:2d} {typ:<12} 全部 {len(q['answer_keywords'])} 个关键词命中 {doc}")
    elif still_missing:
        problems.append(f"#{qid} 关键词全语料都搜不到: {still_missing}")
        print(f"  ✗ #{qid:2d} {typ:<12} 关键词搜不到: {still_missing}")
    else:
        # 在别的文档里找到了（多跳属正常，单跳需留意）
        stats[typ][0] += 1
        flag = "" if typ == "multi_hop" else "  (单跳题却要跨文档，请核对 gold_doc)"
        print(f"  ~ #{qid:2d} {typ:<12} 关键词在其他文档命中: {missing_but_global}{flag}")

print("\n" + "=" * 72)
print("统计")
print("=" * 72)
for t, (ok, total) in stats.items():
    print(f"  {t:<14} 通过 {ok}/{total}")

print()
if problems:
    print(f"⚠ 发现 {len(problems)} 处需要修正：")
    for p in problems:
        print("   -", p)
    sys.exit(1)
else:
    print("✅ 全部评测题校验通过")
