# -*- coding: utf-8 -*-
"""
清洗 FastAPI 中文文档语料：
- 移除 mkdocs 占位符 {* ../../docs_src/xxx.py *}（代码引用，实际代码未包含）
- 移除 /// tip | 提示 这类 mkdocs 标记行（保留正文）
- 移除 HTML 标签（termy 终端模拟块、details/summary 折叠块等）
- 移除标题锚点 {#anchor}
- 压缩多余空行

用法：python scripts/clean_corpus.py
输出：data/clean/（清洗后的纯净 markdown）
"""
import os, re

# 动态定位项目根目录，避免路径写死导致换机器就跑不了
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "clean")
os.makedirs(OUT, exist_ok=True)

PATTERNS = [
    # mkdocs 代码引用占位符（可能跨行）
    (re.compile(r"\{\*\s*[^*]*?\s*\*\}", re.S), ""),
    # /// tip | 提示、//// tab | Python 3.10+ 这类整行标记（含缩进、含 3 个以上斜杠）
    # 用 /{3,} 而非 /+，避免误伤 /code/requirements.txt 这类路径
    (re.compile(r"^\s*/{3,}\s*\w*\s*\|.*$", re.M), ""),
    (re.compile(r"^\s*/{3,}\s*$", re.M), ""),
    # HTML 标签
    (re.compile(r"</?(?:div|details|summary|abbr|dfn|font|span|u|b|i|code|br)[^>]*>", re.I), ""),
    # 标题锚点 {#first-steps} / { #first-steps }（允许花括号内前导空格）
    (re.compile(r"\{\s*#[^}]*\}"), ""),
    # HTML 注释
    (re.compile(r"<!--.*?-->", re.S), ""),
    # 连续空行压缩
    (re.compile(r"\n{3,}"), "\n\n"),
]

def clean(text: str) -> str:
    for pat, rep in PATTERNS:
        text = pat.sub(rep, text)
    # 去掉行尾空白
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip() + "\n"

before_total = after_total = 0
rows = []
for f in sorted(os.listdir(RAW)):
    if not f.endswith(".md"):
        continue
    src = open(os.path.join(RAW, f), encoding="utf-8").read()
    dst = clean(src)
    open(os.path.join(OUT, f), "w", encoding="utf-8").write(dst)
    before_total += len(src)
    after_total += len(dst)
    rows.append((f, len(src), len(dst)))

print("=" * 68)
print(f"{'文档':<52}{'清洗前':>8}{'清洗后':>8}")
print("=" * 68)
for f, a, b in rows:
    print(f"{f:<52}{a:>8}{b:>8}")
print("=" * 68)
print(f"{'合计':<52}{before_total:>8}{after_total:>8}")
print(f"清洗掉 {before_total - after_total} 字符（{(1-after_total/before_total)*100:.1f}%）")
print(f"\n输出目录: {OUT}")

# 校验：还有没有残留占位符
left = 0
for f in os.listdir(OUT):
    t = open(os.path.join(OUT, f), encoding="utf-8").read()
    left += len(re.findall(r"\{\*|\{#|///|<div|<details", t))
print(f"残留标记数: {left}  {'✅ 已清理干净' if left == 0 else '⚠ 仍有残留'}")
