# -*- coding: utf-8 -*-
"""从 jsdelivr CDN 下载 FastAPI 官方中文文档作为 RAG 语料"""
import json, os, time, urllib.request

# 输出目录基于脚本位置推断（scripts/ 的上一级 = 项目根），
# 不要写死绝对路径，否则别人 clone 下来跑会直接往不存在的目录写。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw")
os.makedirs(OUT, exist_ok=True)
CDN = "https://cdn.jsdelivr.net/gh/fastapi/fastapi@master"

# 挑选：tutorial 全量 + 高频 advanced + deployment + 参考
TARGETS = [
    "index.md",
    "tutorial/first-steps.md",
    "tutorial/path-params.md",
    "tutorial/query-params.md",
    "tutorial/body.md",
    "tutorial/query-params-str-validations.md",
    "tutorial/path-params-numeric-validations.md",
    "tutorial/body-multiple-params.md",
    "tutorial/body-fields.md",
    "tutorial/body-nested-models.md",
    "tutorial/extra-models.md",
    "tutorial/response-model.md",
    "tutorial/response-status-code.md",
    "tutorial/handling-errors.md",
    "tutorial/dependencies/index.md",
    "tutorial/dependencies/first-steps.md",
    "tutorial/dependencies/classes-as-dependencies.md",
    "tutorial/dependencies/sub-dependencies.md",
    "tutorial/dependencies/global-dependencies.md",
    "tutorial/security/index.md",
    "tutorial/security/first-steps.md",
    "tutorial/security/oauth2-jwt.md",
    "tutorial/metadata.md",
    "tutorial/static-files.md",
    "tutorial/testing.md",
    "tutorial/debugger.md",
    "advanced/middleware.md",
    "advanced/events.md",
    "advanced/settings.md",
    "advanced/websockets.md",
    "advanced/behind-a-proxy.md",
    "advanced/custom-response.md",
    "advanced/response-headers.md",
    "advanced/dataclasses.md",
    "deployment/index.md",
    "deployment/concepts.md",
    "deployment/docker.md",
    "async.md",
    "alternatives.md",
    "benchmarks.md",
]

def fetch(path):
    url = f"{CDN}/docs/zh/docs/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")

ok, fail = 0, []
for i, p in enumerate(TARGETS, 1):
    name = p.replace("/", "__")
    dest = os.path.join(OUT, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 200:
        ok += 1
        continue
    try:
        txt = fetch(p)
        open(dest, "w", encoding="utf-8").write(txt)
        ok += 1
        print(f"[{i:2d}/{len(TARGETS)}] {len(txt):6d} 字  {p}")
    except Exception as e:
        fail.append((p, str(e)[:60]))
        print(f"[{i:2d}/{len(TARGETS)}] 失败  {p}  {e}")
    time.sleep(0.15)

print("\n" + "=" * 56)
print(f"成功 {ok} / {len(TARGETS)}")
if fail:
    print("失败:", fail)

total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".md"))
chars = 0
for f in os.listdir(OUT):
    if f.endswith(".md"):
        chars += len(open(os.path.join(OUT, f), encoding="utf-8").read())
print(f"语料目录: {OUT}")
print(f"文件数 {len([f for f in os.listdir(OUT) if f.endswith('.md')])} | 总字节 {total/1024:.0f} KB | 总字符 {chars}")
