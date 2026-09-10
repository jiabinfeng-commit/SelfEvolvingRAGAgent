#!/usr/bin/env python3
"""
从 ModelScope 下载 embedding 模型到本地目录。

为什么不用 HuggingFace：
  国内 ECS 访问 huggingface.co / hf-mirror.com 基本不通（连接超时），
  ModelScope（魔搭）是阿里系镜像站，国内直连可用，且 BAAI/bge-*
  系列模型在上面是官方同步的。

用法：
    python scripts/fetch_model.py                      # 下载 config 里指定的默认模型
    python scripts/fetch_model.py BAAI/bge-base-zh-v1.5

下载完成后，把 config.py 的 EMBED_MODEL 指向本地目录即可离线加载。
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API_LIST = "https://modelscope.cn/api/v1/models/{model}/repo/files?Revision=master&Recursive=true"
API_FILE = "https://modelscope.cn/api/v1/models/{model}/repo?Revision=master&FilePath={path}"

# 模型缓存根目录，可通过环境变量覆盖
ROOT = Path(os.environ.get("MODEL_ROOT", "/opt/rag-agent/models"))

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def list_files(model: str) -> list[dict]:
    url = API_LIST.format(model=model)
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    if data.get("Code") != 200:
        raise RuntimeError(f"ModelScope API 返回异常: {data}")
    return [f for f in data["Data"]["Files"] if f["Type"] == "blob"]


def download(model: str, rel_path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  skip (已存在)  {rel_path}")
        return
    url = API_FILE.format(model=model, path=urllib.parse.quote(rel_path))
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fp:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fp.write(chunk)
    tmp.rename(dest)


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    target = ROOT / model.replace("/", "__")
    print(f"模型: {model}\n目标: {target}\n")

    files = list_files(model)
    total = sum(f["Size"] for f in files)
    print(f"共 {len(files)} 个文件，{total / 1024 / 1024:.1f} MB\n")

    for f in files:
        rel = f["Path"]
        size_mb = f["Size"] / 1024 / 1024
        print(f"  ↓ {rel}  ({size_mb:.2f} MB)")
        download(model, rel, target / rel)

    print(f"\n完成: {target}")
    print(f"config.py 里改成: EMBED_MODEL = r'{target}'")


if __name__ == "__main__":
    main()
