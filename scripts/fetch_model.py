#!/usr/bin/env python3
"""
从 ModelScope 下载模型到本地目录（国内直连可用，无需 HF）。

为什么不用 HuggingFace：
  国内 ECS 访问 huggingface.co / hf-mirror.com 基本不通（连接超时），
  ModelScope（魔搭）是阿里系镜像站，国内直连可用，且 BAAI/bge-*
  系列模型在上面是官方同步的。

默认行为（一条命令搞定）：
  同时下载 embedding 模型（bge-small-zh-v1.5）与 reranker 模型
  （bge-reranker-v2-m3），下载完成后把它们的「本地绝对路径」自动写回
  项目根目录的 .env（EMBED_MODEL / RERANKER_MODEL），并提示确认
  RERANKER_MODE=auto。这样重启服务后 cross-encoder 会直接离线加载，
  不会再因为连不上 HF 而退到 bi。

用法：
    python scripts/fetch_model.py                      # 下载默认两个模型
    python scripts/fetch_model.py BAAI/bge-base-zh-v1.5   # 只下载指定模型
    MODEL_ROOT=/data/models python scripts/fetch_model.py  # 指定模型根目录
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API_LIST = "https://modelscope.cn/api/v1/models/{model}/repo/files?Revision=master&Recursive=true"
API_FILE = "https://modelscope.cn/api/v1/models/{model}/repo?Revision=master&FilePath={path}"

# 默认同时下载 embedding 与 reranker
DEFAULT_MODELS = ["BAAI/bge-small-zh-v1.5", "BAAI/bge-reranker-v2-m3"]


def resolve_model_root() -> Path:
    """
    模型根目录，逻辑与 core/config.py 的 BASE_DIR / MODEL_ROOT 保持一致：
      1) 环境变量 MODEL_ROOT 优先
      2) 否则 RAG_BASE_DIR
      3) 否则若存在 /opt/rag-agent（云上标志）则用之
      4) 否则用仓库根
    """
    if os.environ.get("MODEL_ROOT"):
        return Path(os.environ["MODEL_ROOT"])
    if os.environ.get("RAG_BASE_DIR"):
        base = Path(os.environ["RAG_BASE_DIR"])
    elif os.path.isdir("/opt/rag-agent"):
        base = Path("/opt/rag-agent")
    else:
        base = Path(__file__).resolve().parent.parent  # 仓库根
    return base / "models"


def find_dotenv() -> Path | None:
    """按 core/config.py 同样的顺序找 .env。"""
    here = Path(__file__).resolve().parent
    for c in [
        here.parent / ".env",                 # 仓库根（本地）
        Path("/opt/rag-agent") / ".env",      # 云上
        Path(os.getcwd()) / ".env",
    ]:
        if c.is_file():
            return c
    return None


def update_env(env_path: Path, updates: dict) -> None:
    """
    把模型本地路径写回 .env。
    - 只覆盖「默认占位 / 空值」：已是自定义本地路径则保留，避免误改用户配置。
    - .env 不存在时不做任何写入，仅打印可复制行（避免从零生成导致丢失凭据）。
    """
    lines = env_path.read_text(encoding="utf-8").splitlines()
    seen = set()
    out = []
    for line in lines:
        stripped = line.strip()
        matched = False
        for key, val in updates.items():
            if stripped.startswith(key + "=") or stripped.startswith(key + " ="):
                seen.add(key)
                cur = stripped.split("=", 1)[1].strip()
                is_default = (cur == "" or cur.startswith("BAAI/"))
                out.append(f"{key}={val}" if is_default else line)
                matched = True
                break
        if not matched:
            out.append(line)
    for key, val in updates.items():
        if key not in seen:
            out.append(f"{key}={val}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def list_files(model: str) -> list:
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


def model_local_name(model: str) -> str:
    return model.replace("/", "__")


def main() -> None:
    models = sys.argv[1:] or DEFAULT_MODELS
    root = resolve_model_root()
    root.mkdir(parents=True, exist_ok=True)
    print(f"模型根目录: {root}\n")

    # model -> 对应的 .env 键（embedding 用 EMBED_MODEL，reranker 用 RERANKER_MODEL）
    env_updates = {}
    for model in models:
        target = root / model_local_name(model)
        key = "RERANKER_MODEL" if "reranker" in model else "EMBED_MODEL"
        print(f"模型: {model}\n目标: {target}\n")
        files = list_files(model)
        total = sum(f["Size"] for f in files)
        print(f"共 {len(files)} 个文件，{total / 1024 / 1024:.1f} MB\n")
        for f in files:
            rel = f["Path"]
            print(f"  ↓ {rel}  ({f['Size'] / 1024 / 1024:.2f} MB)")
            download(model, rel, target / rel)
        env_updates[key] = str(target)
        print(f"完成: {target}\n")

    # 下载完，自动把本地路径写回 .env
    env_path = find_dotenv()
    if env_path:
        update_env(env_path, env_updates)
        print(f"[ok] 已把本地路径写入 {env_path}：")
        for k, v in env_updates.items():
            print(f"     {k}={v}")
    else:
        print("[提示] 未找到 .env，请手动把以下两行加入你的环境配置：")
        for k, v in env_updates.items():
            print(f"     {k}={v}")

    print("\n下一步：确认 .env 里 RERANKER_MODE=auto（或 cross），重启服务即可用上 cross-encoder。")


if __name__ == "__main__":
    main()
