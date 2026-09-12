# -*- coding: utf-8 -*-
"""
全局配置

设计原则：
- 凭据/路径/运行参数都可用环境变量覆盖
- 优先级：环境变量 > .env 文件 > 代码内默认（仅非敏感字段有默认）
- 凭据不进代码，必须从 .env 读；.env 本身不进 git

.env 文件约定：
  - 项目根放一份 .env（含真实密码，仅本机可见）
  - 同一目录放 .env.example（只含键名 + 注释，可入仓作模板）
  - .env 已加入 .gitignore
"""
import os
import sys

# ---------- 加载 .env ----------
try:
    from dotenv import load_dotenv
    _HERE = os.path.dirname(os.path.abspath(__file__))
    # 按这个顺序找：仓库根、app 同级、当前工作目录
    for _p in [
        os.path.join(_HERE, "..", ".env"),
        os.path.join(_HERE, ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]:
        if os.path.isfile(_p):
            load_dotenv(_p, override=False)
            break
except ImportError:
    pass


def _get(name: str, default):
    """
    os.getenv 的加强版：「环境变量存在但为空」与「未设置」都走 default。
    dotenv 会把 .env 里的 `KEY=` 解析成空串，这两种情况我们应该一致对待。
    """
    v = os.getenv(name)
    if v:                       # 非空走真值
        return v
    if v is None:               # 未设置
        return default
    return default              # 设了但是空 → 走 default


def _getint(name: str, default: int) -> int:
    v = _get(name, None)
    if v is None:
        return default
    return int(v)


def _getbool(name: str, default: bool) -> bool:
    """读布尔配置：true/yes/on/1 视为真，其余（含未设置/空）走 default。"""
    v = _get(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# ---------- 项目根目录 ----------
# 优先级：环境变量 > /opt/rag-agent（云上标志位）> 仓库根（本地 = core 的父目录）
_LOCAL_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getenv("RAG_BASE_DIR"):
    BASE_DIR = os.environ["RAG_BASE_DIR"]
elif os.path.isdir("/opt/rag-agent"):
    BASE_DIR = "/opt/rag-agent"
else:
    BASE_DIR = _LOCAL_DEFAULT

# ---------- 语料 ----------
CORPUS_DIR = _get(
    "RAG_CORPUS_DIR",
    os.path.join(BASE_DIR, "corpus", "clean"),
)

# ---------- PostgreSQL（存 chunk 原文与元数据） ----------
# 非敏感字段保留默认值；密码等敏感字段无默认，缺了直接报错（见 validate()）
PG_HOST = _get("PG_HOST", "localhost")
PG_PORT = _getint("PG_PORT", 5432)
PG_DB = _get("PG_DB", "rag")
PG_USER = _get("PG_USER", "rag")
PG_PASSWORD = _get("PG_PASSWORD", "")       # ← 不再硬编码

# ---------- Milvus Lite（只存向量） ----------
MILVUS_PATH = _get("MILVUS_PATH", os.path.join(BASE_DIR, "data", "milvus.db"))
COLLECTION_NAME = _get("MILVUS_COLLECTION", "chunks")

# ---------- HuggingFace 镜像 ----------
# 国内直连 huggingface.co 不通（实测 6s 超时、一个字节都收不到），
# 于是 sentence-transformers / huggingface_hub 任何"去 HF 取模型"的动作都会卡死。
# 默认切到 hf-mirror.com 这个社区镜像；要切回官方源，在 .env 里写：
#     HF_ENDPOINT=https://huggingface.co
#
# 【为什么必须写在这里、而且必须写进 os.environ】
# huggingface_hub 是在 import 时就读取 HF_ENDPOINT 决定下载地址的，
# 之后再改 os.environ 就不生效了。而本项目所有模块都会先 import core.config，
# 且 sentence_transformers 是在 BGEEmbedder.__init__ 里**惰性** import 的
# （见 core/embedder.py），所以在这里赋值一定早于 huggingface_hub 被加载。
# 用"非空才写"而不是 setdefault：.env 里写成 `HF_ENDPOINT=`（空串）时，
# setdefault 会因为"键已存在"而跳过，反而把空串留在环境里 → 下载地址变成空。
HF_ENDPOINT = _get("HF_ENDPOINT", "https://hf-mirror.com")
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT

# ---------- Embedding ----------
# bge-small-zh-v1.5：512 维，约 100MB，中文效果好且轻量
#
# 国内机器访问 huggingface.co 不通，模型由 scripts/fetch_model.py 从
# ModelScope 预先下载到本地。这里优先用本地目录，不存在时再退回 HF 官方 id。
MODEL_ROOT = _get("MODEL_ROOT", os.path.join(BASE_DIR, "models"))
_LOCAL_EMBED = os.path.join(MODEL_ROOT, "BAAI__bge-small-zh-v1.5")
EMBED_MODEL = _get(
    "EMBED_MODEL",
    _LOCAL_EMBED if os.path.isdir(_LOCAL_EMBED) else "BAAI/bge-small-zh-v1.5",
)
EMBED_DIM = _getint("EMBED_DIM", 512)
EMBED_BATCH_SIZE = _getint("EMBED_BATCH_SIZE", 32)

# bge 系列做检索时，查询侧建议加这句前缀（官方推荐，能小幅提升召回）
QUERY_PREFIX = _get("QUERY_PREFIX", "为这个句子生成表示以用于检索相关文章：")

# ---------- Reranker（阶段 4 扩展：混合检索融合后精排） ----------
# 默认开。模式：auto = 优先 cross-encoder（BAAI/bge-reranker-v2-m3），
# 加载不到（模型没下/没网）就自动退 bi（复用现有 bge embedding，零下载）。
# cross 模式需要把模型下到本地；下载后可设 RERANKER_MODEL 指向本地目录，auto 会自动用上。
# 防御性：本地模型目录存在时优先用它（此时 RERANKER_MODEL 没在 .env 显式设置才会走到这里）。
# 注意：若 .env 里写了非空 RERANKER_MODEL（如默认的 HF id），本回退不触发，
#       需靠 scripts/fetch_model.py 下载后把本地路径写回 .env。
_LOCAL_RERANKER = os.path.join(MODEL_ROOT, "BAAI__bge-reranker-v2-m3")
RERANKER_ENABLED = _getbool("RERANKER_ENABLED", True)
RERANKER_MODEL = _get(
    "RERANKER_MODEL",
    _LOCAL_RERANKER if os.path.isdir(_LOCAL_RERANKER) else "BAAI/bge-reranker-v2-m3",
)
RERANKER_MODE = _get("RERANKER_MODE", "auto")            # cross / bi / auto
RERANKER_CANDIDATE_TOP_N = _getint("RERANKER_CANDIDATE_TOP_N", 20)

# ---------- LLM（阶段 2：检索到上下文后，用它生成最终答案） ----------
# 后端可切换：
#   ollama  —— 本地免费（默认），需要本机起了 Ollama 服务
#   openai  —— 任意 OpenAI 兼容 API（如 gpt-4o-mini、DeepSeek、通义百炼等），需要 API Key
LLM_BACKEND = _get("LLM_BACKEND", "openai")
# ollama 模式填模型名（如 qwen3.7-max / llama3）；openai 模式填模型 id（如 gpt-4o-mini）
LLM_MODEL = _get("LLM_MODEL", "qwen3.7-max")
OLLAMA_URL = _get("OLLAMA_URL", "http://localhost:11434")
OPENAI_API_KEY = _get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = _get("OPENAI_BASE_URL", "")
# 召回多少块拼进 prompt（阶段 2 的 top-k）
RAG_TOP_K = _getint("RAG_TOP_K", 5)
# 单块上下文拼进 prompt 时的最大字符数（避免 prompt 过长把无关内容塞进来）
RAG_CONTEXT_MAX_CHARS = _getint("RAG_CONTEXT_MAX_CHARS", 1200)


# ---------- 校验 ----------
def validate() -> list:
    """
    检查启动必需的配置，缺失项返回错误列表。
    应在 main 入口开头调用，让错误早点暴露。
    """
    errs = []
    if not PG_PASSWORD:
        errs.append("PG_PASSWORD 未设置（请在 .env 或环境变量里填）")
    if not os.path.isdir(CORPUS_DIR):
        errs.append(f"语料目录不存在: {CORPUS_DIR}")
    if not os.path.isdir(EMBED_MODEL) and not EMBED_MODEL.startswith(("BAAI/", "./", "/")):
        errs.append(f"EMBED_MODEL 路径不存在且不是已知的 HF id: {EMBED_MODEL}")
    # 阶段 2 新增：LLM 后端合法性校验
    if LLM_BACKEND not in ("ollama", "openai"):
        errs.append(f"未知 LLM_BACKEND: {LLM_BACKEND}（可选 ollama / openai）")
    if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
        errs.append("LLM_BACKEND=openai 时需要设置 OPENAI_API_KEY")
    return errs


def summary() -> str:
    """打印配置摘要（隐藏密码）"""
    return (
        f"语料目录    : {CORPUS_DIR}\n"
        f"PostgreSQL  : {PG_USER}:****@{PG_HOST}:{PG_PORT}/{PG_DB}\n"
        f"Milvus Lite : {MILVUS_PATH} (collection={COLLECTION_NAME})\n"
        f"Embedding   : {EMBED_MODEL} ({EMBED_DIM} 维)\n"
        f"Reranker    : {'开' if RERANKER_ENABLED else '关'} (mode={RERANKER_MODE})\n"
        f"LLM         : {LLM_BACKEND} / {LLM_MODEL}"
    )


if __name__ == "__main__":
    print(summary())
    errs = validate()
    if errs:
        print("\n配置错误：")
        for e in errs:
            print(f"  ✗ {e}")
        sys.exit(1)
    print("\n配置 OK ✓")
