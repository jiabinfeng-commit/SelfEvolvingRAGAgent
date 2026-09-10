# -*- coding: utf-8 -*-
"""
LLM 客户端（阶段 2：根据检索到的上下文生成最终答案）

为什么单独抽一层？
- 阶段 2 当前最小闭环只关心"给我 system + user 两段文本，返回一段回答"。
- 具体用哪家模型、本地还是云端，应该可插拔：今天用 Ollama 本地跑，
  明天想换 OpenAI / DeepSeek / 通义百炼，只改配置不改业务代码。

设计：
- BaseLLM：统一接口 generate(system, user) -> str
- OllamaLLM：打本地 Ollama 的 /api/chat（免费、无需 key，适合开发期）
- OpenAILLM：打任意 OpenAI 兼容 /chat/completions（需要 key）
- 两者都用 Python 标准库 urllib 发 HTTP，不引入 requests 依赖

为什么不引 requests？项目一贯保持依赖精简（见阶段 1 踩坑：pip 装包慢且重），
urllib 在标准库里，发 JSON POST 足够了。
"""
import json
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import List, Dict

from core import config


class BaseLLM(ABC):
    """所有 LLM 后端统一接口"""

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """
        生成回答。

        :param system: 系统提示词（这里通常是"只依据上下文作答"的指令）
        :param user:   用户提示词（这里通常是"上下文 + 问题"）
        :return:       模型生成的纯文本回答
        """
        ...

    @staticmethod
    def _post_json(url: str, payload: Dict, headers: Dict, timeout: int = 120) -> Dict:
        """
        标准库发 JSON POST 的小工具（避免引入 requests）。

        :raises RuntimeError: HTTP 非 2xx 或网络异常时，把错误信息带上，方便排查
        """
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body)
        except urllib.error.HTTPError as e:
            # 服务器返回了错误状态码（如 404 模型不存在、401 key 错误）
            detail = e.read().decode("utf-8", errors="ignore")[:300]
            raise RuntimeError(f"LLM 请求失败 HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            # 连不上（Ollama 没起 / 地址错 / 没网）
            raise RuntimeError(f"LLM 连接失败（检查服务是否启动/地址是否正确）: {e.reason}") from e


class OllamaLLM(BaseLLM):
    """
    本地 Ollama 后端（默认）

    前提：本机已启动 Ollama 并 pull 了对应模型，例如
        ollama run qwen2.5:7b
    接口：POST {OLLAMA_URL}/api/chat
    请求体：{"model":..., "messages":[{"role":"system"},{"role":"user"}], "stream":false}
    返回：{"message":{"content":"..."}, ...}
    """

    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or config.LLM_MODEL
        self.base_url = (base_url or config.OLLAMA_URL).rstrip("/")
        self.chat_url = f"{self.base_url}/api/chat"

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,   # 一次性返回，方便解析；不要流式
        }
        headers = {"Content-Type": "application/json"}
        resp = self._post_json(self.chat_url, payload, headers)
        # Ollama 正常路径：resp["message"]["content"]
        return (resp.get("message") or {}).get("content", "").strip()


class OpenAILLM(BaseLLM):
    """
    OpenAI 兼容后端（gpt-4o-mini / DeepSeek / 通义百炼 等）

    前提：设置 OPENAI_API_KEY（和可选的 OPENAI_BASE_URL）
    接口：POST {OPENAI_BASE_URL}/chat/completions
    请求体：{"model":..., "messages":[{"role":"system"},{"role":"user"}]}
    返回：{"choices":[{"message":{"content":"..."}}]}
    """

    def __init__(self, model: str = None, api_key: str = None, base_url: str = None):
        if not (api_key or config.OPENAI_API_KEY):
            raise ValueError("OpenAI 后端需要 OPENAI_API_KEY")
        self.model = model or config.LLM_MODEL
        self.api_key = api_key or config.OPENAI_API_KEY
        self.base_url = (base_url or config.OPENAI_BASE_URL).rstrip("/")
        self.chat_url = f"{self.base_url}/chat/completions"

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = self._post_json(self.chat_url, payload, headers)
        # OpenAI 路径：resp["choices"][0]["message"]["content"]
        choices = resp.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content", "").strip()


def get_llm(backend: str = None, **kwargs) -> BaseLLM:
    """
    工厂：按名字拿 LLM 后端。

    :param backend: ollama / openai，缺省读 config.LLM_BACKEND
    """
    name = backend or config.LLM_BACKEND
    if name == "ollama":
        return OllamaLLM(**kwargs)
    if name == "openai":
        return OpenAILLM(**kwargs)
    raise ValueError(f"未知 LLM_BACKEND: {name}（可选 ollama / openai）")
