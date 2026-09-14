# -*- coding: utf-8 -*-
"""
统一日志启动器（阶段 10 可观测性基建）

为什么需要它？
-----------------------------------------------------------------------
项目之前几乎没有日志，出错基本靠 print，定位问题极慢——跟 Java + IDEA
那种「框架默认刷一堆结构化日志、异常点一下就停在源码」的体验差一大截。

这个模块做四件事：
  1. 统一日志格式：时间 | 级别 | 模块:函数:行号 | 消息（loguru 还带颜色）。
  2. 首选 loguru（体验最好），没装就退而用标准库 logging（兜底，照样能跑）。
  3. 把 FastAPI / uvicorn / sentence-transformers / transformers 这些「用标准库
     logging 打日志」的第三方库，也收编进同一条通道，它们的 INFO/DEBUG 会按我们的
     格式打印出来（默认这些库的 INFO 是被静音的，所以你以前几乎看不到它们说话）。
  4. 装全局未捕获异常钩子：主线程 + 子线程一旦漏异常，直接打完整堆栈（含现场），
     不再只丢一句孤零零的 traceback。

用法（在每个脚本入口顶部、import core 之后调一次即可）：
    from core.logging_setup import init_logging, get_logger
    init_logging()                       # 级别读 RAG_LOG_LEVEL，默认 INFO
    logger = get_logger(__name__)

级别控制：环境变量 RAG_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR，
          或 core/config.py 里的 LOG_LEVEL。
落盘（可选）：环境变量 RAG_LOG_FILE=路径，或 init_logging(log_file="xxx")，
          自动按 20MB 切割、保留 3 份。
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Optional

# 装了 loguru 用 loguru（带颜色、带 backtrace 上下文），没装退标准库 logging。
try:
    from loguru import logger as _loguru
    _HAS_LOGURU = True
except Exception:  # pragma: no cover
    _HAS_LOGURU = False

# 给 loguru 日志绑一个「组件名」基线默认值。
# 说明：loguru 的 {name} 令牌永远是调用方模块名（不可控），我们用自定义 extra 字段
# {extra[comp]} 来显示组件名——自己的模块显示 __name__，第三方库显示它自己的 logger 名。
if _HAS_LOGURU:
    _loguru = _loguru.bind(comp="root")

_INITIALIZED = False


def _resolve_level() -> str:
    env = os.getenv("RAG_LOG_LEVEL")
    if env:
        return env.strip().upper()
    try:
        from core import config  # 懒导入，避免循环依赖
        return getattr(config, "LOG_LEVEL", "INFO").upper()
    except Exception:
        return "INFO"


def _format() -> str:
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[comp]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{message}"
    )


def init_logging(level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """
    初始化全局日志。幂等：重复调用只会重新应用级别 / sink，不会叠加重复输出。
    """
    global _INITIALIZED
    lvl = (level or _resolve_level()).upper()
    log_file = log_file or os.getenv("RAG_LOG_FILE") or None

    if _HAS_LOGURU:
        _init_loguru(lvl, log_file)
    else:
        _init_stdlib(lvl, log_file)

    _capture_stdlib()      # 收编第三方库的 logging
    _install_excepthook()  # 漏网异常也别跑掉
    _INITIALIZED = True

    if _HAS_LOGURU:
        _loguru.debug("日志通道已初始化（loguru），级别={}", lvl)
    else:
        logging.getLogger("root").info(
            "日志通道已初始化（标准库 logging，未装 loguru），级别=%s", lvl
        )


def _init_loguru(lvl: str, log_file: Optional[str]) -> None:
    _loguru.remove()  # 清掉 loguru 默认 sink，重新配，避免重复
    sinks = [{
        "sink": sys.stderr,
        "level": lvl,
        "format": _format(),
        "colorize": True,
        "backtrace": True,    # 异常时打印完整回溯
        "diagnose": True,     # 回溯里附带变量值（定位问题的关键）
    }]
    if log_file:
        sinks.append({
            "sink": log_file,
            "level": lvl,
            "format": _format(),
            "rotation": "20 MB",
            "retention": 3,
            "encoding": "utf-8",
        })
    for kw in sinks:
        _loguru.add(**kw)


def _init_stdlib(lvl: str, log_file: Optional[str]) -> None:
    root = logging.getLogger()
    root.setLevel(lvl)
    for h in list(root.handlers):  # 清旧 handler，避免和别人设的叠加
        root.removeHandler(h)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _capture_stdlib() -> None:
    """把标准库 logging（FastAPI/uvicorn/transformers 都用它）桥接到我们的日志。"""
    if _HAS_LOGURU:
        class _Intercept(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    lvl = _loguru.level(record.levelname).name
                except Exception:
                    lvl = record.levelno
                # 用 record.name 当作组件名，保证第三方日志能看出来源
                _loguru.opt(depth=6, exception=record.exc_info).bind(
                    comp=record.name
                ).log(lvl, record.getMessage())

        logging.basicConfig(handlers=[_Intercept()], level=0, force=True)
    # 没装 loguru 时，_init_stdlib 已经配好 root，第三方库直接走 root，无需额外动作。


def _install_excepthook() -> None:
    if _HAS_LOGURU:
        def _hook(exc_type, exc_value, tb):
            _loguru.opt(depth=1, exception=(exc_type, exc_value, tb)).error(
                "未捕获异常（进程即将退出）"
            )
        sys.excepthook = _hook

        def _thread_hook(args):
            _loguru.opt(depth=1, exception=args.exc_value).error(
                "线程内未捕获异常: {}", getattr(args, "thread", None) and args.thread.name
            )
        threading.excepthook = _thread_hook
    else:
        def _hook(exc_type, exc_value, tb):
            logging.getLogger("root").error("未捕获异常", exc_info=(exc_type, exc_value, tb))
        sys.excepthook = _hook


def get_logger(name: Optional[str] = None):
    """拿一个 logger。loguru 下绑定组件名 comp，标准库下直接用同名 logger。"""
    if _HAS_LOGURU:
        return _loguru.bind(comp=name or "app")
    return logging.getLogger(name or "root")
