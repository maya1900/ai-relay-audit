from __future__ import annotations

import os
import re

from .models import ApiConfig


def normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if url.endswith("/v1"):
        return url[:-3]
    return url


def redact_secrets(text: str, config: ApiConfig) -> str:
    """从面向用户/报告的文本中移除 API key，避免服务端回显导致泄露。"""
    if config.api_key and config.api_key in text:
        text = text.replace(config.api_key, "***REDACTED***")
    return text


def _strip_env_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or value[index - 1].isspace():
                return value[:index].rstrip()
    return value.strip()


def parse_env_lines(lines: list[str]) -> dict[str, str]:
    """解析项目本地 .env 的简单 KEY=value 语法，不执行 shell 展开。"""
    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = _strip_env_comment(value.strip())
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_dotenv(path: str = ".env", override: bool = False) -> dict[str, str]:
    """加载 .env 到 os.environ；默认不覆盖已有环境变量。"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as file:
        parsed = parse_env_lines(file.readlines())
    loaded: dict[str, str] = {}
    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def filter_models(models: list[str], pattern: str | None, limit: int | None) -> list[str]:
    if limit is not None and limit < 0:
        raise ValueError("--limit must be >= 0")
    filtered = list(models)
    if pattern:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Invalid model filter regex: {exc}") from exc
        filtered = [model for model in filtered if regex.search(model)]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered
