from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .models import ApiConfig
from .scoring import family_for_model, is_reasoning_model


RETRYABLE_STATUS = {429, 500, 502, 503, 504}
ANTHROPIC_THINKING_BUDGET_TOKENS = 2048
ANTHROPIC_THINKING_MAX_TOKENS = 4096
ANTHROPIC_ADAPTIVE_THINKING_MAX_TOKENS = 8192


def build_api_url(config: ApiConfig, path: str) -> str:
    """Build a request URL, preserving explicit /v1 or /v1beta paths."""
    if path.startswith(("http://", "https://")):
        return path
    normalized_path = path if path.startswith("/") else f"/{path}"
    has_explicit_version = normalized_path.startswith(("/v1/", "/v1?", "/v1beta/", "/v1beta?"))
    prefix = "" if has_explicit_version else "/v1"
    return f"{config.base_url}{prefix}{normalized_path}"


def api_request(
    config: ApiConfig,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    retries: int = 2,
) -> dict[str, Any]:
    url = build_api_url(config, path)
    data = None
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            body = redact_secrets(exc.read().decode("utf-8", errors="replace"), config)
            if exc.code in RETRYABLE_STATUS and attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {exc.code}: {body[:1200]}") from exc
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise RuntimeError(f"Network error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON response: {exc}") from exc
    raise RuntimeError("Request failed after retries")


def fetch_models(config: ApiConfig) -> list[str]:
    data = api_request(config, "GET", "/models")
    models = []
    for item in data.get("data", []):
        model_id = item.get("id")
        if isinstance(model_id, str):
            models.append(model_id)
    return sorted(set(models))


def preflight_check(config: ApiConfig, model: str) -> tuple[bool, str]:
    """预检测：快速验证模型是否可用。

    返回: (is_available, message)
    - is_available: 模型是否可用
    - message: 可用时为空，不可用时为错误信息
    """
    try:
        # 使用极简的请求，超时设为 5 秒
        temp_config = ApiConfig(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=5,
            max_tokens=10,
            temperature=0,
            api_style=config.api_style,
        )

        # 极简问题，只需要模型能响应即可
        system = "Reply with just 'ok'."
        user = "ok?"

        try:
            response, usage, latency_ms = chat(temp_config, model, system, user)
            if response and len(response.strip()) > 0:
                return True, ""
            return False, "模型返回空响应"
        except Exception as exc:
            error = str(exc)
            # 判断是否是模型不存在的错误
            if any(keyword in error.lower() for keyword in ["model", "not found", "does not exist", "invalid", "unknown"]):
                return False, f"模型不存在或不可用: {error[:200]}"
            # 其他错误（网络、权限等）也标记为不可用
            return False, f"预检测失败: {error[:200]}"

    except Exception as exc:
        return False, f"预检测异常: {str(exc)[:200]}"


def reasoning_token_budget(config: ApiConfig) -> int:
    # 推理模型会先消耗大量隐藏推理 token，预算过小会导致可见输出为空。
    return max(config.max_tokens, 2048)


def chat_openai(config: ApiConfig, model: str, system: str, user: str, stream: bool = False) -> tuple[str, dict[str, Any] | None, int]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": stream,
    }
    if is_reasoning_model(model):
        # 推理模型只接受默认 temperature，且改用 max_completion_tokens。
        payload["max_completion_tokens"] = reasoning_token_budget(config)
    else:
        payload["temperature"] = config.temperature
        payload["max_tokens"] = config.max_tokens
    started = time.perf_counter()

    if stream:
        # Streaming mode: collect chunks
        url = build_api_url(config, "/chat/completions")
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")

        chunks: list[str] = []
        usage: dict[str, Any] | None = None

        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            for line in response:
                line_text = line.decode("utf-8", errors="replace").strip()
                if not line_text or not line_text.startswith("data: "):
                    continue
                line_text = line_text[6:]  # Remove "data: " prefix
                if line_text == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(line_text)
                    choices = chunk_data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            chunks.append(content)
                    # Try to get usage from the final chunk
                    if "usage" in chunk_data:
                        usage = chunk_data["usage"]
                except json.JSONDecodeError:
                    continue

        latency_ms = int((time.perf_counter() - started) * 1000)
        text = "".join(chunks)
        return text, usage, latency_ms
    else:
        # Non-streaming mode (original implementation)
        data = api_request(config, "POST", "/chat/completions", payload)
        latency_ms = int((time.perf_counter() - started) * 1000)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices in response: {json.dumps(data, ensure_ascii=False)[:1000]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
        if content is None:
            content = ""
        text = str(content)
        return text, data.get("usage"), latency_ms


def extract_responses_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
            elif isinstance(content.get("output_text"), str):
                chunks.append(content["output_text"])
    return "".join(chunks)


def chat_openai_responses(
    config: ApiConfig,
    model: str,
    system: str,
    user: str,
    extra_headers: dict[str, str] | None = None,
) -> tuple[str, dict[str, Any] | None, int]:
    reasoning = is_reasoning_model(model)
    payload: dict[str, Any] = {
        "model": model,
        "instructions": system,
        "input": user,
        "max_output_tokens": reasoning_token_budget(config) if reasoning else config.max_tokens,
    }
    if not reasoning:
        payload["temperature"] = config.temperature
    started = time.perf_counter()
    data = api_request(config, "POST", "/responses", payload, extra_headers=extra_headers)
    latency_ms = int((time.perf_counter() - started) * 1000)
    text = extract_responses_text(data)
    if not text:
        raise RuntimeError(f"No text in responses payload: {json.dumps(data, ensure_ascii=False)[:1000]}")
    return text, data.get("usage"), latency_ms


def chat_anthropic(
    config: ApiConfig,
    model: str,
    system: str,
    user: str,
    extra_headers: dict[str, str] | None = None,
    thinking: str | dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None, int, dict[str, Any] | None]:
    """调用 Anthropic Messages API。

    返回: (text, usage, latency_ms, full_response)
    - full_response 包含完整的响应体，用于提取 thinking signature 等字段。
    """
    payload: dict[str, Any] = {
        "model": model,
        "system": system,
        "max_tokens": _anthropic_max_tokens(config, model, thinking),
        "messages": [{"role": "user", "content": user}],
    }
    if _anthropic_supports_temperature(model):
        payload["temperature"] = config.temperature
    if thinking:
        payload["thinking"] = _anthropic_thinking_payload(model, thinking)
    started = time.perf_counter()
    # 同时带 x-api-key（Anthropic 官方）与 Authorization: Bearer（多数中转站），
    # 以最大化兼容性；官方端点会忽略多余的 Authorization 头。
    headers = {
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
    }
    if extra_headers:
        headers.update(extra_headers)
    data = api_request(config, "POST", "/messages", payload, extra_headers=headers)
    latency_ms = int((time.perf_counter() - started) * 1000)
    content = data.get("content")
    if isinstance(content, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    else:
        text = str(content or "")
    usage = data.get("usage")
    return text, usage, latency_ms, data


def chat_gemini(config: ApiConfig, model: str, system: str, user: str) -> tuple[str, dict[str, Any] | None, int]:
    """调用 Google Gemini API。

    Gemini API 使用不同的格式：
    - Endpoint: /v1beta/models/{model}:generateContent 或 /v1/models/{model}:generateContent
    - 需要在 URL 参数中传递 key（?key=xxx）
    - system instruction 在 systemInstruction 字段中
    - 返回格式与 OpenAI 不同
    """
    # Gemini API key 通常通过 URL 参数传递
    payload: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user}]
            }
        ],
        "generationConfig": {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens,
        }
    }

    # 添加 system instruction（如果支持）
    if system:
        payload["systemInstruction"] = {
            "parts": [{"text": system}]
        }

    started = time.perf_counter()

    # Gemini 可能使用 v1 或 v1beta，尝试两种路径
    # API key 通过 URL 参数传递
    endpoint = f"/v1/models/{model}:generateContent"

    # 对于 Gemini，API key 通常在 URL 参数中
    # 我们需要修改 api_request 的调用方式
    # 先尝试标准方式（某些中转可能仍用 Authorization header）
    try:
        # 尝试在 URL 中传递 key
        data = api_request(config, "POST", f"{endpoint}?key={config.api_key}", payload, extra_headers={"Content-Type": "application/json"})
    except Exception:
        # 如果失败，尝试 v1beta
        endpoint = f"/v1beta/models/{model}:generateContent"
        data = api_request(config, "POST", f"{endpoint}?key={config.api_key}", payload, extra_headers={"Content-Type": "application/json"})

    latency_ms = int((time.perf_counter() - started) * 1000)

    # 解析 Gemini 响应格式
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"No candidates in Gemini response: {json.dumps(data, ensure_ascii=False)[:1000]}")

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    if not parts:
        raise RuntimeError(f"No parts in Gemini response: {json.dumps(data, ensure_ascii=False)[:1000]}")

    # 拼接所有 text parts
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))

    # Gemini usage 格式不同
    usage_metadata = data.get("usageMetadata", {})
    usage = None
    if usage_metadata:
        usage = {
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0),
        }

    return text, usage, latency_ms


# 需要特殊处理的探针（如需要 thinking signature）会直接调用 chat_anthropic_with_thinking
STYLE_CALLERS: dict[str, Callable[[ApiConfig, str, str, str], tuple[str, dict[str, Any] | None, int]]] = {
    "anthropic": lambda cfg, m, s, u: chat_anthropic(cfg, m, s, u)[:3],
    "openai-chat": chat_openai,
    "openai-responses": chat_openai_responses,
    "gemini": chat_gemini,
}


def chat_anthropic_with_thinking(
    config: ApiConfig, model: str, system: str, user: str, thinking: str | dict[str, Any] = "enabled"
) -> tuple[str, dict[str, Any] | None, int, dict[str, Any] | None]:
    """专门用于需要 thinking 的探针。"""
    return chat_anthropic(config, model, system, user, thinking=thinking)


def _normalized_model(model: str) -> str:
    return model.lower().replace("_", "-").replace(".", "-")


def _anthropic_adaptive_thinking_only(model: str) -> bool:
    return "opus-4-7" in _normalized_model(model)


def _anthropic_supports_temperature(model: str) -> bool:
    return not _anthropic_adaptive_thinking_only(model)


def _anthropic_thinking_payload(model: str, thinking: str | dict[str, Any]) -> dict[str, Any]:
    if _anthropic_adaptive_thinking_only(model):
        return {"type": "adaptive", "display": "summarized"}
    if isinstance(thinking, dict):
        return dict(thinking)
    if thinking == "adaptive":
        return {"type": "adaptive", "display": "summarized"}
    return {"type": thinking, "budget_tokens": ANTHROPIC_THINKING_BUDGET_TOKENS}


def _anthropic_max_tokens(config: ApiConfig, model: str, thinking: str | dict[str, Any] | None) -> int:
    if not thinking:
        return config.max_tokens
    payload = _anthropic_thinking_payload(model, thinking)
    if payload.get("type") == "adaptive":
        return max(config.max_tokens, ANTHROPIC_ADAPTIVE_THINKING_MAX_TOKENS)
    budget = payload.get("budget_tokens")
    budget_tokens = budget if isinstance(budget, int) and not isinstance(budget, bool) else ANTHROPIC_THINKING_BUDGET_TOKENS
    return max(config.max_tokens, budget_tokens + 1024, ANTHROPIC_THINKING_MAX_TOKENS)


def normalize_api_style(style: str) -> str:
    aliases = {
        "openai": "openai-chat",
        "responses": "openai-responses",
        "google": "gemini",
    }
    normalized = (style or "auto").lower()
    return aliases.get(normalized, normalized)


def chat(config: ApiConfig, model: str, system: str, user: str) -> tuple[str, dict[str, Any] | None, int]:
    style = normalize_api_style(config.api_style)
    if style in STYLE_CALLERS:
        return STYLE_CALLERS[style](config, model, system, user)
    if style != "auto":
        raise ValueError(f"Unsupported api_style: {config.api_style}")

    # auto：若该模型已解析出可用协议，直接复用，避免每个探针重复试错。
    cached = config.resolved_styles.get(model)
    if cached and cached in STYLE_CALLERS:
        try:
            return STYLE_CALLERS[cached](config, model, system, user)
        except Exception:  # noqa: BLE001 - 缓存协议失效则回退到完整探测。
            config.resolved_styles.pop(model, None)

    family = family_for_model(model)
    if family == "claude":
        order = ["anthropic", "openai-chat", "openai-responses"]
    elif family == "gpt":
        order = ["openai-responses", "openai-chat"]
    elif family == "gemini":
        order = ["gemini", "openai-chat", "openai-responses"]
    else:
        order = ["openai-responses", "openai-chat", "anthropic", "gemini"]
    errors = []
    for name in order:
        try:
            result = STYLE_CALLERS[name](config, model, system, user)
            config.resolved_styles[model] = name
            return result
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    raise RuntimeError("auto failed: " + "; ".join(errors))
