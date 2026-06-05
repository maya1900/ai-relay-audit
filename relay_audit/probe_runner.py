from __future__ import annotations

from typing import Any

from .api import (
    chat,
    chat_anthropic_full,
    chat_anthropic_pdf,
    chat_anthropic_tool_use,
    chat_anthropic_with_thinking,
    chat_openai,
    chat_openai_json_schema,
    chat_openai_tool_call,
    normalize_api_style,
    stream_anthropic_events,
)
from .models import ApiConfig, Probe, ProbeResult
from .reporters import Reporter
from .scoring import family_for_model


CLAUDE_PROTOCOL_PROBES = {
    "claude_message_protocol",
    "claude_tool_use",
    "claude_pdf",
    "claude_sse_protocol",
}

OPENAI_PROTOCOL_PROBES = {
    "openai_tool_calls",
    "openai_json_schema_protocol",
}

RESPONSE_DATA_SCORERS = CLAUDE_PROTOCOL_PROBES | OPENAI_PROTOCOL_PROBES


def _skipped_result(probe: Probe, family: str, expected: str) -> ProbeResult:
    return ProbeResult(
        probe,
        "skipped",
        0,
        f"该探针仅适用于 {expected} 模型，当前模型家族为 {family}",
        "",
        None,
        None,
        None,
        None,
    )


def run_probe(
    config: ApiConfig,
    model: str,
    probe: Probe,
    reporter: Reporter,
    show_prompts: bool,
    index: int,
    total: int,
) -> ProbeResult:
    reporter.probe_start(probe, index, total, show_prompts)
    try:
        response_data: dict[str, Any] | None = None
        stream_response: str | None = None
        non_stream_response: str | None = None
        family = family_for_model(model)
        actual_api_style = normalize_api_style(config.api_style)

        # 特殊处理 thinking signature 探针：需要调用带 thinking 参数的 Anthropic API
        if probe.probe_id == "claude_thinking_signature":
            if family != "claude":
                # 非 Claude 模型跳过此探针
                result = _skipped_result(probe, family, "Claude")
                reporter.probe_result(result, show_prompts)
                return result

            # 对 Claude 模型使用 extended thinking
            response, usage, latency_ms, response_data = chat_anthropic_with_thinking(
                config, model, probe.system, probe.user, thinking="enabled"
            )
            actual_api_style = "anthropic"
        elif probe.probe_id in CLAUDE_PROTOCOL_PROBES:
            if family != "claude":
                result = _skipped_result(probe, family, "Claude")
                reporter.probe_result(result, show_prompts)
                return result
            actual_api_style = "anthropic"
            if probe.probe_id == "claude_message_protocol":
                response, usage, latency_ms, response_data = chat_anthropic_full(
                    config,
                    model,
                    probe.system,
                    [{"role": "user", "content": probe.user}],
                    {"max_tokens": min(config.max_tokens, 128)},
                )
            elif probe.probe_id == "claude_tool_use":
                response, usage, latency_ms, response_data = chat_anthropic_tool_use(config, model)
            elif probe.probe_id == "claude_pdf":
                response, usage, latency_ms, response_data = chat_anthropic_pdf(config, model)
            else:
                response, usage, latency_ms, response_data = stream_anthropic_events(
                    config, model, probe.system, probe.user
                )
        elif probe.probe_id in OPENAI_PROTOCOL_PROBES:
            if family != "gpt":
                result = _skipped_result(probe, family, "OpenAI/GPT")
                reporter.probe_result(result, show_prompts)
                return result
            actual_api_style = "openai-chat"
            if probe.probe_id == "openai_tool_calls":
                response, usage, latency_ms, response_data = chat_openai_tool_call(config, model)
            else:
                response, usage, latency_ms, response_data = chat_openai_json_schema(config, model)
        elif probe.probe_id == "stream_consistency":
            # 特殊处理 stream 一致性探针：需要分别调用 stream 和 non-stream
            # 只对 OpenAI 兼容接口测试（Anthropic 和 Gemini 的 streaming 格式不同）
            stream_response = None
            non_stream_response = None

            # 先尝试 non-stream 请求
            try:
                non_stream_response, usage, latency_ms = chat(config, model, probe.system, probe.user)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Non-stream request failed: {exc}") from exc

            # 再尝试 stream 请求（仅对 OpenAI 风格）
            resolved_style = normalize_api_style(config.api_style)
            if resolved_style in ("auto", "openai-chat"):
                try:
                    stream_response, _, _ = chat_openai(config, model, probe.system, probe.user, stream=True)
                except Exception as exc:  # noqa: BLE001
                    # Stream 失败时记录错误，但不阻断整个探针
                    # 评分器会处理 stream_response=None 的情况
                    pass

            # 将 non_stream_response 作为主要响应用于显示
            response = non_stream_response
        else:
            # 常规探针使用标准 chat 函数
            response, usage, latency_ms = chat(config, model, probe.system, probe.user)
            if actual_api_style == "auto":
                actual_api_style = config.resolved_styles.get(model, "auto")

        # 调用评分器
        if probe.probe_id == "claude_thinking_signature":
            score, reason = probe.scorer(response, response_data)
        elif probe.probe_id in RESPONSE_DATA_SCORERS:
            if probe.probe_id == "openai_json_schema_protocol":
                score, reason = probe.scorer(response, response_data, usage)
            else:
                score, reason = probe.scorer(response, response_data)
        elif probe.probe_id == "protocol_fingerprint":
            score, reason = probe.scorer(response, usage, family, actual_api_style)
        elif probe.probe_id == "stream_consistency":
            score, reason = probe.scorer(response, stream_response, non_stream_response)
        else:
            score, reason = probe.scorer(response)

        result = ProbeResult(probe, "ok", score, reason, response, latency_ms, usage, None, response_data)
    except Exception as exc:  # noqa: BLE001 - audit should continue per model.
        error = str(exc)
        result = ProbeResult(probe, "error", 0, f"接口请求失败：{error}", "", None, None, error, None)
    reporter.probe_result(result, show_prompts)
    return result
