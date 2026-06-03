from __future__ import annotations

import os
import re

from .models import ApiConfig, AuditConfig, ModelCostEstimate, ModelPricing, Probe, RunEstimate
from .probes import applicable_probes
from .api import reasoning_token_budget
from .scoring import is_reasoning_model


OFFICIAL_PRICING: tuple[tuple[re.Pattern[str], ModelPricing], ...] = tuple(
    (re.compile(pattern), pricing)
    for pattern, pricing in [
        # OpenAI: 只匹配价格明确的标准型号；pro/mini/nano 等变体不猜价。
        (r"(?:^|[/\s_-])gpt-?5\.3-codex(?:$|[/\s_-])", ModelPricing("GPT-5.3-Codex", 1.75, 14.0, "OpenAI")),
        (r"(?:^|[/\s_-])gpt-?5\.5(?:$|[/\s_-](?!pro|mini|nano))", ModelPricing("GPT-5.5", 5.0, 30.0, "OpenAI")),
        (r"(?:^|[/\s_-])gpt-?5\.4-mini(?:$|[/\s_-])", ModelPricing("GPT-5.4 mini", 0.75, 4.5, "OpenAI")),
        (r"(?:^|[/\s_-])gpt-?5\.4(?:$|[/\s_-](?!pro|mini|nano))", ModelPricing("GPT-5.4", 2.5, 15.0, "OpenAI")),
        (r"(?:^|[/\s_-])gpt-?5(?:$|[/\s_-](?!pro|mini|nano))", ModelPricing("GPT-5", 1.25, 10.0, "OpenAI")),
        (r"(?:^|[/\s_-])o3(?:$|[/\s_-](?!mini|pro|preview))", ModelPricing("o3", 2.0, 8.0, "OpenAI")),
        (r"(?:^|[/\s_-])o4-mini(?:$|[/\s_-])", ModelPricing("o4-mini", 1.1, 4.4, "OpenAI")),
        # Anthropic: Opus 4.5+ 与 Opus 4/4.1 价格不同，必须先匹配更具体的 4.5+。
        (r"claude-(?:opus-4[\.-][5-9]|4[\.-][5-9].*opus)", ModelPricing("Claude Opus 4.5+", 5.0, 25.0, "Anthropic")),
        (r"claude[_-](?:4[._-].*opus|opus[_-]4)", ModelPricing("Claude 4 Opus", 15.0, 75.0, "Anthropic")),
        (r"claude[_-](?:4[._-].*sonnet|sonnet[_-]4)", ModelPricing("Claude Sonnet 4/4.5", 3.0, 15.0, "Anthropic")),
        (r"claude.*haiku[_-]4", ModelPricing("Claude Haiku 4.5", 1.0, 5.0, "Anthropic")),
    ]
)


def pricing_for_model(model: str) -> ModelPricing | None:
    """按模型 ID 模糊匹配官方价格；未命中则不估算，避免误导。"""
    normalized = model.lower().replace("_", "-")
    for pattern, pricing in OFFICIAL_PRICING:
        if pattern.search(normalized):
            return pricing
    return None


def estimate_text_tokens(text: str) -> int:
    """粗略 token 估算：英文约 4 字符/token，中文按 2 字符/token 近似。"""
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, (ascii_chars + 3) // 4 + (non_ascii_chars + 1) // 2)


def estimate_probe_input_tokens(probe: Probe) -> int:
    # 给消息结构、角色名和 JSON 包装预留少量 overhead。
    return estimate_text_tokens(probe.system) + estimate_text_tokens(probe.user) + 16


def estimate_model_cost(model: str, probes: list[Probe], config: ApiConfig) -> ModelCostEstimate:
    input_tokens = sum(estimate_probe_input_tokens(probe) for probe in probes)
    output_budget = reasoning_token_budget(config) if is_reasoning_model(model) else config.max_tokens
    output_tokens = len(probes) * output_budget
    pricing = pricing_for_model(model)
    if pricing is None:
        return ModelCostEstimate(model, None, input_tokens, output_tokens, None, None, None)
    input_cost = input_tokens / 1_000_000 * pricing.input_per_million
    output_cost = output_tokens / 1_000_000 * pricing.output_per_million
    return ModelCostEstimate(model, pricing.label, input_tokens, output_tokens, input_cost, output_cost, pricing.source)


def format_usd(value: float) -> str:
    if value < 0.01:
        return f"${value:.4f}"
    if value < 1:
        return f"${value:.3f}"
    return f"${value:.2f}"


def build_run_estimate(cfg: AuditConfig, models: list[str]) -> RunEstimate:
    probes_by_model: dict[str, int] = {}
    cost_by_model: dict[str, ModelCostEstimate] = {}
    probe_requests = 0
    max_output_tokens = 0
    estimated_input_tokens = 0
    for model in models:
        probes = applicable_probes(model, cfg.all_targeted, cfg.probes_config, cfg.mode)
        probe_count = len(probes)
        probes_by_model[model] = probe_count
        probe_requests += probe_count
        cost = estimate_model_cost(model, probes, cfg.api)
        cost_by_model[model] = cost
        estimated_input_tokens += cost.input_tokens
        max_output_tokens += cost.output_tokens
    return RunEstimate(
        models=list(models),
        probes_by_model=probes_by_model,
        probe_requests=probe_requests,
        max_output_tokens=max_output_tokens,
        estimated_input_tokens=estimated_input_tokens,
        cost_by_model=cost_by_model,
    )


def format_run_estimate(estimate: RunEstimate, output_dir: str | None = None, save_report: bool | None = None, include_ping: bool = False) -> list[str]:
    request_count = estimate.probe_requests + (1 if include_ping else 0)
    lines = [
        f"Models: {estimate.model_count}",
        f"Probe requests: {estimate.probe_requests}" + (" (+1 TUI connectivity ping)" if include_ping else ""),
        f"Total API requests: {request_count}",
        f"Estimated input tokens: {estimate.estimated_input_tokens}",
        f"Max output token budget: {estimate.max_output_tokens}",
    ]
    if estimate.estimated_cost is None:
        lines.append("Estimated official cost: unknown (some models are not in the built-in official pricing table).")
    else:
        lines.append(f"Estimated official cost upper bound: {format_usd(estimate.estimated_cost)}")
    if estimate.models:
        per_model = ", ".join(f"{model}: {estimate.probes_by_model[model]}" for model in estimate.models)
        lines.append(f"Probes by model: {per_model}")
        cost_parts = []
        for model in estimate.models:
            item = estimate.cost_by_model[model]
            total = item.total_cost
            if total is None:
                cost_parts.append(f"{model}: unknown")
            else:
                cost_parts.append(f"{model}: {format_usd(total)} ({item.pricing_label})")
        lines.append("Cost by model: " + ", ".join(cost_parts))
    if output_dir is not None:
        lines.append(f"Output dir: {os.path.abspath(output_dir)}")
    if save_report is not None:
        lines.append(f"Auto-save report: {'yes' if save_report else 'no'}")
    return lines
