from __future__ import annotations

import argparse
import dataclasses
from typing import Any, Callable


DEFAULT_TIMEOUT = 90
DEFAULT_MAX_TOKENS = 900


@dataclasses.dataclass
class ApiConfig:
    base_url: str
    api_key: str
    timeout: int
    max_tokens: int
    temperature: float
    api_style: str = "auto"
    # auto 模式下，按模型缓存首个成功的调用协议，避免后续探针重复试错。
    resolved_styles: dict[str, str] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )


@dataclasses.dataclass
class AuditConfig:
    """单一运行配置源：封装一次审计所需的 ApiConfig 与全部运行选项。

    TUI 的 state（经 make_namespace_from_tui）汇聚到 from_namespace，避免在多处分别构造
    ApiConfig。
    """

    api: ApiConfig
    models: list[str]
    model_filter: str | None
    limit: int | None
    all_targeted: bool
    hide_prompts: bool
    output_dir: str
    save_report: bool
    baseline: str | None = None
    probes_config: str | None = None
    mode: str = "standard"  # quick, standard, full
    long_context: str = "off"  # off, 32k, 100k, 200k, max
    long_context_tokens: int | None = None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "AuditConfig":
        from .utils import normalize_base_url

        api = ApiConfig(
            base_url=normalize_base_url(getattr(args, "base_url", "") or ""),
            api_key=getattr(args, "api_key", "") or "",
            timeout=int(getattr(args, "timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT),
            max_tokens=int(getattr(args, "max_tokens", DEFAULT_MAX_TOKENS) or DEFAULT_MAX_TOKENS),
            temperature=float(getattr(args, "temperature", 0.0) or 0.0),
            api_style=getattr(args, "api_style", "auto") or "auto",
        )
        raw_models = getattr(args, "models", None)
        models = [item.strip() for item in raw_models.split(",") if item.strip()] if raw_models else []
        return cls(
            api=api,
            models=models,
            model_filter=getattr(args, "model_filter", None) or None,
            limit=getattr(args, "limit", None),
            all_targeted=bool(getattr(args, "all_targeted", False)),
            hide_prompts=bool(getattr(args, "hide_prompts", False)),
            output_dir=getattr(args, "output_dir", "reports") or "reports",
            save_report=bool(getattr(args, "save_report", True)),
            baseline=getattr(args, "baseline", None) or None,
            probes_config=getattr(args, "probes_config", None) or None,
            mode=getattr(args, "mode", "standard") or "standard",
            long_context=getattr(args, "long_context", "off") or "off",
            long_context_tokens=getattr(args, "long_context_tokens", None),
        )


@dataclasses.dataclass
class Probe:
    probe_id: str
    title: str
    category: str
    weight: int
    families: tuple[str, ...]
    system: str
    user: str
    scorer: Callable[[str], tuple[float, str]]
    scorer_id: str = ""
    mode: str = "standard"  # quick, standard, full


@dataclasses.dataclass
class ProbeResult:
    probe: Probe
    status: str
    score: float
    reason: str
    response: str
    latency_ms: int | None
    usage: dict[str, Any] | None
    error: str | None = None
    response_data: dict[str, Any] | None = None  # 完整响应体，用于 thinking signature 等高级检测


@dataclasses.dataclass
class SevereIssue:
    """严重问题记录。"""
    probe_id: str
    probe_title: str
    severity: str  # "critical", "high", "medium"
    score: float
    reason: str
    icon: str  # "🔴", "🟠", "🟡"


@dataclasses.dataclass(frozen=True)
class ModelPricing:
    label: str
    input_per_million: float
    output_per_million: float
    source: str


@dataclasses.dataclass
class ModelCostEstimate:
    model: str
    pricing_label: str | None
    input_tokens: int
    output_tokens: int
    input_cost: float | None
    output_cost: float | None
    source: str | None

    @property
    def total_cost(self) -> float | None:
        if self.input_cost is None or self.output_cost is None:
            return None
        return self.input_cost + self.output_cost


@dataclasses.dataclass
class RunEstimate:
    models: list[str]
    probes_by_model: dict[str, int]
    probe_requests: int
    max_output_tokens: int
    estimated_input_tokens: int
    cost_by_model: dict[str, ModelCostEstimate]

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def estimated_cost(self) -> float | None:
        costs = [estimate.total_cost for estimate in self.cost_by_model.values()]
        if not costs or any(cost is None for cost in costs):
            return None
        return sum(cost for cost in costs if cost is not None)


DecisionSummaryRow = dict[str, Any]
