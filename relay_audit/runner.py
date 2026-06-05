from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Callable

from .api import fetch_models, preflight_check
from .models import ApiConfig, AuditConfig, DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT, ProbeResult
from .pricing import build_run_estimate, format_run_estimate
from .probe_runner import run_probe
from .probes import applicable_probes, configured_probes
from .reporters import ConsoleReporter, Reporter
from .reporting import (
    authenticity_note,
    availability_rate,
    capability_score,
    compare_reports,
    detect_severe_issues,
    format_comparison_markdown,
    load_report_json,
    overall_rating,
    report_dict_from_results,
    write_reports,
)
from .scoring import family_for_model
from .utils import filter_models, load_dotenv, normalize_base_url


BASELINE_DIR = "data/baselines"
OFFICIAL_BASE_URLS = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
}
OFFICIAL_API_STYLES = {
    "openai": "auto",
    "anthropic": "anthropic",
    "gemini": "gemini",
}
PROVIDER_KEY_ENV = {
    "openai": ("OPENAI_API_KEY", "AI_RELAY_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "AI_RELAY_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "AI_RELAY_API_KEY"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Relay Audit terminal UI and baseline utilities."
    )
    parser.add_argument("--tui", action="store_true", help="Start the full-screen terminal UI. This is the default.")
    subparsers = parser.add_subparsers(dest="command")

    baseline = subparsers.add_parser("baseline", help="Generate or compare official full-mode baselines.")
    baseline_subparsers = baseline.add_subparsers(dest="baseline_command", required=True)

    generate = baseline_subparsers.add_parser("generate", help="Run full mode against an official provider and save a fixed baseline JSON.")
    generate.add_argument("--provider", required=True, choices=sorted(OFFICIAL_BASE_URLS), help="Official provider used for the baseline.")
    generate.add_argument("--model", required=True, help="Model ID to baseline.")
    generate.add_argument("--base-url", help="Override the official provider base URL.")
    generate.add_argument("--api-key", help="Official provider API key. Defaults to provider-specific env vars.")
    generate.add_argument("--api-style", choices=["auto", "openai-chat", "openai-responses", "anthropic", "gemini"], help="Override provider default API style.")
    generate.add_argument("--output-dir", default=BASELINE_DIR, help=f"Baseline output directory. Default: {BASELINE_DIR}.")
    generate.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Request timeout seconds. Default: {DEFAULT_TIMEOUT}.")
    generate.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max output tokens per probe. Default: {DEFAULT_MAX_TOKENS}.")
    generate.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Default: 0.")
    generate.add_argument("--all-targeted", action="store_true", help="Run all targeted probes instead of model-family targeted probes.")
    generate.add_argument("--hide-prompts", action="store_true", help="Hide prompts in console output.")

    compare = baseline_subparsers.add_parser("compare", help="Compare a relay report JSON against an official baseline JSON.")
    compare.add_argument("--current", required=True, help="Current relay audit report JSON.")
    compare.add_argument("--baseline", help="Baseline JSON. If omitted, derive from --model and --baseline-dir.")
    compare.add_argument("--model", help="Model ID used to derive the baseline path when --baseline is omitted.")
    compare.add_argument("--baseline-dir", default=BASELINE_DIR, help=f"Baseline directory. Default: {BASELINE_DIR}.")
    compare.add_argument("--mode", default="full", help="Baseline mode suffix. Default: full.")
    compare.add_argument("--output", help="Optional Markdown output path for the comparison.")
    return parser.parse_args()


def sanitize_baseline_model_name(model: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip()).strip("._-")
    return sanitized or "model"


def baseline_path_for_model(model: str, output_dir: str = BASELINE_DIR, mode: str = "full") -> str:
    return os.path.join(output_dir, f"{sanitize_baseline_model_name(model)}_{mode}.json")


def provider_api_key(provider: str, explicit_key: str | None = None) -> str:
    if explicit_key:
        return explicit_key
    for env_name in PROVIDER_KEY_ENV.get(provider, ()):
        value = os.getenv(env_name)
        if value:
            return value
    names = ", ".join(PROVIDER_KEY_ENV.get(provider, ()))
    raise ValueError(f"Missing API key for {provider}; pass --api-key or set one of: {names}")


def baseline_config_from_args(args: argparse.Namespace) -> AuditConfig:
    provider = str(args.provider)
    base_url = normalize_base_url(args.base_url or OFFICIAL_BASE_URLS[provider])
    api_style = args.api_style or OFFICIAL_API_STYLES[provider]
    api = ApiConfig(
        base_url=base_url,
        api_key=provider_api_key(provider, args.api_key),
        timeout=int(args.timeout),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        api_style=api_style,
    )
    return AuditConfig(
        api=api,
        models=[str(args.model)],
        model_filter=None,
        limit=None,
        all_targeted=bool(args.all_targeted),
        hide_prompts=bool(args.hide_prompts),
        output_dir=str(args.output_dir),
        save_report=False,
        baseline=None,
        probes_config=None,
        mode="full",
    )


def write_json_atomic(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def build_baseline_report(
    model_results: dict[str, list[ProbeResult]],
    config: ApiConfig,
    provider: str,
    mode: str = "full",
) -> dict[str, Any]:
    data = report_dict_from_results(model_results, config)
    data["baseline"] = {
        "provider": provider,
        "mode": mode,
        "api_style": config.api_style,
        "source": "official-api",
    }
    return data


def run_baseline_generate(args: argparse.Namespace) -> int:
    cfg = baseline_config_from_args(args)
    reporter = ConsoleReporter()
    model_results, _paths = run_audit(cfg, reporter)
    report = build_baseline_report(model_results, cfg.api, str(args.provider), "full")
    path = baseline_path_for_model(str(args.model), str(args.output_dir), "full")
    write_json_atomic(path, report)
    print(f"Baseline JSON: {path}")
    return 0


def run_baseline_compare(args: argparse.Namespace) -> int:
    baseline_path = args.baseline
    if not baseline_path:
        if not args.model:
            raise ValueError("baseline compare requires --baseline or --model")
        baseline_path = baseline_path_for_model(str(args.model), str(args.baseline_dir), str(args.mode))
    baseline_report = load_report_json(str(baseline_path))
    current_report = load_report_json(str(args.current))
    comparison = compare_reports(baseline_report, current_report)
    markdown = format_comparison_markdown(comparison)
    if args.output:
        os.makedirs(os.path.dirname(str(args.output)) or ".", exist_ok=True)
        with open(str(args.output), "w", encoding="utf-8") as file:
            file.write(markdown)
            file.write("\n")
    print(markdown)
    return 0


def run_baseline_command(args: argparse.Namespace) -> int:
    if args.baseline_command == "generate":
        return run_baseline_generate(args)
    if args.baseline_command == "compare":
        return run_baseline_compare(args)
    raise ValueError(f"Unsupported baseline command: {args.baseline_command}")


def run_audit(
    cfg: AuditConfig,
    reporter: Reporter,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[dict[str, list[ProbeResult]], tuple[str | None, str | None]]:
    if not cfg.api.base_url:
        raise ValueError("Missing Base URL.")
    if not cfg.api.api_key:
        raise ValueError("Missing API key.")
    if cfg.probes_config:
        configured_probes(cfg.probes_config)
    config = cfg.api

    def cancelled() -> bool:
        return should_cancel is not None and should_cancel()

    if cfg.models:
        models = list(cfg.models)
    else:
        reporter.section("获取模型列表 /v1/models")
        models = fetch_models(config)
        reporter.info(f"发现 {len(models)} 个模型")

    models = filter_models(models, cfg.model_filter, cfg.limit)
    if not models:
        raise ValueError("No models to audit.")

    # 预检测：快速验证每个模型是否可用
    reporter.section("预检测模型可用性")
    available_models: list[str] = []
    unavailable_models: list[tuple[str, str]] = []

    for model in models:
        is_available, error_msg = preflight_check(config, model)
        if is_available:
            available_models.append(model)
            reporter.info(f"✓ {model}: 可用")
        else:
            unavailable_models.append((model, error_msg))
            reporter.info(f"✗ {model}: {error_msg}")

    if not available_models:
        error_details = "\n".join(f"  - {model}: {msg}" for model, msg in unavailable_models)
        raise ValueError(f"所有模型预检测均失败:\n{error_details}\n\n建议: 检查 API key、模型名称、base URL 和网络连接。")

    if unavailable_models:
        reporter.info(f"\n跳过 {len(unavailable_models)} 个不可用模型，继续检测 {len(available_models)} 个可用模型。")

    models = available_models

    estimate = build_run_estimate(cfg, models)
    reporter.section("检测配置")
    reporter.info(f"Base URL: {config.base_url}")
    reporter.info(f"Models: {', '.join(models)}")
    reporter.info(f"Output dir: {os.path.abspath(cfg.output_dir)}")
    reporter.info("说明: 真假检测为黑盒一致性评估，不是供应商级证明。")
    reporter.section("运行前确认")
    for line in format_run_estimate(estimate, cfg.output_dir, cfg.save_report):
        reporter.info(line)

    was_cancelled = False
    model_results: dict[str, list[ProbeResult]] = {}
    for model in models:
        if cancelled():
            was_cancelled = True
            break
        reporter.section(f"开始检测模型: {model} | family={family_for_model(model)}")
        probes = applicable_probes(model, cfg.all_targeted, cfg.probes_config, cfg.mode)
        total = len(probes)
        results: list[ProbeResult] = []
        for index, probe in enumerate(probes, 1):
            if cancelled():
                was_cancelled = True
                break
            results.append(
                run_probe(config, model, probe, reporter, not cfg.hide_prompts, index, total)
            )
            reporter.progress(index, total)
        model_results[model] = results
        cap = capability_score(results)
        avail = availability_rate(results)
        auth, auth_reason = authenticity_note(model, results)
        severe_issues = detect_severe_issues(model, results)
        reporter.model_done(model, cap, avail, overall_rating(cap, avail, severe_issues), auth, auth_reason)
        if was_cancelled:
            break

    if was_cancelled:
        reporter.section("已取消")
        reporter.info("检测被用户取消，以下仅为已完成部分。")

    comparison = None
    if cfg.baseline and model_results:
        reporter.section("Baseline 对比")
        baseline_report = load_report_json(cfg.baseline)
        current_report = report_dict_from_results(model_results, config)
        comparison = compare_reports(baseline_report, current_report)
        reporter.info(format_comparison_markdown(comparison))

    md_path = None
    json_path = None
    if cfg.save_report and model_results:
        reporter.section("生成报告")
        md_path, json_path = write_reports(cfg.output_dir, model_results, config, comparison)
        reporter.info(f"Markdown: {md_path}")
        reporter.info(f"JSON: {json_path}")
    return model_results, (md_path, json_path)


def main() -> int:
    load_dotenv()
    try:
        args = parse_args()
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        if getattr(args, "command", None) == "baseline":
            return run_baseline_command(args)
        from .tui import run_tui

        return run_tui()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0
