from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable

from .api import fetch_models, preflight_check
from .models import AuditConfig, ProbeResult
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
from .utils import filter_models, load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the AI Relay Audit terminal UI."
    )
    parser.add_argument("--tui", action="store_true", help="Start the full-screen terminal UI. This is the default.")
    return parser.parse_args()


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
        parse_args()
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        from .tui import run_tui

        return run_tui()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0
