from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import statistics
from typing import Any

from .models import ApiConfig, DecisionSummaryRow, ProbeResult, SevereIssue
from .scoring import family_for_model


PROTOCOL_PROBE_IDS = {
    "protocol_fingerprint",
    "claude_thinking_signature",
    "claude_message_protocol",
    "claude_tool_use",
    "claude_pdf",
    "claude_sse_protocol",
    "openai_tool_calls",
    "openai_json_schema_protocol",
}


def indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def weighted_score(results: list[ProbeResult]) -> float:
    total_weight = sum(result.probe.weight for result in results)
    if total_weight == 0:
        return 0
    return sum(result.score * result.probe.weight for result in results) / total_weight


def capability_score(results: list[ProbeResult]) -> float | None:
    """仅基于成功探针的加权能力分；没有任何探针成功时返回 None。

    这样接口不可用（请求失败）就不会被算作 0 分而拖垮能力评估——
    失败由单独的可用性指标承担。
    """
    ok = [result for result in results if result.status == "ok"]
    if not ok:
        return None
    return weighted_score(ok)


def availability_rate(results: list[ProbeResult]) -> float:
    """成功返回可用响应的探针请求占比（百分数）。"""
    total = len(results)
    if total == 0:
        return 0.0
    ok = sum(1 for result in results if result.status == "ok")
    return ok / total * 100


def availability_summary(results: list[ProbeResult]) -> tuple[int, int, str]:
    total = len(results)
    errors = sum(1 for result in results if result.status == "error")
    if total == 0:
        return 0, 0, "无检测结果"
    if errors == total:
        return errors, total, "接口不可用：所有检测请求均失败，无法评估模型能力或真实性。"
    if errors:
        return errors, total, f"接口不稳定：{errors}/{total} 个检测请求失败；能力分仅基于成功探针，可用性单独给出。"
    return errors, total, "接口可用：所有检测请求均完成。"


def detect_severe_issues(model: str, results: list[ProbeResult]) -> list[SevereIssue]:
    """检测严重问题并分类。

    返回按严重性排序的问题列表：critical > high > medium
    """
    issues: list[SevereIssue] = []
    family = family_for_model(model)

    for result in results:
        if result.status != "ok":
            continue

        # 1. 协议指纹异常 - Critical
        if result.probe.probe_id == "protocol_fingerprint" and result.score < 30:
            issues.append(SevereIssue(
                probe_id=result.probe.probe_id,
                probe_title=result.probe.title,
                severity="critical",
                score=result.score,
                reason=result.reason,
                icon="🔴"
            ))

        # 2. Thinking signature 缺失（Claude 模型）- Critical
        elif result.probe.probe_id == "claude_thinking_signature" and result.score < 50:
            issues.append(SevereIssue(
                probe_id=result.probe.probe_id,
                probe_title=result.probe.title,
                severity="critical",
                score=result.score,
                reason=result.reason,
                icon="🔴"
            ))

        # 3. Stream 严重不一致 - High
        elif result.probe.probe_id == "stream_consistency" and result.score < 50:
            # 只有在两者都存在但不一致时才标记为严重（50 分表示一方缺失）
            if "严重不一致" in result.reason or "篡改" in result.reason:
                issues.append(SevereIssue(
                    probe_id=result.probe.probe_id,
                    probe_title=result.probe.title,
                    severity="high",
                    score=result.score,
                    reason=result.reason,
                    icon="🟠"
                ))

        # 4. 协议级 detector 偏差 - High/Critical
        elif result.probe.probe_id in PROTOCOL_PROBE_IDS and result.score < 60:
            severity = "critical" if result.score < 30 else "high"
            icon = "🔴" if severity == "critical" else "🟠"
            issues.append(SevereIssue(
                probe_id=result.probe.probe_id,
                probe_title=result.probe.title,
                severity=severity,
                score=result.score,
                reason=result.reason,
                icon=icon,
            ))

        # 4. 身份混乱 - Medium
        elif result.probe.probe_id == "identity_limits" and result.score < 40:
            issues.append(SevereIssue(
                probe_id=result.probe.probe_id,
                probe_title=result.probe.title,
                severity="medium",
                score=result.score,
                reason=result.reason,
                icon="🟡"
            ))

        # 5. 高权重探针极低分 - Medium
        elif result.probe.weight >= 15 and result.score < 30:
            issues.append(SevereIssue(
                probe_id=result.probe.probe_id,
                probe_title=result.probe.title,
                severity="medium",
                score=result.score,
                reason=result.reason,
                icon="🟡"
            ))

    # 按严重性排序：critical > high > medium，同级按分数升序
    severity_order = {"critical": 0, "high": 1, "medium": 2}
    issues.sort(key=lambda x: (severity_order[x.severity], x.score))

    return issues


def reliability_band(score: float) -> str:
    if score >= 90:
        return "A 优秀：能力强，接口行为高度稳定"
    if score >= 80:
        return "B 良好：可用于多数正式任务，建议保留抽检"
    if score >= 65:
        return "C 可用：存在明显短板或不稳定，关键任务需谨慎"
    if score >= 50:
        return "D 风险：仅建议低风险场景试用"
    return "E 不建议：能力或接口可信度不足"


def overall_rating(capability: float | None, availability: float, severe_issues: list[SevereIssue] | None = None) -> str:
    """把能力分映射为 A–E，并按可用性和严重问题降级。

    可用性不足说明接口/路由不稳定，此时即便成功探针表现优异也不应给出高评级：
    - 无成功探针：无法评分。
    - 可用性 100%：能力分直接决定评级（A–E 全段）。
    - 可用性 <100%：封顶到 C（不允许 A/B），低于 60% 追加更强提示。

    严重问题降级：
    - Critical 问题：降级 1-2 级（最多到 C）
    - 2+ High 问题：降级 1 级
    - 3+ Medium 问题：追加标注
    """
    if capability is None:
        return "N/A 接口不可用：无法评分"

    # 基础评级（考虑可用性）
    if availability >= 100:
        base_rating = reliability_band(capability)
    else:
        capped = reliability_band(min(capability, 79.0))
        note = "（接口不稳定，评级已限级）" if availability >= 60 else "（接口高度不稳定，评级仅供参考）"
        base_rating = capped + note

    # 严重问题降级
    if not severe_issues:
        return base_rating

    critical_count = sum(1 for issue in severe_issues if issue.severity == "critical")
    high_count = sum(1 for issue in severe_issues if issue.severity == "high")
    medium_count = sum(1 for issue in severe_issues if issue.severity == "medium")

    # 提取基础等级字母（A-E）
    base_letter = base_rating[0] if base_rating and base_rating[0] in "ABCDEN" else "N"

    # 降级逻辑
    downgrade_levels = 0
    downgrade_reason = ""

    if critical_count > 0:
        downgrade_levels = min(2, critical_count)  # Critical 最多降 2 级
        downgrade_reason = f"严重问题（{critical_count} Critical）"
    elif high_count >= 2:
        downgrade_levels = 1
        downgrade_reason = f"多项高危问题（{high_count} High）"
    elif medium_count >= 3:
        downgrade_reason = f"多项中等问题（{medium_count} Medium）"

    if downgrade_levels > 0 and base_letter in "ABCDE":
        # 降级映射
        level_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        letters = ["A", "B", "C", "D", "E"]
        current_level = level_map.get(base_letter, 2)
        new_level = min(4, current_level + downgrade_levels)  # 最多降到 E
        new_letter = letters[new_level]

        # 提取基础评级的描述部分
        description = base_rating.split("：", 1)[1] if "：" in base_rating else base_rating.split(" ", 1)[1] if " " in base_rating else ""

        return f"{new_letter} ⬇️ {description}（降级：{downgrade_reason}）"
    elif downgrade_reason and medium_count >= 3:
        # 中等问题不降级，只追加标注
        return f"{base_rating}（{downgrade_reason}）"

    return base_rating


def authenticity_note(model: str, results: list[ProbeResult]) -> tuple[str, str]:
    """黑盒一致性评估，取保守口径：永不证明真实，证据不足即不下结论。"""
    family = family_for_model(model)
    availability = availability_rate(results)
    if availability <= 0:
        return "无法判断", "所有请求失败；问题在中转站或该模型路由，不是模型输出能力。"
    if availability < 60:
        return "无法判断", "成功请求过少（接口不稳定），证据不足以评估真实性。"

    # 检查严重问题，优先返回特定标注
    severe_issues = detect_severe_issues(model, results)
    for issue in severe_issues:
        # 协议指纹异常
        if issue.probe_id == "protocol_fingerprint" and issue.severity == "critical":
            return "疑似协议转换/模型替换", f"协议指纹异常：{issue.reason}"

        # Thinking signature 缺失（Claude 模型）
        if issue.probe_id == "claude_thinking_signature" and issue.severity == "critical":
            return "疑似非官方 Claude", f"Thinking signature 检测失败：{issue.reason}"

    if family == "unknown":
        return "无法按名称判断", "模型 ID 不含清晰 GPT/Claude 家族特征，无法做名称一致性判断；仅能参考通用能力。"

    universal_cap = capability_score([r for r in results if r.probe.category == "universal"])
    targeted_cap = capability_score([r for r in results if r.probe.category == "targeted"])
    identity = next(
        (r for r in results if r.probe.probe_id == "identity_limits" and r.status == "ok"),
        None,
    )

    if targeted_cap is not None and targeted_cap < 65:
        return "疑似不匹配", (
            f"模型 ID 显示 {family}，但针对性探针偏低（{targeted_cap:.0f}/100）。"
            "黑盒检测不能定罪，建议追查路由/上游配置。"
        )
    if identity is None or identity.score < 60:
        return "证据不足", "身份与黑盒限制自述缺失或不规范，不足以支撑一致性判断。"
    if (
        universal_cap is not None
        and universal_cap >= 80
        and (targeted_cap is None or targeted_cap >= 75)
        and identity.score >= 60
        and availability >= 100
    ):
        return "未发现不一致", (
            "名称、通用能力、针对性探针与身份自述基本一致；"
            "但黑盒 API 无法证明真实底层模型，仅为一致性观察。"
        )
    return "证据不足", "部分能力探针表现一般或接口不完全稳定，不能可靠确认模型真实性。"


def recommendation_for(capability: float | None, availability: float, authenticity: str) -> str:
    if capability is None or availability <= 0:
        return "Do not use: endpoint unavailable"
    if availability < 60:
        return "Do not use for critical tasks: endpoint unstable"
    if "疑似" in authenticity:
        return "Use only after checking relay routing"
    if capability >= 80 and availability >= 100:
        return "Suitable for routine use; keep periodic checks"
    if capability >= 65:
        return "Use with caution; verify important outputs"
    return "Not recommended for critical tasks"


def build_decision_summary(model_results: dict[str, list[ProbeResult]]) -> list[DecisionSummaryRow]:
    rows: list[DecisionSummaryRow] = []
    for model, results in model_results.items():
        cap = capability_score(results)
        avail = availability_rate(results)
        auth, auth_reason = authenticity_note(model, results)
        errors, total, availability_text = availability_summary(results)
        ok = total - errors
        severe_issues = detect_severe_issues(model, results)
        rating = overall_rating(cap, avail, severe_issues)
        rows.append(
            {
                "model": model,
                "capability": cap,
                "availability": avail,
                "rating": rating,
                "authenticity": auth,
                "authenticity_reason": auth_reason,
                "ok_requests": ok,
                "total_requests": total,
                "availability_summary": availability_text,
                "recommendation": recommendation_for(cap, avail, auth),
                "severe_issues": [dataclasses.asdict(issue) for issue in severe_issues],
            }
        )
    return rows


def suggest_next_steps(error: str, context: dict[str, Any] | None = None) -> list[str]:
    """根据常见错误给出下一步建议；用于 TUI/报告边界。"""
    del context  # 预留给未来按 api_style/path 细分。
    text = error.lower()
    suggestions: list[str] = []
    if "base-url" in text or "base url" in text:
        suggestions.append("填写 Base URL；可带或不带 /v1。")
    if "api key" in text or "401" in text or "403" in text or "authorization" in text:
        suggestions.append("检查 API key、账户额度/权限，以及中转站是否接受 Bearer Authorization。")
    if "http 404" in text:
        suggestions.append("检查 Base URL 是否正确；若模型调用失败，尝试切换 API style。")
    if "429" in text:
        suggestions.append("触发限流：减少模型/探针数量，稍后重试，或增大请求间隔/额度。")
    if "network error" in text or "timed out" in text or "timeout" in text:
        suggestions.append("检查网络/VPN/代理和中转站地址；慢站点可增大 timeout。")
    if "invalid json" in text or "no choices" in text or "no text" in text:
        suggestions.append("接口返回格式与当前 API style 不匹配；尝试 openai-chat、openai-responses、anthropic 或 gemini。")
    if "auto failed" in text:
        suggestions.append("auto 协议探测失败：手动指定 api-style 逐个复测。")
    if "invalid model filter" in text or "bad escape" in text or "missing" in text and "regex" in text:
        suggestions.append("修正 Model filter 正则，或清空筛选条件。")
    if "no models" in text:
        suggestions.append("检查 /v1/models 是否可用、Model filter/Limit 是否过窄，或在 Model 字段手动填写单个模型。")
    if not suggestions:
        suggestions.append("查看上方错误详情；必要时先用 F5 选择或在 Model 字段手动填写单个模型，再切换 API style 复测。")
    return suggestions


def format_suggestions(error: str, prefix: str = "  - ") -> str:
    return "\n".join(prefix + item for item in suggest_next_steps(error))


def _report_summary_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary = report.get("summary")
    if isinstance(summary, list):
        return {str(row.get("model")): row for row in summary if isinstance(row, dict) and row.get("model")}
    rows: dict[str, dict[str, Any]] = {}
    models = report.get("models") or {}
    if not isinstance(models, dict):
        return rows
    for model, results in models.items():
        if not isinstance(results, list):
            continue
        total = len(results)
        ok_results = [item for item in results if isinstance(item, dict) and item.get("status") == "ok"]
        total_weight = sum(float(item.get("weight") or 0) for item in ok_results)
        capability = None
        if total_weight > 0:
            capability = sum(float(item.get("score") or 0) * float(item.get("weight") or 0) for item in ok_results) / total_weight
        availability = (len(ok_results) / total * 100) if total else 0.0
        rows[str(model)] = {
            "model": str(model),
            "capability": capability,
            "availability": availability,
            "rating": overall_rating(capability, availability),
            "authenticity": "N/A (old report)",
            "recommendation": recommendation_for(capability, availability, ""),
        }
    return rows


def _report_probe_map(report: dict[str, Any], model: str) -> dict[str, dict[str, Any]]:
    models = report.get("models") or {}
    if not isinstance(models, dict):
        return {}
    results = models.get(model) or []
    if not isinstance(results, list):
        return {}
    return {
        str(item.get("probe_id")): item
        for item in results
        if isinstance(item, dict) and item.get("probe_id")
    }


def _delta(current: Any, baseline: Any) -> float | None:
    if current is None or baseline is None:
        return None
    try:
        return float(current) - float(baseline)
    except (TypeError, ValueError):
        return None


def _probe_field_signature(item: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    signature: dict[str, Any] = {}
    usage = item.get("usage")
    if isinstance(usage, dict):
        signature["usage_keys"] = sorted(str(key) for key in usage)
    response_data = item.get("response_data")
    if isinstance(response_data, dict):
        signature["response_data_keys"] = sorted(str(key) for key in response_data)
        content = response_data.get("content")
        if isinstance(content, list):
            signature["content_types"] = sorted(
                str(block.get("type"))
                for block in content
                if isinstance(block, dict) and block.get("type")
            )
        choices = response_data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                signature["message_keys"] = sorted(str(key) for key in message)
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    signature["tool_call_types"] = sorted(
                        str(call.get("type"))
                        for call in tool_calls
                        if isinstance(call, dict) and call.get("type")
                    )
        events = response_data.get("events")
        if isinstance(events, list):
            names = []
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_name = event.get("event")
                data = event.get("data")
                if not event_name and isinstance(data, dict):
                    event_name = data.get("type")
                if event_name:
                    names.append(str(event_name))
            signature["sse_events"] = names
    return signature


def _field_changes(before: dict[str, Any] | None, after: dict[str, Any] | None) -> list[dict[str, Any]]:
    before_sig = _probe_field_signature(before)
    after_sig = _probe_field_signature(after)
    changes = []
    for field in sorted(set(before_sig) | set(after_sig)):
        if before_sig.get(field) != after_sig.get(field):
            changes.append({"field": field, "before": before_sig.get(field), "after": after_sig.get(field)})
    return changes


def load_report_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise ValueError("Invalid report JSON: expected top-level models object")
    return data


def compare_reports(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    baseline_summary = _report_summary_map(baseline)
    current_summary = _report_summary_map(current)
    baseline_models = set(baseline_summary)
    current_models = set(current_summary)
    changed_models = []
    for model in sorted(baseline_models & current_models):
        before = baseline_summary[model]
        after = current_summary[model]
        probe_changes = []
        before_probes = _report_probe_map(baseline, model)
        after_probes = _report_probe_map(current, model)
        for probe_id in sorted(set(before_probes) | set(after_probes)):
            old_probe = before_probes.get(probe_id)
            new_probe = after_probes.get(probe_id)
            if old_probe is None:
                probe_changes.append({"probe_id": probe_id, "change": "added"})
                continue
            if new_probe is None:
                probe_changes.append({"probe_id": probe_id, "change": "removed"})
                continue
            old_score = old_probe.get("score")
            new_score = new_probe.get("score")
            old_status = old_probe.get("status")
            new_status = new_probe.get("status")
            score_delta = _delta(new_score, old_score)
            field_changes = _field_changes(old_probe, new_probe)
            if old_status != new_status or (score_delta is not None and abs(score_delta) >= 0.01) or field_changes:
                probe_changes.append(
                    {
                        "probe_id": probe_id,
                        "change": "changed",
                        "status_before": old_status,
                        "status_after": new_status,
                        "score_delta": score_delta,
                        "field_changes": field_changes,
                    }
                )
        cap_delta = _delta(after.get("capability"), before.get("capability"))
        avail_delta = _delta(after.get("availability"), before.get("availability"))
        rating_changed = before.get("rating") != after.get("rating")
        authenticity_changed = before.get("authenticity") != after.get("authenticity")
        if probe_changes or (cap_delta is not None and abs(cap_delta) >= 0.01) or (avail_delta is not None and abs(avail_delta) >= 0.01) or rating_changed or authenticity_changed:
            changed_models.append(
                {
                    "model": model,
                    "capability_before": before.get("capability"),
                    "capability_after": after.get("capability"),
                    "capability_delta": cap_delta,
                    "availability_before": before.get("availability"),
                    "availability_after": after.get("availability"),
                    "availability_delta": avail_delta,
                    "rating_before": before.get("rating"),
                    "rating_after": after.get("rating"),
                    "authenticity_before": before.get("authenticity"),
                    "authenticity_after": after.get("authenticity"),
                    "probe_changes": probe_changes,
                }
            )
    return {
        "added_models": sorted(current_models - baseline_models),
        "removed_models": sorted(baseline_models - current_models),
        "changed_models": changed_models,
    }


def _format_delta(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}{suffix}"


def format_comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = ["## Baseline Comparison", ""]
    added = comparison.get("added_models") or []
    removed = comparison.get("removed_models") or []
    changed = comparison.get("changed_models") or []
    if not added and not removed and not changed:
        lines.append("No model, score, rating, authenticity, or probe-level changes detected.")
        return "\n".join(lines)
    if added:
        lines.append("- Added models: " + ", ".join(added))
    if removed:
        lines.append("- Removed models: " + ", ".join(removed))
    if changed:
        lines.extend(["", "| Model | Capability Δ | Availability Δ | Rating | Authenticity | Probe changes |", "| --- | ---: | ---: | --- | --- | ---: |"])
        for row in changed:
            rating = f"{row.get('rating_before')} → {row.get('rating_after')}" if row.get("rating_before") != row.get("rating_after") else str(row.get("rating_after"))
            auth = f"{row.get('authenticity_before')} → {row.get('authenticity_after')}" if row.get("authenticity_before") != row.get("authenticity_after") else str(row.get("authenticity_after"))
            lines.append(
                f"| {row['model']} | {_format_delta(row.get('capability_delta'))} | {_format_delta(row.get('availability_delta'), '%')} | {rating} | {auth} | {len(row.get('probe_changes') or [])} |"
            )
        for row in changed:
            probe_changes = row.get("probe_changes") or []
            if probe_changes:
                lines.extend(["", f"### Probe changes: {row['model']}", ""])
                for change in probe_changes:
                    field_changes = change.get("field_changes") or []
                    field_text = ""
                    if field_changes:
                        field_text = "; fields: " + ", ".join(str(item.get("field")) for item in field_changes)
                    if change.get("change") == "changed":
                        lines.append(
                            f"- {change['probe_id']}: {change.get('status_before')} → {change.get('status_after')}, score Δ {_format_delta(change.get('score_delta'))}{field_text}"
                        )
                    else:
                        lines.append(f"- {change['probe_id']}: {change.get('change')}")
    return "\n".join(lines)


def _compact_signature_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or ""
    return str(value)


def build_report(model_results: dict[str, list[ProbeResult]], config: ApiConfig, comparison: dict[str, Any] | None = None) -> str:
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# AI Relay Audit Report",
        "",
        f"- Time: {now}",
        f"- Base URL: {config.base_url}",
        f"- API style: {config.api_style}",
        f"- Temperature: {config.temperature}",
        f"- Max tokens: {config.max_tokens}",
        "",
        "## Decision Summary",
        "",
        "| Model | Rating | Capability | Availability | Authenticity | Recommendation |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in build_decision_summary(model_results):
        cap_text = f"{row['capability']:.1f}/100" if row["capability"] is not None else "N/A"
        model = str(row["model"]).replace("|", "\\|")
        rating = str(row["rating"]).replace("|", "\\|")
        auth = str(row["authenticity"]).replace("|", "\\|")
        rec = str(row["recommendation"]).replace("|", "\\|")
        lines.append(f"| {model} | {rating} | {cap_text} | {row['availability']:.0f}% | {auth} | {rec} |")
    lines.extend(
        [
            "",
            "## Important Limit",
            "",
            "Black-box prompts cannot prove the real vendor model with cryptographic certainty. The authenticity result is an evidence-based consistency assessment using model ID, self-description, behavior, and targeted probes.",
            "",
        ]
    )
    for model, results in model_results.items():
        cap = capability_score(results)
        avail = availability_rate(results)
        auth, auth_reason = authenticity_note(model, results)
        errors, total, availability_text = availability_summary(results)
        ok = total - errors
        latencies = [r.latency_ms for r in results if r.latency_ms is not None]
        avg_latency = int(statistics.mean(latencies)) if latencies else None
        capability_line = f"{cap:.1f}/100 ({ok}/{total} probes scored)" if cap is not None else "N/A (no successful probe)"

        # 检测严重问题
        severe_issues = detect_severe_issues(model, results)

        # 按严重性分组
        critical_issues = [issue for issue in severe_issues if issue.severity == "critical"]
        high_issues = [issue for issue in severe_issues if issue.severity == "high"]
        medium_issues = [issue for issue in severe_issues if issue.severity == "medium"]

        lines.extend(
            [
                f"## {model}",
                "",
            ]
        )

        # 显示严重问题汇总
        if critical_issues:
            lines.append(f"\n⚠️  **CRITICAL ISSUES ({len(critical_issues)})**:")
            for issue in critical_issues:
                lines.append(f"- {issue.icon} {issue.probe_title}: 分数 {issue.score:.1f}, {issue.reason}")
            lines.append("")

        if high_issues:
            lines.append(f"⚠️  **HIGH ISSUES ({len(high_issues)})**:")
            for issue in high_issues:
                lines.append(f"- {issue.icon} {issue.probe_title}: 分数 {issue.score:.1f}, {issue.reason}")
            lines.append("")

        if medium_issues:
            lines.append(f"⚠️  **MEDIUM ISSUES ({len(medium_issues)})**:")
            for issue in medium_issues:
                lines.append(f"- {issue.icon} {issue.probe_title}: 分数 {issue.score:.1f}, {issue.reason}")
            lines.append("")

        lines.extend(
            [
                f"- Capability: {capability_line}",
                f"- Availability: {avail:.0f}% ({ok}/{total} requests OK) — {availability_text}",
                f"- Rating: {overall_rating(cap, avail, severe_issues)}",
                f"- Authenticity: {auth}",
                f"- Authenticity reason: {auth_reason}",
                f"- Average latency: {avg_latency if avg_latency is not None else 'N/A'} ms",
            ]
        )
        evidence_rows = []
        for result in results:
            if not result.response_data and result.probe.probe_id not in PROTOCOL_PROBE_IDS:
                continue
            signature = _probe_field_signature({"usage": result.usage, "response_data": result.response_data})
            if signature:
                evidence_rows.append((result.probe.probe_id, signature))
        if evidence_rows:
            lines.extend(
                [
                    "",
                    "### Protocol Evidence",
                    "",
                    "| Probe | Usage keys | Response keys | Content/message/tool/SSE evidence |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for probe_id, signature in evidence_rows:
                usage_keys = _compact_signature_value(signature.get("usage_keys")).replace("|", "\\|")
                response_keys = _compact_signature_value(signature.get("response_data_keys")).replace("|", "\\|")
                evidence = "; ".join(
                    part
                    for part in [
                        "content=" + _compact_signature_value(signature.get("content_types")) if signature.get("content_types") else "",
                        "message_keys=" + _compact_signature_value(signature.get("message_keys")) if signature.get("message_keys") else "",
                        "tool_calls=" + _compact_signature_value(signature.get("tool_call_types")) if signature.get("tool_call_types") else "",
                        "sse=" + _compact_signature_value(signature.get("sse_events")) if signature.get("sse_events") else "",
                    ]
                    if part
                ).replace("|", "\\|")
                lines.append(f"| {probe_id} | {usage_keys} | {response_keys} | {evidence} |")
        lines.extend(
            [
                "",
                "| Probe | Category | Weight | Score | Status | Reason |",
                "| --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        # 创建探针 ID 到严重问题的映射
        issue_map = {issue.probe_id: issue for issue in severe_issues}

        for result in results:
            # 严重问题标记
            severity_icon = ""
            if result.probe.probe_id in issue_map:
                severity_icon = f" {issue_map[result.probe.probe_id].icon}"

            reason = result.reason.replace("|", "\\|").replace("\n", " ")
            error = (result.error or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {result.probe.title}{severity_icon} | {result.probe.category} | {result.probe.weight} | {result.score:.1f} | {result.status} | {reason or error} |"
            )
        if errors:
            lines.extend(["", "### Error Summary", ""])
            unique_errors = sorted({result.error for result in results if result.error})
            for error in unique_errors:
                lines.append(f"- {error}")
        lines.append("")
    if comparison is not None:
        lines.extend([format_comparison_markdown(comparison), ""])
    return "\n".join(lines)


def write_reports(output_dir: str, model_results: dict[str, list[ProbeResult]], config: ApiConfig, comparison: dict[str, Any] | None = None) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_dir, f"audit_report_{stamp}.md")
    json_path = os.path.join(output_dir, f"audit_report_{stamp}.json")
    report = build_report(model_results, config, comparison)
    serializable: dict[str, Any] = {
        "base_url": config.base_url,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": build_decision_summary(model_results),
        "models": {
            model: [
                {
                    "probe_id": result.probe.probe_id,
                    "title": result.probe.title,
                    "category": result.probe.category,
                    "weight": result.probe.weight,
                    "status": result.status,
                    "score": result.score,
                    "reason": result.reason,
                    "latency_ms": result.latency_ms,
                    "usage": result.usage,
                    "error": result.error,
                    "response": result.response,
                    "response_data": result.response_data,
                }
                for result in results
            ]
            for model, results in model_results.items()
        },
    }
    if comparison is not None:
        serializable["comparison"] = comparison
    # Validate JSON before writing either report, so a serialization bug cannot leave a half-saved run.
    json_text = json.dumps(serializable, ensure_ascii=False, indent=2)
    md_tmp = md_path + ".tmp"
    json_tmp = json_path + ".tmp"
    with open(md_tmp, "w", encoding="utf-8") as file:
        file.write(report)
    with open(json_tmp, "w", encoding="utf-8") as file:
        file.write(json_text)
    os.replace(md_tmp, md_path)
    os.replace(json_tmp, json_path)
    return md_path, json_path


def report_dict_from_results(model_results: dict[str, list[ProbeResult]], config: ApiConfig, comparison: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "base_url": config.base_url,
        "summary": build_decision_summary(model_results),
        "models": {
            model: [
                {
                    "probe_id": result.probe.probe_id,
                    "title": result.probe.title,
                    "category": result.probe.category,
                    "weight": result.probe.weight,
                    "status": result.status,
                    "score": result.score,
                    "reason": result.reason,
                    "latency_ms": result.latency_ms,
                    "usage": result.usage,
                    "error": result.error,
                    "response": result.response,
                    "response_data": result.response_data,
                }
                for result in results
            ]
            for model, results in model_results.items()
        },
    }
    if comparison is not None:
        data["comparison"] = comparison
    return data
