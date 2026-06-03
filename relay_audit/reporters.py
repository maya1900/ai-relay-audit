from __future__ import annotations

import json
import queue
from typing import Any

from .models import Probe, ProbeResult
from .reporting import capability_score, format_suggestions, indent


class Reporter:
    """审计过程的输出汇聚点。基类方法均为空操作，子类负责具体渲染。

    取代过去 TUI 用 redirect_stdout 捕获 print 的做法：run_audit/run_probe 只调用
    这些回调，由 ConsoleReporter（标准输出）或 TuiReporter（推入队列）决定去向。
    """

    def section(self, title: str) -> None:  # noqa: D401
        pass

    def info(self, text: str) -> None:
        pass

    def probe_start(self, probe: Probe, index: int, total: int, show_prompt: bool) -> None:
        pass

    def probe_result(self, result: "ProbeResult", show_prompt: bool) -> None:
        pass

    def model_done(
        self,
        model: str,
        capability: float | None,
        availability: float,
        rating: str,
        auth: str,
        auth_reason: str,
    ) -> None:
        pass

    def progress(self, done: int, total: int) -> None:
        pass


def _capability_text(capability: float | None) -> str:
    return f"{capability:.1f}/100" if capability is not None else "N/A（无成功探针）"


class ConsoleReporter(Reporter):
    """标准输出 reporter，供内部测试和非 TUI 调用复用。"""

    def section(self, title: str) -> None:
        print_section(title)

    def info(self, text: str) -> None:
        print(text)

    def probe_start(self, probe: Probe, index: int, total: int, show_prompt: bool) -> None:
        print(f"\n[{probe.probe_id}] {probe.title} | 权重 {probe.weight}")
        if show_prompt:
            print("- System:")
            print(indent(probe.system))
            print("- User:")
            print(indent(probe.user))

    def probe_result(self, result: "ProbeResult", show_prompt: bool) -> None:
        if result.status == "error":
            print(f"- 失败: {result.error}")
            print("- 下一步建议:")
            print(format_suggestions(result.error or result.reason))
            return
        print(f"- 延迟: {result.latency_ms} ms")
        if result.usage:
            print(f"- Token 用量: {json.dumps(result.usage, ensure_ascii=False)}")
        print("- 模型回复:")
        print(indent(result.response.strip() or "<empty>"))
        print(f"- 判分: {result.score:.1f}/100")
        print(f"- 理由: {result.reason}")

    def model_done(
        self,
        model: str,
        capability: float | None,
        availability: float,
        rating: str,
        auth: str,
        auth_reason: str,
    ) -> None:
        print(f"\n模型 {model} 能力分: {_capability_text(capability)} | 可用性: {availability:.0f}%")
        print(f"评级: {rating}")
        print(f"真实性评估: {auth} - {auth_reason}")


class TuiReporter(Reporter):
    """把审计过程渲染成紧凑文本行推入队列，并把完整细节作为哨兵事件交给 TUI。"""

    def __init__(self, output_queue: "queue.Queue[str]", language: str = "zh") -> None:
        self.queue = output_queue
        self.language = language

    def zh(self) -> bool:
        return self.language == "zh"

    def text(self, en: str, zh: str) -> str:
        return zh if self.zh() else en

    def _emit(self, text: str) -> None:
        for line in text.split("\n"):
            self.queue.put(line)

    def _event(self, name: str, payload: dict[str, Any]) -> None:
        self.queue.put(f"__{name}__ {json.dumps(payload, ensure_ascii=False)}")

    def section(self, title: str) -> None:
        self._emit(f"\n◇ {title}")

    def info(self, text: str) -> None:
        self._emit(self.localize_info(text))

    def localize_info(self, text: str) -> str:
        if not self.zh():
            return text
        replacements = [
            ("Models: ", "模型："),
            ("Client adapters: ", "客户端适配："),
            ("Probe requests: ", "探针请求数："),
            ("Total API requests: ", "总 API 请求数："),
            ("Estimated input tokens: ", "预估输入 token："),
            ("Max output token budget: ", "最大输出 token 预算："),
            ("Estimated official cost: unknown (some models are not in the built-in official pricing table).", "官方价格预估：未知（部分模型不在内置官方价格表中）。"),
            ("Estimated official cost upper bound: ", "官方价格预估上限："),
            ("Probes by model: ", "每个模型探针数："),
            ("Cost by model: ", "每个模型费用："),
            ("Output dir: ", "输出目录："),
            ("Auto-save report: yes", "自动保存报告：是"),
            ("Auto-save report: no", "自动保存报告：否"),
            ("Save report file is off. Press s after completion to save this report.", "未开启自动保存。检测完成后可按 s 保存本次报告。"),
            ("No models available to select.", "没有可选择的模型。"),
        ]
        for old, new in replacements:
            if text.startswith(old):
                return new + text[len(old) :]
            if text == old:
                return new
        return text

    def probe_start(self, probe: Probe, index: int, total: int, show_prompt: bool) -> None:
        self._emit(f"▶ [{index}/{total}] {probe.probe_id} — {probe.title}")
        if show_prompt:
            self._event(
                "DETAIL",
                {
                    "title": f"{self.text('Prompt', '提示词')}: {probe.probe_id}",
                    "lines": [
                        f"[{probe.probe_id}] {probe.title}",
                        "",
                        self.text("System:", "系统提示词："),
                        indent(probe.system),
                        "",
                        self.text("User:", "用户提示词："),
                        indent(probe.user),
                    ],
                },
            )

    def probe_result(self, result: "ProbeResult", show_prompt: bool) -> None:
        # 判断是否为严重问题
        is_severe = result.status == "ok" and result.score < 30 and result.probe.weight >= 15
        severity_marker = " ⚠️ SEVERE" if is_severe else ""

        if result.status == "error":
            self._emit(f"✗ {result.probe.probe_id} {self.text('ERROR', '失败')} — {result.error}")
            suggestions = suggest_next_steps(result.error or result.reason)
            self._event(
                "ERROR",
                {
                    "probe_id": result.probe.probe_id,
                    "title": result.probe.title,
                    "error": result.error or result.reason,
                    "suggestions": suggestions,
                },
            )
        else:
            usage_text = f" | tokens={json.dumps(result.usage, ensure_ascii=False)}" if result.usage else ""
            ok_text = self.text("OK", "通过")
            self._emit(f"✓ {result.probe.probe_id} {ok_text} — {result.score:.1f}/100 | {result.latency_ms} ms{usage_text}{severity_marker}")

        detail_lines = [
            f"[{result.probe.probe_id}] {result.probe.title}",
            f"{self.text('Status', '状态')}: {result.status}",
            f"{self.text('Score', '分数')}: {result.score:.1f}/100{severity_marker}",
            f"{self.text('Latency', '延迟')}: {result.latency_ms if result.latency_ms is not None else 'N/A'} ms",
            f"{self.text('Reason', '理由')}: {result.reason}",
        ]
        if result.error:
            detail_lines.extend(["", self.text("Error:", "错误："), result.error, "", self.text("Suggestions:", "下一步建议：")])
            detail_lines.extend(f"- {item}" for item in suggest_next_steps(result.error))
        if result.usage:
            detail_lines.extend(["", self.text("Usage:", "Token 用量："), json.dumps(result.usage, ensure_ascii=False, indent=2)])
        if result.response:
            detail_lines.extend(["", self.text("Response:", "模型回复："), result.response.strip() or "<empty>"])
        self._event("DETAIL", {"title": f"{self.text('Result', '结果')}: {result.probe.probe_id}", "lines": detail_lines})

    def model_done(
        self,
        model: str,
        capability: float | None,
        availability: float,
        rating: str,
        auth: str,
        auth_reason: str,
    ) -> None:
        if self.zh():
            self._emit(f"\n模型 {model} 能力分: {_capability_text(capability)} | 可用性: {availability:.0f}%")
            self._emit(f"评级: {rating}")
            self._emit(f"真实性评估: {auth} - {auth_reason}")
        else:
            self._emit(f"\nModel {model} capability: {_capability_text(capability)} | availability: {availability:.0f}%")
            self._emit(f"Rating: {rating}")
            self._emit(f"Authenticity: {auth} - {auth_reason}")

    def progress(self, done: int, total: int) -> None:
        self.queue.put(f"__PROGRESS__ {done}/{total}")


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
