from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import queue
import re
import sys
import threading
import time
import unicodedata
from typing import Any, Callable

try:
    import curses
except ImportError:  # curses 在标准 Windows Python 发行版上不可用
    curses = None  # type: ignore[assignment]

from .api import chat, fetch_models
from .models import ApiConfig, AuditConfig, DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT, ProbeResult
from .pricing import build_run_estimate, format_run_estimate, format_usd
from .probes import applicable_probes
from .reporters import TuiReporter
from .reporting import build_decision_summary, build_report, format_suggestions, suggest_next_steps, write_reports
from .scoring import family_for_model
from .utils import filter_models


def make_namespace_from_tui(state: dict[str, Any]) -> argparse.Namespace:
    limit_value = str(state["limit"]).strip()
    models_value = str(state["model"]).strip()
    filter_value = str(state["model_filter"]).strip()
    return argparse.Namespace(
        base_url=str(state["base_url"]).strip(),
        api_key=str(state["api_key"]).strip(),
        models=models_value or None,
        model_filter=filter_value or None,
        limit=int(limit_value) if limit_value else None,
        timeout=int(str(state["timeout"]).strip() or DEFAULT_TIMEOUT),
        max_tokens=int(str(state["max_tokens"]).strip() or DEFAULT_MAX_TOKENS),
        temperature=float(str(state["temperature"]).strip() or 0),
        api_style=str(state.get("api_style", "auto")).strip() or "auto",
        mode=str(state.get("mode", "standard")).strip() or "standard",
        output_dir=str(state["output_dir"]).strip() or "reports",
        save_report=bool(state.get("save_report")),
        all_targeted=bool(state["all_targeted"]),
        hide_prompts=bool(state["hide_prompts"]),
        tui=True,
    )


def model_family_summary(models_value: str) -> str:
    models = [item.strip() for item in models_value.split(",") if item.strip()]
    if not models:
        return "empty"
    if len(models) > 1:
        return "multiple not allowed"
    return family_for_model(models[0])


def display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def truncate_display(text: str, width: int) -> str:
    if width <= 0:
        return ""
    current_width = 0
    chars: list[str] = []
    for char in text:
        char_width = 0 if unicodedata.combining(char) else (2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1)
        if current_width + char_width > width:
            break
        chars.append(char)
        current_width += char_width
    return "".join(chars)


def pad_display(text: str, width: int) -> str:
    shown = truncate_display(text, width)
    return shown + " " * max(0, width - display_width(shown))


def _wrap_long_token(token: str, width: int) -> list[str]:
    parts: list[str] = []
    remaining = token
    while remaining:
        part = truncate_display(remaining, width)
        if not part:
            break
        parts.append(part)
        remaining = remaining[len(part) :]
    return parts or [""]


def wrap_for_width(text: str, width: int) -> list[str]:
    if width <= 8:
        return [truncate_display(text, width)]
    words: list[str] = []
    for raw_word in text.split():
        if display_width(raw_word) <= width:
            words.append(raw_word)
        else:
            words.extend(_wrap_long_token(raw_word, width))
    lines: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif display_width(current) + 1 + display_width(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def wrapped_log_lines(logs: list[str], width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in logs:
        segments = str(raw_line).replace("\t", "    ").splitlines() or [""]
        for line in segments:
            if not line:
                lines.append("")
                continue
            lines.extend(wrap_for_width(line, width))
    return lines


def validate_field(key: str, value: str) -> str | None:
    """即时校验单个字段；返回错误文案，合法则返回 None。"""
    value = value.strip()
    if key in {"timeout", "max_tokens"}:
        if not value:
            return "不能为空，需为正整数"
        if not value.isdigit() or int(value) <= 0:
            return "需为正整数"
    elif key == "limit":
        if value:
            if not value.isdigit():
                return "需为非负整数或留空"
            if int(value) < 0:
                return "需为非负整数或留空"
    elif key == "temperature":
        if value:
            try:
                float(value)
            except ValueError:
                return "需为数字"
    return None


def run_tui() -> int:
    if curses is None:
        print(
            "TUI 需要 curses 模块，标准 Windows Python 不自带。\n"
            "Windows 如需 TUI，可先 pip install windows-curses。",
            file=sys.stderr,
        )
        return 2
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("TUI requires an interactive terminal. Run it directly in your shell: python3 ai_relay_audit.py --tui", file=sys.stderr)
        return 2
    return curses.wrapper(lambda stdscr: TuiApp(stdscr).run())


class TuiApp:
    """全屏终端界面。把原 tui_main 的闭包拆成方法，状态收敛到实例属性。

    健壮性：所有绘制经 _safe_addstr 裁剪并吞掉 curses.error；终端过小时显示提示而非崩溃；
    检测过程实时显示进度、可用 Esc/c 取消；数值字段即时校验。
    """

    MIN_WIDTH = 50
    MIN_HEIGHT = 12

    def __init__(self, stdscr: Any) -> None:
        self.stdscr = stdscr
        self.state: dict[str, Any] = {
            "base_url": os.getenv("AI_RELAY_BASE_URL", ""),
            "api_key": os.getenv("AI_RELAY_API_KEY", ""),
            "model": "",
            "model_filter": "",
            "limit": "",
            "timeout": str(DEFAULT_TIMEOUT),
            "max_tokens": str(DEFAULT_MAX_TOKENS),
            "temperature": "0",
            "api_style": "auto",
            "mode": "standard",
            "output_dir": "reports",
            "save_report": False,
            "all_targeted": False,
            "hide_prompts": False,
        }
        self.fields: list[tuple[str, str]] = [
            ("base_url", "Base URL"),
            ("api_key", "API Key"),
            ("model", "Model"),
            ("model_filter", "Model filter"),
            ("limit", "Limit"),
            ("timeout", "Timeout"),
            ("max_tokens", "Max tokens"),
            ("temperature", "Temperature"),
            ("api_style", "API style"),
            ("mode", "Mode"),
            ("output_dir", "Output dir"),
            ("save_report", "Save report file"),
            ("all_targeted", "All targeted probes"),
            ("hide_prompts", "Hide prompts"),
        ]
        self.field_labels_zh = {
            "base_url": "中转站地址",
            "api_key": "API 密钥",
            "model": "模型",
            "model_filter": "模型筛选",
            "limit": "数量限制",
            "timeout": "超时秒数",
            "max_tokens": "最大 tokens",
            "temperature": "随机性",
            "api_style": "接口风格",
            "mode": "检测模式",
            "output_dir": "输出目录",
            "save_report": "保存报告",
            "all_targeted": "全量探针",
            "hide_prompts": "隐藏提示词",
        }
        self.language = "zh"
        self.selected = 0
        self.logs: list[str] = [
            "欢迎使用 AI Relay Audit。",
            "功能：检测中转站 /v1/models、模型可用性、结构化输出、多约束推理、提示注入抵抗、代码任务与身份一致性。",
            "针对 GPT/Claude 模型会额外运行家族特定探针；报告会给出能力分、可用性、真实性观察和决策摘要。",
            "第 1 步：填写中转站地址和 API 密钥。",
            "第 2 步：按 F5 拉取并选择模型，也可以手动填写模型。",
            "第 3 步：按 F9 开始检测。",
            "实时日志默认保持简洁；检测后可按 r 看报告、d 看详情、e 看错误、s 保存、x 清空日志。",
            "提示：PageUp/PageDown 滚动当前视图；运行中 Esc/c 可取消。按 g 可切换中/英界面。",
        ]
        self.status = "Idle"
        self.worker: threading.Thread | None = None
        self.queue: queue.Queue[str] = queue.Queue()
        self.fetched_models: list[str] = []
        self.log_scroll = 0
        self.cancel_event = threading.Event()
        self.progress: tuple[int, int] | None = None
        self.view_mode = "log"
        self.spinner_frames = "|/-\\"
        self.last_report_text = ""
        self.detail_items: list[dict[str, Any]] = []
        self.error_items: list[dict[str, Any]] = []
        self.last_model_results: dict[str, list[ProbeResult]] | None = None
        self.last_report_config: ApiConfig | None = None
        self.last_report_output_dir: str | None = None
        self.last_saved_paths: tuple[str, str] | None = None
        self.last_report_unsaved = False
        self.help_text: dict[str, tuple[str, str]] = {
            "base_url": (
                "Relay endpoint, for example https://api.example.com or https://api.example.com/v1.",
                "Enter your relay URL, then move to API Key.",
            ),
            "api_key": (
                "Bearer API key used for /v1/models and /v1/chat/completions. It is masked on screen.",
                "Enter the key. Then press F5 to pick a model, or fill Model manually.",
            ),
            "model": (
                "One model ID to audit. TUI intentionally tests one model per run.",
                "Press F5 to fetch and choose a model, or type one manually. Then press F9.",
            ),
            "model_filter": (
                "Regex applied to fetched /v1/models before the picker opens.",
                "Use patterns like (gpt|claude), or leave empty to show all fetched models.",
            ),
            "limit": (
                "Maximum number of fetched models kept after Model filter. Empty means no limit.",
                "Set a small number for long model lists, then press F5 to pick from the filtered set.",
            ),
            "timeout": (
                "Request timeout in seconds for each API call.",
                "Keep 90 unless the relay is slow; use 180 for unstable or distant endpoints.",
            ),
            "max_tokens": (
                "Maximum answer length for each probe.",
                "Keep 900 for normal audits. Lower values may truncate code or reasoning tests.",
            ),
            "temperature": (
                "Sampling randomness. Audits should be deterministic.",
                "Keep 0 for repeatable results.",
            ),
            "api_style": (
                "Endpoint style: auto, openai-chat, openai-responses, anthropic, or gemini.",
                "Keep auto for normal relays.",
            ),
            "mode": (
                "Detection mode: quick (fast validation), standard (balanced), or full (comprehensive).",
                "quick = 2 probes, standard = 7 probes (default), full = all targeted probes.",
            ),
            "output_dir": (
                "Directory used only when Save report file is enabled.",
                "Keep reports unless you want saved files in a different folder.",
            ),
            "save_report": (
                "Save Markdown and JSON reports to Output dir after showing the report in the log.",
                "Leave off for quick checks. Turn on when you want report files.",
            ),
            "all_targeted": (
                "Runs both GPT-specific and Claude-specific probes against this model.",
                "Leave off for normal checks. Turn on when you suspect model spoofing.",
            ),
            "hide_prompts": (
                "Hides full probe prompts from the live log.",
                "Leave off when you want a transparent audit trail; turn on for shorter logs.",
            ),
        }
        self.help_text_zh: dict[str, tuple[str, str]] = {
            "base_url": ("中转站地址，例如 https://api.example.com 或 https://api.example.com/v1。", "填写地址后移动到 API 密钥。"),
            "api_key": ("用于 /v1/models 和模型调用的 Bearer API key，界面会打码显示。", "填写密钥后按 F5 选模型，或手动填写模型。"),
            "model": ("本次要检测的单个模型 ID。TUI 模式刻意保持一次只测一个模型。", "按 F5 拉取并选择模型，或手动填写；然后按 F9。"),
            "model_filter": ("拉取模型列表后应用的正则筛选。", "可填 (gpt|claude)，留空则显示全部模型。"),
            "limit": ("筛选后最多保留多少个模型，留空表示不限。", "模型列表很长时可先限制数量，再按 F5 选择。"),
            "timeout": ("每次 API 请求的超时秒数。", "默认 90；中转站慢或网络远时可用 180。"),
            "max_tokens": ("每个探针允许的最大输出 token。", "默认 900；过低可能截断代码或推理题。"),
            "temperature": ("采样随机性。检测建议保持确定性。", "保持 0 以便复测结果更稳定。"),
            "api_style": ("调用协议：auto、openai-chat、openai-responses、anthropic 或 gemini。", "通常保持 auto；失败时再手动切换协议复测。"),
            "mode": ("检测模式：quick（快速）、standard（标准）或 full（完整）。", "quick=2探针，standard=7探针（默认），full=含所有针对性探针。"),
            "output_dir": ("保存报告时使用的目录。", "默认 reports；需要保存到别处时再修改。"),
            "save_report": ("检测完成后自动保存 Markdown 和 JSON 报告。", "快速检查可关闭；完成后也可按 s 手动保存。"),
            "all_targeted": ("对当前模型同时运行 GPT 和 Claude 针对性探针。", "常规检测关闭；怀疑模型冒充时开启。"),
            "hide_prompts": ("实时日志不展示完整 prompt。", "想保留透明审计轨迹时关闭；想减少详情时开启。"),
        }

    # ---- curses helpers -------------------------------------------------

    @staticmethod
    def _safe_addstr(win: Any, y: int, x: int, text: str, attr: int = 0) -> None:
        """裁剪到窗口范围并吞掉 curses.error，避免越界写入导致崩溃。"""
        if y < 0 or x < 0:
            return
        try:
            max_y, max_x = win.getmaxyx()
        except curses.error:
            return
        if y >= max_y or x >= max_x:
            return
        # 末列写入在多数终端会抛错，预留 1 列安全边距；按显示宽度裁剪，避免中文宽字符换行到左侧。
        text = truncate_display(text, max(0, max_x - x - 2))
        if not text:
            return
        try:
            win.addstr(y, x, text, attr)
        except curses.error:
            pass

    # ---- log handling ---------------------------------------------------

    def add_log(self, line: str) -> None:
        self.logs.append(line)
        if len(self.logs) > 1000:
            del self.logs[:200]
        if self.log_scroll > 0:
            self.log_scroll += 1

    def clear_logs(self) -> None:
        self.logs.clear()
        self.log_scroll = 0
        if self.view_mode == "log":
            self.set_log_bottom()

    def _report_preview_lines(self) -> list[str]:
        if not self.last_report_text:
            return [self.text("No report yet. Run an audit first, then press r again.", "还没有报告。请先运行检测，完成后再按 r 查看报告。")]
        return self.last_report_text.splitlines()

    def _detail_lines(self) -> list[str]:
        if not self.detail_items:
            return [self.text("No probe details yet. Run an audit first.", "还没有探针详情。请先运行一次检测。")]
        lines: list[str] = []
        for index, item in enumerate(self.detail_items[-30:], 1):
            if lines:
                lines.append("")
            lines.append(f"## {index}. {item.get('title', self.text('Detail', '详情'))}")
            lines.extend(str(line) for line in item.get("lines", []))
        return lines

    def _error_lines(self) -> list[str]:
        if not self.error_items:
            return [self.text("No errors recorded in this session.", "当前会话还没有记录错误。")]
        lines = [self.text("Errors and next-step suggestions", "错误和下一步建议"), ""]
        for index, item in enumerate(self.error_items, 1):
            lines.append(f"{index}. [{item.get('probe_id', 'n/a')}] {item.get('title', '')}")
            lines.append(str(item.get("error", "")))
            suggestions = item.get("suggestions") or []
            if suggestions:
                lines.append(self.text("Suggestions:", "建议："))
                lines.extend(f"  - {suggestion}" for suggestion in suggestions)
            lines.append("")
        return lines

    def view_lines(self) -> list[str]:
        if self.view_mode == "report":
            return self._report_preview_lines()
        if self.view_mode == "details":
            return self._detail_lines()
        if self.view_mode == "errors":
            return self._error_lines()
        if self.worker_running():
            # 运行中在底部留出呼吸空间，最新日志不要紧贴页脚。
            return self.logs + ["", "", ""]
        return self.logs

    def scroll_log(self, delta: int) -> None:
        height, width = self.stdscr.getmaxyx()
        left_width = min(54, max(34, width // 3))
        log_width = width - left_width - 4
        visible_height = max(1, height - 5)
        total_lines = len(wrapped_log_lines(self.view_lines(), max(10, log_width - 1)))
        max_scroll = max(0, total_lines - visible_height)
        self.log_scroll = min(max_scroll, max(0, self.log_scroll + delta))

    def set_log_bottom(self) -> None:
        self.log_scroll = 0

    def set_view(self, mode: str) -> None:
        self.view_mode = mode
        self.set_log_bottom()

    # ---- worker lifecycle ----------------------------------------------

    def worker_running(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def start_worker(self, target: Callable[[], None]) -> None:
        if self.worker_running():
            self.add_log(self.text("A task is already running.", "已有任务正在运行。"))
            return
        self.status = "Running"
        self.progress = None
        self.cancel_event.clear()
        self.worker = threading.Thread(target=target, daemon=True)
        self.worker.start()

    def fetch_models_worker(self) -> None:
        reporter = TuiReporter(self.queue, self.language)
        try:
            cfg = AuditConfig.from_namespace(make_namespace_from_tui(self.state))
            if not cfg.api.base_url or not cfg.api.api_key:
                raise ValueError(self.text("Base URL and API Key are required before fetching models.", "获取模型列表前必须填写中转站地址和 API 密钥。"))
            reporter.section(self.text("Fetch model list", "获取模型列表"))
            loaded_models = fetch_models(cfg.api)
            models = filter_models(loaded_models, cfg.model_filter, cfg.limit)
            self.fetched_models = models
            reporter.info(
                self.text(
                    f"Loaded {len(loaded_models)} models; {len(models)} after filter/limit. Select one in the popup.",
                    f"已加载 {len(loaded_models)} 个模型；筛选/限制后剩 {len(models)} 个。请在弹窗中选择一个模型。",
                )
            )
            self.queue.put("__SELECT_MODEL__")
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            reporter.info(self.text(f"Fetch failed: {error}", f"模型列表获取失败：{error}"))
            reporter.info(self.text("Suggestions:", "下一步建议："))
            reporter.info(format_suggestions(error))
            self.queue.put(
                "__ERROR__ "
                + json.dumps(
                    {"probe_id": "fetch_models", "title": self.text("Fetch models", "获取模型列表"), "error": error, "suggestions": suggest_next_steps(error)},
                    ensure_ascii=False,
                )
            )
        finally:
            self.queue.put("__DONE__")

    def audit_worker(self) -> None:
        reporter = TuiReporter(self.queue, self.language)
        try:
            self.set_view("log")
            self.detail_items = []
            self.error_items = []
            self.last_report_text = ""
            self.last_model_results = None
            self.last_report_config = None
            self.last_report_output_dir = None
            self.last_saved_paths = None
            self.last_report_unsaved = False
            cfg = AuditConfig.from_namespace(make_namespace_from_tui(self.state))
            missing = []
            if not cfg.api.base_url:
                missing.append(self.text("Base URL", "中转站地址"))
            if not cfg.api.api_key:
                missing.append(self.text("API Key", "API 密钥"))
            if not cfg.models:
                missing.append(self.text("Model", "模型"))
            if missing:
                raise ValueError(self.text("Required before F9: ", "F9 前请先填写：") + ", ".join(missing))
            if len(cfg.models) > 1:
                raise ValueError(self.text("TUI runs one model at a time. Please keep only one model in Model.", "TUI 每次只检测一个模型，请只保留一个模型。"))
            model = cfg.models[0]
            reporter.section("连接测试")
            try:
                response, _usage, latency_ms = chat(cfg.api, model, "Reply with exactly OK.", "ping")
                reporter.info(
                    self.text(
                        f"Model call reachable via api_style={cfg.api.api_style}: {latency_ms} ms, response={response.strip()[:80]!r}",
                        f"模型调用可达，api_style={cfg.api.api_style}：{latency_ms} ms，回复={response.strip()[:80]!r}",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"模型调用不可用，已停止检测：{exc}") from exc
            run_cfg = dataclasses.replace(cfg, save_report=False)
            from .runner import run_audit

            model_results, _ = run_audit(run_cfg, reporter, should_cancel=self.cancel_event.is_set)
            if model_results:
                self.remember_latest_report(model_results, cfg)
            else:
                self.last_report_text = ""
            if model_results:
                reporter.section(self.text("Audit complete", "检测完成"))
                reporter.info("╔══════════════════════════════════════════════════════════════╗")
                reporter.info(self.text("║                    AUDIT COMPLETE                         ║", "║                    AUDIT COMPLETE / 检测完成                 ║"))
                reporter.info("╚══════════════════════════════════════════════════════════════╝")
                for row in build_decision_summary(model_results):
                    cap_text = f"{row['capability']:.1f}/100" if row["capability"] is not None else "N/A"
                    reporter.info(f"★ {self.text('Model', '模型')}：{row['model']}")
                    reporter.info(f"  {self.text('Rating', '评级')}：{row['rating']}")
                    reporter.info(f"  {self.text('Capability', '能力分')}：{cap_text}    {self.text('Availability', '可用性')}：{row['availability']:.0f}%")
                    reporter.info(f"  {self.text('Authenticity', '真实性')}：{row['authenticity']}")
                    reporter.info(f"  {self.text('Recommendation', '建议')}：{row['recommendation']}")
                reporter.info(self.text("Next: r report | d details | e errors | s save", "下一步：r 报告 | d 详情 | e 错误 | s 保存"))
            if self.state["save_report"] and model_results:
                md_path, json_path = write_reports(cfg.output_dir, model_results, cfg.api)
                self.last_saved_paths = (md_path, json_path)
                self.last_report_unsaved = False
                reporter.section(self.text("Report saved", "报告已保存"))
                reporter.info(f"Markdown: {md_path}")
                reporter.info(f"JSON: {json_path}")
            elif model_results:
                self.last_report_unsaved = True
                reporter.section(self.text("Report not saved", "报告未保存"))
                reporter.info(self.text("Save report file is off. Press s after completion to save this report.", "未开启自动保存。检测完成后可按 s 保存本次报告。"))
            else:
                self.last_report_unsaved = False
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            reporter.info(self.text(f"Audit failed: {error}", f"检测失败：{error}"))
            reporter.info(self.text("Suggestions:", "下一步建议："))
            reporter.info(format_suggestions(error))
            self.queue.put(
                "__ERROR__ "
                + json.dumps(
                    {"probe_id": "audit", "title": "Audit run", "error": error, "suggestions": suggest_next_steps(error)},
                    ensure_ascii=False,
                )
            )
        finally:
            self.queue.put("__DONE__")

    def remember_latest_report(self, model_results: dict[str, list[ProbeResult]], cfg: AuditConfig) -> None:
        self.last_report_text = build_report(model_results, cfg.api)
        self.last_model_results = model_results
        self.last_report_config = cfg.api
        self.last_report_output_dir = cfg.output_dir

    def preflight(self) -> str | None:
        """F9 前快速校验，避免到线程里才报错。返回错误文案或 None。"""
        missing = []
        if not str(self.state["base_url"]).strip():
            missing.append(self.text("Base URL", "中转站地址"))
        if not str(self.state["api_key"]).strip():
            missing.append(self.text("API Key", "API 密钥"))
        if not str(self.state["model"]).strip():
            missing.append(self.text("Model", "模型"))
        if missing:
            return self.text("Please fill ", "请先填写 ") + ", ".join(missing)
        if "," in str(self.state["model"]):
            return self.text("TUI runs one model at a time. Please keep only one Model.", "TUI 每次只检测一个模型，请只保留一个模型。")
        for key in ("timeout", "max_tokens", "temperature", "limit"):
            err = validate_field(key, str(self.state[key]))
            if err:
                return f"{key}: {err}"
        filter_value = str(self.state["model_filter"]).strip()
        if filter_value:
            try:
                re.compile(filter_value)
            except re.error as exc:
                return self.text(f"model_filter: Invalid regex: {exc}", f"模型筛选：正则无效：{exc}")
        try:
            cfg = AuditConfig.from_namespace(make_namespace_from_tui(self.state))
            if not filter_models(cfg.models, cfg.model_filter, cfg.limit):
                return "Model filter/limit 排除了当前 Model，请调整筛选或清空它。"
        except ValueError as exc:
            return str(exc)
        return None

    def save_latest_report(self) -> None:
        if self.last_saved_paths:
            md_path, json_path = self.last_saved_paths
            self.add_log(self.text("Report already saved.", "报告已保存过。"))
            self.add_log(f"Markdown: {md_path}")
            self.add_log(f"JSON: {json_path}")
            return
        if not self.last_model_results or not self.last_report_config:
            self.add_log(self.text("No completed report is available to save yet.", "还没有可保存的已完成报告。"))
            return
        output_dir = self.last_report_output_dir or str(self.state["output_dir"]).strip() or "reports"
        try:
            md_path, json_path = write_reports(output_dir, self.last_model_results, self.last_report_config)
        except Exception as exc:  # noqa: BLE001
            self.add_log(self.text(f"Save failed: {exc}", f"保存失败：{exc}"))
            self.add_log(self.text("Suggestions:", "下一步建议："))
            for suggestion in suggest_next_steps(str(exc)):
                self.add_log(f"  - {suggestion}")
            return
        self.last_saved_paths = (md_path, json_path)
        self.last_report_unsaved = False
        self.add_log(self.text("Report saved", "报告已保存"))
        self.add_log(f"Markdown: {md_path}")
        self.add_log(f"JSON: {json_path}")

    # ---- popups ---------------------------------------------------------

    def confirm_run_popup(self) -> bool:
        cfg = AuditConfig.from_namespace(make_namespace_from_tui(self.state))
        models = filter_models(cfg.models, cfg.model_filter, cfg.limit)
        estimate = build_run_estimate(cfg, models)
        if self.zh():
            request_count = estimate.probe_requests + 1
            per_model = ", ".join(f"{model}: {estimate.probes_by_model[model]}" for model in estimate.models)
            if estimate.estimated_cost is None:
                cost_line = "官方价格预估：未知（部分模型不在内置官方价格表中）"
            else:
                cost_line = f"官方价格预估上限：{format_usd(estimate.estimated_cost)}"
            cost_parts = []
            for model in estimate.models:
                item = estimate.cost_by_model[model]
                total = item.total_cost
                if total is None:
                    cost_parts.append(f"{model}: unknown")
                else:
                    cost_parts.append(f"{model}: {format_usd(total)}（{item.pricing_label}）")
            lines = [
                "确认开始检测？",
                "",
                f"模型数：{estimate.model_count}",
                f"探针请求数：{estimate.probe_requests}（另有 1 次 TUI 连通性测试）",
                f"总 API 请求数：{request_count}",
                f"预估输入 token：{estimate.estimated_input_tokens}",
                f"最大输出 token 预算：{estimate.max_output_tokens}",
                cost_line,
                f"每个模型探针数：{per_model}",
                "每个模型费用：" + "，".join(cost_parts),
                f"输出目录：{os.path.abspath(cfg.output_dir)}",
                f"自动保存报告：{'是' if self.state['save_report'] else '否'}",
            ]
        else:
            lines = ["Run this audit?", ""] + format_run_estimate(
                estimate, cfg.output_dir, bool(self.state["save_report"]), include_ping=True
            )
        height, width = self.stdscr.getmaxyx()
        wrapped: list[str] = []
        max_text_width = max(30, min(76, width - 8))
        for line in lines:
            wrapped.extend(wrap_for_width(line, max_text_width) if line else [""])
        win_height = min(max(9, len(wrapped) + 4), max(7, height - 4))
        win_width = min(max(50, max((len(line) for line in wrapped), default=0) + 6), max(20, width - 4))
        y = max(1, (height - win_height) // 2)
        x = max(0, (width - win_width) // 2)
        win = curses.newwin(win_height, win_width, y, x)
        win.keypad(True)
        offset = 0
        while True:
            win.erase()
            with contextlib.suppress(curses.error):
                win.border()
            visible = win_height - 4
            max_scroll = max(0, len(wrapped) - visible)
            offset = min(offset, max_scroll)
            for row, line in enumerate(wrapped[offset : offset + visible]):
                attr = curses.A_BOLD if row == 0 and offset == 0 else curses.A_NORMAL
                self._safe_addstr(win, 1 + row, 2, line, attr)
            self._safe_addstr(win, win_height - 2, 2, self.text("Enter run | Esc cancel | PgUp/PgDn scroll", "Enter 开始 | Esc 取消 | PgUp/PgDn 滚动"))
            win.refresh()
            ch = win.getch()
            if ch in (10, 13):
                return True
            if ch == 27:
                self.add_log(self.text("Run cancelled before start.", "已在开始前取消运行。"))
                return False
            if ch == curses.KEY_NPAGE:
                offset = min(max_scroll, offset + visible)
            elif ch == curses.KEY_PPAGE:
                offset = max(0, offset - visible)

    def edit_field(self, key: str, label: str) -> None:
        if isinstance(self.state[key], bool):
            self.state[key] = not self.state[key]
            return
        if key == "api_style":
            styles = ["auto", "openai-responses", "openai-chat", "anthropic", "gemini"]
            current = str(self.state[key]).lower()
            self.state[key] = styles[(styles.index(current) + 1) % len(styles)] if current in styles else "auto"
            return
        if key == "mode":
            modes = ["quick", "standard", "full"]
            current = str(self.state[key]).lower()
            self.state[key] = modes[(modes.index(current) + 1) % len(modes)] if current in modes else "standard"
            return
        # 编辑弹窗需要显示光标，便于确认当前插入/删除位置。
        curses.curs_set(1)
        height, width = self.stdscr.getmaxyx()
        prompt = f"{self.field_label(key, label)}: "
        value = str(self.state[key])
        win_width = max(40, min(width - 4, 100))
        win_height = min(6, max(5, height - 2))
        y = max(1, height // 2 - 3)
        x = max(0, (width - win_width) // 2)
        edit_win = curses.newwin(win_height, win_width, y, x)
        edit_win.keypad(True)
        pos = len(value)
        while True:
            error = validate_field(key, value)
            edit_win.erase()
            with contextlib.suppress(curses.error):
                edit_win.border()
            display = value
            if key == "api_key" and display:
                display = "*" * len(value)
            max_value_width = max(1, win_width - display_width(prompt) - 5)
            view_start = 0
            while view_start < pos and display_width(display[view_start:pos]) > max_value_width:
                view_start += 1
            shown = truncate_display(display[view_start:], max_value_width)
            cursor_x = 2 + display_width(prompt) + display_width(display[view_start:pos])
            cursor_x = min(max(2, cursor_x), win_width - 2)
            self._safe_addstr(edit_win, 1, 2, self.text("Edit field", "编辑字段"))
            self._safe_addstr(edit_win, 2, 2, prompt + shown)
            self._safe_addstr(edit_win, 3, 2, self.text("Enter save | Esc cancel | Ctrl+U clear", "Enter 保存 | Esc 取消 | Ctrl+U 清空"))
            if error:
                self._safe_addstr(edit_win, 4, 2, f"! {error}", _curses_attr(4))
            with contextlib.suppress(curses.error):
                edit_win.move(2, cursor_x)
            edit_win.refresh()
            ch = edit_win.getch()
            if ch in (10, 13):
                if error:
                    continue  # 校验未过，保持编辑
                self.state[key] = value
                break
            if ch == 27:
                break
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                if pos > 0:
                    value = value[: pos - 1] + value[pos:]
                    pos -= 1
            elif ch == curses.KEY_LEFT:
                pos = max(0, pos - 1)
            elif ch == curses.KEY_RIGHT:
                pos = min(len(value), pos + 1)
            elif ch == 21:
                value = ""
                pos = 0
            elif 32 <= ch <= 126:
                value = value[:pos] + chr(ch) + value[pos:]
                pos += 1
        curses.curs_set(0)

    def select_model_popup(self, models: list[str]) -> None:
        if not models:
            self.add_log(self.text("No models available to select.", "没有可选择的模型。"))
            return
        height, width = self.stdscr.getmaxyx()
        win_height = min(max(10, min(len(models), 12) + 5), max(7, height - 4))
        win_width = min(max(58, max(len(model) for model in models) + 6), max(20, width - 4))
        y = max(1, (height - win_height) // 2)
        x = max(0, (width - win_width) // 2)
        win = curses.newwin(win_height, win_width, y, x)
        win.keypad(True)
        selected_model = 0
        offset = 0
        search = ""
        filtered = list(models)
        while True:
            lowered = search.lower()
            filtered = [model for model in models if lowered in model.lower()] if lowered else list(models)
            if not filtered:
                selected_model = 0
                offset = 0
            else:
                selected_model = min(selected_model, len(filtered) - 1)
            win.erase()
            with contextlib.suppress(curses.error):
                win.border()
            self._safe_addstr(win, 1, 2, self.text("Select one model", "选择一个模型"), curses.A_BOLD)
            self._safe_addstr(win, 2, 2, f"{self.text('Search', '搜索')}: {search} | {len(filtered)}/{len(models)}")
            visible = win_height - 6
            if filtered:
                if selected_model < offset:
                    offset = selected_model
                if selected_model >= offset + visible:
                    offset = selected_model - visible + 1
                offset = min(offset, max(0, len(filtered) - visible))
                for row, model in enumerate(filtered[offset : offset + visible]):
                    idx = offset + row
                    attr = curses.A_REVERSE if idx == selected_model else curses.A_NORMAL
                    shown = model
                    if len(shown) > win_width - 5:
                        shown = shown[: win_width - 6] + "~"
                    self._safe_addstr(win, 3 + row, 2, shown, attr)
            else:
                self._safe_addstr(win, 3, 2, self.text("No matches. Backspace or Ctrl+U to change search.", "没有匹配项。Backspace 删除，Ctrl+U 清空搜索。"), _curses_attr(4))
            self._safe_addstr(win, win_height - 2, 2, self.text("Type search | Enter choose | Esc cancel | Ctrl+U clear", "输入搜索 | Enter 选择 | Esc 取消 | Ctrl+U 清空"))
            win.refresh()
            ch = win.getch()
            if ch in (10, 13):
                if filtered:
                    self.state["model"] = filtered[selected_model]
                    self.add_log(self.text(f"Selected model: {filtered[selected_model]}", f"已选择模型：{filtered[selected_model]}"))
                break
            if ch == 27:
                self.add_log(self.text("Model selection cancelled.", "已取消模型选择。"))
                break
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                search = search[:-1]
                selected_model = 0
                offset = 0
            elif ch == 21:
                search = ""
                selected_model = 0
                offset = 0
            elif ch == curses.KEY_UP:
                selected_model = max(0, selected_model - 1)
            elif ch == curses.KEY_DOWN:
                selected_model = min(max(0, len(filtered) - 1), selected_model + 1)
            elif ch == curses.KEY_NPAGE:
                selected_model = min(max(0, len(filtered) - 1), selected_model + visible)
            elif ch == curses.KEY_PPAGE:
                selected_model = max(0, selected_model - visible)
            elif 32 <= ch <= 126:
                search += chr(ch)
                selected_model = 0
                offset = 0

    # ---- drawing --------------------------------------------------------

    def _status_text(self) -> str:
        status_names = {
            "Idle": self.text("Idle", "空闲"),
            "Running": self.text("Running", "运行中"),
            "Cancelling": self.text("Cancelling", "取消中"),
            "Selecting": self.text("Selecting", "选择中"),
        }
        if self.status in {"Running", "Cancelling", "Selecting"}:
            frame = self.spinner_frames[int(time.monotonic() * 8) % len(self.spinner_frames)]
            if self.status == "Running" and self.progress is not None:
                done, total = self.progress
                return f"{frame} {status_names['Running']} {done}/{total}"
            return f"{frame} {status_names.get(self.status, self.status)}"
        return status_names.get(self.status, self.status)

    def zh(self) -> bool:
        return self.language == "zh"

    def text(self, en: str, zh: str) -> str:
        return zh if self.zh() else en

    def field_label(self, key: str, fallback: str) -> str:
        return self.field_labels_zh.get(key, fallback) if self.zh() else fallback

    def help_for(self, key: str) -> tuple[str, str]:
        source = self.help_text_zh if self.zh() else self.help_text
        return source.get(key, ("", ""))

    def toggle_language(self) -> None:
        self.language = "en" if self.zh() else "zh"
        self.add_log("Language switched to English." if not self.zh() else "界面语言已切换为中文。")

    def command_lines(self) -> list[str]:
        if self.zh():
            return [
                "g 中/英  l 日志  r 报告",
                "d 详情  e 错误  s 保存  x 清空",
                "F5 拉模型  F9 运行  q 退出",
            ]
        return [
            "g lang  l log  r report",
            "d details  e errors  s save  x clear",
            "F5 fetch  F9 run  q quit",
        ]

    def family_summary_text(self) -> str:
        summary = model_family_summary(str(self.state["model"]))
        if self.zh():
            return {"empty": "未填写", "multiple not allowed": "多个模型不支持"}.get(summary, summary)
        return summary

    def _view_title(self) -> str:
        titles = {
            "log": self.text("Live Log", "实时日志"),
            "report": self.text("Report", "完整报告"),
            "details": self.text("Probe Details", "探针详情"),
            "errors": self.text("Errors", "错误建议"),
        }
        return titles.get(self.view_mode, titles["log"])

    def _progress_bar(self, width: int) -> str:
        if self.progress is None or width <= 4:
            return ""
        done, total = self.progress
        if total <= 0:
            return ""
        fill_width = max(1, width - 2)
        filled = min(fill_width, max(0, int(fill_width * done / total)))
        return "[" + "█" * filled + "░" * (fill_width - filled) + "]"

    def draw(self) -> None:
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        title_attr = _curses_attr(1, bold=True)
        ok_attr = _curses_attr(2)
        warn_attr = _curses_attr(3)

        if height < self.MIN_HEIGHT or width < self.MIN_WIDTH:
            self._safe_addstr(self.stdscr, 0, 0, self.text("Terminal too small.", "终端窗口太小。"))
            self._safe_addstr(self.stdscr, 1, 0, self.text(f"Need at least {self.MIN_WIDTH} x {self.MIN_HEIGHT}.", f"至少需要 {self.MIN_WIDTH} x {self.MIN_HEIGHT}。"))
            self._safe_addstr(self.stdscr, 2, 0, self.text("Resize the window. (q quits)", "请放大窗口。（q 退出）"))
            with contextlib.suppress(curses.error):
                self.stdscr.refresh()
            return

        self._safe_addstr(self.stdscr, 0, 2, "AI Relay Audit TUI", title_attr)
        repo_text = self.text("Author: maya1900 | Git: local ai-relay-audit", "作者: maya1900 | Git: 本地 ai-relay-audit")
        self._safe_addstr(self.stdscr, 0, max(20, width - display_width(repo_text) - 2), repo_text, warn_attr)
        with contextlib.suppress(curses.error):
            self.stdscr.hline(1, 0, "-", width)

        left_width = min(54, max(34, width // 3))
        self._safe_addstr(self.stdscr, 2, 2, self.text("Configuration", "配置"), title_attr)
        for idx, (key, label) in enumerate(self.fields):
            y = 4 + idx
            if y >= height - 2:
                break
            is_selected = idx == self.selected
            selected_attr = curses.A_REVERSE if is_selected else curses.A_NORMAL
            value = self.state[key]
            if isinstance(value, bool):
                shown = "[x]" if value else "[ ]"
            elif key == "api_key" and value:
                shown = "*" * min(len(str(value)), 18)
            else:
                shown = str(value)
            shown_label = self.field_label(key, label)
            label_width = 18
            value_width = max(8, left_width - 23)
            if display_width(shown) > value_width:
                shown = truncate_display(shown, value_width - 1) + "~"
            if is_selected:
                self._safe_addstr(self.stdscr, y, 2, " " * max(1, left_width - 4), selected_attr)
            self._safe_addstr(self.stdscr, y, 2, pad_display(shown_label, label_width), selected_attr)
            self._safe_addstr(self.stdscr, y, 21, pad_display(shown, value_width), selected_attr)

        summary_y = 5 + len(self.fields)
        if summary_y < height - 3:
            self._safe_addstr(self.stdscr, summary_y, 2, self.text("Detected family", "识别家族"), title_attr)
            summary = self.family_summary_text()
            if display_width(summary) > left_width - 22:
                summary = truncate_display(summary, left_width - 23) + "~"
            self._safe_addstr(self.stdscr, summary_y, 21, summary, ok_attr)

        help_y = summary_y + 2
        if help_y < height - 7:
            key, _ = self.fields[self.selected]
            description, next_step = self.help_for(key)
            available = height - help_y - 2
            desc_lines = wrap_for_width(description, left_width - 4)
            next_lines = wrap_for_width(next_step, left_width - 4)
            self._safe_addstr(self.stdscr, help_y, 2, self.text("Help", "帮助"), title_attr)
            desc_limit = max(1, min(4, available // 3))
            for row, text in enumerate(desc_lines[:desc_limit]):
                self._safe_addstr(self.stdscr, help_y + 1 + row, 2, text)
            next_y = help_y + 2 + desc_limit
            if next_y < height - 5:
                self._safe_addstr(self.stdscr, next_y, 2, self.text("Next", "下一步"), title_attr)
                next_limit = max(1, min(3, height - next_y - 5))
                for row, text in enumerate(next_lines[:next_limit]):
                    self._safe_addstr(self.stdscr, next_y + 1 + row, 2, text, warn_attr)
            command_y = max(next_y + 2 + min(len(next_lines), max(1, min(3, height - next_y - 5))), height - 6)
            if command_y < height - 1:
                self._safe_addstr(self.stdscr, command_y, 2, self.text("Keys", "快捷键"), title_attr)
                for row, text in enumerate(self.command_lines()):
                    if command_y + 1 + row < height - 1:
                        self._safe_addstr(self.stdscr, command_y + 1 + row, 2, text, warn_attr)

        status_attr = ok_attr if self.status == "Idle" else warn_attr
        status_text = self.text("Status", "状态") + f": {self._status_text()}"
        self._safe_addstr(self.stdscr, height - 1, 2, status_text, status_attr)
        footer = self.text("Enter edit/toggle, Tab next, arrows move", "Enter 编辑/切换，Tab 下一个，方向键移动")
        self._safe_addstr(self.stdscr, height - 1, max(20, width - display_width(footer) - 2), footer)

        log_x = left_width + 2
        log_width = width - log_x - 2
        if log_width > 20:
            with contextlib.suppress(curses.error):
                self.stdscr.vline(2, left_width, "|", height - 4)
            scroll_hint = "" if self.log_scroll == 0 else f" (scrolled {self.log_scroll})"
            self._safe_addstr(self.stdscr, 2, log_x, f"{self._view_title()}{scroll_hint}", title_attr)
            if self.worker_running() and self.progress is not None:
                bar = self._progress_bar(max(8, min(30, log_width - 20)))
                if bar:
                    self._safe_addstr(self.stdscr, 2, max(log_x + 18, width - len(bar) - 2), bar, ok_attr)
            visible_height = height - 5
            all_log_lines = wrapped_log_lines(self.view_lines(), log_width - 1)
            max_start = max(0, len(all_log_lines) - visible_height)
            start = max(0, max_start - self.log_scroll)
            visible_logs = all_log_lines[start : start + visible_height]
            for i, clean in enumerate(visible_logs):
                attr = curses.A_NORMAL
                if clean.startswith(("✓", "★", "Next:")):
                    attr = ok_attr
                elif clean.startswith("✗") or clean.startswith("Audit failed") or clean.startswith("Fetch failed"):
                    attr = _curses_attr(4)
                elif clean.startswith(("▶", "◇", "╔", "║", "╚")):
                    attr = warn_attr | curses.A_BOLD
                self._safe_addstr(self.stdscr, 4 + i, log_x, clean, attr)
        with contextlib.suppress(curses.error):
            self.stdscr.refresh()

    # ---- event loop -----------------------------------------------------

    def drain_queue(self) -> None:
        while True:
            try:
                line = self.queue.get_nowait()
            except queue.Empty:
                break
            if line == "__DONE__":
                self.status = "Idle"
            elif line == "__SELECT_MODEL__":
                self.status = "Selecting"
                self.draw()
                self.select_model_popup(self.fetched_models)
                self.status = "Idle"
            elif line.startswith("__PROGRESS__"):
                with contextlib.suppress(ValueError):
                    done, total = line.split(" ", 1)[1].split("/", 1)
                    self.progress = (int(done), int(total))
            elif line.startswith("__DETAIL__ "):
                with contextlib.suppress(json.JSONDecodeError):
                    item = json.loads(line.split(" ", 1)[1])
                    if isinstance(item, dict):
                        self.detail_items.append(item)
            elif line.startswith("__ERROR__ "):
                with contextlib.suppress(json.JSONDecodeError):
                    item = json.loads(line.split(" ", 1)[1])
                    if isinstance(item, dict):
                        self.error_items.append(item)
            elif line:
                self.add_log(line)

    def request_cancel(self) -> None:
        if self.worker_running():
            self.cancel_event.set()
            self.status = "Cancelling"
            self.add_log("正在取消检测……（当前探针结束后停止）")

    def handle_key(self, ch: int) -> bool:
        """处理一个按键；返回 False 表示请求退出。"""
        if ch in (ord("q"), ord("Q")) and not self.worker_running():
            return False
        if self.worker_running() and ch in (27, ord("c"), ord("C")):
            self.request_cancel()
            return True
        if ch in (ord("g"), ord("G")):
            self.toggle_language()
        elif ch in (ord("l"), ord("L")):
            self.set_view("log")
        elif ch in (ord("r"), ord("R")):
            self.set_view("report")
        elif ch in (ord("d"), ord("D")):
            self.set_view("details")
        elif ch in (ord("e"), ord("E")):
            self.set_view("errors")
        elif ch in (ord("x"), ord("X")):
            self.clear_logs()
        elif not self.worker_running() and ch in (ord("s"), ord("S")):
            self.save_latest_report()
        elif ch in (9, curses.KEY_DOWN):
            self.selected = (self.selected + 1) % len(self.fields)
        elif ch in (curses.KEY_BTAB, curses.KEY_UP):
            self.selected = (self.selected - 1) % len(self.fields)
        elif ch in (10, 13):
            key, label = self.fields[self.selected]
            self.edit_field(key, label)
        elif ch == curses.KEY_F5:
            self.start_worker(self.fetch_models_worker)
        elif ch == curses.KEY_F9:
            err = self.preflight()
            if err:
                self.add_log("无法开始：" + err)
                self.add_log("下一步建议:")
                for suggestion in suggest_next_steps(err):
                    self.add_log(f"  - {suggestion}")
            elif self.confirm_run_popup():
                self.start_worker(self.audit_worker)
                self.set_log_bottom()
        elif ch == curses.KEY_NPAGE:
            self.scroll_log(-10)
        elif ch == curses.KEY_PPAGE:
            self.scroll_log(10)
        elif ch == curses.KEY_HOME:
            self.scroll_log(100000)
        elif ch == curses.KEY_END:
            self.set_log_bottom()
        return True

    def run(self) -> int:
        curses.curs_set(0)
        self.stdscr.keypad(True)
        curses.use_default_colors()
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, curses.COLOR_RED, -1)

        while True:
            self.drain_queue()
            self.draw()
            self.stdscr.timeout(150)
            ch = self.stdscr.getch()
            if ch == -1:
                continue
            if not self.handle_key(ch):
                return 0


def _curses_attr(pair: int, bold: bool = False) -> int:
    """颜色对编号转 curses 属性；无色终端退化为普通/加粗。"""
    if curses is not None and curses.has_colors():
        attr = curses.color_pair(pair)
        return attr | curses.A_BOLD if bold else attr
    return curses.A_BOLD if bold and curses is not None else 0
