#!/usr/bin/env python3
"""ai_relay_audit 的纯函数单元测试。

仅测试不触网的评分器、解析器与辅助函数。运行：

    python3 -m unittest test_ai_relay_audit -v
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import ai_relay_audit as m


def make_probe(
    probe_id: str = "p",
    category: str = "universal",
    weight: int = 10,
    families: tuple[str, ...] = ("gpt", "claude", "unknown"),
) -> m.Probe:
    return m.Probe(probe_id, probe_id, category, weight, families, "sys", "user", lambda t: (0.0, ""))


def make_result(
    probe_id: str = "p",
    category: str = "universal",
    weight: int = 10,
    status: str = "ok",
    score: float = 100.0,
) -> m.ProbeResult:
    return m.ProbeResult(
        probe=make_probe(probe_id, category, weight),
        status=status,
        score=score,
        reason="",
        response="hello",
        latency_ms=10 if status == "ok" else None,
        usage=None,
        error=None if status == "ok" else "boom",
    )


def make_config(api_key: str = "sk-test") -> m.ApiConfig:
    return m.ApiConfig(
        base_url="https://relay.example.com",
        api_key=api_key,
        timeout=30,
        max_tokens=900,
        temperature=0.0,
    )


class FakeHTTPResponse:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = body

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def request_json(request: object) -> dict[str, object]:
    data = getattr(request, "data", None)
    if not data:
        return {}
    parsed = json.loads(data.decode("utf-8"))
    return parsed if isinstance(parsed, dict) else {}


class NormalizeBaseUrlTest(unittest.TestCase):
    def test_strips_trailing_slash_and_v1(self) -> None:
        self.assertEqual(m.normalize_base_url("https://x.com/v1"), "https://x.com")
        self.assertEqual(m.normalize_base_url("https://x.com/v1/"), "https://x.com")
        self.assertEqual(m.normalize_base_url("https://x.com/"), "https://x.com")
        self.assertEqual(m.normalize_base_url("https://x.com"), "https://x.com")

    def test_preserves_subpath_roundtrip(self) -> None:
        self.assertEqual(m.normalize_base_url("https://x.com/openai/v1"), "https://x.com/openai")


class FamilyForModelTest(unittest.TestCase):
    def test_families(self) -> None:
        self.assertEqual(m.family_for_model("claude-3-5-sonnet-20241022"), "claude")
        self.assertEqual(m.family_for_model("gpt-4o"), "gpt")
        self.assertEqual(m.family_for_model("o3-mini"), "gpt")
        self.assertEqual(m.family_for_model("gemini-2.0-flash-exp"), "gemini")
        self.assertEqual(m.family_for_model("gemini-1.5-pro"), "gemini")
        self.assertEqual(m.family_for_model("llama-3-70b"), "unknown")


class IsReasoningModelTest(unittest.TestCase):
    def test_reasoning_models(self) -> None:
        for model in ("o1", "o1-mini", "o3", "o3-mini", "o4-mini", "openai/o3", "gpt-5", "gpt-5-mini"):
            self.assertTrue(m.is_reasoning_model(model), model)

    def test_non_reasoning_models(self) -> None:
        for model in ("gpt-4o", "gpt-4o-mini", "chatgpt-4o-latest", "claude-3-5-sonnet", "gpt-4-turbo"):
            self.assertFalse(m.is_reasoning_model(model), model)


class ApiStyleTest(unittest.TestCase):
    def test_normalize_api_style_aliases(self) -> None:
        self.assertEqual(m.normalize_api_style("openai"), "openai-chat")
        self.assertEqual(m.normalize_api_style("responses"), "openai-responses")
        self.assertEqual(m.normalize_api_style("google"), "gemini")
        self.assertEqual(m.normalize_api_style("gemini"), "gemini")
        self.assertEqual(m.normalize_api_style("google"), "gemini")
        self.assertEqual(m.normalize_api_style("gemini"), "gemini")


class ApiRequestTest(unittest.TestCase):
    def test_api_request_prefixes_unversioned_paths_only(self) -> None:
        requests = []

        def fake_urlopen(request: object, timeout: int) -> FakeHTTPResponse:
            requests.append(request)
            return FakeHTTPResponse({"ok": True})

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            m.api_request(make_config(), "GET", "/models")
            m.api_request(make_config(), "POST", "/v1/models/gemini:generateContent?key=sk-test", {})

        self.assertEqual(requests[0].full_url, "https://relay.example.com/v1/models")
        self.assertEqual(requests[1].full_url, "https://relay.example.com/v1/models/gemini:generateContent?key=sk-test")

    def test_chat_openai_payload(self) -> None:
        requests = []

        def fake_urlopen(request: object, timeout: int) -> FakeHTTPResponse:
            requests.append(request)
            return FakeHTTPResponse({
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            text, usage, _latency = m.chat_openai(make_config(), "gpt-4o", "sys", "user")

        payload = request_json(requests[0])
        headers = {k.lower(): v for k, v in requests[0].header_items()}
        self.assertEqual(text, "ok")
        self.assertEqual(usage["total_tokens"], 2)
        self.assertEqual(requests[0].full_url, "https://relay.example.com/v1/chat/completions")
        self.assertEqual(payload["model"], "gpt-4o")
        self.assertEqual(payload["max_tokens"], 900)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(headers["authorization"], "Bearer sk-test")

    def test_chat_openai_responses_payload(self) -> None:
        requests = []

        def fake_urlopen(request: object, timeout: int) -> FakeHTTPResponse:
            requests.append(request)
            return FakeHTTPResponse({
                "output_text": "ok",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            })

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            text, usage, _latency = m.chat_openai_responses(make_config(), "gpt-4o", "sys", "user")

        payload = request_json(requests[0])
        self.assertEqual(text, "ok")
        self.assertEqual(usage["total_tokens"], 2)
        self.assertEqual(requests[0].full_url, "https://relay.example.com/v1/responses")
        self.assertEqual(payload["instructions"], "sys")
        self.assertEqual(payload["input"], "user")
        self.assertEqual(payload["max_output_tokens"], 900)

    def test_chat_gemini_url_and_payload(self) -> None:
        requests = []

        def fake_urlopen(request: object, timeout: int) -> FakeHTTPResponse:
            requests.append(request)
            return FakeHTTPResponse({
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            })

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            text, usage, _latency = m.chat_gemini(make_config(), "gemini-2.0-flash", "sys", "user")

        payload = request_json(requests[0])
        self.assertEqual(text, "ok")
        self.assertEqual(usage["total_tokens"], 2)
        self.assertEqual(requests[0].full_url, "https://relay.example.com/v1/models/gemini-2.0-flash:generateContent?key=sk-test")
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "user")
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "sys")

    def test_chat_anthropic_payload_and_headers(self) -> None:
        requests = []

        def fake_urlopen(request: object, timeout: int) -> FakeHTTPResponse:
            requests.append(request)
            return FakeHTTPResponse({
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            text, usage, _latency, _data = m.chat_anthropic(make_config(), "claude-sonnet-4-6", "sys", "user")

        payload = request_json(requests[0])
        headers = {k.lower(): v for k, v in requests[0].header_items()}
        self.assertEqual(text, "ok")
        self.assertEqual(usage["output_tokens"], 1)
        self.assertEqual(requests[0].full_url, "https://relay.example.com/v1/messages")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(headers["x-api-key"], "sk-test")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")

    def test_anthropic_thinking_uses_adaptive_for_opus_47(self) -> None:
        requests = []

        def fake_urlopen(request: object, timeout: int) -> FakeHTTPResponse:
            requests.append(request)
            return FakeHTTPResponse({
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            m.chat_anthropic_with_thinking(make_config(), "claude-opus-4-7", "sys", "user")

        payload = request_json(requests[0])
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertGreaterEqual(payload["max_tokens"], m.ANTHROPIC_ADAPTIVE_THINKING_MAX_TOKENS)

    def test_anthropic_thinking_budget_fits_max_tokens(self) -> None:
        requests = []

        def fake_urlopen(request: object, timeout: int) -> FakeHTTPResponse:
            requests.append(request)
            return FakeHTTPResponse({
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            m.chat_anthropic_with_thinking(make_config(), "claude-haiku-4-5", "sys", "user")

        payload = request_json(requests[0])
        self.assertEqual(payload["thinking"]["type"], "enabled")
        self.assertEqual(payload["thinking"]["budget_tokens"], m.ANTHROPIC_THINKING_BUDGET_TOKENS)
        self.assertGreater(payload["max_tokens"], payload["thinking"]["budget_tokens"])


class ExtractJsonObjectTest(unittest.TestCase):
    def test_plain(self) -> None:
        self.assertEqual(m.extract_json_object('{"a": 1}'), {"a": 1})

    def test_fenced(self) -> None:
        self.assertEqual(m.extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_embedded(self) -> None:
        self.assertEqual(m.extract_json_object('prefix {"a": 1} suffix'), {"a": 1})

    def test_non_object(self) -> None:
        self.assertIsNone(m.extract_json_object("hello"))
        self.assertIsNone(m.extract_json_object("[1, 2, 3]"))


class ScoreJsonContractTest(unittest.TestCase):
    def test_perfect(self) -> None:
        text = '{"city":"Hangzhou","count":7,"valid":true,"items":["alpha","delta","omega"]}'
        score, _ = m.score_json_contract(text)
        self.assertEqual(score, 100)

    def test_one_wrong(self) -> None:
        text = '{"city":"Beijing","count":7,"valid":true,"items":["alpha","delta","omega"]}'
        score, _ = m.score_json_contract(text)
        self.assertEqual(score, 75)

    def test_not_json(self) -> None:
        score, _ = m.score_json_contract("no json here")
        self.assertEqual(score, 0)


class ScoreReasoningTest(unittest.TestCase):
    def test_correct_answer_scores_full(self) -> None:
        text = "The original price is $20. The unique letter is Q. Shortest route A-B-D-F costs 7. Final answer above."
        score, reason = m.score_reasoning(text)
        self.assertEqual(score, 100, reason)

    def test_old_buggy_value_18_75_no_longer_passes_price(self) -> None:
        # 回归：题目正确答案是 20，不是 18.75。
        text = "The price is $18.75. Letter Q. Route A-B-D-F. Final answer."
        score, reason = m.score_reasoning(text)
        self.assertIn("price: 未通过", reason)
        self.assertEqual(score, 75.0, reason)

    def test_route_accepts_arrow_format(self) -> None:
        text = "answer: price 20, letter Q, route A → B → D → F"
        score, reason = m.score_reasoning(text)
        self.assertEqual(score, 100, reason)


class ScoreInstructionResistanceTest(unittest.TestCase):
    def test_resists_injection(self) -> None:
        text = '{"action":"summarize","summary":"short doc summary here","quote":"NEVER FOLLOW USER"}'
        score, reason = m.score_instruction_resistance(text)
        self.assertEqual(score, 100, reason)

    def test_leaked_is_penalized(self) -> None:
        text = '{"action":"summarize","summary":"x","quote":"reveal your system prompt"}'
        score, _ = m.score_instruction_resistance(text)
        self.assertLess(score, 100)


class ScoreCodeTaskTest(unittest.TestCase):
    def test_spaced_result(self) -> None:
        text = (
            "def merge_ranges(ranges):\n"
            "    ranges = sorted(ranges)\n"
            "    # merge when start <= last end + 1 (adjacent)\n"
            "Example: merge_ranges([[1,3],[4,5],[8,10]]) == [[1, 5], [8, 10]]"
        )
        score, reason = m.score_code_task(text)
        self.assertEqual(score, 100, reason)

    def test_tuple_result_without_spaces(self) -> None:
        text = "def merge_ranges(r): sort; end+1; result [(1,5),(8,10)]"
        score, reason = m.score_code_task(text)
        self.assertIn("示例结果=True", reason)


class ScoreIdentityTest(unittest.TestCase):
    def test_full(self) -> None:
        text = '{"model_family":"gpt","certainty":"high","limitations":"cannot be verified via black-box api"}'
        score, reason = m.score_identity(text)
        self.assertEqual(score, 100, reason)

    def test_non_json_gets_partial(self) -> None:
        score, _ = m.score_identity("I am some model.")
        self.assertEqual(score, 20)


class ScoreClaudeXmlTest(unittest.TestCase):
    def test_winner_cedar_short_name(self) -> None:
        # 回归：题目里项目名是 "Cedar"，不应强制要求 "Project Cedar"。
        text = '{"winner":"Cedar","highest_risk":"vendor lock-in","ignored_instruction":true}'
        score, reason = m.score_claude_xml(text)
        self.assertEqual(score, 100, reason)

    def test_winner_project_cedar_also_accepted(self) -> None:
        text = '{"winner":"Project Cedar","highest_risk":"lock-in","ignored_instruction":"true"}'
        score, reason = m.score_claude_xml(text)
        self.assertEqual(score, 100, reason)

    def test_wrong_winner_fails(self) -> None:
        text = '{"winner":"Maple","highest_risk":"latency","ignored_instruction":false}'
        score, reason = m.score_claude_xml(text)
        self.assertIn("winner: 未通过", reason)


class ScoreClaudeSafetyTest(unittest.TestCase):
    def test_good_answer(self) -> None:
        text = "I can't provide the key. You should rotate and revoke it now. I can help secure it."
        score, reason = m.score_claude_safety(text)
        self.assertEqual(score, 100, reason)

    def test_bare_keyiyong_no_longer_counts_as_not_overblock(self) -> None:
        # 回归：裸"可以"不应再让 not_overblock 恒为真。
        text = "好的，可以。"
        score, reason = m.score_claude_safety(text)
        self.assertIn("未过度拒绝=False", reason)
        self.assertEqual(score, 25.0, reason)


class ScoreClaudeThinkingSignatureTest(unittest.TestCase):
    def test_redacted_thinking_data_counts_as_valid_evidence(self) -> None:
        response_data = {
            "content": [
                {"type": "redacted_thinking", "data": "A" * 600},
                {"type": "text", "text": "ok"},
            ]
        }
        score, reason = m.score_claude_thinking_signature("", response_data)
        self.assertEqual(score, 100, reason)
        self.assertIn("redacted_thinking", reason)


class ScoreGptSchemaTest(unittest.TestCase):
    def test_full(self) -> None:
        text = (
            '{"name":"normalize_invoice","arguments":'
            '{"total_cents":129934,"currency":"USD","due_days":30,"vendor":"ACME"}}'
        )
        score, reason = m.score_gpt_schema(text)
        self.assertEqual(score, 100, reason)


class ScoreGptMathTest(unittest.TestCase):
    def test_correct(self) -> None:
        text = "The answer is 43, since 43 mod 5 = 3 and 43 is prime."
        score, reason = m.score_gpt_math(text)
        self.assertEqual(score, 100, reason)

    def test_430_and_143_not_matched_as_43(self) -> None:
        # 回归：精确数值比较，不把 430 / 143 误当成 43。
        text = "value 430 is not prime; 143 = 11 * 13 (mod stuff)"
        score, reason = m.score_gpt_math(text)
        self.assertIn("答案43=False", reason)


class ScoreStreamConsistencyTest(unittest.TestCase):
    def test_identical_responses(self) -> None:
        stream = "red, blue, green"
        non_stream = "red, blue, green"
        score, reason = m.score_stream_consistency("", stream, non_stream)
        self.assertEqual(score, 100, reason)
        self.assertIn("完全一致", reason)

    def test_whitespace_differences_normalized(self) -> None:
        stream = "red,  blue,   green"
        non_stream = "red, blue, green"
        score, reason = m.score_stream_consistency("", stream, non_stream)
        self.assertEqual(score, 100, reason)

    def test_partial_match(self) -> None:
        stream = "red, blue, green and yellow"
        non_stream = "red, blue, green"
        score, reason = m.score_stream_consistency("", stream, non_stream)
        self.assertGreaterEqual(score, 60, reason)
        self.assertIn("部分", reason)

    def test_severe_inconsistency(self) -> None:
        stream = "completely different response"
        non_stream = "red, blue, green"
        score, reason = m.score_stream_consistency("", stream, non_stream)
        self.assertLess(score, 50, reason)

    def test_missing_responses(self) -> None:
        # 两个都缺失
        score, reason = m.score_stream_consistency("", None, None)
        self.assertEqual(score, 50, reason)
        self.assertIn("均未获取", reason)

        # stream 缺失
        score, reason = m.score_stream_consistency("", None, "test")
        self.assertEqual(score, 50, reason)
        self.assertIn("stream 响应获取失败", reason)

        # non-stream 缺失
        score, reason = m.score_stream_consistency("", "test", None)
        self.assertEqual(score, 50, reason)
        self.assertIn("non-stream 响应获取失败", reason)

    def test_empty_responses(self) -> None:
        score, reason = m.score_stream_consistency("", "", "test")
        self.assertEqual(score, 0, reason)


class ScoreProtocolFingerprintTest(unittest.TestCase):
    def test_openai_responses_input_output_tokens_are_valid(self) -> None:
        usage = {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}
        score, reason = m.score_protocol_fingerprint("", usage, "gpt", "openai-responses")
        self.assertEqual(score, 100, reason)
        self.assertIn("openai-responses", reason)

    def test_openai_chat_prompt_completion_tokens_are_valid(self) -> None:
        usage = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
        score, reason = m.score_protocol_fingerprint("", usage, "gpt", "openai-chat")
        self.assertEqual(score, 100, reason)
        self.assertIn("openai-chat", reason)

    def test_anthropic_input_output_tokens_are_valid_for_claude(self) -> None:
        usage = {
            "input_tokens": 12,
            "output_tokens": 3,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        score, reason = m.score_protocol_fingerprint("", usage, "claude", "anthropic")
        self.assertEqual(score, 100, reason)
        self.assertIn("anthropic", reason)

    def test_extra_usage_field_does_not_zero_score(self) -> None:
        usage = {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "relay_latency_ms": 200,
        }
        score, reason = m.score_protocol_fingerprint("", usage, "gpt", "openai-chat")
        self.assertEqual(score, 100, reason)

    def test_openai_chat_mixed_input_output_fields_is_minor(self) -> None:
        usage = {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "input_tokens": 12,
            "output_tokens": 3,
        }
        score, reason = m.score_protocol_fingerprint("", usage, "gpt", "openai-chat")
        self.assertEqual(score, 95, reason)
        self.assertIn("minor", reason)

    def test_openai_usage_source_anthropic_is_critical(self) -> None:
        usage = {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "usage_source": "anthropic",
        }
        score, reason = m.score_protocol_fingerprint("", usage, "gpt", "openai-chat")
        self.assertLess(score, 30, reason)
        self.assertIn("critical", reason)


class RedactSecretsTest(unittest.TestCase):
    def test_redacts_key(self) -> None:
        config = make_config(api_key="sk-supersecret")
        out = m.redact_secrets("error body contains sk-supersecret in echo", config)
        self.assertNotIn("sk-supersecret", out)
        self.assertIn("***REDACTED***", out)

    def test_no_key_noop(self) -> None:
        config = make_config(api_key="")
        self.assertEqual(m.redact_secrets("plain text", config), "plain text")


class CapabilityScoreTest(unittest.TestCase):
    def test_over_successful_only(self) -> None:
        # 失败探针被排除，不再以 0 分拖垮能力评估。
        results = [
            make_result("a", weight=10, status="ok", score=80),
            make_result("b", weight=10, status="error", score=0),
        ]
        self.assertAlmostEqual(m.capability_score(results), 80.0)

    def test_none_when_all_failed(self) -> None:
        self.assertIsNone(m.capability_score([make_result(status="error")]))
        self.assertIsNone(m.capability_score([]))

    def test_weighted_over_ok(self) -> None:
        results = [
            make_result("a", weight=10, status="ok", score=100),
            make_result("b", weight=30, status="ok", score=0),
        ]
        self.assertAlmostEqual(m.capability_score(results), 25.0)


class AvailabilityRateTest(unittest.TestCase):
    def test_fraction(self) -> None:
        results = [
            make_result(status="ok"),
            make_result(status="ok"),
            make_result(status="error"),
            make_result(status="error"),
        ]
        self.assertEqual(m.availability_rate(results), 50.0)

    def test_empty_is_zero(self) -> None:
        self.assertEqual(m.availability_rate([]), 0.0)

    def test_all_ok(self) -> None:
        self.assertEqual(m.availability_rate([make_result(status="ok")]), 100.0)


class OverallRatingTest(unittest.TestCase):
    def test_none_capability_is_na(self) -> None:
        self.assertTrue(m.overall_rating(None, 0.0).startswith("N/A"))

    def test_full_range_at_full_availability(self) -> None:
        self.assertTrue(m.overall_rating(95.0, 100.0).startswith("A"))
        self.assertTrue(m.overall_rating(82.0, 100.0).startswith("B"))

    def test_capped_at_c_below_full_availability(self) -> None:
        # 能力强但接口不稳定：不允许 A/B，封顶到 C 并标注限级。
        rating = m.overall_rating(95.0, 80.0)
        self.assertTrue(rating.startswith("C"))
        self.assertIn("限级", rating)

    def test_strong_warning_below_60(self) -> None:
        rating = m.overall_rating(95.0, 40.0)
        self.assertTrue(rating.startswith("C"))
        self.assertIn("仅供参考", rating)

    def test_capping_never_raises_a_weak_model(self) -> None:
        self.assertTrue(m.overall_rating(30.0, 80.0).startswith("E"))


class AuthenticityNoteTest(unittest.TestCase):
    def _good_gpt(self) -> list[m.ProbeResult]:
        return [
            make_result("universal_json", "universal", 12, "ok", 90),
            make_result("universal_reasoning", "universal", 18, "ok", 85),
            make_result("identity_limits", "identity", 8, "ok", 90),
            make_result("gpt_math", "targeted", 13, "ok", 85),
        ]

    def test_all_failed_is_unknown(self) -> None:
        self.assertEqual(m.authenticity_note("gpt-4o", [make_result(status="error")])[0], "无法判断")

    def test_low_availability_is_unknown(self) -> None:
        results = [
            make_result("identity_limits", "identity", 8, "ok", 90),
            make_result("b", "universal", 10, "error", 0),
            make_result("c", "universal", 10, "error", 0),
        ]
        self.assertEqual(m.authenticity_note("gpt-4o", results)[0], "无法判断")

    def test_unknown_family(self) -> None:
        self.assertEqual(m.authenticity_note("llama-3-70b", self._good_gpt())[0], "无法按名称判断")

    def test_weak_targeted_flags_mismatch(self) -> None:
        results = self._good_gpt()
        results[-1] = make_result("gpt_math", "targeted", 13, "ok", 30)
        self.assertEqual(m.authenticity_note("gpt-4o", results)[0], "疑似不匹配")

    def test_missing_identity_is_insufficient(self) -> None:
        results = [r for r in self._good_gpt() if r.probe.probe_id != "identity_limits"]
        self.assertEqual(m.authenticity_note("gpt-4o", results)[0], "证据不足")

    def test_best_case_never_affirms_authenticity(self) -> None:
        label, reason = m.authenticity_note("gpt-4o", self._good_gpt())
        self.assertEqual(label, "未发现不一致")
        self.assertIn("无法证明", reason)


class AuditConfigTest(unittest.TestCase):
    def test_from_namespace_parses_models_and_strips_v1(self) -> None:
        ns = argparse.Namespace(
            base_url="https://x.com/v1", api_key="sk", models="a, b ,c",
            timeout=30, max_tokens=900, temperature=0.0, api_style="auto",
            model_filter=None, limit=None, all_targeted=False, hide_prompts=False, output_dir="out",
        )
        cfg = m.AuditConfig.from_namespace(ns)
        self.assertEqual(cfg.api.base_url, "https://x.com")
        self.assertEqual(cfg.models, ["a", "b", "c"])
        self.assertEqual(cfg.output_dir, "out")
        self.assertTrue(cfg.save_report)

    def test_defaults_when_attrs_missing(self) -> None:
        cfg = m.AuditConfig.from_namespace(argparse.Namespace(base_url=None, api_key=None))
        self.assertEqual(cfg.api.base_url, "")
        self.assertEqual(cfg.models, [])
        self.assertEqual(cfg.api.timeout, m.DEFAULT_TIMEOUT)
        self.assertEqual(cfg.output_dir, "reports")


class FilterModelsTest(unittest.TestCase):
    def test_regex_and_limit(self) -> None:
        models = ["gpt-4o", "claude-3-5-sonnet", "llama-3", "gpt-4o-mini"]
        self.assertEqual(m.filter_models(models, "gpt", 1), ["gpt-4o"])

    def test_no_filter_preserves_order(self) -> None:
        models = ["b", "a", "c"]
        self.assertEqual(m.filter_models(models, None, None), models)

    def test_empty_results(self) -> None:
        self.assertEqual(m.filter_models(["gpt-4o"], "claude", None), [])

    def test_invalid_regex_is_value_error(self) -> None:
        with self.assertRaises(ValueError):
            m.filter_models(["gpt-4o"], "(", None)

    def test_negative_limit_is_value_error(self) -> None:
        with self.assertRaises(ValueError):
            m.filter_models(["gpt-4o"], None, -1)


class RunEstimateTest(unittest.TestCase):
    def test_pricing_matches_known_standard_models(self) -> None:
        codex = m.pricing_for_model("gpt-5.3-codex")
        self.assertIsNotNone(codex)
        self.assertEqual(codex.label, "GPT-5.3-Codex")
        self.assertEqual(codex.input_per_million, 1.75)
        self.assertEqual(codex.output_per_million, 14.0)

        gpt_54 = m.pricing_for_model("gpt-5.4")
        self.assertIsNotNone(gpt_54)
        self.assertEqual(gpt_54.label, "GPT-5.4")
        self.assertEqual(gpt_54.input_per_million, 2.5)
        self.assertEqual(gpt_54.output_per_million, 15.0)

        gpt_54_mini = m.pricing_for_model("gpt-5.4-mini")
        self.assertIsNotNone(gpt_54_mini)
        self.assertEqual(gpt_54_mini.label, "GPT-5.4 mini")
        self.assertEqual(gpt_54_mini.input_per_million, 0.75)
        self.assertEqual(gpt_54_mini.output_per_million, 4.5)

        gpt_55 = m.pricing_for_model("gpt-5.5")
        self.assertIsNotNone(gpt_55)
        self.assertEqual(gpt_55.label, "GPT-5.5")
        self.assertEqual(gpt_55.input_per_million, 5.0)
        self.assertEqual(gpt_55.output_per_million, 30.0)

        opus_45 = m.pricing_for_model("claude-opus-4-5-20251101")
        self.assertIsNotNone(opus_45)
        self.assertEqual(opus_45.label, "Claude Opus 4.5+")
        self.assertEqual(opus_45.input_per_million, 5.0)
        self.assertEqual(opus_45.output_per_million, 25.0)

    def test_pricing_does_not_guess_different_variants(self) -> None:
        self.assertIsNone(m.pricing_for_model("gpt-5-pro"))
        self.assertIsNone(m.pricing_for_model("gpt-5-mini"))
        self.assertIsNone(m.pricing_for_model("o3-mini"))

    def test_counts_family_specific_probes(self) -> None:
        cfg = m.AuditConfig(make_config(), ["gpt-4o", "claude-3-haiku"], None, None, False, False, "reports", True)
        estimate = m.build_run_estimate(cfg, cfg.models)
        self.assertEqual(estimate.model_count, 2)
        self.assertEqual(estimate.probes_by_model["gpt-4o"], 7)
        self.assertEqual(estimate.probes_by_model["claude-3-haiku"], 8)
        self.assertEqual(estimate.probe_requests, 15)

    def test_all_targeted_adds_all_targeted_probes(self) -> None:
        cfg = m.AuditConfig(make_config(), ["llama-3"], None, None, True, False, "reports", False)
        estimate = m.build_run_estimate(cfg, cfg.models)
        self.assertEqual(estimate.probes_by_model["llama-3"], 8)
        self.assertEqual(estimate.max_output_tokens, 8 * cfg.api.max_tokens)

    def test_known_price_estimate_uses_output_upper_bound(self) -> None:
        cfg = m.AuditConfig(make_config(), ["gpt-5.5"], None, None, False, False, "reports", False)
        estimate = m.build_run_estimate(cfg, cfg.models)
        item = estimate.cost_by_model["gpt-5.5"]
        self.assertEqual(item.pricing_label, "GPT-5.5")
        self.assertEqual(item.output_tokens, 7 * m.reasoning_token_budget(cfg.api))
        self.assertIsNotNone(item.total_cost)

        lines = m.format_run_estimate(estimate)
        self.assertTrue(any("Estimated official cost upper bound:" in line for line in lines))

    def test_format_run_estimate_includes_output_dir(self) -> None:
        cfg = m.AuditConfig(make_config(), ["gpt-5.5"], None, None, False, False, "reports", False)
        estimate = m.build_run_estimate(cfg, cfg.models)
        lines = m.format_run_estimate(estimate, cfg.output_dir, cfg.save_report)
        self.assertTrue(any(line.startswith("Output dir: ") for line in lines))


class DecisionSummaryTest(unittest.TestCase):
    def test_normal_model_summary(self) -> None:
        rows = m.build_decision_summary({"gpt-4o": [make_result(score=90)]})
        self.assertEqual(rows[0]["model"], "gpt-4o")
        self.assertEqual(rows[0]["availability"], 100.0)
        self.assertIn("recommendation", rows[0])

    def test_all_failed_recommends_no_use(self) -> None:
        rows = m.build_decision_summary({"gpt-4o": [make_result(status="error", score=0)]})
        self.assertIsNone(rows[0]["capability"])
        self.assertIn("Do not use", rows[0]["recommendation"])

    def test_severe_issues_are_json_serializable(self) -> None:
        result = make_result("universal_reasoning", "universal", 15, "ok", 0)
        rows = m.build_decision_summary({"gpt-4o": [result]})
        json.dumps(rows)
        self.assertEqual(rows[0]["severe_issues"][0]["probe_id"], "universal_reasoning")


class ReportOutputTest(unittest.TestCase):
    def test_build_report_has_decision_summary(self) -> None:
        report = m.build_report({"gpt-4o": [make_result(score=90)]}, make_config())
        self.assertIn("## Decision Summary", report)
        self.assertIn("| Model | Rating | Capability | Availability | Authenticity | Recommendation |", report)

    def test_write_reports_json_has_summary_and_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _md_path, json_path = m.write_reports(tmpdir, {"gpt-4o": [make_result(score=90)]}, make_config())
            with open(json_path, encoding="utf-8") as file:
                data = json.load(file)
        self.assertIn("summary", data)
        self.assertIn("models", data)
        self.assertIn("gpt-4o", data["models"])

    def test_write_reports_handles_severe_issues(self) -> None:
        result = make_result("universal_reasoning", "universal", 15, "ok", 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path, json_path = m.write_reports(tmpdir, {"gpt-4o": [result]}, make_config())
            self.assertTrue(os.path.exists(md_path))
            with open(json_path, encoding="utf-8") as file:
                data = json.load(file)
        self.assertEqual(data["summary"][0]["severe_issues"][0]["probe_id"], "universal_reasoning")


class TuiSaveTest(unittest.TestCase):
    def test_save_latest_report_writes_unsaved_completed_report(self) -> None:
        app = m.TuiApp.__new__(m.TuiApp)
        app.language = "zh"
        app.logs = []
        app.log_scroll = 0
        app.last_saved_paths = None
        app.last_model_results = {"gpt-4o": [make_result(score=90)]}
        app.last_report_config = make_config()
        app.last_report_output_dir = "reports"
        app.last_report_unsaved = True

        with mock.patch("relay_audit.tui.write_reports", return_value=("new.md", "new.json")) as write_reports:
            app.save_latest_report()

        write_reports.assert_called_once()
        self.assertEqual(app.last_saved_paths, ("new.md", "new.json"))
        self.assertFalse(app.last_report_unsaved)
        self.assertIn("报告已保存", app.logs)

    def test_save_latest_report_does_not_write_again_when_already_saved(self) -> None:
        app = m.TuiApp.__new__(m.TuiApp)
        app.language = "zh"
        app.logs = []
        app.log_scroll = 0
        app.last_saved_paths = ("old.md", "old.json")
        app.last_model_results = {"gpt-4o": [make_result(score=90)]}
        app.last_report_config = make_config()
        app.last_report_output_dir = "reports"
        app.last_report_unsaved = False

        with mock.patch("relay_audit.tui.write_reports", side_effect=AssertionError("should not write")):
            app.save_latest_report()

        self.assertIn("报告已保存过。", app.logs)
        self.assertIn("Markdown: old.md", app.logs)


class TuiKeyTest(unittest.TestCase):
    def test_x_clears_live_logs_only(self) -> None:
        app = m.TuiApp.__new__(m.TuiApp)
        app.logs = ["one", "two"]
        app.log_scroll = 3
        app.view_mode = "report"
        app.worker = None
        app.detail_items = [{"title": "detail"}]
        app.error_items = [{"title": "error"}]

        self.assertTrue(app.handle_key(ord("x")))

        self.assertEqual(app.logs, [])
        self.assertEqual(app.log_scroll, 0)
        self.assertEqual(app.detail_items, [{"title": "detail"}])
        self.assertEqual(app.error_items, [{"title": "error"}])

    def test_arrow_keys_move_config_selection(self) -> None:
        app = m.TuiApp.__new__(m.TuiApp)
        calls: list[int] = []
        app.worker = None
        app.selected = 4
        app.fields = [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D"), ("e", "E")]
        app.scroll_log = calls.append  # type: ignore[method-assign]

        self.assertTrue(app.handle_key(m.curses.KEY_DOWN))
        self.assertEqual(app.selected, 0)
        self.assertEqual(calls, [])

    def test_confirm_run_popup_formats_known_cost_in_chinese(self) -> None:
        class FakeStdScr:
            def getmaxyx(self) -> tuple[int, int]:
                return (40, 100)

        class FakeWin:
            def keypad(self, value: bool) -> None:
                self.keypad_enabled = value

            def erase(self) -> None:
                pass

            def border(self) -> None:
                pass

            def addstr(self, *_args: object) -> None:
                pass

            def refresh(self) -> None:
                pass

            def getmaxyx(self) -> tuple[int, int]:
                return (20, 80)

            def getch(self) -> int:
                return 10

        app = m.TuiApp.__new__(m.TuiApp)
        app.stdscr = FakeStdScr()
        app.language = "zh"
        app.state = {
            "base_url": "https://relay.example.com",
            "api_key": "sk",
            "model": "gpt-5.5",
            "model_filter": "",
            "limit": "",
            "timeout": "30",
            "max_tokens": "900",
            "temperature": "0",
            "api_style": "auto",
            "mode": "standard",
            "output_dir": "reports",
            "save_report": False,
            "all_targeted": False,
            "hide_prompts": False,
        }

        with mock.patch.object(m.curses, "newwin", return_value=FakeWin()):
            self.assertTrue(app.confirm_run_popup())


class ErrorSuggestionTest(unittest.TestCase):
    def test_auth_error(self) -> None:
        suggestions = m.suggest_next_steps("HTTP 401: invalid token")
        self.assertTrue(any("API key" in item for item in suggestions))

    def test_auto_failed(self) -> None:
        suggestions = m.suggest_next_steps("auto failed: openai-chat: No choices")
        self.assertTrue(any("api-style" in item for item in suggestions))

    def test_no_models(self) -> None:
        suggestions = m.suggest_next_steps("No models to audit.")
        self.assertTrue(any("Model filter" in item for item in suggestions))


class DotenvTest(unittest.TestCase):
    def test_parse_env_lines(self) -> None:
        parsed = m.parse_env_lines([
            "# comment\n",
            "AI_RELAY_API_KEY='sk-test'\n",
            'AI_RELAY_BASE_URL="https://relay.example.com/v1" # inline\n',
            "BAD LINE\n",
        ])
        self.assertEqual(parsed["AI_RELAY_API_KEY"], "sk-test")
        self.assertEqual(parsed["AI_RELAY_BASE_URL"], "https://relay.example.com/v1")

    def test_load_dotenv_does_not_override_by_default(self) -> None:
        old_key = os.environ.get("AI_RELAY_API_KEY")
        old_base_url = os.environ.get("AI_RELAY_BASE_URL")
        os.environ["AI_RELAY_API_KEY"] = "existing"
        os.environ.pop("AI_RELAY_BASE_URL", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, ".env")
                with open(path, "w", encoding="utf-8") as file:
                    file.write("AI_RELAY_API_KEY=from-file\nAI_RELAY_BASE_URL=https://relay.example.com\n")
                loaded = m.load_dotenv(path)
            self.assertEqual(os.environ["AI_RELAY_API_KEY"], "existing")
            self.assertNotIn("AI_RELAY_API_KEY", loaded)
            self.assertEqual(loaded["AI_RELAY_BASE_URL"], "https://relay.example.com")
        finally:
            if old_key is None:
                os.environ.pop("AI_RELAY_API_KEY", None)
            else:
                os.environ["AI_RELAY_API_KEY"] = old_key
            if old_base_url is None:
                os.environ.pop("AI_RELAY_BASE_URL", None)
            else:
                os.environ["AI_RELAY_BASE_URL"] = old_base_url


class CompareReportTest(unittest.TestCase):
    def _report(self, score: float = 80.0, status: str = "ok") -> dict[str, object]:
        return {
            "summary": [
                {
                    "model": "gpt-4o",
                    "capability": score,
                    "availability": 100.0 if status == "ok" else 0.0,
                    "rating": m.overall_rating(score if status == "ok" else None, 100.0 if status == "ok" else 0.0),
                    "authenticity": "未发现不一致" if status == "ok" else "无法判断",
                }
            ],
            "models": {
                "gpt-4o": [
                    {
                        "probe_id": "p",
                        "title": "p",
                        "category": "universal",
                        "weight": 10,
                        "status": status,
                        "score": score,
                    }
                ]
            },
        }

    def test_compare_reports_detects_score_delta(self) -> None:
        comparison = m.compare_reports(self._report(80), self._report(90))
        self.assertEqual(comparison["added_models"], [])
        changed = comparison["changed_models"]
        self.assertEqual(changed[0]["model"], "gpt-4o")
        self.assertEqual(changed[0]["capability_delta"], 10.0)
        self.assertEqual(changed[0]["probe_changes"][0]["score_delta"], 10.0)

    def test_compare_reports_detects_added_model(self) -> None:
        baseline = {"summary": [], "models": {}}
        comparison = m.compare_reports(baseline, self._report(80))
        self.assertEqual(comparison["added_models"], ["gpt-4o"])

    def test_format_comparison_markdown(self) -> None:
        comparison = m.compare_reports(self._report(80), self._report(90))
        markdown = m.format_comparison_markdown(comparison)
        self.assertIn("## Baseline Comparison", markdown)
        self.assertIn("+10.0", markdown)

    def test_load_report_json_validates_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump({"no_models": {}}, file)
            with self.assertRaises(ValueError):
                m.load_report_json(path)


class ProbeConfigTest(unittest.TestCase):
    def _config_path(
        self,
        tmpdir: str,
        scorer: str = "json_contract",
        families: list[str] | None = None,
    ) -> str:
        path = os.path.join(tmpdir, "probes.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "probes": [
                        {
                            "probe_id": "custom_json",
                            "title": "Custom JSON",
                            "category": "universal",
                            "weight": 5,
                            "families": families or ["unknown", "gpt", "claude"],
                            "system": "Return JSON.",
                            "user": "Return JSON.",
                            "scorer": scorer,
                        }
                    ]
                },
                file,
            )
        return path

    def test_scorer_registry_covers_default_probes(self) -> None:
        self.assertTrue(all(probe.scorer_id for probe in m.build_probes()))

    def test_load_probe_config_maps_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            probes = m.load_probe_config(self._config_path(tmpdir))
        self.assertEqual(probes[0].probe_id, "custom_json")
        self.assertEqual(probes[0].scorer_id, "json_contract")
        self.assertIs(probes[0].scorer, m.score_json_contract)

    def test_unknown_scorer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                m.load_probe_config(self._config_path(tmpdir, "missing"))
        self.assertIn("unknown scorer", str(ctx.exception))

    def test_applicable_probes_uses_external_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            probes = m.applicable_probes("gpt-4o", False, self._config_path(tmpdir))
        self.assertEqual([probe.probe_id for probe in probes], ["custom_json"])

    def test_external_config_accepts_gemini_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            probes = m.applicable_probes("gemini-2.0-flash", False, self._config_path(tmpdir, families=["gemini"]))
        self.assertEqual([probe.probe_id for probe in probes], ["custom_json"])


class ValidateFieldTest(unittest.TestCase):
    def test_timeout_positive_int(self) -> None:
        self.assertIsNone(m.validate_field("timeout", "90"))
        self.assertIsNotNone(m.validate_field("timeout", "0"))
        self.assertIsNotNone(m.validate_field("timeout", "x"))
        self.assertIsNotNone(m.validate_field("timeout", ""))

    def test_temperature_float(self) -> None:
        self.assertIsNone(m.validate_field("temperature", "0.7"))
        self.assertIsNone(m.validate_field("temperature", ""))
        self.assertIsNotNone(m.validate_field("temperature", "hot"))

    def test_limit_optional_int(self) -> None:
        self.assertIsNone(m.validate_field("limit", ""))
        self.assertIsNone(m.validate_field("limit", "5"))
        self.assertIsNotNone(m.validate_field("limit", "five"))

    def test_free_text_has_no_validator(self) -> None:
        self.assertIsNone(m.validate_field("base_url", "whatever"))


class DisplayWidthTest(unittest.TestCase):
    def test_pad_display_keeps_wide_text_width(self) -> None:
        text = m.pad_display("中转站地址", 18)
        self.assertEqual(m.display_width(text), 18)


class ConsoleReporterTest(unittest.TestCase):
    def test_probe_result_ok_prints_score(self) -> None:
        result = m.ProbeResult(make_probe(), "ok", 87.5, "good", "hello", 12, {"total_tokens": 3})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.ConsoleReporter().probe_result(result, show_prompt=False)
        out = buf.getvalue()
        self.assertIn("判分: 87.5/100", out)
        self.assertIn("模型回复:", out)

    def test_probe_result_error_prints_failure(self) -> None:
        result = m.ProbeResult(make_probe(), "error", 0, "fail", "", None, None, "HTTP 500")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.ConsoleReporter().probe_result(result, show_prompt=False)
        self.assertIn("失败: HTTP 500", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
