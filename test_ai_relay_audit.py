#!/usr/bin/env python3
"""ai_relay_audit 的纯函数单元测试。

仅测试不触网的评分器、解析器与辅助函数。运行：

    python3 -m unittest test_ai_relay_audit -v
"""

from __future__ import annotations

import unittest

import ai_relay_audit as m


def make_config(api_key: str = "sk-test") -> m.ApiConfig:
    return m.ApiConfig(
        base_url="https://relay.example.com",
        api_key=api_key,
        timeout=30,
        max_tokens=900,
        temperature=0.0,
    )


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
        self.assertEqual(m.family_for_model("llama-3-70b"), "unknown")


class IsReasoningModelTest(unittest.TestCase):
    def test_reasoning_models(self) -> None:
        for model in ("o1", "o1-mini", "o3", "o3-mini", "o4-mini", "openai/o3", "gpt-5", "gpt-5-mini"):
            self.assertTrue(m.is_reasoning_model(model), model)

    def test_non_reasoning_models(self) -> None:
        for model in ("gpt-4o", "gpt-4o-mini", "chatgpt-4o-latest", "claude-3-5-sonnet", "gpt-4-turbo"):
            self.assertFalse(m.is_reasoning_model(model), model)


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


class RedactSecretsTest(unittest.TestCase):
    def test_redacts_key(self) -> None:
        config = make_config(api_key="sk-supersecret")
        out = m.redact_secrets("error body contains sk-supersecret in echo", config)
        self.assertNotIn("sk-supersecret", out)
        self.assertIn("***REDACTED***", out)

    def test_no_key_noop(self) -> None:
        config = make_config(api_key="")
        self.assertEqual(m.redact_secrets("plain text", config), "plain text")


if __name__ == "__main__":
    unittest.main()
