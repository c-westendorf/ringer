"""Tests for the opencode-json token parser and its config wiring.

Fixture: tests/fixtures/opencode_v2_stream.jsonl is a trimmed, synthetic opencode
v2 `run --format json` stream. It is fabricated to exercise the parser; it is
NOT real worker output and contains no real client or patient data.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import ringer
from ringer import (
    EngineConfig,
    load_engines,
    parse_opencode_json_tokens,
    parse_token_count,
)

FIXTURE = Path(__file__).parent / "fixtures" / "opencode_v2_stream.jsonl"


def _read_fixture() -> str:
    return FIXTURE.read_text(encoding="utf-8")


class OpenCodeJsonParserTests(unittest.TestCase):
    def test_multi_step_stream_sums_correctly(self) -> None:
        # Step sums (input+output+reasoning+cache.write):
        # 601+65+0+0       = 666
        # 120+40+10+500    = 670
        # 7+3+2+1          = 13
        # 50+5+0+0         = 55  (missing reasoning + cache.write count as 0)
        # step with no tokens dict -> 0
        # step with no part      -> 0
        # total = 1404
        self.assertEqual(parse_opencode_json_tokens(_read_fixture()), 1404)

    def test_cache_read_excluded(self) -> None:
        text = (
            '{"type":"step_finish","part":{"tokens":'
            '{"input":100,"output":10,"reasoning":0,'
            '"cache":{"read":99999,"write":5}}}}\n'
        )
        # cache.read (99999) excluded; 100+10+0+5 = 115
        self.assertEqual(parse_opencode_json_tokens(text), 115)

    def test_malformed_lines_skipped(self) -> None:
        text = (
            "not json at all\n"
            "{this is not json}\n"
            '{"type":"step_finish","part":{"tokens":{"input":7,"output":1}}}\n'
            "   \n"
            '{"type":"run_finish"}\n'
        )
        # only one valid step_finish -> 7+1 = 8
        self.assertEqual(parse_opencode_json_tokens(text), 8)

    def test_no_steps_gives_none(self) -> None:
        text = (
            '{"type":"session_start"}\n'
            '{"type":"run_finish"}\n'
            "not json\n"
        )
        self.assertIsNone(parse_opencode_json_tokens(text))

    def test_empty_text_gives_none(self) -> None:
        self.assertIsNone(parse_opencode_json_tokens(""))

    def test_step_finish_without_tokens_contributes_zero_but_counts(self) -> None:
        # A step_finish with no tokens object is still a step, so the result is
        # 0 (not None).
        text = '{"type":"step_finish","part":{"reason":"stop"}}\n'
        self.assertEqual(parse_opencode_json_tokens(text), 0)


class ParseTokenCountDispatchTests(unittest.TestCase):
    def test_dispatches_to_opencode_json(self) -> None:
        self.assertEqual(
            parse_token_count(_read_fixture(), token_regex=None, token_parser="opencode-json"),
            1404,
        )

    def test_unknown_parser_returns_none(self) -> None:
        # Defensive: load_engines rejects unknown names, but parse_token_count
        # must not raise.
        self.assertIsNone(
            parse_token_count("anything", token_regex=None, token_parser="nope")
        )

    def test_existing_regex_behaviour_unchanged(self) -> None:
        regex = r"tokens\s+used\s*:?\s*([0-9][0-9,]*)"
        self.assertEqual(parse_token_count("tokens used: 1,234", regex), 1234)
        self.assertEqual(parse_token_count("tokens used\n5,678", regex), 5678)
        # Default (no args) still works.
        self.assertEqual(parse_token_count("tokens used: 9"), 9)
        self.assertIsNone(parse_token_count("nothing here", regex))


class EngineConfigTokenParserTests(unittest.TestCase):
    def _engine(self, section: dict) -> EngineConfig:
        return load_engines({"opencode": section})["opencode"]

    def test_token_parser_accepted(self) -> None:
        engine = self._engine(
            {
                "bin": "/usr/local/bin/opencode",
                "args_template": ["run", "{spec}"],
                "token_parser": "opencode-json",
            }
        )
        self.assertEqual(engine.token_parser, "opencode-json")
        self.assertIsNone(engine.token_regex)

    def test_setting_both_regex_and_parser_is_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._engine(
                {
                    "bin": "/usr/local/bin/opencode",
                    "args_template": ["run", "{spec}"],
                    "token_regex": r"tokens\s+used\s*:?\s*([0-9]+)",
                    "token_parser": "opencode-json",
                }
            )
        self.assertIn("token_parser", str(ctx.exception))
        self.assertIn("token_regex", str(ctx.exception))

    def test_unknown_parser_is_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._engine(
                {
                    "bin": "/usr/local/bin/opencode",
                    "args_template": ["run", "{spec}"],
                    "token_parser": "bogus",
                }
            )
        self.assertIn("token_parser", str(ctx.exception))

    def test_parser_overrides_base_regex_inheritance(self) -> None:
        # A custom engine overriding the built-in codex engine should not
        # silently inherit codex's token_regex when token_parser is set.
        engines = load_engines(
            {
                "codex": {
                    "bin": "codex",
                    "args_template": ["exec", "{spec}"],
                    "token_parser": "opencode-json",
                }
            }
        )
        codex = engines["codex"]
        self.assertEqual(codex.token_parser, "opencode-json")
        self.assertIsNone(codex.token_regex)

    def test_neither_set_inherits_base_regex(self) -> None:
        engines = load_engines(
            {
                "codex": {
                    "bin": "codex",
                    "args_template": ["exec", "{spec}"],
                }
            }
        )
        self.assertIsNotNone(engines["codex"].token_regex)
        self.assertIsNone(engines["codex"].token_parser)


if __name__ == "__main__":
    unittest.main()