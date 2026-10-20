"""DeepSeek V4 Flash behind the server: its pre-tokenizer, its DSML tool calls, its turn openers and its template.

    python -m unittest serve.test_dsv4
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402

D = "｜DSML｜"


class PreTokenizer(unittest.TestCase):
    TOKENS = ["1", "2", "3", "4", "5", "12", "123", "45", "a", "b", "ab", " ", "你", "好", "你好",
              "'", "s", "'s", "x"]
    MERGES = ["1 2", "12 3", "4 5", "a b", "你 好", "' s"]

    def pieces(self, text):
        tok = ST.Tokenizer(self.TOKENS, self.MERGES, pre="joyai-llm")
        return tok._pieces(text)

    def test_digits_then_ideographs_then_the_rest(self):
        self.assertEqual(self.pieces("12345ab"), ["123", "45", "ab"])
        self.assertEqual(self.pieces("ab你好ab"), ["ab", "你好", "ab"])
        self.assertEqual(self.pieces("x's 12"), ["x", "'s", " ", "12"])

    def test_a_special_token_decodes_to_its_text(self):
        # its characters are outside the byte alphabet the other tokens are written in
        tokens = ["<｜begin▁of▁sentence｜>", "a", "Ġb", "<think>"]
        tok = ST.Tokenizer(tokens, [], token_types=[3, 1, 1, 4], pre="joyai-llm")
        self.assertEqual(tok.decode([0, 1, 2, 3]), "<｜begin▁of▁sentence｜>a b<think>")


class ToolCalls(unittest.TestCase):
    SCHEMA = [{"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "n": {"type": "integer"},
                                                              "content": {"type": "string"}}}},
              {"name": "now", "parameters": {"properties": {}}}]
    CONTENT = 'a value may hold </' + D + 'invoke> and "quotes"\nend'
    TEXT = ("thinking</think>I will write it.\n\n<" + D + "tool_calls>\n<" + D + 'invoke name="write">\n<' + D +
            'parameter name="path" string="true">doc.md</' + D + 'parameter>\n<' + D + 'parameter name="n" '
            'string="false">3</' + D + 'parameter>\n<' + D + 'parameter name="content" string="true">' + CONTENT +
            "</" + D + "parameter>\n</" + D + "invoke>\n<" + D + 'invoke name="now">\n</' + D + "invoke>\n</" + D +
            "tool_calls>")

    def events(self, stream_tools, step):
        p = OutputParser(thinking=True, tools=self.SCHEMA, stream_tools=stream_tools, call_format="dsml")
        evs = []
        for i in range(0, len(self.TEXT), step):
            evs += p.feed(self.TEXT[i:i + step])
        return evs + p.finish()

    def test_calls_are_parsed_however_they_arrive(self):
        for stream_tools in (False, True):
            for step in (1, 7, 10_000):
                with self.subTest(stream_tools=stream_tools, step=step):
                    evs = self.events(stream_tools, step)
                    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
                    self.assertEqual(calls, [("write", {"path": "doc.md", "n": 3, "content": self.CONTENT}),
                                             ("now", {})])
                    self.assertEqual("".join(e.text for e in evs if e.kind == "reasoning"), "thinking")
                    self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "I will write it.")

    def test_streamed_call_is_announced_first(self):
        kinds = [e.kind for e in self.events(True, 1) if e.kind.startswith("tool_")]
        self.assertEqual(kinds, ["tool_start", "tool_args", "tool_call"] * 2)


# what of DeepSeek V4's chat template (Unsloth's) the server has to meet: tools as OpenAI sends them, a call's
# arguments read with from_json, tool results in the user's turn, and thinking from enable_thinking
TEMPLATE = """{%- set thinking = enable_thinking | default(false) -%}{%- set d = '｜DSML｜' -%}
{{- bos_token -}}{%- if tools -%}## Tools
{% for t in tools %}{{ t['function'] | tojson }}
{% endfor %}{%- endif -%}
{%- for m in messages -%}
{%- if m['role'] == 'user' -%}<｜User｜>{{ m['content'] }}
{%- elif m['role'] == 'tool' -%}<｜User｜><tool_result>{{ m['content'] }}</tool_result>
{%- elif m['role'] == 'assistant' -%}<｜Assistant｜></think>{{ m['content'] }}
{%- for c in m['tool_calls'] or [] -%}{%- set args = c['function']['arguments'] -%}
{%- if args is string -%}{%- set args = args | from_json -%}{%- endif -%}
<{{ d }}invoke name="{{ c['function']['name'] }}">
{% for k, v in args.items() %}<{{ d }}parameter name="{{ k }}" string="false">{{ v | tojson }}</{{ d }}parameter>
{% endfor %}</{{ d }}invoke>{%- endfor -%}<｜end▁of▁sentence｜>
{%- endif -%}{%- endfor -%}
{%- if add_generation_prompt -%}<｜Assistant｜>{{ '<think>' if thinking else '</think>' }}{%- endif -%}"""


class DsTokens(ByteTokenizer):
    SPECIALS = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>",
                "<｜User｜>", "<｜Assistant｜>"]
    pre = "joyai-llm"
    special_ids = {"tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.eos_token_id": 257}


class ServiceForDsv4(unittest.TestCase):
    def setUp(self):
        tok = DsTokens()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "chat_template.jinja"
            path.write_text(TEMPLATE, encoding="utf-8")
            self.svc = Service(MockEngine(tok, "hi"), tok, ChatTemplate(path))

    def test_stop_ids_turns_and_call_format(self):
        self.assertEqual(self.svc.call_format, "dsml")
        self.assertEqual(self.svc.stop_ids, {257})
        self.assertEqual(self.svc.turn_ids, frozenset({258, 259}))

    def test_prompt_with_tools_and_a_call_in_the_history(self):
        tools = [{"name": "now", "description": "the time", "parameters": {"type": "object", "properties": {}}}]
        messages = [{"role": "user", "content": "time?"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"type": "function", "function": {"name": "now", "arguments": "{}"}}]},
                    {"role": "tool", "content": "12:00"}]
        prompt = self.svc.render_prompt(messages, tools, {"enable_thinking": False})
        self.assertTrue(prompt.startswith("<｜begin▁of▁sentence｜>## Tools"))
        self.assertIn('"name": "now"', prompt)
        self.assertIn("<" + D + 'invoke name="now">', prompt)
        self.assertIn("<tool_result>12:00</tool_result>", prompt)
        self.assertTrue(prompt.endswith("<｜Assistant｜></think>"))
        thinking = self.svc.render_prompt(messages[:1], None, {})
        self.assertTrue(thinking.endswith("<｜Assistant｜><think>"))


if __name__ == "__main__":
    unittest.main()
