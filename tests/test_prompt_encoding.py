from __future__ import annotations

from mtplx.benchmarks.schema import PromptCase, encode_prompt_case


class DummyTokenizer:
    def encode(self, text):
        return [ord(ch) for ch in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is True
        assert add_generation_prompt is True
        assert kwargs in ({}, {"enable_thinking": False})
        rendered = "".join(f"{m['role']}:{m['content']}\n" for m in messages) + "assistant:"
        return [ord(ch) for ch in rendered]


class GemmaNoTemplateTokenizer:
    bos_token = "<bos>"

    def get_vocab(self):
        return {
            "<|think|>": 1,
            "<|channel>": 2,
            "<channel|>": 3,
            "<|turn>": 4,
            "<turn|>": 5,
        }

    def encode(self, text, add_special_tokens=False):
        self.last_text = text
        return [ord(ch) for ch in text]


class NoTemplateTokenizer(DummyTokenizer):
    def apply_chat_template(self, *_args, **_kwargs):
        raise ValueError("Cannot use chat template functions because tokenizer.chat_template is not set")


def test_encode_prompt_case_raw():
    case = PromptCase(id="x", category="raw", prompt="abc")
    assert encode_prompt_case(DummyTokenizer(), case, chat_template=False) == [97, 98, 99]


def test_encode_prompt_case_chat_template_wraps_user_prompt():
    case = PromptCase(id="x", category="chat", prompt="hello")
    encoded = encode_prompt_case(DummyTokenizer(), case, chat_template=True)
    assert encoded[:5] == [117, 115, 101, 114, 58]


def test_encode_prompt_case_uses_explicit_messages():
    case = PromptCase(
        id="x",
        category="chat",
        prompt="ignored",
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    )
    encoded = encode_prompt_case(DummyTokenizer(), case, chat_template=True)
    assert chr(encoded[0]) == "s"


def test_encode_prompt_case_can_disable_thinking():
    case = PromptCase(id="x", category="chat", prompt="hello")
    encoded = encode_prompt_case(DummyTokenizer(), case, chat_template=True, enable_thinking=False)
    assert encoded[:5] == [117, 115, 101, 114, 58]


def test_encode_prompt_case_gemma_without_template_uses_gemma_chat_encoding():
    tokenizer = GemmaNoTemplateTokenizer()
    case = PromptCase(id="x", category="chat", prompt="hello")
    encoded = encode_prompt_case(tokenizer, case, chat_template=True, enable_thinking=False)

    assert encoded
    assert "<|turn>user\nhello<turn|>" in tokenizer.last_text
    assert "<|channel>thought\n<channel|>" in tokenizer.last_text


def test_encode_prompt_case_missing_template_falls_back_to_plain_chat_encoding():
    tokenizer = NoTemplateTokenizer()
    case = PromptCase(id="x", category="chat", prompt="hello")

    encoded = encode_prompt_case(tokenizer, case, chat_template=True, enable_thinking=False)

    assert encoded[:5] == [117, 115, 101, 114, 58]
