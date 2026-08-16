# SPDX-License-Identifier: Apache-2.0
"""oMLX-style OpenAI/OpenCode bridge primitives for MTPLX.

This package deliberately mirrors the clean oMLX protocol shape: preserve the
client transcript, let native Qwen tool formatting do the heavy lifting when it
is available, suppress tool-control markup from visible streaming text, and
parse generated tool calls at completion instead of cancelling the model mid
stream.
"""

from .adapter import normalize_messages_for_template
from .thinking import ThinkingParser, extract_thinking
from .tool_calling import (
    ToolCallExtraction,
    ToolCallStreamFilter,
    extract_tool_calls_with_thinking,
    parse_tool_calls,
)

__all__ = [
    "ThinkingParser",
    "ToolCallExtraction",
    "ToolCallStreamFilter",
    "extract_thinking",
    "extract_tool_calls_with_thinking",
    "normalize_messages_for_template",
    "parse_tool_calls",
]
