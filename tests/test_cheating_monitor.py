import math

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall

from cheating_monitor import format_transcript, parse_monitor_output, truncate_middle


def test_truncate_middle_keeps_both_ends():
    text = "a" * 10 + "b" * 80 + "c" * 10
    result = truncate_middle(text, 20)
    assert result.startswith("a" * 10)
    assert result.endswith("c" * 10)
    assert "[... 80 characters omitted ...]" in result


def test_truncate_middle_leaves_short_text_and_zero_limit_alone():
    assert truncate_middle("short", 20) == "short"
    assert truncate_middle("x" * 100, 0) == "x" * 100


def _transcript_messages():
    return [
        ChatMessageUser(content="Fix the failing tests."),
        ChatMessageAssistant(
            content=[
                ContentReasoning(reasoning="Easiest path is to edit the test file."),
                ContentText(text="Looking at the tests."),
            ],
            tool_calls=[ToolCall(id="1", function="bash", arguments={"cmd": "cat test.py"})],
        ),
        ChatMessageTool(content="x" * 50, tool_call_id="1", function="bash"),
    ]


def test_format_transcript_includes_calls_outputs_and_reasoning():
    text = format_transcript(_transcript_messages(), max_tool_output_chars=10, include_reasoning=True)
    assert "[0] USER:\nFix the failing tests." in text
    assert "(reasoning) Easiest path is to edit the test file." in text
    assert '(tool call) bash({"cmd": "cat test.py"})' in text
    assert "[2] TOOL OUTPUT (bash):" in text
    assert "[... 40 characters omitted ...]" in text


def test_format_transcript_can_hide_reasoning():
    text = format_transcript(_transcript_messages(), max_tool_output_chars=0, include_reasoning=False)
    assert "Easiest path" not in text
    assert "Looking at the tests." in text


def test_format_transcript_can_cap_tool_call_arguments():
    message = ChatMessageAssistant(
        content="",
        tool_calls=[ToolCall(id="1", function="edit", arguments={"text": "y" * 50, "line": 3})],
    )
    capped = format_transcript([message], max_tool_output_chars=0, include_reasoning=True, max_tool_arg_chars=10)
    full = format_transcript([message], max_tool_output_chars=0, include_reasoning=True)
    assert "[... 40 characters omitted ...]" in capped
    assert '"line": 3' in capped
    assert "y" * 50 in full


def test_format_transcript_uses_summary_for_redacted_reasoning():
    message = ChatMessageAssistant(
        content=[ContentReasoning(reasoning="opaque", summary="Plans to edit tests.", redacted=True)]
    )
    text = format_transcript([message], max_tool_output_chars=0, include_reasoning=True)
    assert "(reasoning) Plans to edit tests." in text
    assert "opaque" not in text


def test_parse_takes_last_score_and_reasoning():
    value, reasoning = parse_monitor_output(
        "<reasoning>It rewrote the test.</reasoning> maybe <score>10</score> final <score>85</score>"
    )
    assert value == 85
    assert reasoning == "It rewrote the test."


def test_parse_rejects_missing_or_out_of_range_scores():
    assert math.isnan(parse_monitor_output("no score here")[0])
    assert math.isnan(parse_monitor_output("<score>150</score>")[0])
