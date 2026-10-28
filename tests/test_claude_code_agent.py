import json
import zipfile

import pytest
from inspect_ai.log import EvalLog, EvalSample, EvalSpec, read_eval_log, write_eval_log
from inspect_ai.log._log import EvalConfig, EvalDataset
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
)

from claude_code_agent import REDACTED, scrub_log, stream_to_messages

SECRET = "sk-ant-oat01-test-secret-value"


def _stream(*events: dict) -> str:
    return "\n".join(json.dumps(event) for event in events)


def _assistant(message_id: str, *blocks: dict, parent: str | None = None) -> dict:
    return {
        "type": "assistant",
        "parent_tool_use_id": parent,
        "message": {"id": message_id, "model": "claude-opus-5-5", "content": list(blocks)},
    }


def _tool_result(call_id: str, content, is_error: bool = False) -> dict:
    return {
        "type": "user",
        "parent_tool_use_id": None,
        "message": {
            "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": is_error}
            ]
        },
    }


def test_converts_turns_and_merges_split_assistant_events():
    stream = _stream(
        {"type": "system", "subtype": "init", "model": "claude-opus-5-5"},
        _assistant("m1", {"type": "thinking", "thinking": "Check the tests first."}),
        _assistant("m1", {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}),
        _tool_result("t1", "test.py"),
        _assistant("m2", {"type": "text", "text": "done"}),
        {"type": "result", "subtype": "success", "result": "done", "num_turns": 2, "is_error": False},
    )
    messages, result = stream_to_messages(stream)

    assert [type(m) for m in messages] == [ChatMessageAssistant, ChatMessageTool, ChatMessageAssistant]
    first = messages[0]
    assert isinstance(first.content[0], ContentReasoning)
    assert first.tool_calls[0].function == "Bash"
    assert first.tool_calls[0].arguments == {"command": "ls"}
    assert messages[1].function == "Bash"
    assert messages[1].text == "test.py"
    assert result["num_turns"] == 2


def test_tool_errors_list_content_and_subagents():
    stream = _stream(
        _assistant("m1", {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}),
        _tool_result("t1", [{"type": "text", "text": "no such file"}, {"type": "image"}], is_error=True),
        _assistant("m2", {"type": "text", "text": "sub"}, parent="t9"),
        {"type": "user", "message": {"content": "subagent prompt"}, "parent_tool_use_id": "t9"},
    )
    messages, result = stream_to_messages(stream)

    tool = messages[1]
    assert tool.error is not None
    assert tool.text == "no such file\n[image omitted]"
    assert messages[2].metadata == {"parent_tool_use_id": "t9"}
    assert isinstance(messages[3], ChatMessageUser)
    assert result is None


def test_skips_blank_and_malformed_lines():
    stream = "\n\nnot json\n" + _stream(_assistant("m1", {"type": "text", "text": "hi"}))
    messages, _ = stream_to_messages(stream)
    assert len(messages) == 1


def _write_log(path, text: str) -> None:
    log = EvalLog(
        eval=EvalSpec(
            created="2026-10-28T00:00:00+00:00",
            task="t",
            dataset=EvalDataset(),
            model="none/none",
            config=EvalConfig(),
        ),
        samples=[EvalSample(id=1, epoch=1, input=text, target="", messages=[ChatMessageUser(content=text)])],
    )
    write_eval_log(log, str(path))


@pytest.mark.parametrize("suffix", [".eval", ".json"])
def test_scrub_removes_every_copy(tmp_path, suffix):
    path = tmp_path / f"run{suffix}"
    _write_log(path, f"token is {SECRET} and again {SECRET}")

    scrub_log(path, SECRET)

    sample = read_eval_log(str(path)).samples[0]
    assert sample.input == f"token is {REDACTED} and again {REDACTED}"
    if suffix == ".eval":
        with zipfile.ZipFile(path) as archive:
            assert all(SECRET.encode() not in archive.read(n) for n in archive.namelist())
    else:
        assert SECRET not in path.read_text()


def test_scrub_refuses_empty_secret(tmp_path):
    with pytest.raises(ValueError):
        scrub_log(tmp_path / "x.eval", "")
