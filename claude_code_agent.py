"""Run the unmodified Claude Code CLI as the agent inside a METR task container.

Claude Code signs in with the user's own subscription token (`claude setup-token`) and talks
to Anthropic directly. Inspect only starts the container, launches Claude Code, and turns its
stream-json output into messages so the METR scorer and the cheating monitor can read the run.

The token reaches the container through stdin, so Inspect's trace files never see it.
Inspect's eval log does record stdin, so every log must go through `scrub_log` before it is
shared. `run_claude_code.py` does that automatically.
"""

import json
import os
import shlex
import zipfile
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    Content,
    ContentReasoning,
    ContentText,
    ModelOutput,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolCall, ToolCallError
from inspect_ai.util import sandbox

TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
REDACTED = "[REDACTED]"

RUN_DIR = "/root/claude_code_run"
STREAM_FILE = f"{RUN_DIR}/stream.jsonl"
STDERR_FILE = f"{RUN_DIR}/stderr.log"
PROMPT_FILE = f"{RUN_DIR}/prompt.md"
# Inspect's read_file runs as the sandbox's default user, which cannot read /root.
# Outputs are copied here only after Claude Code has exited, then removed.
EXPORT_DIR = "/tmp/claude_code_export"

AGENT_USER = "agent"
AGENT_HOME = "/home/agent"
CLAUDE_BINARY = f"{AGENT_HOME}/.local/bin/claude"

SUBMISSION_SUFFIX = """

When you have finished, end with a final message that contains only your submission.
If the task does not ask for a submission, end with a final message that says "done"."""


def official_install_command(version: str) -> str:
    """Anthropic's native installer, which puts the binary in ~/.local/bin."""
    return f"curl -fsSL https://claude.ai/install.sh | bash -s {shlex.quote(version)}"


def _run_script(timeout_seconds: int, model: str | None) -> str:
    """Root shell script that runs Claude Code as the agent user.

    The token arrives on stdin and is exported only for the Claude Code process.
    Output goes to files under /root, which the agent user cannot read or change.
    """
    model_flag = f" --model {shlex.quote(model)}" if model else ""
    claude_command = (
        f"cd {AGENT_HOME} && exec {CLAUDE_BINARY} -p"
        " --output-format stream-json --verbose"
        " --dangerously-skip-permissions --permission-prompts none"
        f"{model_flag}"
    )
    return f"""set -eu
umask 077
IFS= read -r {TOKEN_ENV}
export {TOKEN_ENV}
export DISABLE_AUTOUPDATER=1
set +e
timeout --signal=INT --kill-after=60 {timeout_seconds} \\
  runuser -u {AGENT_USER} -- bash -c {shlex.quote(claude_command)} \\
  < {PROMPT_FILE} > {STREAM_FILE} 2> {STDERR_FILE}
echo "exit_code=$?"
"""


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for item in content or []:
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
        else:
            parts.append(f"[{item.get('type', 'unknown')} omitted]")
    return "\n".join(parts)


def _assistant_content(blocks: list[dict[str, Any]]) -> tuple[list[Content], list[ToolCall]]:
    content: list[Content] = []
    tool_calls: list[ToolCall] = []
    for block in blocks:
        kind = block.get("type")
        if kind == "text":
            content.append(ContentText(text=block.get("text", "")))
        elif kind == "thinking":
            content.append(ContentReasoning(reasoning=block.get("thinking", "")))
        elif kind == "redacted_thinking":
            content.append(ContentReasoning(reasoning="", redacted=True))
        elif kind == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block["id"],
                    function=block.get("name", "unknown"),
                    arguments=block.get("input") or {},
                )
            )
    return content, tool_calls


def stream_to_messages(stream: str) -> tuple[list[ChatMessage], dict[str, Any] | None]:
    """Convert Claude Code stream-json output into Inspect messages.

    Consecutive assistant events with the same message id are merged into one message.
    Messages from subagents keep their parent tool call id in message metadata.
    Returns the messages and the final `result` event, if the run produced one.
    """
    messages: list[ChatMessage] = []
    tool_names: dict[str, str] = {}
    result: dict[str, Any] | None = None
    last_assistant_id: str | None = None

    for line in stream.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        parent = event.get("parent_tool_use_id")
        metadata = {"parent_tool_use_id": parent} if parent else None

        if kind == "assistant":
            message = event.get("message", {})
            content, tool_calls = _assistant_content(message.get("content", []))
            for call in tool_calls:
                tool_names[call.id] = call.function
            message_id = message.get("id")
            previous = messages[-1] if messages else None
            if (
                message_id
                and message_id == last_assistant_id
                and isinstance(previous, ChatMessageAssistant)
            ):
                previous_content = previous.content
                if isinstance(previous_content, str):
                    previous_content = [ContentText(text=previous_content)]
                previous.content = [*previous_content, *content]
                previous.tool_calls = [*(previous.tool_calls or []), *tool_calls]
            else:
                messages.append(
                    ChatMessageAssistant(
                        content=content,
                        tool_calls=tool_calls or None,
                        model=message.get("model"),
                        metadata=metadata,
                    )
                )
            last_assistant_id = message_id

        elif kind == "user":
            last_assistant_id = None
            content = event.get("message", {}).get("content")
            if isinstance(content, str):
                messages.append(ChatMessageUser(content=content, metadata=metadata))
                continue
            for block in content or []:
                if block.get("type") == "tool_result":
                    call_id = block.get("tool_use_id", "")
                    text = _tool_result_text(block.get("content"))
                    messages.append(
                        ChatMessageTool(
                            content=text,
                            tool_call_id=call_id,
                            function=tool_names.get(call_id),
                            error=ToolCallError("unknown", text) if block.get("is_error") else None,
                            metadata=metadata,
                        )
                    )
                elif block.get("type") == "text":
                    messages.append(ChatMessageUser(content=block.get("text", ""), metadata=metadata))

        elif kind == "result":
            result = event

    return messages, result


@solver
def claude_code(
    timeout_hours: float = 2.0,
    model: str | None = None,
    version: str = "stable",
    install_command: str | None = None,
) -> Solver:
    """Run Claude Code on the task with the user's subscription token.

    Args:
        timeout_hours: Wall-clock limit. Claude Code gets SIGINT at the limit so it can end its turn.
        model: Claude Code `--model` value, such as "opus". None uses the plan's default.
        version: Claude Code version for the official installer, such as "stable" or "2.1.290".
        install_command: Replaces the official installer. Only for testing with a fake binary.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        token = os.environ.get(TOKEN_ENV)
        if not token:
            raise RuntimeError(f"Set {TOKEN_ENV} from `claude setup-token` before running.")

        sbx = sandbox()
        install = install_command or official_install_command(version)
        installed = await sbx.exec(["bash", "-lc", install], user=AGENT_USER, timeout=900)
        if not installed.success:
            raise RuntimeError(f"Claude Code install failed: {installed.stderr[-2000:]}")

        prompt = state.input_text + SUBMISSION_SUFFIX
        await _root_exec(
            f"mkdir -p {RUN_DIR} && chmod 700 {RUN_DIR} && cat > {PROMPT_FILE}", input=prompt
        )

        timeout_seconds = int(timeout_hours * 3600)
        run = await sbx.exec(
            ["bash", "-c", _run_script(timeout_seconds, model)],
            input=token + "\n",
            user="root",
            timeout=timeout_seconds + 300,
        )

        await _root_exec(
            f"mkdir -p {EXPORT_DIR} && install -m 644 {STREAM_FILE} {STDERR_FILE} {EXPORT_DIR}/"
        )
        stream = await sbx.read_file(f"{EXPORT_DIR}/stream.jsonl")
        stderr = await sbx.read_file(f"{EXPORT_DIR}/stderr.log")
        await _root_exec(f"rm -rf {EXPORT_DIR}")
        stream = stream.replace(token, REDACTED)
        stderr = stderr.replace(token, REDACTED)

        messages, result = stream_to_messages(stream)
        state.messages = [ChatMessageUser(content=prompt), *messages]
        final_text = (result or {}).get("result") or ""
        state.output = ModelOutput.from_content(model="claude-code", content=final_text)
        state.metadata["claude_code"] = {
            "run_output": run.stdout.strip(),
            "stderr_tail": stderr[-4000:],
            "result": {k: v for k, v in (result or {}).items() if k != "result"},
            "requested_model": model,
            "version": version,
        }
        return state

    return solve


async def _root_exec(script: str, input: str | None = None) -> None:
    result = await sandbox().exec(["bash", "-c", script], input=input, user="root")
    if not result.success:
        raise RuntimeError(f"Sandbox command failed: {script}\n{result.stderr[-2000:]}")


def scrub_log(path: str | Path, secret: str) -> None:
    """Replace every copy of `secret` in an eval log, then confirm none is left."""
    if not secret:
        raise ValueError("Refusing to scrub with an empty secret.")
    path = Path(path)
    log = read_eval_log(str(path))
    cleaned = log.model_dump_json().replace(secret, REDACTED)
    write_eval_log(type(log).model_validate_json(cleaned), str(path))
    if _file_contains(path, secret):
        raise RuntimeError(f"Secret still present in {path} after scrubbing.")


def _file_contains(path: Path, secret: str) -> bool:
    needle = secret.encode()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return any(needle in archive.read(name) for name in archive.namelist())
    return needle in path.read_bytes()
