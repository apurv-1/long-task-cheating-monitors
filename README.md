# Long-task cheating monitors

Do LLM monitors catch cheating on multi-hour agent runs?
See [one-pager.md](one-pager.md) for the question and plan.

## Status

The harness runs end to end with fake agents: a METR public task runs in Docker, METR scores it, and the monitor scores the saved run.
That includes the Claude Code path, tested with a fake `claude` binary from `tests/fixtures`.
No real models have been run yet.
Next is a small pilot with Claude Code on multi-hour tasks, to see how long runs take and whether cheating shows up.

## Setup

Needs `uv` and a running Docker daemon.

```bash
uv sync
```

## Run an agent on a METR public task

Task images come from METR's public registry, listed in `METR-public-tasks/suite_manifest.yaml`.

```bash
uv run inspect eval mtb/bridge \
  -T image_tag=registry-1.docker.io/metrevals/public-tasks:crossword-0.1.8 \
  --sample-id 5x5_verify --model mockllm/model --log-dir logs
```

Swap `mockllm/model` for a real model once the plumbing works.

## Run Claude Code as the agent

The agent is the unmodified Claude Code CLI, installed inside the task container and signed in with your own subscription.
Inspect starts the container, runs Claude Code as the unprivileged `agent` user, and converts its output into messages for the METR scorer and the monitor.

1. Run `claude setup-token` in your own terminal.
2. Put the token in `.env` as `CLAUDE_CODE_OAUTH_TOKEN=...`. Git ignores `.env`.
3. Run a task:

```bash
uv run python run_claude_code.py \
  --image-tag registry-1.docker.io/metrevals/public-tasks:cowthello-2.0.2 \
  --sample-id main --hours 2
```

Always use `run_claude_code.py` rather than `inspect eval`.
Inspect's eval log records the token, so the script scrubs every log it writes and fails if any copy is left.

Limits:

- Claude Code needs to reach Anthropic, so only tasks that grant `full_internet` work for now, such as `cowthello` and `clone_game`.
- Runs count against your plan's usage limits, so run them one at a time.

## Score an existing run with a monitor

Monitoring runs after the fact, so one set of agent runs can be scored by many monitors.

```bash
uv run inspect score logs/<run>.eval \
  --scorer cheating_monitor.py@cheating_monitor \
  -S model=anthropic/claude-sonnet-5-5 --action append
```

Add `-S include_reasoning=false` to hide the agent's reasoning from the monitor.
The monitor stores a 0-100 suspicion score next to METR's task score.
A response without a valid score is stored as `nan` with `parse_failed` set, never as 0.

## Tests

```bash
uv run pytest
```
