# Long-task cheating monitors

Do LLM monitors catch cheating on multi-hour agent runs?
See [one-pager.md](one-pager.md) for the question and plan.

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

## Score an existing run with a monitor

Monitoring runs after the fact, so one set of agent runs can be scored by many monitors.

```bash
uv run inspect score logs/<run>.eval \
  --scorer cheating_monitor.py@cheating_monitor \
  -S model=anthropic/claude-sonnet-5-5 --action append
```

The monitor stores a 0-100 suspicion score next to METR's task score.
A response without a valid score is stored as `nan` with `parse_failed` set, never as 0.

## Tests

```bash
uv run pytest
```
