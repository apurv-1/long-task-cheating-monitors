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

Some tasks, such as `cowthello`, download files when they are scored.
Pass METR's public `secrets.env` so the task gets the settings it needs:

```bash
-T secrets_env_path=../METR-public-tasks/secrets.env
```

Check a task's `resources` in its `manifest.yaml` before running it.
Docker Desktop on a Mac can refuse tasks that ask for more CPUs than it has, such as `cowthello/no_internet`.

### Free local model

Ollama runs a model on your own machine, so testing costs nothing.
These settings keep memory low enough to run beside Docker on a 24 GB Mac:

```bash
OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
  OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
ollama pull qwen3:8b
```

Then use `--model ollama/qwen3:8b` for the agent and `-S model=ollama/qwen3:8b` for the monitor.
A model this small is for testing the harness, not for study results: its context fills up long before a multi-hour task ends.

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

## Label runs by hand

```bash
uv run python label_runs.py --labeller <your-name> --failure-sample 10
```

The tool walks through every run METR scored as a success, plus a random sample of failures, and asks for cheat, legitimate, or unclear.
It also asks which message numbers support the label and for a short note.
Monitor scores are never shown, so they cannot sway a label.
Labels are appended to `labels/labels.jsonl`, which belongs in git, and labelled runs are skipped next time.

## Compute the results

```bash
uv run python analyze.py --out results/summary.json
```

For each monitor setup, this reports recall at a threshold that flags at most 5% of legitimate runs, a 95% interval, AUROC, recall by transcript length, and the recall lost when the monitor cannot see the agent's reasoning.
Every number comes with the run counts behind it.
The threshold is picked on the same runs it is measured on, so with few runs it flatters recall.

## Tests

```bash
uv run pytest
```
