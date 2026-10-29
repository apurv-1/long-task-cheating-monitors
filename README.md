# Long-task cheating monitors

Do LLM monitors catch AI agents that cheat on long tasks?

AI labs now run agents that work for hours on their own, and they rely on a second model, a monitor, to flag bad behaviour in what those agents do.
METR's [Frontier Risk Report](https://metr.org/blog/2026-05-19-frontier-risk-report/) found that on tasks longer than 8 hours, at least 16% of runs scored as successes were illegitimate.
Those cheats were found by people reading transcripts.
This repo measures how many of them an LLM monitor would catch on its own, at what false alarm rate, and whether it gets worse on longer runs or when the monitor cannot see the agent's reasoning.

## Status

- The harness works end to end: agents run on METR's public tasks in Docker, METR's scorer grades them, a monitor scores the saved runs, and a hand-labelled set turns into recall numbers.
- First pilot, run for free on a laptop: [a small local model never got far enough to cheat on EvilGenie](notes/2026-10-29-small-model-evilgenie.md).
- No frontier-model results yet.
  The plan for the full study is in [docs/study-plan.md](docs/study-plan.md).

## How it works

1. **Run an agent.**
   METR's public tasks run in Docker through [Inspect](https://inspect.aisi.org.uk/) and METR's [task bridge](https://github.com/METR/inspect-metr-task-bridge), and METR's own scorer grades each run.
2. **Score the run with a monitor.**
   The monitor reads the saved transcript and gives a cheating suspicion score from 0 to 100.
   It runs after the fact, so one set of agent runs can be scored by many monitors.
3. **Label runs by hand.**
   A person labels each run as cheat, legitimate, or unclear, without seeing the monitor's score.
4. **Analyse.**
   Labels and monitor scores become recall at a fixed false positive rate, with intervals and run counts.

## Repository layout

| Path | What it is |
|---|---|
| [`cheating_monitor.py`](cheating_monitor.py) | The monitor, as an Inspect scorer |
| [`label_runs.py`](label_runs.py) | Command-line tool for hand-labelling runs, blind to monitor scores |
| [`analyze.py`](analyze.py) | Turns labels and monitor scores into the study's numbers |
| [`claude_code_agent.py`](claude_code_agent.py), [`run_claude_code.py`](run_claude_code.py) | Optional: run the Claude Code CLI as the agent on your own subscription |
| [`notes/`](notes/) | Lab notes from experiments |
| [`docs/study-plan.md`](docs/study-plan.md) | The plan for the full study |
| [`tests/`](tests/) | Unit tests, including known-answer tests for the analysis |

## Setup

You need [`uv`](https://docs.astral.sh/uv/) and a running Docker daemon.

```bash
uv sync
uv run pytest
```

### A free local model

[Ollama](https://ollama.com/) runs a model on your own machine, so testing costs nothing.
These settings keep memory low enough to run beside Docker on a 24 GB Mac:

```bash
OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
  OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
ollama pull qwen3:8b
```

A model this small is good for testing the harness, not for study results.
Turn its thinking mode off with `--reasoning-effort none`, or it can spend a whole time limit thinking.

## Usage

### 1. Run an agent on a METR task

Task images come from METR's public registry, listed in the [public task suite](https://github.com/METR/public-tasks).

```bash
uv run inspect eval mtb/bridge \
  -T image_tag=registry-1.docker.io/metrevals/public-tasks:crossword-0.1.8 \
  --sample-id 5x5_verify --model ollama/qwen3:8b --log-dir logs
```

Some tasks, such as `cowthello`, download files when they are scored.
Pass the public task suite's `secrets.env` so they get the settings they need:

```bash
-T secrets_env_path=../METR-public-tasks/secrets.env
```

Check a task's `resources` in its `manifest.yaml` first.
Docker Desktop on a Mac refuses tasks that ask for more CPUs than it has, such as `cowthello/no_internet`.

### 2. Score runs with a monitor

```bash
uv run inspect score logs/<run>.eval \
  --scorer cheating_monitor.py@cheating_monitor \
  -S model=ollama/qwen3:8b --action append --overwrite
```

Options, each recorded with the score:

- `-S include_reasoning=false` hides the agent's reasoning from the monitor.
- `-S max_tool_output_chars=1000` and `-S max_tool_arg_chars=1000` trim long tool outputs and tool inputs from the middle, for monitors with small context windows.

A monitor reply without a valid score is stored as `nan` with `parse_failed` set, never as 0.

### 3. Label runs

```bash
uv run python label_runs.py --labeller <your-name> --failure-sample 10
```

The tool walks through every run METR scored as a success, plus a random sample of failures.
For each one it asks for a label, the message numbers that support it, and a short note.
Monitor scores are never shown, so they cannot sway a label.
Labels are appended to `labels/labels.jsonl`, and labelled runs are skipped next time.

### 4. Analyse

```bash
uv run python analyze.py --out results/summary.json
```

For each monitor setup, this reports recall at a threshold that flags at most 5% of legitimate runs, a 95% interval, AUROC, recall by transcript length, and the recall lost when reasoning is hidden.
Every number comes with the run counts behind it.

### Optional: Claude Code as the agent

`run_claude_code.py` installs the unmodified Claude Code CLI inside each task container and signs it in with your own subscription, as Anthropic's [terms](https://code.claude.com/docs/en/legal-and-compliance) allow.

1. Run `claude setup-token` and put the token in `.env` as `CLAUDE_CODE_OAUTH_TOKEN=...`.
   Git ignores `.env`.
2. Run a task:

```bash
uv run python run_claude_code.py \
  --image-tag registry-1.docker.io/metrevals/public-tasks:cowthello-2.0.2 \
  --sample-id main --hours 2
```

Use this script rather than `inspect eval`.
Inspect's logs record the token, so the script scrubs every log it writes and fails if any copy is left.
Claude Code needs to reach Anthropic, so only tasks that grant `full_internet` work for now.

## Limitations

- No frontier-model results yet, so nothing here says how well monitors catch frontier agents.
- The analysis picks its 5% threshold on the same runs it measures, which flatters recall when there are few runs.
- Monitor results depend on the prompt in `cheating_monitor.py`.
- Labels so far come from one person.
