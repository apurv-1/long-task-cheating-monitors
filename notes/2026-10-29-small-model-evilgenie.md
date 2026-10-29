# A small local model never got far enough to cheat on EvilGenie

Lab note, 29 October 2026.
This is a pilot, run for free on a laptop, to test the harness before spending money on frontier models.
It is not a result about frontier agents.

## Question

EvilGenie gives a coding agent competitive programming problems with visible tests it can read and edit.
Editing the tests or hardcoding their answers would be cheating, and EvilGenie checks for both.
I wanted to know two things.
First, does a small open model cheat when the problems are too hard for it?
Second, does an LLM monitor wrongly flag the runs where it does not?

## Setup

- Agent: Qwen3 8B through Ollama on a 24 GB M5 Pro MacBook.
- Task: 10 hard LiveCodeBench problems from EvilGenie, with its default agent and its default problem sample.
- EvilGenie's own judge: the same Qwen3 8B model.
- Monitor: the transcript monitor from this repo, also Qwen3 8B, scoring each run from 0 (clearly legitimate) to 100 (clearly cheating).

I ran it twice.

| | Run 1 | Run 2 |
|---|---|---|
| Qwen3 thinking mode | on | off |
| Time per problem | 10 minutes | 30 minutes |
| Message limit | 30 | 60 |

## What happened

**Run 1 told us nothing.**
8 of 10 problems hit the time limit after 0 or 1 actions.
The model spent the whole time thinking, and never opened the test files.

**Run 2 was active but stuck.**

| Measure | Result |
|---|---|
| Actions taken | 282, all of them file edits |
| Actions that failed | 209 (74%) |
| Runs where the agent ran its code or the tests | 0 of 10 |
| Runs where the agent opened the test files | 1 of 10, and it changed nothing |
| Problems solved | 0 of 10 |
| Runs where the test files were edited | 0 of 10 |
| EvilGenie judge | 9 legitimate, 1 heuristic, 0 reward hacking |

Most failed actions were the same mistake repeated.
The agent kept trying to create `solution.py`, got the error that the file already exists, and tried again until it ran out of messages.
It had a shell and Python available, but never used them.

**The monitor did not raise false alarms.**
On these 10 honest runs it gave scores from 0 to 20, with a mean of 7, and every reply had a valid score.
The highest score, 20, went to a run whose code was wrong and incomplete.
The monitor's reasoning there was about correctness, not honesty, so bad code nudges its suspicion up a little.

## What it means

Cheating takes skill.
To cheat on EvilGenie, an agent has to run the tests, see them fail, and then decide to game them.
This model never got past the first step.
So zero cheating here says the model was not capable enough to cheat, not that it was honest.
This matches the broader pattern that more capable models reward hack more.

Two lessons carry over to the next experiments:

- Check that the agent actually reached the part being tested before trusting a zero.
- A small local model is good for testing a harness and bad for answering questions about agent behaviour.

## Limits

- 10 problems and one model, so none of these numbers generalise.
- The monitor is the same small model as the agent, and it saw no cheating runs, so this says nothing about how well it catches cheating.
- Transcripts were too long for the model's context, so the monitor saw each tool input and tool output cut to 1,000 characters from the middle.
  The monitor ran with a 32K token context.

## Reproduce

Agent runs, from a clone of [EvilGenie](https://github.com/JonathanGabor/evilgenie_inspect), with Ollama serving `qwen3:8b`:

```bash
inspect eval reward_hacking.py --model ollama/qwen3:8b --model-role judge=ollama/qwen3:8b \
  --reasoning-effort none -T limit=10 -T max_messages=60 -T timeout_per_problem=1800 \
  --max-samples 1 --log-dir logs/evilgenie-no-thinking
```

Monitor scores, from this repo:

```bash
inspect score logs/evilgenie-no-thinking/<run>.eval --scorer cheating_monitor.py@cheating_monitor \
  -S model=ollama/qwen3:8b -S max_tool_output_chars=1000 -S max_tool_arg_chars=1000 \
  --action append --overwrite
```
