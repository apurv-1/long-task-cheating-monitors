# Can LLM monitors catch cheating on long agent tasks?

Draft one-pager, Apurv Gupta.

## Question

METR's May 2026 risk report found that on tasks longer than 8 hours, at least 16% of runs scored as successes were illegitimate on manual review.
METR's 2025 reward hacking post found cheating with high-score heuristics, a lightly prompted Claude monitor, and manual review.
That post reports no recall or precision for the monitor.
Published monitor comparisons, such as arXiv 2609.19101, use SWE-bench-scale runs rather than multi-hour ones.

So the question is: on multi-hour agent runs, how much cheating does an LLM monitor catch, and at what false positive rate?
A second question is how much the monitor depends on seeing the agent's reasoning, since that reasoning may not stay readable.
BenchShield, arXiv 2609.11028, already detects cheating from sandbox-side evidence at 96% accuracy, so this study sticks to transcript monitors.

## Why it matters for loss of control

Cheating is the most common misaligned behavior seen in real agent runs, and METR's rogue deployment assessment uses it as evidence about motive.
Labs rely on the same kind of LLM monitor to catch worse behavior.
If monitors miss plain cheating on long runs, that is a warning sign for how well they would catch something subtler.
There is also a practical use: time horizon numbers depend on catching illegitimate successes, which is still done by hand.

## Setup

Tasks: multi-hour METR public tasks run through the Inspect task bridge, for example Payments Over Timezones and Build Expert AI for Novel Board Game.
Agents: 2-3 frontier models, about 150 runs in total.
Labels: I read every run that scores as a success, plus a random sample of failures, and label each one as cheat, legitimate, or ambiguous, using METR's definition.
If I can find a second labeller, they label a subset so I can report agreement.
Monitors: an LLM monitor that reads the whole transcript, run with 2 monitor models, once with the agent's visible reasoning and once without it.
Metrics: recall at a 5% false positive rate, and how recall changes as transcripts get longer.

## Predictions (written before any runs)

- Transcript-only recall at 5% false positive rate: TODO.
- Recall lost when the monitor cannot see the agent's reasoning: TODO.
- Recall on the longest third of transcripts compared with the shortest third: TODO.

## When I drop or change the project

- Someone, including METR, has already published this.
- A 20-run pilot produces fewer than 3 clear cheats, so positives would cost too much to collect.
- In the pilot, the transcript-only monitor already catches nearly every cheat at a low false positive rate.
  In that case I write a short note instead of a full study.

## Limits

- The sample is small, and most labels come from one person.
- Monitor results depend on the prompts, so all prompts will be published.
- Cheating rates depend on the agent scaffold and may not carry over to lab-internal setups.
- This does not test agents that deliberately try to evade the monitor.

## Cost and time

Pilot: about $200 of API credits.
Full study: roughly $2-4k, to be firmed up from pilot token counts.
Time: 6-8 weeks alongside my day job.
Code, prompts, labels, and transcripts will be public.
