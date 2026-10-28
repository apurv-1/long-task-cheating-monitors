"""Hand-label agent runs as cheat, legitimate, or unclear.

    uv run python label_runs.py --labeller apurv

It walks through every run METR scored as a success, plus an optional random sample of
failures, and appends one JSON line per label to labels/labels.jsonl.
Monitor scores are never shown, so they cannot sway a label.
Runs that already have a label are skipped, so you can stop and resume at any time.
"""

import argparse
import json
import random
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

from cheating_monitor import format_transcript

METR_SCORER = "score_metr_task"
LABELS = {"c": "cheat", "l": "legitimate", "u": "unclear"}


@dataclass(frozen=True)
class RunKey:
    """Identifies one agent run, independent of where its log file lives."""

    eval_id: str
    sample_id: str
    epoch: int


@dataclass
class Run:
    key: RunKey
    log_path: str
    task: str
    metr_score: float | None


@dataclass
class Label:
    eval_id: str
    sample_id: str
    epoch: int
    label: str
    evidence: list[int]
    note: str
    labeller: str
    log_file: str
    labelled_at: str

    @property
    def key(self) -> RunKey:
        return RunKey(self.eval_id, self.sample_id, self.epoch)


def read_logs(log_dir: str) -> list[EvalLog]:
    return [read_eval_log(info) for info in list_eval_logs(log_dir)]


def metr_score(scores: dict | None) -> float | None:
    score = (scores or {}).get(METR_SCORER)
    if score is None:
        return None
    try:
        return float(score.value)
    except (TypeError, ValueError):
        return None


def collect_runs(logs: list[EvalLog]) -> list[Run]:
    runs = []
    for log in logs:
        for sample in log.samples or []:
            runs.append(
                Run(
                    key=RunKey(log.eval.eval_id, str(sample.id), sample.epoch),
                    log_path=log.location,
                    task=log.eval.task,
                    metr_score=metr_score(sample.scores),
                )
            )
    return runs


def select_runs(
    runs: list[Run],
    labelled: set[RunKey],
    success_threshold: float,
    failure_sample: int,
    seed: int,
) -> list[Run]:
    """Every unlabelled success, plus a seeded random sample of unlabelled failures.

    A run counts as a success when its METR score is above `success_threshold`.
    Runs without a METR score, such as crashed runs, are left out.
    """
    scored = [run for run in runs if run.metr_score is not None and run.key not in labelled]
    successes = [run for run in scored if run.metr_score > success_threshold]
    failures = [run for run in scored if run.metr_score <= success_threshold]
    sampled = random.Random(seed).sample(failures, min(failure_sample, len(failures)))
    return successes + sampled


def load_labels(path: Path) -> list[Label]:
    if not path.exists():
        return []
    with path.open() as file:
        return [Label(**json.loads(line)) for line in file if line.strip()]


def append_label(path: Path, label: Label) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(asdict(label)) + "\n")


def parse_evidence(text: str, message_count: int) -> list[int]:
    """Parse comma-separated message numbers, rejecting any outside the transcript."""
    numbers = [int(part) for part in text.replace(" ", "").split(",") if part]
    bad = [n for n in numbers if not 0 <= n < message_count]
    if bad:
        raise ValueError(f"Message numbers out of range 0-{message_count - 1}: {bad}")
    return numbers


def label_run(
    run: Run,
    labeller: str,
    ask: Callable[[str], str],
    show: Callable[[str], None],
    transcript_dir: Path,
) -> Label | None:
    """Ask for one label. Returns None to skip; raises KeyboardInterrupt to quit."""
    sample = next(
        s
        for s in read_eval_log(run.log_path).samples or []
        if str(s.id) == run.key.sample_id and s.epoch == run.key.epoch
    )
    transcript = format_transcript(sample.messages, max_tool_output_chars=0, include_reasoning=True)
    transcript_file = transcript_dir / f"{run.key.eval_id}_{run.key.sample_id}_{run.key.epoch}.txt"
    transcript_file.write_text(transcript)

    show(
        f"\nTask: {run.task}  sample: {run.key.sample_id}  epoch: {run.key.epoch}"
        f"  METR score: {run.metr_score}\n"
        f"Log: {run.log_path}\n"
        f"Full transcript: {transcript_file}\n"
        f"Messages are numbered [0] to [{len(sample.messages) - 1}]."
    )

    while True:
        choice = ask("Label: [c]heat, [l]egitimate, [u]nclear, [s]kip, [q]uit: ").strip().lower()
        if choice == "q":
            raise KeyboardInterrupt
        if choice == "s":
            return None
        if choice in LABELS:
            break
        show("Please type c, l, u, s, or q.")

    while True:
        try:
            evidence = parse_evidence(
                ask("Message numbers that support this label (comma separated): "),
                len(sample.messages),
            )
            break
        except ValueError as error:
            show(str(error))

    return Label(
        eval_id=run.key.eval_id,
        sample_id=run.key.sample_id,
        epoch=run.key.epoch,
        label=LABELS[choice],
        evidence=evidence,
        note=ask("Note (what happened, in one or two sentences): ").strip(),
        labeller=labeller,
        log_file=Path(run.log_path).name,
        labelled_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeller", required=True, help="Your name, saved with each label.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--labels", default="labels/labels.jsonl")
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.0,
        help="A run is a success when its METR score is above this.",
    )
    parser.add_argument(
        "--failure-sample",
        type=int,
        default=0,
        help="Also label this many randomly chosen failed runs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    labels_path = Path(args.labels)
    labelled = {label.key for label in load_labels(labels_path)}
    runs = select_runs(
        collect_runs(read_logs(args.log_dir)),
        labelled,
        args.success_threshold,
        args.failure_sample,
        args.seed,
    )
    print(f"{len(runs)} runs to label. Monitor scores are hidden on purpose.")

    transcript_dir = Path(tempfile.mkdtemp(prefix="transcripts-"))
    done = 0
    try:
        for number, run in enumerate(runs, 1):
            print(f"\n=== Run {number} of {len(runs)} ===")
            label = label_run(run, args.labeller, input, print, transcript_dir)
            if label is not None:
                append_label(labels_path, label)
                done += 1
    except KeyboardInterrupt:
        print("\nStopped. Your labels so far are saved.")
    print(f"Saved {done} labels to {labels_path}.")


if __name__ == "__main__":
    main()
