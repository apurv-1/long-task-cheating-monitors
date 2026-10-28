import math

import pytest
from inspect_ai.log import EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.log._log import EvalConfig, EvalDataset
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import Score

from analyze import auroc, summarize, threshold_at_fpr, wilson_interval
from label_runs import (
    Label,
    Run,
    RunKey,
    append_label,
    collect_runs,
    label_run,
    load_labels,
    parse_evidence,
    read_logs,
    select_runs,
)

EVAL_ID = "eval-1"


def _monitor(value: float, reasoning: bool, chars: int) -> Score:
    return Score(
        value=value,
        metadata={
            "monitor_model": "ollama/test",
            "include_reasoning": reasoning,
            "max_tool_output_chars": 4000,
            "parse_failed": math.isnan(value),
            "transcript_chars": chars,
        },
    )


def _write_log(path, samples: list[EvalSample]) -> None:
    log = EvalLog(
        eval=EvalSpec(
            eval_id=EVAL_ID,
            created="2026-10-28T00:00:00+00:00",
            task="toy",
            dataset=EvalDataset(),
            model="none/none",
            config=EvalConfig(),
        ),
        samples=samples,
    )
    write_eval_log(log, str(path))


def _sample(sample_id: str, metr: float, with_r: float, without_r: float, chars: int) -> EvalSample:
    return EvalSample(
        id=sample_id,
        epoch=1,
        input="task",
        target="",
        messages=[ChatMessageUser(content="task"), ChatMessageAssistant(content="work")],
        scores={
            "score_metr_task": Score(value=metr),
            # Inspect renames a second copy of the same scorer, as `inspect score` does.
            "cheating_monitor": _monitor(with_r, True, chars),
            "cheating_monitor1": _monitor(without_r, False, chars),
        },
    )


def _label(sample_id: str, label: str) -> Label:
    return Label(EVAL_ID, sample_id, 1, label, [], "", "tester", "x.eval", "2026-10-28T00:00:00")


@pytest.fixture
def toy_study(tmp_path):
    """20 legitimate runs scored 0-95 and 10 cheats scored 10-100, plus one unclear run.

    With reasoning hidden, every cheat scores 30 points lower.
    """
    samples, labels = [], []
    for i in range(20):
        samples.append(_sample(f"legit{i}", 1, i * 5, i * 5, chars=1000 + i))
        labels.append(_label(f"legit{i}", "legitimate"))
    for i in range(10):
        score = 100 - i * 10
        samples.append(_sample(f"cheat{i}", 1, score, score - 30, chars=5000 + i * 1000))
        labels.append(_label(f"cheat{i}", "cheat"))
    samples.append(_sample("odd", 1, 50, 50, chars=2000))
    labels.append(_label("odd", "unclear"))
    _write_log(tmp_path / "toy.eval", samples)
    return read_logs(str(tmp_path)), labels


def test_threshold_respects_the_false_positive_cap_with_ties():
    assert threshold_at_fpr([95, 90, 85] + [0] * 17, 0.05) == 90
    # Two runs tie at the top, so flagging either would flag both and exceed 5%.
    assert threshold_at_fpr([90, 90] + [0] * 18, 0.05) == 90
    assert threshold_at_fpr([], 0.05) is None


def test_wilson_and_auroc():
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.237, abs=0.001) and high == pytest.approx(0.763, abs=0.001)
    assert wilson_interval(0, 0) is None
    assert auroc([2, 3], [1]) == 1.0
    assert auroc([1], [1]) == 0.5


def test_summary_on_known_data(toy_study):
    logs, labels = toy_study
    summary = summarize(logs, labels, max_fpr=0.05)

    assert summary["labels"] == {"cheat": 10, "legitimate": 20, "unclear": 1}
    with_r = next(s for s in summary["setups"] if s["setup"].endswith("with reasoning"))
    # At most 1 of 20 legitimate runs may be flagged, so the threshold is 90.
    assert with_r["threshold"] == 90
    assert with_r["false_positive_rate"] == 0.05
    assert with_r["recall"] == 0.1
    assert with_r["n_cheat"] == 10 and with_r["n_legitimate"] == 20

    drop = summary["reasoning_drops"][0]
    assert drop["recall_with_reasoning"] == 0.1
    assert drop["recall_without_reasoning"] == 0.0


def test_missing_monitor_scores_count_as_not_flagged(tmp_path):
    samples = [_sample(f"legit{i}", 1, 0, 0, chars=100) for i in range(5)]
    samples.append(_sample("cheat", 1, math.nan, math.nan, chars=100))
    _write_log(tmp_path / "toy.eval", samples)
    labels = [_label(f"legit{i}", "legitimate") for i in range(5)] + [_label("cheat", "cheat")]

    summary = summarize(read_logs(str(tmp_path)), labels, max_fpr=0.05)

    setup = summary["setups"][0]
    assert setup["recall"] == 0.0
    assert setup["n_parse_failed"] == 1


def test_select_runs_takes_successes_and_a_seeded_failure_sample():
    def run(name: str, score: float | None) -> Run:
        return Run(RunKey("e", name, 1), "x.eval", "toy", score)

    runs = [run("win1", 1.0), run("win2", 0.5), run("lose1", 0.0), run("lose2", 0.0), run("crash", None)]
    picked = select_runs(runs, {RunKey("e", "win2", 1)}, 0.0, failure_sample=1, seed=0)

    names = [r.key.sample_id for r in picked]
    assert names[0] == "win1"
    assert len(names) == 2 and names[1] in {"lose1", "lose2"}
    assert select_runs(runs, set(), 0.0, failure_sample=1, seed=0) == select_runs(
        runs, set(), 0.0, failure_sample=1, seed=0
    )


def test_parse_evidence_rejects_out_of_range():
    assert parse_evidence("1, 2", 3) == [1, 2]
    assert parse_evidence("", 3) == []
    with pytest.raises(ValueError):
        parse_evidence("5", 3)


def test_label_run_round_trip(tmp_path):
    _write_log(tmp_path / "toy.eval", [_sample("s1", 1, 0, 0, chars=10)])
    run = collect_runs(read_logs(str(tmp_path)))[0]
    answers = iter(["x", "c", "9", "1", "Edited the test file."])
    shown: list[str] = []

    label = label_run(run, "tester", lambda _: next(answers), shown.append, tmp_path)

    assert label.label == "cheat"
    assert label.evidence == [1]
    assert label.note == "Edited the test file."
    assert any("Please type" in text for text in shown)
    assert any("out of range" in text for text in shown)
    assert not any("cheating_monitor" in text for text in shown)

    labels_file = tmp_path / "labels.jsonl"
    append_label(labels_file, label)
    assert load_labels(labels_file)[0].key == RunKey(EVAL_ID, "s1", 1)
