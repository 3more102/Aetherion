"""M35 budget calculations use no model, dataset or hardware."""
import pytest

from aetherion.pilot_budget import check_pilot_budget


def budget(**changes):
    args = dict(steps=100, context=256, batch_size=2, accumulation=4,
                schedule_steps=200, save_every=30,
                retain_step_checkpoints=True, max_total_tokens=204800,
                max_checkpoint_writes=4)
    args.update(changes)
    return check_pilot_budget(**args)


def test_exact_positions_and_write_count():
    report = budget()
    assert report["max_planned_sampled_token_positions"] == 204800
    assert report["sampled_positions_per_optimizer_step"] == 2048
    assert report["max_planned_checkpoint_writes"] == 4
    assert report["max_planned_checkpoint_files"] == 4
    assert report["result"] == "requested_work_within_operator_limits"


def test_periodic_interval_boundary_and_final_write():
    assert budget(steps=90, max_total_tokens=200000)["max_planned_checkpoint_writes"] == 3
    assert budget(steps=91, max_total_tokens=200000)["max_planned_checkpoint_writes"] == 4
    assert budget(steps=5, save_every=10, max_total_tokens=12000)["max_planned_checkpoint_writes"] == 1
    assert budget(steps=5, save_every=0, max_total_tokens=12000)["max_planned_checkpoint_writes"] == 1
    assert budget(steps=5, save_every=None, max_total_tokens=12000)["save_every_steps"] == 50


def test_non_retained_saves_overwrite_but_writes_still_capped():
    report = budget(retain_step_checkpoints=False)
    assert report["max_planned_checkpoint_files"] == 1
    assert report["max_planned_checkpoint_writes"] == 4
    with pytest.raises(ValueError, match="checkpoint writes"):
        budget(retain_step_checkpoints=False, max_checkpoint_writes=3)


def test_resume_is_conservatively_full_target_not_remaining():
    record = budget(resume=True)
    assert record["max_planned_sampled_token_positions"] == 204800
    assert record["resume_requested"] is True
    assert "conservative" in record["limitations"]


@pytest.mark.parametrize("changes,error", [
    ({"max_total_tokens": 204799}, "token positions"),
    ({"max_checkpoint_writes": 3}, "checkpoint writes"),
    ({"schedule_steps": 99}, "exceed schedule"),
    ({"save_every": -1}, "save-every"),
    ({"save_every": True}, "save-every"),
    ({"steps": 0}, "positive integer"),
    ({"batch_size": False}, "positive integer"),
    ({"accumulation": 1.0}, "positive integer"),
    ({"max_total_tokens": 1.5}, "positive integer"),
    ({"max_checkpoint_writes": True}, "positive integer"),
    ({"retain_step_checkpoints": 1}, "boolean"),
])
def test_budget_rejections(changes, error):
    with pytest.raises(ValueError, match=error):
        budget(**changes)


def test_execute_missing_budget_rejects_before_benchmark_and_corpus(monkeypatch):
    import pilot_train
    import sys

    called = []
    monkeypatch.setattr(pilot_train, "validate_pilot_benchmark",
                        lambda *a, **kw: called.append("benchmark"))
    monkeypatch.setattr(pilot_train, "pilot_command",
                        lambda *a, **kw: called.append("corpus"))
    monkeypatch.setattr(pilot_train.subprocess, "run",
                        lambda *a, **kw: called.append("launched"))
    monkeypatch.setattr(sys, "argv", [
        "pilot_train.py", "--data-dir", "fake", "--tokenizer", "fake",
        "--review", "fake", "--allow-license", "CC0-1.0",
        "--steps", "10", "--schedule-steps", "20", "--output", "fake.pt",
        "--device", "cuda:0", "--execute",
        "--benchmark-report", "fake.json", "--max-peak-reserved-gib", "12",
    ])
    with pytest.raises(SystemExit) as exc:
        pilot_train.main()
    assert exc.value.code == 2
    assert called == []


def test_execute_over_budget_rejects_before_benchmark_and_corpus(monkeypatch):
    import pilot_train
    import sys

    called = []
    monkeypatch.setattr(pilot_train, "validate_pilot_benchmark",
                        lambda *a, **kw: called.append("benchmark"))
    monkeypatch.setattr(pilot_train, "pilot_command",
                        lambda *a, **kw: called.append("corpus"))
    monkeypatch.setattr(pilot_train.subprocess, "run",
                        lambda *a, **kw: called.append("launched"))
    monkeypatch.setattr(sys, "argv", [
        "pilot_train.py", "--data-dir", "fake", "--tokenizer", "fake",
        "--review", "fake", "--allow-license", "CC0-1.0",
        "--steps", "10", "--schedule-steps", "20", "--output", "fake.pt",
        "--device", "cuda:0", "--execute",
        "--benchmark-report", "fake.json", "--max-peak-reserved-gib", "12",
        "--max-total-tokens", "1", "--max-checkpoint-writes", "2",
    ])
    with pytest.raises(SystemExit) as exc:
        pilot_train.main()
    assert exc.value.code == 2
    assert called == []
