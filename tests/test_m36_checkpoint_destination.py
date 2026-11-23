"""M36 checkpoint destination preflight requires no torch or hardware."""
from types import SimpleNamespace

import pytest
from aetherion.checkpoint_destination import check_checkpoint_destination, GIB


def check(tmp_path, **kwargs):
    options = dict(output=str(tmp_path / "pilot.pt"), steps=100,
                   save_every=30, retain_step_checkpoints=False,
                   min_free_gib=1, disk_usage=lambda _: SimpleNamespace(free=2 * GIB))
    options.update(kwargs)
    return check_checkpoint_destination(**options)


def test_new_output_with_operator_space_floor(tmp_path):
    r = check(tmp_path)
    assert r["observed_free_bytes"] == 2 * GIB
    assert r["min_requested_free_bytes"] == GIB
    assert r["result"] == "point_in_time_destination_check_passed"


def test_nonretained_existing_output_rejected_unless_explicit_override(tmp_path):
    p = tmp_path / "pilot.pt"
    p.write_bytes(b"old")
    with pytest.raises(ValueError, match="already exists"):
        check(tmp_path)
    assert check(tmp_path, allow_overwrite_output=True)["operator_approved_overwrite"]
    assert p.read_bytes() == b"old"  # preflight never mutates data


def test_retained_collision_checks_exact_save_schedule(tmp_path):
    (tmp_path / "pilot-step00000012.pt").write_bytes(b"other-step")
    (tmp_path / "pilot-step00000060.pt").write_bytes(b"collision")
    with pytest.raises(ValueError, match="scheduled snapshot"):
        check(tmp_path, retain_step_checkpoints=True)
    (tmp_path / "pilot-step00000060.pt").unlink()
    assert check(tmp_path, retain_step_checkpoints=True)["existing_candidate_files_examined"] == 1
    (tmp_path / "pilot-step00000100.pt").write_bytes(b"final")
    with pytest.raises(ValueError, match="scheduled snapshot"):
        check(tmp_path, retain_step_checkpoints=True)


def test_retained_final_only_and_default_interval(tmp_path):
    (tmp_path / "pilot-step00000050.pt").write_bytes(b"periodic")
    with pytest.raises(ValueError, match="scheduled snapshot"):
        check(tmp_path, retain_step_checkpoints=True, save_every=None)
    assert check(tmp_path, retain_step_checkpoints=True, save_every=0)["result"].endswith("passed")


def test_existing_directory_and_symlink_never_replaced(tmp_path):
    folder = tmp_path / "folder.pt"
    folder.mkdir()
    with pytest.raises(ValueError, match="symlink or directory"):
        check(tmp_path, output=str(folder), allow_overwrite_output=True)
    link = tmp_path / "linked.pt"
    try:
        link.symlink_to(folder)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink not supported")
    with pytest.raises(ValueError, match="symlink or directory"):
        check(tmp_path, output=str(link), allow_overwrite_output=True)


def test_insufficient_disk_and_missing_anchor(tmp_path):
    with pytest.raises(ValueError, match="Insufficient free"):
        check(tmp_path, disk_usage=lambda _: SimpleNamespace(free=GIB-1))
    assert check(tmp_path, output=str(tmp_path / "notcreated" / "pilot.pt"))["result"].endswith("passed")


@pytest.mark.parametrize("change, error", [
    ({"output": ""}, "nonempty"),
    ({"steps": 0}, "positive integer"),
    ({"save_every": -2}, "nonnegative"),
    ({"save_every": True}, "nonnegative"),
    ({"min_free_gib": 0}, "finite and positive"),
    ({"min_free_gib": float("nan")}, "finite and positive"),
    ({"min_free_gib": float("inf")}, "finite and positive"),
    ({"min_free_gib": True}, "finite and positive"),
    ({"retain_step_checkpoints": 2}, "boolean"),
    ({"allow_overwrite_output": 1}, "boolean"),
    ({"retain_step_checkpoints": True, "allow_overwrite_output": True}, "invalid with retained"),
])
def test_invalid_inputs(tmp_path, change, error):
    with pytest.raises(ValueError, match=error):
        check(tmp_path, **change)


def test_disk_usage_error_fails_closed(tmp_path):
    def failing(_):
        raise OSError("unmounted")
    with pytest.raises(ValueError, match="Cannot read"):
        check(tmp_path, disk_usage=failing)


def test_execute_requires_disk_floor_before_hardware_and_corpus(monkeypatch):
    import sys
    import pilot_train
    called = []
    monkeypatch.setattr(pilot_train, "validate_pilot_benchmark",
                        lambda *a, **kw: called.append("benchmark"))
    monkeypatch.setattr(pilot_train, "pilot_command",
                        lambda *a, **kw: called.append("corpus"))
    monkeypatch.setattr(pilot_train.subprocess, "run",
                        lambda *a, **kw: called.append("train"))
    monkeypatch.setattr(sys, "argv", [
        "pilot_train.py", "--data-dir", "fixture", "--tokenizer", "fixture",
        "--review", "fixture", "--allow-license", "CC0-1.0",
        "--steps", "2", "--schedule-steps", "4",
        "--max-total-tokens", "8192", "--max-checkpoint-writes", "2",
        "--device", "cuda:0", "--output", "fixture.pt", "--execute",
        "--benchmark-report", "fixture.json", "--max-peak-reserved-gib", "12",
    ])
    with pytest.raises(SystemExit) as exc:
        pilot_train.main()
    assert exc.value.code == 2
    assert called == []


def test_execute_disk_gate_blocks_training_after_other_gates(monkeypatch):
    import sys
    import pilot_train
    called = []
    monkeypatch.setattr(pilot_train, "validate_pilot_benchmark",
                        lambda *a, **kw: {"device": "cuda:0"})
    monkeypatch.setattr(pilot_train, "pilot_command",
                        lambda *a, **kw: called.append("corpus") or ["train.py"])
    monkeypatch.setattr(pilot_train, "check_live_accelerator",
                        lambda *a, **kw: called.append("live") or {"result": "test"})
    def fail_storage(*a, **kw):
        called.append("storage")
        raise ValueError("Insufficient free checkpoint disk space")
    monkeypatch.setattr(pilot_train, "check_checkpoint_destination", fail_storage)
    monkeypatch.setattr(pilot_train.subprocess, "run",
                        lambda *a, **kw: called.append("train"))
    monkeypatch.setattr(sys, "argv", [
        "pilot_train.py", "--data-dir", "fixture", "--tokenizer", "fixture",
        "--review", "fixture", "--allow-license", "CC0-1.0",
        "--steps", "2", "--schedule-steps", "4",
        "--max-total-tokens", "8192", "--max-checkpoint-writes", "2",
        "--min-free-checkpoint-gib", "2",
        "--device", "cuda:0", "--output", "fixture.pt", "--execute",
        "--benchmark-report", "fixture.json", "--max-peak-reserved-gib", "12",
    ])
    with pytest.raises(SystemExit) as exc:
        pilot_train.main()
    assert exc.value.code == 2
    assert called == ["corpus", "live", "storage"]
