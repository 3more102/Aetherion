"""M37: exclusive, fail-closed lock tests; no GPU or corpus required."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from aetherion.pilot_lock import pilot_execution_lock


def test_one_output_cannot_launch_twice(tmp_path):
    output = str(tmp_path / "pilot.pt")
    with pilot_execution_lock(output) as held:
        lock = Path(held["lock_directory"])
        assert lock.is_dir()
        meta = json.loads((lock / "owner.json").read_text())
        assert meta["schema_version"] == 1
        assert meta["pid"] == os.getpid()
        assert meta["output"] == output
        with pytest.raises(ValueError, match="already locked"):
            with pilot_execution_lock(output):
                pytest.fail("duplicate launcher acquired lock")
    assert not lock.exists()


def test_different_outputs_can_coexist_and_relative_alias_collides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pilot_execution_lock("pilot.pt") as first:
        with pytest.raises(ValueError, match="already locked"):
            with pilot_execution_lock(str(tmp_path / "pilot.pt")):
                pass
        with pilot_execution_lock("second.pt") as second:
            assert first["lock_directory"] != second["lock_directory"]
    assert not list(tmp_path.glob(".*.aetherion-pilot.lock"))


def test_exception_releases_only_owned_lock(tmp_path):
    path = str(tmp_path / "pilot.pt")
    with pytest.raises(RuntimeError, match="training failed"):
        with pilot_execution_lock(path):
            raise RuntimeError("training failed")
    with pilot_execution_lock(path):
        pass


def test_interrupted_lock_never_auto_reclaimed(tmp_path):
    path = str(tmp_path / "pilot.pt")
    stale = tmp_path / ".pilot.pt.aetherion-pilot.lock"
    stale.mkdir()
    with pytest.raises(ValueError, match="Investigate"):
        with pilot_execution_lock(path):
            pass
    assert stale.is_dir()


def test_altered_owner_metadata_keeps_lock(tmp_path):
    path = str(tmp_path / "pilot.pt")
    with pilot_execution_lock(path) as report:
        lock = Path(report["lock_directory"])
        (lock / "owner.json").write_text('{"token": "someone-else"}')
    assert lock.is_dir()
    (lock / "owner.json").unlink()
    lock.rmdir()


@pytest.mark.parametrize("path", ["", "  ", ".", "/"])
def test_invalid_paths_are_rejected(path):
    with pytest.raises(ValueError, match="nonempty|name a file"):
        with pilot_execution_lock(path):
            pass


def configured_pilot(monkeypatch, tmp_path, calls, *, fail_subprocess=False):
    import pilot_train
    output = tmp_path / "pilot.pt"
    monkeypatch.setattr(pilot_train, "validate_pilot_benchmark",
                        lambda *a, **kw: {"kind": "synthetic_test"})
    monkeypatch.setattr(pilot_train, "pilot_command",
                        lambda *a, **kw: calls.append("corpus") or ["mock_train"])
    monkeypatch.setattr(pilot_train, "check_live_accelerator",
                        lambda *a, **kw: calls.append("live") or {"kind": "synthetic_test"})
    monkeypatch.setattr(pilot_train, "check_checkpoint_destination",
                        lambda *a, **kw: calls.append("storage") or {"kind": "synthetic_test"})

    def run_mock(*args, **kw):
        calls.append("training")
        assert (tmp_path / ".pilot.pt.aetherion-pilot.lock").is_dir()
        if fail_subprocess:
            raise subprocess.CalledProcessError(9, "mock_train")

    monkeypatch.setattr(pilot_train.subprocess, "run", run_mock)
    monkeypatch.setattr(sys, "argv", [
        "pilot_train.py", "--data-dir", "fixture", "--tokenizer", "fixture",
        "--review", "fixture", "--allow-license", "CC0-1.0",
        "--steps", "2", "--schedule-steps", "4", "--device", "cuda:0",
        "--output", str(output), "--execute",
        "--max-total-tokens", "8192", "--max-checkpoint-writes", "2",
        "--min-free-checkpoint-gib", "0.001",
        "--benchmark-report", "fixture.json", "--max-peak-reserved-gib", "8",
    ])
    return pilot_train, output


def test_launch_holds_lock_through_storage_and_child_exit(tmp_path, monkeypatch):
    calls = []
    pilot_train, output = configured_pilot(monkeypatch, tmp_path, calls)
    pilot_train.main()
    assert calls == ["corpus", "live", "storage", "training"]
    assert not (tmp_path / ".pilot.pt.aetherion-pilot.lock").exists()


def test_second_launch_refused_before_storage_or_training(tmp_path, monkeypatch):
    calls = []
    pilot_train, output = configured_pilot(monkeypatch, tmp_path, calls)
    with pilot_execution_lock(str(output)):
        with pytest.raises(SystemExit) as exc:
            pilot_train.main()
        assert exc.value.code == 2
        assert calls == ["corpus", "live"]


def test_failed_child_releases_lock_for_diagnosis_and_retry(tmp_path, monkeypatch):
    calls = []
    pilot_train, _ = configured_pilot(
        monkeypatch, tmp_path, calls, fail_subprocess=True)
    with pytest.raises(subprocess.CalledProcessError):
        pilot_train.main()
    assert calls[-1] == "training"
    assert not (tmp_path / ".pilot.pt.aetherion-pilot.lock").exists()
