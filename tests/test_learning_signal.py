"""M16 checks a true optimizer learning signal, not model intelligence."""
import json
import math
import os
import subprocess
import sys

import pytest
import torch

from overfit_check import overfit_probe


def test_synthetic_overfit_reduces_loss():
    torch.set_num_threads(1)
    first = overfit_probe(steps=40, seed=17)
    assert first["passed"]
    assert first["tokens_seen"] == 320
    assert first["unique_training_sequences"] == 1
    assert first["checkpoint_saved"] is False
    assert math.isfinite(first["final_nll"])
    assert 0 < first["final_nll"] < 0.5 * first["initial_nll"]
    second = overfit_probe(steps=40, seed=17)
    assert first == second


def test_probe_rejects_invalid_settings():
    with pytest.raises(ValueError, match="steps"):
        overfit_probe(steps=0)
    with pytest.raises(ValueError, match="seed"):
        overfit_probe(steps=1, seed=-1)


def test_overfit_cli_json_report():
    torch.set_num_threads(1)
    done = subprocess.run([sys.executable, "overfit_check.py", "--steps", "40"],
                          capture_output=True, text=True, check=True,
                          env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    report = json.loads(done.stdout)
    assert report["passed"] is True
    assert report["diagnostic"] == "synthetic_training_overfit"
