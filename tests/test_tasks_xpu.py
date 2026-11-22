import json
import subprocess
import sys

import pytest
import torch

from aetherion.devices import resolve_device, xpu_available
from aetherion.task_eval import score, summarize, load_tasks
from benchmark_training import benchmark


def test_numeric():
    task = {"kind": "numeric", "answer": "5/8"}
    assert score(task, "0.625")["correct"]
    assert not score(task, "5/7")["correct"]
    assert not score(task, "1/0")["correct"]
    assert not score(task, "answer: 5/8")["correct"]


def test_python_not_executed(tmp_path):
    task = {"kind": "python", "answer": "def f(x):\n    return x*x"}
    assert score(task, "def f(x): return x * x") == {"valid_syntax": True, "ast_exact": True}
    assert score(task, "return 1") == {"valid_syntax": False, "ast_exact": False}
    target = tmp_path / "never-created"
    result = score(task, f"open({str(target)!r}, 'w').write('x')")
    assert result["valid_syntax"] and not result["ast_exact"]
    assert not target.exists()


def test_offline_cli(tmp_path):
    tasks, _ = load_tasks("data/eval_smoke_tasks.jsonl")
    path = tmp_path / "preds.jsonl"
    path.write_text("".join(json.dumps({"id":t["id"], "prediction":t["answer"]})+"\n" for t in tasks))
    result = subprocess.run([sys.executable, "evaluate_tasks.py", "--tasks",
                             "data/eval_smoke_tasks.jsonl", "--predictions", str(path)],
                            check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["by_kind"]["numeric"]["correct_rate"] == 1.0
    assert report["by_kind"]["python"]["ast_exact_rate"] == 1.0
    with pytest.raises(ValueError, match="ids"):
        summarize(tasks, {})


def test_auto_xpu(monkeypatch):
    import aetherion.devices as devices
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(devices, "xpu_available", lambda: True)
    assert resolve_device("auto") == "xpu"
    monkeypatch.setattr(devices, "xpu_available", lambda: False)
    with pytest.raises(ValueError, match="XPU unavailable"):
        resolve_device("xpu")


@pytest.mark.skipif(not xpu_available(), reason="Requires actual Intel GPU and XPU PyTorch")
def test_physical_xpu_smoke():
    result = benchmark(device="xpu", context=8, steps=1, warmup=0)
    assert result["tokens"] == 8 and result["tokens_per_second"] > 0
    assert result["xpu_peak_allocated_bytes"] > 0
