"""M34: injected hardware mocks, never claim access to a real accelerator."""
from types import SimpleNamespace

import pytest
from aetherion.live_device_gate import check_live_accelerator

GIB = 1024**3


class FakeGpu:
    def __init__(self, *, name="synthetic CUDA model", available=True, count=1,
                 capacity=16*GIB, free=12*GIB, total=16*GIB):
        self.name = name
        self.available = available
        self.count = count
        self.capacity = capacity
        self.free = free
        self.total = total

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def get_device_name(self, index):
        return self.name

    def get_device_properties(self, index):
        return SimpleNamespace(total_memory=self.capacity)

    def mem_get_info(self, index):
        return self.free, self.total


def report(kind="cuda"):
    return {"device": f"{kind}:0", "hardware": "synthetic CUDA model",
            "sampled_peak_reserved_bytes": 8*GIB,
            "sampled_peak_allocated_bytes": 7*GIB}


def validate(gpu=None, *, kind="cuda", **patch):
    backend = FakeGpu(**patch) if gpu is None else gpu
    return check_live_accelerator(report(kind), device=f"{kind}:0",
                                  torch_backend=SimpleNamespace(**{kind: backend}))


def test_exact_live_match_and_headroom():
    result = validate()
    assert result["result"] == "point_in_time_preflight_passed"
    assert result["required_free_bytes"] == 10*GIB
    assert result["live_free_bytes"] == 12*GIB
    assert "cannot authenticate" in result["limitations"]


def test_xpu_uses_exact_same_checks_when_memory_api_supported():
    assert validate(kind="xpu")["device"] == "xpu:0"


@pytest.mark.parametrize("changes, error", [
    ({"available": False}, "not available"),
    ({"count": 0}, "not available"),
    ({"name": "different device"}, "identity differs"),
    ({"capacity": 6*GIB}, "exceeds device capacity"),
    ({"free": 9*GIB}, "Insufficient free"),
    ({"free": -1}, "nonnegative integer"),
    ({"free": 20*GIB}, "accounting is invalid"),
    ({"total": 17*GIB}, "exceeds device capacity"),
])
def test_bad_device_rejected(changes, error):
    with pytest.raises(ValueError, match=error):
        validate(**changes)


def test_missing_live_api_rejected():
    fake = FakeGpu()
    fake.mem_get_info = None
    with pytest.raises(ValueError, match="APIs are unavailable"):
        validate(fake)


def test_mismatched_device_or_peak_rejected():
    with pytest.raises(ValueError, match="matching indexed"):
        check_live_accelerator(report(), device="cuda:1", torch_backend=SimpleNamespace(cuda=FakeGpu()))
    with pytest.raises(ValueError, match="matching indexed"):
        check_live_accelerator(report(), device="cpu", torch_backend=SimpleNamespace(cuda=FakeGpu()))
    altered = report()
    altered["sampled_peak_reserved_bytes"] = 20*GIB
    with pytest.raises(ValueError, match="exceeds live accelerator capacity"):
        check_live_accelerator(altered, device="cuda:0", torch_backend=SimpleNamespace(cuda=FakeGpu()))
    altered = report()
    altered["sampled_peak_allocated_bytes"] = 9*GIB
    with pytest.raises(ValueError, match="exceeds live accelerator capacity"):
        check_live_accelerator(altered, device="cuda:0", torch_backend=SimpleNamespace(cuda=FakeGpu()))
