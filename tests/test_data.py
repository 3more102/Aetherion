"""Provenance, exact-dedup and split-isolation regression tests."""
import json
from pathlib import Path
import pytest
from aetherion.data import prepare


def make_dataset(path: Path, count: int = 12):
    items = []
    for i in range(count):
        items.append({"id": f"d{i}", "source": f"source{i}", "license": "CC0-1.0",
                      "authorized": True, "text": f"Original test document {i}.\r\nText"})
    items.append({"id": "duplicate", "source": "reprint", "license": "CC0-1.0",
                  "authorized": True, "text": items[0]["text"]})
    path.write_text("".join(json.dumps(x) + "\n" for x in items), encoding="utf-8")


def test_preparation_is_reproducible_and_no_group_leaks(tmp_path):
    path = tmp_path / "input.jsonl"
    make_dataset(path)
    a, b = tmp_path / "a", tmp_path / "b"
    m1, m2 = prepare([path], a), prepare([path], b)
    assert m1 == m2
    assert m1["exact_duplicates_removed"] == 1
    groups = []
    hashes = []
    for name in ("train", "validation", "test"):
        docs = [json.loads(line) for line in (a / f"{name}.jsonl").read_text().splitlines()]
        assert docs
        groups.append({d["group"] for d in docs})
        hashes.append({d["sha256"] for d in docs})
        assert "\r" not in (a / f"{name}.txt").read_text()
        assert (a / f"{name}.txt").read_bytes() == (b / f"{name}.txt").read_bytes()
    for i in range(3):
        for j in range(i + 1, 3):
            assert groups[i].isdisjoint(groups[j])
            assert hashes[i].isdisjoint(hashes[j])
    assert (a / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        prepare([path], a)


def test_permission_and_source_group_requirements(tmp_path):
    path = tmp_path / "no-auth.jsonl"
    path.write_text(json.dumps({"id":"1","source":"x","license":"CC0","text":"hello"}), encoding="utf-8")
    with pytest.raises(ValueError, match="authorized=true"):
        prepare([path], tmp_path / "out")
    path = tmp_path / "too-few.jsonl"
    make_dataset(path, count=2)
    with pytest.raises(ValueError, match="three distinct"):
        prepare([path], tmp_path / "out2")
