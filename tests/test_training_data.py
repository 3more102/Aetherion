"""Test explicit train/validation isolation and metadata in training loader."""
from hashlib import sha256
import torch
from train import load_data


def test_explicit_validation_file(tmp_path):
    train_bytes = b"training text is separate" * 3
    val_bytes = b"validation text is different" * 3
    path_a, path_b = tmp_path / "train.txt", tmp_path / "validation.txt"
    path_a.write_bytes(train_bytes)
    path_b.write_bytes(val_bytes)
    train, val, hashes = load_data(str(path_a), str(path_b))
    assert train.dtype == torch.long and val.dtype == torch.long
    assert bytes(train.tolist()) == train_bytes
    assert bytes(val.tolist()) == val_bytes
    assert hashes["explicit_validation_file"] is True
    assert hashes["train_sha256"] == sha256(train_bytes).hexdigest()
    assert hashes["validation_sha256"] == sha256(val_bytes).hexdigest()
