"""M21: PWA ZIP packaging guards and no-model privacy rule."""
import json
from hashlib import sha256
from pathlib import Path
from struct import pack
from zipfile import ZipFile

import pytest

from mobile.package_pwa import STATIC, build_pwa_zip


def fake_png(size: int) -> bytes:
    # A valid PNG header/IHDR; file content is not rendered by the packager.
    return b"\x89PNG\r\n\x1a\n" + pack(">I", 13) + b"IHDR" + pack(">II", size, size)


@pytest.fixture
def staged(tmp_path):
    root = tmp_path / "mobile"
    (root / "vendor").mkdir(parents=True)
    (root / "icons").mkdir()
    for name in STATIC:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith("icon-192.png"):
            path.write_bytes(fake_png(192))
        elif name.endswith("icon-512.png"):
            path.write_bytes(fake_png(512))
        elif name == "manifest.webmanifest":
            path.write_text(json.dumps({"start_url": "./", "scope": "./"}))
        elif name != "precache-assets.json":
            path.write_text("M21 synthetic fixture")
    for name in ("ort.wasm.min.js", "ort-wasm-simd-threaded.wasm"):
        (root / "vendor" / name).write_bytes(b"synthetic-runtime")
    expected = [p for p in STATIC if p != "precache-assets.json"] + [
        "precache-assets.json", "vendor/ort.wasm.min.js", "vendor/ort-wasm-simd-threaded.wasm"
    ]
    (root / "precache-assets.json").write_text(json.dumps({
        "schema_version": 1, "assets": ["./"] + ["./" + name for name in expected]
    }))
    return root


def test_deterministic_zip_manifest_and_model_exclusion(staged, tmp_path):
    private_dir = staged / "artifacts"
    private_dir.mkdir()
    (private_dir / "secret-model.onnx").write_bytes(b"never ship this")
    (staged / ".env").write_text("SECRET=DO_NOT_SHIP")
    one, two = tmp_path / "one.zip", tmp_path / "two.zip"
    first = build_pwa_zip(staged, one)
    second = build_pwa_zip(staged, two)
    assert first["model_weights_included"] is False
    assert first["asset_count"] == len(STATIC) + 2
    assert one.read_bytes() == two.read_bytes()
    with ZipFile(one) as archive:
        names = set(archive.namelist())
        assert "PACKAGE_INFO.json" in names
        assert "vendor/ort-wasm-simd-threaded.wasm" in names
        assert "artifacts/secret-model.onnx" not in names
        assert ".env" not in names
        info = json.loads(archive.read("PACKAGE_INFO.json"))
        assert info["models_included"] is False
        for entry in info["assets"]:
            assert sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]


def test_reject_missing_pwa_runtime(staged, tmp_path):
    (staged / "vendor" / "ort.wasm.min.js").unlink()
    with pytest.raises(ValueError, match="runtime"):
        build_pwa_zip(staged, tmp_path / "out.zip")


def test_reject_incomplete_pwa_precache(staged, tmp_path):
    manifest_path = staged / "precache-assets.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["assets"].remove("./vendor/ort-wasm-simd-threaded.wasm")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing"):
        build_pwa_zip(staged, tmp_path / "out.zip")


def test_reject_invalid_icon_dimensions(staged, tmp_path):
    (staged / "icons" / "icon-512.png").write_bytes(fake_png(192))
    with pytest.raises(ValueError, match="512px"):
        build_pwa_zip(staged, tmp_path / "out.zip")
