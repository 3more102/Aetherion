"""Build a static HTTPS-deployable Aetherion PWA ZIP (no model weights).

Stages only the app shell, pinned self-hosted ONNX Runtime Web assets,
generated PNG icons and integrity inventory. No credentials, node_modules,
model checkpoints, or local prompt data are ever packaged.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


STATIC = (
    "index.html",
    "app.js",
    "model-vault.js",
    "device-check.js",
    "sw.js",
    "manifest.webmanifest",
    "icon.svg",
    "icons/icon-192.png",
    "icons/icon-512.png",
    "precache-assets.json",
)
EXTENSIONS = {".js", ".mjs", ".wasm"}
EPOCH = (2026, 11, 22, 0, 0, 0)


def _png_dimensions(file: Path) -> tuple[int, int]:
    with file.open("rb") as stream:
        head = stream.read(24)
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG icon: {file.name}")
    return struct.unpack(">II", head[16:24])


def build_pwa_zip(source: str | Path, destination: str | Path) -> dict:
    root, output = Path(source).resolve(), Path(destination).resolve()
    if not root.is_dir():
        raise ValueError(f"Mobile directory missing: {root}")
    files = list(STATIC)
    vendor = root / "vendor"
    if not vendor.is_dir():
        raise ValueError("Staged vendor/ missing; run npm install && npm run stage-runtime in mobile/")
    for path in sorted(vendor.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name.startswith("ort") and path.suffix in EXTENSIONS:
            files.append("vendor/" + path.name)
    if "vendor/ort.wasm.min.js" not in files or not any(p.endswith(".wasm") for p in files):
        raise ValueError("Pinned WebAssembly runtime files are missing")

    for relative in files:
        path = root / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing, untrusted, or unsafe PWA asset: {relative}")
    for size in (192, 512):
        if _png_dimensions(root / f"icons/icon-{size}.png") != (size, size):
            raise ValueError(f"Invalid {size}px PWA icon")

    manifest = json.loads((root / "manifest.webmanifest").read_text(encoding="utf-8"))
    if manifest.get("start_url") != "./" or manifest.get("scope") != "./":
        raise ValueError("PWA must be hosted in its own scoped directory")
    precache = json.loads((root / "precache-assets.json").read_text(encoding="utf-8"))
    if precache.get("schema_version") != 1:
        raise ValueError("Unsupported offline precache manifest")
    asset_paths = precache.get("assets")
    if not isinstance(asset_paths, list) or not asset_paths:
        raise ValueError("Offline precache manifest missing assets")
    required_paths = set(files)
    actual = {item[2:] if item != "./" else "index.html"
              for item in asset_paths if isinstance(item, str) and item.startswith("./")}
    if not required_paths.issubset(actual):
        raise ValueError("Offline manifest is missing one or more packaged assets")
    if actual - required_paths:
        raise ValueError("Offline manifest includes an unbundled or unexpected asset")

    inventory = {
        "schema_version": 1,
        "distribution": "static-pwa-zip-not-an-apk-or-ipa",
        "models_included": False,
        "hosting": "HTTPS required (localhost allowed for development)",
        "assets": [
            {
                "path": name,
                "bytes": (root / name).stat().st_size,
                "sha256": sha256((root / name).read_bytes()).hexdigest(),
            }
            for name in sorted(set(files))
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output == root or output in [root / name for name in files]:
        raise ValueError("Output would overwrite a PWA asset")
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(set(files)):
            info = ZipInfo(name, date_time=EPOCH)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (root / name).read_bytes(), compress_type=ZIP_DEFLATED,
                             compresslevel=6)
        info = ZipInfo("PACKAGE_INFO.json", date_time=EPOCH)
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    return {
        "output": str(output),
        "bundle_bytes": output.stat().st_size,
        "asset_count": len(inventory["assets"]),
        "onnx_runtime_web_staged": True,
        "model_weights_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobile-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "dist" /
                                                "aetherion-mobile-pwa.zip"))
    args = parser.parse_args()
    print(json.dumps(build_pwa_zip(args.mobile_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
