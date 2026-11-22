"""Generate *untrained* deterministic fixtures for browser WASM CI only."""
from dataclasses import asdict
from pathlib import Path
import sys

import torch

from aetherion.mobile_export import export_mobile_checkpoint
from aetherion.mobile_quantization import quantize_mobile_int4
from aetherion.mobile_cached import export_mobile_cached
from aetherion.model import Aetherion, ModelConfig


def main() -> None:
    dest = Path(sys.argv[1])
    dest.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(18)
    config = ModelConfig(block_size=16, d_model=32, n_heads=4, n_layers=2)
    model = Aetherion(config).eval()
    checkpoint = dest / "synthetic.pt"
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, checkpoint)
    fp32_dir = dest / "fp32"
    int4_dir = dest / "int4"
    export_mobile_checkpoint(checkpoint, fp32_dir, window=8)
    report = quantize_mobile_int4(
        fp32_dir / "model.onnx", fp32_dir / "manifest.json", int4_dir
    )
    export_mobile_cached(checkpoint, dest / "cached", window=8)
    print("M18 SYNTHETIC FIXTURE ONLY:", report["quantization"])
    print("M19 SYNTHETIC KV-CACHE FIXTURE ONLY: 8-byte context")


if __name__ == "__main__":
    main()
