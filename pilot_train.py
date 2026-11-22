"""Guarded Aetherion-125M pilot launcher; no data or weights are bundled."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from aetherion.corpus_gate import audit_corpus, require_review
from aetherion.presets import get_preset
from aetherion.tokenizer_runtime import load_verified_tokenizer


def pilot_command(args):
    directory = Path(args.data_dir)
    report = audit_corpus(directory, args.allow_license)
    require_review(report, args.review)
    tokenizer, meta = load_verified_tokenizer(args.tokenizer)
    cfg = get_preset("125m")
    if tokenizer.get_vocab_size() != cfg.vocab_size:
        raise ValueError("125M pilot needs exactly 32000 actual vocabulary entries")
    if meta.get("manifest_sha256") != report["manifest_sha256"]:
        raise ValueError("Tokenizer and reviewed corpus manifest differ")
    if not 2 <= args.context <= cfg.block_size:
        raise ValueError("Pilot context must be in [2, 2048]")
    if args.steps < 1 or args.schedule_steps < args.steps or not 0 <= args.warmup_steps < args.schedule_steps:
        raise ValueError("Invalid pilot schedule")
    if args.batch_size < 1 or args.accumulation < 1:
        raise ValueError("Invalid pilot batch")
    command = [sys.executable, str(Path(__file__).with_name("train.py")),
               "--data", str(directory / "train.txt"), "--val-data", str(directory / "validation.txt"),
               "--tokenizer", args.tokenizer, "--d-model", str(cfg.d_model), "--layers", str(cfg.n_layers),
               "--heads", str(cfg.n_heads), "--block-size", str(args.context),
               "--steps", str(args.steps), "--schedule-steps", str(args.schedule_steps),
               "--warmup-steps", str(args.warmup_steps), "--batch-size", str(args.batch_size),
               "--accumulation", str(args.accumulation), "--precision", args.precision,
               "--device", args.device, "--output", args.output]
    if args.resume:
        command += ["--resume", args.resume]
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--allow-license", action="append", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--schedule-steps", type=int, required=True)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--execute", action="store_true", help="Explicitly start training; default only prints plan")
    args = p.parse_args()
    cmd = pilot_command(args)
    if args.execute:
        subprocess.run(cmd, check=True)
    else:
        print(json.dumps({"executed": False, "command": cmd}, indent=2))


if __name__ == "__main__":
    main()
