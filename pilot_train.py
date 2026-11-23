"""Guarded Aetherion-125M pilot launcher; no data or weights are bundled."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from aetherion.corpus_gate import audit_corpus, require_review
from aetherion.contamination import require_no_cross_split_findings
from aetherion.hardware_gate import validate_pilot_benchmark
from aetherion.live_device_gate import check_live_accelerator
from aetherion.pilot_budget import check_pilot_budget
from aetherion.presets import get_preset
from aetherion.tokenizer_runtime import load_verified_tokenizer


def pilot_command(args):
    directory = Path(args.data_dir)
    report = audit_corpus(directory, args.allow_license)
    require_review(report, args.review)
    # M27: a self-attested review cannot override known lexical leakage.
    # The bounded scan must complete before tokenizer/model/hardware execution.
    require_no_cross_split_findings(directory, args.allow_license)
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
    if getattr(args, "retain_step_checkpoints", False):
        command += ["--retain-step-checkpoints"]
    save_every = getattr(args, "save_every", None)
    if save_every is not None:
        if type(save_every) is not int or save_every < 0:
            raise ValueError("Pilot save-every must be a nonnegative integer")
        command += ["--save-every", str(save_every)]
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
    p.add_argument("--retain-step-checkpoints", action="store_true",
                   help="Preserve separate numbered snapshots; requires available disk space")
    p.add_argument("--save-every", type=int, default=None,
                   help="Optional snapshot interval in optimizer steps; default is trainer's 50")
    p.add_argument("--benchmark-report", help="Local exact-configuration 125M accelerator benchmark JSON")
    p.add_argument("--max-peak-reserved-gib", type=float,
                   help="Explicit upper bound on measured accelerator peak reserved GiB")
    p.add_argument("--max-total-tokens", type=int,
                   help="Required for execution: upper bound on sampled target token positions")
    p.add_argument("--max-checkpoint-writes", type=int,
                   help="Required for execution: upper bound on planned checkpoint write operations")
    p.add_argument("--execute", action="store_true", help="Explicitly start training; default only prints plan")
    args = p.parse_args()
    hardware_report = None
    work_budget = None
    if args.execute:
        if args.max_total_tokens is None or args.max_checkpoint_writes is None:
            p.error("--execute requires --max-total-tokens and --max-checkpoint-writes")
        try:
            work_budget = check_pilot_budget(
                steps=args.steps, context=args.context, batch_size=args.batch_size,
                accumulation=args.accumulation, schedule_steps=args.schedule_steps,
                save_every=args.save_every,
                retain_step_checkpoints=args.retain_step_checkpoints,
                max_total_tokens=args.max_total_tokens,
                max_checkpoint_writes=args.max_checkpoint_writes,
                resume=bool(args.resume),
            )
        except ValueError as exc:
            p.error(str(exc))
        if not args.benchmark_report or args.max_peak_reserved_gib is None:
            p.error("--execute requires --benchmark-report and --max-peak-reserved-gib")
        try:
            hardware_report = validate_pilot_benchmark(
                args.benchmark_report, context=args.context,
                batch_size=args.batch_size, accumulation=args.accumulation,
                precision=args.precision, device=args.device,
                max_peak_reserved_gib=args.max_peak_reserved_gib,
            )
        except (ValueError, OSError) as exc:
            p.error(str(exc))
    cmd = pilot_command(args)
    if args.execute:
        # Live identity and free-memory snapshot must be checked immediately
        # before launching training, after corpus and benchmark preflights.
        try:
            live_report = check_live_accelerator(hardware_report, device=args.device)
        except (ValueError, OSError, RuntimeError) as exc:
            p.error(str(exc))
        print(json.dumps({
            "hardware_preflight": hardware_report,
            "live_accelerator_preflight": live_report,
            "work_budget_preflight": work_budget,
        }, indent=2), flush=True)
        subprocess.run(cmd, check=True)
    else:
        print(json.dumps({"executed": False, "command": cmd}, indent=2))


if __name__ == "__main__":
    main()
