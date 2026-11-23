"""Train Aetherion Micro on byte sequences with verifiable resume support."""
import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from secrets import token_hex

import torch
from aetherion import Aetherion, ModelConfig
from aetherion.checkpoint import file_sha256, load_training_checkpoint, save_training_checkpoint
from aetherion.training import learning_rate, optimizer_step, precision_context


from aetherion.devices import resolve_device


def load_data(train_path: str, val_path: str | None):
    """Read distinct prepared files when provided; preserve demo fallback."""
    train_bytes = Path(train_path).read_bytes()
    if val_path is None:
        cutoff = int(0.9 * len(train_bytes))
        print("WARNING: using a raw 90/10 byte split for demo only; use --val-data for real experiments")
        train_bytes, val_bytes = train_bytes[:cutoff], train_bytes[cutoff:]
    else:
        val_bytes = Path(val_path).read_bytes()
    train_sha = sha256(train_bytes).hexdigest()
    val_sha = sha256(val_bytes).hexdigest()
    return (torch.tensor(list(train_bytes), dtype=torch.long),
            torch.tensor(list(val_bytes), dtype=torch.long),
            {"train_sha256": train_sha, "validation_sha256": val_sha,
             "explicit_validation_file": val_path is not None})



def load_bpe_data(train_path: str, val_path: str | None, tokenizer_path: str):
    """Load token IDs only from prepared splits matching a verified BPE artifact."""
    import json
    from aetherion.tokenizer_runtime import load_verified_tokenizer

    if val_path is None:
        raise ValueError("BPE training requires --val-data (separate validation.txt)")
    train_file, val_file = Path(train_path).resolve(), Path(val_path).resolve()
    if (train_file.parent != val_file.parent or
            train_file.name != "train.txt" or val_file.name != "validation.txt"):
        raise ValueError("BPE training requires prepared train.txt and validation.txt in one directory")
    manifest_bytes = (train_file.parent / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported prepared-data manifest schema")
    for name, path in (("train.txt", train_file), ("validation.txt", val_file)):
        expected = manifest.get("outputs_sha256", {}).get(name)
        if expected is None or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared dataset checksum mismatch: {name}")
    tokenizer, metadata = load_verified_tokenizer(tokenizer_path)
    if (metadata.get("manifest_sha256") != sha256(manifest_bytes).hexdigest()
            or metadata.get("train_sha256") !=
            manifest.get("outputs_sha256", {}).get("train.jsonl")):
        raise ValueError("Tokenizer was fitted on a different prepared training dataset")
    train_raw, val_raw = train_file.read_bytes(), val_file.read_bytes()
    train_ids = tokenizer.encode(train_raw.decode("utf-8"), add_special_tokens=False).ids
    val_ids = tokenizer.encode(val_raw.decode("utf-8"), add_special_tokens=False).ids
    hashes = {
        "train_sha256": sha256(train_raw).hexdigest(),
        "validation_sha256": sha256(val_raw).hexdigest(),
        "explicit_validation_file": True,
        "tokenizer_type": "bpe",
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        "prepared_manifest_sha256": metadata["manifest_sha256"],
    }
    return (torch.tensor(train_ids, dtype=torch.long),
            torch.tensor(val_ids, dtype=torch.long), hashes, tokenizer.get_vocab_size())


def main():
    parser = argparse.ArgumentParser(description="Aetherion Micro training")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--val-data", default=None, help="Separate validation text from prepare_data.py")
    parser.add_argument("--tokenizer", default=None, help="Verified BPE tokenizer.json (requires prepared train/validation)")
    parser.add_argument("--steps", type=int, default=200, help="Total training steps, including any resumed steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--accumulation", type=int, default=1, help="Microbatches per optimizer update")
    parser.add_argument("--schedule-steps", type=int, default=0, help="Fixed cosine horizon; zero keeps constant LR")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50, help="Save after every N steps (0 disables periodic saves)")
    parser.add_argument("--retain-step-checkpoints", action="store_true",
                        help="Opt-in: save immutable --output-stem-stepXXXXXXXX.pt files, never overwrite")
    parser.add_argument("--resume", default=None, help="Resume optimizer, weights and RNG state from this checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="checkpoints/demo.pt")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.save_every < 0:
        parser.error("--steps, --batch-size, --eval-every must be >= 1 and --save-every >= 0")
    if args.lr <= 0:
        parser.error("--lr must be > 0")

    if args.accumulation < 1 or (args.schedule_steps and args.steps > args.schedule_steps):
        parser.error("Invalid accumulation or total steps exceed schedule horizon")
    try:
        learning_rate(1, peak=args.lr, warmup=args.warmup_steps,
                      horizon=args.schedule_steps, min_ratio=args.min_lr_ratio)
    except ValueError as exc:
        parser.error(str(exc))

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    precision_context(device, args.precision)
    if args.tokenizer:
        train, val, hashes, vocab_size = load_bpe_data(args.data, args.val_data, args.tokenizer)
    else:
        train, val, hashes = load_data(args.data, args.val_data)
        vocab_size = 256
    if min(len(train), len(val)) < args.block_size + 2:
        raise ValueError("Train and validation data must each contain at least block_size+2 tokens")

    cfg = ModelConfig(vocab_size=vocab_size, block_size=args.block_size, d_model=args.d_model,
                      n_layers=args.layers, n_heads=args.heads)
    model = Aetherion(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    hparams = {"batch_size": args.batch_size, "learning_rate": args.lr, "weight_decay": 0.1}
    # Keep legacy constant-LR FP32 checkpoint metadata byte-for-byte compatible.
    if args.accumulation != 1 or args.schedule_steps or args.precision != "fp32" or torch.device(device).type == "xpu":
        hparams.update(accumulation=args.accumulation, schedule_steps=args.schedule_steps,
                       warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio,
                       precision=args.precision, device=str(torch.device(device)))
    train_rng = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    start_step = 0
    resume_lineage = {}
    # Detect changes to the selected parent artifact while it is being loaded.
    resume_sha = (file_sha256(args.resume)
                  if args.resume is not None and args.retain_step_checkpoints else None)
    if args.resume is not None:
        start_step = load_training_checkpoint(
            args.resume, model=model, optimizer=optimizer, train_rng=train_rng,
            config=asdict(cfg), seed=args.seed, data=hashes, hparams=hparams,
            lineage_out=resume_lineage,
        )
        if start_step >= args.steps:
            parser.error(f"--steps must exceed checkpoint step ({start_step}); it is a total step target")
        if resume_sha is not None and file_sha256(args.resume) != resume_sha:
            raise ValueError("Resume checkpoint changed during load")
        print(f"resumed_checkpoint={args.resume} from_step={start_step}")
    if args.retain_step_checkpoints:
        # A legacy checkpoint is a fresh chain root, not verified ancestry.
        run_id = resume_lineage["run_id"] if resume_lineage else token_hex(16)
        parent_sha = resume_sha if resume_lineage else None
        parent_step = start_step if resume_lineage else None
        if args.resume and not resume_lineage:
            print("WARNING: legacy resume starts a new lineage; preceding snapshots are unlinked")
    else:
        run_id, parent_sha, parent_step = None, None, None
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")

    def get_batch(split: torch.Tensor, generator: torch.Generator):
        starts = torch.randint(0, len(split) - cfg.block_size, (args.batch_size,), generator=generator)
        x = torch.stack([split[i:i + cfg.block_size] for i in starts])
        y = torch.stack([split[i + 1:i + cfg.block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    output = Path(args.output)

    def save(step: int):
        nonlocal parent_sha, parent_step
        target = (output.with_name(f"{output.stem}-step{step:08d}{output.suffix}")
                  if args.retain_step_checkpoints else output)
        lineage = ({"version": 1, "run_id": run_id,
                    "parent_sha256": parent_sha, "parent_step": parent_step}
                   if args.retain_step_checkpoints else None)
        save_training_checkpoint(
            target, model=model, optimizer=optimizer, train_rng=train_rng,
            config=asdict(cfg), step=step, seed=args.seed, data=hashes, hparams=hparams,
            overwrite=not args.retain_step_checkpoints, lineage=lineage,
        )
        if args.retain_step_checkpoints:
            parent_sha, parent_step = file_sha256(target), step
        print(f"saved_checkpoint={target} step={step}")

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        lr = learning_rate(step, peak=args.lr, warmup=args.warmup_steps,
                           horizon=args.schedule_steps, min_ratio=args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = optimizer_step(model, optimizer, lambda: get_batch(train, train_rng),
                              accumulation=args.accumulation, device=device, precision=args.precision)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            # Step-specific validation sampling prevents extra end-of-run evaluations
            # from perturbing training RNG or later validation measurements.
            val_rng = torch.Generator(device="cpu").manual_seed(args.seed + 1_000_003 + step)
            model.eval()
            with torch.inference_mode(), precision_context(device, args.precision):
                vx, vy = get_batch(val, val_rng)
                _, val_loss = model(vx, vy)
            print(f"step={step:4d} train_loss={loss:.3f} val_loss={val_loss.item():.3f} lr={lr:.6g}")
        if args.save_every and step % args.save_every == 0:
            save(step)

    if not args.save_every or args.steps % args.save_every:
        save(args.steps)


if __name__ == "__main__":
    main()
