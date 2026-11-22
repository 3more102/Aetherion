"""Single-node DDP training prototype (CPU/Gloo or CUDA/NCCL).

Run with torchrun; all ranks read identical data, sample independently and
synchronize gradients. No optimizer sharding or multi-node support claimed.
"""
import argparse
import json
from hashlib import sha256
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import timedelta
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from aetherion import Aetherion, ModelConfig
from aetherion.presets import PRESETS, get_preset, parameter_breakdown
from aetherion.training import learning_rate, precision_context
from aetherion.checkpoint import save_training_checkpoint, load_training_checkpoint
from train import load_data, load_bpe_data


def run(args):
    if any(k not in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise ValueError("Launch via torchrun --standalone --nproc-per-node=N")
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if rank != local or world != int(os.environ.get("LOCAL_WORLD_SIZE", world)):
        raise ValueError("Only single-node training is supported")
    if args.device == "cuda":
        if not torch.cuda.is_available() or world > torch.cuda.device_count():
            raise ValueError("Require one visible CUDA GPU per rank")
        device = torch.device("cuda", local)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device, backend = torch.device("cpu"), "gloo"
        torch.set_num_threads(args.cpu_threads)
    precision_context(device, args.precision)
    if not 1 <= args.steps <= args.schedule_steps:
        raise ValueError("Require fixed schedule horizon >= total steps")
    if min(args.batch_size, args.accumulation, args.cpu_threads) < 1 or not math.isfinite(args.lr):
        raise ValueError("Invalid training arguments")
    learning_rate(1, peak=args.lr, warmup=args.warmup_steps, horizon=args.schedule_steps)
    if args.save_every < 1:
        raise ValueError("Save interval must be positive")
    if Path(args.output).exists() and (not args.resume or Path(args.output).resolve() != Path(args.resume).resolve()):
        raise ValueError("Refusing to overwrite unrelated checkpoint")
    if args.tokenizer:
        train, val, fingerprints, vocab = load_bpe_data(args.data, args.val_data, args.tokenizer)
    else:
        train, val, fingerprints = load_data(args.data, args.val_data)
        vocab = 256
    cfg = get_preset(args.preset)
    if args.preset != "micro" and vocab != cfg.vocab_size:
        raise ValueError("Non-micro presets require the matching 32000-entry BPE tokenizer")
    if args.context is not None:
        if not 2 <= args.context <= cfg.block_size:
            raise ValueError("Invalid context")
        cfg = replace(cfg, block_size=args.context)
    cfg = replace(cfg, vocab_size=vocab)
    if min(train.numel(), val.numel()) <= cfg.block_size:
        raise ValueError("Need more tokens than context in both splits")
    if cfg.dropout != 0:
        raise ValueError("This resume design requires dropout=0")
    # No allocation of the larger designs without an explicit opt-in.
    if parameter_breakdown(cfg)["total"] >= 300_000_000 and not args.allow_large:
        raise ValueError("Large preset requires --allow-large after memory assessment")
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=5))
    try:
        # Compare fixed-size SHA-256 digests with tensor collectives. Unlike
        # all_gather_object this needs neither NumPy nor unsafe pickle decoding.
        local_fingerprint = sha256(json.dumps(
            fingerprints, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).digest()
        local_digest = torch.tensor(list(local_fingerprint), dtype=torch.uint8, device=device)
        gathered = [torch.empty_like(local_digest) for _ in range(world)]
        dist.all_gather(gathered, local_digest)
        if any(not torch.equal(peer, local_digest) for peer in gathered):
            raise ValueError("Dataset fingerprints differ across ranks")
        torch.manual_seed(args.seed)
        raw = Aetherion(cfg).to(device)
        raw.activation_checkpointing = args.activation_checkpointing
        optimizer = torch.optim.AdamW(raw.parameters(), lr=args.lr, weight_decay=0.1)
        metadata = dict(distributed=True, world_size=world, backend=backend,
                        batch_size=args.batch_size, accumulation=args.accumulation,
                        learning_rate=args.lr, schedule_steps=args.schedule_steps,
                        warmup_steps=args.warmup_steps, precision=args.precision,
                        activation_checkpointing=args.activation_checkpointing,
                        weight_decay=0.1, sampling="rank-step-seed-v1")
        rng = torch.Generator(device="cpu")
        start = 0
        if args.resume:
            start = load_training_checkpoint(args.resume, model=raw, optimizer=optimizer,
                                             train_rng=rng, config=asdict(cfg), seed=args.seed,
                                             data=fingerprints, hparams=metadata)
        if start >= args.steps:
            raise ValueError("Resume step must be less than total steps")
        model = DDP(raw, device_ids=[local] if backend == "nccl" else None)
        tokens_per_update = world * args.batch_size * args.accumulation * cfg.block_size
        if rank == 0:
            print(f"world_size={world} backend={backend} parameters={sum(p.numel() for p in raw.parameters())} tokens_per_update={tokens_per_update}", flush=True)

        def batch(split, generator):
            starts = torch.randint(0, len(split)-cfg.block_size, (args.batch_size,), generator=generator)
            x = torch.stack([split[i:i+cfg.block_size] for i in starts])
            y = torch.stack([split[i+1:i+cfg.block_size+1] for i in starts])
            return x.to(device), y.to(device)

        for step in range(start + 1, args.steps + 1):
            # Dropout is zero. Rank/step-derived CPU seeds make sampling
            # independent of rank-0 checkpoint RNG and prior validation.
            rng.manual_seed((args.seed + step * world + rank) % (2**63))
            model.train()
            lr = learning_rate(step, peak=args.lr, warmup=args.warmup_steps,
                               horizon=args.schedule_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            local_loss = torch.zeros((), device=device)
            for micro in range(args.accumulation):
                sync = model.no_sync() if micro + 1 < args.accumulation else nullcontext()
                with sync:
                    x, y = batch(train, rng)
                    with precision_context(device, args.precision):
                        _, loss = model(x, y)
                    local_loss += loss.detach().float()
                    (loss / args.accumulation).backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            if rank == 0:
                print(f"step={step} global_train_loss={local_loss.item()/(world*args.accumulation):.6f} lr={lr:.6g}", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                if rank == 0:
                    raw.eval()
                    v_rng = torch.Generator().manual_seed(args.seed + 10_000_000 + step)
                    with torch.inference_mode(), precision_context(device, args.precision):
                        x, y = batch(val, v_rng)
                        _, loss = raw(x, y)
                    print(f"sampled_validation_loss={loss.item():.6f}", flush=True)
                    save_training_checkpoint(args.output, model=raw, optimizer=optimizer,
                                             train_rng=rng, config=asdict(cfg), step=step,
                                             seed=args.seed, data=fingerprints, hparams=metadata)
                dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--preset", choices=sorted(PRESETS), default="micro")
    p.add_argument("--data", default="data/demo.txt")
    p.add_argument("--val-data")
    p.add_argument("--tokenizer")
    p.add_argument("--context", type=int)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--schedule-steps", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--allow-large", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
