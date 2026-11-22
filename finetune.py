"""Experimental supervised fine-tuning from a trusted BPE base checkpoint."""
import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import torch
from torch.nn import functional as F

from aetherion import Aetherion, ModelConfig
from aetherion.checkpoint import save_training_checkpoint, load_training_checkpoint
from aetherion.instruct import TEMPLATE, IGNORE, load_splits, batch_examples
from aetherion.tokenizer_runtime import load_verified_tokenizer
from train import resolve_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if min(args.steps, args.batch_size) < 1 or not 0 < args.lr < float("inf"):
        p.error("Invalid training settings")
    if Path(args.base).resolve() == Path(args.output).resolve():
        p.error("Do not overwrite the base checkpoint")
    if Path(args.output).exists() and (not args.resume or Path(args.resume).resolve() != Path(args.output).resolve()):
        p.error("Refusing to overwrite unrelated checkpoint")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    base_bytes = Path(args.base).read_bytes()
    import io
    base = torch.load(io.BytesIO(base_bytes), map_location="cpu", weights_only=True)
    base_info = base.get("data", {})
    tokenizer, meta = load_verified_tokenizer(args.tokenizer)
    cfg = ModelConfig(**base["config"])
    if (base_info.get("tokenizer_type") != "bpe" or base_info.get("stage") == "sft"
            or base_info.get("tokenizer_sha256") != meta["tokenizer_sha256"]
            or cfg.vocab_size != tokenizer.get_vocab_size()):
        p.error("Require matching original BPE base checkpoint")
    train, val, hashes = load_splits(args.train, args.validation, tokenizer, cfg.block_size)
    hashes.update(stage="sft", template=TEMPLATE, tokenizer_type="bpe",
                  tokenizer_sha256=meta["tokenizer_sha256"], base_sha256=sha256(base_bytes).hexdigest())
    del base_bytes
    model = Aetherion(cfg)
    model.load_state_dict(base["state_dict"])
    del base
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    hparams = {"batch_size": args.batch_size, "learning_rate": args.lr}
    rng = torch.Generator().manual_seed(args.seed + 1)
    start = 0
    if args.resume:
        start = load_training_checkpoint(args.resume, model=model, optimizer=optimizer,
                                         train_rng=rng, config=asdict(cfg), seed=args.seed,
                                         data=hashes, hparams=hparams)
    if start >= args.steps:
        p.error("Total steps must exceed checkpoint step")
    pad = tokenizer.token_to_id("<|eos|>")
    for step in range(start + 1, args.steps + 1):
        model.train()
        idx = torch.randint(len(train), (args.batch_size,), generator=rng).tolist()
        x, y = batch_examples([train[i] for i in idx], pad, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        print(f"step={step} response_loss={loss.item():.6f}")
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(val), args.batch_size):
            x, y = batch_examples(val[i:i + args.batch_size], pad, device)
            logits, _ = model(x)
            total += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
            count += (y != IGNORE).sum().item()
    print(f"validation_response_nll={total/count:.6f} targets={count}")
    save_training_checkpoint(args.output, model=model, optimizer=optimizer, train_rng=rng,
                             config=asdict(cfg), step=args.steps, seed=args.seed,
                             data=hashes, hparams=hparams)


if __name__ == "__main__":
    main()
