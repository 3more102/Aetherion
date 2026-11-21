"""Train Aetherion Micro with legacy bytes or prepared BPE JSONL splits."""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from aetherion import Aetherion, ModelConfig
from aetherion.encoding import encode_bytes, encode_documents, load_bpe, sha256_file


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Aetherion Micro training")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--val-data", help="Independent validation file; required for BPE")
    parser.add_argument("--tokenizer", help="Trained JSON ByteLevel BPE tokenizer")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="checkpoints/demo.pt")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        parser.error("--steps, --batch-size, and --eval-every must be >= 1")
    if args.block_size < 2 or args.lr <= 0:
        parser.error("--block-size must be >= 2 and --lr must be > 0")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    data_hash = sha256_file(args.data)
    val_hash = sha256_file(args.val_data) if args.val_data else None
    if args.val_data and data_hash == val_hash:
        raise ValueError("Training and validation files must have different contents")

    if args.tokenizer:
        if not args.val_data:
            parser.error("--val-data is required when using --tokenizer")
        if Path(args.data).name != "train.jsonl" or Path(args.val_data).name != "validation.jsonl":
            parser.error("BPE requires prepared train.jsonl and validation.jsonl files")
        tokenizer = load_bpe(args.tokenizer)
        train = encode_documents(args.data, tokenizer)
        val = encode_documents(args.val_data, tokenizer)
        vocab_size = tokenizer.get_vocab_size()
        encoding = "bpe"
        tokenizer_hash = sha256_file(args.tokenizer)
    else:
        if args.data.endswith(".jsonl") or (args.val_data and args.val_data.endswith(".jsonl")):
            parser.error("JSONL datasets require --tokenizer; legacy byte mode expects UTF-8 text")
        data = encode_bytes(args.data)
        if args.val_data:
            train, val = data, encode_bytes(args.val_data)
        else:
            cutoff = int(0.9 * len(data))
            train, val = data[:cutoff], data[cutoff:]
        vocab_size = 256
        encoding = "byte"
        tokenizer_hash = None

    if min(len(train), len(val)) < args.block_size + 2:
        raise ValueError("Each split needs at least block_size+2 token IDs; use more data or a smaller context")

    cfg = ModelConfig(vocab_size=vocab_size, block_size=args.block_size,
                      d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
    model = Aetherion(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    print(f"device={device} encoding={encoding} vocab={vocab_size} parameters={sum(p.numel() for p in model.parameters()):,}")

    def get_batch(split: torch.Tensor):
        starts = torch.randint(0, len(split) - cfg.block_size, (args.batch_size,))
        x = torch.stack([split[i:i + cfg.block_size] for i in starts])
        y = torch.stack([split[i + 1:i + cfg.block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch(train)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.inference_mode():
                vx, vy = get_batch(val)
                _, val_loss = model(vx, vy)
            print(f"step={step:4d} train_loss={loss.item():.3f} val_loss={val_loss.item():.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": asdict(cfg),
        "state_dict": model.state_dict(),
        "step": args.steps,
        "seed": args.seed,
        "encoding": encoding,
        "tokenizer_sha256": tokenizer_hash,
        "data_sha256": data_hash,
        "val_data_sha256": val_hash,
        "train_tokens": len(train),
        "val_tokens": len(val),
    }, output)
    print(f"saved_checkpoint={output}")


if __name__ == "__main__":
    main()
