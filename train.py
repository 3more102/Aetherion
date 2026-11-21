"""Train Aetherion Micro from UTF-8 bytes or a verified BPE dataset."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import torch
from aetherion import Aetherion, ModelConfig


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_data(train_path: str, val_path: str | None):
    """Read separate files when given, retaining the legacy raw-byte demo."""
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
             "explicit_validation_file": val_path is not None,
             "tokenizer_type": "bytes"})


def load_bpe_data(train_path: str, val_path: str | None, tokenizer_dir: str):
    """Use only checksum-verified prepared splits with the matching tokenizer."""
    from aetherion.tokenizer import AetherionTokenizer
    if val_path is None:
        raise ValueError("BPE training requires --val-data (a separate prepared validation.txt)")
    train_file = Path(train_path).resolve()
    val_file = Path(val_path).resolve()
    if (train_file.parent != val_file.parent or
            train_file.name != "train.txt" or val_file.name != "validation.txt"):
        raise ValueError("BPE training requires train.txt and validation.txt in one prepared data directory")
    dataset_manifest = train_file.parent / "manifest.json"
    expected_manifest_bytes = dataset_manifest.read_bytes()
    manifest = json.loads(expected_manifest_bytes)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported prepared-data manifest schema")
    for name, path in (("train.txt", train_file), ("validation.txt", val_file)):
        wanted = manifest.get("outputs_sha256", {}).get(name)
        if wanted is None or sha256(path.read_bytes()).hexdigest() != wanted:
            raise ValueError(f"Prepared dataset checksum mismatch: {name}")

    tokenizer = AetherionTokenizer.load(tokenizer_dir)
    manifest_hash = sha256(expected_manifest_bytes).hexdigest()
    if (tokenizer.metadata["source_manifest_sha256"] != manifest_hash or
            tokenizer.metadata["source_train_sha256"] !=
            manifest.get("outputs_sha256", {}).get("train.jsonl")):
        raise ValueError("Tokenizer was trained on a different prepared dataset")
    raw_train, raw_val = train_file.read_bytes(), val_file.read_bytes()
    train_ids = tokenizer.encode(raw_train.decode("utf-8"))
    val_ids = tokenizer.encode(raw_val.decode("utf-8"))
    hashes = {
        "train_sha256": sha256(raw_train).hexdigest(),
        "validation_sha256": sha256(raw_val).hexdigest(),
        "explicit_validation_file": True,
        "tokenizer_type": "bpe",
        "tokenizer_sha256": tokenizer.sha256,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "source_manifest_sha256": manifest_hash,
    }
    return (torch.tensor(train_ids, dtype=torch.long),
            torch.tensor(val_ids, dtype=torch.long), hashes, tokenizer.vocab_size)


def main():
    parser = argparse.ArgumentParser(description="Aetherion Micro training")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--val-data", default=None, help="Separate validation text from prepare_data.py")
    parser.add_argument("--tokenizer-dir", default=None, help="Verified BPE tokenizer artifact directory")
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

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if args.tokenizer_dir:
        train, val, hashes, vocab_size = load_bpe_data(args.data, args.val_data, args.tokenizer_dir)
    else:
        train, val, hashes = load_data(args.data, args.val_data)
        vocab_size = 256
    if min(len(train), len(val)) < args.block_size + 2:
        raise ValueError("Train and validation must each have at least block_size+2 tokens")

    cfg = ModelConfig(vocab_size=vocab_size, block_size=args.block_size,
                      d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
    model = Aetherion(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} "
          f"tokenizer={hashes['tokenizer_type']} vocab_size={vocab_size}")

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
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict(),
                "step": args.steps, "seed": args.seed, "data": hashes}, output)
    print(f"saved_checkpoint={output}")


if __name__ == "__main__":
    main()
