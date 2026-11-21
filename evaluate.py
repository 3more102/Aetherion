"""Reproducible validation-only loss for Aetherion Micro byte checkpoints.

Uses the same final-10%-of-bytes demo split as train.py. For published
benchmarks, replace this with document-separated held-out evaluation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from aetherion import Aetherion, ModelConfig
from train import resolve_device


def evaluate_checkpoint(checkpoint_path: str | Path, data_path: str | Path, device: str = 'cpu', max_batches: int = 0) -> dict:
    """Evaluate every next byte in disjoint val chunks, or the first N chunks.

    Loss is per UTF-8 byte token. Chunk boundaries reset attention context.
    """
    if max_batches < 0:
        raise ValueError('max_batches must be >= 0')
    checkpoint_path, data_path = Path(checkpoint_path), Path(data_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    cfg = ModelConfig(**checkpoint['config'])
    model = Aetherion(cfg)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()

    raw = data_path.read_bytes()
    cutoff = int(0.9 * len(raw))
    val = raw[cutoff:]
    if len(val) < 2:
        raise ValueError('Validation split needs at least two bytes')

    loss_sum = 0.0
    tokens = 0
    batches = 0
    with torch.inference_mode():
        for offset in range(0, len(val) - 1, cfg.block_size):
            if max_batches and batches >= max_batches:
                break
            chunk = val[offset: offset + cfg.block_size + 1]
            x = torch.tensor([list(chunk[:-1])], dtype=torch.long, device=device)
            targets = torch.tensor([list(chunk[1:])], dtype=torch.long, device=device)
            logits, _ = model(x)
            loss_sum += F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), targets.reshape(-1), reduction='sum'
            ).item()
            tokens += targets.numel()
            batches += 1
    nll = loss_sum / tokens
    return {
        'metric': 'validation_next_byte',
        'nll_nats_per_byte': nll,
        'perplexity_per_byte': math.exp(nll) if nll < 700 else None,
        'evaluated_byte_tokens': tokens,
        'evaluated_chunks': batches,
        'context_length': cfg.block_size,
        'dataset_sha256': hashlib.sha256(raw).hexdigest(),
        'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'split_policy': 'last_10_percent_by_byte_demo_only',
        'boundary_policy': 'attention_context_resets_each_chunk',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate Aetherion Micro byte-level validation NLL')
    parser.add_argument('--checkpoint', default='checkpoints/demo.pt')
    parser.add_argument('--data', default='data/demo.txt')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-batches', type=int, default=0, help='0 evaluates the entire validation split')
    parser.add_argument('--output', default=None, help='Optional JSON metrics output')
    args = parser.parse_args()
    if args.max_batches < 0:
        parser.error('--max-batches must be >= 0')
    results = evaluate_checkpoint(args.checkpoint, args.data, resolve_device(args.device), args.max_batches)
    result_json = json.dumps(results, indent=2, allow_nan=False)
    print(result_json)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result_json + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
