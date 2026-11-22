"""Single-turn SFT examples with response-only supervision and explicit EOS."""
import json
from hashlib import sha256
from pathlib import Path

import torch

TEMPLATE = "aetherion-single-turn-v1"
IGNORE = -100
RESERVED = ("<|unk|>", "<|bos|>", "<|eos|>")


def prompt_ids(tokenizer, instruction):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction must be nonempty")
    if any(marker in instruction for marker in RESERVED):
        raise ValueError("Reserved token in instruction")
    bos = tokenizer.token_to_id("<|bos|>")
    if bos is None:
        raise ValueError("BOS missing")
    # Encode response separately so the supervision boundary is exact.
    return [bos] + tokenizer.encode("User: " + instruction + "\nAssistant: ",
                                    add_special_tokens=False).ids


def encode_example(tokenizer, instruction, response, context):
    if not isinstance(response, str) or not response.strip() or any(x in response for x in RESERVED):
        raise ValueError("Response must be nonempty and contain no reserved tokens")
    prefix = prompt_ids(tokenizer, instruction)
    eos = tokenizer.token_to_id("<|eos|>")
    if eos is None:
        raise ValueError("EOS missing")
    completion = tokenizer.encode(response, add_special_tokens=False).ids + [eos]
    tokens = prefix + completion
    if len(tokens) - 1 > context:
        raise ValueError("SFT example exceeds context; refusing silent truncation")
    x = torch.tensor(tokens[:-1], dtype=torch.long)
    y = torch.tensor([IGNORE] * (len(prefix) - 1) + completion, dtype=torch.long)
    return x, y


def load_splits(train_path, val_path, tokenizer, context):
    groups_seen, instructions_seen = set(), set()
    all_examples, fingerprints = [], {}
    for split, path in (("train", train_path), ("validation", val_path)):
        raw = Path(path).read_bytes()
        fingerprints[split + "_sha256"] = sha256(raw).hexdigest()
        examples, groups = [], set()
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("authorized") is not True:
                raise ValueError("Instruction records require authorized=true self-attestation")
            for key in ("source", "license", "group", "instruction", "response"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError("Missing nonempty field: " + key)
            instruction = row["instruction"].strip()
            group = row["group"]
            if group in groups_seen or instruction in instructions_seen:
                raise ValueError("Duplicate instruction or cross-split group")
            instructions_seen.add(instruction)
            groups.add(group)
            examples.append(encode_example(tokenizer, instruction, row["response"], context))
        if not examples:
            raise ValueError("Empty " + split + " split")
        groups_seen.update(groups)
        all_examples.append(examples)
    return (*all_examples, fingerprints)


def batch_examples(examples, pad_id, device):
    length = max(x.numel() for x, _ in examples)
    x_batch = torch.full((len(examples), length), pad_id, dtype=torch.long)
    y_batch = torch.full_like(x_batch, IGNORE)
    for i, (x, y) in enumerate(examples):
        x_batch[i, :x.numel()] = x
        y_batch[i, :y.numel()] = y
    # Right padding cannot affect earlier causal positions; padded labels ignored.
    return x_batch.to(device), y_batch.to(device)
