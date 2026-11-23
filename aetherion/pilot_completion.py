"""M38: fail-closed verification of the final checkpoint of a guarded pilot.

This is an artifact consistency check, never proof that training occurred,
quality improved, or the data are authorized. Only inspect trusted local
checkpoints; weights_only avoids general pickle execution, not all risks.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile


def sha256_file(path: str | Path) -> str:
    h = sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def final_checkpoint_path(output: str | Path, *, steps: int, retained: bool) -> Path:
    if type(steps) is not int or steps < 1 or type(retained) is not bool:
        raise ValueError('Invalid final checkpoint step or retention option')
    path = Path(output)
    if not path.name or path.name in ('.', '..'):
        raise ValueError('Final checkpoint needs a filename')
    if retained:
        path = path.with_name(f'{path.stem}-step{steps:08d}{path.suffix}')
    return path


def expected_state_shapes(cfg) -> dict[str, tuple[int, ...]]:
    """Inspect exact decoder shapes without constructing a 125M model."""
    n, d, v = cfg.n_layers, cfg.d_model, cfg.vocab_size
    if min(n, d, v) <= 0 or d % cfg.n_heads:
        raise ValueError('Invalid model config for checkpoint verification')
    hidden = math.ceil((8 * d / 3) / 32) * 32
    shapes = {'embed.weight': (v, d), 'lm_head.weight': (v, d), 'norm.weight': (d,)}
    for i in range(n):
        p = f'blocks.{i}.'
        shapes.update({
            p + 'attn_norm.weight': (d,),
            p + 'attn.qkv.weight': (3 * d, d),
            p + 'attn.out.weight': (d, d),
            p + 'ffn_norm.weight': (d,),
            p + 'ffn.gate_up.weight': (2 * hidden, d),
            p + 'ffn.down.weight': (d, hidden),
        })
    return shapes


def verify_pilot_completion(*, output: str, steps: int, retained: bool,
                            cfg, data_dir: str | Path, tokenizer_path: str | Path,
                            batch_size: int, accumulation: int, schedule_steps: int,
                            warmup_steps: int, precision: str, device: str,
                            seed: int = 42) -> dict:
    """Require an exact final v2 checkpoint with coherent frozen provenance.

    Loads only after the trainer exits successfully. Does not certify
    honest gradient updates or authenticate locally supplied metadata.
    """
    import torch

    path = final_checkpoint_path(output, steps=steps, retained=retained)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Final pilot checkpoint is missing or not a regular non-symlink file')
    checkpoint_sha_before = sha256_file(path)
    try:
        # mmap avoids eagerly copying the entire 125M payload into RAM.
        checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except (RuntimeError, ValueError, OSError, EOFError, TypeError) as exc:
        raise ValueError('Cannot read trusted final checkpoint') from exc
    if not isinstance(checkpoint, dict) or checkpoint.get('format_version') != 2:
        raise ValueError('Final checkpoint is not resumable format-v2')
    if type(checkpoint.get('step')) is not int or checkpoint['step'] != steps:
        raise ValueError('Final checkpoint optimizer step does not match planned target')
    if checkpoint.get('seed') != seed or checkpoint.get('config') != asdict(cfg):
        raise ValueError('Final checkpoint seed or model architecture mismatch')

    hparams = {
        'batch_size': batch_size, 'learning_rate': 3e-4, 'weight_decay': .1,
        'accumulation': accumulation, 'schedule_steps': schedule_steps,
        'warmup_steps': warmup_steps, 'min_lr_ratio': .1,
        'precision': precision, 'device': str(torch.device(device)),
    }
    if checkpoint.get('train_hparams') != hparams:
        raise ValueError('Final checkpoint training hyperparameters mismatch')
    data = checkpoint.get('data')
    if not isinstance(data, dict) or data.get('explicit_validation_file') is not True:
        raise ValueError('Final checkpoint lacks separate held-out validation provenance')
    folder = Path(data_dir)
    tokenizer = Path(tokenizer_path)
    for field, file in (
        ('train_sha256', folder / 'train.txt'),
        ('validation_sha256', folder / 'validation.txt'),
        ('prepared_manifest_sha256', folder / 'manifest.json'),
        ('tokenizer_sha256', tokenizer),
    ):
        if not file.is_file() or file.is_symlink() or data.get(field) != sha256_file(file):
            raise ValueError(f'Final checkpoint provenance mismatch: {field}')
    if data.get('tokenizer_type') != 'bpe' or data.get('tokenizer_vocab_size') != cfg.vocab_size:
        raise ValueError('Final checkpoint tokenizer mode/vocabulary mismatch')

    state = checkpoint.get('state_dict')
    expected_shapes = expected_state_shapes(cfg)
    if not isinstance(state, dict) or set(state) != set(expected_shapes):
        raise ValueError('Final checkpoint state_dict architecture keys mismatch')
    for key, shape in expected_shapes.items():
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
            raise ValueError(f'Final checkpoint parameter shape or dtype mismatch: {key}')
    if state['embed.weight'].untyped_storage().data_ptr() != state['lm_head.weight'].untyped_storage().data_ptr():
        raise ValueError('Final checkpoint embedding and LM head are not tied')
    opt = checkpoint.get('optimizer_state_dict')
    if not isinstance(opt, dict) or not isinstance(opt.get('state'), dict) or not opt['state']:
        raise ValueError('Final checkpoint optimizer state is absent')
    if not isinstance(checkpoint.get('torch_rng_state'), torch.Tensor) or not isinstance(checkpoint.get('train_generator_state'), torch.Tensor):
        raise ValueError('Final checkpoint RNG restoration state is absent')
    if checkpoint_sha_before != sha256_file(path):
        raise ValueError('Final checkpoint changed while being inspected')
    return {
        'schema_version': 1,
        'kind': 'aetherion-pilot-final-checkpoint-consistency-receipt',
        'result': 'final_checkpoint_consistency_checked',
        'checkpoint': str(path),
        'checkpoint_sha256': checkpoint_sha_before,
        'checkpoint_bytes': path.stat().st_size,
        'checkpoint_step': steps,
        'model_config': asdict(cfg),
        'unique_parameter_count_from_shapes': sum(math.prod(v) for k, v in expected_shapes.items() if k != 'lm_head.weight'),
        'train_sha256': data['train_sha256'],
        'validation_sha256': data['validation_sha256'],
        'prepared_manifest_sha256': data['prepared_manifest_sha256'],
        'tokenizer_sha256': data['tokenizer_sha256'],
        'limitations': (
            'Local structural and checksum consistency only. Does not prove that '
            'optimization was run honestly, source data are licensed, the model '
            'learned anything, or assistant quality/safety/phone inference.'
        ),
    }


def write_completion_receipt(report: dict, target: str | Path) -> Path:
    """Create a durable JSON receipt atomically without overwriting prior evidence."""
    destination = Path(target)
    if destination.is_symlink() or destination.exists():
        raise FileExistsError(f'Completion receipt destination already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as file:
            json.dump(report, file, sort_keys=True, indent=2, allow_nan=False)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.link(temp, destination)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return destination
