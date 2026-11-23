"""M38 CPU fixture verification; does not train a large model."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from aetherion.pilot_completion import (
    expected_state_shapes, final_checkpoint_path,
    verify_pilot_completion, write_completion_receipt,
)


@dataclass
class TinyCfg:
    vocab_size: int = 64
    block_size: int = 8
    n_layers: int = 1
    n_heads: int = 2
    d_model: int = 16
    dropout: float = 0.0


@pytest.fixture
def artifact(tmp_path):
    cfg = TinyCfg()
    folder = tmp_path / 'prepared'
    folder.mkdir()
    for filename in ('train.txt', 'validation.txt', 'manifest.json'):
        (folder / filename).write_text('synthetic-' + filename)
    token = tmp_path / 'tokenizer.json'
    token.write_text('synthetic tokenizer')
    def digest(path):
        return sha256(path.read_bytes()).hexdigest()
    data = { 'train_sha256': digest(folder/'train.txt'),
             'validation_sha256': digest(folder/'validation.txt'),
             'prepared_manifest_sha256': digest(folder/'manifest.json'),
             'tokenizer_sha256': digest(token),
             'tokenizer_type': 'bpe', 'tokenizer_vocab_size': cfg.vocab_size,
             'explicit_validation_file': True }
    state = {name: torch.zeros(shape) for name, shape in expected_state_shapes(cfg).items()}
    state['lm_head.weight'] = state['embed.weight']
    checkpoint = {
        'format_version': 2, 'step': 2, 'seed': 42,
        'config': vars(cfg), 'train_hparams': {
            'batch_size': 1, 'learning_rate': 3e-4, 'weight_decay': .1,
            'accumulation': 4, 'schedule_steps': 4, 'warmup_steps': 0,
            'min_lr_ratio': .1, 'precision': 'fp32', 'device': 'cpu',
        },
        'data': data, 'state_dict': state,
        'optimizer_state_dict': {'state': {0: {'step': torch.tensor(2.)}}},
        'torch_rng_state': torch.get_rng_state(),
        'train_generator_state': torch.Generator().get_state(),
    }
    output = tmp_path / 'pilot.pt'
    torch.save(checkpoint, output)
    args = dict(output=str(output), steps=2, retained=False, cfg=cfg,
                data_dir=folder, tokenizer_path=token,
                batch_size=1, accumulation=4, schedule_steps=4,
                warmup_steps=0, precision='fp32', device='cpu')
    return checkpoint, args


def test_accepts_exact_metadata_and_checksum(artifact):
    _, args = artifact
    result = verify_pilot_completion(**args)
    assert result['checkpoint_step'] == 2
    assert len(result['checkpoint_sha256']) == 64
    assert result['unique_parameter_count_from_shapes'] > 0
    assert 'does not prove' in result['limitations'].lower()


def test_final_retained_checkpoint_selection(artifact, tmp_path):
    checkpoint, args = artifact
    assert final_checkpoint_path(args['output'], steps=2, retained=True).name == 'pilot-step00000002.pt'
    args['retained'] = True
    with pytest.raises(ValueError, match='missing'):
        verify_pilot_completion(**args)
    torch.save(checkpoint, final_checkpoint_path(args['output'], steps=2, retained=True))
    assert verify_pilot_completion(**args)['checkpoint_step'] == 2


@pytest.mark.parametrize('field, value, error', [
    ('step', 1, 'step'), ('seed', 99, 'seed'),
    ('format_version', 1, 'format-v2'),
    ('optimizer_state_dict', {'state': {}}, 'optimizer'),
])
def test_bad_final_metadata_fails(artifact, field, value, error):
    obj, args = artifact
    obj[field] = value
    torch.save(obj, args['output'])
    with pytest.raises(ValueError, match=error):
        verify_pilot_completion(**args)


def test_rejects_wrong_state_shape_or_untied_weights(artifact):
    obj, args = artifact
    obj['state_dict']['norm.weight'] = torch.ones(4)
    torch.save(obj, args['output'])
    with pytest.raises(ValueError, match='shape or dtype'):
        verify_pilot_completion(**args)
    obj['state_dict']['norm.weight'] = torch.ones(16)
    obj['state_dict']['lm_head.weight'] = torch.zeros(64,16)
    torch.save(obj, args['output'])
    with pytest.raises(ValueError, match='not tied'):
        verify_pilot_completion(**args)


def test_rejects_drifted_data_or_tokenizer(artifact):
    obj, args = artifact
    (args['data_dir'] / 'train.txt').write_text('changed')
    with pytest.raises(ValueError, match='train_sha256'):
        verify_pilot_completion(**args)
    (args['data_dir'] / 'train.txt').write_text('synthetic-train.txt')
    args['tokenizer_path'].write_text('changed')
    with pytest.raises(ValueError, match='tokenizer_sha256'):
        verify_pilot_completion(**args)


def test_rejects_missing_or_symlink_checkpoint(artifact, tmp_path):
    obj, args = artifact
    Path(args['output']).unlink()
    with pytest.raises(ValueError, match='missing'):
        verify_pilot_completion(**args)
    try:
        Path(args['output']).symlink_to(tmp_path / 'nonexistent.pt')
    except (OSError, NotImplementedError):
        pytest.skip('symlink unsupported')
    with pytest.raises(ValueError, match='missing'):
        verify_pilot_completion(**args)


def test_receipt_is_atomic_and_exclusive(artifact, tmp_path):
    _, args = artifact
    report = verify_pilot_completion(**args)
    target = tmp_path / 'reports' / 'completion.json'
    write_completion_receipt(report, target)
    assert json.loads(target.read_text()) == report
    with pytest.raises(FileExistsError):
        write_completion_receipt(report, target)
    assert json.loads(target.read_text()) == report
