"""M35: Exact planned token and checkpoint-write budget for guarded 125M pilots.

The result counts sampled training positions, not unique corpus tokens or
financial cost. Resume is conservatively treated as a full target-step run.
"""

from __future__ import annotations

DEFAULT_SAVE_EVERY = 50  # train.py default; keep in sync with its CLI


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"Pilot {label} must be a positive integer")
    return value


def check_pilot_budget(*, steps: int, context: int, batch_size: int,
                       accumulation: int, schedule_steps: int,
                       save_every: int | None, retain_step_checkpoints: bool,
                       max_total_tokens: int, max_checkpoint_writes: int,
                       resume: bool = False) -> dict:
    """Fail closed on a user-declared upper budget before expensive I/O.

    Steps are the total target optimizer updates, including any prior steps
    before resume; the resulting workload is therefore an upper bound for
    a resumed run, not an estimate of remaining work.
    """
    steps = _positive_int(steps, "steps")
    context = _positive_int(context, "context")
    batch_size = _positive_int(batch_size, "batch size")
    accumulation = _positive_int(accumulation, "accumulation")
    schedule_steps = _positive_int(schedule_steps, "schedule steps")
    max_total_tokens = _positive_int(max_total_tokens, "max total tokens")
    max_checkpoint_writes = _positive_int(max_checkpoint_writes, "max checkpoint writes")
    if steps > schedule_steps:
        raise ValueError("Pilot steps exceed schedule steps")
    if type(retain_step_checkpoints) is not bool or type(resume) is not bool:
        raise ValueError("Pilot retention and resume flags must be boolean")
    if save_every is None:
        interval = DEFAULT_SAVE_EVERY
    elif type(save_every) is int and save_every >= 0:
        interval = save_every
    else:
        raise ValueError("Pilot save-every must be a nonnegative integer")

    positions_per_update = context * batch_size * accumulation
    total_positions = steps * positions_per_update
    if total_positions > max_total_tokens:
        raise ValueError("Pilot planned token positions exceed max-total-tokens")

    # Every periodic save is a write; the trainer saves the final step too.
    writes = (steps // interval + int(steps % interval != 0)) if interval else 1
    if writes > max_checkpoint_writes:
        raise ValueError("Pilot checkpoint writes exceed max-checkpoint-writes")

    return {
        "schema_version": 1,
        "kind": "aetherion-125m-pilot-work-budget",
        "result": "requested_work_within_operator_limits",
        "total_step_target": steps,
        "schedule_step_target": schedule_steps,
        "microbatches_per_step": accumulation,
        "tokens_per_microbatch": context * batch_size,
        "sampled_positions_per_optimizer_step": positions_per_update,
        "max_planned_sampled_token_positions": total_positions,
        "operator_max_total_tokens": max_total_tokens,
        "save_every_steps": interval,
        "max_planned_checkpoint_writes": writes,
        "max_planned_checkpoint_files": writes if retain_step_checkpoints else 1,
        "operator_max_checkpoint_writes": max_checkpoint_writes,
        "retained_checkpoints": retain_step_checkpoints,
        "resume_requested": resume,
        "limitations": (
            "This is a conservative full-target calculation, not measured GPU time, "
            "unique training tokens, device RAM/disk, electricity, monetary cost or "
            "proof of data rights. A resumed run usually performs fewer actual steps; "
            "unexpected failures can create partial artifacts."
        ),
    }
