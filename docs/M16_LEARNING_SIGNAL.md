# M16: Synthetic learning-signal gate

Aetherion has training/generation smoke tests, but successfully running a
forward/backward pass is not sufficient to show that an optimizer can improve
a next-token objective. This gate deliberately checks that a tiny decoder
learns one deterministic eight-token cycle.

```bash
python overfit_check.py --steps 40 --seed 17
python -m pytest -q tests/test_learning_signal.py
```

The command emits a JSON report with exact token/step counts, model parameter
count, initial/final negative log-likelihood, loss ratio and a pass/fail field.
A non-passing report exits nonzero. It saves **no model weights** and uses
**no external corpus**. The CI test checks same-environment repeatability on
CPU and detects regression of the optimizer learning signal.

This is intentional **memorization** of one short synthetic example, not
validation-set improvement, generalization, human preference, reasoning,
language understanding, or readiness to compete with modern assistants.
Seed control alone does not guarantee identical values across PyTorch versions
and different hardware. See the official
[PyTorch reproducibility note](https://pytorch.org/docs/stable/notes/randomness.html).

The next real training gates remain authorized human-reviewed data,
verified train-only tokenizer, measured device memory and throughput,
held-out validation loss/perplexity, saved checkpoints and independent
non-contaminated task evaluation. No 125M pretraining run is claimed.
