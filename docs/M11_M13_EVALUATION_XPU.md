# M11–M13: task evaluation and optional Intel XPU

M11 scores strict numeric outputs (exact rational equality) and exact text.
M12 checks Python compilation and AST equivalence without executing model
output. A structurally different correct program may fail AST matching;
syntax or AST scores are not functional correctness or pass@k.

Use independently authored/licensed held-out tasks with unique id, kind,
prompt and answer fields. The bundled four smoke tasks are public development
fixtures and must not be used to claim model capability.

    python evaluate_tasks.py --tasks data/eval_smoke_tasks.jsonl --predictions predictions.jsonl

Each prediction row has id and prediction. When the M8–M10 chat module is
available, use --checkpoint checkpoints/instruct.pt --tokenizer
artifacts/tokenizer.json instead of --predictions. Generation is greedy.
Reports record task and prediction/checkpoint SHA, per-kind counts and metrics.

M13 adds device selection, BF16 autocast, XPU RNG-state checkpointing and
allocator profiling. Consult the official PyTorch Intel GPU setup guide:
https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html

    python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
    python benchmark_training.py --device xpu --preset micro --context 32
    python benchmark_training.py --device xpu --precision bf16 --preset micro --context 32
    python -m pytest -q tests/test_tasks_xpu.py

The hardware test skips when no XPU is available. CPU CI cannot establish
Intel Arc performance, BF16 operator support or bitwise resume equivalence
on a specific GPU. Measure first; no XPU speedup is claimed. All checkpoint
state must remain on a trusted local source. Distributed training is separate.
