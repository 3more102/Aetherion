# M14–M15: distributed training and architecture scaling

Single-node DDP replicates weights, gradients and optimizer state on every
rank. It does not shard memory. CPU/Gloo and CUDA/NCCL are supported by
the prototype. No multi-node, XPU distributed, FSDP or ZeRO implementation
is claimed. Only dropout-free configs are accepted for step/rank-derived
sampling on resume. Exact GPU reproducibility is not established.

    torchrun --standalone --nproc-per-node=2 train_distributed.py --device cpu --context 8 --steps 2 --output checkpoints/ddp-smoke.pt

For a CUDA pilot, provide real reviewed prepared data and BPE tokenizer,
--device cuda, --preset 125m, --context 256, and one visible GPU per rank.
Use --activation-checkpointing to trade recomputation for activation memory.
Resume with the same world size, backend, data, precision, schedule and
sampling options. Rank 0 writes an atomic checkpoint after synchronized updates.
Samples may coincide by chance; random sampling is not disjoint sharding.

Architecture-only sizing:

    python inspect_model.py --preset 350m --json
    python inspect_model.py --preset 1.3b --json
    python inspect_model.py --preset 7b --json

These presets retain tied embeddings and full multi-head attention. Model
names are size classes, not exact counts. The FP32 AdamW parameter, gradient
and two-moment estimate uses 16 bytes per unique parameter. This omits
activations, communication buffers, allocator overhead and temporary optimizer
buffers. Lower-precision compute alone does not eliminate FP32 state.
Large distributed presets require --allow-large and actual memory assessment.
No model weights or large-model training results are provided.
