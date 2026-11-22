# M8–M10: experimental instruction tuning, cached inference, local chat

M8 uses supervised next-token cross-entropy on assistant response tokens and
EOS only. Prompts and padding are ignored. Supply separate train and validation
JSONL, with an instruction, response, source, license, group and authorized=true
per row. Group overlap and exact instruction duplicates are refused. These
checks and self-attestation are not a substitute for a human rights, privacy,
contamination and safety review. Examples exceeding context are rejected.
Start from a trusted, trained BPE base checkpoint, not a random fixture.

    python finetune.py --base checkpoints/pilot.pt --tokenizer artifacts/tokenizer.json --train data/instruct/train.jsonl --validation data/instruct/validation.jsonl --steps 100 --output checkpoints/instruct.pt

Resume with the same inputs and optimizer settings, --resume checkpoints/instruct.pt
and a larger total --steps. Saves occur at the end of each invocation.

M9 adds inference-only per-layer K/V caching with relative RoPE offsets and
correct causal masks for multi-token cached inputs. At window overflow the
cache is rebuilt to preserve original sliding-window semantics. Dynamic
concatenation is not a static or paged KV memory system. No measured speedup
is claimed.

M10 provides local single-turn interactive or one-shot chat:

    python chat.py --checkpoint checkpoints/instruct.pt --tokenizer artifacts/tokenizer.json --tokens 64

Only matching SFT checkpoints are accepted. The CLI has no conversation
history, tools, authentication, deployment server or safety guarantees.
Smoke fixtures use random base weights and synthetic instructions; they
establish pipeline behavior, not meaningful assistant capability.
