# Aetherion

**Aetherion** is an open-source experimental language-model project. Our goal is to understand, implement, train, evaluate, and eventually serve an original decoder-only Transformer.

> **Status: Phase 0 / engineering prototype.** This repository starts with a small model using random weights and byte-level tokenization. It is **not** a competitive general-purpose assistant, and no frontier-level capabilities are claimed.

## Project tracks

| Track | Goal | State |
|---|---|---|
| Aetherion-Micro | Reproducible train/generate/test loop on CPU or GPU | Phase 0 |
| Aetherion-125M | Trained tokenizer, clean data, ~125M parameter baseline | Planned |
| Aetherion-Instruct | Instruction tuning, evaluation, model card | Planned |
| Aetherion-Scale | Distributed training and larger model feasibility | Research |

## Quick start

Python 3.10+ and PyTorch:

```bash
git clone https://github.com/3more102/Aetherion.git
cd Aetherion
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python train.py --steps 5 --batch-size 2 --d-model 32 --layers 2 --heads 4 --block-size 32
python generate.py --prompt "Hello" --tokens 30
python evaluate.py --checkpoint checkpoints/demo.pt --data data/demo.txt --device cpu
```

The sample corpus is synthetic demonstration text, not sufficient to build a chat assistant. Training a strong model from scratch requires vastly more data and compute.

## Architecture

Input bytes → token embeddings → N × (RMSNorm → causal attention + RoPE → residual → RMSNorm → SwiGLU → residual) → RMSNorm → tied LM head → next-token loss.

The research target for Aetherion-125M is **14 layers, width 768, 12 attention heads, 32k vocabulary, 2048 context and SwiGLU**, estimated around **124M parameters** with tied embeddings. This is a target, not implemented or trained in the Micro prototype.

## Documentation

- [Build plan: execution gates and next PRs](docs/BUILD_PLAN.md)
- [Engineering roadmap](docs/ROADMAP.md)
- [Architecture and design choices](docs/ARCHITECTURE.md)
- [Dataset and evaluation policy](docs/DATA_AND_EVALUATION.md)

## Principles

1. Reproducible tests, pinned experiments, transparent limitations.
2. No claim of pretrained or evaluated capability without measured evidence.
3. Use data with appropriate licenses and provenance; respect privacy, copyright, and opt-outs.
4. Separate pretraining, instruction tuning, inference and evaluation.
5. Publish safety evaluations and limitations before any serious deployment.

## References

- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Hugging Face: train a causal language model from scratch](https://huggingface.co/docs/course/chapter7/6)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/main/quicktour)

## License

MIT (source code). Datasets and later model weights require independent licensing decisions.
