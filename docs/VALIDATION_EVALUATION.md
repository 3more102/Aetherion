# M2 — reproducible validation evaluation

Evaluate a trusted format-v2 checkpoint against its original validation split.

    python evaluate.py --checkpoint checkpoints/demo.pt
    python evaluate.py --checkpoint checkpoints/bpe-micro.pt --data data/processed/v1/train.txt --val-data data/processed/v1/validation.txt --tokenizer artifacts/tokenizer.json

For demo byte mode, the split warning is emitted to stderr; stdout is JSON.
Use the same paths/contents as the training run; fingerprints must match.

The evaluator scores every next-token target exactly once, including partial
final windows. Default stride equals context size; smaller strides increase
available left context. Stride 1 uses maximal available context for every
target but is much more expensive. Compare results only with identical
tokenization, data, context and stride. The JSON report records these fields,
checkpoint and validation SHA-256, evaluated target count, mean NLL (nats),
and perplexity. The test split is deliberately not accessed.

These are engineering metrics, not evidence of instruction following,
generalization, or frontier-model quality. The synthetic byte demo is not
a deployment benchmark. A properly licensed, contamination-reviewed corpus
and independent test suite are still needed.
