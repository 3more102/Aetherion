"""Score offline predictions or generate deterministic SFT responses when chat is installed."""
import argparse
import json
from pathlib import Path
from hashlib import sha256

from aetherion.task_eval import load_tasks, summarize


def read_predictions(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    result = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not isinstance(row.get("prediction"), str) or row["id"] in result):
            raise ValueError("Expected unique prediction ids and text predictions")
        result[row["id"]] = row["prediction"]
    return result, sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions")
    source.add_argument("--checkpoint")
    p.add_argument("--tokenizer")
    p.add_argument("--device", default="auto")
    p.add_argument("--tokens", type=int, default=64)
    args = p.parse_args()
    tasks, task_hash = load_tasks(args.tasks)
    provenance = {"tasks_sha256": task_hash}
    if args.predictions:
        predictions, fingerprint = read_predictions(args.predictions)
        provenance.update(mode="offline", predictions_sha256=fingerprint)
    else:
        if not args.tokenizer or args.tokens < 1:
            p.error("Model evaluation requires tokenizer and positive generation budget")
        try:
            from chat import load_chat, reply
        except ImportError:
            p.error("Model evaluation requires the M8–M10 chat implementation")
        from train import resolve_device
        from aetherion.tokenizer_runtime import load_verified_tokenizer
        _, meta = load_verified_tokenizer(args.tokenizer)
        model, tok = load_chat(args.checkpoint, args.tokenizer, resolve_device(args.device))
        predictions = {task["id"]: reply(model, tok, task["prompt"], args.tokens, 0.0)
                       for task in tasks}
        provenance.update(mode="model", checkpoint_sha256=sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                          tokenizer_sha256=meta["tokenizer_sha256"], max_new_tokens=args.tokens,
                          device=str(next(model.parameters()).device), temperature=0.0)
    print(json.dumps({"schema_version": 1, **provenance, **summarize(tasks, predictions)},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
