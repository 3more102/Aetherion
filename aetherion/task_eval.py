"""Offline mathematics and Python structural evaluation; never execute candidate code."""
import ast
from fractions import Fraction
import re
import json
from pathlib import Path
from hashlib import sha256

NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?")
FENCE = re.compile(r"\A```(?:python|py)?[ \t]*\n(.*?)\n```\Z", re.S)


def numeric_answer(value):
    value = value.strip()
    if len(value) > 128 or not NUMBER.fullmatch(value):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def code_source(value):
    value = value.strip()
    match = FENCE.fullmatch(value)
    return match.group(1) if match else value


def code_structure(value):
    if not value.strip() or len(value) > 20000:
        return None
    try:
        tree = ast.parse(code_source(value), mode="exec")
        # Compilation checks language constraints (e.g. return outside function),
        # but never runs any generated program.
        compile(tree, "<candidate>", "exec")
        if not tree.body:
            return None
        return ast.dump(tree, include_attributes=False)
    except (SyntaxError, ValueError, TypeError, RecursionError, OverflowError):
        return None


def score(task, prediction):
    if not isinstance(prediction, str):
        raise ValueError("Prediction must be text")
    expected, kind = task["answer"], task["kind"]
    if kind == "numeric":
        gold = numeric_answer(expected)
        if gold is None:
            raise ValueError("Invalid numeric reference")
        guess = numeric_answer(prediction)
        return {"correct": guess is not None and guess == gold}
    if kind == "exact":
        return {"correct": prediction.strip() == expected.strip()}
    if kind == "python":
        gold = code_structure(expected)
        if gold is None:
            raise ValueError("Invalid Python reference")
        guess = code_structure(prediction)
        return {"valid_syntax": guess is not None, "ast_exact": guess is not None and guess == gold}
    raise ValueError("Unknown task kind: " + kind)


def load_tasks(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    ids = set()
    for task in rows:
        if not isinstance(task, dict) or any(not isinstance(task.get(k), str) or not task[k].strip()
                                                 for k in ("id", "kind", "prompt", "answer")):
            raise ValueError("Tasks require nonempty id, kind, prompt, answer")
        if task["id"] in ids:
            raise ValueError("Duplicate task id")
        ids.add(task["id"])
        score(task, task["answer"])
    if not rows:
        raise ValueError("Empty evaluation tasks")
    return rows, sha256(raw).hexdigest()


def summarize(tasks, predictions):
    ids = {task["id"] for task in tasks}
    if predictions.keys() != ids:
        raise ValueError("Prediction ids must match task ids exactly")
    totals = {}
    details = []
    for task in tasks:
        metrics = score(task, predictions[task["id"]])
        bucket = totals.setdefault(task["kind"], {"count": 0})
        bucket["count"] += 1
        for key, value in metrics.items():
            bucket[key] = bucket.get(key, 0) + int(value)
        details.append({"id": task["id"], "kind": task["kind"], **metrics})
    return {"by_kind": {kind: {**counts, **{key + "_rate": value / counts["count"]
                                            for key, value in counts.items() if key != "count"}}
                         for kind, counts in totals.items()}, "details": details,
            "limitations": "Numeric/exact answer matching and Python syntax/AST matching only; not reasoning-process or functional code correctness."}
