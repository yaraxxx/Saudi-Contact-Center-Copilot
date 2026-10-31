"""
Offline precision/recall of the deterministic Saudi PII detector, per PII type and per dialect.

    uv run python evals/pii_detector_eval.py

No API key needed. Add hard cases to datasets/pii_cases.jsonl whenever you find a miss.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cc_copilot.pii.saudi_patterns import detect_pii  # noqa: E402

DATASET = Path(__file__).parent / "datasets" / "pii_cases.jsonl"


def load_cases(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(cases: list[dict]) -> dict:
    tp, fp, fn = Counter(), Counter(), Counter()
    failures_by_dialect: dict[str, list[str]] = defaultdict(list)

    for case in cases:
        predicted = Counter(m.type.value for m in detect_pii(case["text"]))
        expected = Counter(case["expected"])
        for t in predicted | expected:
            tp[t] += min(predicted[t], expected[t])
            fp[t] += max(predicted[t] - expected[t], 0)
            fn[t] += max(expected[t] - predicted[t], 0)
        if predicted != expected:
            failures_by_dialect[case["dialect"]].append(case["id"])

    per_type = {}
    for t in sorted(tp | fp | fn):
        precision = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 1.0
        recall = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else 1.0
        per_type[t] = {"precision": precision, "recall": recall, "support": tp[t] + fn[t]}

    return {"per_type": per_type, "failures": dict(failures_by_dialect), "cases": len(cases)}


def main() -> None:
    results = evaluate(load_cases())
    print(f"{'type':<12} {'precision':>9} {'recall':>7} {'support':>8}")
    for t, m in results["per_type"].items():
        print(f"{t:<12} {m['precision']:>9.2f} {m['recall']:>7.2f} {m['support']:>8}")
    print(f"\n{results['cases']} cases, failures by dialect: {results['failures'] or 'none'}")


if __name__ == "__main__":
    main()
