"""Golden-set configuration and metrics for Suburb Lookalike evaluation."""
import json
from pathlib import Path

GOLDEN_SET_PATH = Path(__file__).with_name("golden_set.json")
DEFAULT_K = 10
DEFAULT_ALPHA = 0.2
ALPHA_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def load_golden_set(path=GOLDEN_SET_PATH):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "references" in data:
        refs = data["references"]
    else:
        refs = [
            {"reference_code": code, **config}
            for code, config in data.items()
        ]
        data = {"references": refs}

    if len(refs) != 10:
        raise ValueError(f"Golden set must contain exactly 10 references; found {len(refs)}")
    return data


GOLDEN_SET = {
    reference["reference_code"]: {
        key: value
        for key, value in reference.items()
        if key != "reference_code"
    }
    for reference in load_golden_set()["references"]
}

def precision_at_k(returned, expected, k=DEFAULT_K):
    returned = returned[:k]
    if not returned:
        return 0.0
    expected = set(expected)
    return sum(x in expected for x in returned) / len(returned)

def recall_at_k(returned, expected, k=DEFAULT_K):
    if not expected:
        return 0.0
    expected = set(expected)
    return sum(x in expected for x in returned[:k]) / len(expected)

if __name__ == "__main__":
    data = load_golden_set()
    print(f"Loaded {len(data['references'])} golden-set references")
