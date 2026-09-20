from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from engine.similarity import cosine_similarity


# ============================================================
# BASIC RANKING METRICS
# ============================================================

def precision_at_k(
    returned_codes: Iterable[str],
    expected_codes: Iterable[str],
    k: int,
) -> float:
    """
    Precision@K

    Measures how many of the returned top-K suburbs
    are relevant according to the golden set.

    Precision@K = relevant results in top K / K
    """

    returned = list(returned_codes)[:k]
    expected = set(expected_codes)

    if k <= 0:
        return 0.0

    if not returned:
        return 0.0

    hits = sum(1 for code in returned if code in expected)

    return hits / len(returned)


def recall_at_k(
    returned_codes: Iterable[str],
    expected_codes: Iterable[str],
    k: int,
) -> float:
    """
    Recall@K

    Measures how many of the expected relevant suburbs
    were retrieved in the top-K results.

    Recall@K = relevant results retrieved / total expected relevant
    """

    returned = set(list(returned_codes)[:k])
    expected = set(expected_codes)

    if not expected:
        return 0.0

    hits = len(returned.intersection(expected))

    return hits / len(expected)


def count_hits(
    returned_codes: Iterable[str],
    expected_codes: Iterable[str],
    k: int,
) -> int:
    """
    Count the number of relevant expected suburbs
    appearing in the top-K results.
    """

    returned = set(list(returned_codes)[:k])
    expected = set(expected_codes)

    return len(returned.intersection(expected))


def count_negative_matches(
    returned_codes: Iterable[str],
    negative_codes: Iterable[str],
    k: int,
) -> int:
    """
    Count how many known negative suburbs appear
    in the top-K results.

    A value of 0 means the negative sanity check passed
    for that reference.
    """

    returned = set(list(returned_codes)[:k])
    negative = set(negative_codes)

    return len(returned.intersection(negative))


# ============================================================
# SELF-CONSISTENCY
# ============================================================

def self_consistency_check(
    matrix: np.ndarray,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """
    Self-consistency / self-neighbour similarity check.

    For every suburb vector:
      1. Compare it against the complete matrix.
      2. Find its highest similarity.
      3. Confirm that its own vector is the nearest neighbour.
      4. Confirm that self-similarity is approximately 1.

    This check is performed BEFORE removing the reference suburb
    from search results.

    Returns summary statistics and any failures.
    """

    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError(
            "Self-consistency requires a 2D matrix."
        )

    total_rows = matrix.shape[0]

    if total_rows == 0:
        return {
            "passed": True,
            "total_rows": 0,
            "failure_count": 0,
            "failures": [],
            "min_self_similarity": None,
            "avg_self_similarity": None,
            "max_self_similarity": None,
        }

    failures: list[dict[str, Any]] = []
    self_similarities: list[float] = []

    for i in range(total_rows):

        query_vector = matrix[i]

        # Compare this vector against every vector in the matrix.
        scores = cosine_similarity(
            query_vector,
            matrix,
        )

        scores = np.asarray(scores).reshape(-1)

        if len(scores) != total_rows:
            raise ValueError(
                "cosine_similarity returned an unexpected number "
                "of similarity scores."
            )

        # Self similarity is the score at the vector's own index.
        self_similarity = float(scores[i])

        self_similarities.append(self_similarity)

        # Highest similarity should be the vector itself.
        nearest_index = int(np.argmax(scores))

        is_self_nearest = nearest_index == i
        is_self_similarity_valid = (
            abs(self_similarity - 1.0) <= tolerance
        )

        if not (
            is_self_nearest
            and is_self_similarity_valid
        ):
            failures.append(
                {
                    "index": i,
                    "nearest_index": nearest_index,
                    "self_similarity": self_similarity,
                    "is_self_nearest": is_self_nearest,
                    "is_self_similarity_valid": is_self_similarity_valid,
                }
            )

    return {
        "passed": len(failures) == 0,
        "total_rows": total_rows,
        "failure_count": len(failures),
        "failures": failures,
        "min_self_similarity": float(
            np.min(self_similarities)
        ),
        "avg_self_similarity": float(
            np.mean(self_similarities)
        ),
        "max_self_similarity": float(
            np.max(self_similarities)
        ),
    }


# ============================================================
# WEIGHT STABILITY
# ============================================================

def top_k_overlap(
    baseline_codes: Iterable[str],
    changed_codes: Iterable[str],
    k: int,
) -> float:
    """
    Calculate top-K overlap between two ranked result lists.

    Top-K overlap = common suburbs in both top-K lists / K
    """

    baseline = set(list(baseline_codes)[:k])
    changed = set(list(changed_codes)[:k])

    if k <= 0:
        return 0.0

    return len(baseline.intersection(changed)) / k


def weight_stability_summary(
    baseline_codes: Iterable[str],
    changed_codes: Iterable[str],
    k: int,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Summarise weight stability after a small KPI-weight change.

    The evaluation passes when the top-K overlap is at least
    the supplied stability threshold.
    """

    overlap = top_k_overlap(
        baseline_codes,
        changed_codes,
        k,
    )

    return {
        "top_k_overlap": overlap,
        "threshold": threshold,
        "passed": overlap >= threshold,
    }