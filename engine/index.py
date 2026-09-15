import numpy as np
import faiss


def build_faiss_index(matrix):
    """
    Build an exact FAISS cosine-similarity index.

    FAISS IndexFlatIP calculates inner product.

    After L2 normalising the vectors,
    inner product becomes equivalent to
    cosine similarity.
    """

    X = np.asarray(
        matrix,
        dtype=np.float32
    ).copy()

    # Normalise every vector to length 1
    faiss.normalize_L2(X)

    # Exact inner-product index
    index = faiss.IndexFlatIP(
        X.shape[1]
    )

    # Add all suburb vectors
    index.add(X)

    return index, X


def faiss_find_top_n(
    df,
    index,
    normalised_matrix,
    reference_index,
    n=10
):
    """
    Find the N most similar suburbs
    using the FAISS index.
    """

    # Get reference suburb vector
    query = normalised_matrix[
        reference_index
    ].reshape(1, -1)

    # Ask for one extra result because
    # the reference suburb will match itself
    similarities, indices = index.search(
        query,
        n + 1
    )

    result_indices = []
    result_scores = []

    for idx, score in zip(
        indices[0],
        similarities[0]
    ):

        # Remove self-match
        if idx == reference_index:
            continue

        result_indices.append(
            int(idx)
        )

        result_scores.append(
            float(score)
        )

        if len(result_indices) == n:
            break

    results = (
        df.iloc[result_indices][
            [
                "sa2_code",
                "sa2_name",
                "state"
            ]
        ]
        .copy()
    )

    results[
        "raw_similarity"
    ] = result_scores

    # Same 0–1 transformation
    # used by our NumPy implementation
    results[
        "similarity"
    ] = (
        results["raw_similarity"] + 1
    ) / 2

    results[
        "rank"
    ] = range(
        1,
        len(results) + 1
    )

    return results[
        [
            "rank",
            "sa2_code",
            "sa2_name",
            "state",
            "similarity",
            "raw_similarity"
        ]
    ].reset_index(
        drop=True
    )