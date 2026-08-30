import numpy as np


def cosine_similarity(query_vector, matrix):

    query_norm = np.linalg.norm(query_vector)

    matrix_norms = np.linalg.norm(
        matrix,
        axis=1
    )

    scores = (
        matrix @ query_vector
    ) / (
        matrix_norms * query_norm
    )

    return scores


def find_top_n(
    df,
    X,
    reference_index,
    n=10
):

    query = X[reference_index]

    scores = cosine_similarity(
        query,
        X
    )

    ranked = np.argsort(scores)[::-1]

    ranked = [
        i
        for i in ranked
        if i != reference_index
    ]

    top = ranked[:n]

    results = df.iloc[top][
        ["sa2_code", "sa2_name", "state"]
    ].copy()

    results["similarity"] = scores[top]

    return results