import numpy as np


def cosine_similarity(
    query_vector,
    matrix
):

    query_norm = np.linalg.norm(
        query_vector
    )

    matrix_norms = np.linalg.norm(
        matrix,
        axis=1
    )

    denominator = (
        matrix_norms
        * query_norm
    )

    # Protect against division by zero
    denominator[
        denominator == 0
    ] = 1.0

    scores = (
        matrix @ query_vector
    ) / denominator

    return scores



def normalise_similarity(scores):
    """
    Convert cosine similarity
    from -1..1 to 0..1.
    """

    return (
        scores + 1
    ) / 2



def find_top_n(
    df,
    matrix,
    reference_index,
    n=10
):

    query = matrix[
        reference_index
    ]

    scores = cosine_similarity(
        query,
        matrix
    )

    ranked = np.argsort(
        scores
    )[::-1]

    # Remove reference suburb itself
    ranked = [
        i
        for i in ranked
        if i != reference_index
    ]

    top = ranked[:n]

    results = df.iloc[top][
        [
            "sa2_code",
            "sa2_name",
            "state"
        ]
    ].copy()

    results["raw_similarity"] = (
        scores[top]
    )

    results["similarity"] = (
        normalise_similarity(
            scores[top]
        )
    )

    results["rank"] = range(
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
    ]
    
def compare_rankings(
numeric_results,
hybrid_results
):

    numeric_ranks = dict(
        zip(
            numeric_results[
                "sa2_code"
            ],
            numeric_results[
                "rank"
            ]
        )
    )

    hybrid_ranks = dict(
        zip(
            hybrid_results[
                "sa2_code"
            ],
            hybrid_results[
                "rank"
            ]
        )
    )


    comparison = (
        hybrid_results.copy()
    )


    comparison[
        "numeric_rank"
    ] = comparison[
        "sa2_code"
    ].map(
        numeric_ranks
    )


    comparison[
        "hybrid_rank"
    ] = comparison[
        "rank"
    ]


    comparison[
        "rank_delta"
    ] = (
        comparison[
            "numeric_rank"
        ]
        -
        comparison[
            "hybrid_rank"
        ]
    )


    return comparison