import numpy as np


def l2_normalise(matrix):

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    return matrix / norms



def fuse_vectors(
    numeric_matrix,
    text_matrix,
    alpha=0.5
):

    if not 0 <= alpha <= 1:
        raise ValueError(
            "alpha must be between 0 and 1"
        )


    if len(numeric_matrix) != len(text_matrix):
        raise ValueError(
            "Numeric and text matrices must "
            "contain the same number of suburbs"
        )


    numeric_normalised = l2_normalise(
        numeric_matrix
    )

    text_normalised = l2_normalise(
        text_matrix
    )


    numeric_weighted = (
        (1 - alpha)
        * numeric_normalised
    )

    text_weighted = (
        alpha
        * text_normalised
    )


    hybrid_matrix = np.concatenate(
        [
            numeric_weighted,
            text_weighted
        ],
        axis=1
    )


    return hybrid_matrix