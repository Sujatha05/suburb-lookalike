import numpy as np
import pandas as pd

from engine.features import KPI_COLS

from engine.weights import (
    KPI_LABELS,
    weights_to_array
)

from engine.fusion import (
    l2_normalise
)

from engine.similarity import (
    cosine_similarity
)


# ============================================================
# TOP KPI CONTRIBUTIONS
# ============================================================

def get_top_kpi_contributions(
    X_numeric,
    reference_index,
    candidate_index,
    weights,
    top_k=3
):

    """
    Find the KPIs that contributed most
    to the numeric similarity between
    the reference suburb and candidate.
    """

    weight_array = weights_to_array(
        weights
    )


    # Apply user-selected feature weights
    X_weighted = (
        X_numeric
        * weight_array
    )


    # L2 normalise numeric block
    X_normalised = l2_normalise(
        X_weighted
    )


    reference_vector = (
        X_normalised[
            reference_index
        ]
    )


    candidate_vector = (
        X_normalised[
            candidate_index
        ]
    )


    # Contribution of each KPI
    # to cosine similarity
    contributions = (
        reference_vector
        * candidate_vector
    )


    contribution_data = []


    for i, kpi in enumerate(
        KPI_COLS
    ):

        contribution_data.append(
            {
                "kpi": kpi,
                "label": KPI_LABELS[kpi],
                "contribution": contributions[i]
            }
        )


    contribution_data = sorted(
        contribution_data,
        key=lambda x: x["contribution"],
        reverse=True
    )


    return contribution_data[
        :top_k
    ]
    

def get_full_ranks(
    matrix,
    reference_index
):

    """
    Return the rank of every suburb
    relative to the reference suburb.
    """

    query = matrix[
        reference_index
    ]


    scores = cosine_similarity(
        query,
        matrix
    )


    ranked_indices = (
        np.argsort(
            scores
        )[::-1]
    )


    ranked_indices = [
        i
        for i in ranked_indices
        if i != reference_index
    ]


    ranks = {}


    for rank, index in enumerate(
        ranked_indices,
        start=1
    ):

        ranks[index] = rank


    return ranks

def get_rank_delta(
    numeric_matrix,
    hybrid_matrix,
    reference_index,
    candidate_index
):

    numeric_ranks = (
        get_full_ranks(
            numeric_matrix,
            reference_index
        )
    )


    hybrid_ranks = (
        get_full_ranks(
            hybrid_matrix,
            reference_index
        )
    )


    numeric_rank = (
        numeric_ranks[
            candidate_index
        ]
    )


    hybrid_rank = (
        hybrid_ranks[
            candidate_index
        ]
    )


    rank_delta = (
        numeric_rank
        - hybrid_rank
    )


    return {
        "numeric_rank": numeric_rank,
        "hybrid_rank": hybrid_rank,
        "rank_delta": rank_delta
    }
    
def explain_results(
    df,
    results,
    X_numeric,
    X_hybrid,
    reference_index,
    weights
):

    numeric_ranks = (
        get_full_ranks(
            X_numeric,
            reference_index
        )
    )


    hybrid_ranks = (
        get_full_ranks(
            X_hybrid,
            reference_index
        )
    )


    explained_rows = []


    for _, row in results.iterrows():

        sa2_code = row[
            "sa2_code"
        ]


        candidate_matches = df.index[
            df["sa2_code"]
            == sa2_code
        ].tolist()


        if not candidate_matches:
            continue


        candidate_index = (
            candidate_matches[0]
        )


        top_kpis = (
            get_top_kpi_contributions(
                X_numeric,
                reference_index,
                candidate_index,
                weights,
                top_k=3
            )
        )


        top_kpi_names = [
            item["label"]
            for item in top_kpis
        ]


        numeric_rank = (
            numeric_ranks[
                candidate_index
            ]
        )


        hybrid_rank = (
            hybrid_ranks[
                candidate_index
            ]
        )


        rank_delta = (
            numeric_rank
            - hybrid_rank
        )


        explained_rows.append(
            {
                "rank": row["rank"],
                "sa2_code": sa2_code,
                "sa2_name": row["sa2_name"],
                "state": row["state"],
                "similarity": row["similarity"],

                "top_kpis":
                    ", ".join(
                        top_kpi_names
                    ),

                "numeric_rank":
                    numeric_rank,

                "hybrid_rank":
                    hybrid_rank,

                "rank_delta":
                    rank_delta
            }
        )


    return pd.DataFrame(
        explained_rows
    )
  
  
def get_radar_data(
    df,
    reference_index,
    candidate_index
):

    """
    Convert the 16 KPI values into percentile
    scores from 0 to 100 so they can be
    compared on the same radar chart scale.
    """

    percentile_df = (
        df[KPI_COLS]
        .rank(
            pct=True
        )
        * 100
    )


    reference_values = (
        percentile_df.iloc[
            reference_index
        ]
    )


    candidate_values = (
        percentile_df.iloc[
            candidate_index
        ]
    )


    radar_data = []


    for kpi in KPI_COLS:

        radar_data.append(
            {
                "kpi": kpi,

                "label":
                    KPI_LABELS[kpi],

                "reference":
                    float(
                        reference_values[
                            kpi
                        ]
                    ),

                "candidate":
                    float(
                        candidate_values[
                            kpi
                        ]
                    )
            }
        )


    return radar_data

def get_kpi_comparison_table(
    df,
    reference_index,
    candidate_index
):

    """
    Return a KPI-by-KPI comparison
    using percentile ranks from 0 to 100.
    """

    percentile_df = (
        df[KPI_COLS]
        .rank(
            pct=True
        )
        * 100
    )

    reference_values = (
        percentile_df.iloc[
            reference_index
        ]
    )

    candidate_values = (
        percentile_df.iloc[
            candidate_index
        ]
    )

    rows = []

    for kpi in KPI_COLS:

        reference_score = float(
            reference_values[kpi]
        )

        candidate_score = float(
            candidate_values[kpi]
        )

        difference = (
            candidate_score
            - reference_score
        )

        rows.append(
            {
                "KPI":
                    KPI_LABELS[kpi],

                "Reference Percentile":
                    reference_score,

                "Matched Percentile":
                    candidate_score,

                "Difference":
                    difference,

                "Absolute Difference":
                    abs(difference)
            }
        )

    return pd.DataFrame(
        rows
    )