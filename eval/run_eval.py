import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from db.bigquery_client import (
    get_bigquery_client
)

from engine.features import (
    load_features,
    clean_features,
    standardise_features
)

from engine.profiles import (
    generate_all_profiles
)

from engine.text_embed import (
    get_or_create_embeddings
)

from engine.weights import (
    get_preset,
    apply_feature_weights
)

from engine.fusion import (
    fuse_vectors
)

from engine.similarity import (
    cosine_similarity,
    find_top_n
)

from eval.golden_set import (
    GOLDEN_SET
)


# ============================================================
# SETTINGS
# ============================================================

ALPHA_VALUES = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
]


DEFAULT_PRESET = (
    "Balanced"
)


# ============================================================
# PRECISION @ K
# ============================================================

def precision_at_k(
    returned_codes,
    expected_codes,
    k
):

    returned_top_k = (
        returned_codes[:k]
    )


    expected_set = set(
        expected_codes
    )


    hits = sum(
        1
        for code
        in returned_top_k
        if code in expected_set
    )


    return hits / k


# ============================================================
# RECALL @ K
# ============================================================

def recall_at_k(
    returned_codes,
    expected_codes,
    k
):

    if not expected_codes:
        return 0.0


    returned_top_k = set(
        returned_codes[:k]
    )


    expected_set = set(
        expected_codes
    )


    hits = len(
        returned_top_k
        .intersection(
            expected_set
        )
    )


    return (
        hits
        /
        len(expected_set)
    )


# ============================================================
# COUNT EXPECTED MATCHES
# ============================================================

def count_hits(
    returned_codes,
    expected_codes,
    k
):

    returned_top_k = set(
        returned_codes[:k]
    )


    expected_set = set(
        expected_codes
    )


    return len(
        returned_top_k
        .intersection(
            expected_set
        )
    )


# ============================================================
# NEGATIVE MATCH CHECK
# ============================================================

def count_negative_matches(
    returned_codes,
    should_not_match,
    k
):

    if not should_not_match:
        return 0


    returned_top_k = set(
        returned_codes[:k]
    )


    negative_set = set(
        should_not_match
    )


    return len(
        returned_top_k
        .intersection(
            negative_set
        )
    )


# ============================================================
# BUILD CODE -> INDEX MAP
# ============================================================

def build_code_to_index(
    df
):

    return {
        str(code): index
        for index, code
        in enumerate(
            df["sa2_code"]
        )
    }


# ============================================================
# RUN ONE GOLDEN-SET EVALUATION
# ============================================================

def evaluate_matrix(
    df,
    matrix,
    code_to_index,
    configuration_name
):

    rows = []


    for (
        reference_code,
        config
    ) in GOLDEN_SET.items():

        reference_code = str(
            reference_code
        )


        if (
            reference_code
            not in code_to_index
        ):

            print(
                f"WARNING: "
                f"{reference_code} "
                "not found in dataset."
            )

            continue


        reference_index = (
            code_to_index[
                reference_code
            ]
        )


        k = config.get(
            "k",
            10
        )


        results = find_top_n(
            df,
            matrix,
            reference_index,
            n=k
        )


        returned_codes = (
            results[
                "sa2_code"
            ]
            .astype(str)
            .tolist()
        )


        expected_codes = [
            str(code)
            for code
            in config[
                "expected_neighbours"
            ]
        ]


        negative_codes = [
            str(code)
            for code
            in config.get(
                "should_not_match",
                []
            )
        ]


        precision = (
            precision_at_k(
                returned_codes,
                expected_codes,
                k
            )
        )


        recall = (
            recall_at_k(
                returned_codes,
                expected_codes,
                k
            )
        )


        hits = (
            count_hits(
                returned_codes,
                expected_codes,
                k
            )
        )


        negative_hits = (
            count_negative_matches(
                returned_codes,
                negative_codes,
                k
            )
        )


        rows.append(
            {
                "configuration":
                    configuration_name,

                "reference_code":
                    reference_code,

                "reference_suburb":
                    config[
                        "sa2_name"
                    ],

                "k":
                    k,

                "expected_count":
                    len(
                        expected_codes
                    ),

                "hits":
                    hits,

                "precision_at_k":
                    precision,

                "recall_at_k":
                    recall,

                "negative_hits":
                    negative_hits
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# SELF CONSISTENCY CHECK
# ============================================================

def self_consistency_check(
    matrix
):

    failures = []


    for i in range(
        len(matrix)
    ):

        query = matrix[
            i
        ]


        scores = (
            cosine_similarity(
                query,
                matrix
            )
        )


        nearest_index = int(
            np.argmax(
                scores
            )
        )


        nearest_score = float(
            scores[
                nearest_index
            ]
        )


        if (
            nearest_index != i
            or
            not np.isclose(
                nearest_score,
                1.0,
                atol=1e-5
            )
        ):

            failures.append(
                {
                    "reference_index":
                        i,

                    "nearest_index":
                        nearest_index,

                    "similarity":
                        nearest_score
                }
            )


    return failures


# ============================================================
# TOP-K INDEX HELPER
# ============================================================

def get_top_k_codes(
    df,
    matrix,
    reference_index,
    k=10
):

    results = find_top_n(
        df,
        matrix,
        reference_index,
        n=k
    )


    return (
        results[
            "sa2_code"
        ]
        .astype(str)
        .tolist()
    )


# ============================================================
# STABILITY CHECK
# ============================================================

def stability_check(
    df,
    X_numeric,
    X_text,
    code_to_index,
    alpha=0.5,
    k=10
):

    base_weights = (
        get_preset(
            "Balanced"
        )
    )


    base_numeric = (
        apply_feature_weights(
            X_numeric,
            base_weights
        )
    )


    base_hybrid = (
        fuse_vectors(
            base_numeric,
            X_text,
            alpha=alpha
        )
    )


    # Small weight change:
    # increase Prosperity by 10%
    changed_weights = (
        base_weights.copy()
    )


    changed_weights[
        "kpi_1_val"
    ] *= 1.10


    changed_numeric = (
        apply_feature_weights(
            X_numeric,
            changed_weights
        )
    )


    changed_hybrid = (
        fuse_vectors(
            changed_numeric,
            X_text,
            alpha=alpha
        )
    )


    rows = []


    for (
        reference_code,
        config
    ) in GOLDEN_SET.items():

        reference_code = str(
            reference_code
        )


        if (
            reference_code
            not in code_to_index
        ):

            continue


        reference_index = (
            code_to_index[
                reference_code
            ]
        )


        base_codes = (
            get_top_k_codes(
                df,
                base_hybrid,
                reference_index,
                k=k
            )
        )


        changed_codes = (
            get_top_k_codes(
                df,
                changed_hybrid,
                reference_index,
                k=k
            )
        )


        overlap = len(
            set(base_codes)
            .intersection(
                set(changed_codes)
            )
        )


        overlap_ratio = (
            overlap / k
        )


        rows.append(
            {
                "reference_suburb":
                    config[
                        "sa2_name"
                    ],

                "top_k":
                    k,

                "overlap_count":
                    overlap,

                "overlap_ratio":
                    overlap_ratio
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# ALPHA SWEEP
# ============================================================

def run_alpha_sweep(
    df,
    X_numeric,
    X_text,
    code_to_index
):

    all_results = []


    weights = (
        get_preset(
            DEFAULT_PRESET
        )
    )


    X_weighted = (
        apply_feature_weights(
            X_numeric,
            weights
        )
    )


    for alpha in ALPHA_VALUES:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"ALPHA = {alpha:.1f}"
        )

        print(
            "=" * 70
        )


        X_hybrid = (
            fuse_vectors(
                X_weighted,
                X_text,
                alpha=alpha
            )
        )


        result_df = (
            evaluate_matrix(
                df,
                X_hybrid,
                code_to_index,
                configuration_name=(
                    f"alpha_{alpha:.1f}"
                )
            )
        )


        result_df[
            "alpha"
        ] = alpha


        all_results.append(
            result_df
        )


        print(
            result_df[
                [
                    "reference_suburb",
                    "hits",
                    "precision_at_k",
                    "recall_at_k",
                    "negative_hits"
                ]
            ].to_string(
                index=False
            )
        )


        if not result_df.empty:

            print(
                "\nAverage Precision:",
                round(
                    result_df[
                        "precision_at_k"
                    ].mean(),
                    4
                )
            )

            print(
                "Average Recall:",
                round(
                    result_df[
                        "recall_at_k"
                    ].mean(),
                    4
                )
            )


    return pd.concat(
        all_results,
        ignore_index=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nLoading evaluation data..."
    )


    client = (
        get_bigquery_client()
    )


    df = (
        load_features(
            client
        )
    )


    df = (
        clean_features(
            df
        )
    )


    X_numeric, scaler = (
        standardise_features(
            df
        )
    )


    print(
        "Numeric matrix:",
        X_numeric.shape
    )


    profiles = (
        generate_all_profiles(
            df
        )
    )


    X_text = (
        get_or_create_embeddings(
            profiles
        )
    )


    print(
        "Text matrix:",
        X_text.shape
    )


    code_to_index = (
        build_code_to_index(
            df
        )
    )


    # ========================================================
    # BALANCED NUMERIC FEATURES
    # ========================================================

    balanced_weights = (
        get_preset(
            DEFAULT_PRESET
        )
    )


    X_weighted = (
        apply_feature_weights(
            X_numeric,
            balanced_weights
        )
    )


    # ========================================================
    # 1. NUMERIC-ONLY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "NUMERIC-ONLY EVALUATION"
    )

    print(
        "=" * 70
    )


    numeric_matrix = (
        fuse_vectors(
            X_weighted,
            X_text,
            alpha=0.0
        )
    )


    numeric_results = (
        evaluate_matrix(
            df,
            numeric_matrix,
            code_to_index,
            "numeric_only"
        )
    )


    print(
        numeric_results.to_string(
            index=False
        )
    )


    # ========================================================
    # 2. DEFAULT HYBRID
    # ========================================================

    DEFAULT_ALPHA = 0.2


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"DEFAULT HYBRID "
        f"(alpha={DEFAULT_ALPHA})"
    )

    print(
        "=" * 70
    )


    hybrid_matrix = (
        fuse_vectors(
            X_weighted,
            X_text,
            alpha=DEFAULT_ALPHA
        )
    )


    hybrid_results = (
        evaluate_matrix(
            df,
            hybrid_matrix,
            code_to_index,
            (
                f"hybrid_{DEFAULT_ALPHA}"
            )
        )
    )


    print(
        hybrid_results.to_string(
            index=False
        )
    )


    # ========================================================
    # 3. TEXT-ONLY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEXT-ONLY EVALUATION"
    )

    print(
        "=" * 70
    )


    text_matrix = (
        fuse_vectors(
            X_weighted,
            X_text,
            alpha=1.0
        )
    )


    text_results = (
        evaluate_matrix(
            df,
            text_matrix,
            code_to_index,
            "text_only"
        )
    )


    print(
        text_results.to_string(
            index=False
        )
    )


    # ========================================================
    # COMPARISON SUMMARY
    # ========================================================

    comparison = pd.DataFrame(
        [
            {
                "configuration":
                    "Numeric Only",

                "alpha":
                    0.0,

                "avg_precision":
                    numeric_results[
                        "precision_at_k"
                    ].mean(),

                "avg_recall":
                    numeric_results[
                        "recall_at_k"
                    ].mean()
            },

            {
                "configuration":
                    "Hybrid",

                "alpha":
                    DEFAULT_ALPHA,

                "avg_precision":
                    hybrid_results[
                        "precision_at_k"
                    ].mean(),

                "avg_recall":
                    hybrid_results[
                        "recall_at_k"
                    ].mean()
            },

            {
                "configuration":
                    "Text Only",

                "alpha":
                    1.0,

                "avg_precision":
                    text_results[
                        "precision_at_k"
                    ].mean(),

                "avg_recall":
                    text_results[
                        "recall_at_k"
                    ].mean()
            }
        ]
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "NUMERIC VS HYBRID VS TEXT"
    )

    print(
        "=" * 70
    )


    print(
        comparison.to_string(
            index=False
        )
    )


    # ========================================================
    # 4. ALPHA SWEEP
    # ========================================================

    alpha_results = (
        run_alpha_sweep(
            df,
            X_numeric,
            X_text,
            code_to_index
        )
    )


    alpha_summary = (
        alpha_results
        .groupby(
            "alpha",
            as_index=False
        )
        .agg(
            avg_precision=(
                "precision_at_k",
                "mean"
            ),

            avg_recall=(
                "recall_at_k",
                "mean"
            ),

            total_negative_hits=(
                "negative_hits",
                "sum"
            )
        )
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "ALPHA SWEEP SUMMARY"
    )

    print(
        "=" * 70
    )


    print(
        alpha_summary.to_string(
            index=False
        )
    )


    # ========================================================
    # BEST ALPHA
    # ========================================================

    alpha_summary[
        "combined_score"
    ] = (

        alpha_summary[
            "avg_precision"
        ]

        +

        alpha_summary[
            "avg_recall"
        ]

    ) / 2


    best_row = (
        alpha_summary
        .sort_values(
            [
                "combined_score",
                "total_negative_hits"
            ],
            ascending=[
                False,
                True
            ]
        )
        .iloc[0]
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "BEST ALPHA"
    )

    print(
        "=" * 70
    )


    print(
        "Best alpha:",
        best_row[
            "alpha"
        ]
    )


    print(
        "Average precision:",
        round(
            best_row[
                "avg_precision"
            ],
            4
        )
    )


    print(
        "Average recall:",
        round(
            best_row[
                "avg_recall"
            ],
            4
        )
    )


    print(
        "Combined score:",
        round(
            best_row[
                "combined_score"
            ],
            4
        )
    )


    # ========================================================
    # 5. SELF CONSISTENCY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SELF CONSISTENCY CHECK"
    )

    print(
        "=" * 70
    )


    failures = (
        self_consistency_check(
            hybrid_matrix
        )
    )


    if not failures:

        print(
            "PASS - every suburb is its "
            "own nearest neighbour."
        )

    else:

        print(
            "FAIL - self consistency "
            "problems found:"
        )

        print(
            pd.DataFrame(
                failures
            ).to_string(
                index=False
            )
        )


    # ========================================================
    # 6. STABILITY CHECK
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STABILITY CHECK"
    )

    print(
        "=" * 70
    )


    stability_results = (
        stability_check(
            df,
            X_numeric,
            X_text,
            code_to_index,
            alpha=DEFAULT_ALPHA,
            k=10
        )
    )


    print(
        stability_results.to_string(
            index=False
        )
    )


    if not stability_results.empty:

        print(
            "\nAverage Top-10 overlap:",
            round(
                stability_results[
                    "overlap_ratio"
                ].mean(),
                4
            )
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    numeric_results.to_csv(
        "eval/numeric_results.csv",
        index=False
    )


    hybrid_results.to_csv(
        "eval/hybrid_results.csv",
        index=False
    )


    text_results.to_csv(
        "eval/text_results.csv",
        index=False
    )


    alpha_results.to_csv(
        "eval/alpha_results.csv",
        index=False
    )


    alpha_summary.to_csv(
        "eval/alpha_summary.csv",
        index=False
    )


    stability_results.to_csv(
        "eval/stability_results.csv",
        index=False
    )


    print(
        "\nEvaluation CSV files saved "
        "in the eval folder."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()