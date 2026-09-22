from __future__ import annotations

import os
import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# STANDARD IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from langsmith import traceable


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LANGSMITH
# ============================================================

LANGSMITH_ENABLED = (
    str(
        os.getenv(
            "LANGSMITH_TRACING",
            "false",
        )
    ).lower()
    in {
        "true",
        "1",
        "yes",
    }
)

LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT",
    "demografy-suburb-lookalike",
)

print(
    "\nLangSmith tracing:",
    "ENABLED" if LANGSMITH_ENABLED else "DISABLED",
)

print(
    "LangSmith project:",
    LANGSMITH_PROJECT,
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from db.bigquery_client import (
    get_bigquery_client,
)

from engine.features import (
    load_features,
    clean_features,
    standardise_features,
)

from engine.profiles import (
    generate_all_profiles,
)

from engine.text_embed import (
    get_or_create_embeddings,
)

from engine.weights import (
    get_preset,
    apply_feature_weights,
)

from engine.fusion import (
    fuse_vectors,
)

from engine.similarity import (
    find_top_n,
)

from eval.golden_set import (
    GOLDEN_SET,
)

# ============================================================
# METRICS
#
# Metric calculations live in eval/metrics.py.
# This file only RUNS the metrics.
# ============================================================

from eval.metrics import (
    precision_at_k,
    recall_at_k,
    count_hits,
    count_negative_matches,
    self_consistency_check,
    weight_stability_summary,
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
    1.0,
]

DEFAULT_PRESET = "Balanced"

DEFAULT_ALPHA = 0.2

STABILITY_THRESHOLD = 0.50

DEFAULT_K = 10


# ============================================================
# BUILD CODE -> INDEX MAP
# ============================================================

def build_code_to_index(df):
    """
    Create a mapping from SA2 code to dataframe row index.
    """

    return {
        str(code): index
        for index, code in enumerate(
            df["sa2_code"]
        )
    }


# ============================================================
# EVALUATE ONE GOLDEN-SET REFERENCE
#
# Calculates:
#   - Precision@K
#   - Recall@K
#   - Expected-neighbour hits
#   - Negative/inner-city hits
#
# LangSmith:
# One child trace is created for each reference suburb.
# ============================================================

@traceable(
    name="evaluate_reference",
    run_type="chain",
)
def evaluate_reference(
    df,
    matrix,
    reference_code,
    config,
    configuration_name,
):

    reference_code = str(
        reference_code
    )

    k = config.get(
        "k",
        DEFAULT_K,
    )

    reference_suburb = config[
        "sa2_name"
    ]

    # --------------------------------------------------------
    # Find reference suburb
    # --------------------------------------------------------

    code_to_index = build_code_to_index(df)

    if reference_code not in code_to_index:

        print(
            f"WARNING: {reference_code} "
            "not found in dataset."
        )

        return None

    reference_index = code_to_index[
        reference_code
    ]

    # --------------------------------------------------------
    # Similarity search
    # --------------------------------------------------------

    results = find_top_n(
        df,
        matrix,
        reference_index,
        n=k,
    )

    returned_codes = (
        results["sa2_code"]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # Expected neighbours from golden set
    # --------------------------------------------------------

    expected_codes = [
        str(code)
        for code in config[
            "expected_neighbours"
        ]
    ]

    # --------------------------------------------------------
    # Negative / inner-city suburbs
    # --------------------------------------------------------

    negative_codes = [
        str(code)
        for code in config.get(
            "should_not_match",
            [],
        )
    ]

    # --------------------------------------------------------
    # Required ranking metrics
    # --------------------------------------------------------

    precision = precision_at_k(
        returned_codes,
        expected_codes,
        k,
    )

    recall = recall_at_k(
        returned_codes,
        expected_codes,
        k,
    )

    hits = count_hits(
        returned_codes,
        expected_codes,
        k,
    )

    negative_hits = count_negative_matches(
        returned_codes,
        negative_codes,
        k,
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "configuration": configuration_name,

        "reference_code": reference_code,

        "reference_suburb": reference_suburb,

        "k": k,

        "expected_count": len(
            expected_codes
        ),

        "hits": hits,

        "precision_at_k": float(
            precision
        ),

        "recall_at_k": float(
            recall
        ),

        "negative_hits": negative_hits,

        "negative_sanity_passed": (
            negative_hits == 0
        ),

        # Do not store the complete returned list
        # in the CSV. It is only useful inside the trace.
        "returned_codes": returned_codes[:k],
    }


# ============================================================
# RUN GOLDEN-SET EVALUATION
#
# GOLDEN_SET structure:
#
# {
#     "SA2_CODE": {
#         "sa2_name": "...",
#         "expected_neighbours": [...],
#         "should_not_match": [...]
#     }
# }
#
# Therefore GOLDEN_SET.items() is required.
#
# LangSmith:
# One parent trace for this configuration.
# evaluate_reference() creates child traces.
# ============================================================

@traceable(
    name="evaluate_matrix",
    run_type="chain",
)
def evaluate_matrix(
    df,
    matrix,
    code_to_index,
    configuration_name,
):

    rows = []

    for reference_code, config in GOLDEN_SET.items():

        reference_code = str(
            reference_code
        )

        if reference_code not in code_to_index:

            print(
                f"WARNING: {reference_code} "
                "not found in dataset."
            )

            continue

        result = evaluate_reference(
            df,
            matrix,
            reference_code,
            config,
            configuration_name,
        )

        if result is not None:

            # returned_codes are useful for the trace,
            # but not necessary in the CSV.
            csv_result = {
                key: value
                for key, value in result.items()
                if key != "returned_codes"
            }

            rows.append(csv_result)

    return pd.DataFrame(rows)


# ============================================================
# GET TOP-K CODES
# ============================================================

def get_top_k_codes(
    df,
    matrix,
    reference_index,
    k=DEFAULT_K,
):

    results = find_top_n(
        df,
        matrix,
        reference_index,
        n=k,
    )

    return (
        results["sa2_code"]
        .astype(str)
        .tolist()
    )


# ============================================================
# WEIGHT STABILITY CHECK
#
# Requirement:
# Small weight changes should not completely
# reshuffle the rankings.
#
# Test:
# Increase KPI 1 weight by 10%.
#
# Metric calculation:
# eval/metrics.py
# ============================================================

@traceable(
    name="stability_check",
    run_type="chain",
)
def stability_check(
    df,
    X_numeric,
    X_text,
    code_to_index,
    alpha=DEFAULT_ALPHA,
    k=DEFAULT_K,
    stability_threshold=STABILITY_THRESHOLD,
):

    # --------------------------------------------------------
    # Baseline weights
    # --------------------------------------------------------

    base_weights = get_preset(
        DEFAULT_PRESET
    )

    base_numeric = apply_feature_weights(
        X_numeric,
        base_weights,
    )

    base_hybrid = fuse_vectors(
        base_numeric,
        X_text,
        alpha=alpha,
    )

    # --------------------------------------------------------
    # Small weight change
    #
    # Increase KPI 1 by 10%.
    # --------------------------------------------------------

    changed_weights = (
        base_weights.copy()
    )

    changed_weights[
        "kpi_1_val"
    ] *= 1.10

    changed_numeric = apply_feature_weights(
        X_numeric,
        changed_weights,
    )

    changed_hybrid = fuse_vectors(
        changed_numeric,
        X_text,
        alpha=alpha,
    )

    # --------------------------------------------------------
    # Compare rankings
    # --------------------------------------------------------

    rows = []

    for reference_code, config in GOLDEN_SET.items():

        reference_code = str(
            reference_code
        )

        if reference_code not in code_to_index:
            continue

        reference_index = code_to_index[
            reference_code
        ]

        base_codes = get_top_k_codes(
            df,
            base_hybrid,
            reference_index,
            k=k,
        )

        changed_codes = get_top_k_codes(
            df,
            changed_hybrid,
            reference_index,
            k=k,
        )

        # ----------------------------------------------------
        # Weight stability metric
        # ----------------------------------------------------

        stability = weight_stability_summary(
            base_codes,
            changed_codes,
            k,
            threshold=stability_threshold,
        )

        rows.append(
            {
                "reference_suburb":
                    config["sa2_name"],

                "top_k":
                    k,

                "overlap_count":
                    int(
                        stability[
                            "top_k_overlap"
                        ] * k
                    ),

                "overlap_ratio":
                    float(
                        stability[
                            "top_k_overlap"
                        ]
                    ),

                "stability_threshold":
                    float(
                        stability[
                            "threshold"
                        ]
                    ),

                "stability_passed":
                    bool(
                        stability[
                            "passed"
                        ]
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# ALPHA / BLEND EXPERIMENT
#
# alpha = 0.0 -> Numeric only
# alpha = 1.0 -> Text only
#
# This is an EXPERIMENT, not one of the four
# required Section 7.2 metrics.
#
# It supports the Frontend / Eval requirement:
# "golden set + numeric-vs-hybrid evaluation harness"
# and validates the blend control.
#
# LangSmith:
# One parent trace for the sweep.
# ============================================================

@traceable(
    name="run_alpha_sweep",
    run_type="chain",
)
def run_alpha_sweep(
    df,
    X_numeric,
    X_text,
    code_to_index,
):

    all_results = []

    weights = get_preset(
        DEFAULT_PRESET
    )

    X_weighted = apply_feature_weights(
        X_numeric,
        weights,
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

        X_hybrid = fuse_vectors(
            X_weighted,
            X_text,
            alpha=alpha,
        )

        result_df = evaluate_matrix(
            df,
            X_hybrid,
            code_to_index,
            configuration_name=(
                f"alpha_{alpha:.1f}"
            ),
        )

        if result_df.empty:
            continue

        result_df["alpha"] = alpha

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
                    "negative_hits",
                ]
            ].to_string(
                index=False
            )
        )

        print(
            "\nAverage Precision:",
            round(
                result_df[
                    "precision_at_k"
                ].mean(),
                4,
            )
        )

        print(
            "Average Recall:",
            round(
                result_df[
                    "recall_at_k"
                ].mean(),
                4,
            )
        )

    if not all_results:

        return pd.DataFrame()

    return pd.concat(
        all_results,
        ignore_index=True,
    )


# ============================================================
# MAIN EVALUATION
#
# This is the parent LangSmith trace.
#
# It contains:
#   - Golden-set evaluation
#   - Numeric vs Hybrid vs Text
#   - Alpha experiment
#   - Self-consistency
#   - Stability
#   - Negative sanity
# ============================================================

@traceable(
    name="demografy_evaluation",
    run_type="chain",
)
def main():

    print(
        "\nLoading evaluation data..."
    )

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    client = get_bigquery_client()

    df = load_features(
        client
    )

    df = clean_features(
        df
    )

    # scaler is not needed after standardisation
    X_numeric, _ = standardise_features(
        df
    )

    print(
        "Numeric matrix:",
        X_numeric.shape
    )

    # ========================================================
    # 2. GENERATE PROFILES / TEXT EMBEDDINGS
    # ========================================================

    profiles = generate_all_profiles(
        df
    )

    X_text = get_or_create_embeddings(
        profiles
    )

    print(
        "Text matrix:",
        X_text.shape
    )

    # ========================================================
    # BUILD SA2 CODE INDEX
    # ========================================================

    code_to_index = build_code_to_index(
        df
    )

    # ========================================================
    # 3. BALANCED NUMERIC FEATURES
    # ========================================================

    balanced_weights = get_preset(
        DEFAULT_PRESET
    )

    X_weighted = apply_feature_weights(
        X_numeric,
        balanced_weights,
    )

    # ========================================================
    # 4. NUMERIC-ONLY EVALUATION
    #
    # alpha = 0
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

    numeric_matrix = fuse_vectors(
        X_weighted,
        X_text,
        alpha=0.0,
    )

    numeric_results = evaluate_matrix(
        df,
        numeric_matrix,
        code_to_index,
        "numeric_only",
    )

    if not numeric_results.empty:

        print(
            numeric_results.to_string(
                index=False
            )
        )

    # ========================================================
    # 5. DEFAULT HYBRID EVALUATION
    #
    # This is the main configuration used by the app.
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"DEFAULT HYBRID EVALUATION "
        f"(alpha={DEFAULT_ALPHA})"
    )

    print(
        "=" * 70
    )

    hybrid_matrix = fuse_vectors(
        X_weighted,
        X_text,
        alpha=DEFAULT_ALPHA,
    )

    hybrid_results = evaluate_matrix(
        df,
        hybrid_matrix,
        code_to_index,
        f"hybrid_{DEFAULT_ALPHA}",
    )

    if not hybrid_results.empty:

        print(
            hybrid_results.to_string(
                index=False
            )
        )

    # ========================================================
    # 6. TEXT-ONLY EVALUATION
    #
    # alpha = 1
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

    text_matrix = fuse_vectors(
        X_weighted,
        X_text,
        alpha=1.0,
    )

    text_results = evaluate_matrix(
        df,
        text_matrix,
        code_to_index,
        "text_only",
    )

    if not text_results.empty:

        print(
            text_results.to_string(
                index=False
            )
        )

    # ========================================================
    # 7. NUMERIC VS HYBRID VS TEXT
    #
    # This directly supports the Frontend / Eval
    # requirement for a numeric-vs-hybrid evaluation harness.
    # ========================================================

    comparison_rows = []

    if not numeric_results.empty:

        comparison_rows.append(
            {
                "configuration":
                    "Numeric Only",

                "alpha":
                    0.0,

                "avg_precision":
                    float(
                        numeric_results[
                            "precision_at_k"
                        ].mean()
                    ),

                "avg_recall":
                    float(
                        numeric_results[
                            "recall_at_k"
                        ].mean()
                    ),

                "total_negative_hits":
                    int(
                        numeric_results[
                            "negative_hits"
                        ].sum()
                    ),
            }
        )

    if not hybrid_results.empty:

        comparison_rows.append(
            {
                "configuration":
                    "Hybrid",

                "alpha":
                    DEFAULT_ALPHA,

                "avg_precision":
                    float(
                        hybrid_results[
                            "precision_at_k"
                        ].mean()
                    ),

                "avg_recall":
                    float(
                        hybrid_results[
                            "recall_at_k"
                        ].mean()
                    ),

                "total_negative_hits":
                    int(
                        hybrid_results[
                            "negative_hits"
                        ].sum()
                    ),
            }
        )

    if not text_results.empty:

        comparison_rows.append(
            {
                "configuration":
                    "Text Only",

                "alpha":
                    1.0,

                "avg_precision":
                    float(
                        text_results[
                            "precision_at_k"
                        ].mean()
                    ),

                "avg_recall":
                    float(
                        text_results[
                            "recall_at_k"
                        ].mean()
                    ),

                "total_negative_hits":
                    int(
                        text_results[
                            "negative_hits"
                        ].sum()
                    ),
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
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

    if not comparison.empty:

        print(
            comparison.to_string(
                index=False
            )
        )

    # ========================================================
    # 8. ALPHA / BLEND EXPERIMENT
    # ========================================================

    alpha_results = run_alpha_sweep(
        df,
        X_numeric,
        X_text,
        code_to_index,
    )

    if not alpha_results.empty:

        alpha_summary = (
            alpha_results
            .groupby(
                "alpha",
                as_index=False,
            )
            .agg(
                avg_precision=(
                    "precision_at_k",
                    "mean",
                ),
                avg_recall=(
                    "recall_at_k",
                    "mean",
                ),
                total_negative_hits=(
                    "negative_hits",
                    "sum",
                ),
            )
        )

    else:

        alpha_summary = pd.DataFrame()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ALPHA / BLEND EXPERIMENT"
    )

    print(
        "=" * 70
    )

    if not alpha_summary.empty:

        print(
            alpha_summary.to_string(
                index=False
            )
        )

    # ========================================================
    # 9. INNER-CITY / NEGATIVE SANITY CHECK
    #
    # Requirement:
    # "should NOT return inner-city suburbs"
    #
    # We count all known negative matches from the
    # golden set across the alpha experiment.
    # ========================================================

    if not alpha_results.empty:

        total_negative_hits = int(
            alpha_results[
                "negative_hits"
            ].sum()
        )

    else:

        total_negative_hits = 0

    negative_sanity_passed = (
        total_negative_hits == 0
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INNER-CITY / NEGATIVE SANITY CHECK"
    )

    print(
        "=" * 70
    )

    print(
        "Total negative hits:",
        total_negative_hits,
    )

    print(
        "Status:",
        (
            "PASS"
            if negative_sanity_passed
            else "FAIL"
        ),
    )

    # ========================================================
    # 10. SELF-CONSISTENCY CHECK
    #
    # Requirement:
    # Every suburb is its own nearest neighbour
    # with similarity 1.0 BEFORE self-match is removed.
    #
    # We test the hybrid matrix.
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SELF-CONSISTENCY CHECK"
    )

    print(
        "=" * 70
    )

    self_consistency = self_consistency_check(
        hybrid_matrix
    )

    self_consistency_passed = bool(
        self_consistency[
            "passed"
        ]
    )

    print(
        "Total rows:",
        self_consistency[
            "total_rows"
        ],
    )

    print(
        "Failure count:",
        self_consistency[
            "failure_count"
        ],
    )

    print(
        "Minimum self-similarity:",
        self_consistency[
            "min_self_similarity"
        ],
    )

    print(
        "Average self-similarity:",
        self_consistency[
            "avg_self_similarity"
        ],
    )

    print(
        "Maximum self-similarity:",
        self_consistency[
            "max_self_similarity"
        ],
    )

    print(
        "Status:",
        (
            "PASS"
            if self_consistency_passed
            else "FAIL"
        ),
    )

    if not self_consistency_passed:

        print(
            "\nSelf-consistency failures:"
        )

        print(
            pd.DataFrame(
                self_consistency[
                    "failures"
                ]
            ).to_string(
                index=False
            )
        )

    # ========================================================
    # 11. WEIGHT STABILITY CHECK
    #
    # Requirement:
    # Small weight changes should not completely
    # reshuffle rankings.
    #
    # Test:
    # KPI 1 increased by 10%.
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "WEIGHT STABILITY CHECK"
    )

    print(
        "=" * 70
    )

    stability_results = stability_check(
        df,
        X_numeric,
        X_text,
        code_to_index,
        alpha=DEFAULT_ALPHA,
        k=DEFAULT_K,
        stability_threshold=STABILITY_THRESHOLD,
    )

    if not stability_results.empty:

        print(
            stability_results.to_string(
                index=False
            )
        )

        average_overlap = float(
            stability_results[
                "overlap_ratio"
            ].mean()
        )

        stability_passed = bool(
            average_overlap
            >= STABILITY_THRESHOLD
        )

    else:

        average_overlap = None
        stability_passed = False

    print(
        "\nAverage Top-10 overlap:",
        (
            round(
                average_overlap,
                4,
            )
            if average_overlap is not None
            else "N/A"
        ),
    )

    print(
        "Stability threshold:",
        STABILITY_THRESHOLD,
    )

    print(
        "Status:",
        (
            "PASS"
            if stability_passed
            else "FAIL"
        ),
    )

    # ========================================================
    # 12. SAVE EVALUATION RESULTS
    # ========================================================

    numeric_results.to_csv(
        "eval/numeric_results.csv",
        index=False,
    )

    hybrid_results.to_csv(
        "eval/hybrid_results.csv",
        index=False,
    )

    text_results.to_csv(
        "eval/text_results.csv",
        index=False,
    )

    alpha_results.to_csv(
        "eval/alpha_results.csv",
        index=False,
    )

    alpha_summary.to_csv(
        "eval/alpha_summary.csv",
        index=False,
    )

    stability_results.to_csv(
        "eval/stability_results.csv",
        index=False,
    )

    comparison.to_csv(
        "eval/method_comparison.csv",
        index=False,
    )

    print(
        "\nEvaluation CSV files saved "
        "in the eval folder."
    )

    # ========================================================
    # 13. CALCULATE MAIN SUMMARY VALUES
    # ========================================================

    numeric_avg_precision = (
        float(
            numeric_results[
                "precision_at_k"
            ].mean()
        )
        if not numeric_results.empty
        else None
    )

    numeric_avg_recall = (
        float(
            numeric_results[
                "recall_at_k"
            ].mean()
        )
        if not numeric_results.empty
        else None
    )

    hybrid_avg_precision = (
        float(
            hybrid_results[
                "precision_at_k"
            ].mean()
        )
        if not hybrid_results.empty
        else None
    )

    hybrid_avg_recall = (
        float(
            hybrid_results[
                "recall_at_k"
            ].mean()
        )
        if not hybrid_results.empty
        else None
    )

    text_avg_precision = (
        float(
            text_results[
                "precision_at_k"
            ].mean()
        )
        if not text_results.empty
        else None
    )

    text_avg_recall = (
        float(
            text_results[
                "recall_at_k"
            ].mean()
        )
        if not text_results.empty
        else None
    )

    # ========================================================
    # 14. REQUIRED SECTION 7.2 REPORT
    #
    # This is deliberately focused on the actual
    # requirements.
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RANKING-QUALITY EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Golden-set references : {len(GOLDEN_SET)}"
    )

    print(
        f"K                     : {DEFAULT_K}"
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    print(
        "\n1. Precision@K"
    )

    print(
        f"   Numeric : "
        f"{numeric_avg_precision:.4f}"
        if numeric_avg_precision is not None
        else "   Numeric : N/A"
    )

    print(
        f"   Hybrid  : "
        f"{hybrid_avg_precision:.4f}"
        if hybrid_avg_precision is not None
        else "   Hybrid  : N/A"
    )

    print(
        f"   Text    : "
        f"{text_avg_precision:.4f}"
        if text_avg_precision is not None
        else "   Text    : N/A"
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    print(
        "\n2. Recall@K"
    )

    print(
        f"   Numeric : "
        f"{numeric_avg_recall:.4f}"
        if numeric_avg_recall is not None
        else "   Numeric : N/A"
    )

    print(
        f"   Hybrid  : "
        f"{hybrid_avg_recall:.4f}"
        if hybrid_avg_recall is not None
        else "   Hybrid  : N/A"
    )

    print(
        f"   Text    : "
        f"{text_avg_recall:.4f}"
        if text_avg_recall is not None
        else "   Text    : N/A"
    )

    # --------------------------------------------------------
    # Self-consistency
    # --------------------------------------------------------

    print(
        "\n3. Self-consistency"
    )

    print(
        "   Rows tested          :",
        self_consistency[
            "total_rows"
        ],
    )

    print(
        "   Failures             :",
        self_consistency[
            "failure_count"
        ],
    )

    print(
        "   Minimum similarity   :",
        self_consistency[
            "min_self_similarity"
        ],
    )

    print(
        "   Status               :",
        (
            "PASS"
            if self_consistency_passed
            else "FAIL"
        ),
    )

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    print(
        "\n4. Weight stability"
    )

    print(
        "   Average Top-10 overlap :",
        (
            round(
                average_overlap,
                4,
            )
            if average_overlap is not None
            else "N/A"
        ),
    )

    print(
        "   Threshold              :",
        STABILITY_THRESHOLD,
    )

    print(
        "   Status                 :",
        (
            "PASS"
            if stability_passed
            else "FAIL"
        ),
    )

    # --------------------------------------------------------
    # Negative sanity
    # --------------------------------------------------------

    print(
        "\n5. Inner-city / negative sanity"
    )

    print(
        "   Negative hits :",
        total_negative_hits,
    )

    print(
        "   Status        :",
        (
            "PASS"
            if negative_sanity_passed
            else "FAIL"
        ),
    )

    # --------------------------------------------------------
    # Numeric vs Hybrid
    # --------------------------------------------------------

    print(
        "\n6. Numeric vs Hybrid evaluation"
    )

    print(
        f"   Numeric Precision@10 : "
        f"{numeric_avg_precision:.4f}"
        if numeric_avg_precision is not None
        else "   Numeric Precision@10 : N/A"
    )

    print(
        f"   Numeric Recall@10    : "
        f"{numeric_avg_recall:.4f}"
        if numeric_avg_recall is not None
        else "   Numeric Recall@10    : N/A"
    )

    print(
        f"   Hybrid Precision@10  : "
        f"{hybrid_avg_precision:.4f}"
        if hybrid_avg_precision is not None
        else "   Hybrid Precision@10  : N/A"
    )

    print(
        f"   Hybrid Recall@10     : "
        f"{hybrid_avg_recall:.4f}"
        if hybrid_avg_recall is not None
        else "   Hybrid Recall@10     : N/A"
    )

    # --------------------------------------------------------
    # Alpha experiment
    # --------------------------------------------------------

    print(
        "\n7. Alpha / blend experiment"
    )

    if not alpha_summary.empty:

        print(
            alpha_summary.to_string(
                index=False
            )
        )

    else:

        print(
            "   No alpha results produced."
        )

   
    # ========================================================
    # FINAL LANGSMITH OUTPUT
    #
    # IMPORTANT:
    # Keep this small.
    #
    # LangSmith already contains the child traces and
    # detailed evaluation runs.
    # ========================================================

    final_output = {

        # ----------------------------------------------------
        # Evaluation identification
        # ----------------------------------------------------

        "evaluation":
            "Ranking-quality evaluation",

        "golden_set_size":
            int(len(GOLDEN_SET)),

        "k":
            DEFAULT_K,

        # ----------------------------------------------------
        # Required metric results
        # ----------------------------------------------------

        "precision_at_k": {
            "numeric":
                numeric_avg_precision,

            "hybrid":
                hybrid_avg_precision,

            "text":
                text_avg_precision,
        },

        "recall_at_k": {
            "numeric":
                numeric_avg_recall,

            "hybrid":
                hybrid_avg_recall,

            "text":
                text_avg_recall,
        },

        # ----------------------------------------------------
        # Self-consistency
        # ----------------------------------------------------

        "self_consistency": {
            "total_rows":
                int(
                    self_consistency[
                        "total_rows"
                    ]
                ),

            "failure_count":
                int(
                    self_consistency[
                        "failure_count"
                    ]
                ),

            "min_similarity":
                (
                    float(
                        self_consistency[
                            "min_self_similarity"
                        ]
                    )
                    if self_consistency[
                        "min_self_similarity"
                    ] is not None
                    else None
                ),

            "passed":
                self_consistency_passed,
        },

        # ----------------------------------------------------
        # Weight stability
        # ----------------------------------------------------

        "weight_stability": {
            "average_top_k_overlap":
                (
                    float(
                        average_overlap
                    )
                    if average_overlap is not None
                    else None
                ),

            "threshold":
                STABILITY_THRESHOLD,

            "passed":
                stability_passed,
        },

        # ----------------------------------------------------
        # Negative sanity
        # ----------------------------------------------------

        "negative_sanity": {
            "total_negative_hits":
                total_negative_hits,

            "passed":
                negative_sanity_passed,
        },

        # ----------------------------------------------------
        # Alpha experiment
        # ----------------------------------------------------

        "alpha_values_tested":
            ALPHA_VALUES,

        # ----------------------------------------------------
        # LangSmith
        # ----------------------------------------------------

        "langsmith_enabled":
            LANGSMITH_ENABLED,

        "langsmith_project":
            LANGSMITH_PROJECT,
    }

    # --------------------------------------------------------
    # Add alpha summary to LangSmith if available
    #
    # No combined_score.
    # --------------------------------------------------------

    if not alpha_summary.empty:

        final_output[
            "alpha_summary"
        ] = alpha_summary.to_dict(
            orient="records"
        )

    return final_output


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    output = main()

    print(
            "=" * 70
        )

    
    print(
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Precision@10 — Hybrid:",
        (
            output[
                "precision_at_k"
            ]["hybrid"]
        ),
    )

    print(
        "Recall@10 — Hybrid:",
        (
            output[
                "recall_at_k"
            ]["hybrid"]
        ),
    )

    print(
        "Self-consistency:",
        (
            "PASS"
            if output[
                "self_consistency"
            ]["passed"]
            else "FAIL"
        ),
    )

    print(
        "Weight stability:",
        (
            "PASS"
            if output[
                "weight_stability"
            ]["passed"]
            else "FAIL"
        ),
    )

    print(
        "Negative sanity:",
        (
            "PASS"
            if output[
                "negative_sanity"
            ]["passed"]
            else "FAIL"
        ),
    )

    print(
        "LangSmith:",
        (
            "ENABLED"
            if output[
                "langsmith_enabled"
            ]
            else "DISABLED"
        ),
    )

    print(
        "=" * 70
    )