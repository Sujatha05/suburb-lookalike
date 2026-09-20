from __future__ import annotations

import os
from typing import Any, Callable

import pandas as pd
from dotenv import load_dotenv
from langsmith import traceable


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# LANGSMITH STATUS
# ============================================================

def is_langsmith_enabled() -> bool:
    """
    Return True when LangSmith tracing is enabled.
    """

    tracing_value = (
        os.getenv("LANGSMITH_TRACING", "false")
        .strip()
        .lower()
    )

    api_key = os.getenv("LANGSMITH_API_KEY")

    return (
        tracing_value == "true"
        and bool(api_key)
    )


# ============================================================
# SAFE TRACE INPUTS
# ============================================================

def _safe_lookup_inputs(
    inputs: dict[str, Any]
) -> dict[str, Any]:
    """
    Keep only small lookup configuration values
    in LangSmith.

    Large objects such as DataFrames, numeric matrices,
    embeddings and hybrid matrices are intentionally
    excluded.
    """

    request = inputs.get(
        "request",
        inputs
    )

    if not isinstance(request, dict):
        return {
            "request": str(request)
        }

    weights = request.get(
        "weights",
        {}
    )

    return {

        "reference_sa2_code":
            request.get(
                "reference_sa2_code"
            ),

        "alpha":
            request.get(
                "alpha"
            ),

        "preset":
            request.get(
                "preset"
            ),

        "n_matches":
            request.get(
                "n_matches"
            ),

        "user_id":
            request.get(
                "user_id"
            ),

        "tier":
            request.get(
                "tier"
            ),

        "weights":
            weights,
    }


# ============================================================
# SAFE TRACE OUTPUTS
# ============================================================

def _safe_lookup_outputs(
    outputs: Any
) -> dict[str, Any]:
    """
    Convert lookup output into a small,
    LangSmith-friendly representation.

    Streamlit still receives the full result.
    This function only controls what LangSmith records.
    """

    if not isinstance(outputs, dict):
        return {
            "result": str(outputs)
        }

    safe_output = {

        "reference_sa2_code":
            outputs.get(
                "reference_sa2_code"
            ),

        "reference_suburb":
            outputs.get(
                "reference_suburb"
            ),

        "state":
            outputs.get(
                "state"
            ),

        "alpha":
            outputs.get(
                "alpha"
            ),

        "preset":
            outputs.get(
                "preset"
            ),

        "n_matches":
            outputs.get(
                "n_matches"
            ),

        "latency_ms":
            outputs.get(
                "latency_ms"
            ),
    }


    # --------------------------------------------------------
    # SAFE SEARCH RESULTS
    # --------------------------------------------------------

    results = outputs.get(
        "results"
    )

    if isinstance(results, pd.DataFrame):

        useful_columns = [
            col
            for col in [
                "rank",
                "sa2_code",
                "sa2_name",
                "state",
                "similarity",
            ]
            if col in results.columns
        ]

        safe_results = (
            results[
                useful_columns
            ]
            .head(25)
        )

        # Detailed but small result representation
        safe_output["matches"] = (
            safe_results
            .to_dict(
                orient="records"
            )
        )

        # Explicit neighbour-code list
        if "sa2_code" in safe_results.columns:

            safe_output["neighbour_codes"] = (
                safe_results[
                    "sa2_code"
                ]
                .astype(str)
                .tolist()
            )

        else:

            safe_output["neighbour_codes"] = []

    else:

        safe_output["matches"] = results
        safe_output["neighbour_codes"] = []


    return safe_output


# ============================================================
# LOOKUP TRACE DECORATOR
# ============================================================

def trace_lookup(
    function: Callable
) -> Callable:
    """
    Add LangSmith tracing to a complete
    suburb-lookalike lookup.

    Example:

        @trace_lookup
        def run_lookup(request):
            ...
    """

    return traceable(
        name="suburb-lookalike-search",
        run_type="chain",
        process_inputs=_safe_lookup_inputs,
        process_outputs=_safe_lookup_outputs,
    )(
        function
    )


# ============================================================
# OPTIONAL TRACE CONFIG
# ============================================================

def build_trace_config(
    user_id: str | None = None,
    tier: str | None = None,
    preset: str | None = None,
) -> dict:
    """
    Build optional metadata and tags for LangSmith.

    This is not required for the basic traced
    Python lookup.
    """

    metadata = {}

    if user_id is not None:
        metadata["user_id"] = str(user_id)

    if tier is not None:
        metadata["tier"] = str(tier)

    if preset is not None:
        metadata["preset"] = str(preset)

    tags = [
        "suburb-lookalike",
        "faiss",
        "hybrid-similarity",
    ]

    if tier:
        tags.append(
            f"tier:{tier}"
        )

    return {
        "run_name":
            "suburb-lookalike-lookup",

        "metadata":
            metadata,

        "tags":
            tags,
    }