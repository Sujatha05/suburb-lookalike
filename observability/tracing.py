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

    Expected .env settings:

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=...
    LANGSMITH_PROJECT=suburb-lookalike
    """

    tracing_value = (
        os.getenv(
            "LANGSMITH_TRACING",
            "false"
        )
        .strip()
        .lower()
    )

    api_key = os.getenv(
        "LANGSMITH_API_KEY"
    )

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
    Remove large objects from LangSmith traces.

    We do NOT want to upload:
    - complete pandas DataFrame
    - numeric feature matrix
    - text embedding matrix
    - hybrid matrix

    Only small lookup configuration values should be traced.
    """

    request = inputs.get(
        "request",
        inputs
    )

    if not isinstance(
        request,
        dict
    ):
        return {
            "request": str(request)
        }


    weights = request.get(
        "weights",
        {}
    )


    return {

        "reference_index":
            request.get(
                "reference_index"
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
    Convert the lookup result into a small,
    LangSmith-friendly representation.

    The actual function still returns the full result
    to Streamlit. This only controls what LangSmith sees.
    """

    if not isinstance(
        outputs,
        dict
    ):
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
    }


    # --------------------------------------------------------
    # Store only the small results table in LangSmith.
    # Do NOT store internal matrices.
    # --------------------------------------------------------

    results = outputs.get(
        "results"
    )


    if isinstance(
        results,
        pd.DataFrame
    ):

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


        safe_output[
            "matches"
        ] = (
            results[
                useful_columns
            ]
            .head(25)
            .to_dict(
                orient="records"
            )
        )

    else:

        safe_output[
            "matches"
        ] = results


    return safe_output


# ============================================================
# LOOKUP TRACE DECORATOR
# ============================================================

def trace_lookup(
    function: Callable
) -> Callable:
    """
    Add LangSmith tracing to the complete suburb
    lookalike lookup.

    Usage:

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
# LANGCHAIN / LANGSMITH CONFIG
# ============================================================

def build_trace_config(
    user_id: str | None = None,
    tier: str | None = None,
    preset: str | None = None,
) -> dict:
    """
    Build metadata and tags that can be supplied when
    invoking the LangChain lookup chain.

    Example:

        lookup_chain.invoke(
            request,
            config=build_trace_config(
                user_id="USER123",
                tier="pro",
                preset="Investor"
            )
        )
    """

    metadata = {}


    if user_id is not None:

        metadata[
            "user_id"
        ] = str(user_id)


    if tier is not None:

        metadata[
            "tier"
        ] = str(tier)


    if preset is not None:

        metadata[
            "preset"
        ] = str(preset)


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