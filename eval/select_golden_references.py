import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from db.bigquery_client import (
    get_bigquery_client
)

from engine.features import (
    load_features,
    clean_features,
    KPI_COLS
)


# ============================================================
# EXISTING GOLDEN REFERENCES
# ============================================================

EXISTING_REFERENCES = {
    "206041117",   # Carlton
    "206061138",   # Toorak
    "311061331",   # Logan Central
    "115011555",   # Castle Hill - North
    "702011050",   # Petermann - Simpson
}


# ============================================================
# ARCHETYPE DEFINITIONS
# ============================================================
#
# Positive KPIs:
# Higher percentile is desirable for this archetype.
#
# Negative KPIs:
# Lower percentile is desirable for this archetype.
# ============================================================

ARCHETYPES = {

    "Investor / Premium Rental": {

        "positive": {
            "kpi_15_val": 1.0,   # Investment Potential
            "kpi_14_val": 1.0,   # Premium Rental
            "kpi_7_val": 0.6,    # Rental Access
        },

        "negative": {
        }
    },


    "Retiree / Downsizer": {

        "positive": {
            "kpi_12_val": 1.0,   # Retirees and Downsizer
            "kpi_8_val": 0.7,    # Resident Anchor
            "kpi_6_val": 0.5,    # Resident Equity
        },

        "negative": {
            "kpi_10_val": 0.3,   # Young Family
        }
    },


    "Migration / Diversity Hub": {

        "positive": {
            "kpi_2_val": 1.0,    # Diversity
            "kpi_3_val": 1.0,    # Migration Footprint
        },

        "negative": {
        }
    },


    "Stable Owner-Occupier": {

        "positive": {
            "kpi_6_val": 1.0,    # Resident Equity
            "kpi_8_val": 1.0,    # Resident Anchor
            "kpi_16_val": 1.0,   # Generational Stability
        },

        "negative": {
            "kpi_7_val": 0.5,    # Rental Access
        }
    },


    "High-Mobility Rental Area": {

        "positive": {
            "kpi_9_val": 1.0,    # Household Mobility
            "kpi_7_val": 1.0,    # Rental Access
        },

        "negative": {
            "kpi_8_val": 0.7,    # Resident Anchor
            "kpi_16_val": 0.4,   # Generational Stability
        }
    }
}


# ============================================================
# CALCULATE ARCHETYPE SCORE
# ============================================================

def calculate_score(
    percentiles,
    positive,
    negative
):

    score = pd.Series(
        0.0,
        index=percentiles.index
    )

    total_weight = 0.0


    # Higher percentile = better
    for kpi, weight in positive.items():

        score += (
            percentiles[kpi]
            * weight
        )

        total_weight += weight


    # Lower percentile = better
    for kpi, weight in negative.items():

        score += (
            (100 - percentiles[kpi])
            * weight
        )

        total_weight += weight


    if total_weight > 0:

        score = (
            score
            /
            total_weight
        )


    return score


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nLoading suburb data..."
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


    # Convert each KPI to percentile 0-100
    percentiles = (
        df[KPI_COLS]
        .rank(
            pct=True
        )
        * 100
    )


    # Do not recommend one of our
    # existing golden references again
    available_mask = (
        ~df["sa2_code"]
        .astype(str)
        .isin(
            EXISTING_REFERENCES
        )
    )


    for (
        archetype_name,
        definition
    ) in ARCHETYPES.items():

        score = (
            calculate_score(
                percentiles,
                definition[
                    "positive"
                ],
                definition[
                    "negative"
                ]
            )
        )


        candidates = (
            df.loc[
                available_mask,
                [
                    "sa2_code",
                    "sa2_name",
                    "state"
                ]
            ]
            .copy()
        )


        candidates[
            "archetype_score"
        ] = score[
            available_mask
        ]


        candidates = (
            candidates
            .sort_values(
                "archetype_score",
                ascending=False
            )
            .head(15)
        )


        print(
            "\n"
            + "=" * 90
        )

        print(
            archetype_name.upper()
        )

        print(
            "=" * 90
        )


        print(
            candidates.to_string(
                index=False
            )
        )


if __name__ == "__main__":

    main()