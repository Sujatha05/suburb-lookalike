import numpy as np

from engine.features import KPI_COLS

KPI_LABELS = {

    "kpi_1_val":
        "Prosperity",

    "kpi_2_val":
        "Diversity",

    "kpi_3_val":
        "Migration Footprint",

    "kpi_4_val":
        "Learning Level",

    "kpi_5_val":
        "Social Housing",

    "kpi_6_val":
        "Resident Equity",

    "kpi_7_val":
        "Rental Access",

    "kpi_8_val":
        "Resident Anchor",

    "kpi_9_val":
        "Household Mobility",

    "kpi_10_val":
        "Young Family",

    "kpi_11_val":
        "Disadvantage Concentration",

    "kpi_12_val":
        "Retirees and Downsizer",

    "kpi_13_val":
        "Housing Density Mix",

    "kpi_14_val":
        "Premium Rental",

    "kpi_15_val":
        "Investment Potential",

    "kpi_16_val":
        "Generational Stability"
}


DEFAULT_WEIGHTS = {
    column: 1.0
    for column in KPI_COLS
}


WEIGHT_PRESETS = {

    "Balanced": {

        "kpi_1_val": 1.0,
        "kpi_2_val": 1.0,
        "kpi_3_val": 1.0,
        "kpi_4_val": 1.0,
        "kpi_5_val": 1.0,
        "kpi_6_val": 1.0,
        "kpi_7_val": 1.0,
        "kpi_8_val": 1.0,
        "kpi_9_val": 1.0,
        "kpi_10_val": 1.0,
        "kpi_11_val": 1.0,
        "kpi_12_val": 1.0,
        "kpi_13_val": 1.0,
        "kpi_14_val": 1.0,
        "kpi_15_val": 1.0,
        "kpi_16_val": 1.0,
    },


    "Family-focused": {

        "kpi_1_val": 1.2,
        "kpi_2_val": 1.0,
        "kpi_3_val": 0.8,
        "kpi_4_val": 1.5,
        "kpi_5_val": 0.7,
        "kpi_6_val": 1.2,
        "kpi_7_val": 1.0,
        "kpi_8_val": 1.5,
        "kpi_9_val": 0.8,
        "kpi_10_val": 2.0,
        "kpi_11_val": 0.7,
        "kpi_12_val": 0.8,
        "kpi_13_val": 1.2,
        "kpi_14_val": 0.8,
        "kpi_15_val": 1.0,
        "kpi_16_val": 1.5,
    },


    "Investor": {

        "kpi_1_val": 1.5,
        "kpi_2_val": 0.8,
        "kpi_3_val": 1.0,
        "kpi_4_val": 0.8,
        "kpi_5_val": 0.5,
        "kpi_6_val": 1.5,
        "kpi_7_val": 2.0,
        "kpi_8_val": 1.0,
        "kpi_9_val": 1.5,
        "kpi_10_val": 0.7,
        "kpi_11_val": 0.7,
        "kpi_12_val": 1.0,
        "kpi_13_val": 1.5,
        "kpi_14_val": 1.8,
        "kpi_15_val": 2.0,
        "kpi_16_val": 1.2,
    },


    "Lifestyle": {

        "kpi_1_val": 1.3,
        "kpi_2_val": 1.5,
        "kpi_3_val": 1.0,
        "kpi_4_val": 1.2,
        "kpi_5_val": 0.7,
        "kpi_6_val": 1.0,
        "kpi_7_val": 1.0,
        "kpi_8_val": 1.3,
        "kpi_9_val": 1.0,
        "kpi_10_val": 1.0,
        "kpi_11_val": 0.7,
        "kpi_12_val": 1.2,
        "kpi_13_val": 1.3,
        "kpi_14_val": 1.3,
        "kpi_15_val": 1.0,
        "kpi_16_val": 1.4,
    }
}


def get_preset(name):

    if name not in WEIGHT_PRESETS:

        raise ValueError(
            f"Unknown preset: {name}"
        )

    return WEIGHT_PRESETS[
        name
    ].copy()


def weights_to_array(weights):

    return np.array(
        [
            weights[col]
            for col in KPI_COLS
        ],
        dtype=np.float32
    )


def apply_feature_weights(
    X_numeric,
    weights
):

    weight_array = (
        weights_to_array(
            weights
        )
    )

    if np.any(
        weight_array < 0
    ):

        raise ValueError(
            "Feature weights cannot "
            "be negative."
        )

    if X_numeric.shape[1] != len(
        weight_array
    ):

        raise ValueError(
            "Numeric matrix contains "
            f"{X_numeric.shape[1]} features "
            f"but {len(weight_array)} "
            "weights were supplied."
        )

    weighted_matrix = (
        X_numeric
        * weight_array
    )

    return weighted_matrix