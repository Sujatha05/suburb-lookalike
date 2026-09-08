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
    find_top_n
)


# ----------------------------------------
# Load data
# ----------------------------------------

print("\nLoading data...")

client = get_bigquery_client()

df = load_features(
    client
)

df = clean_features(
    df
)


# ----------------------------------------
# Numeric features
# ----------------------------------------

X_numeric, scaler = (
    standardise_features(
        df
    )
)


# ----------------------------------------
# Text embeddings
# ----------------------------------------

profiles = generate_all_profiles(
    df
)

X_text = (
    get_or_create_embeddings(
        profiles
    )
)


# ----------------------------------------
# Reference suburb
# ----------------------------------------

reference_name = "Karabar"

reference_index = df.index[
    df["sa2_name"]
    == reference_name
][0]


print(
    "\nREFERENCE:"
)

print(
    df.iloc[
        reference_index
    ]["sa2_name"]
)


# ----------------------------------------
# Test presets
# ----------------------------------------

presets = [
    "Balanced",
    "Family-focused",
    "Investor",
    "Lifestyle"
]


for preset_name in presets:

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"PRESET: {preset_name}"
    )

    print(
        "=" * 60
    )


    weights = get_preset(
        preset_name
    )


    X_weighted = (
        apply_feature_weights(
            X_numeric,
            weights
        )
    )


    X_hybrid = fuse_vectors(
        X_weighted,
        X_text,
        alpha=0.5
    )


    results = find_top_n(
        df,
        X_hybrid,
        reference_index,
        n=5
    )


    print(
        results[
            [
                "rank",
                "sa2_name",
                "state",
                "similarity"
            ]
        ].to_string(
            index=False
        )
    )
    
print(
    "\n\nTESTING ALPHA"
)


weights = get_preset(
    "Balanced"
)

X_weighted = (
    apply_feature_weights(
        X_numeric,
        weights
    )
)


for alpha in [
    0.0,
    0.25,
    0.50,
    0.75,
    1.0
]:

    X_hybrid = fuse_vectors(
        X_weighted,
        X_text,
        alpha=alpha
    )


    results = find_top_n(
        df,
        X_hybrid,
        reference_index,
        n=5
    )


    print(
        "\n"
        + "-" * 50
    )

    print(
        f"ALPHA = {alpha}"
    )

    print(
        "-" * 50
    )


    print(
        results[
            [
                "rank",
                "sa2_name",
                "similarity"
            ]
        ].to_string(
            index=False
        )
    )