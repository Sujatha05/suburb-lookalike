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

from engine.explain import (
    explain_results
)


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\nLoading data..."
)

client = (
    get_bigquery_client()
)

df = load_features(
    client
)

df = clean_features(
    df
)


# ============================================================
# NUMERIC FEATURES
# ============================================================

X_numeric, scaler = (
    standardise_features(
        df
    )
)


# ============================================================
# EMBEDDINGS
# ============================================================

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


# ============================================================
# WEIGHTS
# ============================================================

weights = get_preset(
    "Balanced"
)


X_weighted = (
    apply_feature_weights(
        X_numeric,
        weights
    )
)


# ============================================================
# HYBRID VECTOR
# ============================================================

X_hybrid = (
    fuse_vectors(
        X_weighted,
        X_text,
        alpha=0.5
    )
)


# ============================================================
# REFERENCE SUBURB
# ============================================================

reference_name = (
    "Karabar"
)


reference_index = df.index[
    df["sa2_name"]
    == reference_name
][0]


print(
    "\nReference suburb:"
)

print(
    df.iloc[
        reference_index
    ]["sa2_name"]
)


# ============================================================
# TOP MATCHES
# ============================================================

results = find_top_n(
    df,
    X_hybrid,
    reference_index,
    n=10
)


# ============================================================
# EXPLAIN RESULTS
# ============================================================

explained = (
    explain_results(
        df,
        results,
        X_weighted,
        X_hybrid,
        reference_index,
        weights
    )
)


print(
    "\nEXPLAINED RESULTS\n"
)


print(
    explained[
        [
            "rank",
            "sa2_name",
            "similarity",
            "top_kpis",
            "numeric_rank",
            "hybrid_rank",
            "rank_delta"
        ]
    ].to_string(
        index=False
    )
)