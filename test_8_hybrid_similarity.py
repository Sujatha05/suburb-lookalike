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

from engine.fusion import (
    fuse_vectors
)

from engine.similarity import (
    find_top_n
)


# ------------------------------------------------
# 1. Load suburb data
# ------------------------------------------------

print("\nLoading suburb data...")

client = get_bigquery_client()

df = load_features(
    client
)

df = clean_features(
    df
)


# ------------------------------------------------
# 2. Build numeric vectors
# ------------------------------------------------

print(
    "\nPreparing numeric vectors..."
)

X_numeric, scaler = (
    standardise_features(
        df
    )
)


# ------------------------------------------------
# 3. Load cached Gemini embeddings
# ------------------------------------------------

print(
    "\nLoading text embeddings..."
)

profiles = generate_all_profiles(
    df
)

X_text = (
    get_or_create_embeddings(
        profiles
    )
)


# ------------------------------------------------
# 4. Create hybrid vectors
# ------------------------------------------------

ALPHA = 0.5

print(
    f"\nBuilding hybrid vectors "
    f"with alpha={ALPHA}..."
)

X_hybrid = fuse_vectors(
    X_numeric,
    X_text,
    alpha=ALPHA
)


# ------------------------------------------------
# 5. Choose reference suburb
# ------------------------------------------------

reference_name = "Karabar"

matches = df.index[
    df["sa2_name"]
    == reference_name
].tolist()


if not matches:

    raise ValueError(
        f"{reference_name} "
        f"not found"
    )


reference_index = (
    matches[0]
)


reference = df.iloc[
    reference_index
]


print(
    "\nREFERENCE SUBURB"
)

print(
    f"{reference['sa2_name']} "
    f"({reference['state']})"
)


# ------------------------------------------------
# 6. Numeric-only results
# ------------------------------------------------

numeric_results = find_top_n(
    df,
    X_numeric,
    reference_index,
    n=10
)


print(
    "\nNUMERIC-ONLY RESULTS"
)

print(
    numeric_results[
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


# ------------------------------------------------
# 7. Hybrid results
# ------------------------------------------------

hybrid_results = find_top_n(
    df,
    X_hybrid,
    reference_index,
    n=10
)


print(
    "\nHYBRID RESULTS"
)

print(
    hybrid_results[
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

from engine.similarity import (
    compare_rankings
)


comparison = compare_rankings(
    numeric_results,
    hybrid_results
)


print(
    "\nNUMERIC VS HYBRID "
    "RANK CHANGE"
)

print(
    comparison[
        [
            "sa2_name",
            "numeric_rank",
            "hybrid_rank",
            "rank_delta"
        ]
    ].to_string(
        index=False
    )
)