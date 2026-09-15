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

from engine.index import (
    build_faiss_index,
    faiss_find_top_n
)


# ============================================================
# SETTINGS
# ============================================================

ALPHA = 0.2

REFERENCE_SUBURB = (
    "Carlton"
)

N = 10


# ============================================================
# LOAD DATA
# ============================================================

print(
    "Loading data..."
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


# ============================================================
# TEXT EMBEDDINGS
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
# APPLY BALANCED WEIGHTS
# ============================================================

weights = (
    get_preset(
        "Balanced"
    )
)

X_weighted = (
    apply_feature_weights(
        X_numeric,
        weights
    )
)


# ============================================================
# BUILD HYBRID MATRIX
# ============================================================

X_hybrid = (
    fuse_vectors(
        X_weighted,
        X_text,
        alpha=ALPHA
    )
)


print(
    "Hybrid matrix:",
    X_hybrid.shape
)


# ============================================================
# FIND REFERENCE SUBURB
# ============================================================

matches = df.index[
    df["sa2_name"]
    .str.lower()
    ==
    REFERENCE_SUBURB.lower()
].tolist()


if not matches:

    raise ValueError(
        f"{REFERENCE_SUBURB} "
        "not found."
    )


reference_index = (
    matches[0]
)


print(
    "Reference suburb:",
    df.iloc[
        reference_index
    ]["sa2_name"]
)

print(
    "SA2 code:",
    df.iloc[
        reference_index
    ]["sa2_code"]
)


# ============================================================
# NUMPY SEARCH
# ============================================================

numpy_results = (
    find_top_n(
        df,
        X_hybrid,
        reference_index,
        n=N
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "NUMPY RESULTS"
)

print(
    "=" * 70
)

print(
    numpy_results[
        [
            "rank",
            "sa2_code",
            "sa2_name",
            "similarity"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# FAISS SEARCH
# ============================================================

faiss_index, X_faiss = (
    build_faiss_index(
        X_hybrid
    )
)


faiss_results = (
    faiss_find_top_n(
        df,
        faiss_index,
        X_faiss,
        reference_index,
        n=N
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "FAISS RESULTS"
)

print(
    "=" * 70
)

print(
    faiss_results[
        [
            "rank",
            "sa2_code",
            "sa2_name",
            "similarity"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# COMPARE RESULTS
# ============================================================

numpy_codes = (
    numpy_results[
        "sa2_code"
    ]
    .astype(str)
    .tolist()
)


faiss_codes = (
    faiss_results[
        "sa2_code"
    ]
    .astype(str)
    .tolist()
)


print(
    "\n"
    + "=" * 70
)

print(
    "NUMPY VS FAISS CHECK"
)

print(
    "=" * 70
)


if numpy_codes == faiss_codes:

    print(
        "PASS - FAISS and NumPy "
        "returned identical neighbours "
        "in identical order."
    )

else:

    print(
        "FAIL - FAISS and NumPy "
        "returned different results."
    )


    for rank, (
        numpy_code,
        faiss_code
    ) in enumerate(
        zip(
            numpy_codes,
            faiss_codes
        ),
        start=1
    ):

        status = (
            "MATCH"
            if numpy_code == faiss_code
            else "DIFFERENT"
        )

        print(
            f"Rank {rank:2}: "
            f"NumPy={numpy_code} "
            f"FAISS={faiss_code} "
            f"{status}"
        )