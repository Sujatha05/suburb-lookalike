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
    l2_normalise,
    fuse_vectors
)


print("\nLoading suburb data...")

client = get_bigquery_client()

df = load_features(
    client
)

df = clean_features(
    df
)


print(
    f"Loaded {len(df)} suburbs."
)


print("\nPreparing numeric features...")

X_numeric, scaler = standardise_features(
    df
)

print(
    "Numeric matrix shape:"
)

print(
    X_numeric.shape
)


print("\nLoading text embeddings...")

profiles = generate_all_profiles(
    df
)

X_text = get_or_create_embeddings(
    profiles
)

print(
    "Text embedding shape:"
)

print(
    X_text.shape
)


print("\nBuilding hybrid vectors...")

X_hybrid = fuse_vectors(
    X_numeric,
    X_text,
    alpha=0.5
)


print(
    "Hybrid matrix shape:"
)

print(
    X_hybrid.shape
)


print(
    "\nFirst numeric vector dimensions:"
)

print(
    len(X_numeric[0])
)


print(
    "\nFirst text vector dimensions:"
)

print(
    len(X_text[0])
)


print(
    "\nFirst hybrid vector dimensions:"
)

print(
    len(X_hybrid[0])
)


numeric_normalised = l2_normalise(
    X_numeric
)

text_normalised = l2_normalise(
    X_text
)


print(
    "\nNumeric vector L2 norm:"
)

print(
    sum(
        numeric_normalised[0] ** 2
    ) ** 0.5
)


print(
    "\nText vector L2 norm:"
)

print(
    sum(
        text_normalised[0] ** 2
    ) ** 0.5
)