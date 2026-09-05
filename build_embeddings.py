from db.bigquery_client import get_bigquery_client

from engine.features import (
    load_features,
    clean_features
)

from engine.profiles import (
    generate_all_profiles
)

from engine.text_embed import (
    get_or_create_embeddings
)


print("\nLoading suburb data...")


client = get_bigquery_client()

df = load_features(client)

df = clean_features(df)


print(
    f"Loaded {len(df)} suburbs."
)


print("\nGenerating text profiles...")


profiles = generate_all_profiles(
    df
)


print(
    f"Generated {len(profiles)} profiles."
)


print("\nGenerating/loading embeddings...")


embeddings = get_or_create_embeddings(
    profiles
)


print("\nEMBEDDING BUILD COMPLETE")


print(
    "Embedding matrix shape:"
)

print(
    embeddings.shape
)


print(
    "\nFirst suburb:"
)

print(
    df.iloc[0]["sa2_name"]
)


print(
    "\nFirst 10 embedding values:"
)

print(
    embeddings[0][:10]
)