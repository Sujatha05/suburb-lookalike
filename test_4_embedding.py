from db.bigquery_client import get_bigquery_client

from engine.features import (
    load_features,
    clean_features
)

from engine.profiles import generate_profile
from engine.text_embed import embed_profile


client = get_bigquery_client()

df = load_features(client)

df = clean_features(df)


profile = generate_profile(
    df.iloc[0]
)


print("\nPROFILE")
print(profile)


vector = embed_profile(
    profile
)


print("\nVECTOR LENGTH")
print(len(vector))


print("\nFIRST 10 VALUES")
print(vector[:10])