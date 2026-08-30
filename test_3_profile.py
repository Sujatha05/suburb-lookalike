from db.bigquery_client import get_bigquery_client

from engine.features import (
    load_features,
    clean_features
)

from engine.profiles import generate_profile


client = get_bigquery_client()

df = load_features(client)

df = clean_features(df)


karabar = df.iloc[0]

profile = generate_profile(karabar)

print(profile)