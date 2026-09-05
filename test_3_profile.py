from db.bigquery_client import get_bigquery_client
from engine.profiles import generate_all_profiles
from engine.features import load_features, clean_features


from engine.profiles import generate_profile


client = get_bigquery_client()

df = load_features(client)

df = clean_features(df)

profiles = generate_all_profiles(df)

print("NUMBER OF PROFILES")
print(len(profiles))

print("\nFIRST PROFILE")
print(profiles[0])

print("\nSECOND PROFILE")
print(profiles[1])