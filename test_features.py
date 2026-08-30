from db.bigquery_client import get_bigquery_client

from engine.features import (
    KPI_COLS,
    load_features,
    clean_features,
    standardise_features,
)


client = get_bigquery_client()


# -------------------------
# Load data
# -------------------------

df = load_features(client)

print("\nDATA LOADED")
print(df.head())

print("\nShape:")
print(df.shape)


# -------------------------
# Check NULLs before
# -------------------------

print("\nNULLS BEFORE CLEANING")
print(df[KPI_COLS].isnull().sum())


# -------------------------
# Clean data
# -------------------------

df = clean_features(df)


# -------------------------
# Check NULLs after
# -------------------------

print("\nNULLS AFTER CLEANING")
print(df[KPI_COLS].isnull().sum())


# -------------------------
# Standardise
# -------------------------

X, scaler = standardise_features(df)


print("\nNUMERIC MATRIX SHAPE")
print(X.shape)


print("\nFIRST SUBURB")
print(
    df.iloc[0][
        ["sa2_code", "sa2_name", "state"]
    ]
)


print("\nRAW KPI VALUES")
print(
    df.iloc[0][KPI_COLS].values
)


print("\nSTANDARDISED KPI VALUES")
print(
    X[0]
)