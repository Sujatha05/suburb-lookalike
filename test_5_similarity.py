from db.bigquery_client import get_bigquery_client

from engine.features import (
    load_features,
    clean_features,
    standardise_features,
)

from engine.similarity import find_top_n


# -----------------------------------
# Load and prepare data
# -----------------------------------

client = get_bigquery_client()

df = load_features(client)

df = clean_features(df)

X, scaler = standardise_features(df)


# -----------------------------------
# Select a suburb
# -----------------------------------

suburb_name = "Karabar"


match = df[
    df["sa2_name"] == suburb_name
]


if match.empty:

    print(
        f"Suburb '{suburb_name}' not found."
    )

else:

    reference_index = match.index[0]

    print("\nREFERENCE SUBURB")
    print(
        df.iloc[reference_index][
            ["sa2_code", "sa2_name", "state"]
        ]
    )


    # -----------------------------------
    # Find similar suburbs
    # -----------------------------------

    results = find_top_n(
        df=df,
        X=X,
        reference_index=reference_index,
        n=10
    )


    print("\nTOP 10 SIMILAR SUBURBS")

    print(
        results.to_string(
            index=False
        )
    )