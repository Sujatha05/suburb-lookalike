import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.bigquery_client import (
    get_bigquery_client
)

from engine.features import (
    load_features,
    clean_features,
    standardise_features
)

from engine.similarity import (
    find_top_n
)


# ============================================================
# REFERENCE SUBURBS
# ============================================================

REFERENCE_SUBURBS = [
#    "Carlton",
#    "Toorak",
#    "Logan Central",
#    "Castle Hill - North",
#    "Petermann - Simpson"
    "Taylor",
    "Wheelers Hill",
    "St Albans - North",
    "Loftus - Yarrawarrah",
    "Redbank Plains"
]


# ============================================================
# LOAD DATA
# ============================================================

client = get_bigquery_client()

df = load_features(
    client
)

df = clean_features(
    df
)


X_numeric, scaler = (
    standardise_features(
        df
    )
)


# ============================================================
# FIND NUMERIC CANDIDATES
# ============================================================

for suburb in REFERENCE_SUBURBS:

    matches = df.index[
        df["sa2_name"]
        .str.lower()
        ==
        suburb.lower()
    ].tolist()


    if not matches:

        print(
            f"\n{suburb} not found."
        )

        continue


    reference_index = (
        matches[0]
    )


    results = find_top_n(
        df,
        X_numeric,
        reference_index,
        n=20
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"CANDIDATES FOR {suburb.upper()}"
    )

    print(
        "=" * 80
    )


    print(
        results[
            [
                "rank",
                "sa2_code",
                "sa2_name",
                "state",
                "similarity"
            ]
        ].to_string(
            index=False
        )
    )