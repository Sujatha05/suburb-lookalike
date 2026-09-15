import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.bigquery_client import (
    get_bigquery_client
)

from engine.features import (
    load_features,
    clean_features,
    KPI_COLS
)

from engine.weights import (
    KPI_LABELS
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

percentiles = (
    df[KPI_COLS]
    .rank(
        pct=True
    )
    * 100
)


# ============================================================
# DISPLAY REFERENCE DETAILS
# ============================================================

for suburb_name in REFERENCE_SUBURBS:

    matches = df[
        df["sa2_name"]
        .str.lower()
        ==
        suburb_name.lower()
    ]


    print(
        "\n"
        + "=" * 70
    )

    print(
        suburb_name.upper()
    )

    print(
        "=" * 70
    )


    if matches.empty:

        print(
            "Suburb not found."
        )

        continue


    for index, row in matches.iterrows():

        print(
            "\nSA2 Code:",
            row["sa2_code"]
        )

        print(
            "State:",
            row["state"]
        )


        for kpi in KPI_COLS:

            percentile = (
                percentiles.loc[
                    index,
                    kpi
                ]
            )

            print(
                f"{KPI_LABELS[kpi]:30}",
                f"value={row[kpi]:8.3f}",
                f"percentile={percentile:6.1f}"
            )