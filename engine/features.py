from sklearn.preprocessing import StandardScaler


KPI_COLS = [
    f"kpi_{i}_val"
    for i in range(1, 11)
]


def load_features(client):

    sql = f"""
        SELECT
            sa2_code,
            sa2_name,
            state,
            {", ".join(KPI_COLS)}
        FROM `demografy.prod_tables.a_master_view`
    """

    df = client.query(sql).to_dataframe()

    df = df.reset_index(drop=True)

    return df


def clean_features(df):

    df = df.copy()

    df[KPI_COLS] = df[KPI_COLS].fillna(
        df[KPI_COLS].median()
    )

    return df


def standardise_features(df):

    scaler = StandardScaler()

    X = scaler.fit_transform(
        df[KPI_COLS]
    )

    return X, scaler