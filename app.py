import streamlit as st

from db.bigquery_client import get_bigquery_client
from engine.features import (
    load_features,
    clean_features,
    standardise_features,
)
from engine.similarity import find_top_n


st.title("Demografy Suburb Lookalike Finder")


@st.cache_resource
def load_data():

    client = get_bigquery_client()

    df = load_features(client)

    df = clean_features(df)

    X, scaler = standardise_features(df)

    return df, X, scaler


df, X, scaler = load_data()


df["display_name"] = (
    df["sa2_name"]
    + " ("
    + df["state"]
    + ")"
)


selected = st.selectbox(
    "Select a suburb",
    df["display_name"]
)


if st.button("Find Similar Suburbs"):

    reference_index = df.index[
        df["display_name"] == selected
    ][0]

    results = find_top_n(
        df,
        X,
        reference_index,
        n=10
    )

    st.dataframe(results)