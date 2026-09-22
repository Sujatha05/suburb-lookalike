
from __future__ import annotations

import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from db.bigquery_client import get_bigquery_client
from engine.features import (
    load_features,
    clean_features,
    standardise_features,
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_K = 2
MAX_K = 10

RANDOM_STATE = 42
N_INIT = 10

TOP_N = 10


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

def cluster_suburbs(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """
    Assign each suburb to a K-Means cluster.
    """

    if X is None or len(X) == 0:
        raise ValueError("Feature matrix is empty.")

    if n_clusters < 2:
        raise ValueError(
            "n_clusters must be at least 2."
        )

    if n_clusters >= len(X):
        raise ValueError(
            "n_clusters must be smaller than "
            "the number of suburbs."
        )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=N_INIT,
    )

    labels = model.fit_predict(X)

    return labels


# ============================================================
# EVALUATE K VALUES
# ============================================================

def evaluate_k_values(
    X: np.ndarray,
    min_k: int = MIN_K,
    max_k: int = MAX_K,
) -> pd.DataFrame:
    """
    Evaluate K values using:

    - K-Means inertia
    - Silhouette score
    """

    if X is None or len(X) == 0:
        raise ValueError(
            "Feature matrix is empty."
        )

    max_allowed_k = min(
        max_k,
        len(X) - 1,
    )

    results = []

    print("\nEvaluating K values...")
    print("-" * 70)

    for k in range(
        min_k,
        max_allowed_k + 1,
    ):

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
        )

        labels = model.fit_predict(X)

        inertia = model.inertia_

        silhouette = silhouette_score(
            X,
            labels,
        )

        results.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette_score": silhouette,
            }
        )

        print(
            f"K = {k:2d} | "
            f"Inertia = {inertia:,.2f} | "
            f"Silhouette = {silhouette:.4f}"
        )

    return pd.DataFrame(results)


# ============================================================
# SELECT BEST K
# ============================================================

def select_best_k(
    results: pd.DataFrame,
) -> int:
    """
    Select K with the highest silhouette score.
    """

    if results.empty:
        raise ValueError(
            "No K evaluation results available."
        )

    best_row = results.loc[
        results["silhouette_score"].idxmax()
    ]

    return int(best_row["k"])


# ============================================================
# ADD CLUSTER LABELS
# ============================================================

def add_cluster_labels(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Add K-Means cluster labels to the DataFrame.
    """

    result = df.copy()

    if len(result) != len(labels):
        raise ValueError(
            "Number of labels does not match "
            "number of suburbs."
        )

    result["cluster"] = labels

    return result


# ============================================================
# FIND SUBURB
# ============================================================

def find_suburb(
    df: pd.DataFrame,
    suburb_name: str,
) -> pd.DataFrame:
    """
    Find an exact suburb name,
    ignoring case and surrounding spaces.
    """

    return df[
        df["sa2_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        == suburb_name.strip().lower()
    ]


# ============================================================
# TOP 10 SIMILAR SUBURBS
# ============================================================

def find_top_similar_suburbs(
    df: pd.DataFrame,
    X: np.ndarray,
    suburb_index: int,
    cluster_id: int,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """
    Find the Top N numerically similar suburbs
    within the selected suburb's K-Means cluster.

    Similarity is calculated using cosine similarity
    on the standardised KPI feature vectors.
    """

    # --------------------------------------------------------
    # Get all rows belonging to the selected cluster
    # --------------------------------------------------------

    cluster_indices = np.where(
        df["cluster"].to_numpy() == cluster_id
    )[0]

    if len(cluster_indices) == 0:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Reference suburb vector
    # --------------------------------------------------------

    reference_vector = X[
        suburb_index
    ].reshape(1, -1)

    # --------------------------------------------------------
    # Vectors for suburbs in same cluster
    # --------------------------------------------------------

    cluster_vectors = X[
        cluster_indices
    ]

    # --------------------------------------------------------
    # Calculate cosine similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        reference_vector,
        cluster_vectors,
    )[0]

    # --------------------------------------------------------
    # Build result DataFrame
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "index": cluster_indices,
            "similarity": similarities,
        }
    )

    # --------------------------------------------------------
    # Remove the reference suburb itself
    # --------------------------------------------------------

    results = results[
        results["index"] != suburb_index
    ]

    # --------------------------------------------------------
    # Sort from most similar to least similar
    # --------------------------------------------------------

    results = results.sort_values(
        "similarity",
        ascending=False,
    )

    # --------------------------------------------------------
    # Top N
    # --------------------------------------------------------

    results = results.head(top_n)

    # --------------------------------------------------------
    # Add suburb names
    # --------------------------------------------------------

    results["suburb"] = (
        df.iloc[
            results["index"]
        ]["sa2_name"]
        .values
    )

    # --------------------------------------------------------
    # Return clean output
    # --------------------------------------------------------

    return results[
        [
            "suburb",
            "similarity",
        ]
    ].reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("DEMOGRAFY SUBURB CLUSTERING + TOP 10 SIMILAR SUBURBS")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading suburb data from BigQuery...")

    client = get_bigquery_client()

    df = load_features(client)

    print(
        f"Loaded {len(df):,} suburbs."
    )

    # ========================================================
    # CLEAN FEATURES
    # ========================================================

    print("\nCleaning KPI data...")

    df_clean = clean_features(df)

    # ========================================================
    # STANDARDISE FEATURES
    # ========================================================

    print("\nStandardising KPI features...")

    X_numeric, _scaler = standardise_features(
        df_clean
    )

    print(
        f"Feature matrix: "
        f"{X_numeric.shape[0]:,} suburbs x "
        f"{X_numeric.shape[1]} KPIs"
    )

    # ========================================================
    # CHECK DATA ALIGNMENT
    # ========================================================

    if len(df) != len(X_numeric):

        raise ValueError(
            "The number of DataFrame rows does not "
            "match the number of feature vectors."
        )

    # ========================================================
    # FIND BEST K
    # ========================================================

    print("\n" + "=" * 70)
    print("FINDING A SUITABLE NUMBER OF CLUSTERS")
    print("=" * 70)

    k_results = evaluate_k_values(
        X=X_numeric,
        min_k=MIN_K,
        max_k=MAX_K,
    )

    best_k = select_best_k(
        k_results
    )

    best_row = k_results[
        k_results["k"] == best_k
    ].iloc[0]

    print("\n" + "-" * 70)

    print(
        f"Selected K: {best_k}"
    )

    print(
        f"Silhouette score: "
        f"{best_row['silhouette_score']:.4f}"
    )

    print(
        "\nK was selected using the highest "
        "silhouette score."
    )

    # ========================================================
    # RUN FINAL K-MEANS
    # ========================================================

    print("\n" + "=" * 70)
    print(
        f"RUNNING FINAL K-MEANS WITH K = {best_k}"
    )
    print("=" * 70)

    labels = cluster_suburbs(
        X=X_numeric,
        n_clusters=best_k,
        random_state=RANDOM_STATE,
    )

    clustered_df = add_cluster_labels(
        df,
        labels,
    )

    # ========================================================
    # CLUSTER SUMMARY
    # ========================================================

    summary = (
        clustered_df
        .groupby("cluster")
        .size()
        .reset_index(
            name="suburb_count"
        )
        .sort_values("cluster")
    )

    print("\nCluster summary:")
    print("-" * 40)

    for _, row in summary.iterrows():

        cluster_id = int(
            row["cluster"]
        )

        suburb_count = int(
            row["suburb_count"]
        )

        print(
            f"Cluster {cluster_id}: "
            f"{suburb_count:,} suburbs"
        )

    # ========================================================
    # ASK FOR SUBURB
    # ========================================================

    print("\n" + "=" * 70)

    suburb_name = input(
        "Enter suburb name: "
    ).strip()

    if not suburb_name:

        print(
            "\nNo suburb entered."
        )

        return

    # ========================================================
    # FIND SUBURB
    # ========================================================

    match = find_suburb(
        clustered_df,
        suburb_name,
    )

    if match.empty:

        print(
            f"\nSuburb '{suburb_name}' "
            "was not found in the dataset."
        )

        print(
            "\nPossible matches:"
        )

        partial = clustered_df[
            clustered_df["sa2_name"]
            .astype(str)
            .str.lower()
            .str.contains(
                suburb_name.lower(),
                na=False,
            )
        ]

        if partial.empty:

            print(
                "  No partial matches found."
            )

        else:

            for name in (
                partial["sa2_name"]
                .drop_duplicates()
                .head(20)
            ):

                print(
                    f"  - {name}"
                )

        return

    # ========================================================
    # FIND ORIGINAL ROW INDEX
    # ========================================================

    selected_index = match.index[0]

    selected = clustered_df.loc[
        selected_index
    ]

    cluster_id = int(
        selected["cluster"]
    )

    # ========================================================
    # DISPLAY SELECTED SUBURB
    # ========================================================

    print("\n" + "=" * 70)
    print("SUBURB CLUSTER RESULT")
    print("=" * 70)

    print(
        f"\nSelected suburb: "
        f"{selected['sa2_name']}"
    )

    if "state" in selected.index:

        print(
            f"State: "
            f"{selected['state']}"
        )

    print(
        f"Cluster: {cluster_id}"
    )

    # ========================================================
    # FIND TOP 10 SIMILAR SUBURBS
    # ========================================================

    top_similar = find_top_similar_suburbs(
        df=clustered_df,
        X=X_numeric,
        suburb_index=selected_index,
        cluster_id=cluster_id,
        top_n=TOP_N,
    )

    # ========================================================
    # DISPLAY TOP 10
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP 10 SIMILAR SUBURBS")
    print("=" * 70)

    if top_similar.empty:

        print(
            "\nNo other suburbs were found "
            "in the selected cluster."
        )

    else:

        for position, row in (
            top_similar
            .iterrows()
        ):

            print(
                f"{position + 1:2d}. "
                f"{row['suburb']:<40} "
                f"Similarity: "
                f"{row['similarity']:.4f}"
            )

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "Clustering and similarity search completed."
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()



