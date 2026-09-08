import os
import base64
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# DATABASE
# ============================================================

from db.bigquery_client import (
    get_bigquery_client
)


# ============================================================
# FEATURES
# ============================================================

from engine.features import (
    load_features,
    clean_features,
    standardise_features,
    KPI_COLS
)


# ============================================================
# TEXT EMBEDDINGS
# ============================================================

from engine.profiles import (
    generate_all_profiles
)

from engine.text_embed import (
    get_or_create_embeddings
)


# ============================================================
# WEIGHTS / FUSION / SIMILARITY
# ============================================================

from engine.weights import (
    KPI_LABELS,
    WEIGHT_PRESETS,
    get_preset,
    apply_feature_weights
)

from engine.fusion import (
    fuse_vectors
)

from engine.similarity import (
    find_top_n
)


# ============================================================
# EXPLAINABILITY
# ============================================================

from engine.explain import (
    explain_results,
    get_radar_data,
    get_kpi_comparison_table
)


# ============================================================
# RBAC
# ============================================================

from auth.users import (
    get_user
)

from auth.rbac import (
    get_tier_config,
    can_lookup,
    get_lookups_remaining
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Demografy Suburb Lookalike Finder",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# EXTERNAL STYLESHEET
# ============================================================

def load_css():

    css_path = (
        Path(__file__).parent
        / "style"
        / "styles.css"
    )

    if css_path.exists():

        with open(
            css_path,
            "r",
            encoding="utf-8"
        ) as css_file:

            st.markdown(
                f"<style>{css_file.read()}</style>",
                unsafe_allow_html=True
            )

    else:

        st.warning(
            f"CSS file not found: {css_path}"
        )


load_css()


# ============================================================
# PLOTLY CHART COLOURS
# ============================================================
# Plotly styling is Python chart configuration, not CSS.

PLOT_PRIMARY = "#9a66ee"
PLOT_CYAN = "#8df2ed"
PLOT_JET = "#272d2d"
PLOT_GREY = "#dbdddc"
PLOT_WHITE = "#ffffff"


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:

    st.session_state[
        "logged_in"
    ] = False


if "user" not in st.session_state:

    st.session_state[
        "user"
    ] = None


if "lookup_count" not in st.session_state:

    st.session_state[
        "lookup_count"
    ] = 0


if "latest_results" not in st.session_state:

    st.session_state[
        "latest_results"
    ] = None


if "latest_reference_index" not in st.session_state:

    st.session_state[
        "latest_reference_index"
    ] = None


if "latest_alpha" not in st.session_state:

    st.session_state[
        "latest_alpha"
    ] = None


if "latest_weights" not in st.session_state:

    st.session_state[
        "latest_weights"
    ] = None


# ============================================================
# HEADER FUNCTION
# ============================================================

def show_header():

    logo_path = (
        Path(__file__).parent
        / "asset"
        / "demografy_logo.png"
    )


    if os.path.exists(
        logo_path
    ):

        logo_col, title_col = (
            st.columns(
                [1.3, 5]
            )
        )


        with logo_col:

            logo_data = base64.b64encode(
                logo_path.read_bytes()
            ).decode("ascii")

            st.html(
                f'<img src="data:image/svg+xml;base64,{logo_data}" '
                f'width="190" alt="Demografy logo">'
            )


        with title_col:

            st.markdown(
                """
                <h1 class="demografy-title">
                    Suburb Lookalike Finder
                </h1>

                <div class="
                    demografy-subtitle
                ">
                    Hybrid demographic
                    similarity search
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.markdown(
            """
            <h1>
                Suburb Lookalike Finder
            </h1>

            <div class="
                demografy-subtitle
            ">
                Discover Australian suburbs
                with similar demographic
                characteristics
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        """
        <div class="
            demografy-divider
        ">
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LOGIN SCREEN
# ============================================================

if not st.session_state[
    "logged_in"
]:

    show_header()


    st.subheader(
        "Sign in"
    )


    st.caption(
        "Enter your Demografy user ID "
        "to continue."
    )


    user_id = st.text_input(
        "User ID",
        placeholder="e.g. user_001"
    )


    login_clicked = st.button(
        "Login",
        type="primary"
    )


    if login_clicked:

        entered_user_id = (
            user_id.strip()
        )


        if not entered_user_id:

            st.warning(
                "Please enter your "
                "user ID."
            )

        else:

            try:

                client = (
                    get_bigquery_client()
                )


                user = get_user(
                    client,
                    entered_user_id
                )


                if user is None:

                    st.error(
                        "Invalid user ID "
                        "or inactive account."
                    )

                else:

                    try:

                        get_tier_config(
                            user["tier"]
                        )

                    except ValueError:

                        st.error(
                            "This account has "
                            "an unsupported tier."
                        )

                        st.stop()


                    st.session_state[
                        "logged_in"
                    ] = True


                    st.session_state[
                        "user"
                    ] = user


                    st.session_state[
                        "lookup_count"
                    ] = 0


                    st.session_state[
                        "latest_results"
                    ] = None


                    st.rerun()


            except Exception as exc:

                st.error(
                    "Unable to complete "
                    "login."
                )

                st.exception(
                    exc
                )


    st.stop()


# ============================================================
# LOAD APPLICATION RESOURCES
# ============================================================

@st.cache_resource
def load_app_data():

    client = (
        get_bigquery_client()
    )


    # --------------------------------------------------------
    # BIGQUERY DATA
    # --------------------------------------------------------

    df = load_features(
        client
    )


    df = clean_features(
        df
    )


    # --------------------------------------------------------
    # NUMERIC FEATURES
    # --------------------------------------------------------

    X_numeric, scaler = (
        standardise_features(
            df
        )
    )


    # --------------------------------------------------------
    # TEXT PROFILES
    # --------------------------------------------------------

    profiles = (
        generate_all_profiles(
            df
        )
    )


    # --------------------------------------------------------
    # CACHED GEMINI EMBEDDINGS
    # --------------------------------------------------------

    X_text = (
        get_or_create_embeddings(
            profiles
        )
    )


    return (
        df,
        X_numeric,
        X_text,
        scaler
    )


with st.spinner(
    "Loading suburb data..."
):

    (
        df,
        X_numeric,
        X_text,
        scaler
    ) = load_app_data()


# ============================================================
# CURRENT USER / TIER
# ============================================================

user = st.session_state[
    "user"
]


tier = str(
    user["tier"]
).lower()


tier_config = (
    get_tier_config(
        tier
    )
)


# ============================================================
# INITIALISE KPI WEIGHT STATE
# ============================================================

if "weights_initialised" not in (
    st.session_state
):

    balanced_weights = (
        get_preset(
            "Balanced"
        )
    )


    for kpi in KPI_COLS:

        st.session_state[
            f"weight_{kpi}"
        ] = balanced_weights[
            kpi
        ]


    st.session_state[
        "current_preset"
    ] = "Balanced"


    st.session_state[
        "weights_initialised"
    ] = True


# ============================================================
# APPLICATION HEADER
# ============================================================

show_header()


# ============================================================
# SIDEBAR ACCOUNT INFORMATION
# ============================================================

st.sidebar.header(
    "Account"
)


st.sidebar.write(
    f"**User:** "
    f"{user['user_id']}"
)


st.sidebar.write(
    f"**Tier:** "
    f"{tier.title()}"
)


if tier == "pro":

    st.sidebar.markdown(
        """
        <span class="
            demografy-badge
        ">
            PRO
        </span>
        """,
        unsafe_allow_html=True
    )


st.sidebar.divider()


# ============================================================
# LOOKUP COUNTER
# ============================================================

lookup_count = (
    st.session_state[
        "lookup_count"
    ]
)


remaining = (
    get_lookups_remaining(
        tier,
        lookup_count
    )
)


lookup_allowed = (
    can_lookup(
        tier,
        lookup_count
    )
)


st.sidebar.metric(
    "Lookups Remaining",
    remaining
)


st.sidebar.caption(
    f"Used this session: "
    f"{lookup_count} / "
    f"{tier_config['lookup_limit']}"
)


# ============================================================
# TIER WARNINGS
# ============================================================

warning_at = (
    tier_config[
        "warning_at"
    ]
)


if (
    warning_at is not None
    and
    lookup_count >= warning_at
    and
    lookup_allowed
):

    st.sidebar.warning(
        f"You have {remaining} "
        "lookups remaining "
        "this session."
    )


if not lookup_allowed:

    st.sidebar.error(
        "Lookup limit reached."
    )


    if tier == "free":

        st.sidebar.info(
            "Upgrade your plan for "
            "additional lookups and "
            "more matches."
        )


# ============================================================
# LOGOUT
# ============================================================

if st.sidebar.button(
    "Logout"
):

    st.session_state.clear()

    st.rerun()


st.sidebar.divider()


# ============================================================
# SEARCH SETTINGS
# ============================================================

st.sidebar.header(
    "Search Settings"
)


# ============================================================
# SA2 LOOKUP
# ============================================================

sa2_codes = (
    df["sa2_code"]
    .astype(str)
    .tolist()
)


display_lookup = dict(

    zip(

        df["sa2_code"]
        .astype(str),

        (
            df["sa2_name"]
            .astype(str)

            + " ("

            + df["state"]
            .astype(str)

            + ")"
        )
    )
)


code_to_index = {

    str(code): index

    for index, code
    in enumerate(
        df["sa2_code"]
    )
}


selected_sa2 = (
    st.sidebar.selectbox(

        "Reference suburb",

        options=sa2_codes,

        format_func=lambda code:
            display_lookup[
                code
            ],

        disabled=not lookup_allowed
    )
)


reference_index = (
    code_to_index[
        selected_sa2
    ]
)


# ============================================================
# NUMBER OF MATCHES
# ============================================================

max_matches = (
    tier_config[
        "max_matches"
    ]
)


default_matches = min(
    10,
    max_matches
)


top_n = (
    st.sidebar.slider(

        "Number of matches",

        min_value=1,

        max_value=max_matches,

        value=default_matches,

        step=1,

        disabled=not lookup_allowed
    )
)


# ============================================================
# HYBRID BLEND
# ============================================================

st.sidebar.subheader(
    "Hybrid Blend"
)


alpha = (
    st.sidebar.slider(

        "Gemini influence (alpha)",

        min_value=0.0,

        max_value=1.0,

        value=0.5,

        step=0.05,

        disabled=not lookup_allowed
    )
)


st.sidebar.caption(
    f"Numeric influence: "
    f"{1 - alpha:.0%}"
)


st.sidebar.caption(
    f"Gemini influence: "
    f"{alpha:.0%}"
)


# ============================================================
# PRO PRESETS
# ============================================================

if tier_config[
    "presets_enabled"
]:

    st.sidebar.subheader(
        "Weight Preset"
    )


    preset_name = (
        st.sidebar.selectbox(

            "Choose preset",

            options=list(
                WEIGHT_PRESETS.keys()
            ),

            disabled=not lookup_allowed
        )
    )


    if st.sidebar.button(
        "Apply Preset",
        disabled=not lookup_allowed
    ):

        preset_weights = (
            get_preset(
                preset_name
            )
        )


        for kpi in KPI_COLS:

            st.session_state[
                f"weight_{kpi}"
            ] = preset_weights[
                kpi
            ]


        st.session_state[
            "current_preset"
        ] = preset_name


        st.rerun()

else:

    st.sidebar.info(
        "Weight presets are available "
        "on the Pro tier."
    )


# ============================================================
# KPI WEIGHTS
# ============================================================

st.sidebar.subheader(
    "KPI Weights"
)


st.sidebar.caption(
    """
    0 = ignored

    1 = normal importance

    2 = double importance
    """
)


weights = {}


for kpi in KPI_COLS:

    weights[
        kpi
    ] = (
        st.sidebar.slider(

            KPI_LABELS[
                kpi
            ],

            min_value=0.0,

            max_value=2.0,

            step=0.1,

            key=f"weight_{kpi}",

            disabled=not lookup_allowed
        )
    )


# ============================================================
# WEIGHT NUMERIC MATRIX
# ============================================================

X_weighted = (
    apply_feature_weights(
        X_numeric,
        weights
    )
)


# ============================================================
# FUSE NUMERIC + TEXT
# ============================================================

X_hybrid = (
    fuse_vectors(
        X_weighted,
        X_text,
        alpha=alpha
    )
)


# ============================================================
# SEARCH SUMMARY
# ============================================================

reference = (
    df.iloc[
        reference_index
    ]
)


st.subheader(
    "Search Configuration"
)


summary_col1, summary_col2, (
    summary_col3
) = st.columns(
    3
)


with summary_col1:

    st.metric(
        "Reference Suburb",
        (
            f"{reference['sa2_name']} "
            f"({reference['state']})"
        )
    )


with summary_col2:

    st.metric(
        "Numeric Influence",
        f"{1 - alpha:.0%}"
    )


with summary_col3:

    st.metric(
        "Gemini Influence",
        f"{alpha:.0%}"
    )


# ============================================================
# FIND LOOKALIKES BUTTON
# ============================================================

st.write("")


find_clicked = (
    st.button(

        "🔍 Find Lookalikes",

        type="primary",

        disabled=not lookup_allowed
    )
)


# ============================================================
# RUN LOOKUP
# ============================================================

if find_clicked:

    with st.spinner(
        "Finding similar suburbs..."
    ):

        results = (
            find_top_n(
                df,
                X_hybrid,
                reference_index,
                n=top_n
            )
        )


        explained_results = (
            explain_results(
                df,
                results,
                X_weighted,
                X_hybrid,
                reference_index,
                weights
            )
        )


        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        st.session_state[
            "latest_results"
        ] = explained_results


        st.session_state[
            "latest_reference_index"
        ] = reference_index


        st.session_state[
            "latest_alpha"
        ] = alpha


        st.session_state[
            "latest_weights"
        ] = weights.copy()


        # ----------------------------------------------------
        # INCREMENT SESSION LOOKUP
        # ----------------------------------------------------

        st.session_state[
            "lookup_count"
        ] += 1


    # Rerun so sidebar counter
    # immediately reflects new value
    st.rerun()


# ============================================================
# LOOKUP LIMIT MESSAGE
# ============================================================

if not lookup_allowed:

    st.warning(
        "You have reached your "
        "lookup limit for this session."
    )


    if tier == "free":

        st.info(
            "Upgrade your plan to run "
            "more lookups and return "
            "more suburb matches."
        )


# ============================================================
# DISPLAY LAST SEARCH RESULTS
# ============================================================

if (
    st.session_state[
        "latest_results"
    ] is not None
):

    explained_results = (
        st.session_state[
            "latest_results"
        ]
    )


    result_reference_index = (
        st.session_state[
            "latest_reference_index"
        ]
    )


    result_reference = (
        df.iloc[
            result_reference_index
        ]
    )


    st.markdown(
        """
        <div class="
            demografy-divider
        ">
        </div>
        """,
        unsafe_allow_html=True
    )


    st.subheader(
        "Most Similar Suburbs"
    )


    st.caption(
        (
            "Results for "
            f"{result_reference['sa2_name']} "
            f"({result_reference['state']})"
        )
    )


    # ========================================================
    # RESULTS TABLE
    # ========================================================

    display_results = (
        explained_results[
            [
                "rank",
                "sa2_name",
                "state",
                "similarity",
                "top_kpis",
                "numeric_rank",
                "hybrid_rank",
                "rank_delta"
            ]
        ].copy()
    )


    display_results[
        "similarity"
    ] = (
        display_results[
            "similarity"
        ]
        * 100
    ).round(2)


    display_results.columns = [

        "Rank",

        "Suburb",

        "State",

        "Similarity %",

        "Top 3 Contributing KPIs",

        "Numeric Rank",

        "Hybrid Rank",

        "Rank Δ"
    ]


    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )


    st.caption(
        """
        Rank Δ shows how much the suburb
        moved after Gemini text similarity
        was added.

        Positive = moved higher.
        Negative = moved lower.
        """
    )


    # ========================================================
    # RADAR COMPARISON
    # ========================================================

    st.subheader(
        "KPI Profile Comparison"
    )


    st.write(
        """
        Select one of the matched suburbs
        to compare its 16-KPI profile
        against the reference suburb.
        """
    )


    match_codes = (
        explained_results[
            "sa2_code"
        ]
        .astype(str)
        .tolist()
    )


    match_lookup = dict(

        zip(

            explained_results[
                "sa2_code"
            ]
            .astype(str),

            (
                explained_results[
                    "sa2_name"
                ]
                .astype(str)

                + " ("

                + explained_results[
                    "state"
                ]
                .astype(str)

                + ")"
            )
        )
    )


    selected_match_code = (
        st.selectbox(

            "Compare with",

            options=match_codes,

            format_func=lambda code:
                match_lookup[
                    code
                ],

            key="radar_match"
        )
    )


    candidate_index = (
        code_to_index[
            selected_match_code
        ]
    )


    # ========================================================
    # RADAR DATA
    # ========================================================

    radar_data = (
        get_radar_data(
            df,
            result_reference_index,
            candidate_index
        )
    )


    labels = [
        item["label"]
        for item in radar_data
    ]


    reference_scores = [
        item["reference"]
        for item in radar_data
    ]


    candidate_scores = [
        item["candidate"]
        for item in radar_data
    ]


    reference_name = (
        f"{df.iloc[result_reference_index]['sa2_name']} "
        f"({df.iloc[result_reference_index]['state']})"
    )


    candidate_name = (
        f"{df.iloc[candidate_index]['sa2_name']} "
        f"({df.iloc[candidate_index]['state']})"
    )


    # ========================================================
    # RADAR CHART
    # ========================================================

    fig = go.Figure()


    fig.add_trace(

        go.Scatterpolar(

            r=reference_scores,

            theta=labels,

            fill="toself",

            name=reference_name,

            line=dict(
                color=PLOT_PRIMARY,
                width=3
            ),

            fillcolor=(
                "rgba("
                "154, 102, 238, 0.18"
                ")"
            )
        )
    )


    fig.add_trace(

        go.Scatterpolar(

            r=candidate_scores,

            theta=labels,

            fill="toself",

            name=candidate_name,

            line=dict(
                color=PLOT_CYAN,
                width=3
            ),

            fillcolor=(
                "rgba("
                "141, 242, 237, 0.20"
                ")"
            )
        )
    )


    fig.update_layout(

        polar=dict(

            bgcolor=PLOT_WHITE,

            radialaxis=dict(

                visible=True,

                range=[
                    0,
                    100
                ],

                gridcolor=
                    PLOT_GREY,

                linecolor=
                    PLOT_GREY
            ),

            angularaxis=dict(

                gridcolor=
                    PLOT_GREY
            )
        ),

        paper_bgcolor=
            PLOT_WHITE,

        font=dict(
            color=PLOT_JET
        ),

        legend=dict(
            orientation="h"
        ),

        showlegend=True,

        height=700,

        margin=dict(
            l=90,
            r=90,
            t=100,
            b=70
        ),

        title=(
            f"{reference_name} vs "
            f"{candidate_name}"
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.caption(
        """
        Radar values represent percentile
        ranks across all suburbs.

        100 = among the highest values.

        50 = around the middle.

        0 = among the lowest values.
        """
    )


    # ========================================================
    # KPI COMPARISON TABLE
    # ========================================================

    st.subheader(
        "KPI Comparison Details"
    )


    comparison_table = (
        get_kpi_comparison_table(
            df,
            result_reference_index,
            candidate_index
        )
    )


    numeric_columns = [

        "Reference Percentile",

        "Matched Percentile",

        "Difference"
    ]


    if (
        "Absolute Difference"
        in comparison_table.columns
    ):

        numeric_columns.append(
            "Absolute Difference"
        )


    for column in numeric_columns:

        comparison_table[
            column
        ] = (
            comparison_table[
                column
            ].round(1)
        )


    st.dataframe(
        comparison_table,
        use_container_width=True,
        hide_index=True
    )


    st.caption(
        """
        Difference = matched suburb
        percentile minus reference
        suburb percentile.

        Positive values mean the matched
        suburb ranks higher on that KPI.

        Negative values mean the reference
        suburb ranks higher.
        """
    )


# ============================================================
# DEVELOPMENT INFORMATION
# ============================================================

with st.expander(
    "Development Information"
):

    st.write(
        "Numeric matrix:",
        X_numeric.shape
    )


    st.write(
        "Text matrix:",
        X_text.shape
    )


    st.write(
        "Current hybrid matrix:",
        X_hybrid.shape
    )


    st.write(
        "Number of KPIs:",
        len(KPI_COLS)
    )


    st.write(
        "User tier:",
        tier
    )


    st.write(
        "Lookup count:",
        st.session_state[
            "lookup_count"
        ]
    )


    st.write(
        "Maximum matches:",
        tier_config[
            "max_matches"
        ]
    )