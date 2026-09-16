import os
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

from engine.index import (
    build_faiss_index,
    faiss_find_top_n
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

    header = st.container()

    with header:
        logo_col, title_col, user_col = st.columns([1.15, 4.8, 1.2], vertical_alignment="center")

        with logo_col:
            try:
                if logo_path.exists() and logo_path.is_file():
                    st.image(str(logo_path), width=170)
                else:
                    raise FileNotFoundError("Logo file not found")
            except Exception:
                # Keep the application usable even if the supplied
                # logo file is missing or is not a valid image.
                st.markdown(
                    '<div class="brand-wordmark">Demografy</div>',
                    unsafe_allow_html=True
                )

        with title_col:
            st.markdown(
                """
                <div class="header-title">Suburb Lookalike Finder</div>
                <div class="header-subtitle">16-KPI hybrid demographic similarity · Gemini embeddings · FAISS</div>
                """,
                unsafe_allow_html=True
            )

        with user_col:
            if st.session_state.get("logged_in") and st.session_state.get("user"):
                current_user = st.session_state["user"]
                st.markdown(
                    f'<div class="header-user">{current_user["user_id"]}</div>',
                    unsafe_allow_html=True
                )

    st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)


# ============================================================
# LOGIN SCREEN
# ============================================================

if not st.session_state[
    "logged_in"
]:

    logo_path = (
        Path(__file__).parent
        / "asset"
        / "demografy_logo.png"
    )

    logo_html = ""

    login_logo_left, login_logo_mid, login_logo_right = st.columns([1, 1.1, 1])
    with login_logo_mid:
        try:
            if logo_path.exists() and logo_path.is_file():
                st.image(str(logo_path), width=220)
            else:
                raise FileNotFoundError("Logo file not found")
        except Exception:
            st.markdown(
                '<div class="login-wordmark">Demografy</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        f"""
        <div class="login-shell">
            <div class="login-brand">
                <div class="login-title">Suburb Lookalike Finder</div>
                <div class="login-subtitle">Sign in to discover Australian suburbs with similar demographic profiles.</div>
            </div>
            <div class="login-note">Access is controlled by your Demografy user ID and subscription tier.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    login_left, login_mid, login_right = st.columns([1.2, 1, 1.2])

    with login_mid:
        user_id = st.text_input(
            "User ID",
            placeholder="e.g. user_001"
        )

        login_clicked = st.button(
            "Sign in",
            type="primary",
            use_container_width=True
        )

        if login_clicked:
            entered_user_id = user_id.strip()

            if not entered_user_id:
                st.warning("Please enter your user ID.")
            else:
                try:
                    client = get_bigquery_client()
                    user = get_user(client, entered_user_id)

                    if user is None:
                        st.error("Invalid user ID or inactive account.")
                    else:
                        try:
                            get_tier_config(user["tier"])
                        except ValueError:
                            st.error("This account has an unsupported tier.")
                            st.stop()

                        st.session_state["logged_in"] = True
                        st.session_state["user"] = user
                        st.session_state["lookup_count"] = 0
                        st.session_state["latest_results"] = None
                        st.rerun()

                except Exception as exc:
                    st.error("Unable to complete login.")
                    st.exception(exc)

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
# SIDEBAR ACCOUNT / LOOKUP STATUS
# ============================================================

lookup_count = st.session_state["lookup_count"]
remaining = get_lookups_remaining(tier, lookup_count)
lookup_allowed = can_lookup(tier, lookup_count)
warning_at = tier_config["warning_at"]

with st.sidebar:
    st.markdown('<div class="sidebar-label">Account</div>', unsafe_allow_html=True)

    badge = ' <span class="demografy-badge">PRO</span>' if tier == "pro" else ""
    st.markdown(
        f"""
        <div class="account-card">
            <div class="account-line">{user['user_id']} · {tier.title()} {badge}</div>
            <div class="account-small">{remaining} / {tier_config['lookup_limit']} lookups left</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if warning_at is not None and lookup_count >= warning_at and lookup_allowed:
        st.warning(f"You have {remaining} lookups remaining this session.")

    if not lookup_allowed:
        st.error("Lookup limit reached.")
        if tier == "free":
            st.info("Upgrade your plan for additional lookups and more matches.")

    if st.button("Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.divider()
    st.markdown('<div class="sidebar-label">Lookup</div>', unsafe_allow_html=True)


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

        value=0.2,

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
# PRO PRESETS - COMPACT ONE-LINE BUTTONS
# ============================================================

if tier_config["presets_enabled"]:

    st.sidebar.markdown(
        '<div class="sidebar-label">Presets</div>',
        unsafe_allow_html=True
    )

    # Keep labels short so all four buttons remain on one line.
    preset_labels = {
        "Balanced": "Balanced",
        "Family-focused": "Family",
        "Investor": "Investor",
        "Lifestyle": "Lifestyle",
    }

    available_presets = [
        name for name in preset_labels
        if name in WEIGHT_PRESETS
    ]

    # Include any extra preset names without breaking the application.
    for name in WEIGHT_PRESETS:
        if name not in available_presets:
            available_presets.append(name)
            preset_labels[name] = name

    preset_cols = st.sidebar.columns(len(available_presets), gap="small")
    active_preset = st.session_state.get("current_preset", "Balanced")

    for preset_col, preset_name in zip(preset_cols, available_presets):
        with preset_col:
            is_active = preset_name == active_preset
            if st.button(
                preset_labels[preset_name],
                key=f"preset_btn_{preset_name}",
                disabled=not lookup_allowed,
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                preset_weights = get_preset(preset_name)

                for kpi in KPI_COLS:
                    st.session_state[f"weight_{kpi}"] = preset_weights[kpi]

                st.session_state["current_preset"] = preset_name
                st.rerun()

    st.sidebar.caption(
        f"Active preset: {preset_labels.get(active_preset, active_preset)}"
    )

else:
    st.sidebar.info("Weight presets are available on the Pro tier.")


# ============================================================
# KPI WEIGHTS
# ============================================================

weights = {}

with st.sidebar.expander(
    "Advanced KPI weights",
    expanded=False
):
    st.caption("0 = ignored · 1 = normal · 2 = double importance")

    for kpi in KPI_COLS:
        weights[kpi] = st.slider(
            KPI_LABELS[kpi],
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            key=f"weight_{kpi}",
            disabled=not lookup_allowed
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

reference = df.iloc[reference_index]

st.markdown('<div class="section-label">Search overview</div>', unsafe_allow_html=True)

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns([1.45, 0.8, 1.0, 1.0])

with summary_col1:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">Reference suburb</div>
            <div class="summary-value summary-suburb">{reference['sa2_name']}</div>
            <div class="summary-detail">{reference['state']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with summary_col2:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">Matches</div>
            <div class="summary-value">{top_n}</div>
            <div class="summary-detail">Top N results</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with summary_col3:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">Hybrid blend α</div>
            <div class="summary-value">{alpha:.2f}</div>
            <div class="summary-detail">{1-alpha:.0%} numeric · {alpha:.0%} text</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with summary_col4:
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">Account</div>
            <div class="summary-value">{tier.title()}</div>
            <div class="summary-detail">{remaining} lookups left</div>
        </div>
        """,
        unsafe_allow_html=True
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
# RUN LOOKUP DIRECTLY WITH FAISS
# ============================================================

if find_clicked:

    try:

        with st.spinner(
            "Finding similar suburbs..."
        ):

            # Rebuild the exact FAISS index because the KPI
            # weights and alpha can change the hybrid vectors.
            faiss_index, X_faiss = build_faiss_index(
                X_hybrid
            )

            results = faiss_find_top_n(
                df,
                faiss_index,
                X_faiss,
                reference_index,
                n=top_n
            )

            explained_results = explain_results(
                df,
                results,
                X_weighted,
                X_hybrid,
                reference_index,
                weights
            )

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

            st.session_state[
                "lookup_count"
            ] += 1

        st.rerun()

    except Exception as exc:

        st.error(
            "The suburb lookup could not be completed."
        )

        st.exception(exc)


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
# EMPTY STATE
# ============================================================

if st.session_state["latest_results"] is None and lookup_allowed:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">⌕</div>
            <div class="empty-title">Ready to find suburb lookalikes</div>
            <div class="empty-copy">Choose a reference suburb, number of matches, blend and optional preset, then select <b>Find Lookalikes</b>.</div>
        </div>
        """,
        unsafe_allow_html=True
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


    st.markdown(
        f"""
        <div class="content-card">
            <div class="section-label">Lookalike results</div>
            <div class="results-title">Suburbs most like {result_reference['sa2_name']} ({result_reference['state']})</div>
            <div class="results-subtitle">Hybrid ranking · {len(explained_results)} matches · α = {st.session_state.get('latest_alpha', 0.2):.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
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
    # KPI PROFILE COMPARISON
    # ========================================================

    st.subheader(
        "KPI Profile Comparison"
    )

    st.write(
        """
        Compare the reference suburb with up to
        three matched suburbs across all 16 KPIs.
        """
    )


    # ========================================================
    # BUILD MATCH LOOKUP
    # ========================================================

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


    # ========================================================
    # SELECT UP TO 3 MATCHED SUBURBS
    # ========================================================

    default_matches = (
        match_codes[:3]
    )


    selected_match_codes = (
        st.multiselect(

            "Compare with up to 3 matched suburbs",

            options=match_codes,

            default=default_matches,

            max_selections=3,

            format_func=lambda code:
                match_lookup[
                    code
                ],

            key="radar_matches"
        )
    )


    # ========================================================
    # REFERENCE SUBURB NAME
    # ========================================================

    reference_name = (

        f"{df.iloc[result_reference_index]['sa2_name']} "
        f"({df.iloc[result_reference_index]['state']})"
    )


    # ========================================================
    # CREATE RADAR CHART
    # ========================================================

    if selected_match_codes:

        fig = go.Figure()


        # ----------------------------------------------------
        # USE FIRST SELECTED SUBURB TO GET KPI LABELS
        # AND REFERENCE VALUES
        # ----------------------------------------------------

        first_candidate_index = (
            code_to_index[
                selected_match_codes[0]
            ]
        )


        first_radar_data = (
            get_radar_data(
                df,
                result_reference_index,
                first_candidate_index
            )
        )


        labels = [

            item["label"]

            for item
            in first_radar_data
        ]


        reference_scores = [

            item["reference"]

            for item
            in first_radar_data
        ]


        # ----------------------------------------------------
        # REFERENCE SUBURB
        # ----------------------------------------------------

        fig.add_trace(

            go.Scatterpolar(

                r=reference_scores,

                theta=labels,

                fill="toself",

                name=reference_name,

                line=dict(
                    color=PLOT_PRIMARY,
                    width=4
                ),

                fillcolor=(
                    "rgba("
                    "154, 102, 238, 0.16"
                    ")"
                )
            )
        )


        # ----------------------------------------------------
        # COLOURS FOR THE 3 MATCHED SUBURBS
        # ----------------------------------------------------

        comparison_colours = [

            "#00A6A6",

            "#E67E22",

            "#379634"
        ]


        # ----------------------------------------------------
        # ADD EACH SELECTED MATCHED SUBURB
        # ----------------------------------------------------

        for position, match_code in enumerate(
            selected_match_codes
        ):

            candidate_index = (
                code_to_index[
                    match_code
                ]
            )


            radar_data = (
                get_radar_data(
                    df,
                    result_reference_index,
                    candidate_index
                )
            )


            candidate_scores = [

                item["candidate"]

                for item
                in radar_data
            ]


            candidate_name = (

                f"{df.iloc[candidate_index]['sa2_name']} "
                f"({df.iloc[candidate_index]['state']})"
            )


            fig.add_trace(

                go.Scatterpolar(

                    r=candidate_scores,

                    theta=labels,

                    fill=None,

                    name=candidate_name,

                    line=dict(

                        color=
                            comparison_colours[
                                position
                            ],

                        width=3
                    )
                )
            )


        # ====================================================
        # RADAR CHART LAYOUT
        # ====================================================

        fig.update_layout(

            polar=dict(

                bgcolor=PLOT_WHITE,

                radialaxis=dict(

                    visible=True,

                    range=[
                        0,
                        100
                    ],

                    tickvals=[
                        0,
                        20,
                        40,
                        60,
                        80,
                        100
                    ],

                    ticksuffix="%",

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

                orientation="h",

                yanchor="bottom",

                y=1.08,

                xanchor="center",

                x=0.5
            ),

            showlegend=True,

            height=720,

            margin=dict(
                l=90,
                r=90,
                t=130,
                b=70
            ),

            title=(
                f"{reference_name} vs "
                f"{len(selected_match_codes)} "
                f"matched suburb"
                f"{'s' if len(selected_match_codes) > 1 else ''}"
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.caption(
            """
            Radar values represent percentile ranks
            across all suburbs.

            100 = among the highest values.

            50 = around the middle.

            0 = among the lowest values.
            """
        )


        # ====================================================
        # KPI COMPARISON DETAILS
        # ====================================================

        st.subheader(
            "KPI Comparison Details"
        )


        st.write(
            """
            Select one suburb below for a detailed
            KPI-by-KPI comparison with the reference suburb.
            """
        )


        detail_match_code = (
            st.selectbox(

                "Detailed comparison with",

                options=
                    selected_match_codes,

                format_func=lambda code:
                    match_lookup[
                        code
                    ],

                key="detail_match"
            )
        )


        detail_candidate_index = (
            code_to_index[
                detail_match_code
            ]
        )


        comparison_table = (
            get_kpi_comparison_table(
                df,
                result_reference_index,
                detail_candidate_index
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
            Difference = matched suburb percentile
            minus reference suburb percentile.

            Positive values mean the matched suburb
            ranks higher on that KPI.

            Negative values mean the reference suburb
            ranks higher.
            """
        )

    else:

        st.info(
            "Select at least one matched suburb "
            "to display the KPI comparison."
        )
 

# ============================================================
# DEVELOPMENT INFORMATION
# ============================================================

if os.getenv("SHOW_DEVELOPMENT_INFO", "false").lower() == "true":
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
