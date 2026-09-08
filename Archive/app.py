import streamlit as st
import plotly.graph_objects as go
import base64
from pathlib import Path


from db.bigquery_client import (
    get_bigquery_client
)

from engine.features import (
    load_features,
    clean_features,
    standardise_features
)

from engine.profiles import (
    generate_all_profiles
)

from engine.text_embed import (
    get_or_create_embeddings
)

from engine.weights import (
    KPI_COLS,
    KPI_LABELS,
    get_preset,
    apply_feature_weights
)

from engine.fusion import (
    fuse_vectors
)

from engine.similarity import (
    find_top_n
)

from engine.explain import (
    explain_results,
    get_radar_data,
    get_kpi_comparison_table
)

from auth.users import (
    get_user
)

from auth.rbac import (
    get_tier_config,
    can_lookup,
    get_lookups_remaining
)


# ============================================================
# DEMOGRAFY BRAND COLOURS
# ============================================================

BRAND_PRIMARY = "#9a66ee"
BRAND_BLUE = "#5e17eb"
BRAND_CYAN = "#8df2ed"
BRAND_MAUVE = "#cb6ce6"
BRAND_MINT = "#d8f2d0"
BRAND_SKY = "#cae4fb"

BRAND_BLACK = "#000000"
BRAND_JET = "#272d2d"
BRAND_GREY = "#dbdddc"
BRAND_WHITE = "#ffffff"
BRAND_GREY_OLIVE = "#818585"

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Suburb Lookalike Finder",
    page_icon="🏘️",
    layout="wide"
)


st.markdown(
    f"""
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .stApp {{
        background-color: {BRAND_WHITE};
        color: {BRAND_JET};
    }}


    /* --------------------------------------------------------
       HEADINGS
    -------------------------------------------------------- */

    h1, h2, h3 {{
        color: {BRAND_JET};
        font-weight: 700;
    }}


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {{
        background: linear-gradient(
            180deg,
            {BRAND_WHITE} 0%,
            #f7f5fb 100%
        );
        border-right: 1px solid {BRAND_GREY};
    }}


    /* --------------------------------------------------------
       PRIMARY BUTTON
    -------------------------------------------------------- */

    div.stButton > button[kind="primary"] {{
        background-color: {BRAND_PRIMARY};
        border-color: {BRAND_PRIMARY};
        color: {BRAND_WHITE};
        font-weight: 600;
        border-radius: 8px;
    }}

    div.stButton > button[kind="primary"]:hover {{
        background-color: {BRAND_BLUE};
        border-color: {BRAND_BLUE};
    }}


    /* --------------------------------------------------------
       NORMAL BUTTON
    -------------------------------------------------------- */

    div.stButton > button {{
        border-radius: 8px;
    }}


    /* --------------------------------------------------------
       METRIC CARDS
    -------------------------------------------------------- */

    div[data-testid="stMetric"] {{
        background-color: #faf9fd;
        border: 1px solid {BRAND_GREY};
        padding: 15px;
        border-radius: 10px;
    }}


    /* --------------------------------------------------------
       DATAFRAME
    -------------------------------------------------------- */

    div[data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
    }}


    /* --------------------------------------------------------
       BRAND BADGE
    -------------------------------------------------------- */

    .demografy-badge {{
        display: inline-block;
        background-color: {BRAND_PRIMARY};
        color: {BRAND_WHITE};
        padding: 5px 10px;
        border-radius: 14px;
        font-size: 13px;
        font-weight: 600;
    }}


    /* --------------------------------------------------------
       SUBTITLE
    -------------------------------------------------------- */

    .demografy-subtitle {{
        color: {BRAND_GREY_OLIVE};
        font-size: 16px;
        margin-top: -10px;
        margin-bottom: 20px;
    }}


    /* --------------------------------------------------------
       SECTION DIVIDER
    -------------------------------------------------------- */

    .demografy-divider {{
        height: 3px;
        border-radius: 4px;

        background: linear-gradient(
            90deg,
            {BRAND_CYAN},
            {BRAND_PRIMARY},
            {BRAND_MAUVE}
        );

        margin-top: 5px;
        margin-bottom: 25px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

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
    

# ============================================================
# LOAD DATA ONCE
# ============================================================

@st.cache_resource
def load_app_data():

    print(
        "Loading application data..."
    )

    client = get_bigquery_client()


    # ----------------------------
    # Load suburb data
    # ----------------------------

    df = load_features(
        client
    )

    df = clean_features(
        df
    )


    # ----------------------------
    # Numeric vectors
    # ----------------------------

    X_numeric, scaler = (
        standardise_features(
            df
        )
    )


    # ----------------------------
    # Text profiles
    # ----------------------------

    profiles = (
        generate_all_profiles(
            df
        )
    )


    # ----------------------------
    # Gemini embeddings
    # ----------------------------

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


# ============================================================
# LOGIN
# ============================================================

def show_login_page():

    header_left, header_right = st.columns(
        [1.2, 5]
    )


    with header_left:

        logo_path = (
            Path(__file__).parent
            / "asset"
            / "demografy_logo.png"
        )

        logo_data = base64.b64encode(
            logo_path.read_bytes()
        ).decode("ascii")

        st.html(
            f'<img src="data:image/svg+xml;base64,{logo_data}" '
            f'width="190" alt="Demografy logo">'
        )


    with header_right:

        st.markdown(
            """
            <h1 style="margin-bottom:0;">
                Suburb Lookalike Finder
            </h1>

            <div class="demografy-subtitle">
                Hybrid demographic similarity search
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="demografy-divider"></div>',
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div style="
            max-width:500px;
            padding:30px;
            border:1px solid {BRAND_GREY};
            border-radius:12px;
            background-color:#faf9fd;
            margin-top:20px;
        ">
            <h3 style="
                margin-top:0;
                color:{BRAND_JET};
            ">
                Sign in
            </h3>
            <p style="
                color:{BRAND_GREY_OLIVE};
            ">
                Enter your Demografy user ID to continue.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


    user_id = st.text_input(
        "User ID",
        placeholder="e.g. user_001"
    )


    login_button = st.button(
        "Login",
        type="primary"
    )


    if login_button:

        if not user_id.strip():

            st.warning(
                "Please enter your user ID."
            )

        else:

            client = (
                get_bigquery_client()
            )


            user = get_user(
                client,
                user_id.strip()
            )


            if user is None:

                st.error(
                    "Invalid user ID or "
                    "inactive account."
                )

            else:

                st.session_state[
                    "logged_in"
                ] = True

                st.session_state[
                    "user"
                ] = user

                st.session_state[
                    "lookup_count"
                ] = 0


                st.rerun()


    st.stop()
    
    
if not st.session_state[
    "logged_in"
]:

    show_login_page()


def logout():

    for key in [
        "user",
        "logged_in",
    ]:

        st.session_state.pop(
            key,
            None
        )

    st.rerun()


# -------------------------
# LOGIN CHECK
# -------------------------

if not st.session_state.get(
    "logged_in",
    False
):

    show_login_page()

    st.stop()


# -------------------------
# LOGGED-IN APP
# -------------------------

#st.sidebar.write(
#    f"User: "
#    f"{st.session_state['user']['user_id']}"
#)

#st.sidebar.write(
#    f"Tier: "
#    f"{st.session_state['user']['tier']}"
#)


if st.sidebar.button(
    "Logout"
):
    logout()


# ============================================================
# CURRENT USER
# ============================================================
  
user = st.session_state[
    "user"
]


tier = user[
    "tier"
]


tier_config = (
    get_tier_config(
        tier
    )
)


st.sidebar.header(
    "Account"
)


st.sidebar.write(
    f"**User:** {user['user_id']}"
)


st.sidebar.write(
    f"**Tier:** {tier.title()}"
)

if tier == "pro":

    st.sidebar.markdown(
        """
        <span class="demografy-badge">
            PRO
        </span>
        """,
        unsafe_allow_html=True
    )
    
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


st.sidebar.metric(
    "Lookups Remaining",
    remaining
)

lookup_allowed = (
    can_lookup(
        tier,
        st.session_state[
            "lookup_count"
        ]
    )
)

warning_at = (
    tier_config[
        "warning_at"
    ]
)


lookup_count = (
    st.session_state[
        "lookup_count"
    ]
)


if (
    warning_at is not None
    and
    lookup_count >= warning_at
):

    remaining = (
        get_lookups_remaining(
            tier,
            lookup_count
        )
    )


    st.sidebar.warning(
        f"You have {remaining} "
        "lookups remaining "
        "this session."
    )

if not lookup_allowed:

    st.warning(
        "You have reached your lookup "
        "limit for this session."
    )


if tier == "free":

    st.info(
        "Upgrade your plan for "
        "additional searches and "
        "more matches."
    )
        
find_button = st.button(

    "🔍 Find Lookalikes",

    type="primary",

    disabled=not lookup_allowed
)

# ============================================================
# LOAD APPLICATION DATA
# ============================================================

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
# APP TITLE
# ============================================================

    
    
st.title(
    "🏘️ Suburb Lookalike Finder"
)

st.write(
    """
    Find Australian suburbs with similar demographic
    characteristics using numeric KPI similarity and
    Gemini text embeddings.
    """
)


# ============================================================
# SUBURB DISPLAY LABEL
# ============================================================

df["display_name"] = (
    df["sa2_name"]
    + " ("
    + df["state"]
    + ")"
)


display_lookup = dict(
    zip(
        df["sa2_code"],
        df["display_name"]
    )
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Search Settings"
)

# ============================================================
# SUBURB SELECTOR
# ============================================================

selected_sa2 = st.sidebar.selectbox(

    "Select reference suburb",

    options=df[
        "sa2_code"
    ].tolist(),

    format_func=lambda code:
        display_lookup[code]
)


reference_index = df.index[
    df["sa2_code"]
    == selected_sa2
][0]


# ============================================================
# NUMBER OF MATCHES
# ============================================================

#top_n = st.sidebar.slider(
#   "Number of matches",
#    min_value=5,
#    max_value=20,
#    value=10,
#    step=1
#)

max_matches = (
    tier_config[
        "max_matches"
    ]
)


default_matches = min(
    10,
    max_matches
)


top_n = st.sidebar.slider(

    "Number of matches",

    min_value=1,

    max_value=max_matches,

    value=default_matches,

    step=1
)

# ============================================================
# BLEND WEIGHT
# ============================================================

st.sidebar.subheader(
    "Hybrid Blend"
)


alpha = st.sidebar.slider(

    "Gemini influence (alpha)",

    min_value=0.0,

    max_value=1.0,

    value=0.5,

    step=0.05
)


st.sidebar.caption(
    f"""
    Numeric influence: {1 - alpha:.0%}

    Gemini influence: {alpha:.0%}
    """
)


# ============================================================
# PRESET SELECTOR
# ============================================================

st.sidebar.subheader(
    "Weight Preset"
)


if tier_config[
    "presets_enabled"
]:

#    st.sidebar.subheader(
#        "Weight Preset"
#    )


    preset_name = (
        st.sidebar.selectbox(

            "Choose preset",

            [
                "Balanced",
                "Family-focused",
                "Investor",
                "Lifestyle"
            ]
        )
    )


    if st.sidebar.button(
        "Apply Preset"
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
        
else:

    st.sidebar.info(
        "Weight presets are available "
        "on the Pro tier."
    )

#    apply_preset = st.sidebar.button(
#       "Apply Preset",
#       disabled=True
#   )

# ============================================================
# INITIALISE SLIDER VALUES
# ============================================================

if "current_preset" not in st.session_state:

    st.session_state[
        "current_preset"
    ] = "Balanced"


    initial_weights = (
        get_preset(
            "Balanced"
        )
    )


    for kpi in KPI_COLS:

        st.session_state[
            f"weight_{kpi}"
        ] = initial_weights[
            kpi
        ]


# ============================================================
# KPI WEIGHT SLIDERS
# ============================================================

st.sidebar.subheader(
    "KPI Weights"
)


st.sidebar.caption(
    """
    0 = ignore KPI

    1 = normal importance

    2 = double importance
    """
)


weights = {}


for kpi in KPI_COLS:

    label = KPI_LABELS[
        kpi
    ]


    weights[
        kpi
    ] = st.sidebar.slider(

        label,

        min_value=0.0,

        max_value=2.0,

        step=0.1,

        key=f"weight_{kpi}"
    )


# ============================================================
# BUILD WEIGHTED NUMERIC MATRIX
# ============================================================

X_weighted = (
    apply_feature_weights(
        X_numeric,
        weights
    )
)


# ============================================================
# BUILD HYBRID MATRIX
# ============================================================

X_hybrid = (
    fuse_vectors(
        X_weighted,
        X_text,
        alpha=alpha
    )
)


# ============================================================
# FIND MATCHES
# ============================================================

#results = find_top_n(
#    df,
#    X_hybrid,
#    reference_index,
#    n=top_n
#)

if find_button:

    results = find_top_n(
        df,
        X_hybrid,
        reference_index,
        n=top_n
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


    st.session_state[
        "lookup_count"
    ] += 1


    st.session_state[
        "latest_results"
    ] = explained_results


    st.session_state[
        "latest_reference_index"
    ] = reference_index


if (
    "latest_results"
    in st.session_state
):

    explained_results = (
        st.session_state[
            "latest_results"
        ]
    )


if explained_results is None:

    st.info(
        "Choose a reference suburb and click Find Lookalikes to see results."
    )

    st.stop()


# ============================================================
# MAIN DISPLAY
# ============================================================

reference = df.iloc[
    reference_index
]


st.subheader(
    "Reference Suburb"
)


st.success(
    f"""
    {reference['sa2_name']}
    ({reference['state']})
    """
)


# ============================================================
# SEARCH SUMMARY
# ============================================================

col1, col2, col3 = st.columns(
    3
)


with col1:

    st.metric(
        "Matches",
        top_n
    )


with col2:

    st.metric(
        "Numeric Influence",
        f"{1 - alpha:.0%}"
    )


with col3:

    st.metric(
        "Gemini Influence",
        f"{alpha:.0%}"
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

st.subheader(
    "Most Similar Suburbs"
)


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


display_results.columns = [
    "Rank",
    "Suburb",
    "State",
    "Similarity",
    "Top 3 Contributing KPIs",
    "Numeric Rank",
    "Hybrid Rank",
    "Rank Δ"
]


display_results[
    "Similarity"
] = (
    display_results[
        "Similarity"
    ]
    * 100
).round(2)


st.dataframe(
    display_results,
    width="stretch",
    hide_index=True
)

st.caption(
    """
    Rank Δ shows how much a suburb moved after
    Gemini text similarity was added.

    Positive = moved higher

    Negative = moved lower

    0 = no change
    """
)

# ============================================================
# RADAR COMPARISON
# ============================================================

st.subheader(
    "KPI Profile Comparison"
)


st.write(
    """
    Select a matched suburb to compare its
    demographic KPI profile against the
    reference suburb.
    """
)


# ============================================================
# MATCH SELECTOR
# ============================================================

match_lookup = dict(

    zip(

        explained_results[
            "sa2_code"
        ],

        (
            explained_results[
                "sa2_name"
            ]
            + " ("
            + explained_results[
                "state"
            ]
            + ")"
        )
    )
)


selected_match_code = (
    st.selectbox(

        "Compare with",

        options=
            explained_results[
                "sa2_code"
            ].tolist(),

        format_func=lambda code:
            match_lookup[code],

        key="radar_match"
    )
)


candidate_index = (
    df.index[
        df["sa2_code"]
        == selected_match_code
    ][0]
)

radar_data = (
    get_radar_data(
        df,
        reference_index,
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

# ============================================================
# CREATE RADAR CHART
# ============================================================

reference_name = (
    f"{df.iloc[reference_index]['sa2_name']} "
    f"({df.iloc[reference_index]['state']})"
)


candidate_name = (
    f"{df.iloc[candidate_index]['sa2_name']} "
    f"({df.iloc[candidate_index]['state']})"
)


fig = go.Figure()


# ============================================================
# REFERENCE SUBURB
# ============================================================

fig.add_trace(
    go.Scatterpolar(
        r=reference_scores,
        theta=labels,
        fill="toself",
        name=reference_name,

        line=dict(
            color=BRAND_PRIMARY,
            width=3
        ),

        fillcolor="rgba(154, 102, 238, 0.18)"
    )
)


# ============================================================
# MATCHED SUBURB
# ============================================================

fig.add_trace(
    go.Scatterpolar(
        r=candidate_scores,
        theta=labels,
        fill="toself",
        name=candidate_name,

        line=dict(
            color=BRAND_CYAN,
            width=3
        ),

        fillcolor="rgba(141, 242, 237, 0.20)"
    )
)


# ============================================================
# RADAR CHART SETTINGS
# ============================================================

fig.update_layout(

    polar=dict(

        bgcolor=BRAND_WHITE,

        radialaxis=dict(
            visible=True,
            range=[0, 100],
            gridcolor=BRAND_GREY,
            linecolor=BRAND_GREY
        ),

        angularaxis=dict(
            gridcolor=BRAND_GREY
        )
    ),

    paper_bgcolor=BRAND_WHITE,

    font=dict(
        color=BRAND_JET
    ),

    showlegend=True,

    height=700,

    title=(
        f"{reference_name} vs "
        f"{candidate_name}"
    )
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.caption(
    """
    KPI values are displayed as percentile ranks
    across all suburbs.

    100 = among the highest values in the dataset.

    50 = around the middle of the dataset.

    0 = among the lowest values in the dataset.
    """
)


# ============================================================
# KPI COMPARISON TABLE
# ============================================================

st.subheader(
    "KPI Comparison Details"
)


comparison_table = (
    get_kpi_comparison_table(
        df,
        reference_index,
        candidate_index
    )
)


comparison_table[
    "Reference Percentile"
] = (
    comparison_table[
        "Reference Percentile"
    ].round(1)
)


comparison_table[
    "Matched Percentile"
] = (
    comparison_table[
        "Matched Percentile"
    ].round(1)
)


comparison_table[
    "Difference"
] = (
    comparison_table[
        "Difference"
    ].round(1)
)

comparison_table[
    "Absolute Difference"
] = (
    comparison_table[
        "Absolute Difference"
    ].round(1)
)


st.dataframe(
    comparison_table,
    width="stretch",
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

# ============================================================
# DEBUG / DEVELOPMENT INFORMATION
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
        "Hybrid matrix:",
        X_hybrid.shape
    )

    st.write(
        "Reference SA2:",
        selected_sa2
    )

    st.write(
        "Alpha:",
        alpha
    )

    st.write(
        "Preset:",
        st.session_state[
            "current_preset"
        ]
    )