# Suburb Lookalike Finder

AI-assisted Streamlit application for finding Australian SA2 areas with similar demographic profiles. It combines **16 numeric KPIs** with **Gemini text embeddings**, applies configurable KPI weights, fuses both representations, and uses **FAISS** to return ranked lookalikes.

## Features

- BigQuery SA2 data source and 16 demographic KPIs
- Standardised numeric features and configurable KPI weighting
- Deterministic suburb text profiles and cached Gemini embeddings
- Hybrid numeric + semantic representation with default **alpha = 0.20**
- FAISS Top-N similarity search
- Top contributing KPI explanations and numeric-vs-hybrid rank delta
- Plotly radar comparison against up to three matched suburbs
- Free / Basic / Pro lookup controls
- Golden Set evaluation with Precision@10, Recall@10 and stability checks
- Optional clustering work under `stretch/`

## Technology

| Area | Technology |
|---|---|
| UI | Streamlit |
| Data | Google BigQuery, pandas, NumPy |
| Preprocessing | scikit-learn |
| Embeddings | Google Gemini (`google-genai`) |
| Search | FAISS |
| Charts | Plotly |
| Configuration | python-dotenv |

> `requirements.txt` currently also contains LangChain/LangSmith packages. The current `app.py` performs its main lookup directly through FAISS; `chains/` and `observability/` are retained separately.

## Project Structure

```text
suburb-lookalike/
├── .streamlit/config.toml
├── asset/demografy_logo.png
├── auth/
│   ├── rbac.py
│   └── users.py
├── cache/
│   ├── profiles.parquet
│   └── text_embeddings.npy
├── chains/lookup_chain.py
├── db/
│   ├── bigquery_client.py
│   ├── customers.py
│   └── explore_data.py
├── docs/data_profile.md
├── engine/
│   ├── explain.py
│   ├── features.py
│   ├── fusion.py
│   ├── index.py
│   ├── profiles.py
│   ├── similarity.py
│   ├── text_embed.py
│   └── weights.py
├── eval/
│   ├── golden_set.py
│   ├── golden_set.json
│   ├── metrics.py
│   ├── run_eval.py
│   └── *_results.csv
├── observability/tracing.py
├── stretch/clustering.py
├── style/
│   ├── demografy.css
│   └── styles.css
├── .env.example
├── app.py
├── build_embeddings.py
├── config.py
└── requirements.txt
```

Do not commit `.env`, virtual environments, `__pycache__`, API keys, or service-account credentials.

## Architecture

```text
                         Google BigQuery
                         SA2 + 16 KPIs
                               |
                               v
                        Load / Clean Data
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
          NUMERIC KPI PATH             SEMANTIC PATH
          StandardScaler               Deterministic
          16 KPI values                text profile
          KPI weighting                     |
                 |                           v
                 |                    Gemini embedding
                 |                           |
                 |                    cached .npy vectors
                 |                           |
                 +-------------+-------------+
                               |
                               v
                         HYBRID FUSION
                           alpha=0.20
                               |
                               v
                             FAISS
                               |
                               v
                       Top-N Lookalikes
                               |
               +---------------+---------------+
               v               v               v
          Ranked table     KPI explanation   Radar chart
                           + rank delta       (up to 3)
```

### Runtime Flow

1. User signs in with a Demografy user ID.
2. Account tier determines lookup limits and available controls.
3. SA2 data is loaded from BigQuery and cleaned.
4. The 16 KPIs are standardised.
5. Deterministic text profiles are generated.
6. Gemini embeddings are loaded from cache or generated when required.
7. User-selected KPI weights are applied.
8. Numeric and semantic vectors are fused.
9. FAISS searches the current hybrid representation.
10. The reference SA2 itself is excluded.
11. Top-N matches are ranked, explained and displayed.

## Why Hybrid Fusion?

The two representations provide complementary information.

**Numeric KPIs** are the primary signal. They provide measurable demographic characteristics and make the result explainable through individual KPI differences.

**Gemini embeddings** provide a semantic representation of the deterministic suburb profile. They are used as a supporting signal rather than replacing the structured KPIs.

Conceptually:

```text
weighted numeric vector -> L2 normalise -> (1-alpha) --+
                                                        +-> concatenate -> hybrid vector
Gemini text vector -----> L2 normalise -> alpha --------+
```

`alpha` is best interpreted as a **blend parameter**, not a literal percentage contribution to the final cosine similarity.

### Why alpha = 0.20?

Golden Set evaluation produced:

| Configuration | Avg Precision@10 | Avg Recall@10 |
|---|---:|---:|
| Numeric only | 0.47 | 0.94 |
| Hybrid, alpha=0.20 | **0.48** | **0.96** |
| Text only | 0.10 | 0.20 |

Alpha 0.20 was retained because it improved the hybrid evaluation while keeping the structured demographic signal dominant. Alpha 0.40 achieved the same aggregate Precision@10 and Recall@10 in the completed sweep, so the lower semantic weighting was selected as the default.

## 16 KPIs

1. Prosperity
2. Diversity
3. Migration Footprint
4. Learning Level
5. Social Housing
6. Resident Equity
7. Rental Access
8. Resident Anchor
9. Household Mobility
10. Young Family
11. Disadvantage Concentration
12. Retirees and Downsizer
13. Housing Density Mix
14. Premium Rental
15. Investment Potential
16. Generational Stability

## Setup

### 1. Open the project

```powershell
cd C:\Projects\AI-Engineering-TDAI-GitHubPush\AI-Projects\suburb-lookalike
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Configure the values expected by your local `config.py`, `db/bigquery_client.py` and `engine/text_embed.py`. Typical values include Gemini and Google Cloud credentials.

```env
GEMINI_API_KEY=your_key
GOOGLE_APPLICATION_CREDENTIALS=path_to_service_account_json
SHOW_DEVELOPMENT_INFO=false
```

Do not commit `.env` or credential files.

### 5. Verify BigQuery

The repository includes `test_1_bigquery.py` for checking connectivity:

```powershell
python test_1_bigquery.py
```

### 6. Build/load embeddings

Cached files are stored under `cache/`:

```text
cache/profiles.parquet
cache/text_embeddings.npy
```

If the cache needs to be generated:

```powershell
python build_embeddings.py
```

Caching avoids regenerating Gemini embeddings on every lookup.

### 7. Run Streamlit

```powershell
streamlit run app.py
```

## Golden Set Evaluation

The `eval/` folder contains the evaluation workflow. The final Golden Set has **10 reference SA2s** and **5 expected neighbours each**, giving **50 expected-neighbour relationships**.

**Precision@10** asks: of the ten returned suburbs, how many are expected?

```text
Precision@10 = expected matches found in Top 10 / 10
```

**Recall@10** asks: of the expected neighbours, how many were found?

```text
Recall@10 = expected matches found in Top 10 / expected neighbours
```

The evaluation folder also contains numeric, hybrid, text, alpha-sweep and stability result files.

## Explainability

Results include similarity, top three contributing KPIs, numeric rank, hybrid rank and rank delta. The KPI Profile Comparison can display the reference suburb and **up to three matched suburbs** on one Plotly radar chart, with a separate one-to-one detailed KPI table.

## Edge Cases

- **Null KPIs:** clean missing values before standardisation/similarity.
- **Duplicate names:** use `sa2_code` internally; display suburb name + state to users.
- **Remote outliers:** validate rather than automatically remove them. The Golden Set includes a remote/distinct-profile case.
- **Gemini rate limits:** reuse cached embeddings and use retry/backoff when generating missing embeddings.

## Demo GIF

Save the final recording as:

```text
asset/demo.gif
```

It will display here:

![Suburb Lookalike Finder Demo](asset/demo.gif)

### Suggested 30-45 second recording

1. Open the application and sign in with a demo user.
2. Select a reference suburb.
3. Show number of matches, alpha and a KPI preset.
4. Click **Find Lookalikes**.
5. Show the ranked table and similarity scores.
6. Scroll to **KPI Profile Comparison**.
7. Compare the reference with three matched suburbs.
8. Finish on the radar chart / detailed KPI comparison.

Avoid showing `.env`, API keys or service-account credentials in the recording.

## Optional Stretch: K-Means Archetypes

`stretch/clustering.py` can be used for an optional clustering enhancement after the core system is stable. K-means can group the standardised 16-KPI profiles into broader demographic archetypes.

This is deliberately separate from FAISS:

- **FAISS:** Which individual suburbs are most similar?
- **K-means:** What broad demographic type does this suburb belong to?

Cluster names should be assigned only after inspecting the actual KPI profile of each cluster.

## Tests

The repository contains staged tests for BigQuery, features, profiles, embeddings, similarity, fusion, weighting, explainability, RBAC and FAISS:

```text
test_1_bigquery.py
test_2_features.py
test_3_profile.py
test_4_embedding.py
test_5_similarity.py
test_6_embedding.py
test_7_fusion.py
test_8_hybrid_similarity.py
test_9_weights.py
test_10_explain.py
test_11_rbac.py
test_12_faiss.py
```

## Security

Before pushing the repository, confirm `.gitignore` excludes:

```gitignore
.env
.venv/
env/
__pycache__/
*.pyc
BigQuery_Service_Account_JSON.json
```

If a real credential has ever been committed to a remote repository, removing it from the current folder is not enough; revoke/rotate the credential.

## Summary

The Suburb Lookalike Finder combines structured demographic KPIs with a complementary Gemini representation. The numeric KPIs remain the dominant signal, the semantic representation refines ranking, and FAISS provides efficient interactive retrieval. Golden Set evaluation supports the default **alpha=0.20** configuration.
