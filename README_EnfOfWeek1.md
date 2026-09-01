# 🏘️ Suburb Look-alike Finder — Demografy

An AI-powered suburb similarity engine that identifies Australian suburbs with similar demographic characteristics.

This project is being developed as part of an AI Engineering project for **Demografy**.

---

## 📌 Project Overview

The **Suburb Look-alike Finder** allows a user to select an Australian suburb and discover other suburbs with similar demographic characteristics.

The solution uses demographic KPI data from Google BigQuery and converts each suburb into a numerical feature vector. Similar suburbs can then be identified using vector similarity techniques.

The final solution is designed to use a **hybrid similarity approach**, combining:

- Numeric demographic KPI features
- Natural-language suburb profiles
- Gemini text embeddings
- Vector similarity search

Week 1 focuses on establishing a reliable end-to-end foundation before introducing the full hybrid similarity model.

---

# 🚀 Week 1 — MVP Foundation

The objective of Week 1 is to build and validate the core data and similarity pipeline.

### Week 1 Deliverables

- [x] Set up GitHub repository and project structure
- [x] Configure `.gitignore` for secrets and generated files
- [x] Configure read-only Google BigQuery connection
- [x] Explore and profile the source demographic dataset
- [x] Analyse KPI ranges and NULL patterns
- [x] Validate suburb identifiers and duplicate suburb names
- [x] Build numeric feature pipeline
- [x] Handle missing KPI values
- [x] Standardise demographic KPIs
- [x] Build NumPy brute-force cosine similarity baseline
- [x] Return Top-N similar suburbs
- [x] Exclude the selected suburb from its own recommendations
- [x] Draft deterministic suburb text-profile generator
- [x] Test Gemini embedding API access
- [ ] Scaffold Streamlit user interface

---

# 🏗️ Current Architecture

```text
Google BigQuery
      │
      ▼
a_master_view
      │
      ▼
Load 10 demographic KPIs
      │
      ▼
Pandas DataFrame
      │
      ▼
NULL Handling
Median Imputation
      │
      ▼
StandardScaler
      │
      ▼
Numeric Feature Matrix
(2473 × 10)
      │
      ▼
NumPy Cosine Similarity
      │
      ▼
Rank All Suburbs
      │
      ▼
Exclude Selected Suburb
      │
      ▼
Top-N Similar Suburbs
```

The Week 1 implementation deliberately uses a NumPy brute-force similarity search so that correctness can be validated before introducing more advanced vector-search technologies.

---

# 📊 Dataset

The application currently reads demographic data from the production BigQuery view:

```text
demografy.prod_tables.a_master_view
```

The Week 1 exploration identified:

| Metric | Result |
|---|---:|
| Total rows | 2,473 |
| Distinct SA2 codes | 2,473 |
| Numeric KPI features | 10 |
| Duplicate suburb names | 0 |
| Numeric feature matrix | 2,473 × 10 |

`sa2_code` is retained as the internal suburb identifier.

---

# 📈 KPI Data Exploration

The following ranges were observed during Week 1 data profiling:

| KPI | Minimum | Maximum | NULLs |
|---|---:|---:|---:|
| KPI 1 | 0.00 | 64.96 | 120 |
| KPI 2 | 0.00 | 1.00 | 45 |
| KPI 3 | 0.00 | 100.00 | 43 |
| KPI 4 | 0.00 | 100.00 | 45 |
| KPI 5 | 0.00 | 88.19 | 77 |
| KPI 6 | 0.00 | 100.00 | 81 |
| KPI 7 | 0.00 | 100.00 | 113 |
| KPI 8 | 0.00 | 100.00 | 44 |
| KPI 9 | 7.23 | 100.00 | 120 |
| KPI 10 | 0.00 | 57.14 | 41 |

This analysis demonstrated that the KPIs operate on significantly different numerical scales.

For example, KPI 2 operates between `0–1`, while several other KPIs operate between `0–100`.

Therefore, raw KPI values should not be directly compared when calculating suburb similarity.

---

# 🧹 Data Preprocessing

## Missing Values

NULL values are handled using **median imputation**.

For each KPI:

```python
df[KPI_COLS] = df[KPI_COLS].fillna(
    df[KPI_COLS].median()
)
```

Each missing value is therefore replaced by the median of its corresponding KPI rather than using a single replacement value across all features.

After preprocessing:

```text
NULL values across all 10 KPIs = 0
```

---

## Feature Standardisation

The project uses:

```python
sklearn.preprocessing.StandardScaler
```

to standardise the 10 KPI features.

This transforms features onto comparable statistical scales so that KPIs with larger numeric ranges do not dominate the similarity calculation.

The resulting numeric feature matrix has the shape:

```text
(2473, 10)
```

representing:

```text
2,473 suburbs × 10 demographic KPIs
```

---

# 🔎 Similarity Engine

Week 1 implements a **NumPy brute-force cosine similarity baseline**.

For a selected suburb:

1. Retrieve its standardised 10-dimensional feature vector.
2. Compare the vector against every suburb in the dataset.
3. Calculate cosine similarity scores.
4. Rank suburbs from highest to lowest similarity.
5. Exclude the selected suburb itself.
6. Return the Top-N most similar suburbs.

Conceptually:

```text
Selected Suburb
      │
      ▼
10-Dimensional Vector
      │
      ▼
Compare Against 2,473 Suburbs
      │
      ▼
Cosine Similarity
      │
      ▼
Sort Scores
      │
      ▼
Remove Self-Match
      │
      ▼
Top-N Look-alike Suburbs
```

This provides a simple and transparent baseline against which future search implementations can be validated.

---

# 🤖 Gemini Embeddings

The project also begins development of the semantic component of the similarity engine.

A deterministic text-profile generator converts demographic characteristics into a natural-language suburb description.

Example concept:

```text
Karabar is a suburb in New South Wales with lower prosperity,
high cultural diversity, strong rental access and lower
young-family presence.
```

This profile can then be converted into a vector embedding using the Gemini Embedding API.

The current implementation uses:

```text
gemini-embedding-001
```

with:

```text
output_dimensionality = 768
```

The resulting 768-dimensional vector will later be combined with the numeric KPI representation to create a **hybrid suburb vector**.

---

# 🧠 Planned Hybrid Similarity

The final similarity engine is designed to combine two different representations.

```text
                  Suburb
                     │
             ┌───────┴───────┐
             │               │
             ▼               ▼
       Numeric KPIs      Text Profile
             │               │
             ▼               ▼
      StandardScaler       Gemini
             │           Embedding
             ▼               ▼
       10-D Vector       768-D Vector
             │               │
             └───────┬───────┘
                     ▼
               Hybrid Vector
                     │
                     ▼
              Similarity Search
                     │
                     ▼
              Look-alike Suburbs
```

The numeric component captures measurable demographic characteristics, while the embedding component provides a semantic representation of the suburb profile.

---

# 📁 Project Structure

```text
suburb-lookalike/
│
├── app.py
│
├── db/
│   ├── __init__.py
│   ├── bigquery_client.py
│   └── explore_data.py
│
├── engine/
│   ├── __init__.py
│   ├── features.py
│   ├── profiles.py
│   ├── similarity.py
│   └── text_embed.py
│
├── docs/
│   └── data_profile.md
│
├── test_bigquery.py
├── test_features.py
├── test_profile.py
├── test_embedding.py
├── test_similarity.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# ⚙️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| Google BigQuery | Demographic data source |
| Pandas | Data loading and transformation |
| NumPy | Vector operations and similarity calculations |
| Scikit-learn | KPI standardisation |
| Gemini API | Text embeddings |
| Streamlit | Application user interface |
| Git / GitHub | Version control and collaboration |

---

# 📦 Installation

Clone the repository:

```bash
git clone <https://github.com/Sujatha05/suburb-lookalike>
```

Move into the project directory:

```bash
cd suburb-lookalike
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Create a `.env` file in the project root:

```env
GOOGLE_APPLICATION_CREDENTIALS=<service-account.json>
BIGQUERY_PROJECT=demografy
GEMINI_API_KEY=<gemini_api_key>
```

> **Important:** Credentials, API keys and service-account JSON files must never be committed to GitHub.

The project `.gitignore` should exclude files such as:

```text
.env
*.json
service-account*.json
.venv/
__pycache__/
cache/
*.npy
```

---

# 🧪 Running Week 1 Tests

## Test BigQuery Connection

```bash
python test_1_bigquery.py
```

## Explore Source Data

Run the exploration module from the project root:

```bash
python -m db.explore_data
```

## Test Feature Pipeline

```bash
python test_2_features.py
```

Expected result:

```text
NULLS AFTER CLEANING
kpi_1_val     0
...
kpi_10_val    0

NUMERIC MATRIX SHAPE
(2473, 10)
```

## Test Numeric Similarity

```bash
python test_5_similarity.py
```

This should return the Top-N suburbs most similar to the selected reference suburb.

## Test Text Profile

```bash
python test_3_profile.py
```

## Test Gemini Embedding

```bash
python test_4_embedding.py
```

The embedding test should successfully convert a generated suburb profile into a vector representation.

---

# 🖥️ Streamlit UI

The Week 1 user-interface scaffold will provide:

- Searchable suburb selection
- State-aware suburb labels
- Top-N look-alike results
- Similarity scores
- Simple results table

The initial UI intentionally remains simple while the underlying similarity engine is validated.

---

# 🔮 Next Steps

Following completion of the Week 1 foundation, development will progress toward:

- Complete deterministic suburb profile generation
- Generate and cache embeddings for all suburbs
- L2-normalise embedding vectors
- Develop numeric + semantic hybrid similarity
- Introduce configurable alpha weighting
- Compare numeric-only vs hybrid recommendations
- Build the Streamlit user experience
- Add richer suburb comparison and explanation features
- Introduce FAISS for scalable vector search
- Evaluate recommendation quality

---

# 🎯 Week 1 Outcome

At the end of Week 1, the project has established a working pipeline:

```text
BigQuery
   ↓
Demographic Data
   ↓
Data Quality Checks
   ↓
NULL Handling
   ↓
Feature Standardisation
   ↓
Numeric Vector Representation
   ↓
Cosine Similarity
   ↓
Top-N Look-alike Suburbs
```

In parallel, the Gemini embedding integration establishes the foundation for the next stage:

```text
Demographic KPIs
      ↓
Deterministic Text Profile
      ↓
Gemini Embedding
      ↓
Semantic Vector
      ↓
Future Hybrid Similarity Engine
```

This provides a validated baseline on which the remaining MVP functionality can be built.

---

## Status

**Week 1:** 🟢 Core pipeline operational  
**Current focus:** Streamlit UI and hybrid similarity preparation
