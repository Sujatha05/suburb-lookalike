# Suburb Lookalike

An AI-powered suburb similarity engine that combines demographic KPIs
with semantic text embeddings.

## Architecture

BigQuery
↓
Feature Engineering
↓
Numeric Representation
↓
Text Profile Generation
↓
Gemini Embeddings
↓
Vector Fusion
↓
FAISS
↓
Nearest Neighbours
↓
Streamlit

## Features

- BigQuery demographic data
- KPI cleaning and normalisation
- Natural-language suburb profiles
- Gemini text embeddings
- Numeric + semantic vector fusion
- FAISS similarity search
- Explainable similarity scores
- RBAC
- Evaluation using precision@k and recall@k

## Tech Stack

- Python
- Streamlit
- Google BigQuery
- Gemini Embeddings
- FAISS
- NumPy
- Pandas
- Scikit-learn

## Project Structure

...

## Setup

Create a virtual environment:

python -m venv .venv

Activate it:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Create `.env`:

GCP_PROJECT_ID=...
GEMINI_API_KEY=...

Run:

streamlit run app.py