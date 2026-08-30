import os

from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()


def get_bigquery_client():
    return bigquery.Client(
        project=os.getenv("BIGQUERY_PROJECT")
    )