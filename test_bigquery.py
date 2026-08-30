from db.bigquery_client import get_bigquery_client

client = get_bigquery_client()


sql = """
SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT sa2_code) AS suburb_count
FROM `demografy.prod_tables.a_master_view`
"""


df = client.query(sql).to_dataframe()


print(df)