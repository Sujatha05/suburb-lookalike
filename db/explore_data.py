import sys
from pathlib import Path

# Add project root to path so imports work when running this script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.bigquery_client import get_bigquery_client


client = get_bigquery_client()


# -----------------------------------------
# 1. Suburb count
# -----------------------------------------

sql_suburb_count = """
SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT sa2_code) AS suburb_count
FROM `demografy.prod_tables.a_master_view`
"""

df_count = client.query(
    sql_suburb_count
).to_dataframe()

print("\nSUBURB COUNT")
print(df_count)


# -----------------------------------------
# 2. KPI ranges
# -----------------------------------------

sql_kpi_ranges = """
SELECT
    MIN(kpi_1_val) AS kpi_1_min,
    MAX(kpi_1_val) AS kpi_1_max,

    MIN(kpi_2_val) AS kpi_2_min,
    MAX(kpi_2_val) AS kpi_2_max,

    MIN(kpi_3_val) AS kpi_3_min,
    MAX(kpi_3_val) AS kpi_3_max,

    MIN(kpi_4_val) AS kpi_4_min,
    MAX(kpi_4_val) AS kpi_4_max,

    MIN(kpi_5_val) AS kpi_5_min,
    MAX(kpi_5_val) AS kpi_5_max,

    MIN(kpi_6_val) AS kpi_6_min,
    MAX(kpi_6_val) AS kpi_6_max,

    MIN(kpi_7_val) AS kpi_7_min,
    MAX(kpi_7_val) AS kpi_7_max,

    MIN(kpi_8_val) AS kpi_8_min,
    MAX(kpi_8_val) AS kpi_8_max,

    MIN(kpi_9_val) AS kpi_9_min,
    MAX(kpi_9_val) AS kpi_9_max,

    MIN(kpi_10_val) AS kpi_10_min,
    MAX(kpi_10_val) AS kpi_10_max,

    MIN(kpi_11_val) AS kpi_11_min,
    MAX(kpi_11_val) AS kpi_11_max,

    MIN(kpi_12_val) AS kpi_12_min,
    MAX(kpi_12_val) AS kpi_12_max,

    MIN(kpi_13_val) AS kpi_13_min,
    MAX(kpi_13_val) AS kpi_13_max,

    MIN(kpi_14_val) AS kpi_14_min,
    MAX(kpi_14_val) AS kpi_14_max,

    MIN(kpi_15_val) AS kpi_15_min,
    MAX(kpi_15_val) AS kpi_15_max, 
    
    MIN(kpi_16_val) AS kpi_16_min,
    MAX(kpi_16_val) AS kpi_16_max

FROM `demografy.prod_tables.a_master_view`
"""

df_ranges = client.query(
    sql_kpi_ranges
).to_dataframe()

print("\nKPI RANGES")
print(df_ranges.T)


# -----------------------------------------
# 3. NULL patterns
# -----------------------------------------

sql_nulls = """
SELECT
    COUNTIF(kpi_1_val IS NULL) AS kpi_1_nulls,
    COUNTIF(kpi_2_val IS NULL) AS kpi_2_nulls,
    COUNTIF(kpi_3_val IS NULL) AS kpi_3_nulls,
    COUNTIF(kpi_4_val IS NULL) AS kpi_4_nulls,
    COUNTIF(kpi_5_val IS NULL) AS kpi_5_nulls,
    COUNTIF(kpi_6_val IS NULL) AS kpi_6_nulls,
    COUNTIF(kpi_7_val IS NULL) AS kpi_7_nulls,
    COUNTIF(kpi_8_val IS NULL) AS kpi_8_nulls,
    COUNTIF(kpi_9_val IS NULL) AS kpi_9_nulls,
    COUNTIF(kpi_10_val IS NULL) AS kpi_10_nulls,
    COUNTIF(kpi_11_val IS NULL) AS kpi_11_nulls,
    COUNTIF(kpi_12_val IS NULL) AS kpi_12_nulls,
    COUNTIF(kpi_13_val IS NULL) AS kpi_13_nulls,
    COUNTIF(kpi_14_val IS NULL) AS kpi_14_nulls,
    COUNTIF(kpi_15_val IS NULL) AS kpi_15_nulls,
    COUNTIF(kpi_16_val IS NULL) AS kpi_16_nulls

FROM `demografy.prod_tables.a_master_view`
"""

df_nulls = client.query(
    sql_nulls
).to_dataframe()

print("\nNULL COUNTS")
print(df_nulls.T)