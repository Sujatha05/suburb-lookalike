from google.cloud import bigquery


# ============================================================
# USER LOOKUP
# ============================================================

def get_user(client, user_id):

    """
    Look up an active user from the
    Demografy customer reference table.
    """

    sql = """
        SELECT
            user_id,
            email,
            tier,
            is_active
        FROM
            `demografy.ref_tables.dev_customers`
        WHERE
            user_id = @user_id
            AND is_active = TRUE
        LIMIT 1
    """


    job_config = bigquery.QueryJobConfig(

        query_parameters=[
            bigquery.ScalarQueryParameter(
                "user_id",
                "STRING",
                user_id
            )
        ]
    )


    result = (
        client.query(
            sql,
            job_config=job_config
        )
        .to_dataframe()
    )


    if result.empty:
        return None


    user = result.iloc[0]


    return {
        "user_id":
            user["user_id"],

        "email":
            user["email"],

        "tier":
            str(
                user["tier"]
            ).lower(),

        "is_active":
            bool(
                user["is_active"]
            )
    }