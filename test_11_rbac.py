from db.bigquery_client import (
    get_bigquery_client
)

from auth.users import (
    get_user
)

from auth.rbac import (
    get_tier_config
)


client = get_bigquery_client()


user_id = input(
    "Enter user ID: "
)


user = get_user(
    client,
    user_id
)


if user is None:

    print(
        "\nUser not found "
        "or account is inactive."
    )

else:

    print(
        "\nUSER FOUND"
    )

    print(
        "User ID:",
        user["user_id"]
    )

    print(
        "Email:",
        user["email"]
    )

    print(
        "Tier:",
        user["tier"]
    )


    config = (
        get_tier_config(
            user["tier"]
        )
    )


    print(
        "\nTIER SETTINGS"
    )

    print(
        "Lookup limit:",
        config["lookup_limit"]
    )

    print(
        "Max matches:",
        config["max_matches"]
    )

    print(
        "Presets enabled:",
        config[
            "presets_enabled"
        ]
    )