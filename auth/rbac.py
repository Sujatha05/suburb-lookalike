# ============================================================
# TIER CONFIGURATION
# ============================================================

TIER_CONFIG = {

    "free": {
        "lookup_limit": 5,
        "max_matches": 5,
        "warning_at": None,
        "presets_enabled": False
    },

    "basic": {
        "lookup_limit": 20,
        "max_matches": 10,
        "warning_at": 15,
        "presets_enabled": False
    },

    "pro": {
        "lookup_limit": 50,
        "max_matches": 25,
        "warning_at": 45,
        "presets_enabled": True
    }
}


# ============================================================
# GET TIER SETTINGS
# ============================================================

def get_tier_config(tier):

    tier = tier.lower()

    if tier not in TIER_CONFIG:

        raise ValueError(
            f"Unknown user tier: {tier}"
        )

    return TIER_CONFIG[
        tier
    ].copy()


# ============================================================
# CHECK LOOKUP AVAILABILITY
# ============================================================

def can_lookup(
    tier,
    lookup_count
):

    config = get_tier_config(
        tier
    )

    return (
        lookup_count
        <
        config["lookup_limit"]
    )


# ============================================================
# LOOKUPS REMAINING
# ============================================================

def get_lookups_remaining(
    tier,
    lookup_count
):

    config = get_tier_config(
        tier
    )

    remaining = (
        config["lookup_limit"]
        - lookup_count
    )

    return max(
        0,
        remaining
    )