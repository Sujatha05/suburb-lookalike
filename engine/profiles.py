def generate_profile(row):

    prosperity = row["kpi_1_val"]
    diversity = row["kpi_2_val"]
    migration = row["kpi_3_val"]
    education = row["kpi_4_val"]
    social_housing = row["kpi_5_val"]
    equity = row["kpi_6_val"]
    rental = row["kpi_7_val"]
    stability = row["kpi_8_val"]
    mobility = row["kpi_9_val"]
    young_family = row["kpi_10_val"]

    parts = []

    # Prosperity
    if prosperity >= 45:
        parts.append("high prosperity")
    elif prosperity >= 25:
        parts.append("moderate prosperity")
    else:
        parts.append("lower prosperity")

    # Diversity
    if diversity >= 0.70:
        parts.append("high cultural diversity")
    elif diversity >= 0.45:
        parts.append("moderate cultural diversity")
    else:
        parts.append("low cultural diversity")

    # Education
    if education >= 70:
        parts.append("high educational attainment")

    # Rental market
    if rental >= 60:
        parts.append("strong rental access")
    elif rental <= 30:
        parts.append("limited rental access")

    # Social housing
    if social_housing >= 20:
        parts.append("higher social housing presence")

    # Young families
    if young_family >= 25:
        parts.append("strong young-family presence")
    else:
        parts.append("lower young-family presence")

    profile = (
        f"{row['sa2_name']} is a suburb in {row['state']} with "
        + ", ".join(parts)
        + "."
    )

    return profile