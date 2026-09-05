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
    Disadvantage_Concentration = row["kpi_11_val"]
    Retirees_and_Downsizer = row["kpi_12_val"]
    Housing_density_mix = row["kpi_13_val"]
    Premium_rental = row["kpi_14_val"]
    Investment_Potential = row["kpi_15_val"]
    Generational_Stability = row["kpi_16_val"]

    parts = []

    # Prosperity (range: 0 - 64.96)
    if prosperity >= 45:
        parts.append("high prosperity")
    elif prosperity >= 25:
        parts.append("moderate prosperity")
    else:
        parts.append("lower prosperity")


    # Diversity (range: 0 - 1)
    if diversity >= 0.70:
        parts.append("high diversity")
    elif diversity >= 0.40:
        parts.append("moderate diversity")
    else:
        parts.append("lower diversity")


    # Migration (range: 0 - 100)
    if migration >= 70:
        parts.append("high migration")
    elif migration >= 40:
        parts.append("moderate migration")
    else:
        parts.append("lower migration")


    # Education (range: 0 - 100)
    if education >= 70:
        parts.append("high education")
    elif education >= 40:
        parts.append("moderate education")
    else:
        parts.append("lower education")


    # Social Housing (range: 0 - 88.19)
    if social_housing >= 60:
        parts.append("high social housing")
    elif social_housing >= 30:
        parts.append("moderate social housing")
    else:
        parts.append("lower social housing")


    # Equity (range: 0 - 100)
    if equity >= 70:
        parts.append("high equity")
    elif equity >= 40:
        parts.append("moderate equity")
    else:
        parts.append("lower equity")


    # Rental (range: 0 - 100)
    if rental >= 70:
        parts.append("high rental")
    elif rental >= 40:
        parts.append("moderate rental")
    else:
        parts.append("lower rental")


    # Stability (range: 0 - 100)
    if stability >= 70:
        parts.append("high stability")
    elif stability >= 40:
        parts.append("moderate stability")
    else:
        parts.append("lower stability")


    # Mobility (range: 7.23 - 100)
    if mobility >= 70:
        parts.append("high mobility")
    elif mobility >= 40:
        parts.append("moderate mobility")
    else:
        parts.append("lower mobility")


    # Young Family (range: 0 - 57.14)
    if young_family >= 40:
        parts.append("high young-family presence")
    elif young_family >= 20:
        parts.append("moderate young-family presence")
    else:
        parts.append("lower young-family presence")


    # Disadvantage Concentration (range: 0 - 92.26)
    if Disadvantage_Concentration >= 65:
        parts.append("high disadvantage concentration")
    elif Disadvantage_Concentration >= 30:
        parts.append("moderate disadvantage concentration")
    else:
        parts.append("lower disadvantage concentration")


    # Retirees and Downsizers (range: 0 - 100)
    if Retirees_and_Downsizer >= 70:
        parts.append("high retiree and downsizer presence")
    elif Retirees_and_Downsizer >= 40:
        parts.append("moderate retiree and downsizer presence")
    else:
        parts.append("lower retiree and downsizer presence")


    # Housing Density Mix (range: 0 - 100)
    if Housing_density_mix >= 70:
        parts.append("high housing density mix")
    elif Housing_density_mix >= 40:
        parts.append("moderate housing density mix")
    else:
        parts.append("lower housing density mix")


    # Premium Rental (range: 0 - 88.57)
    if Premium_rental >= 60:
        parts.append("high premium rental presence")
    elif Premium_rental >= 30:
        parts.append("moderate premium rental presence")
    else:
        parts.append("lower premium rental presence")


    # Investment Potential (range: 23.31 - 73.55)
    if Investment_Potential >= 60:
        parts.append("high investment potential")
    elif Investment_Potential >= 45:
        parts.append("moderate investment potential")
    else:
        parts.append("lower investment potential")


    # Generational Stability (range: 0 - 100)
    if Generational_Stability >= 70:
        parts.append("high generational stability")
    elif Generational_Stability >= 40:
        parts.append("moderate generational stability")
    else:
        parts.append("lower generational stability")

    profile = (
        f"{row['sa2_name']} is a suburb in {row['state']} with "
        + ", ".join(parts)
        + "."
    )

    return profile

def generate_all_profiles(df):

    profiles = df.apply(
        generate_profile,
        axis=1
    )

    return profiles.tolist()