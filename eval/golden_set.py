GOLDEN_SET = {

    # ========================================================
    # 1. CARLTON
    # ========================================================

    "206041117": {

        "sa2_name": "Carlton",
        "state": "Victoria",

        "profile":
            "Inner-city, diverse, high-rental, "
            "low-young-family areas",

        "expected_neighbours": [
            "206051513",  # St Kilda - Central
            "206041506",  # North Melbourne
            "206041503",  # Melbourne CBD - East
            "206041505",  # Melbourne CBD - West
            "206071141",  # Collingwood
        ],

        "should_not_match": [
        ],

        "k": 10,

        "validation": {
            "minimum_expected_in_top_k": 3
        }
    },


    # ========================================================
    # 2. TOORAK
    # ========================================================

    "206061138": {

        "sa2_name": "Toorak",
        "state": "Victoria",

        "profile":
            "High-prosperity, high-equity, "
            "established areas",

        "expected_neighbours": [
            "503011031",  # Claremont (WA)
            "206051128",  # Albert Park
            "206061135",  # Armadale
            "118011650",  # Double Bay - Darling Point
            "118011347",  # Woollahra
        ],

        "should_not_match": [
        ],

        "k": 10
    },


    # ========================================================
    # 3. LOGAN CENTRAL
    # ========================================================

    "311061331": {

        "sa2_name": "Logan Central",
        "state": "Queensland",

        "profile":
            "High social-housing, "
            "lower-prosperity areas",

        "expected_neighbours": [
            "311061336",  # Woodridge
            "311061330",  # Kingston (Qld)
            "402021030",  # Elizabeth
            "310011274",  # Inala - Richlands
            "116031313",  # Bidwill - Hebersham - Emerton
        ],

        "should_not_match": [
        ],

        "k": 10,

        "validation": {
            "minimum_precision": 0.5
        }
    },


    # ========================================================
    # 4. CASTLE HILL - NORTH
    # ========================================================

    "115011555": {

        "sa2_name": "Castle Hill - North",
        "state": "New South Wales",

        "profile":
            "High young-family, high-learning, "
            "high-equity areas",

        "expected_neighbours": [
            "115011291",  # Baulkham Hills (West) - Bella Vista
            "115011557",  # Castle Hill - West
            "115011558",  # Cherrybrook
            "115011621",  # Kellyville - East
            "115011296",  # West Pennant Hills
        ],

        "should_not_match": [
        ],

        "k": 10
    },


    # ========================================================
    # 5. PETERMANN - SIMPSON
    # ========================================================

    "702011050": {

        "sa2_name": "Petermann - Simpson",
        "state": "Northern Territory",

        "profile":
            "Remote, low-density, "
            "distinct-profile areas",

        "expected_neighbours": [
            "511031283",  # Leinster - Leonora
            "510021267",  # East Pilbara
            "702011052",  # Sandover - Plenty
            "315021404",  # Carpentaria
            "511041290",  # Meekatharra
        ],

        "should_not_match": [
            "206041503",  # Melbourne CBD - East
            "206041505",  # Melbourne CBD - West
            "305011106",  # Fortitude Valley
            "305011105",  # Brisbane City
        ],

        "k": 10,

        "validation": {
            "should_not_match_inner_city": True
        }
    },
    
    
    # ========================================================
    # 6. TAYLOR
    # Investor / Premium Rental
    # ========================================================

    "801041121": {

        "sa2_name": "Taylor",
        "state": "Australian Capital Territory",

        "profile":
            "High investment-potential, premium-rental, "
            "young-growth area",

        "expected_neighbours": [
            "801041122",  # Throsby
            "801041120",  # Moncrieff
            "801101136",  # Denman Prospect
            "212031555",  # Clyde North - North
            "801011143",  # Strathnairn
        ],

        "should_not_match": [
        ],

        "k": 10
    },


    # ========================================================
    # 7. WHEELERS HILL
    # Retiree / Downsizer
    # ========================================================

    "212051327": {

        "sa2_name": "Wheelers Hill",
        "state": "Victoria",

        "profile":
            "Established, high-equity, high-resident-anchor, "
            "retiree and downsizer area",

        "expected_neighbours": [
            "211041273",  # Vermont South
            "207021160",  # Templestowe Lower
            "207021156",  # Bulleen
            "212051321",  # Glen Waverley - East
            "207021159",  # Templestowe
        ],

        "should_not_match": [
        ],

        "k": 10
    },


    # ========================================================
    # 8. ST ALBANS - NORTH
    # Migration / Diversity Hub
    # ========================================================

    "213011334": {

        "sa2_name": "St Albans - North",
        "state": "Victoria",

        "profile":
            "Highly diverse, strong migration-footprint, "
            "high-rental-access area",

        "expected_neighbours": [
            "213011335",  # St Albans - South
            "209041529",  # Lalor - East
            "212041460",  # Noble Park - West
            "212041317",  # Springvale
            "213011337",  # Sunshine North
        ],

        "should_not_match": [
        ],

        "k": 10
    },


    # ========================================================
    # 9. LOFTUS - YARRAWARRAH
    # Stable Owner-Occupier
    # ========================================================

    "128021608": {

        "sa2_name": "Loftus - Yarrawarrah",
        "state": "New South Wales",

        "profile":
            "Stable owner-occupier, high-equity, "
            "strong resident-anchor area",

        "expected_neighbours": [
            "128021607",  # Engadine
            "107041147",  # Helensburgh
            "121021404",  # Berowra - Brooklyn - Cowan
            "128021533",  # Heathcote - Waterfall
            "111011213",  # Valentine - Eleebana
        ],

        "should_not_match": [
        ],

        "k": 10
    },


    # ========================================================
    # 10. REDBANK PLAINS
    # High-Mobility Rental Area
    # ========================================================

    "310041302": {

        "sa2_name": "Redbank Plains",
        "state": "Queensland",

        "profile":
            "High-rental-access, high-household-mobility, "
            "young-family growth area",

        "expected_neighbours": [
            "311031312",  # Browns Plains
            "311031317",  # Marsden
            "310041299",  # Collingwood Park - Redbank
            "311031314",  # Crestmead
            "311031311",  # Boronia Heights - Park Ridge
        ],

        "should_not_match": [
        ],

        "k": 10
    }
}