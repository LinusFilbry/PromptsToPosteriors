DUTCH_BOOK_SYSTEM_PROMPT = "You are an expert predictor of future events."


def build_user_prompt(question: str) -> str:
    return (
        f'With what probability is the answer to the following question "Yes"? \n\n {question} '
        f'\n\nReply with exactly one decimal number between 0 and 1, which represents the probability.'
    )


# --- Question data -----------------------------------------------------
# Positive/q1 questions below are sourced verbatim (or near-verbatim) from the
# ForecastBench question set:
# https://github.com/forecastingresearch/forecastbench-datasets/blob/main/datasets/question_sets/2026-06-21-llm.json

SCENARIO_1_PAIRS = [
    {
        "name": "democrat_2028",
        "positive": "Will a Democrat win the 2028 US presidential election?",
        "negation": "Will a candidate who is NOT a Democrat win the 2028 US presidential election?",
    },
    {
        "name": "prc_taiwan_blockade",
        "positive": "Will the People's Republic of China blockade or quarantine Taiwan before September 1, 2026?",
        "negation": "Will the People's Republic of China NOT blockade or quarantine Taiwan before September 1, 2026?",
    },
    {
        "name": "us_attack_cuba",
        "positive": "Will the United States attack Cuba before 2027?",
        "negation": "Will the United States NOT attack Cuba before 2027?",
    },
    {
        "name": "cuba_govt_lose_power",
        "positive": "Will the Cuban government lose power before 2027?",
        "negation": "Will the Cuban government still hold power on January 1, 2027?",
    },
    {
        "name": "musk_tesla_ceo",
        "positive": "Will Elon Musk be removed as the Chief Executive Officer of Tesla before 2027?",
        "negation": "Will Elon Musk still be the Chief Executive Officer of Tesla on January 1, 2027?",
    },
    {
        "name": "tesla_cybertruck_discontinued",
        "positive": "Will Tesla discontinue the Cybertruck before January 1, 2027?",
        "negation": "Will Tesla NOT discontinue the Cybertruck before January 1, 2027?",
    },
    {
        "name": "netanyahu_ceases_pm",
        "positive": "Will Benjamin Netanyahu cease to be Prime Minister of Israel during 2026?",
        "negation": "Will Benjamin Netanyahu remain Prime Minister of Israel throughout all of 2026?",
    },
    {
        "name": "starmer_ceases_pm",
        "positive": "Will Keir Starmer cease to be Prime Minister of the UK during 2026?",
        "negation": "Will Keir Starmer remain Prime Minister of the UK throughout all of 2026?",
    },
    {
        "name": "peltola_alaska_senate",
        "positive": "Will Mary Peltola win the 2026 US Senate election in Alaska?",
        "negation": "Will a candidate who is NOT Mary Peltola win the 2026 US Senate election in Alaska?",
    },
    {
        "name": "bitcoin_120k",
        "positive": "Will Bitcoin trade above $120,000 before August 1, 2026?",
        "negation": "Will Bitcoin remain at or below $120,000 through August 1, 2026?",
    },
]

# Ordered loose (superset) -> strict (subset). Probabilities should be non-increasing along each series.
SCENARIO_2_SERIES = {
    "donbass_control": {
        "template": "Will Russia control at least {x}% of Donbass at the end of October 2026?",
        "values": [70, 75, 80, 85, 90, 95],
    },
    "nvidia_price": {
        "template": "Will Nvidia's stock price close below ${x} on any day in 2026?",
        "values": [150, 140, 130, 120, 110, 100],
    },
    "sp500_crash": {
        "template": "Will the S&P 500 decline by more than {x}% in any 5-trading-day period before 2027?",
        "values": [5, 7, 10, 12, 15, 20],
    },
    "case_shiller_index": {
        "template": "Will the S&P CoreLogic Case-Shiller U.S. National Home Price Index exceed {x} by July 2027?",
        "values": [320, 330, 340, 345, 350, 355],
    },
    "bitcoin_above": {
        "template": "Will Bitcoin's price exceed ${x} at any point before 2027?",
        "values": [80000, 90000, 100000, 110000, 115000, 120000],
    },
    "anthropic_valuation": {
        "template": "Will Anthropic's valuation reach at least ${x} trillion by December 31, 2026?",
        "values": [1.0, 1.25, 1.5, 1.6, 1.7, 1.75],
    },
    "mlb_attendance": {
        "template": "Will total MLB regular-season attendance in 2026 exceed {x} million?",
        "values": [69, 70, 71, 71.5, 72, 72.5],
    },
    "china_youth_unemployment": {
        "template": "Will China's official youth unemployment rate fall below {x}% in any monthly report released by 31 July 2026?",
        "values": [18, 17, 16, 15.5, 15, 14],
    },
    "reform_uk_polling": {
        "template": "Will Reform UK's poll average fall below {x}% for at least 14 consecutive days in 2026?",
        "values": [24, 22, 21, 20, 18, 15],
    },
    "spiderman_rating": {
        "template": "Will Spider-Man: Beyond the Spider-Verse get a {x}% or higher on Rotten Tomatoes?",
        "values": [90, 92, 93, 95, 96, 98],
    },
}

SCENARIO_3_PAIRS = [
    {
        "name": "india_pakistan_deaths",
        "q1": "Will there be at least 1,000 deaths due to direct conflict between India and Pakistan before 2027?",
        "q2": "Will the death toll from direct conflict between India and Pakistan reach four figures before 2027?",
    },
    {
        "name": "usps_mail_ballots",
        "q1": "Will federal courts block the USPS mail-in ballot restrictions before the 2028 US presidential election?",
        "q2": "Will federal courts strike down the USPS mail-in ballot restrictions before the 2028 US presidential election?",
    },
    {
        "name": "nvidia_100",
        "q1": "Will Nvidia's stock price close below $100 on any day in 2026?",
        "q2": "Will Nvidia stock dip under the $100 mark at least once during 2026?",
    },
    {
        "name": "sp500_crash",
        "q1": "Will the US stock market crash by more than 10% in any 5-day trading period before 2027?",
        "q2": "Will the S&P 500 fall by over 10% within some single 5-trading-day stretch before 2027?",
    },
    {
        "name": "starship_refuel",
        "q1": "Will SpaceX successfully refuel a Starship in orbit during 2026?",
        "q2": "Will SpaceX complete an in-orbit propellant transfer between two Starships at some point in 2026?",
    },
    {
        "name": "musk_trillionaire",
        "q1": "Will Elon Musk become a trillionaire before July 2026?",
        "q2": "Will Elon Musk's net worth cross the $1 trillion mark before July 2026?",
    },
    {
        "name": "stablecoin_300b",
        "q1": "Will the market capitalization for USDT & USDC stablecoins surpass $300 billion USD by June 30, 2026?",
        "q2": "Will the combined market cap of USDT and USDC exceed $300 billion before July 1, 2026?",
    },
    {
        "name": "china_gdp_q2",
        "q1": "Will China's year-over-year GDP growth rate for Q2 2026 equal or exceed 5.0%, according to the National Bureau of Statistics of China?",
        "q2": "Will China's official year-over-year GDP growth for Q2 2026 come in at 5.0% or higher?",
    },
    {
        "name": "waymo_1200sqmi",
        "q1": "Will Waymo have a service area of 1,200 square miles before July 1, 2026?",
        "q2": "Will Waymo's robotaxi service area expand to 1,200 square miles before July 2026?",
    },
    {
        "name": "hurricane_landfall",
        "q1": "Will a hurricane make landfall in the continental United States before September 1, 2026?",
        "q2": "Will a hurricane make landfall somewhere along the continental US coast before September 2026?",
    },
]
