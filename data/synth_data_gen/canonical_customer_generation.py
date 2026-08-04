################################################################################
# Imports
################################################################################
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from faker import Faker

################################################################################
# 1. Generation configuration
################################################################################

# Seed ensures reproduceability
SEED = 42
NUMBER_OF_CUSTOMERS = 10_000

# The date on which the synthetic dataset is assumed to 
# have been extracted.
SNAPSHOT_DATE = pd.Timestamp("2025-12-31")

# Customers may have first interacted with the company before the 12-month operational activity period.
EARLIEST_REGISTRATION_DATE = pd.Timestamp("2018-01-01")

# Output path
OUTPUT_PATH = Path("data/canonical/canonical_customers.csv")

################################################################################
# 2. Assumptions
################################################################################

# Attendance based customer segments - describe customer relationships with the events organisation
CUSTOMER_SEGMENTS = [
    "New",
    "Occasional",
    "Regular",
    "VIP",
]

# Assumption - assigned segment weighting - based on fictional business model
SEGMENT_PROBABILITIES = [
    0.18,  # New
    0.42,  # Occasional
    0.30,  # Regular
    0.10,  # VIP
]

# Email domains
EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "aol.com",
    "hotmail.co.uk",
    "hotmail.fr",
    "msn.com",
    "yahoo.fr",
    "wanadoo.fr",
    "orange.fr",
]

# Assumption - assigned email weighting - based on most popular email domains by number of live emails (2016) available at: 
# https://email-verify.my-addr.com/list-of-most-popular-email-domains.php
EMAIL_DOMAIN_PROBABILITIES = [
    0.295077,  # gmail.com
    0.288423,  # yahoo.com
    0.258317,  # hotmail.com
    0.053227,  # aol.com
    0.021124,  # hotmail.co.uk
    0.020625,  # hotmail.fr
    0.018130,  # msn.com
    0.016301,  # yahoo.fr
    0.014970,  # wanadoo.fr
    0.013806,  # orange.fr
]

# Assumption - all customers will be aged between 18-85
AGES = list(range(18, 86))

# Assumption - customer age weighting - based on Enland and Wales census data (2021) 
# available at: https://www.ons.gov.uk/census/maps/choropleth/population/age/resident-age-3a/aged-15-years-and-under
AGE_PROBABILITIES = [
    0.014518787,  # 18
    0.014950781,  # 19
    0.014922815,  # 20
    0.015355809,  # 21
    0.015774472,  # 22
    0.015998939,  # 23
    0.016281512,  # 24
    0.016265224,  # 25
    0.016616060,  # 26
    0.016965047,  # 27
    0.017266844,  # 28
    0.017735872,  # 29
    0.018222732,  # 30
    0.017993785,  # 31
    0.018046934,  # 32
    0.018232192,  # 33
    0.017726086,  # 34
    0.017710016,  # 35
    0.017508817,  # 36
    0.017187731,  # 37
    0.017086610,  # 38
    0.017092916,  # 39
    0.017483396,  # 40
    0.017209260,  # 41
    0.016287275,  # 42
    0.015336498,  # 43
    0.015358005,  # 44
    0.015581885,  # 45
    0.015983172,  # 46
    0.016240694,  # 47
    0.016941213,  # 48
    0.017644320,  # 49
    0.018007616,  # 50
    0.017636339,  # 51
    0.018046868,  # 52
    0.017927415,  # 53
    0.018051827,  # 54
    0.018022947,  # 55
    0.018040953,  # 56
    0.017621138,  # 57
    0.017242314,  # 58
    0.016690085,  # 59
    0.016166452,  # 60
    0.015374924,  # 61
    0.015056556,  # 62
    0.014606034,  # 63
    0.013943202,  # 64
    0.013302334,  # 65
    0.012967199,  # 66
    0.012969982,  # 67
    0.012528073,  # 68
    0.012278728,  # 69
    0.012276161,  # 70
    0.012500607,  # 71
    0.012694324,  # 72
    0.013730282,  # 73
    0.013559246,  # 74
    0.010211074,  # 75
    0.010671925,  # 76
    0.009715233,  # 77
    0.008984704,  # 78
    0.007612785,  # 79
    0.007103004,  # 80
    0.007147388,  # 81
    0.006753082,  # 82
    0.006242388,  # 83
    0.005701705,  # 84
    0.005089412,  # 85
]

# Normalise to account for minor rounding differences in source values
AGE_PROBABILITIES = (
    np.array(AGE_PROBABILITIES)
    / np.sum(AGE_PROBABILITIES)
)

# Possible regions to be used to generate customer addresses
REGIONS = [
    "South East",
    "London",
    "North West",
    "East",
    "West Midlands",
    "South West",
    "Yorkshire and the Humber",
    "Scotland",
    "East Midlands",
    "Wales",
    "North East",
    "Northern Ireland",
]

# Assumption - Customer postcode distribution to be based on UK population distribution by region (2024) 
# available at: https://www.statista.com/statistics/294729/uk-population-by-region/?srsltid=AfmBOoq_fXuEgGBrRR3_ILynAo2tt5iie2zB2xweS-CrIJXNxntqNw9k
REGION_PROBABILITIES = [
    0.1392,  # South East
    0.1312,  # London
    0.1117,  # North West
    0.0949,  # East
    0.0893,  # West Midlands
    0.0850,  # South West
    0.0819,  # Yorkshire and the Humber
    0.0801,  # Scotland
    0.0731,  # East Midlands
    0.0460,  # Wales
    0.0398,  # North East
    0.0278,  # Northern Ireland
]

# Normalise to account for minor rounding differences in source values
REGION_PROBABILITIES = (
    np.array(REGION_PROBABILITIES)
    / np.sum(REGION_PROBABILITIES)
)

# Postcode pools - designed to be representative rather than exhaustive
REGION_POSTCODE_DISTRICTS = {
    "South East": [
        ("Brighton", "BN1"),
        ("Guildford", "GU1"),
        ("Reading", "RG1"),
        ("Oxford", "OX1"),
        ("Portsmouth", "PO1"),
        ("Southampton", "SO14"),
    ],
    "London": [
        ("London", "E1"),
        ("London", "N1"),
        ("London", "NW1"),
        ("London", "SE1"),
        ("London", "SW1"),
        ("London", "W1"),
    ],
    "North West": [
        ("Manchester", "M1"),
        ("Liverpool", "L1"),
        ("Preston", "PR1"),
        ("Chester", "CH1"),
        ("Blackburn", "BB1"),
        ("Carlisle", "CA1"),
    ],
    "East": [
        ("Cambridge", "CB1"),
        ("Chelmsford", "CM1"),
        ("Norwich", "NR1"),
        ("Ipswich", "IP1"),
        ("Luton", "LU1"),
        ("Peterborough", "PE1"),
    ],
    "West Midlands": [
        ("Birmingham", "B1"),
        ("Coventry", "CV1"),
        ("Wolverhampton", "WV1"),
        ("Worcester", "WR1"),
        ("Shrewsbury", "SY1"),
        ("Stoke-on-Trent", "ST1"),
    ],
    "South West": [
        ("Bristol", "BS1"),
        ("Bath", "BA1"),
        ("Exeter", "EX1"),
        ("Plymouth", "PL1"),
        ("Gloucester", "GL1"),
        ("Bournemouth", "BH1"),
    ],
    "Yorkshire and the Humber": [
        ("Leeds", "LS1"),
        ("Sheffield", "S1"),
        ("York", "YO1"),
        ("Hull", "HU1"),
        ("Bradford", "BD1"),
        ("Harrogate", "HG1"),
    ],
    "Scotland": [
        ("Glasgow", "G1"),
        ("Edinburgh", "EH1"),
        ("Aberdeen", "AB10"),
        ("Dundee", "DD1"),
        ("Inverness", "IV1"),
        ("Perth", "PH1"),
    ],
    "East Midlands": [
        ("Nottingham", "NG1"),
        ("Leicester", "LE1"),
        ("Derby", "DE1"),
        ("Lincoln", "LN1"),
        ("Northampton", "NN1"),
        ("Chesterfield", "S40"),
    ],
    "Wales": [
        ("Cardiff", "CF10"),
        ("Swansea", "SA1"),
        ("Newport", "NP20"),
        ("Wrexham", "LL11"),
        ("Bangor", "LL57"),
        ("Aberystwyth", "SY23"),
    ],
    "North East": [
        ("Newcastle upon Tyne", "NE1"),
        ("Sunderland", "SR1"),
        ("Durham", "DH1"),
        ("Middlesbrough", "TS1"),
        ("Darlington", "DL1"),
    ],
    "Northern Ireland": [
        ("Belfast", "BT1"),
        ("Derry", "BT48"),
        ("Lisburn", "BT27"),
        ("Newry", "BT34"),
        ("Armagh", "BT61"),
    ],
}

############################################################
# 3. initialise generators
############################################################

# UK localised Faker generator: "en_GB"
fake = Faker("en_GB")

#assign consistent seed to both Faker and Numpy generators - enable reproducible generation process
Faker.seed(SEED)
rng = np.random.default_rng(SEED)


############################################################
# 4. helper functions
############################################################

# Normalise names to build a lowercase strings suitable for use within synthetic email address
def normalise_for_email(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()

    # Remove punctuation, spaces, apostrophes and hyphens.
    value = re.sub(r"[^a-z0-9]", "", value)

    return value

# Generate plausible and unique personal email address using normailsed names.
def generate_unique_email(
    first_name: str,
    surname: str,
    # Record emails generated to ensure uniqueness
    used_emails: set[str],
) -> str:

    # Use normalised names from normalise_for_email
    first = normalise_for_email(first_name)
    last = normalise_for_email(surname)

    # Use predefined email domains and probabilities
    domain = rng.choice(
        EMAIL_DOMAINS,
        p=EMAIL_DOMAIN_PROBABILITIES,
    )

    # Build email
    base_email = f"{first}.{last}@{domain}"
    email = base_email
    suffix = 2

    # Add number in case of already used email
    while email in used_emails:
        email = f"{first}.{last}{suffix}@{domain}"
        suffix += 1

    used_emails.add(email)

    return email

# Generate unique UK-style mobile number - stored as consistent international format
def generate_unique_phone(used_phones: set[str]) -> str:
    while True:
        subscriber_number = rng.integers(0, 1_000_000_000)
        # Consistent international formatting
        phone = f"+44 7{subscriber_number:09d}"

        # Ensure unique phone numbers
        if phone not in used_phones:
            used_phones.add(phone)
            return phone

# Select random date between two boundaries - gives every date within period approximately equal probability
def generate_random_date(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Timestamp:
    number_of_days = (end_date - start_date).days

    if number_of_days <= 0:
        return end_date

    # +1 makes end boundary eligible for selection
    random_days = rng.integers(0, number_of_days + 1)

    return start_date + pd.Timedelta(days=int(random_days))

# Generate date on which customer first interacted with fictional events company.
def generate_registration_date(
    date_of_birth: pd.Timestamp,
    customer_segment: str,
) -> pd.Timestamp:
    
    # Calculate customers 18th birthday
    eighteenth_birthday = date_of_birth + pd.DateOffset(years=18)

    # New customers registered during most recent 12-month period
    if customer_segment == "New":
        earliest_date = pd.Timestamp("2025-01-01")
    # Other customer segments can register anytime after 01/01/2018
    else:
        earliest_date = EARLIEST_REGISTRATION_DATE

    # Ensure all customers are over 18 at registration
    earliest_date = max(earliest_date, eighteenth_birthday)

    return generate_random_date(
        start_date=earliest_date,
        end_date=SNAPSHOT_DATE,
    )

# Generate date of birth using weighted England and Wales age distribution
def generate_date_of_birth() -> pd.Timestamp:

    # Select exact age using predefined probabilities.
    selected_age = int(
        rng.choice(
            AGES,
            p=AGE_PROBABILITIES,
        )
    )

    # Latest possible birth date for a person of this age.
    latest_birth_date = (
        SNAPSHOT_DATE
        - pd.DateOffset(years=selected_age)
    )

    # Earliest possible birth date for a person of this age.
    earliest_birth_date = (
        SNAPSHOT_DATE
        - pd.DateOffset(years=selected_age + 1)
        + pd.Timedelta(days=1)
    )

    # Select a random date within valid birth-date range.
    return generate_random_date(
        start_date=earliest_birth_date,
        end_date=latest_birth_date,
    )

# Generate a synthetic postcode inward code
def generate_postcode_suffix() -> str:
    digit = int(rng.integers(0, 10))
    first_letter = fake.random_letter().upper()
    second_letter = fake.random_letter().upper()

    return f"{digit}{first_letter}{second_letter}"

#Select a UK region using population-based weights to generate a representative city, address and postcode.
def generate_customer_location() -> tuple[str, str, str]:
    region = str(
        rng.choice(
            REGIONS,
            p=REGION_PROBABILITIES,
        )
    )

    location_pool = REGION_POSTCODE_DISTRICTS[region]

    location_index = int(
        rng.integers(
            0,
            len(location_pool),
        )
    )

    city, postcode_district = location_pool[location_index]

    street_address = fake.street_address().replace("\n", ", ")
    address = f"{street_address}, {city}"

    postcode = (
        f"{postcode_district} "
        f"{generate_postcode_suffix()}"
    )

    return region, address, postcode

############################################################
# 5. generate clean canonical customer population
############################################################

# Generate canonical dataset to act as hidden source of truth 
def generate_canonical_customers(
    number_of_customers: int,
) -> pd.DataFrame:

    # Temporarily store customer as dictionary in list
    customer_records = []

    # Prevent duplicate emails or phone numbers
    used_emails: set[str] = set()
    used_phones: set[str] = set()

    # Repeaat process for every customer
    for customer_number in range(1, number_of_customers + 1):

        # Generate canonical ID
        ground_truth_id = f"GT{customer_number:06d}"

        # Generate name
        first_name = fake.first_name()
        surname = fake.last_name()

        # Generate DOB
        date_of_birth = generate_date_of_birth()

        # Assign customer segment
        customer_segment = rng.choice(
            CUSTOMER_SEGMENTS,
            p=SEGMENT_PROBABILITIES,
        )

        # Generate email - calls helper function
        email = generate_unique_email(
            first_name=first_name,
            surname=surname,
            used_emails=used_emails,
        )

        # Generate phone number - calls helper funciton
        telephone_number = generate_unique_phone(
            used_phones=used_phones
        )

        # Generate address
        region, address, postcode = generate_customer_location()

        # Generate reg date - calls helper function
        registration_date = generate_registration_date(
            date_of_birth=date_of_birth,
            customer_segment=customer_segment,
        )

        # Append list with customer details
        customer_records.append(
            {
                "ground_truth_id": ground_truth_id,
                "first_name": first_name,
                "surname": surname,
                "date_of_birth": date_of_birth.date(),
                "email": email,
                "telephone_number": telephone_number,
                "region": region,
                "address": address,
                "postcode": postcode,
                "registration_date": registration_date.date(),
                "customer_segment": customer_segment,
            }
        )
    # Return list as tabular pd dataframe
    return pd.DataFrame(customer_records)


############################################################
# 5. validate and visualise
############################################################

# Check that generated dataset meets basic requirements
def validate_canonical_customers(
    customers: pd.DataFrame,
) -> None:

    # Validate probability arrays:
    assert np.isclose(np.sum(SEGMENT_PROBABILITIES), 1.0)
    assert np.isclose(np.sum(EMAIL_DOMAIN_PROBABILITIES), 1.0)
    assert np.isclose(np.sum(AGE_PROBABILITIES), 1.0)
    assert np.isclose(np.sum(REGION_PROBABILITIES), 1.0)

    # Validate expexted cols
    expected_columns = [
        "ground_truth_id",
        "first_name",
        "surname",
        "date_of_birth",
        "email",
        "telephone_number",
        "region",
        "address",
        "postcode",
        "registration_date",
        "customer_segment",
    ]

    # Validate required columns and number of customers
    assert list(customers.columns) == expected_columns
    assert len(customers) == NUMBER_OF_CUSTOMERS

    # Validate unique identifiers
    assert customers["ground_truth_id"].is_unique
    assert customers["email"].is_unique
    assert customers["telephone_number"].is_unique

    assert set(customers["customer_segment"]).issubset(
        set(CUSTOMER_SEGMENTS)
    )

    assert set(customers["region"]).issubset(
        set(REGIONS)
    )

    # Validate alignment against probability lists
    assert len(CUSTOMER_SEGMENTS) == len(SEGMENT_PROBABILITIES)
    assert len(EMAIL_DOMAINS) == len(EMAIL_DOMAIN_PROBABILITIES)
    assert len(AGES) == len(AGE_PROBABILITIES)
    assert len(REGIONS) == len(REGION_PROBABILITIES)

    # Confirm missing values = 0
    assert customers.isna().sum().sum() == 0

    print("\nCanonical customer validation completed successfully.")
    print(f"Number of customers: {len(customers):,}")

    print("\nCustomer segment distribution:")
    print(
        customers["customer_segment"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    # Convert DOB column to datetime
    dates_of_birth = pd.to_datetime(
        customers["date_of_birth"]
    )

    # Calculate customer age at snapshot date
    customer_ages = (
        SNAPSHOT_DATE.year
        - dates_of_birth.dt.year
        - (
            (
                dates_of_birth.dt.month > SNAPSHOT_DATE.month
            )
            |
            (
                (dates_of_birth.dt.month == SNAPSHOT_DATE.month)
                &
                (dates_of_birth.dt.day > SNAPSHOT_DATE.day)
            )
        ).astype(int)
    )

    # Confirm customers are 18-85
    assert customer_ages.between(18, 85).all()

    # Group ages for easier validation/visualisation
    age_groups = pd.cut(
        customer_ages,
        bins=[17, 24, 34, 44, 54, 64, 74, 85],
        labels=[
            "18-24",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65-74",
            "75-85",
        ],
    )

    # Visualise age group distribution
    print("\nGrouped age distribution:")
    print(
        age_groups
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    # Validate region distributions
    print("\nRegional distribution:")
    print(
        customers["region"]
        .value_counts(normalize=True)
        .reindex(REGIONS)
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    # Validate registration date ranges
    print("\nRegistration date range:")
    print(
        customers["registration_date"].min(),
        "to",
        customers["registration_date"].max(),
    )


############################################################
# 6. Main funnction
############################################################

def main() -> None:
    # Generate
    customers = generate_canonical_customers(
        number_of_customers=NUMBER_OF_CUSTOMERS
    )

    # Validate
    validate_canonical_customers(customers)

    # Create output path
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Export pd dataframe as csv
    customers.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # Final visualisations
    print(f"\nDataset saved to: {OUTPUT_PATH.resolve()}")
    print("\nFirst five records:")
    print(customers.head())


if __name__ == "__main__":
    main()