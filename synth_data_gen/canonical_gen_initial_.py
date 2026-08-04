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

# Assumption - assigned segment probabilities
SEGMENT_PROBABILITIES = [
    0.18, # New
    0.42, # Occasional
    0.30, # Regular
    0.10, # VIP
]

# Email domains
EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "aol.com",
    "hotmail.co.uk	",
    "hotmail.fr",
    "msn.com",
    "yahoo.fr",
    "wanadoo.fr",
    "orange.fr",
]

# Assumption - assigned email probabilities
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
        date_of_birth = pd.Timestamp(
            fake.date_of_birth(
                minimum_age=18,
                maximum_age=85,
            )
        )

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
        # Convert line breaks into commas
        street_address = fake.street_address().replace("\n", ", ")
        city = fake.city()
        address = f"{street_address}, {city}"

        # Generate postcode
        postcode = fake.postcode().upper()

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

    expected_columns = [
        "ground_truth_id",
        "first_name",
        "surname",
        "date_of_birth",
        "email",
        "telephone_number",
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

    # Confirm missing values = 0
    assert customers.isna().sum().sum() == 0

    # Visualisations
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