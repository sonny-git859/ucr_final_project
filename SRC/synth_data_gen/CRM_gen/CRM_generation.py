###############################################################################
# Imports
###############################################################################

from pathlib import Path
import re

import numpy as np
import pandas as pd
from faker import Faker


###############################################################################
# 1. Generation configuration
###############################################################################

# CRM-specific seed ensures reproducibility
CRM_SEED = 101

# Dataset size
UNIQUE_CRM_CUSTOMERS = 7_800
NUMBER_OF_DUPLICATE_RECORDS = 200
NUMBER_OF_CRM_RECORDS = 8_000

# Date on which synthetic environment is assumed to have been extracted
SNAPSHOT_DATE = pd.Timestamp("2025-12-31")

# Input canonical dataset
CANONICAL_INPUT_PATH = Path(
    "data/canonical/canonical_customers.csv"
)

# Output paths
CRM_OUTPUT_PATH = Path(
    "data/raw/crm_customer_records.csv"
)

GROUND_TRUTH_OUTPUT_PATH = Path(
    "data/reference/crm_ground_truth_mapping.csv"
)

PORTAL_MAPPING_OUTPUT_PATH = Path(
    "data/reference/portal_account_mapping.csv"
)

VALIDATION_OUTPUT_PATH = Path(
    "data/reference/crm_validation_summary.csv"
)


###############################################################################
# 2. Weighting assumptions
###############################################################################

# CRM customer selection is weighted using customer_segment
# from canonical dataset
# Higher values represent higher likelihood of appearing in CRM
CRM_SEGMENT_SELECTION_WEIGHTS = {
    "New": 0.60,
    "Occasional": 0.80,
    "Regular": 1.00,
    "VIP": 1.20,
}

# Approximately 55% of represented customers possess a portal account
PORTAL_ACCOUNT_RATE = 0.55

# Preferred contact channel weighting
PREFERRED_CONTACT_CHANNELS = [
    "Email",
    "Telephone",
    "Mail",
]

# based on fictional business model
PREFERRED_CONTACT_CHANNEL_WEIGHTS = {
    "Email": 0.65,
    "Telephone": 0.25,
    "Mail": 0.10,
}


###############################################################################
# 3. Data quality assumptions
###############################################################################

# Rates used to deliberately introduce data quality issues
MISSING_EMAIL_RATE = 0.03
MISSING_PHONE_RATE = 0.10
MISSING_ADDRESS_RATE = 0.04
MISSING_POSTCODE_RATE = 0.03

OUTDATED_ADDRESS_RATE = 0.05

EMAIL_CASE_VARIATION_RATE = 0.06
ABBREVIATED_FIRST_NAME_RATE = 0.03
NAME_TYPO_RATE = 0.02
PHONE_FORMAT_VARIATION_RATE = 0.20
POSTCODE_FORMAT_VARIATION_RATE = 0.10


###############################################################################
# 4. Initialise generators
###############################################################################

# UK-localised Faker generator
fake = Faker("en_GB")

# Assign consistent seed to Faker and NumPy
Faker.seed(CRM_SEED)
rng = np.random.default_rng(CRM_SEED)


###############################################################################
# 5. Helper functions
###############################################################################

# Select a random date between two date boundaries
def generate_random_date(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Timestamp:

    number_of_days = (end_date - start_date).days

    if number_of_days <= 0:
        return end_date

    random_days = rng.integers(
        0,
        number_of_days + 1,
    )

    return start_date + pd.Timedelta(
        days=int(random_days)
    )


# Generate CRM record creation date within 7 days
# of canonical registration date
def generate_record_created_date(
    registration_date: pd.Timestamp,
) -> pd.Timestamp:

    maximum_delay = min(
        7,
        (SNAPSHOT_DATE - registration_date).days,
    )

    if maximum_delay <= 0:
        return registration_date

    delay_days = int(
        rng.integers(
            0,
            maximum_delay + 1,
        )
    )

    return registration_date + pd.Timedelta(
        days=delay_days
    )


# Generate later record creation date for intentionally
# introduced duplicate CRM account
def generate_duplicate_created_date(
    original_created_date: pd.Timestamp,
) -> pd.Timestamp:

    earliest_duplicate_date = (
        original_created_date
        + pd.Timedelta(days=1)
    )

    if earliest_duplicate_date > SNAPSHOT_DATE:
        return original_created_date

    return generate_random_date(
        start_date=earliest_duplicate_date,
        end_date=SNAPSHOT_DATE,
    )


# Introduce a minor spelling error into a string
def introduce_name_typo(
    value: str,
) -> str:

    if pd.isna(value):
        return value

    value = str(value)

    if len(value) < 3:
        return value

    characters = list(value)

    typo_type = rng.choice(
        [
            "swap",
            "remove",
        ]
    )

    if typo_type == "swap" and len(characters) >= 3:

        position = int(
            rng.integers(
                0,
                len(characters) - 1,
            )
        )

        characters[position], characters[position + 1] = (
            characters[position + 1],
            characters[position],
        )

    else:

        position = int(
            rng.integers(
                0,
                len(characters),
            )
        )

        del characters[position]

    return "".join(characters)


# Convert canonical phone number into a different plausible format
def alter_phone_format(
    telephone_number: str,
) -> str:

    if pd.isna(telephone_number):
        return telephone_number

    digits = re.sub(
        r"\D",
        "",
        str(telephone_number),
    )

    # Remove international country code
    if digits.startswith("44"):
        national_number = digits[2:]
    else:
        national_number = digits.lstrip("0")

    local_number = f"0{national_number}"

    phone_formats = [
        local_number,
        f"{local_number[:5]} {local_number[5:8]} {local_number[8:]}",
        f"+44 {national_number[:4]} {national_number[4:]}",
        f"+44{national_number}",
        f"0044 {national_number}",
    ]

    return str(
        rng.choice(phone_formats)
    )


# Remove normal spacing from postcode
def alter_postcode_format(
    postcode: str,
) -> str:

    if pd.isna(postcode):
        return postcode

    return str(postcode).replace(
        " ",
        "",
    )


# Generate an alternative historical address while retaining
# the same city and approximate postcode district
def generate_outdated_address(
    current_address: str,
    current_postcode: str,
) -> tuple[str, str]:

    # Retain city from current canonical address
    if "," in str(current_address):
        city = str(current_address).split(",")[-1].strip()
    else:
        city = fake.city()

    new_street_address = (
        fake.street_address()
        .replace("\n", ", ")
    )

    outdated_address = (
        f"{new_street_address}, {city}"
    )

    # Preserve outward postcode area where possible
    postcode_parts = str(current_postcode).split()

    if len(postcode_parts) > 0:
        outward_code = postcode_parts[0]
    else:
        outward_code = "ZZ1"

    postcode_digit = int(
        rng.integers(
            0,
            10,
        )
    )

    first_letter = fake.random_letter().upper()
    second_letter = fake.random_letter().upper()

    outdated_postcode = (
        f"{outward_code} "
        f"{postcode_digit}{first_letter}{second_letter}"
    )

    return (
        outdated_address,
        outdated_postcode,
    )


# Generate preferred contact channel based only on
# contact information available within CRM record
def generate_preferred_contact_channel(
    email,
    telephone_number,
    address,
    postcode,
) -> str:

    available_channels = []
    available_weights = []

    if pd.notna(email):
        available_channels.append("Email")
        available_weights.append(
            PREFERRED_CONTACT_CHANNEL_WEIGHTS["Email"]
        )

    if pd.notna(telephone_number):
        available_channels.append("Telephone")
        available_weights.append(
            PREFERRED_CONTACT_CHANNEL_WEIGHTS["Telephone"]
        )

    if (
        pd.notna(address)
        and pd.notna(postcode)
    ):
        available_channels.append("Mail")
        available_weights.append(
            PREFERRED_CONTACT_CHANNEL_WEIGHTS["Mail"]
        )

    # CRM generation should normally retain at least one
    # contact route, but provide fallback in case all are missing
    if len(available_channels) == 0:
        return "Unknown"

    available_weights = np.array(
        available_weights,
        dtype=float,
    )

    available_weights = (
        available_weights
        / available_weights.sum()
    )

    return str(
        rng.choice(
            available_channels,
            p=available_weights,
        )
    )


###############################################################################
# 6. Load and validate canonical dataset
###############################################################################

def load_canonical_customers() -> pd.DataFrame:

    canonical = pd.read_csv(
        CANONICAL_INPUT_PATH
    )

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

    # Validate expected canonical structure
    assert list(canonical.columns) == expected_columns

    assert canonical[
        "ground_truth_id"
    ].is_unique

    assert len(canonical) >= UNIQUE_CRM_CUSTOMERS

    # Convert date columns
    canonical["date_of_birth"] = pd.to_datetime(
        canonical["date_of_birth"]
    )

    canonical["registration_date"] = pd.to_datetime(
        canonical["registration_date"]
    )

    return canonical


###############################################################################
# 7. Select CRM customer population
###############################################################################

def select_crm_customers(
    canonical: pd.DataFrame,
) -> pd.DataFrame:

    # Map each canonical customer to a segment-based
    # relative selection weight
    selection_weights = (
        canonical["customer_segment"]
        .map(CRM_SEGMENT_SELECTION_WEIGHTS)
        .astype(float)
    )

    # Normalise weights for use by NumPy
    selection_weights = (
        selection_weights
        / selection_weights.sum()
    )

    # Select exactly 7,800 unique canonical customers
    selected_indices = rng.choice(
        canonical.index.to_numpy(),
        size=UNIQUE_CRM_CUSTOMERS,
        replace=False,
        p=selection_weights.to_numpy(),
    )

    selected_customers = (
        canonical
        .loc[selected_indices]
        .copy()
        .reset_index(drop=True)
    )

    return selected_customers


###############################################################################
# 8. Generate initial clean CRM records
###############################################################################

def generate_clean_crm_records(
    selected_customers: pd.DataFrame,
) -> pd.DataFrame:

    crm_records = []

    # Exactly 55% of represented unique customers receive portal accounts
    number_of_portal_accounts = round(
        UNIQUE_CRM_CUSTOMERS
        * PORTAL_ACCOUNT_RATE
    )

    portal_positions = set(
        rng.choice(
            np.arange(
                UNIQUE_CRM_CUSTOMERS
            ),
            size=number_of_portal_accounts,
            replace=False,
        )
    )

    portal_number = 1

    for record_number, customer in enumerate(
        selected_customers.itertuples(index=False),
        start=1,
    ):

        crm_customer_id = (
            f"CRM{record_number:06d}"
        )

        # Generate portal user ID where customer possesses
        # an online portal account
        if (record_number - 1) in portal_positions:

            portal_user_id = (
                f"USR{portal_number:06d}"
            )

            portal_number += 1

        else:
            portal_user_id = pd.NA

        registration_date = pd.Timestamp(
            customer.registration_date
        )

        record_created_date = (
            generate_record_created_date(
                registration_date
            )
        )

        crm_records.append(
            {
                # Internal generation fields
                "ground_truth_id": customer.ground_truth_id,
                "canonical_registration_date": registration_date,
                "customer_segment": customer.customer_segment,
                "is_duplicate_record": False,
                "duplicate_of_crm_customer_id": pd.NA,

                # Operational CRM fields
                "crm_customer_id": crm_customer_id,
                "portal_user_id": portal_user_id,
                "record_created_date": record_created_date,
                "preferred_contact_channel": pd.NA,
                "first_name": customer.first_name,
                "surname": customer.surname,
                "date_of_birth": pd.Timestamp(
                    customer.date_of_birth
                ),
                "email": customer.email,
                "telephone_number": customer.telephone_number,
                "address": customer.address,
                "postcode": customer.postcode,
            }
        )

    return pd.DataFrame(
        crm_records
    )


###############################################################################
# 9. Generate duplicate CRM records
###############################################################################

def add_duplicate_crm_records(
    crm_records: pd.DataFrame,
) -> pd.DataFrame:

    # Only use records where there is at least one day
    # available for a later duplicate account creation
    duplicate_candidates = crm_records[
        crm_records["record_created_date"]
        < SNAPSHOT_DATE
    ]

    duplicate_indices = rng.choice(
        duplicate_candidates.index.to_numpy(),
        size=NUMBER_OF_DUPLICATE_RECORDS,
        replace=False,
    )

    duplicate_records = []

    next_crm_number = (
        len(crm_records) + 1
    )

    for original_index in duplicate_indices:

        original_record = crm_records.loc[
            original_index
        ]

        duplicate_record = (
            original_record.copy()
        )

        duplicate_record[
            "crm_customer_id"
        ] = (
            f"CRM{next_crm_number:06d}"
        )

        duplicate_record[
            "record_created_date"
        ] = generate_duplicate_created_date(
            pd.Timestamp(
                original_record[
                    "record_created_date"
                ]
            )
        )

        duplicate_record[
            "is_duplicate_record"
        ] = True

        duplicate_record[
            "duplicate_of_crm_customer_id"
        ] = original_record[
            "crm_customer_id"
        ]

        # Duplicate represents another CRM account for the
        # same customer and therefore retains ground truth
        # and existing portal relationship where applicable

        duplicate_records.append(
            duplicate_record
        )

        next_crm_number += 1

    duplicate_dataframe = pd.DataFrame(
        duplicate_records
    )

    crm_records = pd.concat(
        [
            crm_records,
            duplicate_dataframe,
        ],
        ignore_index=True,
    )

    return crm_records


###############################################################################
# 10. Introduce CRM data quality issues
###############################################################################

def introduce_crm_data_quality_issues(
    crm_records: pd.DataFrame,
) -> pd.DataFrame:

    crm_records = crm_records.copy()

    # Initialise hidden data-quality flags
    data_quality_flags = [
        "dq_missing_email",
        "dq_missing_phone",
        "dq_missing_address",
        "dq_missing_postcode",
        "dq_outdated_address",
        "dq_email_case_variation",
        "dq_abbreviated_first_name",
        "dq_name_typo",
        "dq_phone_format_variation",
        "dq_postcode_format_variation",
    ]

    for flag in data_quality_flags:
        crm_records[flag] = False

    for index in crm_records.index:

        #######################################################################
        # Email
        #######################################################################

        if rng.random() < MISSING_EMAIL_RATE:

            crm_records.at[
                index,
                "email",
            ] = pd.NA

            crm_records.at[
                index,
                "dq_missing_email",
            ] = True

        elif rng.random() < EMAIL_CASE_VARIATION_RATE:

            current_email = str(
                crm_records.at[
                    index,
                    "email",
                ]
            )

            # Randomly convert email to uppercase or title-style case
            if rng.random() < 0.5:
                altered_email = (
                    current_email.upper()
                )
            else:
                altered_email = (
                    current_email.swapcase()
                )

            crm_records.at[
                index,
                "email",
            ] = altered_email

            crm_records.at[
                index,
                "dq_email_case_variation",
            ] = True

        #######################################################################
        # Telephone number
        #######################################################################

        if rng.random() < MISSING_PHONE_RATE:

            crm_records.at[
                index,
                "telephone_number",
            ] = pd.NA

            crm_records.at[
                index,
                "dq_missing_phone",
            ] = True

        elif rng.random() < PHONE_FORMAT_VARIATION_RATE:

            crm_records.at[
                index,
                "telephone_number",
            ] = alter_phone_format(
                crm_records.at[
                    index,
                    "telephone_number",
                ]
            )

            crm_records.at[
                index,
                "dq_phone_format_variation",
            ] = True

        #######################################################################
        # Address and postcode
        #######################################################################

        # Introduce an outdated address before applying missingness
        if rng.random() < OUTDATED_ADDRESS_RATE:

            (
                outdated_address,
                outdated_postcode,
            ) = generate_outdated_address(
                crm_records.at[
                    index,
                    "address",
                ],
                crm_records.at[
                    index,
                    "postcode",
                ],
            )

            crm_records.at[
                index,
                "address",
            ] = outdated_address

            crm_records.at[
                index,
                "postcode",
            ] = outdated_postcode

            crm_records.at[
                index,
                "dq_outdated_address",
            ] = True

        # Address may subsequently be missing
        if rng.random() < MISSING_ADDRESS_RATE:

            crm_records.at[
                index,
                "address",
            ] = pd.NA

            crm_records.at[
                index,
                "dq_missing_address",
            ] = True

        # Postcode may independently be missing
        if rng.random() < MISSING_POSTCODE_RATE:

            crm_records.at[
                index,
                "postcode",
            ] = pd.NA

            crm_records.at[
                index,
                "dq_missing_postcode",
            ] = True

        elif (
            pd.notna(
                crm_records.at[
                    index,
                    "postcode",
                ]
            )
            and rng.random()
            < POSTCODE_FORMAT_VARIATION_RATE
        ):

            crm_records.at[
                index,
                "postcode",
            ] = alter_postcode_format(
                crm_records.at[
                    index,
                    "postcode",
                ]
            )

            crm_records.at[
                index,
                "dq_postcode_format_variation",
            ] = True

        #######################################################################
        # Name variations
        #######################################################################

        if (
            rng.random()
            < ABBREVIATED_FIRST_NAME_RATE
        ):

            current_first_name = str(
                crm_records.at[
                    index,
                    "first_name",
                ]
            )

            if len(current_first_name) > 0:

                crm_records.at[
                    index,
                    "first_name",
                ] = (
                    f"{current_first_name[0]}."
                )

                crm_records.at[
                    index,
                    "dq_abbreviated_first_name",
                ] = True

        if rng.random() < NAME_TYPO_RATE:

            # Randomly introduce typo in either first name or surname
            name_field = str(
                rng.choice(
                    [
                        "first_name",
                        "surname",
                    ]
                )
            )

            crm_records.at[
                index,
                name_field,
            ] = introduce_name_typo(
                crm_records.at[
                    index,
                    name_field,
                ]
            )

            crm_records.at[
                index,
                "dq_name_typo",
            ] = True

    ###########################################################################
    # Generate preferred contact channel after missingness has been introduced
    ###########################################################################

    for index in crm_records.index:

        crm_records.at[
            index,
            "preferred_contact_channel",
        ] = generate_preferred_contact_channel(
            email=crm_records.at[
                index,
                "email",
            ],
            telephone_number=crm_records.at[
                index,
                "telephone_number",
            ],
            address=crm_records.at[
                index,
                "address",
            ],
            postcode=crm_records.at[
                index,
                "postcode",
            ],
        )

    return crm_records


###############################################################################
# 11. Validate CRM dataset
###############################################################################

def validate_crm_records(
    crm_records: pd.DataFrame,
    canonical: pd.DataFrame,
) -> pd.DataFrame:

    ###########################################################################
    # Basic validation
    ###########################################################################

    # Confirm expected population sizes
    assert len(
        crm_records
    ) == NUMBER_OF_CRM_RECORDS

    assert (
        crm_records[
            "ground_truth_id"
        ].nunique()
        == UNIQUE_CRM_CUSTOMERS
    )

    assert (
        crm_records[
            "is_duplicate_record"
        ].sum()
        == NUMBER_OF_DUPLICATE_RECORDS
    )

    # Confirm CRM identifiers are unique
    assert crm_records[
        "crm_customer_id"
    ].is_unique

    # Confirm all source identities exist in canonical dataset
    assert set(
        crm_records[
            "ground_truth_id"
        ]
    ).issubset(
        set(
            canonical[
                "ground_truth_id"
            ]
        )
    )

    # Confirm record dates do not exceed snapshot date
    assert (
        pd.to_datetime(
            crm_records[
                "record_created_date"
            ]
        )
        <= SNAPSHOT_DATE
    ).all()

    # Validate initial CRM creation delay
    initial_records = crm_records[
        ~crm_records[
            "is_duplicate_record"
        ]
    ].copy()

    initial_delay_days = (
        pd.to_datetime(
            initial_records[
                "record_created_date"
            ]
        )
        -
        pd.to_datetime(
            initial_records[
                "canonical_registration_date"
            ]
        )
    ).dt.days

    assert initial_delay_days.between(
        0,
        7,
    ).all()

    # Confirm duplicate records represent existing selected customers
    duplicate_records = crm_records[
        crm_records[
            "is_duplicate_record"
        ]
    ]

    assert duplicate_records[
        "duplicate_of_crm_customer_id"
    ].notna().all()

    # Confirm each portal user belongs to only one ground-truth customer
    portal_records = crm_records[
        crm_records[
            "portal_user_id"
        ].notna()
    ]

    portal_identity_counts = (
        portal_records
        .groupby(
            "portal_user_id"
        )[
            "ground_truth_id"
        ]
        .nunique()
    )

    assert (
        portal_identity_counts <= 1
    ).all()

    # Confirm preferred contact channel is compatible
    # with information available in each record
    email_preference = crm_records[
        crm_records[
            "preferred_contact_channel"
        ] == "Email"
    ]

    assert email_preference[
        "email"
    ].notna().all()

    telephone_preference = crm_records[
        crm_records[
            "preferred_contact_channel"
        ] == "Telephone"
    ]

    assert telephone_preference[
        "telephone_number"
    ].notna().all()

    mail_preference = crm_records[
        crm_records[
            "preferred_contact_channel"
        ] == "Mail"
    ]

    assert mail_preference[
        "address"
    ].notna().all()

    assert mail_preference[
        "postcode"
    ].notna().all()

    ###########################################################################
    # Visualise validation results
    ###########################################################################

    print(
        "\nCRM validation completed successfully."
    )

    print(
        f"Total CRM records: "
        f"{len(crm_records):,}"
    )

    print(
        f"Unique canonical customers represented: "
        f"{crm_records['ground_truth_id'].nunique():,}"
    )

    print(
        f"Duplicate CRM records: "
        f"{crm_records['is_duplicate_record'].sum():,}"
    )

    portal_customer_count = (
        crm_records[
            crm_records[
                "portal_user_id"
            ].notna()
        ][
            "ground_truth_id"
        ]
        .nunique()
    )

    portal_customer_rate = (
        portal_customer_count
        / UNIQUE_CRM_CUSTOMERS
    )

    print(
        f"Portal account customers: "
        f"{portal_customer_count:,} "
        f"({portal_customer_rate:.2%})"
    )

    ###########################################################################
    # Customer segment representation
    ###########################################################################

    canonical_segment_counts = (
        canonical[
            "customer_segment"
        ]
        .value_counts()
    )

    crm_segment_counts = (
        initial_records[
            "customer_segment"
        ]
        .value_counts()
    )

    segment_coverage = (
        crm_segment_counts
        / canonical_segment_counts
    ).fillna(0)

    segment_coverage = segment_coverage.reindex(
        [
            "New",
            "Occasional",
            "Regular",
            "VIP",
        ]
    )

    print(
        "\nCRM customer coverage by segment:"
    )

    print(
        segment_coverage
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    ###########################################################################
    # Missing-value validation
    ###########################################################################

    operational_fields = [
        "portal_user_id",
        "email",
        "telephone_number",
        "address",
        "postcode",
    ]

    missing_summary = (
        crm_records[
            operational_fields
        ]
        .isna()
        .mean()
        .mul(100)
        .round(2)
    )

    print(
        "\nMissing value percentages:"
    )

    print(
        missing_summary
        .astype(str)
        .add("%")
    )

    ###########################################################################
    # Data quality issue counts
    ###########################################################################

    data_quality_columns = [
        column
        for column
        in crm_records.columns
        if column.startswith(
            "dq_"
        )
    ]

    data_quality_summary = (
        crm_records[
            data_quality_columns
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nIntroduced data quality issues:"
    )

    print(
        data_quality_summary
    )

    ###########################################################################
    # Create validation summary output
    ###########################################################################

    # Main validation metrics
    validation_summary = pd.DataFrame(
        {
            "metric": [
                "MAIN VALIDATION METRICS",
                "total_crm_records",
                "unique_customers",
                "duplicate_records",
                "portal_customers",
                "portal_customer_rate",
                "missing_email_rate",
                "missing_phone_rate",
                "missing_address_rate",
                "missing_postcode_rate",
            ],
            "value": [
                "",
                len(crm_records),
                crm_records[
                    "ground_truth_id"
                ].nunique(),
                crm_records[
                    "is_duplicate_record"
                ].sum(),
                portal_customer_count,
                round(
                    portal_customer_rate,
                    4,
                ),
                round(
                    crm_records[
                        "email"
                    ].isna().mean(),
                    4,
                ),
                round(
                    crm_records[
                        "telephone_number"
                    ].isna().mean(),
                    4,
                ),
                round(
                    crm_records[
                        "address"
                    ].isna().mean(),
                    4,
                ),
                round(
                    crm_records[
                        "postcode"
                    ].isna().mean(),
                    4,
                ),
            ],
        }
    )

    # CRM customer coverage by segment
    segment_summary = pd.DataFrame(
        {
            "metric": [
                "",
                "CRM CUSTOMER COVERAGE BY SEGMENT",
                "coverage_new",
                "coverage_occasional",
                "coverage_regular",
                "coverage_vip",
            ],
            "value": [
                "",
                "",
                f"{segment_coverage['New']:.2%}",
                f"{segment_coverage['Occasional']:.2%}",
                f"{segment_coverage['Regular']:.2%}",
                f"{segment_coverage['VIP']:.2%}",
            ],
        }
    )

    #  Missing value percentages
    missing_value_summary = pd.DataFrame(
        {
            "metric": [
                "",
                "MISSING VALUE PERCENTAGES",
                "missing_portal_user_id",
                "missing_email",
                "missing_telephone_number",
                "missing_address",
                "missing_postcode",
            ],
            "value": [
                "",
                "",
                f"{missing_summary['portal_user_id']:.2f}%",
                f"{missing_summary['email']:.2f}%",
                f"{missing_summary['telephone_number']:.2f}%",
                f"{missing_summary['address']:.2f}%",
                f"{missing_summary['postcode']:.2f}%",
            ],
        }
    )

    # Data quality issues validations
    data_quality_output = pd.DataFrame(
        {
            "metric": [
                "",
                "INTRODUCED DATA QUALITY ISSUES",
            ] + data_quality_summary.index.tolist(),

            "value": [
                "",
                "",
            ] + data_quality_summary.astype(int).tolist(),
        }
    )

    # Combine validation summary sections
    validation_summary = pd.concat(
        [
            validation_summary,
            segment_summary,
            missing_value_summary,
            data_quality_output,
        ],
        ignore_index=True,
    )

    return validation_summary


###############################################################################
# 12. Export CRM outputs
###############################################################################

def export_crm_outputs(
    crm_records: pd.DataFrame,
    validation_summary: pd.DataFrame,
) -> None:

    # Create output folders
    CRM_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GROUND_TRUTH_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Operational CRM dataset
    ###########################################################################

    operational_columns = [
        "crm_customer_id",
        "portal_user_id",
        "record_created_date",
        "preferred_contact_channel",
        "first_name",
        "surname",
        "date_of_birth",
        "email",
        "telephone_number",
        "address",
        "postcode",
    ]

    operational_crm = crm_records[
        operational_columns
    ].copy()

    # Convert date fields for clean CSV output
    operational_crm[
        "record_created_date"
    ] = pd.to_datetime(
        operational_crm[
            "record_created_date"
        ]
    ).dt.date

    operational_crm[
        "date_of_birth"
    ] = pd.to_datetime(
        operational_crm[
            "date_of_birth"
        ]
    ).dt.date

    operational_crm.to_csv(
        CRM_OUTPUT_PATH,
        index=False,
    )

    ###########################################################################
    # Hidden CRM ground-truth mapping
    ###########################################################################

    data_quality_columns = [
        column
        for column
        in crm_records.columns
        if column.startswith(
            "dq_"
        )
    ]

    ground_truth_columns = [
        "crm_customer_id",
        "ground_truth_id",
        "portal_user_id",
        "is_duplicate_record",
        "duplicate_of_crm_customer_id",
    ] + data_quality_columns

    crm_ground_truth_mapping = crm_records[
        ground_truth_columns
    ].copy()

    crm_ground_truth_mapping.to_csv(
        GROUND_TRUTH_OUTPUT_PATH,
        index=False,
    )

    ###########################################################################
    # Portal account mapping
    ###########################################################################

    # Mapping GTID to portal user_ID for later use in E-commerce
    # and online session logs ensuring consistent relationships
    # across CRM records and online logs
    portal_mapping = (
        crm_records[
            [
                "ground_truth_id",
                "portal_user_id",
            ]
        ]
        .dropna(
            subset=[
                "portal_user_id",
            ]
        )
        .drop_duplicates()
        .sort_values(
            "portal_user_id"
        )
        .reset_index(
            drop=True
        )
    )

    portal_mapping.to_csv(
        PORTAL_MAPPING_OUTPUT_PATH,
        index=False,
    )

    ###########################################################################
    # Validation summary
    ###########################################################################

    validation_summary.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )


###############################################################################
# 13. Main function
###############################################################################

def main() -> None:

    # Load canonical source data
    canonical = load_canonical_customers()

    # Select unique customers for CRM
    selected_customers = (
        select_crm_customers(
            canonical
        )
    )

    # Generate initial clean CRM records
    crm_records = (
        generate_clean_crm_records(
            selected_customers
        )
    )

    # Add intentionally duplicated CRM accounts
    crm_records = (
        add_duplicate_crm_records(
            crm_records
        )
    )

    # Introduce controlled data quality problems
    crm_records = (
        introduce_crm_data_quality_issues(
            crm_records
        )
    )

    # Validate final CRM environment
    validation_summary = (
        validate_crm_records(
            crm_records,
            canonical,
        )
    )

    # Export operational and reference datasets
    export_crm_outputs(
        crm_records,
        validation_summary,
    )

    print(
        f"\nCRM dataset saved to: "
        f"{CRM_OUTPUT_PATH.resolve()}"
    )

    print(
        f"CRM ground truth saved to: "
        f"{GROUND_TRUTH_OUTPUT_PATH.resolve()}"
    )

    print(
        f"Portal mapping saved to: "
        f"{PORTAL_MAPPING_OUTPUT_PATH.resolve()}"
    )

    print(
        "\nFirst five CRM records:"
    )

    print(
        crm_records[
            [
                "crm_customer_id",
                "portal_user_id",
                "record_created_date",
                "preferred_contact_channel",
                "first_name",
                "surname",
                "date_of_birth",
                "email",
                "telephone_number",
                "address",
                "postcode",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()
