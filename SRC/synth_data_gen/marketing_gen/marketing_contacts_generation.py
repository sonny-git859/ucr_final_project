###############################################################################
# Imports
###############################################################################

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


###############################################################################
# 1. Generation configuration
###############################################################################

# Marketing-specific seed ensures reproducibility
MARKETING_SEED = 109

# Target operational dataset size
NUMBER_OF_MARKETING_CONTACTS = 12_000

# Number of unique canonical customers represented
UNIQUE_MARKETING_CUSTOMERS = 8_000

# Operational snapshot
OPERATION_START_DATE = pd.Timestamp(
    "2025-01-01"
)

SNAPSHOT_DATE = pd.Timestamp(
    "2025-12-31"
)

# Input datasets
CANONICAL_INPUT_PATH = Path(
    "data/canonical/canonical_customers.csv"
)

CRM_GROUND_TRUTH_INPUT_PATH = Path(
    "data/reference/crm_ground_truth_mapping.csv"
)

# Output paths
MARKETING_OUTPUT_PATH = Path(
    "data/raw/marketing_contact_lists.csv"
)

GROUND_TRUTH_OUTPUT_PATH = Path(
    "data/reference/marketing_ground_truth_mapping.csv"
)

VALIDATION_OUTPUT_PATH = Path(
    "data/reference/marketing_validation_summary.csv"
)


###############################################################################
# 2. Weighting assumptions
###############################################################################

# Customer inclusion is weighted using canonical customer_segment.
# Higher-engagement segments are more likely to appear in Marketing records.
MARKETING_SEGMENT_SELECTION_WEIGHTS = {
    "New": 0.80,
    "Occasional": 1.00,
    "Regular": 1.20,
    "VIP": 1.40,
}

# Higher-engagement segments are more likely to have multiple
# Marketing contact records.
REPEAT_CONTACT_WEIGHTS = {
    "New": 0.80,
    "Occasional": 1.00,
    "Regular": 1.20,
    "VIP": 1.40,
}

# Current recorded marketing consent is more likely among
# higher-engagement customer segments.
CONSENT_PROBABILITIES = {
    "New": 0.45,
    "Occasional": 0.60,
    "Regular": 0.75,
    "VIP": 0.85,
}

# Historical email engagement increases with customer segment.
EMAIL_OPEN_MEANS = {
    "New": 2.5,
    "Occasional": 4.5,
    "Regular": 7.0,
    "VIP": 10.0,
}

# Probability that an opened email contributes a recorded link click.
LINK_CLICK_PROBABILITIES = {
    "New": 0.10,
    "Occasional": 0.15,
    "Regular": 0.25,
    "VIP": 0.35,
}


###############################################################################
# 3. Marketing behaviour and data quality assumptions
###############################################################################

# Contact names may be missing or contain manual-entry errors.
MISSING_CONTACT_NAME_RATE = 0.05
CONTACT_NAME_TYPO_RATE = 0.05

# Marketing email addresses may be alternative or outdated.
ALTERNATIVE_EMAIL_RATE = 0.05
OUTDATED_EMAIL_RATE = 0.05

# Marketing postcodes may be incomplete, outdated or alternative.
INCOMPLETE_POSTCODE_RATE = 0.05
OUTDATED_POSTCODE_RATE = 0.08
ALTERNATIVE_POSTCODE_RATE = 0.05

# Common domains used when generating alternative or outdated emails.
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


###############################################################################
# 4. Initialise generator
###############################################################################

rng = np.random.default_rng(
    MARKETING_SEED
)


###############################################################################
# 5. Helper functions
###############################################################################

# Select a random date between two date boundaries.
def generate_random_date(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Timestamp:

    start_date = pd.Timestamp(
        start_date
    ).normalize()

    end_date = pd.Timestamp(
        end_date
    ).normalize()

    total_days = int(
        (
            end_date
            - start_date
        ).days
    )

    if total_days <= 0:
        return start_date

    random_days = int(
        rng.integers(
            0,
            total_days + 1,
        )
    )

    return (
        start_date
        + pd.Timedelta(
            days=random_days
        )
    )


# Normalise names for use within synthetic alternative email addresses.
def normalise_for_email(
    value: str,
) -> str:

    value = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    value = (
        value
        .encode(
            "ascii",
            "ignore",
        )
        .decode(
            "ascii"
        )
    )

    value = value.lower()

    value = re.sub(
        r"[^a-z0-9]",
        "",
        value,
    )

    return value


# Introduce a small spelling error into a value.
def introduce_name_typo(
    value: str,
) -> str:

    if pd.isna(value):
        return value

    value = str(value)

    if len(value) < 3:
        return value

    characters = list(
        value
    )

    typo_type = str(
        rng.choice(
            [
                "swap",
                "remove",
            ]
        )
    )

    if (
        typo_type == "swap"
        and len(characters) >= 3
    ):

        position = int(
            rng.integers(
                0,
                len(characters) - 1,
            )
        )

        (
            characters[position],
            characters[position + 1],
        ) = (
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

        del characters[
            position
        ]

    return "".join(
        characters
    )


# Introduce a name typo while retaining firstname_surname formatting.
def introduce_contact_name_typo(
    contact_name: str,
) -> str:

    if (
        pd.isna(contact_name)
        or "_" not in str(contact_name)
    ):
        return contact_name

    first_name, surname = (
        str(contact_name)
        .split(
            "_",
            1,
        )
    )

    selected_field = str(
        rng.choice(
            [
                "first_name",
                "surname",
            ]
        )
    )

    if selected_field == "first_name":
        first_name = introduce_name_typo(
            first_name
        )
    else:
        surname = introduce_name_typo(
            surname
        )

    return (
        f"{first_name}_"
        f"{surname}"
    )


# Generate a plausible alternative email for the same canonical customer.
def generate_email_variant(
    first_name: str,
    surname: str,
    registration_date: pd.Timestamp,
    current_email: str,
    used_emails: set[str],
) -> str:

    first = normalise_for_email(
        first_name
    )

    last = normalise_for_email(
        surname
    )

    year_suffix = str(
        pd.Timestamp(
            registration_date
        ).year
    )[-2:]

    current_domain = (
        str(current_email)
        .split("@")[-1]
        .lower()
    )

    available_domains = [
        domain
        for domain
        in EMAIL_DOMAINS
        if domain != current_domain
    ]

    if not available_domains:
        available_domains = EMAIL_DOMAINS

    domain = str(
        rng.choice(
            available_domains
        )
    )

    local_part_options = [
        f"{first}{last}{year_suffix}",
        f"{first}.{last}{year_suffix}",
        f"{first[0]}{last}{year_suffix}",
        f"{first}{last}",
    ]

    local_part = str(
        rng.choice(
            local_part_options
        )
    )

    email = (
        f"{local_part}@{domain}"
    )

    suffix = 2

    while email.lower() in used_emails:

        email = (
            f"{local_part}{suffix}"
            f"@{domain}"
        )

        suffix += 1

    used_emails.add(
        email.lower()
    )

    return email


# Return the outward portion of a UK postcode as an incomplete value.
def make_postcode_incomplete(
    postcode: str,
) -> str:

    if pd.isna(postcode):
        return postcode

    compact_postcode = re.sub(
        r"\s+",
        "",
        str(postcode).upper(),
    )

    if len(compact_postcode) <= 3:
        return compact_postcode

    return compact_postcode[:-3]


# Generate a different postcode while retaining the outward district.
def generate_postcode_variant(
    postcode: str,
) -> str:

    if pd.isna(postcode):
        return postcode

    compact_postcode = re.sub(
        r"\s+",
        "",
        str(postcode).upper(),
    )

    if len(compact_postcode) <= 3:
        outward_code = compact_postcode
    else:
        outward_code = compact_postcode[:-3]

    if not outward_code:
        outward_code = "ZZ1"

    original_postcode = compact_postcode

    while True:

        inward_digit = str(
            int(
                rng.integers(
                    0,
                    10,
                )
            )
        )

        first_letter = str(
            rng.choice(
                list(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                )
            )
        )

        second_letter = str(
            rng.choice(
                list(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                )
            )
        )

        variant = (
            f"{outward_code} "
            f"{inward_digit}"
            f"{first_letter}"
            f"{second_letter}"
        )

        if (
            variant.replace(" ", "")
            != original_postcode
        ):
            return variant


###############################################################################
# 6. Load and validate source datasets
###############################################################################

def load_source_datasets() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    ###########################################################################
    # Canonical customers
    ###########################################################################

    canonical = pd.read_csv(
        CANONICAL_INPUT_PATH
    )

    expected_canonical_columns = [
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

    assert list(
        canonical.columns
    ) == expected_canonical_columns

    assert canonical[
        "ground_truth_id"
    ].is_unique

    canonical[
        "registration_date"
    ] = pd.to_datetime(
        canonical[
            "registration_date"
        ]
    )

    ###########################################################################
    # CRM ground-truth mapping
    ###########################################################################

    crm_ground_truth = pd.read_csv(
        CRM_GROUND_TRUTH_INPUT_PATH
    )

    required_crm_columns = [
        "ground_truth_id",
    ]

    assert set(
        required_crm_columns
    ).issubset(
        set(
            crm_ground_truth.columns
        )
    )

    assert crm_ground_truth[
        "ground_truth_id"
    ].notna().all()

    return (
        canonical,
        crm_ground_truth,
    )


###############################################################################
# 7. Select Marketing customer population
###############################################################################

def select_marketing_customers(
    canonical: pd.DataFrame,
) -> pd.DataFrame:

    eligible_customers = (
        canonical[
            canonical[
                "registration_date"
            ]
            <=
            SNAPSHOT_DATE
        ]
        .copy()
    )

    assert (
        len(
            eligible_customers
        )
        >=
        UNIQUE_MARKETING_CUSTOMERS
    )

    selection_weights = (
        eligible_customers[
            "customer_segment"
        ]
        .map(
            MARKETING_SEGMENT_SELECTION_WEIGHTS
        )
        .astype(float)
    )

    selection_weights = (
        selection_weights
        / selection_weights.sum()
    )

    selected_indices = rng.choice(
        eligible_customers.index.to_numpy(),
        size=UNIQUE_MARKETING_CUSTOMERS,
        replace=False,
        p=selection_weights.to_numpy(),
    )

    selected_customers = (
        eligible_customers
        .loc[
            selected_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return selected_customers


###############################################################################
# 8. Generate Marketing contact plan
###############################################################################

def generate_marketing_contact_plan(
    selected_customers: pd.DataFrame,
) -> pd.DataFrame:

    marketing_contact_plan = []

    ###########################################################################
    # Ensure every selected customer is represented at least once
    ###########################################################################

    for customer in (
        selected_customers
        .itertuples(
            index=False
        )
    ):

        marketing_contact_plan.append(
            {
                "ground_truth_id": customer.ground_truth_id,
                "is_repeat_contact": False,
            }
        )

    ###########################################################################
    # Generate remaining repeated contacts using segment weighting
    ###########################################################################

    remaining_contact_count = (
        NUMBER_OF_MARKETING_CONTACTS
        - len(
            marketing_contact_plan
        )
    )

    if remaining_contact_count < 0:

        raise ValueError(
            "Unique Marketing customer target exceeds the configured "
            "Marketing contact target."
        )

    repeat_weights = (
        selected_customers[
            "customer_segment"
        ]
        .map(
            REPEAT_CONTACT_WEIGHTS
        )
        .astype(float)
    )

    repeat_weights = (
        repeat_weights
        / repeat_weights.sum()
    )

    repeated_customer_indices = rng.choice(
        selected_customers.index.to_numpy(),
        size=remaining_contact_count,
        replace=True,
        p=repeat_weights.to_numpy(),
    )

    for customer_index in (
        repeated_customer_indices
    ):

        customer = selected_customers.loc[
            customer_index
        ]

        marketing_contact_plan.append(
            {
                "ground_truth_id": customer[
                    "ground_truth_id"
                ],
                "is_repeat_contact": True,
            }
        )

    marketing_contact_plan = pd.DataFrame(
        marketing_contact_plan
    )

    marketing_contact_plan = (
        marketing_contact_plan
        .merge(
            selected_customers,
            on="ground_truth_id",
            how="left",
            validate="many_to_one",
        )
    )

    assert (
        len(
            marketing_contact_plan
        )
        ==
        NUMBER_OF_MARKETING_CONTACTS
    )

    assert (
        marketing_contact_plan[
            "ground_truth_id"
        ].nunique()
        ==
        UNIQUE_MARKETING_CUSTOMERS
    )

    assert (
        marketing_contact_plan[
            "is_repeat_contact"
        ].sum()
        ==
        (
            NUMBER_OF_MARKETING_CONTACTS
            - UNIQUE_MARKETING_CUSTOMERS
        )
    )

    return marketing_contact_plan


###############################################################################
# 9. Generate alternative Marketing identity mappings
###############################################################################

def generate_marketing_identity_mappings(
    selected_customers: pd.DataFrame,
    canonical: pd.DataFrame,
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:

    alternative_email_mapping = {}
    outdated_email_mapping = {}
    alternative_postcode_mapping = {}
    outdated_postcode_mapping = {}

    used_emails = set(
        canonical[
            "email"
        ]
        .astype(str)
        .str.lower()
        .tolist()
    )

    for customer in (
        selected_customers
        .itertuples(
            index=False
        )
    ):

        alternative_email_mapping[
            customer.ground_truth_id
        ] = generate_email_variant(
            first_name=customer.first_name,
            surname=customer.surname,
            registration_date=pd.Timestamp(
                customer.registration_date
            ),
            current_email=customer.email,
            used_emails=used_emails,
        )

        outdated_email_mapping[
            customer.ground_truth_id
        ] = generate_email_variant(
            first_name=customer.first_name,
            surname=customer.surname,
            registration_date=pd.Timestamp(
                customer.registration_date
            ),
            current_email=customer.email,
            used_emails=used_emails,
        )

        alternative_postcode_mapping[
            customer.ground_truth_id
        ] = generate_postcode_variant(
            customer.postcode
        )

        outdated_postcode_mapping[
            customer.ground_truth_id
        ] = generate_postcode_variant(
            customer.postcode
        )

    return (
        alternative_email_mapping,
        outdated_email_mapping,
        alternative_postcode_mapping,
        outdated_postcode_mapping,
    )


###############################################################################
# 10. Generate clean Marketing contact records
###############################################################################

def generate_clean_marketing_records(
    marketing_contact_plan: pd.DataFrame,
) -> pd.DataFrame:

    marketing_records = []

    for contact in (
        marketing_contact_plan
        .itertuples(
            index=False
        )
    ):

        #######################################################################
        # Generate most recent Marketing contact date
        #######################################################################

        earliest_contact_date = max(
            OPERATION_START_DATE,
            pd.Timestamp(
                contact.registration_date
            ).normalize(),
        )

        last_contact_date = generate_random_date(
            start_date=earliest_contact_date,
            end_date=SNAPSHOT_DATE,
        )

        #######################################################################
        # Generate current consent status
        #######################################################################

        consent_probability = (
            CONSENT_PROBABILITIES[
                contact.customer_segment
            ]
        )

        consent_status = bool(
            rng.random()
            <
            consent_probability
        )

        #######################################################################
        # Generate historical Marketing engagement
        #######################################################################

        emails_opened_count = int(
            rng.poisson(
                EMAIL_OPEN_MEANS[
                    contact.customer_segment
                ]
            )
        )

        links_clicked_count = int(
            rng.binomial(
                n=emails_opened_count,
                p=LINK_CLICK_PROBABILITIES[
                    contact.customer_segment
                ],
            )
        )

        #######################################################################
        # Append clean Marketing contact
        #######################################################################

        marketing_records.append(
            {
                # Internal generation fields
                "ground_truth_id": contact.ground_truth_id,
                "customer_segment": contact.customer_segment,
                "registration_date": contact.registration_date,
                "is_repeat_contact": bool(
                    contact.is_repeat_contact
                ),

                # Operational Marketing fields
                "marketing_contact_id": pd.NA,
                "contact_name": (
                    f"{contact.first_name}_"
                    f"{contact.surname}"
                ),
                "last_contact_date": last_contact_date,
                "consent_status": consent_status,
                "emails_opened_count": emails_opened_count,
                "links_clicked_count": links_clicked_count,
                "email": contact.email,
                "postcode": contact.postcode,
            }
        )

    return pd.DataFrame(
        marketing_records
    )


###############################################################################
# 11. Introduce Marketing data quality issues
###############################################################################

def introduce_marketing_data_quality_issues(
    marketing_records: pd.DataFrame,
    alternative_email_mapping: dict[str, str],
    outdated_email_mapping: dict[str, str],
    alternative_postcode_mapping: dict[str, str],
    outdated_postcode_mapping: dict[str, str],
) -> pd.DataFrame:

    marketing_records = (
        marketing_records
        .copy()
    )

    data_quality_flags = [
        "dq_duplicate_contact",
        "dq_missing_contact_name",
        "dq_contact_name_typo",
        "dq_alternative_email",
        "dq_outdated_email",
        "dq_incomplete_postcode",
        "dq_outdated_postcode",
        "dq_alternative_postcode",
    ]

    for flag in (
        data_quality_flags
    ):

        marketing_records[
            flag
        ] = False

    marketing_records[
        "dq_duplicate_contact"
    ] = marketing_records[
        "is_repeat_contact"
    ]

    for index in (
        marketing_records.index
    ):

        ground_truth_id = (
            marketing_records.at[
                index,
                "ground_truth_id",
            ]
        )

        #######################################################################
        # Contact name
        #######################################################################

        if (
            rng.random()
            <
            MISSING_CONTACT_NAME_RATE
        ):

            marketing_records.at[
                index,
                "contact_name",
            ] = pd.NA

            marketing_records.at[
                index,
                "dq_missing_contact_name",
            ] = True

        elif (
            rng.random()
            <
            CONTACT_NAME_TYPO_RATE
        ):

            marketing_records.at[
                index,
                "contact_name",
            ] = introduce_contact_name_typo(
                marketing_records.at[
                    index,
                    "contact_name",
                ]
            )

            marketing_records.at[
                index,
                "dq_contact_name_typo",
            ] = True

        #######################################################################
        # Email address
        #######################################################################

        if (
            rng.random()
            <
            ALTERNATIVE_EMAIL_RATE
        ):

            marketing_records.at[
                index,
                "email",
            ] = alternative_email_mapping[
                ground_truth_id
            ]

            marketing_records.at[
                index,
                "dq_alternative_email",
            ] = True

        elif (
            rng.random()
            <
            OUTDATED_EMAIL_RATE
        ):

            marketing_records.at[
                index,
                "email",
            ] = outdated_email_mapping[
                ground_truth_id
            ]

            marketing_records.at[
                index,
                "dq_outdated_email",
            ] = True

        #######################################################################
        # Postcode
        #######################################################################

        if (
            rng.random()
            <
            INCOMPLETE_POSTCODE_RATE
        ):

            marketing_records.at[
                index,
                "postcode",
            ] = make_postcode_incomplete(
                marketing_records.at[
                    index,
                    "postcode",
                ]
            )

            marketing_records.at[
                index,
                "dq_incomplete_postcode",
            ] = True

        elif (
            rng.random()
            <
            OUTDATED_POSTCODE_RATE
        ):

            marketing_records.at[
                index,
                "postcode",
            ] = outdated_postcode_mapping[
                ground_truth_id
            ]

            marketing_records.at[
                index,
                "dq_outdated_postcode",
            ] = True

        elif (
            rng.random()
            <
            ALTERNATIVE_POSTCODE_RATE
        ):

            marketing_records.at[
                index,
                "postcode",
            ] = alternative_postcode_mapping[
                ground_truth_id
            ]

            marketing_records.at[
                index,
                "dq_alternative_postcode",
            ] = True

    return marketing_records


###############################################################################
# 12. Finalise Marketing contact IDs
###############################################################################

def finalise_marketing_contact_ids(
    marketing_records: pd.DataFrame,
) -> pd.DataFrame:

    # Sort contacts by most recent contact date before assigning
    # sequential Marketing identifiers.
    marketing_records = (
        marketing_records
        .sort_values(
            "last_contact_date"
        )
        .reset_index(
            drop=True
        )
    )

    marketing_records[
        "marketing_contact_id"
    ] = [
        f"MKT{contact_number:06d}"
        for contact_number
        in range(
            1,
            len(
                marketing_records
            ) + 1,
        )
    ]

    return marketing_records


###############################################################################
# 13. Validate Marketing Contact dataset
###############################################################################

def validate_marketing_records(
    marketing_records: pd.DataFrame,
    canonical: pd.DataFrame,
    crm_ground_truth: pd.DataFrame,
) -> pd.DataFrame:

    ###########################################################################
    # Basic validation
    ###########################################################################

    assert (
        len(
            marketing_records
        )
        ==
        NUMBER_OF_MARKETING_CONTACTS
    )

    assert marketing_records[
        "marketing_contact_id"
    ].is_unique

    assert (
        marketing_records[
            "ground_truth_id"
        ].nunique()
        ==
        UNIQUE_MARKETING_CUSTOMERS
    )

    assert set(
        marketing_records[
            "ground_truth_id"
        ]
    ).issubset(
        set(
            canonical[
                "ground_truth_id"
            ]
        )
    )

    ###########################################################################
    # Duplicate and repeated-contact validation
    ###########################################################################

    expected_repeat_contacts = (
        NUMBER_OF_MARKETING_CONTACTS
        - UNIQUE_MARKETING_CUSTOMERS
    )

    actual_repeat_contacts = int(
        marketing_records[
            "is_repeat_contact"
        ].sum()
    )

    assert (
        actual_repeat_contacts
        ==
        expected_repeat_contacts
    )

    assert (
        marketing_records[
            "dq_duplicate_contact"
        ]
        ==
        marketing_records[
            "is_repeat_contact"
        ]
    ).all()

    ###########################################################################
    # Contact-date validation
    ###########################################################################

    last_contact_dates = pd.to_datetime(
        marketing_records[
            "last_contact_date"
        ]
    )

    registration_dates = pd.to_datetime(
        marketing_records[
            "registration_date"
        ]
    ).dt.normalize()

    assert (
        last_contact_dates
        >=
        OPERATION_START_DATE
    ).all()

    assert (
        last_contact_dates
        <=
        SNAPSHOT_DATE
    ).all()

    assert (
        last_contact_dates
        >=
        registration_dates
    ).all()

    ###########################################################################
    # Operational field validation
    ###########################################################################

    assert marketing_records[
        "email"
    ].notna().all()

    assert marketing_records[
        "postcode"
    ].notna().all()

    non_missing_names = marketing_records[
        marketing_records[
            "contact_name"
        ].notna()
    ]

    assert non_missing_names[
        "contact_name"
    ].astype(str).str.fullmatch(
        r".+_.+"
    ).all()

    assert marketing_records[
        "consent_status"
    ].isin(
        [
            True,
            False,
        ]
    ).all()

    assert (
        marketing_records[
            "emails_opened_count"
        ]
        >=
        0
    ).all()

    assert (
        marketing_records[
            "links_clicked_count"
        ]
        >=
        0
    ).all()

    assert (
        marketing_records[
            "links_clicked_count"
        ]
        <=
        marketing_records[
            "emails_opened_count"
        ]
    ).all()

    ###########################################################################
    # Marketing-only customer validation
    ###########################################################################

    marketing_customer_ids = set(
        marketing_records[
            "ground_truth_id"
        ]
    )

    crm_customer_ids = set(
        crm_ground_truth[
            "ground_truth_id"
        ]
    )

    marketing_only_customer_ids = (
        marketing_customer_ids
        - crm_customer_ids
    )

    marketing_only_customer_count = len(
        marketing_only_customer_ids
    )

    assert marketing_only_customer_count > 0

    marketing_only_customer_rate = (
        marketing_only_customer_count
        /
        UNIQUE_MARKETING_CUSTOMERS
    )

    ###########################################################################
    # Validation metrics
    ###########################################################################

    total_contacts = len(
        marketing_records
    )

    unique_customers = (
        marketing_records[
            "ground_truth_id"
        ].nunique()
    )

    duplicate_contact_rate = (
        actual_repeat_contacts
        /
        total_contacts
    )

    consent_rate = (
        marketing_records[
            "consent_status"
        ].mean()
    )

    nonconsenting_contacts = marketing_records[
        ~marketing_records[
            "consent_status"
        ]
    ]

    nonconsenting_with_engagement = int(
        (
            (
                nonconsenting_contacts[
                    "emails_opened_count"
                ]
                >
                0
            )
            |
            (
                nonconsenting_contacts[
                    "links_clicked_count"
                ]
                >
                0
            )
        ).sum()
    )

    assert nonconsenting_with_engagement > 0

    ###########################################################################
    # Customer coverage by segment
    ###########################################################################

    unique_marketing_customers = (
        marketing_records[
            [
                "ground_truth_id",
                "customer_segment",
            ]
        ]
        .drop_duplicates(
            "ground_truth_id"
        )
    )

    canonical_segment_counts = (
        canonical[
            "customer_segment"
        ]
        .value_counts()
    )

    marketing_segment_counts = (
        unique_marketing_customers[
            "customer_segment"
        ]
        .value_counts()
    )

    segment_coverage = (
        marketing_segment_counts
        /
        canonical_segment_counts
    ).fillna(0)

    segment_coverage = (
        segment_coverage
        .reindex(
            [
                "New",
                "Occasional",
                "Regular",
                "VIP",
            ]
        )
    )

    ###########################################################################
    # Average contact frequency by segment
    ###########################################################################

    customer_contact_counts = (
        marketing_records
        .groupby(
            [
                "ground_truth_id",
                "customer_segment",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "contact_count",
            }
        )
    )

    average_contacts_by_segment = (
        customer_contact_counts
        .groupby(
            "customer_segment"
        )[
            "contact_count"
        ]
        .mean()
        .reindex(
            [
                "New",
                "Occasional",
                "Regular",
                "VIP",
            ]
        )
    )

    ###########################################################################
    # Consent and engagement behaviour by segment
    ###########################################################################

    consent_rate_by_segment = (
        marketing_records
        .groupby(
            "customer_segment"
        )[
            "consent_status"
        ]
        .mean()
        .reindex(
            [
                "New",
                "Occasional",
                "Regular",
                "VIP",
            ]
        )
    )

    average_opens_by_segment = (
        marketing_records
        .groupby(
            "customer_segment"
        )[
            "emails_opened_count"
        ]
        .mean()
        .reindex(
            [
                "New",
                "Occasional",
                "Regular",
                "VIP",
            ]
        )
    )

    average_clicks_by_segment = (
        marketing_records
        .groupby(
            "customer_segment"
        )[
            "links_clicked_count"
        ]
        .mean()
        .reindex(
            [
                "New",
                "Occasional",
                "Regular",
                "VIP",
            ]
        )
    )

    assert (
        consent_rate_by_segment.diff().dropna()
        >
        0
    ).all()

    assert (
        average_opens_by_segment.diff().dropna()
        >
        0
    ).all()

    assert (
        average_clicks_by_segment.diff().dropna()
        >
        0
    ).all()

    ###########################################################################
    # Data quality issue counts
    ###########################################################################

    data_quality_columns = [
        column
        for column
        in marketing_records.columns
        if column.startswith(
            "dq_"
        )
    ]

    data_quality_summary = (
        marketing_records[
            data_quality_columns
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    ###########################################################################
    # Visualise validation results
    ###########################################################################

    print(
        "\nMarketing Contact validation completed successfully."
    )

    print(
        f"Total marketing contacts: "
        f"{total_contacts:,}"
    )

    print(
        f"Unique customers represented: "
        f"{unique_customers:,}"
    )

    print(
        f"Repeated / duplicate contact records: "
        f"{actual_repeat_contacts:,}"
    )

    print(
        f"Duplicate contact rate: "
        f"{duplicate_contact_rate:.2%}"
    )

    print(
        f"Marketing-only customers not represented in CRM: "
        f"{marketing_only_customer_count:,}"
    )

    print(
        f"Marketing-only customer rate: "
        f"{marketing_only_customer_rate:.2%}"
    )

    print(
        f"Current consent rate: "
        f"{consent_rate:.2%}"
    )

    print(
        f"Non-consenting contacts with historical engagement: "
        f"{nonconsenting_with_engagement:,}"
    )

    print(
        "\nMarketing customer coverage by segment:"
    )

    print(
        segment_coverage
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    print(
        "\nAverage marketing contacts per customer by segment:"
    )

    print(
        average_contacts_by_segment
        .round(2)
    )

    print(
        "\nConsent rate by customer segment:"
    )

    print(
        consent_rate_by_segment
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    print(
        "\nAverage emails opened by customer segment:"
    )

    print(
        average_opens_by_segment
        .round(2)
    )

    print(
        "\nAverage links clicked by customer segment:"
    )

    print(
        average_clicks_by_segment
        .round(2)
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

    validation_summary = pd.DataFrame(
        {
            "metric": [
                "MAIN VALIDATION METRICS",
                "total_marketing_contacts",
                "unique_customers",
                "duplicate_contact_records",
                "duplicate_contact_rate",
                "marketing_only_customers",
                "marketing_only_customer_rate",
                "current_consent_rate",
                "nonconsenting_contacts_with_historical_engagement",
                "contact_timing_valid",
                "email_values_present",
                "postcode_values_present",
                "engagement_counts_nonnegative",
                "link_clicks_not_greater_than_opens",
            ],
            "value": [
                "",
                total_contacts,
                unique_customers,
                actual_repeat_contacts,
                round(
                    duplicate_contact_rate,
                    4,
                ),
                marketing_only_customer_count,
                round(
                    marketing_only_customer_rate,
                    4,
                ),
                round(
                    consent_rate,
                    4,
                ),
                nonconsenting_with_engagement,
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )

    ###########################################################################
    # Customer coverage by segment
    ###########################################################################

    segment_summary = pd.DataFrame(
        {
            "metric": [
                "",
                "MARKETING CUSTOMER COVERAGE BY SEGMENT",
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

    ###########################################################################
    # Average Marketing contact frequency by segment
    ###########################################################################

    frequency_summary = pd.DataFrame(
        {
            "metric": [
                "",
                "AVERAGE MARKETING CONTACTS PER CUSTOMER BY SEGMENT",
                "average_contacts_new",
                "average_contacts_occasional",
                "average_contacts_regular",
                "average_contacts_vip",
            ],
            "value": [
                "",
                "",
                round(
                    average_contacts_by_segment[
                        "New"
                    ],
                    2,
                ),
                round(
                    average_contacts_by_segment[
                        "Occasional"
                    ],
                    2,
                ),
                round(
                    average_contacts_by_segment[
                        "Regular"
                    ],
                    2,
                ),
                round(
                    average_contacts_by_segment[
                        "VIP"
                    ],
                    2,
                ),
            ],
        }
    )

    ###########################################################################
    # Consent behaviour by segment
    ###########################################################################

    consent_summary = pd.DataFrame(
        {
            "metric": [
                "",
                "CONSENT RATE BY CUSTOMER SEGMENT",
                "consent_rate_new",
                "consent_rate_occasional",
                "consent_rate_regular",
                "consent_rate_vip",
            ],
            "value": [
                "",
                "",
                f"{consent_rate_by_segment['New']:.2%}",
                f"{consent_rate_by_segment['Occasional']:.2%}",
                f"{consent_rate_by_segment['Regular']:.2%}",
                f"{consent_rate_by_segment['VIP']:.2%}",
            ],
        }
    )

    ###########################################################################
    # Historical Marketing engagement by segment
    ###########################################################################

    engagement_summary = pd.DataFrame(
        {
            "metric": [
                "",
                "AVERAGE MARKETING ENGAGEMENT BY SEGMENT",
                "average_emails_opened_new",
                "average_emails_opened_occasional",
                "average_emails_opened_regular",
                "average_emails_opened_vip",
                "average_links_clicked_new",
                "average_links_clicked_occasional",
                "average_links_clicked_regular",
                "average_links_clicked_vip",
            ],
            "value": [
                "",
                "",
                round(
                    average_opens_by_segment[
                        "New"
                    ],
                    2,
                ),
                round(
                    average_opens_by_segment[
                        "Occasional"
                    ],
                    2,
                ),
                round(
                    average_opens_by_segment[
                        "Regular"
                    ],
                    2,
                ),
                round(
                    average_opens_by_segment[
                        "VIP"
                    ],
                    2,
                ),
                round(
                    average_clicks_by_segment[
                        "New"
                    ],
                    2,
                ),
                round(
                    average_clicks_by_segment[
                        "Occasional"
                    ],
                    2,
                ),
                round(
                    average_clicks_by_segment[
                        "Regular"
                    ],
                    2,
                ),
                round(
                    average_clicks_by_segment[
                        "VIP"
                    ],
                    2,
                ),
            ],
        }
    )

    ###########################################################################
    # Data quality issues
    ###########################################################################

    data_quality_output = pd.DataFrame(
        {
            "metric": [
                "",
                "INTRODUCED DATA QUALITY ISSUES",
            ]
            +
            data_quality_summary.index.tolist(),

            "value": [
                "",
                "",
            ]
            +
            data_quality_summary.astype(
                int
            ).tolist(),
        }
    )

    ###########################################################################
    # Combine validation summary sections
    ###########################################################################

    validation_summary = pd.concat(
        [
            validation_summary,
            segment_summary,
            frequency_summary,
            consent_summary,
            engagement_summary,
            data_quality_output,
        ],
        ignore_index=True,
    )

    return validation_summary


###############################################################################
# 14. Export Marketing outputs
###############################################################################

def export_marketing_outputs(
    marketing_records: pd.DataFrame,
    validation_summary: pd.DataFrame,
) -> None:

    # Create output folders
    MARKETING_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GROUND_TRUTH_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Operational Marketing dataset
    ###########################################################################

    operational_columns = [
        "marketing_contact_id",
        "contact_name",
        "last_contact_date",
        "consent_status",
        "emails_opened_count",
        "links_clicked_count",
        "email",
        "postcode",
    ]

    operational_marketing = (
        marketing_records[
            operational_columns
        ]
        .copy()
    )

    operational_marketing[
        "last_contact_date"
    ] = pd.to_datetime(
        operational_marketing[
            "last_contact_date"
        ]
    ).dt.date

    operational_marketing.to_csv(
        MARKETING_OUTPUT_PATH,
        index=False,
    )

    ###########################################################################
    # Hidden Marketing ground truth mapping
    ###########################################################################

    data_quality_columns = [
        column
        for column
        in marketing_records.columns
        if column.startswith(
            "dq_"
        )
    ]

    ground_truth_columns = [
        "marketing_contact_id",
        "ground_truth_id",
        "customer_segment",
        "is_repeat_contact",
        "consent_status",
    ] + data_quality_columns

    marketing_ground_truth_mapping = (
        marketing_records[
            ground_truth_columns
        ]
        .copy()
    )

    marketing_ground_truth_mapping.to_csv(
        GROUND_TRUTH_OUTPUT_PATH,
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
# 15. Main function
###############################################################################

def main() -> None:

    # Load canonical population and CRM ground-truth reference
    (
        canonical,
        crm_ground_truth,
    ) = load_source_datasets()

    # Select exactly 8,000 unique Marketing customers
    selected_customers = (
        select_marketing_customers(
            canonical=canonical
        )
    )

    # Generate exactly 12,000 Marketing contact plans
    marketing_contact_plan = (
        generate_marketing_contact_plan(
            selected_customers=selected_customers
        )
    )

    # Generate alternative and outdated identity values
    (
        alternative_email_mapping,
        outdated_email_mapping,
        alternative_postcode_mapping,
        outdated_postcode_mapping,
    ) = generate_marketing_identity_mappings(
        selected_customers=selected_customers,
        canonical=canonical,
    )

    # Generate clean Marketing contact records
    marketing_records = (
        generate_clean_marketing_records(
            marketing_contact_plan=(
                marketing_contact_plan
            )
        )
    )

    # Introduce controlled Marketing data quality issues
    marketing_records = (
        introduce_marketing_data_quality_issues(
            marketing_records=marketing_records,
            alternative_email_mapping=(
                alternative_email_mapping
            ),
            outdated_email_mapping=(
                outdated_email_mapping
            ),
            alternative_postcode_mapping=(
                alternative_postcode_mapping
            ),
            outdated_postcode_mapping=(
                outdated_postcode_mapping
            ),
        )
    )

    # Sort by contact date and assign Marketing contact IDs
    marketing_records = (
        finalise_marketing_contact_ids(
            marketing_records=marketing_records
        )
    )

    # Validate final Marketing environment
    validation_summary = (
        validate_marketing_records(
            marketing_records=marketing_records,
            canonical=canonical,
            crm_ground_truth=crm_ground_truth,
        )
    )

    # Export operational and reference datasets
    export_marketing_outputs(
        marketing_records=marketing_records,
        validation_summary=validation_summary,
    )

    print(
        f"\nMarketing dataset saved to: "
        f"{MARKETING_OUTPUT_PATH.resolve()}"
    )

    print(
        f"Marketing ground truth saved to: "
        f"{GROUND_TRUTH_OUTPUT_PATH.resolve()}"
    )

    print(
        f"Marketing validation summary saved to: "
        f"{VALIDATION_OUTPUT_PATH.resolve()}"
    )

    print(
        "\nFirst five Marketing contact records:"
    )

    print(
        marketing_records[
            [
                "marketing_contact_id",
                "contact_name",
                "last_contact_date",
                "consent_status",
                "emails_opened_count",
                "links_clicked_count",
                "email",
                "postcode",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()
