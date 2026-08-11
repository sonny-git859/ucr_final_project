###############################################################################
# Imports
###############################################################################

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from faker import Faker


###############################################################################
# 1. Generation configuration
###############################################################################

# E-commerce-specific seed ensures reproducibility
ECOMMERCE_SEED = 103

# Number of unique canonical customers represented as purchasers
UNIQUE_ECOMMERCE_CUSTOMERS = 8_000

# Total expected attendance across Events reference dataset
EXPECTED_TOTAL_ATTENDANCE = 20_082

# Operational snapshot
OPERATION_START_DATE = pd.Timestamp("2025-01-01")
SNAPSHOT_DATE = pd.Timestamp("2025-12-31")

# Input datasets
CANONICAL_INPUT_PATH = Path(
    "data/canonical/canonical_customers.csv"
)

EVENTS_INPUT_PATH = Path(
    "data/events/events.csv"
)

PORTAL_MAPPING_INPUT_PATH = Path(
    "data/reference/portal_account_mapping.csv"
)

CRM_INPUT_PATH = Path(
    "data/raw/crm_customer_records.csv"
)

CRM_GROUND_TRUTH_INPUT_PATH = Path(
    "data/reference/crm_ground_truth_mapping.csv"
)

# Output paths
ECOMMERCE_OUTPUT_PATH = Path(
    "data/raw/ecommerce_transactions.csv"
)

GROUND_TRUTH_OUTPUT_PATH = Path(
    "data/reference/ecommerce_ground_truth_mapping.csv"
)

VALIDATION_OUTPUT_PATH = Path(
    "data/reference/ecommerce_validation_summary.csv"
)


###############################################################################
# 2. Weighting assumptions
###############################################################################

# Customer selection is weighted using canonical customer_segment.
# Higher values represent greater relative likelihood of purchasing tickets.
ECOMMERCE_SEGMENT_SELECTION_WEIGHTS = {
    "New": 0.80,
    "Occasional": 1.00,
    "Regular": 1.20,
    "VIP": 1.40,
}

# Existing portal customers are slightly more likely to appear
# within E-commerce records.
PORTAL_CUSTOMER_SELECTION_BOOST = 1.25

# Customers in higher engagement segments are more likely
# to make purchases.
REPEAT_PURCHASE_WEIGHTS = {
    "New": 0.60,
    "Occasional": 0.80,
    "Regular": 1.00,
    "VIP": 1.20,
}

# New and Occasional portal customers are more likely to
# use guest checkout than Regular and VIP customers.
GUEST_CHECKOUT_PROBABILITIES = {
    "New": 0.50,
    "Occasional": 0.30,
    "Regular": 0.18,
    "VIP": 0.10,
}

# Ticket quantities per transaction.
# Most transactions contain one or two tickets.
TICKET_QUANTITIES = [
    1,
    2,
    3,
    4,
]

TICKET_QUANTITY_PROBABILITIES = [
    0.64,
    0.24,
    0.08,
    0.04,
]

# Age weighting used during transaction assignment.
# Customers aged 60+ are more likely to purchase seated tickets.
AGE_THRESHOLD_SEATED = 60

UNDER_60_STANDING_WEIGHT = 1.30
UNDER_60_SEATED_WEIGHT = 0.70

OVER_60_STANDING_WEIGHT = 0.65
OVER_60_SEATED_WEIGHT = 1.60

# Card brands used for synthetic transactions
CARD_BRANDS = [
    "Visa",
    "Mastercard",
    "American Express",
]

CARD_BRAND_PROBABILITIES = [
    0.55,
    0.40,
    0.05,
]


###############################################################################
# 3. Data quality assumptions
###############################################################################

# Guest transactions contain greater levels of manually entered
# and incomplete information than registered transactions.

GUEST_MISSING_EMAIL_RATE = 0.04
REGISTERED_MISSING_EMAIL_RATE = 0.01

GUEST_MISSING_PHONE_RATE = 0.12
REGISTERED_MISSING_PHONE_RATE = 0.04

GUEST_ALTERNATIVE_EMAIL_RATE = 0.12
REGISTERED_ALTERNATIVE_EMAIL_RATE = 0.04

GUEST_EMAIL_TYPO_RATE = 0.03
REGISTERED_EMAIL_TYPO_RATE = 0.01

GUEST_NAME_TYPO_RATE = 0.04
REGISTERED_NAME_TYPO_RATE = 0.01

GUEST_INCORRECT_PHONE_RATE = 0.04
REGISTERED_INCORRECT_PHONE_RATE = 0.01

# Billing details can be historical or formatted differently
HISTORICAL_BILLING_ADDRESS_RATE = 0.06
POSTCODE_FORMAT_VARIATION_RATE = 0.10

# Cardholder name may differ in formatting from purchaser name
CARDHOLDER_NAME_VARIATION_RATE = 0.08


###############################################################################
# 4. Initialise generators
###############################################################################

# UK-localised Faker generator
fake = Faker("en_GB")

# Assign consistent seed to Faker and NumPy
Faker.seed(ECOMMERCE_SEED)
rng = np.random.default_rng(
    ECOMMERCE_SEED
)


###############################################################################
# 5. Helper functions
###############################################################################

# Select a random datetime between two datetime boundaries
def generate_random_datetime(
    start_datetime: pd.Timestamp,
    end_datetime: pd.Timestamp,
) -> pd.Timestamp:

    total_seconds = int(
        (
            end_datetime
            - start_datetime
        ).total_seconds()
    )

    if total_seconds <= 0:
        return start_datetime

    random_seconds = int(
        rng.integers(
            0,
            total_seconds + 1,
        )
    )

    return (
        start_datetime
        + pd.Timedelta(
            seconds=random_seconds
        )
    )


# Calculate customer age at snapshot date
def calculate_age(
    date_of_birth: pd.Timestamp,
    reference_date: pd.Timestamp,
) -> int:

    age = (
        reference_date.year
        - date_of_birth.year
    )

    if (
        reference_date.month,
        reference_date.day,
    ) < (
        date_of_birth.month,
        date_of_birth.day,
    ):
        age -= 1

    return age


# Normalise name for use within alternative synthetic email
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


# Generate plausible alternative email belonging to the same
# hidden canonical customer
def generate_alternative_email(
    first_name: str,
    surname: str,
    date_of_birth: pd.Timestamp,
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
        date_of_birth.year
    )[-2:]

    current_domain = (
        str(current_email)
        .split("@")[-1]
        .lower()
    )

    alternative_domains = [
        "gmail.com",
        "outlook.com",
        "hotmail.com",
        "yahoo.com",
        "icloud.com",
    ]

    available_domains = [
        domain
        for domain
        in alternative_domains
        if domain != current_domain
    ]

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

    while email in used_emails:

        email = (
            f"{local_part}{suffix}"
            f"@{domain}"
        )

        suffix += 1

    used_emails.add(
        email
    )

    return email


# Introduce small spelling error into name
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


# Introduce typo into local section of an email address
def introduce_email_typo(
    email: str,
) -> str:

    if (
        pd.isna(email)
        or "@" not in str(email)
    ):
        return email

    local_part, domain = (
        str(email)
        .split(
            "@",
            1,
        )
    )

    if len(local_part) < 3:
        return email

    altered_local = introduce_name_typo(
        local_part
    )

    return (
        f"{altered_local}"
        f"@{domain}"
    )


# Introduce an incorrect digit into phone number
def introduce_phone_error(
    telephone_number: str,
) -> str:

    if pd.isna(
        telephone_number
    ):
        return telephone_number

    digits = re.sub(
        r"\D",
        "",
        str(telephone_number),
    )

    # Convert into UK national mobile representation
    if digits.startswith("44"):
        national_number = digits[2:]
    else:
        national_number = (
            digits.lstrip("0")
        )

    if len(
        national_number
    ) < 5:
        return telephone_number

    characters = list(
        national_number
    )

    # Avoid changing the initial mobile prefix digit
    position = int(
        rng.integers(
            2,
            len(characters),
        )
    )

    original_digit = (
        characters[position]
    )

    replacement_digits = [
        str(number)
        for number
        in range(10)
        if str(number)
        != original_digit
    ]

    characters[
        position
    ] = str(
        rng.choice(
            replacement_digits
        )
    )

    altered_number = "".join(
        characters
    )

    return (
        f"+44 {altered_number}"
    )


# Remove standard postcode spacing
def alter_postcode_format(
    postcode: str,
) -> str:

    if pd.isna(
        postcode
    ):
        return postcode

    return (
        str(postcode)
        .replace(
            " ",
            "",
        )
    )


# Generate alternative historical billing address while
# retaining the same city and outward postcode district
def generate_historical_billing_address(
    current_address: str,
    current_postcode: str,
) -> tuple[str, str]:

    if "," in str(
        current_address
    ):

        city = (
            str(current_address)
            .split(",")[-1]
            .strip()
        )

    else:
        city = fake.city()

    street_address = (
        fake.street_address()
        .replace(
            "\n",
            ", ",
        )
    )

    historical_address = (
        f"{street_address}, {city}"
    )

    postcode_parts = (
        str(current_postcode)
        .split()
    )

    if len(
        postcode_parts
    ) > 0:
        outward_code = (
            postcode_parts[0]
        )
    else:
        outward_code = "ZZ1"

    postcode_digit = int(
        rng.integers(
            0,
            10,
        )
    )

    first_letter = (
        fake
        .random_letter()
        .upper()
    )

    second_letter = (
        fake
        .random_letter()
        .upper()
    )

    historical_postcode = (
        f"{outward_code} "
        f"{postcode_digit}"
        f"{first_letter}"
        f"{second_letter}"
    )

    return (
        historical_address,
        historical_postcode,
    )


# Generate variation between purchaser and cardholder name
def alter_cardholder_name(
    first_name: str,
    surname: str,
) -> str:

    options = [
        f"{first_name[0]}. {surname}",
        f"{first_name} {surname[0]}.",
        f"{first_name.upper()} {surname.upper()}",
    ]

    return str(
        rng.choice(
            options
        )
    )


# Split exact attendance quantity into realistic
# transaction-level ticket quantities.
def split_ticket_quantity(
    total_tickets: int,
) -> list[int]:

    transaction_quantities = []

    remaining_tickets = int(
        total_tickets
    )

    while remaining_tickets > 0:

        allowed_quantities = [
            quantity
            for quantity
            in TICKET_QUANTITIES
            if quantity
            <= remaining_tickets
        ]

        allowed_probabilities = np.array(
            [
                TICKET_QUANTITY_PROBABILITIES[
                    TICKET_QUANTITIES.index(
                        quantity
                    )
                ]
                for quantity
                in allowed_quantities
            ],
            dtype=float,
        )

        allowed_probabilities = (
            allowed_probabilities
            / allowed_probabilities.sum()
        )

        quantity = int(
            rng.choice(
                allowed_quantities,
                p=allowed_probabilities,
            )
        )

        transaction_quantities.append(
            quantity
        )

        remaining_tickets -= quantity

    return transaction_quantities


# Return preference weight based on customer age and
# requested ticket type.
def get_ticket_type_weight(
    age: int,
    ticket_type: str,
) -> float:

    if age >= AGE_THRESHOLD_SEATED:

        if ticket_type == "Seated":
            return (
                OVER_60_SEATED_WEIGHT
            )

        return (
            OVER_60_STANDING_WEIGHT
        )

    if ticket_type == "Standing":
        return (
            UNDER_60_STANDING_WEIGHT
        )

    return (
        UNDER_60_SEATED_WEIGHT
    )


###############################################################################
# 6. Load and validate source datasets
###############################################################################

def load_source_datasets() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
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
        "date_of_birth"
    ] = pd.to_datetime(
        canonical[
            "date_of_birth"
        ]
    )

    canonical[
        "registration_date"
    ] = pd.to_datetime(
        canonical[
            "registration_date"
        ]
    )

    ###########################################################################
    # Events reference dataset
    ###########################################################################

    events = pd.read_csv(
        EVENTS_INPUT_PATH
    )

    expected_event_columns = [
        "event_id",
        "event_name",
        "event_category",
        "event_date",
        "event_standing_attendance",
        "event_seated_attendance",
        "event_total_attendance",
        "ticket_price_standing",
        "ticket_price_seated",
    ]

    assert list(
        events.columns
    ) == expected_event_columns

    assert events[
        "event_id"
    ].is_unique

    events[
        "event_date"
    ] = pd.to_datetime(
        events[
            "event_date"
        ]
    )

    # Validate attendance totals
    assert (
        events[
            "event_standing_attendance"
        ]
        +
        events[
            "event_seated_attendance"
        ]
        ==
        events[
            "event_total_attendance"
        ]
    ).all()

    assert (
        events[
            "event_total_attendance"
        ].sum()
        ==
        EXPECTED_TOTAL_ATTENDANCE
    )

    ###########################################################################
    # Portal account mapping
    ###########################################################################

    portal_mapping = pd.read_csv(
        PORTAL_MAPPING_INPUT_PATH
    )

    expected_portal_columns = [
        "ground_truth_id",
        "portal_user_id",
    ]

    assert list(
        portal_mapping.columns
    ) == expected_portal_columns

    assert portal_mapping[
        "ground_truth_id"
    ].is_unique

    assert portal_mapping[
        "portal_user_id"
    ].is_unique

    ###########################################################################
    # CRM records
    ###########################################################################

    crm = pd.read_csv(
        CRM_INPUT_PATH
    )

    crm_ground_truth = pd.read_csv(
        CRM_GROUND_TRUTH_INPUT_PATH
    )

    # Attach hidden ground truth to CRM only for
    # synthetic generation purposes
    crm = crm.merge(
        crm_ground_truth[
            [
                "crm_customer_id",
                "ground_truth_id",
                "is_duplicate_record",
            ]
        ],
        on="crm_customer_id",
        how="left",
        validate="one_to_one",
    )

    assert crm[
        "ground_truth_id"
    ].notna().all()

    # Ensure Boolean representation is consistent
    crm[
        "is_duplicate_record"
    ] = (
        crm[
            "is_duplicate_record"
        ]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    crm[
        "record_created_date"
    ] = pd.to_datetime(
        crm[
            "record_created_date"
        ]
    )

    # Prefer original CRM record over intentionally
    # introduced duplicate when copying account details
    crm = (
        crm
        .sort_values(
            [
                "ground_truth_id",
                "is_duplicate_record",
                "record_created_date",
            ]
        )
        .drop_duplicates(
            subset=[
                "ground_truth_id",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    # Rename CRM contact fields before later merge
    crm = crm.rename(
        columns={
            "first_name": "crm_first_name",
            "surname": "crm_surname",
            "email": "crm_email",
            "telephone_number": "crm_telephone_number",
            "address": "crm_address",
            "postcode": "crm_postcode",
        }
    )

    crm_profile_columns = [
        "ground_truth_id",
        "crm_first_name",
        "crm_surname",
        "crm_email",
        "crm_telephone_number",
        "crm_address",
        "crm_postcode",
    ]

    crm_profiles = crm[
        crm_profile_columns
    ].copy()

    return (
        canonical,
        events,
        portal_mapping,
        crm_profiles,
    )


###############################################################################
# 7. Select E-commerce purchaser population
###############################################################################

def select_ecommerce_customers(
    canonical: pd.DataFrame,
    events: pd.DataFrame,
    portal_mapping: pd.DataFrame,
) -> pd.DataFrame:

    customer_population = (
        canonical
        .merge(
            portal_mapping,
            on="ground_truth_id",
            how="left",
        )
        .copy()
    )

    # Customers must have registered before at least one
    # event takes place in the operational snapshot.
    latest_event_date = (
        events[
            "event_date"
        ].max()
    )

    eligible_customers = (
        customer_population[
            customer_population[
                "registration_date"
            ]
            < latest_event_date
        ]
        .copy()
    )

    assert (
        len(
            eligible_customers
        )
        >=
        UNIQUE_ECOMMERCE_CUSTOMERS
    )

    # Apply segment-based purchase weighting
    selection_weights = (
        eligible_customers[
            "customer_segment"
        ]
        .map(
            ECOMMERCE_SEGMENT_SELECTION_WEIGHTS
        )
        .astype(float)
    )

    # Existing portal users are slightly more likely
    # to make an E-commerce purchase
    portal_boost = np.where(
        eligible_customers[
            "portal_user_id"
        ].notna(),
        PORTAL_CUSTOMER_SELECTION_BOOST,
        1.0,
    )

    selection_weights = (
        selection_weights
        * portal_boost
    )

    selection_weights = (
        selection_weights
        / selection_weights.sum()
    )

    selected_indices = rng.choice(
        eligible_customers.index.to_numpy(),
        size=UNIQUE_ECOMMERCE_CUSTOMERS,
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

    # Calculate age at snapshot for later
    # ticket-type assignment weighting
    selected_customers[
        "customer_age"
    ] = selected_customers.apply(
        lambda row:
        calculate_age(
            date_of_birth=row[
                "date_of_birth"
            ],
            reference_date=SNAPSHOT_DATE,
        ),
        axis=1,
    )

    return selected_customers


###############################################################################
# 8. Generate transaction plan from event attendance
###############################################################################

def generate_transaction_plan(
    events: pd.DataFrame,
) -> pd.DataFrame:

    transaction_plan = []

    for event in events.itertuples(
        index=False
    ):

        #######################################################################
        # Standing ticket transactions
        #######################################################################

        standing_quantities = (
            split_ticket_quantity(
                event.event_standing_attendance
            )
        )

        for quantity in standing_quantities:

            transaction_plan.append(
                {
                    "event_id": event.event_id,
                    "event_name": event.event_name,
                    "event_category": event.event_category,
                    "event_date": event.event_date,
                    "ticket_type": "Standing",
                    "ticket_quantity": quantity,
                    "unit_price": float(
                        event.ticket_price_standing
                    ),
                }
            )

        #######################################################################
        # Seated ticket transactions
        #######################################################################

        seated_quantities = (
            split_ticket_quantity(
                event.event_seated_attendance
            )
        )

        for quantity in seated_quantities:

            transaction_plan.append(
                {
                    "event_id": event.event_id,
                    "event_name": event.event_name,
                    "event_category": event.event_category,
                    "event_date": event.event_date,
                    "ticket_type": "Seated",
                    "ticket_quantity": quantity,
                    "unit_price": float(
                        event.ticket_price_seated
                    ),
                }
            )

    transaction_plan = pd.DataFrame(
        transaction_plan
    )

    # There must be enough transaction records to ensure
    # every selected purchaser can make at least one purchase.
    assert (
        len(
            transaction_plan
        )
        >=
        UNIQUE_ECOMMERCE_CUSTOMERS
    )

    return transaction_plan


###############################################################################
# 9. Assign purchasers to transaction records
###############################################################################

def assign_purchasers_to_transactions(
    transaction_plan: pd.DataFrame,
    selected_customers: pd.DataFrame,
) -> pd.DataFrame:

    transaction_plan = (
        transaction_plan
        .copy()
        .reset_index(
            drop=True
        )
    )

    assignments = np.full(
        len(
            transaction_plan
        ),
        None,
        dtype=object,
    )

    unassigned_transactions = np.ones(
        len(
            transaction_plan
        ),
        dtype=bool,
    )

    event_dates = pd.to_datetime(
        transaction_plan[
            "event_date"
        ]
    ).to_numpy()

    ticket_types = (
        transaction_plan[
            "ticket_type"
        ]
        .to_numpy()
    )

    ###########################################################################
    # Ensure every selected customer receives at least one transaction
    ###########################################################################

    # Late-registering customers are allocated first because
    # they have fewer eligible future events.
    customers_by_registration = (
        selected_customers
        .sort_values(
            "registration_date",
            ascending=False,
        )
    )

    for customer in (
        customers_by_registration
        .itertuples(
            index=False
        )
    ):

        registration_date = np.datetime64(
            customer.registration_date
        )

        candidate_indices = np.flatnonzero(
            unassigned_transactions
            &
            (
                event_dates
                >
                registration_date
            )
        )

        if len(
            candidate_indices
        ) == 0:

            raise ValueError(
                "Unable to allocate a future transaction "
                f"to {customer.ground_truth_id}."
            )

        candidate_weights = np.array(
            [
                get_ticket_type_weight(
                    age=int(
                        customer.customer_age
                    ),
                    ticket_type=str(
                        ticket_types[
                            candidate_index
                        ]
                    ),
                )
                for candidate_index
                in candidate_indices
            ],
            dtype=float,
        )

        candidate_weights = (
            candidate_weights
            / candidate_weights.sum()
        )

        selected_transaction_index = int(
            rng.choice(
                candidate_indices,
                p=candidate_weights,
            )
        )

        assignments[
            selected_transaction_index
        ] = customer.ground_truth_id

        unassigned_transactions[
            selected_transaction_index
        ] = False

    ###########################################################################
    # Assign remaining transactions
    ###########################################################################

    customer_ground_truth_ids = (
        selected_customers[
            "ground_truth_id"
        ]
        .to_numpy()
    )

    customer_registration_dates = (
        selected_customers[
            "registration_date"
        ]
        .to_numpy(
            dtype="datetime64[ns]"
        )
    )

    customer_ages = (
        selected_customers[
            "customer_age"
        ]
        .to_numpy()
    )

    customer_segments = (
        selected_customers[
            "customer_segment"
        ]
        .to_numpy()
    )

    customer_portal_status = (
        selected_customers[
            "portal_user_id"
        ]
        .notna()
        .to_numpy()
    )

    remaining_indices = np.flatnonzero(
        unassigned_transactions
    )

    for transaction_index in (
        remaining_indices
    ):

        event_date = event_dates[
            transaction_index
        ]

        ticket_type = str(
            ticket_types[
                transaction_index
            ]
        )

        eligible_mask = (
            customer_registration_dates
            <
            event_date
        )

        eligible_customer_indices = (
            np.flatnonzero(
                eligible_mask
            )
        )

        repeat_weights = np.array(
            [
                REPEAT_PURCHASE_WEIGHTS[
                    customer_segments[
                        customer_index
                    ]
                ]
                for customer_index
                in eligible_customer_indices
            ],
            dtype=float,
        )

        ticket_weights = np.array(
            [
                get_ticket_type_weight(
                    age=int(
                        customer_ages[
                            customer_index
                        ]
                    ),
                    ticket_type=ticket_type,
                )
                for customer_index
                in eligible_customer_indices
            ],
            dtype=float,
        )

        # Portal customers receive a small additional
        # weighting for repeat E-commerce activity.
        portal_weights = np.where(
            customer_portal_status[
                eligible_customer_indices
            ],
            1.20,
            1.0,
        )

        combined_weights = (
            repeat_weights
            * ticket_weights
            * portal_weights
        )

        combined_weights = (
            combined_weights
            / combined_weights.sum()
        )

        selected_customer_index = int(
            rng.choice(
                eligible_customer_indices,
                p=combined_weights,
            )
        )

        assignments[
            transaction_index
        ] = customer_ground_truth_ids[
            selected_customer_index
        ]

    ###########################################################################
    # Attach selected purchaser identities
    ###########################################################################

    transaction_plan[
        "ground_truth_id"
    ] = assignments

    assert transaction_plan[
        "ground_truth_id"
    ].notna().all()

    transaction_plan = (
        transaction_plan
        .merge(
            selected_customers,
            on="ground_truth_id",
            how="left",
            validate="many_to_one",
        )
    )

    return transaction_plan


###############################################################################
# 10. Assign guest and registered checkout status
###############################################################################

def assign_guest_transactions(
    transaction_plan: pd.DataFrame,
) -> pd.DataFrame:

    transaction_plan = (
        transaction_plan
        .copy()
    )

    # Retain customer's underlying portal account for
    # generation and validation purposes
    transaction_plan[
        "available_portal_user_id"
    ] = transaction_plan[
        "portal_user_id"
    ]

    transaction_plan[
        "guest_transaction"
    ] = False

    for index in (
        transaction_plan.index
    ):

        portal_user_id = (
            transaction_plan.at[
                index,
                "available_portal_user_id",
            ]
        )

        customer_segment = (
            transaction_plan.at[
                index,
                "customer_segment",
            ]
        )

        #######################################################################
        # Customers without a portal account must use guest checkout
        #######################################################################

        if pd.isna(
            portal_user_id
        ):

            guest_transaction = True

        #######################################################################
        # Customers with portal accounts may still choose guest checkout
        #######################################################################

        else:

            guest_probability = (
                GUEST_CHECKOUT_PROBABILITIES[
                    customer_segment
                ]
            )

            guest_transaction = (
                rng.random()
                <
                guest_probability
            )

        transaction_plan.at[
            index,
            "guest_transaction",
        ] = guest_transaction

        #######################################################################
        # Guest checkout does not retain portal_user_id
        #######################################################################

        if guest_transaction:

            transaction_plan.at[
                index,
                "portal_user_id",
            ] = pd.NA

    return transaction_plan


###############################################################################
# 11. Generate clean E-commerce records
###############################################################################

def generate_clean_ecommerce_records(
    transaction_plan: pd.DataFrame,
    crm_profiles: pd.DataFrame,
) -> pd.DataFrame:

    transaction_plan = (
        transaction_plan
        .merge(
            crm_profiles,
            on="ground_truth_id",
            how="left",
            validate="many_to_one",
        )
    )

    transaction_records = []

    for transaction in (
        transaction_plan
        .itertuples(
            index=False
        )
    ):

        #######################################################################
        # Generate transaction date
        #######################################################################

        earliest_transaction_date = max(
            OPERATION_START_DATE,
            pd.Timestamp(
                transaction.registration_date
            ),
        )

        latest_transaction_date = (
            pd.Timestamp(
                transaction.event_date
            )
            -
            pd.Timedelta(
                seconds=1
            )
        )

        transaction_date_time = (
            generate_random_datetime(
                start_datetime=earliest_transaction_date,
                end_datetime=latest_transaction_date,
            )
        )

        #######################################################################
        # Select source contact details
        #######################################################################

        # Registered transactions use CRM details where available.
        # Guest transactions begin from canonical customer details.
        if (
            not transaction.guest_transaction
        ):

            purchaser_first_name = (
                transaction.crm_first_name
                if pd.notna(
                    transaction.crm_first_name
                )
                else transaction.first_name
            )

            purchaser_surname = (
                transaction.crm_surname
                if pd.notna(
                    transaction.crm_surname
                )
                else transaction.surname
            )

            email = (
                transaction.crm_email
                if pd.notna(
                    transaction.crm_email
                )
                else transaction.email
            )

            telephone_number = (
                transaction.crm_telephone_number
                if pd.notna(
                    transaction.crm_telephone_number
                )
                else transaction.telephone_number
            )

        else:

            purchaser_first_name = (
                transaction.first_name
            )

            purchaser_surname = (
                transaction.surname
            )

            email = (
                transaction.email
            )

            telephone_number = (
                transaction.telephone_number
            )

        #######################################################################
        # Billing details
        #######################################################################

        billing_address = (
            transaction.address
        )

        billing_postcode = (
            transaction.postcode
        )

        #######################################################################
        # Payment details
        #######################################################################

        card_brand = str(
            rng.choice(
                CARD_BRANDS,
                p=CARD_BRAND_PROBABILITIES,
            )
        )

        cardholder_name = (
            f"{transaction.first_name} "
            f"{transaction.surname}"
        )

        card_last_4 = (
            f"{int(rng.integers(0, 10_000)):04d}"
        )

        #######################################################################
        # Calculate transaction total
        #######################################################################

        transaction_total = round(
            int(
                transaction.ticket_quantity
            )
            *
            float(
                transaction.unit_price
            ),
            2,
        )

        #######################################################################
        # Append clean transaction
        #######################################################################

        transaction_records.append(
            {
                # Internal generation fields
                "ground_truth_id": transaction.ground_truth_id,
                "customer_segment": transaction.customer_segment,
                "customer_age": transaction.customer_age,
                "registration_date": transaction.registration_date,
                "available_portal_user_id":
                    transaction.available_portal_user_id,

                # Operational transaction fields
                "transaction_id": pd.NA,
                "transaction_date_time": transaction_date_time,
                "guest_transaction": bool(
                    transaction.guest_transaction
                ),
                "portal_user_id": transaction.portal_user_id,
                "billing_address": billing_address,
                "billing_postcode": billing_postcode,
                "event_id": transaction.event_id,
                "event_name": transaction.event_name,
                "event_category": transaction.event_category,
                "event_date": transaction.event_date,
                "ticket_type": transaction.ticket_type,
                "ticket_quantity": int(
                    transaction.ticket_quantity
                ),
                "unit_price": float(
                    transaction.unit_price
                ),
                "transaction_total": transaction_total,
                "card_brand": card_brand,
                "cardholder_name": cardholder_name,
                "card_last_4": card_last_4,
                "email": email,
                "purchaser_first_name": purchaser_first_name,
                "purchaser_surname": purchaser_surname,
                "telephone_number": telephone_number,
            }
        )

    return pd.DataFrame(
        transaction_records
    )


###############################################################################
# 12. Generate alternative email mapping
###############################################################################

def generate_alternative_email_mapping(
    selected_customers: pd.DataFrame,
    canonical: pd.DataFrame,
) -> dict[str, str]:

    alternative_email_mapping = {}

    # Prevent generated alternative emails from conflicting
    # with existing canonical email addresses.
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

        alternative_email = (
            generate_alternative_email(
                first_name=customer.first_name,
                surname=customer.surname,
                date_of_birth=pd.Timestamp(
                    customer.date_of_birth
                ),
                current_email=customer.email,
                used_emails=used_emails,
            )
        )

        alternative_email_mapping[
            customer.ground_truth_id
        ] = alternative_email

    return (
        alternative_email_mapping
    )


###############################################################################
# 13. Introduce E-commerce data quality issues
###############################################################################

def introduce_ecommerce_data_quality_issues(
    transaction_records: pd.DataFrame,
    alternative_email_mapping: dict[str, str],
) -> pd.DataFrame:

    transaction_records = (
        transaction_records
        .copy()
    )

    data_quality_flags = [
        "dq_missing_email",
        "dq_missing_phone",
        "dq_alternative_email",
        "dq_email_typo",
        "dq_name_typo",
        "dq_historical_billing_address",
        "dq_postcode_format_variation",
        "dq_incorrect_phone",
        "dq_cardholder_name_variation",
    ]

    for flag in (
        data_quality_flags
    ):

        transaction_records[
            flag
        ] = False

    for index in (
        transaction_records.index
    ):

        guest_transaction = bool(
            transaction_records.at[
                index,
                "guest_transaction",
            ]
        )

        #######################################################################
        # Select different rates for guest and registered checkouts
        #######################################################################

        if guest_transaction:

            missing_email_rate = (
                GUEST_MISSING_EMAIL_RATE
            )

            missing_phone_rate = (
                GUEST_MISSING_PHONE_RATE
            )

            alternative_email_rate = (
                GUEST_ALTERNATIVE_EMAIL_RATE
            )

            email_typo_rate = (
                GUEST_EMAIL_TYPO_RATE
            )

            name_typo_rate = (
                GUEST_NAME_TYPO_RATE
            )

            incorrect_phone_rate = (
                GUEST_INCORRECT_PHONE_RATE
            )

        else:

            missing_email_rate = (
                REGISTERED_MISSING_EMAIL_RATE
            )

            missing_phone_rate = (
                REGISTERED_MISSING_PHONE_RATE
            )

            alternative_email_rate = (
                REGISTERED_ALTERNATIVE_EMAIL_RATE
            )

            email_typo_rate = (
                REGISTERED_EMAIL_TYPO_RATE
            )

            name_typo_rate = (
                REGISTERED_NAME_TYPO_RATE
            )

            incorrect_phone_rate = (
                REGISTERED_INCORRECT_PHONE_RATE
            )

        #######################################################################
        # Email
        #######################################################################

        if (
            rng.random()
            <
            missing_email_rate
        ):

            transaction_records.at[
                index,
                "email",
            ] = pd.NA

            transaction_records.at[
                index,
                "dq_missing_email",
            ] = True

        elif (
            rng.random()
            <
            alternative_email_rate
        ):

            ground_truth_id = (
                transaction_records.at[
                    index,
                    "ground_truth_id",
                ]
            )

            transaction_records.at[
                index,
                "email",
            ] = (
                alternative_email_mapping[
                    ground_truth_id
                ]
            )

            transaction_records.at[
                index,
                "dq_alternative_email",
            ] = True

        elif (
            rng.random()
            <
            email_typo_rate
        ):

            transaction_records.at[
                index,
                "email",
            ] = introduce_email_typo(
                transaction_records.at[
                    index,
                    "email",
                ]
            )

            transaction_records.at[
                index,
                "dq_email_typo",
            ] = True

        #######################################################################
        # Telephone number
        #######################################################################

        if (
            rng.random()
            <
            missing_phone_rate
        ):

            transaction_records.at[
                index,
                "telephone_number",
            ] = pd.NA

            transaction_records.at[
                index,
                "dq_missing_phone",
            ] = True

        elif (
            rng.random()
            <
            incorrect_phone_rate
        ):

            transaction_records.at[
                index,
                "telephone_number",
            ] = introduce_phone_error(
                transaction_records.at[
                    index,
                    "telephone_number",
                ]
            )

            transaction_records.at[
                index,
                "dq_incorrect_phone",
            ] = True

        #######################################################################
        # Purchaser name
        #######################################################################

        if (
            rng.random()
            <
            name_typo_rate
        ):

            selected_name_field = str(
                rng.choice(
                    [
                        "purchaser_first_name",
                        "purchaser_surname",
                    ]
                )
            )

            transaction_records.at[
                index,
                selected_name_field,
            ] = introduce_name_typo(
                transaction_records.at[
                    index,
                    selected_name_field,
                ]
            )

            transaction_records.at[
                index,
                "dq_name_typo",
            ] = True

        #######################################################################
        # Billing address and postcode
        #######################################################################

        if (
            rng.random()
            <
            HISTORICAL_BILLING_ADDRESS_RATE
        ):

            (
                historical_address,
                historical_postcode,
            ) = generate_historical_billing_address(
                current_address=(
                    transaction_records.at[
                        index,
                        "billing_address",
                    ]
                ),
                current_postcode=(
                    transaction_records.at[
                        index,
                        "billing_postcode",
                    ]
                ),
            )

            transaction_records.at[
                index,
                "billing_address",
            ] = historical_address

            transaction_records.at[
                index,
                "billing_postcode",
            ] = historical_postcode

            transaction_records.at[
                index,
                "dq_historical_billing_address",
            ] = True

        if (
            rng.random()
            <
            POSTCODE_FORMAT_VARIATION_RATE
        ):

            transaction_records.at[
                index,
                "billing_postcode",
            ] = alter_postcode_format(
                transaction_records.at[
                    index,
                    "billing_postcode",
                ]
            )

            transaction_records.at[
                index,
                "dq_postcode_format_variation",
            ] = True

        #######################################################################
        # Cardholder name
        #######################################################################

        if (
            rng.random()
            <
            CARDHOLDER_NAME_VARIATION_RATE
        ):

            transaction_records.at[
                index,
                "cardholder_name",
            ] = alter_cardholder_name(
                first_name=str(
                    transaction_records.at[
                        index,
                        "purchaser_first_name",
                    ]
                ),
                surname=str(
                    transaction_records.at[
                        index,
                        "purchaser_surname",
                    ]
                ),
            )

            transaction_records.at[
                index,
                "dq_cardholder_name_variation",
            ] = True

    return transaction_records


###############################################################################
# 14. Finalise transaction IDs
###############################################################################

def finalise_transaction_ids(
    transaction_records: pd.DataFrame,
) -> pd.DataFrame:

    # Sort transactions chronologically before assigning
    # sequential transaction identifiers.
    transaction_records = (
        transaction_records
        .sort_values(
            "transaction_date_time"
        )
        .reset_index(
            drop=True
        )
    )

    transaction_records[
        "transaction_id"
    ] = [
        f"TXN{transaction_number:06d}"
        for transaction_number
        in range(
            1,
            len(
                transaction_records
            ) + 1,
        )
    ]

    return transaction_records


###############################################################################
# 15. Validate E-commerce dataset
###############################################################################

def validate_ecommerce_records(
    transaction_records: pd.DataFrame,
    canonical: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:

    ###########################################################################
    # Basic validation
    ###########################################################################

    assert transaction_records[
        "transaction_id"
    ].is_unique

    assert (
        transaction_records[
            "ground_truth_id"
        ].nunique()
        ==
        UNIQUE_ECOMMERCE_CUSTOMERS
    )

    assert set(
        transaction_records[
            "ground_truth_id"
        ]
    ).issubset(
        set(
            canonical[
                "ground_truth_id"
            ]
        )
    )

    assert set(
        transaction_records[
            "event_id"
        ]
    ).issubset(
        set(
            events[
                "event_id"
            ]
        )
    )

    assert set(
        transaction_records[
            "ticket_type"
        ]
    ).issubset(
        {
            "Standing",
            "Seated",
        }
    )

    assert (
        transaction_records[
            "ticket_quantity"
        ] > 0
    ).all()

    ###########################################################################
    # Guest / registered account validation
    ###########################################################################

    guest_records = (
        transaction_records[
            transaction_records[
                "guest_transaction"
            ]
        ]
    )

    registered_records = (
        transaction_records[
            ~transaction_records[
                "guest_transaction"
            ]
        ]
    )

    # Guest transaction must not expose portal ID
    assert guest_records[
        "portal_user_id"
    ].isna().all()

    # Registered transaction must contain portal ID
    assert registered_records[
        "portal_user_id"
    ].notna().all()

    # Registered portal ID must equal customer's
    # available portal account ID
    assert (
        registered_records[
            "portal_user_id"
        ]
        ==
        registered_records[
            "available_portal_user_id"
        ]
    ).all()

    # Customers without an available portal account
    # must always complete guest transactions
    no_portal_records = transaction_records[
        transaction_records[
            "available_portal_user_id"
        ].isna()
    ]

    assert no_portal_records[
        "guest_transaction"
    ].all()

    ###########################################################################
    # Transaction timing validation
    ###########################################################################

    transaction_dates = pd.to_datetime(
        transaction_records[
            "transaction_date_time"
        ]
    )

    event_dates = pd.to_datetime(
        transaction_records[
            "event_date"
        ]
    )

    registration_dates = pd.to_datetime(
        transaction_records[
            "registration_date"
        ]
    )

    # Count tickets purchased on or after associated event date
    late_purchase_mask = (
        transaction_dates
        >=
        event_dates
    )

    tickets_purchased_after_event_date = int(
        transaction_records.loc[
            late_purchase_mask,
            "ticket_quantity",
        ].sum()
    )

    # Count tickets purchased outside 12-month snapshot
    outside_snapshot_mask = (
        (
            transaction_dates
            <
            OPERATION_START_DATE
        )
        |
        (
            transaction_dates.dt.normalize()
            >
            SNAPSHOT_DATE
        )
    )

    tickets_purchased_outside_snapshot = int(
        transaction_records.loc[
            outside_snapshot_mask,
            "ticket_quantity",
        ].sum()
    )

    assert (
        transaction_dates
        >=
        OPERATION_START_DATE
    ).all()

    assert (
        transaction_dates
        <=
        SNAPSHOT_DATE
    ).all()

    assert (
        transaction_dates
        <
        event_dates
    ).all()

    assert (
        transaction_dates
        >=
        registration_dates
    ).all()

    ###########################################################################
    # Event field validation
    ###########################################################################

    event_lookup = (
        events
        .set_index(
            "event_id"
        )
    )

    expected_event_names = (
        transaction_records[
            "event_id"
        ]
        .map(
            event_lookup[
                "event_name"
            ]
        )
    )

    expected_categories = (
        transaction_records[
            "event_id"
        ]
        .map(
            event_lookup[
                "event_category"
            ]
        )
    )

    expected_event_dates = (
        transaction_records[
            "event_id"
        ]
        .map(
            event_lookup[
                "event_date"
            ]
        )
    )

    assert (
        transaction_records[
            "event_name"
        ]
        ==
        expected_event_names
    ).all()

    assert (
        transaction_records[
            "event_category"
        ]
        ==
        expected_categories
    ).all()

    assert (
        pd.to_datetime(
            transaction_records[
                "event_date"
            ]
        )
        ==
        pd.to_datetime(
            expected_event_dates
        )
    ).all()

    ###########################################################################
    # Unit price validation
    ###########################################################################

    expected_unit_prices = np.where(
        transaction_records[
            "ticket_type"
        ]
        ==
        "Standing",

        transaction_records[
            "event_id"
        ].map(
            event_lookup[
                "ticket_price_standing"
            ]
        ),

        transaction_records[
            "event_id"
        ].map(
            event_lookup[
                "ticket_price_seated"
            ]
        ),
    )

    assert np.allclose(
        transaction_records[
            "unit_price"
        ],
        expected_unit_prices,
    )

    ###########################################################################
    # Transaction total validation
    ###########################################################################

    expected_transaction_totals = (
        transaction_records[
            "ticket_quantity"
        ]
        *
        transaction_records[
            "unit_price"
        ]
    ).round(2)

    assert np.allclose(
        transaction_records[
            "transaction_total"
        ],
        expected_transaction_totals,
    )

    ###########################################################################
    # Ticket attendance reconciliation
    ###########################################################################

    generated_standing_tickets = int(
        transaction_records.loc[
            transaction_records[
                "ticket_type"
            ]
            ==
            "Standing",
            "ticket_quantity",
        ].sum()
    )

    generated_seated_tickets = int(
        transaction_records.loc[
            transaction_records[
                "ticket_type"
            ]
            ==
            "Seated",
            "ticket_quantity",
        ].sum()
    )

    generated_total_tickets = int(
        transaction_records[
            "ticket_quantity"
        ].sum()
    )

    expected_standing_tickets = int(
        events[
            "event_standing_attendance"
        ].sum()
    )

    expected_seated_tickets = int(
        events[
            "event_seated_attendance"
        ].sum()
    )

    expected_total_tickets = int(
        events[
            "event_total_attendance"
        ].sum()
    )

    assert (
        generated_standing_tickets
        ==
        expected_standing_tickets
    )

    assert (
        generated_seated_tickets
        ==
        expected_seated_tickets
    )

    assert (
        generated_total_tickets
        ==
        expected_total_tickets
    )

    ###########################################################################
    # Validate every event and ticket type individually
    ###########################################################################

    expected_standing = (
        events[
            [
                "event_id",
                "event_standing_attendance",
            ]
        ]
        .rename(
            columns={
                "event_standing_attendance":
                    "expected_quantity",
            }
        )
    )

    expected_standing[
        "ticket_type"
    ] = "Standing"

    expected_seated = (
        events[
            [
                "event_id",
                "event_seated_attendance",
            ]
        ]
        .rename(
            columns={
                "event_seated_attendance":
                    "expected_quantity",
            }
        )
    )

    expected_seated[
        "ticket_type"
    ] = "Seated"

    expected_reconciliation = pd.concat(
        [
            expected_standing,
            expected_seated,
        ],
        ignore_index=True,
    )

    generated_reconciliation = (
        transaction_records
        .groupby(
            [
                "event_id",
                "ticket_type",
            ],
            as_index=False,
        )[
            "ticket_quantity"
        ]
        .sum()
        .rename(
            columns={
                "ticket_quantity":
                    "generated_quantity",
            }
        )
    )

    reconciliation = (
        expected_reconciliation
        .merge(
            generated_reconciliation,
            on=[
                "event_id",
                "ticket_type",
            ],
            how="left",
        )
    )

    reconciliation[
        "generated_quantity"
    ] = (
        reconciliation[
            "generated_quantity"
        ]
        .fillna(0)
        .astype(int)
    )

    assert (
        reconciliation[
            "expected_quantity"
        ]
        ==
        reconciliation[
            "generated_quantity"
        ]
    ).all()

    ###########################################################################
    # Validate payment metadata
    ###########################################################################

    assert (
        transaction_records[
            "card_last_4"
        ]
        .astype(str)
        .str.fullmatch(
            r"\d{4}"
        )
        .all()
    )

    assert set(
        transaction_records[
            "card_brand"
        ]
    ).issubset(
        set(
            CARD_BRANDS
        )
    )

    ###########################################################################
    # Visualise validation results
    ###########################################################################

    print(
        "\nE-commerce validation completed successfully."
    )

    print(
        f"Total transactions: "
        f"{len(transaction_records):,}"
    )

    print(
        f"Unique purchasers represented: "
        f"{transaction_records['ground_truth_id'].nunique():,}"
    )

    print(
        f"Total tickets sold: "
        f"{generated_total_tickets:,}"
    )

    print(
        f"Standing tickets sold: "
        f"{generated_standing_tickets:,}"
    )

    print(
        f"Seated tickets sold: "
        f"{generated_seated_tickets:,}"
    )

    ###########################################################################
    # Guest transaction rates
    ###########################################################################

    guest_transaction_rate = (
        transaction_records[
            "guest_transaction"
        ]
        .mean()
    )

    print(
        f"\nGuest transactions: "
        f"{guest_transaction_rate:.2%}"
    )

    guest_rate_by_segment = (
        transaction_records
        .groupby(
            "customer_segment"
        )[
            "guest_transaction"
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

    print(
        "\nGuest transaction rate by customer segment:"
    )

    print(
        guest_rate_by_segment
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    ###########################################################################
    # Guest checkout rates among portal account holders
    ###########################################################################

    portal_customer_records = transaction_records[
        transaction_records[
            "available_portal_user_id"
        ].notna()
    ]

    portal_guest_rate_by_segment = (
        portal_customer_records
        .groupby(
            "customer_segment"
        )[
            "guest_transaction"
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

    print(
        "\nGuest checkout rate among portal account holders:"
    )

    print(
        portal_guest_rate_by_segment
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    ###########################################################################
    # Customer coverage by segment
    ###########################################################################

    unique_purchasers = (
        transaction_records[
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

    purchaser_segment_counts = (
        unique_purchasers[
            "customer_segment"
        ]
        .value_counts()
    )

    segment_coverage = (
        purchaser_segment_counts
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

    print(
        "\nE-commerce customer coverage by segment:"
    )

    print(
        segment_coverage
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    ###########################################################################
    # Ticket type by customer age
    ###########################################################################

    age_ticket_summary = (
        transaction_records
        .assign(
            age_group=np.where(
                transaction_records[
                    "customer_age"
                ]
                >=
                AGE_THRESHOLD_SEATED,
                "60+",
                "Under 60",
            )
        )
        .groupby(
            [
                "age_group",
                "ticket_type",
            ]
        )[
            "ticket_quantity"
        ]
        .sum()
        .unstack(
            fill_value=0
        )
    )

    age_ticket_summary[
        "total"
    ] = (
        age_ticket_summary.sum(
            axis=1
        )
    )

    age_ticket_summary[
        "seated_percentage"
    ] = (
        age_ticket_summary[
            "Seated"
        ]
        /
        age_ticket_summary[
            "total"
        ]
        *
        100
    ).round(2)

    print(
        "\nTicket type behaviour by age:"
    )

    print(
        age_ticket_summary
    )

    ###########################################################################
    # Missing-value validation
    ###########################################################################

    operational_fields = [
        "portal_user_id",
        "email",
        "telephone_number",
        "billing_address",
        "billing_postcode",
    ]

    missing_summary = (
        transaction_records[
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
        in transaction_records.columns
        if column.startswith(
            "dq_"
        )
    ]

    data_quality_summary = (
        transaction_records[
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
    # Total revenue
    ###########################################################################

    total_revenue = round(
        transaction_records[
            "transaction_total"
        ].sum(),
        2,
    )

    print(
        f"\nTotal synthetic ticket revenue: "
        f"£{total_revenue:,.2f}"
    )

    ###########################################################################
    # Create validation summary output
    ###########################################################################

    validation_summary = pd.DataFrame(
        {
            "metric": [
                "MAIN VALIDATION METRICS",
                "total_transactions",
                "unique_purchasers",
                "total_tickets_sold",
                "standing_tickets_sold",
                "seated_tickets_sold",
                "tickets_purchased_after_event_date",
                "tickets_purchased_outside_snapshot",
                "guest_transaction_rate",
                "registered_transaction_rate",
                "total_ticket_revenue",
                "event_ticket_type_reconciliation",
            ],
            "value": [
                "",
                len(
                    transaction_records
                ),
                transaction_records[
                    "ground_truth_id"
                ].nunique(),
                generated_total_tickets,
                tickets_purchased_after_event_date,
                tickets_purchased_outside_snapshot,
                generated_standing_tickets,
                generated_seated_tickets,
                round(
                    guest_transaction_rate,
                    4,
                ),
                round(
                    1
                    -
                    guest_transaction_rate,
                    4,
                ),
                total_revenue,
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
                "ECOMMERCE CUSTOMER COVERAGE BY SEGMENT",
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
    # Guest transaction rates by segment
    ###########################################################################

    guest_summary = pd.DataFrame(
        {
            "metric": [
                "",
                "GUEST TRANSACTION RATE BY SEGMENT",
                "guest_rate_new",
                "guest_rate_occasional",
                "guest_rate_regular",
                "guest_rate_vip",
            ],
            "value": [
                "",
                "",
                f"{guest_rate_by_segment['New']:.2%}",
                f"{guest_rate_by_segment['Occasional']:.2%}",
                f"{guest_rate_by_segment['Regular']:.2%}",
                f"{guest_rate_by_segment['VIP']:.2%}",
            ],
        }
    )

    ###########################################################################
    # Ticket type behaviour by age
    ###########################################################################

    age_60_plus_seated_percentage = age_ticket_summary.loc[
        "60+",
        "seated_percentage",
    ]
    age_under_60_seated_percentage = age_ticket_summary.loc[
        "Under 60",
        "seated_percentage",
    ]

    age_ticket_validation = pd.DataFrame(
        {
            "metric": [
                "",
                "TICKET TYPE BEHAVIOUR BY AGE",
                "age_60_plus_seated_tickets",
                "age_60_plus_standing_tickets",
                "age_60_plus_total_tickets",
                "age_60_plus_seated_percentage",
                "age_under_60_seated_tickets",
                "age_under_60_standing_tickets",
                "age_under_60_total_tickets",
                "age_under_60_seated_percentage",
            ],
            "value": [
                "",
                "",
                int(
                    age_ticket_summary.loc[
                        "60+",
                        "Seated",
                    ]
                ),
                int(
                    age_ticket_summary.loc[
                        "60+",
                        "Standing",
                    ]
                ),
                int(
                    age_ticket_summary.loc[
                        "60+",
                        "total",
                    ]
                ),
                (
                    f"{age_60_plus_seated_percentage:.2f}%",
                ),
                int(
                    age_ticket_summary.loc[
                        "Under 60",
                        "Seated",
                    ]
                ),
                int(
                    age_ticket_summary.loc[
                        "Under 60",
                        "Standing",
                    ]
                ),
                int(
                    age_ticket_summary.loc[
                        "Under 60",
                        "total",
                    ]
                ),
                (
                    f"{age_under_60_seated_percentage:.2f}%"
                ),
            ],
        }
    )

    ###########################################################################
    # Missing value percentages
    ###########################################################################

    missing_value_summary = pd.DataFrame(
        {
            "metric": [
                "",
                "MISSING VALUE PERCENTAGES",
                "missing_portal_user_id",
                "missing_email",
                "missing_telephone_number",
                "missing_billing_address",
                "missing_billing_postcode",
            ],
            "value": [
                "",
                "",
                f"{missing_summary['portal_user_id']:.2f}%",
                f"{missing_summary['email']:.2f}%",
                f"{missing_summary['telephone_number']:.2f}%",
                f"{missing_summary['billing_address']:.2f}%",
                f"{missing_summary['billing_postcode']:.2f}%",
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
            guest_summary,
            age_ticket_validation,
            missing_value_summary,
            data_quality_output,
        ],
        ignore_index=True,
    )

    return validation_summary


###############################################################################
# 16. Export E-commerce outputs
###############################################################################

def export_ecommerce_outputs(
    transaction_records: pd.DataFrame,
    validation_summary: pd.DataFrame,
) -> None:

    # Create output folders
    ECOMMERCE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GROUND_TRUTH_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Operational E-commerce dataset
    ###########################################################################

    operational_columns = [
        "transaction_id",
        "transaction_date_time",
        "guest_transaction",
        "portal_user_id",
        "billing_address",
        "billing_postcode",
        "event_id",
        "event_name",
        "event_category",
        "event_date",
        "ticket_type",
        "ticket_quantity",
        "unit_price",
        "transaction_total",
        "card_brand",
        "cardholder_name",
        "card_last_4",
        "email",
        "purchaser_first_name",
        "purchaser_surname",
        "telephone_number",
    ]

    operational_ecommerce = (
        transaction_records[
            operational_columns
        ]
        .copy()
    )

    operational_ecommerce[
        "event_date"
    ] = pd.to_datetime(
        operational_ecommerce[
            "event_date"
        ]
    ).dt.date

    operational_ecommerce.to_csv(
        ECOMMERCE_OUTPUT_PATH,
        index=False,
    )

    ###########################################################################
    # Hidden E-commerce ground truth mapping
    ###########################################################################

    data_quality_columns = [
        column
        for column
        in transaction_records.columns
        if column.startswith(
            "dq_"
        )
    ]

    ground_truth_columns = [
        "transaction_id",
        "ground_truth_id",
        "guest_transaction",
        "portal_user_id",
        "available_portal_user_id",
        "event_id",
        "ticket_type",
    ] + data_quality_columns

    ecommerce_ground_truth_mapping = (
        transaction_records[
            ground_truth_columns
        ]
        .copy()
    )

    ecommerce_ground_truth_mapping.to_csv(
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
# 17. Main function
###############################################################################

def main() -> None:

    # Load source and reference datasets
    (
        canonical,
        events,
        portal_mapping,
        crm_profiles,
    ) = load_source_datasets()

    # Select approximately 8,000 unique purchasers
    selected_customers = (
        select_ecommerce_customers(
            canonical=canonical,
            events=events,
            portal_mapping=portal_mapping,
        )
    )

    # Generate transaction-level ticket quantities
    # which exactly reconcile with event attendance
    transaction_plan = (
        generate_transaction_plan(
            events=events
        )
    )

    # Assign canonical purchasers to transactions
    transaction_plan = (
        assign_purchasers_to_transactions(
            transaction_plan=transaction_plan,
            selected_customers=selected_customers,
        )
    )

    # Assign guest and registered checkout status
    transaction_plan = (
        assign_guest_transactions(
            transaction_plan
        )
    )

    # Generate clean transaction records
    transaction_records = (
        generate_clean_ecommerce_records(
            transaction_plan=transaction_plan,
            crm_profiles=crm_profiles,
        )
    )

    # Generate one plausible alternative email
    # for each represented customer
    alternative_email_mapping = (
        generate_alternative_email_mapping(
            selected_customers=selected_customers,
            canonical=canonical,
        )
    )

    # Introduce controlled data quality issues
    transaction_records = (
        introduce_ecommerce_data_quality_issues(
            transaction_records=transaction_records,
            alternative_email_mapping=alternative_email_mapping,
        )
    )

    # Sort chronologically and assign transaction IDs
    transaction_records = (
        finalise_transaction_ids(
            transaction_records
        )
    )

    # Validate final E-commerce environment
    validation_summary = (
        validate_ecommerce_records(
            transaction_records=transaction_records,
            canonical=canonical,
            events=events,
        )
    )

    # Export operational and reference datasets
    export_ecommerce_outputs(
        transaction_records=transaction_records,
        validation_summary=validation_summary,
    )

    print(
        f"\nE-commerce dataset saved to: "
        f"{ECOMMERCE_OUTPUT_PATH.resolve()}"
    )

    print(
        f"E-commerce ground truth saved to: "
        f"{GROUND_TRUTH_OUTPUT_PATH.resolve()}"
    )

    print(
        f"E-commerce validation summary saved to: "
        f"{VALIDATION_OUTPUT_PATH.resolve()}"
    )

    print(
        "\nFirst five E-commerce records:"
    )

    print(
        transaction_records[
            [
                "transaction_id",
                "transaction_date_time",
                "guest_transaction",
                "portal_user_id",
                "event_id",
                "event_name",
                "ticket_type",
                "ticket_quantity",
                "unit_price",
                "transaction_total",
                "purchaser_first_name",
                "purchaser_surname",
                "email",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()
