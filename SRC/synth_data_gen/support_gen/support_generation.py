###############################################################################
# Imports
###############################################################################

from pathlib import Path
import re

import numpy as np
import pandas as pd


###############################################################################
# 1. Generation configuration
###############################################################################

# Support-specific seed ensures reproducibility
SUPPORT_SEED = 107

# Target operational dataset size
NUMBER_OF_SUPPORT_TICKETS = 5_000

# Number of unique canonical customers represented
UNIQUE_SUPPORT_CUSTOMERS = 3_000

# Operational snapshot
OPERATION_START_DATETIME = pd.Timestamp(
    "2025-01-01 00:00:00"
)

SNAPSHOT_END_DATETIME = pd.Timestamp(
    "2025-12-31 23:59:59"
)

# Input dataset
CANONICAL_INPUT_PATH = Path(
    "data/canonical/canonical_customers.csv"
)

# Output paths
SUPPORT_OUTPUT_PATH = Path(
    "data/raw/support_ticket_logs.csv"
)

GROUND_TRUTH_OUTPUT_PATH = Path(
    "data/reference/support_ground_truth_mapping.csv"
)

VALIDATION_OUTPUT_PATH = Path(
    "data/reference/support_validation_summary.csv"
)


###############################################################################
# 2. Weighting assumptions
###############################################################################

# Customer inclusion is weighted using canonical customer_segment.
# Higher-engagement segments are more likely to appear in Support records.
SUPPORT_SEGMENT_SELECTION_WEIGHTS = {
    "New": 0.80,
    "Occasional": 1.00,
    "Regular": 1.20,
    "VIP": 1.40,
}

# Higher-engagement segments are more likely to generate repeat tickets.
REPEAT_TICKET_WEIGHTS = {
    "New": 0.80,
    "Occasional": 1.00,
    "Regular": 1.20,
    "VIP": 1.40,
}

# Support contact channels, email is more likely used channel.
CONTACT_CHANNELS = [
    "Email",
    "Phone",
]

CONTACT_CHANNEL_PROBABILITIES = [
    0.70,
    0.30,
]

# Refund requests are the most frequent issue category.
ISSUE_CATEGORIES = [
    "Refund request",
    "Accessibility",
    "Complaint",
    "Enquiry",
]

ISSUE_CATEGORY_WEIGHTS = {
    "Refund request": 0.40,
    "Accessibility": 0.15,
    "Complaint": 0.20,
    "Enquiry": 0.25,
}

# Customers aged 60+ are more likely to create Accessibility tickets.
ACCESSIBILITY_AGE_THRESHOLD = 60
ACCESSIBILITY_OVER_60_MULTIPLIER = 2.00

# Most support tickets are assumed to be resolved within the snapshot.
RESOLVED_TICKET_PROBABILITY = 0.86


###############################################################################
# 3. Support behaviour and data quality assumptions
###############################################################################

# Typical resolution duration in hours by issue category.
RESOLUTION_TIME_MEAN_HOURS = {
    "Refund request": 18.0,
    "Accessibility": 12.0,
    "Complaint": 24.0,
    "Enquiry": 6.0,
}

# Gamma distribution introduces realistic right-skewed resolution times.
RESOLUTION_TIME_GAMMA_SHAPE = 2.00

MINIMUM_RESOLUTION_MINUTES = 5
MAXIMUM_RESOLUTION_MINUTES = 72 * 60

# Manual-entry errors.
NAME_TYPO_RATE = 0.03
EMAIL_TYPO_RATE = 0.07
PHONE_ERROR_RATE = 0.07

# Contact details may be incomplete depending on contact channel.
EMAIL_CHANNEL_MISSING_PHONE_RATE = 0.30
PHONE_CHANNEL_MISSING_EMAIL_RATE = 0.30

# Some tickets are assigned an incorrect but still valid issue category.
INCONSISTENT_CATEGORY_RATE = 0.05


###############################################################################
# 4. Initialise generator
###############################################################################

rng = np.random.default_rng(
    SUPPORT_SEED
)


###############################################################################
# 5. Helper functions
###############################################################################

# Select a random datetime between two datetime boundaries.
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


# Calculate customer age at snapshot date.
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


# Introduce a small spelling error into a name.
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


# Introduce a typo into the local section of an email address.
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


# Introduce an incorrect digit into a telephone number.
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


# Return issue-category probabilities after applying age weighting.
def get_issue_category_probabilities(
    customer_age: int,
) -> np.ndarray:

    weights = np.array(
        [
            ISSUE_CATEGORY_WEIGHTS[
                category
            ]
            for category
            in ISSUE_CATEGORIES
        ],
        dtype=float,
    )

    if customer_age >= ACCESSIBILITY_AGE_THRESHOLD:

        accessibility_index = (
            ISSUE_CATEGORIES.index(
                "Accessibility"
            )
        )

        weights[
            accessibility_index
        ] *= (
            ACCESSIBILITY_OVER_60_MULTIPLIER
        )

    return (
        weights
        / weights.sum()
    )


# Select an incorrect but valid issue category.
def select_incorrect_issue_category(
    intended_category: str,
) -> str:

    alternative_categories = [
        category
        for category
        in ISSUE_CATEGORIES
        if category != intended_category
    ]

    return str(
        rng.choice(
            alternative_categories
        )
    )


# Generate a positive resolution duration formatted as HH:MM.
def generate_resolution_time(
    issue_category: str,
) -> str:

    mean_hours = float(
        RESOLUTION_TIME_MEAN_HOURS[
            issue_category
        ]
    )

    scale = (
        mean_hours
        / RESOLUTION_TIME_GAMMA_SHAPE
    )

    duration_hours = float(
        rng.gamma(
            shape=RESOLUTION_TIME_GAMMA_SHAPE,
            scale=scale,
        )
    )

    duration_minutes = int(
        round(
            duration_hours
            * 60
        )
    )

    duration_minutes = max(
        MINIMUM_RESOLUTION_MINUTES,
        duration_minutes,
    )

    duration_minutes = min(
        MAXIMUM_RESOLUTION_MINUTES,
        duration_minutes,
    )

    hours = (
        duration_minutes
        // 60
    )

    minutes = (
        duration_minutes
        % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}"
    )


# Convert HH:MM resolution duration into total minutes.
def resolution_time_to_minutes(
    value: str,
) -> int:

    hours, minutes = (
        str(value)
        .split(":")
    )

    return (
        int(hours) * 60
        + int(minutes)
    )


###############################################################################
# 6. Load and validate source dataset
###############################################################################

def load_source_dataset() -> pd.DataFrame:

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

    return canonical


###############################################################################
# 7. Select Support customer population
###############################################################################

def select_support_customers(
    canonical: pd.DataFrame,
) -> pd.DataFrame:

    eligible_customers = (
        canonical[
            canonical[
                "registration_date"
            ]
            <=
            SNAPSHOT_END_DATETIME
        ]
        .copy()
    )

    assert (
        len(
            eligible_customers
        )
        >=
        UNIQUE_SUPPORT_CUSTOMERS
    )

    selection_weights = (
        eligible_customers[
            "customer_segment"
        ]
        .map(
            SUPPORT_SEGMENT_SELECTION_WEIGHTS
        )
        .astype(float)
    )

    selection_weights = (
        selection_weights
        / selection_weights.sum()
    )

    selected_indices = rng.choice(
        eligible_customers.index.to_numpy(),
        size=UNIQUE_SUPPORT_CUSTOMERS,
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

    selected_customers[
        "customer_age"
    ] = selected_customers.apply(
        lambda row:
        calculate_age(
            date_of_birth=row[
                "date_of_birth"
            ],
            reference_date=SNAPSHOT_END_DATETIME,
        ),
        axis=1,
    )

    return selected_customers


###############################################################################
# 8. Generate Support ticket plan
###############################################################################

def generate_support_ticket_plan(
    selected_customers: pd.DataFrame,
) -> pd.DataFrame:

    support_ticket_plan = []

    ###########################################################################
    # Ensure every selected customer is represented at least once
    ###########################################################################

    for customer in (
        selected_customers
        .itertuples(
            index=False
        )
    ):

        support_ticket_plan.append(
            {
                "ground_truth_id": customer.ground_truth_id,
            }
        )

    ###########################################################################
    # Generate remaining repeated tickets using segment weighting
    ###########################################################################

    remaining_ticket_count = (
        NUMBER_OF_SUPPORT_TICKETS
        - len(
            support_ticket_plan
        )
    )

    if remaining_ticket_count < 0:

        raise ValueError(
            "Unique Support customer target exceeds the configured "
            "Support ticket target."
        )

    repeat_weights = (
        selected_customers[
            "customer_segment"
        ]
        .map(
            REPEAT_TICKET_WEIGHTS
        )
        .astype(float)
    )

    repeat_weights = (
        repeat_weights
        / repeat_weights.sum()
    )

    repeated_customer_indices = rng.choice(
        selected_customers.index.to_numpy(),
        size=remaining_ticket_count,
        replace=True,
        p=repeat_weights.to_numpy(),
    )

    for customer_index in (
        repeated_customer_indices
    ):

        customer = selected_customers.loc[
            customer_index
        ]

        support_ticket_plan.append(
            {
                "ground_truth_id": customer[
                    "ground_truth_id"
                ],
            }
        )

    support_ticket_plan = pd.DataFrame(
        support_ticket_plan
    )

    support_ticket_plan = (
        support_ticket_plan
        .merge(
            selected_customers,
            on="ground_truth_id",
            how="left",
            validate="many_to_one",
        )
    )

    assert (
        len(
            support_ticket_plan
        )
        ==
        NUMBER_OF_SUPPORT_TICKETS
    )

    assert (
        support_ticket_plan[
            "ground_truth_id"
        ].nunique()
        ==
        UNIQUE_SUPPORT_CUSTOMERS
    )

    return support_ticket_plan


###############################################################################
# 9. Generate clean Support ticket records
###############################################################################

def generate_clean_support_ticket_records(
    support_ticket_plan: pd.DataFrame,
) -> pd.DataFrame:

    support_ticket_records = []

    for ticket in (
        support_ticket_plan
        .itertuples(
            index=False
        )
    ):

        #######################################################################
        # Generate ticket creation time
        #######################################################################

        earliest_ticket_datetime = max(
            OPERATION_START_DATETIME,
            pd.Timestamp(
                ticket.registration_date
            ),
        )

        ticket_created_datetime = (
            generate_random_datetime(
                start_datetime=earliest_ticket_datetime,
                end_datetime=SNAPSHOT_END_DATETIME,
            )
        )

        #######################################################################
        # Generate contact channel
        #######################################################################

        contact_channel = str(
            rng.choice(
                CONTACT_CHANNELS,
                p=CONTACT_CHANNEL_PROBABILITIES,
            )
        )

        #######################################################################
        # Generate issue category
        #######################################################################

        issue_probabilities = (
            get_issue_category_probabilities(
                customer_age=int(
                    ticket.customer_age
                )
            )
        )

        intended_issue_category = str(
            rng.choice(
                ISSUE_CATEGORIES,
                p=issue_probabilities,
            )
        )

        #######################################################################
        # Generate ticket status and resolution time
        #######################################################################

        ticket_status = (
            "Resolved"
            if (
                rng.random()
                <
                RESOLVED_TICKET_PROBABILITY
            )
            else "Open"
        )

        if ticket_status == "Resolved":

            resolution_time = (
                generate_resolution_time(
                    issue_category=(
                        intended_issue_category
                    )
                )
            )

            resolution_minutes = (
                resolution_time_to_minutes(
                    resolution_time
                )
            )

            resolution_datetime = (
                ticket_created_datetime
                + pd.Timedelta(
                    minutes=resolution_minutes
                )
            )

            # A ticket cannot be recorded as Resolved where its
            # generated resolution would fall outside the snapshot.
            if (
                resolution_datetime
                >
                SNAPSHOT_END_DATETIME
            ):
                ticket_status = "Open"
                resolution_time = pd.NA

        else:
            resolution_time = pd.NA

        #######################################################################
        # Append clean Support ticket
        #######################################################################

        support_ticket_records.append(
            {
                # Internal generation fields
                "ground_truth_id": ticket.ground_truth_id,
                "customer_segment": ticket.customer_segment,
                "customer_age": int(
                    ticket.customer_age
                ),
                "registration_date": ticket.registration_date,
                "intended_issue_category": intended_issue_category,

                # Operational Support fields
                "support_ticket_id": pd.NA,
                "support_ticket_created_date_time":
                    ticket_created_datetime,
                "requester_name": (
                    f"{ticket.first_name} "
                    f"{ticket.surname}"
                ),
                "requester_email": ticket.email,
                "requester_phone": ticket.telephone_number,
                "contact_channel": contact_channel,
                "issue_category": intended_issue_category,
                "ticket_status": ticket_status,
                "resolution_time_hours_minutes": resolution_time,
            }
        )

    return pd.DataFrame(
        support_ticket_records
    )


###############################################################################
# 10. Introduce Support data quality issues
###############################################################################

def introduce_support_data_quality_issues(
    support_ticket_records: pd.DataFrame,
) -> pd.DataFrame:

    support_ticket_records = (
        support_ticket_records
        .copy()
    )

    data_quality_flags = [
        "dq_name_typo",
        "dq_email_typo",
        "dq_phone_error",
        "dq_missing_email",
        "dq_missing_phone",
        "dq_incorrect_issue_category",
    ]

    for flag in (
        data_quality_flags
    ):

        support_ticket_records[
            flag
        ] = False

    for index in (
        support_ticket_records.index
    ):

        contact_channel = (
            support_ticket_records.at[
                index,
                "contact_channel",
            ]
        )

        #######################################################################
        # Requester name
        #######################################################################

        if (
            rng.random()
            <
            NAME_TYPO_RATE
        ):

            support_ticket_records.at[
                index,
                "requester_name",
            ] = introduce_name_typo(
                support_ticket_records.at[
                    index,
                    "requester_name",
                ]
            )

            support_ticket_records.at[
                index,
                "dq_name_typo",
            ] = True

        #######################################################################
        # Requester email
        #######################################################################

        if contact_channel == "Phone":

            if (
                rng.random()
                <
                PHONE_CHANNEL_MISSING_EMAIL_RATE
            ):

                support_ticket_records.at[
                    index,
                    "requester_email",
                ] = pd.NA

                support_ticket_records.at[
                    index,
                    "dq_missing_email",
                ] = True

            elif (
                rng.random()
                <
                EMAIL_TYPO_RATE
            ):

                support_ticket_records.at[
                    index,
                    "requester_email",
                ] = introduce_email_typo(
                    support_ticket_records.at[
                        index,
                        "requester_email",
                    ]
                )

                support_ticket_records.at[
                    index,
                    "dq_email_typo",
                ] = True

        elif (
            rng.random()
            <
            EMAIL_TYPO_RATE
        ):

            support_ticket_records.at[
                index,
                "requester_email",
            ] = introduce_email_typo(
                support_ticket_records.at[
                    index,
                    "requester_email",
                ]
            )

            support_ticket_records.at[
                index,
                "dq_email_typo",
            ] = True

        #######################################################################
        # Requester phone
        #######################################################################

        if contact_channel == "Email":

            if (
                rng.random()
                <
                EMAIL_CHANNEL_MISSING_PHONE_RATE
            ):

                support_ticket_records.at[
                    index,
                    "requester_phone",
                ] = pd.NA

                support_ticket_records.at[
                    index,
                    "dq_missing_phone",
                ] = True

            elif (
                rng.random()
                <
                PHONE_ERROR_RATE
            ):

                support_ticket_records.at[
                    index,
                    "requester_phone",
                ] = introduce_phone_error(
                    support_ticket_records.at[
                        index,
                        "requester_phone",
                    ]
                )

                support_ticket_records.at[
                    index,
                    "dq_phone_error",
                ] = True

        elif (
            rng.random()
            <
            PHONE_ERROR_RATE
        ):

            support_ticket_records.at[
                index,
                "requester_phone",
            ] = introduce_phone_error(
                support_ticket_records.at[
                    index,
                    "requester_phone",
                ]
            )

            support_ticket_records.at[
                index,
                "dq_phone_error",
            ] = True

        #######################################################################
        # Issue categorisation
        #######################################################################

        if (
            rng.random()
            <
            INCONSISTENT_CATEGORY_RATE
        ):

            intended_category = (
                support_ticket_records.at[
                    index,
                    "intended_issue_category",
                ]
            )

            support_ticket_records.at[
                index,
                "issue_category",
            ] = select_incorrect_issue_category(
                intended_category=intended_category
            )

            support_ticket_records.at[
                index,
                "dq_incorrect_issue_category",
            ] = True

    return support_ticket_records


###############################################################################
# 11. Finalise Support ticket IDs
###############################################################################

def finalise_support_ticket_ids(
    support_ticket_records: pd.DataFrame,
) -> pd.DataFrame:

    # Sort tickets chronologically before assigning sequential identifiers.
    support_ticket_records = (
        support_ticket_records
        .sort_values(
            "support_ticket_created_date_time"
        )
        .reset_index(
            drop=True
        )
    )

    support_ticket_records[
        "support_ticket_id"
    ] = [
        f"SUP{ticket_number:06d}"
        for ticket_number
        in range(
            1,
            len(
                support_ticket_records
            ) + 1,
        )
    ]

    return support_ticket_records


###############################################################################
# 12. Validate Support ticket dataset
###############################################################################

def validate_support_ticket_records(
    support_ticket_records: pd.DataFrame,
    canonical: pd.DataFrame,
) -> pd.DataFrame:

    ###########################################################################
    # Basic validation
    ###########################################################################

    assert (
        len(
            support_ticket_records
        )
        ==
        NUMBER_OF_SUPPORT_TICKETS
    )

    assert support_ticket_records[
        "support_ticket_id"
    ].is_unique

    assert (
        support_ticket_records[
            "ground_truth_id"
        ].nunique()
        ==
        UNIQUE_SUPPORT_CUSTOMERS
    )

    assert set(
        support_ticket_records[
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
    # Ticket timing validation
    ###########################################################################

    ticket_dates = pd.to_datetime(
        support_ticket_records[
            "support_ticket_created_date_time"
        ]
    )

    registration_dates = pd.to_datetime(
        support_ticket_records[
            "registration_date"
        ]
    )

    assert (
        ticket_dates
        >=
        OPERATION_START_DATETIME
    ).all()

    assert (
        ticket_dates
        <=
        SNAPSHOT_END_DATETIME
    ).all()

    assert (
        ticket_dates
        >=
        registration_dates
    ).all()

    ###########################################################################
    # Contact-channel validation
    ###########################################################################

    assert set(
        support_ticket_records[
            "contact_channel"
        ]
    ).issubset(
        set(
            CONTACT_CHANNELS
        )
    )

    email_tickets = (
        support_ticket_records[
            support_ticket_records[
                "contact_channel"
            ]
            ==
            "Email"
        ]
    )

    phone_tickets = (
        support_ticket_records[
            support_ticket_records[
                "contact_channel"
            ]
            ==
            "Phone"
        ]
    )

    # Email channel requires an email address.
    assert email_tickets[
        "requester_email"
    ].notna().all()

    # Phone channel requires a telephone number.
    assert phone_tickets[
        "requester_phone"
    ].notna().all()

    ###########################################################################
    # Issue-category validation
    ###########################################################################

    assert set(
        support_ticket_records[
            "issue_category"
        ]
    ).issubset(
        set(
            ISSUE_CATEGORIES
        )
    )

    issue_category_counts = (
        support_ticket_records[
            "issue_category"
        ]
        .value_counts()
        .reindex(
            ISSUE_CATEGORIES,
            fill_value=0,
        )
    )

    # Refund requests should remain the most frequent category.
    assert (
        issue_category_counts.idxmax()
        ==
        "Refund request"
    )

    over_60_records = (
        support_ticket_records[
            support_ticket_records[
                "customer_age"
            ]
            >=
            ACCESSIBILITY_AGE_THRESHOLD
        ]
    )

    under_60_records = (
        support_ticket_records[
            support_ticket_records[
                "customer_age"
            ]
            <
            ACCESSIBILITY_AGE_THRESHOLD
        ]
    )

    over_60_accessibility_rate = (
        over_60_records[
            "issue_category"
        ]
        .eq("Accessibility")
        .mean()
    )

    under_60_accessibility_rate = (
        under_60_records[
            "issue_category"
        ]
        .eq("Accessibility")
        .mean()
    )

    assert (
        over_60_accessibility_rate
        >
        under_60_accessibility_rate
    )

    ###########################################################################
    # Ticket-status and resolution-time validation
    ###########################################################################

    assert set(
        support_ticket_records[
            "ticket_status"
        ]
    ).issubset(
        {
            "Open",
            "Resolved",
        }
    )

    open_tickets = (
        support_ticket_records[
            support_ticket_records[
                "ticket_status"
            ]
            ==
            "Open"
        ]
    )

    resolved_tickets = (
        support_ticket_records[
            support_ticket_records[
                "ticket_status"
            ]
            ==
            "Resolved"
        ]
    )

    assert open_tickets[
        "resolution_time_hours_minutes"
    ].isna().all()

    assert resolved_tickets[
        "resolution_time_hours_minutes"
    ].notna().all()

    assert resolved_tickets[
        "resolution_time_hours_minutes"
    ].astype(str).str.fullmatch(
        r"\d{2}:\d{2}"
    ).all()

    resolution_minutes = (
        resolved_tickets[
            "resolution_time_hours_minutes"
        ]
        .apply(
            resolution_time_to_minutes
        )
    )

    assert (
        resolution_minutes
        >=
        MINIMUM_RESOLUTION_MINUTES
    ).all()

    assert (
        resolution_minutes
        <=
        MAXIMUM_RESOLUTION_MINUTES
    ).all()

    resolved_creation_times = pd.to_datetime(
        resolved_tickets[
            "support_ticket_created_date_time"
        ]
    )

    resolved_end_times = (
        resolved_creation_times
        + pd.to_timedelta(
            resolution_minutes.to_numpy(),
            unit="m",
        )
    )

    assert (
        resolved_end_times
        <=
        SNAPSHOT_END_DATETIME
    ).all()

    ###########################################################################
    # Validation metrics
    ###########################################################################

    total_tickets = len(
        support_ticket_records
    )

    unique_customers = (
        support_ticket_records[
            "ground_truth_id"
        ].nunique()
    )

    email_channel_rate = (
        support_ticket_records[
            "contact_channel"
        ]
        .eq("Email")
        .mean()
    )

    resolved_ticket_rate = (
        support_ticket_records[
            "ticket_status"
        ]
        .eq("Resolved")
        .mean()
    )

    average_resolution_hours = (
        resolution_minutes.mean()
        / 60
    )

    ###########################################################################
    # Customer coverage by segment
    ###########################################################################

    unique_support_customers = (
        support_ticket_records[
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

    support_segment_counts = (
        unique_support_customers[
            "customer_segment"
        ]
        .value_counts()
    )

    segment_coverage = (
        support_segment_counts
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
    # Average ticket frequency by segment
    ###########################################################################

    customer_ticket_counts = (
        support_ticket_records
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
                "size": "ticket_count",
            }
        )
    )

    average_tickets_by_segment = (
        customer_ticket_counts
        .groupby(
            "customer_segment"
        )[
            "ticket_count"
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
    # Missing requester details
    ###########################################################################

    email_ticket_missing_phone_rate = (
        email_tickets[
            "requester_phone"
        ]
        .isna()
        .mean()
    )

    phone_ticket_missing_email_rate = (
        phone_tickets[
            "requester_email"
        ]
        .isna()
        .mean()
    )

    ###########################################################################
    # Data quality issue counts
    ###########################################################################

    data_quality_columns = [
        column
        for column
        in support_ticket_records.columns
        if column.startswith(
            "dq_"
        )
    ]

    data_quality_summary = (
        support_ticket_records[
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
        "\nSupport Ticket validation completed successfully."
    )

    print(
        f"Total support tickets: "
        f"{total_tickets:,}"
    )

    print(
        f"Unique customers represented: "
        f"{unique_customers:,}"
    )

    print(
        f"Email contact channel: "
        f"{email_channel_rate:.2%}"
    )

    print(
        f"Resolved tickets: "
        f"{resolved_ticket_rate:.2%}"
    )

    print(
        f"Average resolution time: "
        f"{average_resolution_hours:.2f} hours"
    )

    print(
        "\nIssue category distribution:"
    )

    print(
        issue_category_counts
    )

    print(
        f"\nAccessibility rate for customers aged 60+: "
        f"{over_60_accessibility_rate:.2%}"
    )

    print(
        f"Accessibility rate for customers under 60: "
        f"{under_60_accessibility_rate:.2%}"
    )

    print(
        "\nSupport customer coverage by segment:"
    )

    print(
        segment_coverage
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    print(
        "\nAverage support tickets per customer by segment:"
    )

    print(
        average_tickets_by_segment
        .round(2)
    )

    print(
        f"\nMissing phone among Email tickets: "
        f"{email_ticket_missing_phone_rate:.2%}"
    )

    print(
        f"Missing email among Phone tickets: "
        f"{phone_ticket_missing_email_rate:.2%}"
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
                "total_support_tickets",
                "unique_customers",
                "email_channel_rate",
                "phone_channel_rate",
                "resolved_ticket_rate",
                "open_ticket_rate",
                "average_resolution_time_hours",
                "refund_request_most_frequent",
                "accessibility_over_60_rate",
                "accessibility_under_60_rate",
                "accessibility_age_weighting_valid",
                "ticket_timing_valid",
                "resolution_time_logic_valid",
            ],
            "value": [
                "",
                total_tickets,
                unique_customers,
                round(
                    email_channel_rate,
                    4,
                ),
                round(
                    1
                    -
                    email_channel_rate,
                    4,
                ),
                round(
                    resolved_ticket_rate,
                    4,
                ),
                round(
                    1
                    -
                    resolved_ticket_rate,
                    4,
                ),
                round(
                    average_resolution_hours,
                    2,
                ),
                True,
                round(
                    over_60_accessibility_rate,
                    4,
                ),
                round(
                    under_60_accessibility_rate,
                    4,
                ),
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
                "SUPPORT CUSTOMER COVERAGE BY SEGMENT",
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
    # Average ticket frequency by segment
    ###########################################################################

    frequency_summary = pd.DataFrame(
        {
            "metric": [
                "",
                "AVERAGE SUPPORT TICKETS PER CUSTOMER BY SEGMENT",
                "average_tickets_new",
                "average_tickets_occasional",
                "average_tickets_regular",
                "average_tickets_vip",
            ],
            "value": [
                "",
                "",
                round(
                    average_tickets_by_segment[
                        "New"
                    ],
                    2,
                ),
                round(
                    average_tickets_by_segment[
                        "Occasional"
                    ],
                    2,
                ),
                round(
                    average_tickets_by_segment[
                        "Regular"
                    ],
                    2,
                ),
                round(
                    average_tickets_by_segment[
                        "VIP"
                    ],
                    2,
                ),
            ],
        }
    )

    ###########################################################################
    # Issue category distribution
    ###########################################################################

    issue_summary = pd.DataFrame(
        {
            "metric": [
                "",
                "ISSUE CATEGORY DISTRIBUTION",
                "refund_request_count",
                "accessibility_count",
                "complaint_count",
                "enquiry_count",
            ],
            "value": [
                "",
                "",
                int(
                    issue_category_counts[
                        "Refund request"
                    ]
                ),
                int(
                    issue_category_counts[
                        "Accessibility"
                    ]
                ),
                int(
                    issue_category_counts[
                        "Complaint"
                    ]
                ),
                int(
                    issue_category_counts[
                        "Enquiry"
                    ]
                ),
            ],
        }
    )

    ###########################################################################
    # Missing requester details by contact channel
    ###########################################################################

    missing_summary = pd.DataFrame(
        {
            "metric": [
                "",
                "MISSING REQUESTER DETAILS BY CONTACT CHANNEL",
                "missing_phone_rate_email_tickets",
                "missing_email_rate_phone_tickets",
                "email_channel_email_required",
                "phone_channel_phone_required",
            ],
            "value": [
                "",
                "",
                f"{email_ticket_missing_phone_rate:.2%}",
                f"{phone_ticket_missing_email_rate:.2%}",
                True,
                True,
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
            issue_summary,
            missing_summary,
            data_quality_output,
        ],
        ignore_index=True,
    )

    return validation_summary


###############################################################################
# 13. Export Support outputs
###############################################################################

def export_support_outputs(
    support_ticket_records: pd.DataFrame,
    validation_summary: pd.DataFrame,
) -> None:

    # Create output folders
    SUPPORT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GROUND_TRUTH_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Operational Support dataset
    ###########################################################################

    operational_columns = [
        "support_ticket_id",
        "support_ticket_created_date_time",
        "requester_name",
        "requester_email",
        "requester_phone",
        "contact_channel",
        "issue_category",
        "ticket_status",
        "resolution_time_hours_minutes",
    ]

    operational_support = (
        support_ticket_records[
            operational_columns
        ]
        .copy()
    )

    operational_support.to_csv(
        SUPPORT_OUTPUT_PATH,
        index=False,
    )

    ###########################################################################
    # Hidden Support ground truth mapping
    ###########################################################################

    data_quality_columns = [
        column
        for column
        in support_ticket_records.columns
        if column.startswith(
            "dq_"
        )
    ]

    ground_truth_columns = [
        "support_ticket_id",
        "ground_truth_id",
        "customer_segment",
        "customer_age",
        "intended_issue_category",
        "contact_channel",
        "issue_category",
    ] + data_quality_columns

    support_ground_truth_mapping = (
        support_ticket_records[
            ground_truth_columns
        ]
        .copy()
    )

    support_ground_truth_mapping.to_csv(
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
# 14. Main function
###############################################################################

def main() -> None:

    # Load canonical customer population
    canonical = load_source_dataset()

    # Select exactly 3,000 unique Support customers
    selected_customers = (
        select_support_customers(
            canonical=canonical
        )
    )

    # Generate exactly 5,000 Support ticket plans
    support_ticket_plan = (
        generate_support_ticket_plan(
            selected_customers=selected_customers
        )
    )

    # Generate clean Support ticket records
    support_ticket_records = (
        generate_clean_support_ticket_records(
            support_ticket_plan=support_ticket_plan
        )
    )

    # Introduce controlled Support data quality issues
    support_ticket_records = (
        introduce_support_data_quality_issues(
            support_ticket_records=(
                support_ticket_records
            )
        )
    )

    # Sort chronologically and assign Support ticket IDs
    support_ticket_records = (
        finalise_support_ticket_ids(
            support_ticket_records=(
                support_ticket_records
            )
        )
    )

    # Validate final Support environment
    validation_summary = (
        validate_support_ticket_records(
            support_ticket_records=(
                support_ticket_records
            ),
            canonical=canonical,
        )
    )

    # Export operational and reference datasets
    export_support_outputs(
        support_ticket_records=(
            support_ticket_records
        ),
        validation_summary=validation_summary,
    )

    print(
        f"\nSupport dataset saved to: "
        f"{SUPPORT_OUTPUT_PATH.resolve()}"
    )

    print(
        f"Support ground truth saved to: "
        f"{GROUND_TRUTH_OUTPUT_PATH.resolve()}"
    )

    print(
        f"Support validation summary saved to: "
        f"{VALIDATION_OUTPUT_PATH.resolve()}"
    )

    print(
        "\nFirst five Support ticket records:"
    )

    print(
        support_ticket_records[
            [
                "support_ticket_id",
                "support_ticket_created_date_time",
                "requester_name",
                "requester_email",
                "requester_phone",
                "contact_channel",
                "issue_category",
                "ticket_status",
                "resolution_time_hours_minutes",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()
