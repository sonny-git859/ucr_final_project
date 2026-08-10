###############################################################################
# Imports
###############################################################################

from pathlib import Path

import numpy as np
import pandas as pd


###############################################################################
# 1. Generation configuration
###############################################################################

# Online-session-specific seed ensures reproducibility
ONLINE_SEED = 105

# Target operational dataset size
NUMBER_OF_ONLINE_SESSIONS = 40_000

# Number of unique canonical customers represented
UNIQUE_ONLINE_CUSTOMERS = 8_000

# Operational snapshot
OPERATION_START_DATETIME = pd.Timestamp(
    "2025-01-01 00:00:00"
)

SNAPSHOT_END_DATETIME = pd.Timestamp(
    "2025-12-31 23:59:59"
)

# Input datasets
CANONICAL_INPUT_PATH = Path(
    "data/canonical/canonical_customers.csv"
)

PORTAL_MAPPING_INPUT_PATH = Path(
    "data/reference/portal_account_mapping.csv"
)

ECOMMERCE_INPUT_PATH = Path(
    "data/raw/ecommerce_transactions.csv"
)

ECOMMERCE_GROUND_TRUTH_INPUT_PATH = Path(
    "data/reference/ecommerce_ground_truth_mapping.csv"
)

# Output paths
ONLINE_OUTPUT_PATH = Path(
    "data/raw/online_sessions.csv"
)

GROUND_TRUTH_OUTPUT_PATH = Path(
    "data/reference/online_ground_truth_mapping.csv"
)

VALIDATION_OUTPUT_PATH = Path(
    "data/reference/online_validation_summary.csv"
)


###############################################################################
# 2. Weighting assumptions
###############################################################################

# Customer inclusion is weighted using canonical customer_segment.
# Higher values represent greater relative likelihood of appearing
# within the Online Session Logs.
ONLINE_SEGMENT_SELECTION_WEIGHTS = {
    "New": 0.80,
    "Occasional": 1.00,
    "Regular": 1.20,
    "VIP": 1.40,
}

# Higher-engagement customer segments are more likely to generate
# repeated sessions once represented in the dataset.
SESSION_FREQUENCY_WEIGHTS = {
    "New": 0.80,
    "Occasional": 1.00,
    "Regular": 1.20,
    "VIP": 1.40,
}

# Typical session duration per customer segment.
# represents target mean seconds
# (gamma distribution applied later for further variation)
SESSION_DURATION_MEANS = {
    "New": 300,
    "Occasional": 480,
    "Regular": 720,
    "VIP": 900,
}

# Converted sessions are weighted towards longer durations.
CONVERTED_SESSION_DURATION_MULTIPLIER = 1.50

# Customers with an available portal account still have a 20%
# probability of browsing without logging in.
PORTAL_HOLDER_GUEST_SESSION_RATE = 0.20

# Mobile sessions are more likely than PC sessions.
DEVICE_TYPES = [
    "Mobile",
    "PC",
]

DEVICE_TYPE_PROBABILITIES = [
    0.65,
    0.35,
]

# E-commerce dataset represents multiple sales channels
# (proportion of transactions assumed to originate from online sessions)
ONLINE_CHANNEL_TRANSACTION_RATE = 0.60

# Non-converted sessions may still create an abandoned basket.
NON_CONVERTED_BASKET_CREATION_RATE = 0.22


###############################################################################
# 3. Session behaviour assumptions
###############################################################################

# Min and max generated session duration (30sec - 1hr)
MINIMUM_SESSION_DURATION_SECONDS = 30
MAXIMUM_SESSION_DURATION_SECONDS = 3_600

# Gamma distribution shape used to introduce realistic variation
# for segment-based session-duration assumptions (right skewed distribution)
SESSION_DURATION_GAMMA_SHAPE = 2.20


###############################################################################
# 4. Initialise generator
###############################################################################

rng = np.random.default_rng(
    ONLINE_SEED
)


###############################################################################
# 5. Helper functions
###############################################################################

# Convert values read from CSV into consistent Boolean representation.
def normalise_boolean_series(
    values: pd.Series,
) -> pd.Series:

    return (
        values
        .astype(str)
        .str.lower()
        .eq("true")
    )


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


# Generate a stochastic session duration based on customer segment.
def generate_session_duration(
    customer_segment: str,
    basket_converted: bool,
) -> int:

    target_mean = float(
        SESSION_DURATION_MEANS[
            customer_segment
        ]
    )

    if basket_converted:
        target_mean *= (
            CONVERTED_SESSION_DURATION_MULTIPLIER
        )

    scale = (
        target_mean
        / SESSION_DURATION_GAMMA_SHAPE
    )

    duration_seconds = int(
        round(
            rng.gamma(
                shape=SESSION_DURATION_GAMMA_SHAPE,
                scale=scale,
            )
        )
    )

    duration_seconds = max(
        MINIMUM_SESSION_DURATION_SECONDS,
        duration_seconds,
    )

    duration_seconds = min(
        MAXIMUM_SESSION_DURATION_SECONDS,
        duration_seconds,
    )

    return duration_seconds


# Generate start and end timestamps for a non-converted session.
def generate_non_converted_session_times(
    registration_date: pd.Timestamp,
    duration_seconds: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:

    earliest_start = max(
        OPERATION_START_DATETIME,
        pd.Timestamp(
            registration_date
        ),
    )

    latest_start = (
        SNAPSHOT_END_DATETIME
        - pd.Timedelta(
            seconds=duration_seconds
        )
    )

    if earliest_start > latest_start:
        earliest_start = latest_start

    session_start = (
        generate_random_datetime(
            start_datetime=earliest_start,
            end_datetime=latest_start,
        )
    )

    session_end = (
        session_start
        + pd.Timedelta(
            seconds=duration_seconds
        )
    )

    return (
        session_start,
        session_end,
    )


# Generate a session window containing the associated transaction time.
def generate_converted_session_times(
    transaction_date_time: pd.Timestamp,
    registration_date: pd.Timestamp,
    duration_seconds: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:

    transaction_date_time = pd.Timestamp(
        transaction_date_time
    )

    earliest_allowed = max(
        OPERATION_START_DATETIME,
        pd.Timestamp(
            registration_date
        ),
    )

    latest_allowed = SNAPSHOT_END_DATETIME

    # Position the transaction between 35% and 85% of the way
    # through the generated session before boundary adjustment.
    transaction_position = float(
        rng.uniform(
            0.35,
            0.85,
        )
    )

    seconds_before_transaction = int(
        round(
            duration_seconds
            * transaction_position
        )
    )

    session_start = (
        transaction_date_time
        - pd.Timedelta(
            seconds=seconds_before_transaction
        )
    )

    session_end = (
        session_start
        + pd.Timedelta(
            seconds=duration_seconds
        )
    )

    # Shift the whole session forward where the initial start
    # falls before the allowed operational boundary.
    if session_start < earliest_allowed:

        shift = (
            earliest_allowed
            - session_start
        )

        session_start += shift
        session_end += shift

    # Shift the whole session backwards where the end exceeds
    # the snapshot boundary.
    if session_end > latest_allowed:

        shift = (
            session_end
            - latest_allowed
        )

        session_start -= shift
        session_end -= shift

    # These assertions ensure the linked transaction remains inside
    # the final session window after any boundary adjustment.
    assert session_start <= transaction_date_time
    assert transaction_date_time <= session_end

    return (
        session_start,
        session_end,
    )


###############################################################################
# 6. Load and validate source datasets
###############################################################################

def load_source_datasets() -> tuple[
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
        "registration_date"
    ] = pd.to_datetime(
        canonical[
            "registration_date"
        ]
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
    # E-commerce records and hidden ground truth
    ###########################################################################

    ecommerce = pd.read_csv(
        ECOMMERCE_INPUT_PATH
    )

    ecommerce_ground_truth = pd.read_csv(
        ECOMMERCE_GROUND_TRUTH_INPUT_PATH
    )

    expected_ecommerce_columns = [
        "transaction_id",
        "transaction_date_time",
        "guest_transaction",
        "portal_user_id",
    ]

    assert set(
        expected_ecommerce_columns
    ).issubset(
        set(
            ecommerce.columns
        )
    )

    required_ground_truth_columns = [
        "transaction_id",
        "ground_truth_id",
    ]

    assert set(
        required_ground_truth_columns
    ).issubset(
        set(
            ecommerce_ground_truth.columns
        )
    )

    assert ecommerce[
        "transaction_id"
    ].is_unique

    assert ecommerce_ground_truth[
        "transaction_id"
    ].is_unique

    ecommerce[
        "transaction_date_time"
    ] = pd.to_datetime(
        ecommerce[
            "transaction_date_time"
        ]
    )

    ecommerce[
        "guest_transaction"
    ] = normalise_boolean_series(
        ecommerce[
            "guest_transaction"
        ]
    )

    ecommerce = ecommerce.merge(
        ecommerce_ground_truth[
            [
                "transaction_id",
                "ground_truth_id",
            ]
        ],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )

    assert ecommerce[
        "ground_truth_id"
    ].notna().all()

    return (
        canonical,
        portal_mapping,
        ecommerce,
    )


###############################################################################
# 7. Select Online Session customer population
###############################################################################

def select_online_customers(
    canonical: pd.DataFrame,
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

    eligible_customers = (
        customer_population[
            customer_population[
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
        UNIQUE_ONLINE_CUSTOMERS
    )

    selection_weights = (
        eligible_customers[
            "customer_segment"
        ]
        .map(
            ONLINE_SEGMENT_SELECTION_WEIGHTS
        )
        .astype(float)
    )

    selection_weights = (
        selection_weights
        / selection_weights.sum()
    )

    selected_indices = rng.choice(
        eligible_customers.index.to_numpy(),
        size=UNIQUE_ONLINE_CUSTOMERS,
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
# 8. Select E-commerce transactions linked to Online Sessions
###############################################################################

def select_linked_ecommerce_transactions(
    ecommerce: pd.DataFrame,
    selected_customers: pd.DataFrame,
) -> pd.DataFrame:

    selected_ground_truth_ids = set(
        selected_customers[
            "ground_truth_id"
        ]
    )

    eligible_transactions = (
        ecommerce[
            ecommerce[
                "ground_truth_id"
            ].isin(
                selected_ground_truth_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    online_channel_mask = (
        rng.random(
            len(
                eligible_transactions
            )
        )
        <
        ONLINE_CHANNEL_TRANSACTION_RATE
    )

    linked_transactions = (
        eligible_transactions[
            online_channel_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # Online Session Logs must represent only a subset of the
    # complete E-commerce transaction population.
    assert (
        len(
            linked_transactions
        )
        <
        len(
            ecommerce
        )
    )

    assert (
        len(
            linked_transactions
        )
        <
        NUMBER_OF_ONLINE_SESSIONS
    )

    assert linked_transactions[
        "transaction_id"
    ].is_unique

    return linked_transactions


###############################################################################
# 9. Generate Online Session plan
###############################################################################

def generate_session_plan(
    selected_customers: pd.DataFrame,
    linked_transactions: pd.DataFrame,
) -> pd.DataFrame:

    session_plan = []

    ###########################################################################
    # Create one converted session for each selected E-commerce transaction
    ###########################################################################

    for transaction in (
        linked_transactions
        .itertuples(
            index=False
        )
    ):

        session_plan.append(
            {
                "ground_truth_id": transaction.ground_truth_id,
                "basket_created": True,
                "basket_converted": True,
                "linked_transaction_id": transaction.transaction_id,
                "linked_transaction_date_time":
                    transaction.transaction_date_time,
                "linked_guest_transaction": bool(
                    transaction.guest_transaction
                ),
                "linked_portal_user_id": transaction.portal_user_id,
            }
        )

    ###########################################################################
    # Ensure every selected customer is represented at least once
    ###########################################################################

    represented_ground_truth_ids = {
        record[
            "ground_truth_id"
        ]
        for record
        in session_plan
    }

    unrepresented_customers = (
        selected_customers[
            ~selected_customers[
                "ground_truth_id"
            ].isin(
                represented_ground_truth_ids
            )
        ]
    )

    for customer in (
        unrepresented_customers
        .itertuples(
            index=False
        )
    ):

        basket_created = bool(
            rng.random()
            <
            NON_CONVERTED_BASKET_CREATION_RATE
        )

        session_plan.append(
            {
                "ground_truth_id": customer.ground_truth_id,
                "basket_created": basket_created,
                "basket_converted": False,
                "linked_transaction_id": pd.NA,
                "linked_transaction_date_time": pd.NaT,
                "linked_guest_transaction": pd.NA,
                "linked_portal_user_id": pd.NA,
            }
        )

    ###########################################################################
    # Generate remaining repeated sessions using segment weighting
    ###########################################################################

    remaining_session_count = (
        NUMBER_OF_ONLINE_SESSIONS
        - len(
            session_plan
        )
    )

    if remaining_session_count < 0:

        raise ValueError(
            "Converted and mandatory customer sessions exceed "
            "the configured Online Session target."
        )

    repeat_weights = (
        selected_customers[
            "customer_segment"
        ]
        .map(
            SESSION_FREQUENCY_WEIGHTS
        )
        .astype(float)
    )

    repeat_weights = (
        repeat_weights
        / repeat_weights.sum()
    )

    repeated_customer_indices = rng.choice(
        selected_customers.index.to_numpy(),
        size=remaining_session_count,
        replace=True,
        p=repeat_weights.to_numpy(),
    )

    for customer_index in (
        repeated_customer_indices
    ):

        customer = selected_customers.loc[
            customer_index
        ]

        basket_created = bool(
            rng.random()
            <
            NON_CONVERTED_BASKET_CREATION_RATE
        )

        session_plan.append(
            {
                "ground_truth_id": customer[
                    "ground_truth_id"
                ],
                "basket_created": basket_created,
                "basket_converted": False,
                "linked_transaction_id": pd.NA,
                "linked_transaction_date_time": pd.NaT,
                "linked_guest_transaction": pd.NA,
                "linked_portal_user_id": pd.NA,
            }
        )

    session_plan = pd.DataFrame(
        session_plan
    )

    session_plan = session_plan.merge(
        selected_customers[
            [
                "ground_truth_id",
                "customer_segment",
                "registration_date",
                "portal_user_id",
            ]
        ].rename(
            columns={
                "portal_user_id":
                    "available_portal_user_id",
            }
        ),
        on="ground_truth_id",
        how="left",
        validate="many_to_one",
    )

    assert (
        len(
            session_plan
        )
        ==
        NUMBER_OF_ONLINE_SESSIONS
    )

    assert (
        session_plan[
            "ground_truth_id"
        ].nunique()
        ==
        UNIQUE_ONLINE_CUSTOMERS
    )

    return session_plan


###############################################################################
# 10. Assign guest-session and portal-account status
###############################################################################

def assign_session_account_status(
    session_plan: pd.DataFrame,
) -> pd.DataFrame:

    session_plan = (
        session_plan
        .copy()
    )

    session_plan[
        "guest_session"
    ] = False

    session_plan[
        "session_portal_user_id"
    ] = pd.NA

    for index in (
        session_plan.index
    ):

        basket_converted = bool(
            session_plan.at[
                index,
                "basket_converted",
            ]
        )

        available_portal_user_id = (
            session_plan.at[
                index,
                "available_portal_user_id",
            ]
        )

        #######################################################################
        # Registered converted E-commerce transaction
        #######################################################################

        if (
            basket_converted
            and
            not bool(
                session_plan.at[
                    index,
                    "linked_guest_transaction",
                ]
            )
        ):

            linked_portal_user_id = (
                session_plan.at[
                    index,
                    "linked_portal_user_id",
                ]
            )

            assert pd.notna(
                linked_portal_user_id
            )

            assert (
                linked_portal_user_id
                ==
                available_portal_user_id
            )

            guest_session = False
            portal_user_id = (
                linked_portal_user_id
            )

        #######################################################################
        # Guest E-commerce conversion or non-converted Online Session
        #######################################################################

        else:

            # Customers without an existing portal account can only
            # generate guest sessions.
            if pd.isna(
                available_portal_user_id
            ):

                guest_session = True
                portal_user_id = pd.NA

            # Customers with an available portal account have a 20%
            # probability of browsing as a guest.
            else:

                guest_session = bool(
                    rng.random()
                    <
                    PORTAL_HOLDER_GUEST_SESSION_RATE
                )

                if guest_session:
                    portal_user_id = pd.NA

                else:
                    portal_user_id = (
                        available_portal_user_id
                    )

        session_plan.at[
            index,
            "guest_session",
        ] = guest_session

        session_plan.at[
            index,
            "session_portal_user_id",
        ] = portal_user_id

    return session_plan


###############################################################################
# 11. Generate Online Session records
###############################################################################

def generate_online_session_records(
    session_plan: pd.DataFrame,
) -> pd.DataFrame:

    session_records = []

    for session in (
        session_plan
        .itertuples(
            index=False
        )
    ):

        basket_converted = bool(
            session.basket_converted
        )

        #######################################################################
        # Generate session duration
        #######################################################################

        duration_seconds = (
            generate_session_duration(
                customer_segment=(
                    session.customer_segment
                ),
                basket_converted=(
                    basket_converted
                ),
            )
        )

        #######################################################################
        # Generate session timestamps
        #######################################################################

        if basket_converted:

            (
                session_start,
                session_end,
            ) = generate_converted_session_times(
                transaction_date_time=pd.Timestamp(
                    session.linked_transaction_date_time
                ),
                registration_date=pd.Timestamp(
                    session.registration_date
                ),
                duration_seconds=duration_seconds,
            )

        else:

            (
                session_start,
                session_end,
            ) = generate_non_converted_session_times(
                registration_date=pd.Timestamp(
                    session.registration_date
                ),
                duration_seconds=duration_seconds,
            )

        #######################################################################
        # Generate device type
        #######################################################################

        device_type = str(
            rng.choice(
                DEVICE_TYPES,
                p=DEVICE_TYPE_PROBABILITIES,
            )
        )

        #######################################################################
        # Append session record
        #######################################################################

        session_records.append(
            {
                # Internal generation fields
                "ground_truth_id": session.ground_truth_id,
                "customer_segment": session.customer_segment,
                "registration_date": session.registration_date,
                "available_portal_user_id":
                    session.available_portal_user_id,
                "linked_transaction_date_time":
                    session.linked_transaction_date_time,
                "linked_guest_transaction":
                    session.linked_guest_transaction,
                "linked_portal_user_id":
                    session.linked_portal_user_id,

                # Operational Online Session fields
                "session_id": pd.NA,
                "guest_session": bool(
                    session.guest_session
                ),
                "portal_user_id": session.session_portal_user_id,
                "session_date_time_start": session_start,
                "session_date_time_end": session_end,
                "session_duration_seconds": duration_seconds,
                "device_type": device_type,
                "basket_created": bool(
                    session.basket_created
                ),
                "basket_converted": basket_converted,
                "linked_transaction_id":
                    session.linked_transaction_id,
            }
        )

    return pd.DataFrame(
        session_records
    )


###############################################################################
# 12. Finalise session IDs
###############################################################################

def finalise_session_ids(
    session_records: pd.DataFrame,
) -> pd.DataFrame:

    # Sort sessions chronologically before assigning
    # sequential Online Session identifiers.
    session_records = (
        session_records
        .sort_values(
            "session_date_time_start"
        )
        .reset_index(
            drop=True
        )
    )

    session_records[
        "session_id"
    ] = [
        f"SES{session_number:06d}"
        for session_number
        in range(
            1,
            len(
                session_records
            ) + 1,
        )
    ]

    return session_records


###############################################################################
# 13. Validate Online Session dataset
###############################################################################

def validate_online_session_records(
    session_records: pd.DataFrame,
    canonical: pd.DataFrame,
    portal_mapping: pd.DataFrame,
    ecommerce: pd.DataFrame,
) -> pd.DataFrame:

    ###########################################################################
    # Basic validation
    ###########################################################################

    assert (
        len(
            session_records
        )
        ==
        NUMBER_OF_ONLINE_SESSIONS
    )

    assert session_records[
        "session_id"
    ].is_unique

    assert (
        session_records[
            "ground_truth_id"
        ].nunique()
        ==
        UNIQUE_ONLINE_CUSTOMERS
    )

    assert set(
        session_records[
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
    # Guest-session and portal-account validation
    ###########################################################################

    guest_sessions = (
        session_records[
            session_records[
                "guest_session"
            ]
        ]
    )

    logged_in_sessions = (
        session_records[
            ~session_records[
                "guest_session"
            ]
        ]
    )

    assert guest_sessions[
        "portal_user_id"
    ].isna().all()

    assert logged_in_sessions[
        "portal_user_id"
    ].notna().all()

    no_portal_sessions = session_records[
        session_records[
            "available_portal_user_id"
        ].isna()
    ]

    assert no_portal_sessions[
        "guest_session"
    ].all()

    # Every logged-in session must use the portal account belonging
    # to the same hidden canonical customer.
    assert (
        logged_in_sessions[
            "portal_user_id"
        ]
        ==
        logged_in_sessions[
            "available_portal_user_id"
        ]
    ).all()

    ###########################################################################
    # Session timing validation
    ###########################################################################

    session_start = pd.to_datetime(
        session_records[
            "session_date_time_start"
        ]
    )

    session_end = pd.to_datetime(
        session_records[
            "session_date_time_end"
        ]
    )

    registration_dates = pd.to_datetime(
        session_records[
            "registration_date"
        ]
    )

    assert (
        session_start
        <
        session_end
    ).all()

    assert (
        session_start
        >=
        OPERATION_START_DATETIME
    ).all()

    assert (
        session_end
        <=
        SNAPSHOT_END_DATETIME
    ).all()

    assert (
        session_start
        >=
        registration_dates
    ).all()

    calculated_duration = (
        session_end
        - session_start
    ).dt.total_seconds().astype(int)

    assert (
        calculated_duration
        ==
        session_records[
            "session_duration_seconds"
        ]
    ).all()

    ###########################################################################
    # Device validation
    ###########################################################################

    assert set(
        session_records[
            "device_type"
        ]
    ).issubset(
        set(
            DEVICE_TYPES
        )
    )

    ###########################################################################
    # Basket and conversion validation
    ###########################################################################

    converted_sessions = (
        session_records[
            session_records[
                "basket_converted"
            ]
        ]
        .copy()
    )

    non_converted_sessions = (
        session_records[
            ~session_records[
                "basket_converted"
            ]
        ]
        .copy()
    )

    assert converted_sessions[
        "basket_created"
    ].all()

    assert converted_sessions[
        "linked_transaction_id"
    ].notna().all()

    assert non_converted_sessions[
        "linked_transaction_id"
    ].isna().all()

    assert converted_sessions[
        "linked_transaction_id"
    ].is_unique

    assert set(
        converted_sessions[
            "linked_transaction_id"
        ]
    ).issubset(
        set(
            ecommerce[
                "transaction_id"
            ]
        )
    )

    assert (
        len(
            converted_sessions
        )
        <
        len(
            ecommerce
        )
    )

    ###########################################################################
    # Linked E-commerce transaction validation
    ###########################################################################

    ecommerce_lookup = (
        ecommerce[
            [
                "transaction_id",
                "transaction_date_time",
                "guest_transaction",
                "portal_user_id",
                "ground_truth_id",
            ]
        ]
        .rename(
            columns={
                "transaction_date_time":
                    "ecommerce_transaction_date_time",
                "guest_transaction":
                    "ecommerce_guest_transaction",
                "portal_user_id":
                    "ecommerce_portal_user_id",
                "ground_truth_id":
                    "ecommerce_ground_truth_id",
            }
        )
    )

    converted_validation = (
        converted_sessions
        .merge(
            ecommerce_lookup,
            left_on="linked_transaction_id",
            right_on="transaction_id",
            how="left",
            validate="one_to_one",
        )
    )

    assert converted_validation[
        "transaction_id"
    ].notna().all()

    assert (
        converted_validation[
            "ground_truth_id"
        ]
        ==
        converted_validation[
            "ecommerce_ground_truth_id"
        ]
    ).all()

    converted_start = pd.to_datetime(
        converted_validation[
            "session_date_time_start"
        ]
    )

    converted_end = pd.to_datetime(
        converted_validation[
            "session_date_time_end"
        ]
    )

    converted_transaction_time = pd.to_datetime(
        converted_validation[
            "ecommerce_transaction_date_time"
        ]
    )

    assert (
        converted_start
        <=
        converted_transaction_time
    ).all()

    assert (
        converted_transaction_time
        <=
        converted_end
    ).all()

    # Registered E-commerce conversions must use the exact same
    # portal_user_id within the linked Online Session.
    registered_converted = (
        converted_validation[
            ~converted_validation[
                "ecommerce_guest_transaction"
            ]
        ]
    )

    assert (
        ~registered_converted[
            "guest_session"
        ]
    ).all()

    assert (
        registered_converted[
            "portal_user_id"
        ]
        ==
        registered_converted[
            "ecommerce_portal_user_id"
        ]
    ).all()

    ###########################################################################
    # Validation metrics
    ###########################################################################

    total_sessions = len(
        session_records
    )

    unique_customers = (
        session_records[
            "ground_truth_id"
        ].nunique()
    )

    guest_session_rate = (
        session_records[
            "guest_session"
        ].mean()
    )

    portal_holder_sessions = session_records[
        session_records[
            "available_portal_user_id"
        ].notna()
    ]

    portal_holder_guest_rate = (
        portal_holder_sessions[
            "guest_session"
        ].mean()
    )

    mobile_session_rate = (
        session_records[
            "device_type"
        ]
        .eq("Mobile")
        .mean()
    )

    basket_created_rate = (
        session_records[
            "basket_created"
        ].mean()
    )

    basket_conversion_rate = (
        session_records[
            "basket_converted"
        ].mean()
    )

    converted_session_count = len(
        converted_sessions
    )

    ecommerce_transaction_count = len(
        ecommerce
    )

    # Identify E-commerce transactions belonging to customers
    # represented within the Online Session dataset.
    online_customer_ids = set(
        session_records[
            "ground_truth_id"
        ]
    )

    eligible_ecommerce_records = ecommerce[
        ecommerce[
            "ground_truth_id"
        ].isin(
            online_customer_ids
        )
    ]

    eligible_ecommerce_transactions = len(
        eligible_ecommerce_records
    )

    online_channel_rate_among_eligible = (
        converted_session_count
        /
        eligible_ecommerce_transactions
    )

    ecommerce_online_coverage = (
        converted_session_count
        /
        ecommerce_transaction_count
    )

    ###########################################################################
    # Customer coverage by segment
    ###########################################################################

    unique_online_customers = (
        session_records[
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

    online_segment_counts = (
        unique_online_customers[
            "customer_segment"
        ]
        .value_counts()
    )

    segment_coverage = (
        online_segment_counts
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
    # Session frequency by segment
    ###########################################################################

    customer_session_counts = (
        session_records
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
                "size": "session_count",
            }
        )
    )

    average_sessions_by_segment = (
        customer_session_counts
        .groupby(
            "customer_segment"
        )[
            "session_count"
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
    # Session duration by segment
    ###########################################################################

    average_duration_by_segment = (
        session_records
        .groupby(
            "customer_segment"
        )[
            "session_duration_seconds"
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

    converted_average_duration = (
        converted_sessions[
            "session_duration_seconds"
        ].mean()
    )

    non_converted_average_duration = (
        non_converted_sessions[
            "session_duration_seconds"
        ].mean()
    )

    ###########################################################################
    # Visualise validation results
    ###########################################################################

    print(
        "\nOnline Session validation completed successfully."
    )

    print(
        f"Total sessions: "
        f"{total_sessions:,}"
    )

    print(
        f"Unique customers represented: "
        f"{unique_customers:,}"
    )

    print(
        f"Guest sessions: "
        f"{guest_session_rate:.2%}"
    )

    print(
        f"Guest sessions among portal account holders: "
        f"{portal_holder_guest_rate:.2%}"
    )

    print(
        f"Mobile sessions: "
        f"{mobile_session_rate:.2%}"
    )

    print(
        f"Basket creation rate: "
        f"{basket_created_rate:.2%}"
    )

    print(
        f"Basket conversion rate: "
        f"{basket_conversion_rate:.2%}"
    )

    print(
        f"Converted sessions linked to E-commerce: "
        f"{converted_session_count:,}"
    )

    print(
    f"Eligible E-commerce transactions: "
    f"{eligible_ecommerce_transactions:,}"
    )

    print(
        f"Online channel rate among eligible transactions: "
        f"{online_channel_rate_among_eligible:.2%}"
    )

    print(
        f"E-commerce transactions represented online: "
        f"{ecommerce_online_coverage:.2%}"
    )

    print(
        "\nOnline customer coverage by segment:"
    )

    print(
        segment_coverage
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    print(
        "\nAverage sessions per customer by segment:"
    )

    print(
        average_sessions_by_segment
        .round(2)
    )

    print(
        "\nAverage session duration by segment:"
    )

    print(
        average_duration_by_segment
        .round(2)
    )

    print(
        f"\nAverage converted session duration: "
        f"{converted_average_duration:.2f} seconds"
    )

    print(
        f"Average non-converted session duration: "
        f"{non_converted_average_duration:.2f} seconds"
    )

    ###########################################################################
    # Create validation summary output
    ###########################################################################

    validation_summary = pd.DataFrame(
        {
            "metric": [
                "MAIN VALIDATION METRICS",
                "total_sessions",
                "unique_customers",
                "guest_session_rate",
                "portal_holder_guest_session_rate",
                "mobile_session_rate",
                "basket_creation_rate",
                "basket_conversion_rate",
                "converted_sessions",
                "ecommerce_transactions",
                "eligible_ecommerce_transactions",
                "online_channel_rate_among_eligible_transactions",
                "ecommerce_online_transaction_coverage",
                "linked_transaction_ids_unique",
                "linked_transactions_valid",
                "registered_conversion_portal_match",
                "session_timing_valid",
                "session_duration_reconciliation",
            ],
            "value": [
                "",
                total_sessions,
                unique_customers,
                round(
                    guest_session_rate,
                    4,
                ),
                round(
                    portal_holder_guest_rate,
                    4,
                ),
                round(
                    mobile_session_rate,
                    4,
                ),
                round(
                    basket_created_rate,
                    4,
                ),
                round(
                    basket_conversion_rate,
                    4,
                ),
                converted_session_count,
                ecommerce_transaction_count,
                eligible_ecommerce_transactions,
                round(
                    online_channel_rate_among_eligible,
                    4,
                ),
                round(
                    ecommerce_online_coverage,
                    4,
                ),
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
                "ONLINE CUSTOMER COVERAGE BY SEGMENT",
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
    # Average session frequency by segment
    ###########################################################################

    frequency_summary = pd.DataFrame(
        {
            "metric": [
                "",
                "AVERAGE SESSIONS PER CUSTOMER BY SEGMENT",
                "average_sessions_new",
                "average_sessions_occasional",
                "average_sessions_regular",
                "average_sessions_vip",
            ],
            "value": [
                "",
                "",
                round(
                    average_sessions_by_segment[
                        "New"
                    ],
                    2,
                ),
                round(
                    average_sessions_by_segment[
                        "Occasional"
                    ],
                    2,
                ),
                round(
                    average_sessions_by_segment[
                        "Regular"
                    ],
                    2,
                ),
                round(
                    average_sessions_by_segment[
                        "VIP"
                    ],
                    2,
                ),
            ],
        }
    )

    ###########################################################################
    # Average session duration by segment
    ###########################################################################

    duration_summary = pd.DataFrame(
        {
            "metric": [
                "",
                "AVERAGE SESSION DURATION BY SEGMENT",
                "average_duration_seconds_new",
                "average_duration_seconds_occasional",
                "average_duration_seconds_regular",
                "average_duration_seconds_vip",
                "average_duration_seconds_converted",
                "average_duration_seconds_non_converted",
            ],
            "value": [
                "",
                "",
                round(
                    average_duration_by_segment[
                        "New"
                    ],
                    2,
                ),
                round(
                    average_duration_by_segment[
                        "Occasional"
                    ],
                    2,
                ),
                round(
                    average_duration_by_segment[
                        "Regular"
                    ],
                    2,
                ),
                round(
                    average_duration_by_segment[
                        "VIP"
                    ],
                    2,
                ),
                round(
                    converted_average_duration,
                    2,
                ),
                round(
                    non_converted_average_duration,
                    2,
                ),
            ],
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
            duration_summary,
        ],
        ignore_index=True,
    )

    return validation_summary


###############################################################################
# 14. Export Online Session outputs
###############################################################################

def export_online_session_outputs(
    session_records: pd.DataFrame,
    validation_summary: pd.DataFrame,
) -> None:

    # Create output folders
    ONLINE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    GROUND_TRUTH_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ###########################################################################
    # Operational Online Session dataset
    ###########################################################################

    operational_columns = [
        "session_id",
        "guest_session",
        "portal_user_id",
        "session_date_time_start",
        "session_date_time_end",
        "session_duration_seconds",
        "device_type",
        "basket_created",
        "basket_converted",
        "linked_transaction_id",
    ]

    operational_online_sessions = (
        session_records[
            operational_columns
        ]
        .copy()
    )

    operational_online_sessions.to_csv(
        ONLINE_OUTPUT_PATH,
        index=False,
    )

    ###########################################################################
    # Hidden Online Session ground truth mapping
    ###########################################################################

    ground_truth_columns = [
        "session_id",
        "ground_truth_id",
        "customer_segment",
        "guest_session",
        "portal_user_id",
        "available_portal_user_id",
        "basket_created",
        "basket_converted",
        "linked_transaction_id",
    ]

    online_ground_truth_mapping = (
        session_records[
            ground_truth_columns
        ]
        .copy()
    )

    online_ground_truth_mapping.to_csv(
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

    # Load source and reference datasets
    (
        canonical,
        portal_mapping,
        ecommerce,
    ) = load_source_datasets()

    # Select approximately 8,000 unique customers
    selected_customers = (
        select_online_customers(
            canonical=canonical,
            portal_mapping=portal_mapping,
        )
    )

    # Select a subset of eligible E-commerce transactions
    # assumed to originate from the owned website/app.
    linked_transactions = (
        select_linked_ecommerce_transactions(
            ecommerce=ecommerce,
            selected_customers=selected_customers,
        )
    )

    # Generate exactly 40,000 session plans while ensuring
    # every selected customer is represented at least once.
    session_plan = (
        generate_session_plan(
            selected_customers=selected_customers,
            linked_transactions=linked_transactions,
        )
    )

    # Assign guest-session and portal-account status
    session_plan = (
        assign_session_account_status(
            session_plan=session_plan
        )
    )

    # Generate operational Online Session records
    session_records = (
        generate_online_session_records(
            session_plan=session_plan
        )
    )

    # Sort chronologically and assign session IDs
    session_records = (
        finalise_session_ids(
            session_records
        )
    )

    # Validate final Online Session environment
    validation_summary = (
        validate_online_session_records(
            session_records=session_records,
            canonical=canonical,
            portal_mapping=portal_mapping,
            ecommerce=ecommerce,
        )
    )

    # Export operational and reference datasets
    export_online_session_outputs(
        session_records=session_records,
        validation_summary=validation_summary,
    )

    print(
        f"\nOnline Session dataset saved to: "
        f"{ONLINE_OUTPUT_PATH.resolve()}"
    )

    print(
        f"Online Session ground truth saved to: "
        f"{GROUND_TRUTH_OUTPUT_PATH.resolve()}"
    )

    print(
        f"Online Session validation summary saved to: "
        f"{VALIDATION_OUTPUT_PATH.resolve()}"
    )

    print(
        "\nFirst five Online Session records:"
    )

    print(
        session_records[
            [
                "session_id",
                "guest_session",
                "portal_user_id",
                "session_date_time_start",
                "session_date_time_end",
                "session_duration_seconds",
                "device_type",
                "basket_created",
                "basket_converted",
                "linked_transaction_id",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()
