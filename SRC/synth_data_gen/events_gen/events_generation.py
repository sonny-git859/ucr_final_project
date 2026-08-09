###############################################################################
# Imports
###############################################################################

from pathlib import Path

import numpy as np
import pandas as pd


###############################################################################
# 1. Generation configuration
###############################################################################

# Seed ensures reproducibility
SEED = 42

# Events occur throughout 12-month snapshot
ACTIVITY_START_DATE = pd.Timestamp("2025-01-01")
# End date consistent with SNAPSHOT date in canonical script
ACTIVITY_END_DATE = pd.Timestamp("2025-12-31")

# Output path
OUTPUT_PATH = Path("data/events/events.csv")


###############################################################################
# 2. Assumptions
###############################################################################

# Venue capacity
STANDING_CAPACITY = 500
SEATED_CAPACITY = 200

# Ticket price ranges
MIN_STANDING_PRICE = 20.00
MAX_STANDING_PRICE = 70.00

# Seated tickets assigned random premium
MIN_SEATED_PREMIUM = 5.00
MAX_SEATED_PREMIUM = 30.00

# Event categories
EVENT_CATEGORIES = [
    "Music",
    "Comedy",
    "Theatre",
    "Family",
    "Sport",
]

# Equal category probabilities used as an initial assumption
EVENT_CATEGORY_PROBABILITIES = [
    0.20,
    0.20,
    0.20,
    0.20,
    0.20,
]

# Category-based word pools for event name generation - sourced from LLM
EVENT_NAME_PARTS = {
    "Music": {
        "prefixes": [
            "Electric",
            "Midnight",
            "Neon",
            "Golden",
            "Northern",
            "Velvet",
            "Crimson",
            "Silver",
            "Echoing",
            "Wild",
        ],
        "suffixes": [
            "Echoes",
            "Horizons",
            "Frequency",
            "Revival",
            "Signals",
            "Parade",
            "Static",
            "Anthems",
            "Lights",
            "Sound",
        ],
    },
    "Comedy": {
        "prefixes": [
            "Saturday",
            "Big",
            "Late Night",
            "Unfiltered",
            "Stand-Up",
            "Laughing",
            "Comic",
            "Live",
            "Crowd",
            "Punchline",
        ],
        "suffixes": [
            "Laughs",
            "Comedy Club",
            "Joke Factory",
            "Showcase",
            "Sessions",
            "Circuit",
            "Hour",
            "Revue",
            "Night",
            "Special",
        ],
    },
    "Theatre": {
        "prefixes": [
            "The Last",
            "The Hidden",
            "The Glass",
            "The Silent",
            "The Midnight",
            "The Forgotten",
            "The Golden",
            "The Broken",
            "The Secret",
            "The Final",
        ],
        "suffixes": [
            "Lantern",
            "Kingdom",
            "Voyage",
            "Garden",
            "Letter",
            "Promise",
            "Portrait",
            "Door",
            "Journey",
            "Encore",
        ],
    },
    "Family": {
        "prefixes": [
            "Magical",
            "Great",
            "Enchanted",
            "Curious",
            "Wonderful",
            "Amazing",
            "Colourful",
            "Secret",
            "Winter",
            "Summer",
        ],
        "suffixes": [
            "Adventure",
            "Circus",
            "Kingdom",
            "Quest",
            "Carnival",
            "Workshop",
            "Spectacular",
            "Story",
            "Festival",
            "Discovery",
        ],
    },
    "Sport": {
        "prefixes": [
            "Ultimate",
            "National",
            "Indoor",
            "Saturday",
            "Championship",
            "Rising",
            "Arena",
            "Elite",
            "Premier",
            "City",
        ],
        "suffixes": [
            "Fight Night",
            "Wrestling Live",
            "Boxing Showcase",
            "Darts Open",
            "Futsal Cup",
            "Gymnastics Gala",
            "Martial Arts",
            "Sports Festival",
            "Challenge",
            "Finals",
        ],
    },
}


###############################################################################
# 3. Initialise generator
###############################################################################

rng = np.random.default_rng(SEED)


###############################################################################
# 4. Helper functions
###############################################################################

# Generate unique fictional name based on event category
def generate_event_name(
    event_category: str,
    used_event_names: set[str],
) -> str:

    # Event name based on category
    name_parts = EVENT_NAME_PARTS[event_category]

    # Assign event name prefix
    prefix = str(
        rng.choice(name_parts["prefixes"])
    )

    # Assign event name suffix
    suffix = str(
        rng.choice(name_parts["suffixes"])
    )

    base_event_name = f"{prefix} {suffix}"
    event_name = base_event_name
    duplicate_number = 2

    # Add number to event name to ensure unique names
    while event_name in used_event_names:
        event_name = f"{base_event_name} {duplicate_number}"
        duplicate_number += 1

    used_event_names.add(event_name)

    return event_name


# Generate random standing and seated attendance within venue capacity
def generate_event_attendance() -> tuple[int, int, int]:

    standing_attendance = int(
        rng.integers(
            0,
            STANDING_CAPACITY + 1,
        )
    )

    seated_attendance = int(
        rng.integers(
            0,
            SEATED_CAPACITY + 1,
        )
    )

    total_attendance = (
        standing_attendance
        + seated_attendance
    )

    return (
        standing_attendance,
        seated_attendance,
        total_attendance,
    )


# Generate random standing and seated ticket price
def generate_ticket_prices() -> tuple[float, float]:

    standing_price = round(
        float(
            rng.uniform(
                MIN_STANDING_PRICE,
                MAX_STANDING_PRICE,
            )
        ),
        2,
    )

    seated_premium = round(
        float(
            rng.uniform(
                MIN_SEATED_PREMIUM,
                MAX_SEATED_PREMIUM,
            )
        ),
        2,
    )

    seated_price = round(
        standing_price + seated_premium,
        2,
    )

    return standing_price, seated_price


###############################################################################
# 5. Generate events dataset
###############################################################################

def generate_events() -> pd.DataFrame:

    event_records = []
    used_event_names: set[str] = set()

    # Generate 1 event for every Saturday 12-month snapshot
    event_dates = pd.date_range(
        start=ACTIVITY_START_DATE,
        end=ACTIVITY_END_DATE,
        freq="W-SAT",
    )

    # Generate event fields
    for event_number, event_date in enumerate(
        event_dates,
        start=1,
    ):

        # Generate unique event ID
        event_id = f"EVT{event_number:04d}"

        # Assign event category
        event_category = str(
            rng.choice(
                EVENT_CATEGORIES,
                p=EVENT_CATEGORY_PROBABILITIES,
            )
        )

        # Generate category-based event name
        event_name = generate_event_name(
            event_category=event_category,
            used_event_names=used_event_names,
        )

        # Generate attendance
        (
            standing_attendance,
            seated_attendance,
            total_attendance,
        ) = generate_event_attendance()

        # Generate ticket price
        (
            standing_price,
            seated_price,
        ) = generate_ticket_prices()

        event_records.append(
            {
                "event_id": event_id,
                "event_name": event_name,
                "event_category": event_category,
                "event_date": event_date.date(),
                "event_standing_attendance": standing_attendance,
                "event_seated_attendance": seated_attendance,
                "event_total_attendance": total_attendance,
                "ticket_price_standing": standing_price,
                "ticket_price_seated": seated_price,
            }
        )

    return pd.DataFrame(event_records)


###############################################################################
# 6. Validate and visualise
###############################################################################

def validate_events(
    events: pd.DataFrame,
) -> None:

    expected_columns = [
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

    expected_event_dates = pd.date_range(
        start=ACTIVITY_START_DATE,
        end=ACTIVITY_END_DATE,
        freq="W-SAT",
    )

    # Validate structure and count of events
    assert list(events.columns) == expected_columns
    assert len(events) == len(expected_event_dates)

    # Confrim unique identifiers and names
    assert events["event_id"].is_unique
    assert events["event_name"].is_unique

    # Confirm valid categories
    assert set(events["event_category"]).issubset(
        set(EVENT_CATEGORIES)
    )

    # Confirm probability configuration is valid
    assert len(EVENT_CATEGORIES) == len(
        EVENT_CATEGORY_PROBABILITIES
    )

    assert np.isclose(
        np.sum(EVENT_CATEGORY_PROBABILITIES),
        1.0,
    )

    # Confirm no missing values
    assert events.isna().sum().sum() == 0

    # Convert event date for validation
    event_dates = pd.to_datetime(
        events["event_date"]
    )

    # Confirm all events occur within the activity period
    assert event_dates.between(
        ACTIVITY_START_DATE,
        ACTIVITY_END_DATE,
    ).all()

    # Monday is 0, meaning Saturday is 5
    assert (event_dates.dt.dayofweek == 5).all()

    # Confirm attendance remains within venue capacity
    assert events[
        "event_standing_attendance"
    ].between(
        0,
        STANDING_CAPACITY,
    ).all()

    assert events[
        "event_seated_attendance"
    ].between(
        0,
        SEATED_CAPACITY,
    ).all()

    # Confirm total attendance has been calculated correctly
    calculated_total_attendance = (
        events["event_standing_attendance"]
        + events["event_seated_attendance"]
    )

    assert (
        events["event_total_attendance"]
        == calculated_total_attendance
    ).all()

    # Confirm ticket prices are positive
    assert (
        events["ticket_price_standing"] > 0
    ).all()

    assert (
        events["ticket_price_seated"] > 0
    ).all()

    # Confirm seated prices include a premium
    assert (
        events["ticket_price_seated"]
        >= events["ticket_price_standing"]
    ).all()

    print("\nEvent dataset validation completed successfully.")
    print(f"Number of events: {len(events):,}")

    print("\nEvent date range:")
    print(
        events["event_date"].min(),
        "to",
        events["event_date"].max(),
    )

    print("\nEvent category distribution:")
    print(
        events["event_category"]
        .value_counts()
        .sort_index()
    )

    print("\nAttendance summary:")
    print(
        events[
            [
                "event_standing_attendance",
                "event_seated_attendance",
                "event_total_attendance",
            ]
        ]
        .describe()
        .round(2)
    )

    print("\nTicket price summary:")
    print(
        events[
            [
                "ticket_price_standing",
                "ticket_price_seated",
            ]
        ]
        .describe()
        .round(2)
    )


###############################################################################
# 7. Main function
###############################################################################

def main() -> None:

    # Generate
    events = generate_events()

    # Validate
    validate_events(events)

    # Create output path
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Export DataFrame as CSV
    events.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nDataset saved to: {OUTPUT_PATH.resolve()}")

    print("\nFirst five events:")
    print(events.head())


if __name__ == "__main__":
    main()
