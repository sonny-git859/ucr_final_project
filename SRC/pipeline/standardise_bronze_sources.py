###############################################################################
# Imports
###############################################################################

import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd


###############################################################################
# 1. File paths and source configuration
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INGESTED_DATA_DIR = PROJECT_ROOT / "data" / "ingested_bronze"
STANDARDISED_DATA_DIR = PROJECT_ROOT / "data" / "standardised_silver"

COMMON_IDENTIFIER_COLUMNS = [
    "crm_customer_id",
    "transaction_id",
    "marketing_contact_id",
    "session_id",
    "support_ticket_id",
    "portal_user_id",
    "linked_transaction_id",
]

RAW_IDENTITY_COLUMNS = [
    "first_name_raw",
    "surname_raw",
    "full_name_raw",
    "email_raw",
    "phone_raw",
    "address_raw",
    "postcode_raw",
    "date_of_birth_raw",
]

NORMALISED_IDENTITY_COLUMNS = [
    "first_name_normalised",
    "surname_normalised",
    "full_name_normalised",
    "email_normalised",
    "phone_normalised",
    "address_normalised",
    "postcode_normalised",
    "date_of_birth_normalised",
]

SOURCE_CONFIG = {
    "crm": {
        "input_filename": "crm_customer_records_bronze.csv",
        "output_filename": "crm_customer_records_silver.csv",
        "source_system": "CRM",
        "required_columns": [
            "source_system",
            "source_record_id",
            "crm_customer_id",
            "portal_user_id",
            "first_name",
            "surname",
            "date_of_birth",
            "email",
            "telephone_number",
            "address",
            "postcode",
        ],
        "identity_columns": {
            "first_name": "first_name",
            "surname": "surname",
            "full_name": None,
            "email": "email",
            "phone": "telephone_number",
            "address": "address",
            "postcode": "postcode",
            "date_of_birth": "date_of_birth",
        },
    },
    "ecommerce": {
        "input_filename": "ecommerce_transactions_bronze.csv",
        "output_filename": "ecommerce_transactions_silver.csv",
        "source_system": "ECOMMERCE",
        "required_columns": [
            "source_system",
            "source_record_id",
            "transaction_id",
            "portal_user_id",
            "purchaser_first_name",
            "purchaser_surname",
            "email",
            "telephone_number",
            "billing_address",
            "billing_postcode",
        ],
        "identity_columns": {
            "first_name": "purchaser_first_name",
            "surname": "purchaser_surname",
            "full_name": None,
            "email": "email",
            "phone": "telephone_number",
            "address": "billing_address",
            "postcode": "billing_postcode",
            "date_of_birth": None,
        },
    },
    "marketing": {
        "input_filename": "marketing_contact_lists_bronze.csv",
        "output_filename": "marketing_contact_lists_silver.csv",
        "source_system": "MARKETING",
        "required_columns": [
            "source_system",
            "source_record_id",
            "marketing_contact_id",
            "contact_name",
            "email",
            "postcode",
        ],
        "identity_columns": {
            "first_name": None,
            "surname": None,
            "full_name": "contact_name",
            "email": "email",
            "phone": None,
            "address": None,
            "postcode": "postcode",
            "date_of_birth": None,
        },
    },
    "online": {
        "input_filename": "online_sessions_bronze.csv",
        "output_filename": "online_sessions_silver.csv",
        "source_system": "ONLINE",
        "required_columns": [
            "source_system",
            "source_record_id",
            "session_id",
            "portal_user_id",
            "linked_transaction_id",
        ],
        "identity_columns": {
            "first_name": None,
            "surname": None,
            "full_name": None,
            "email": None,
            "phone": None,
            "address": None,
            "postcode": None,
            "date_of_birth": None,
        },
    },
    "support": {
        "input_filename": "support_ticket_logs_bronze.csv",
        "output_filename": "support_ticket_logs_silver.csv",
        "source_system": "SUPPORT",
        "required_columns": [
            "source_system",
            "source_record_id",
            "support_ticket_id",
            "requester_name",
            "requester_email",
            "requester_phone",
        ],
        "identity_columns": {
            "first_name": None,
            "surname": None,
            "full_name": "requester_name",
            "email": "requester_email",
            "phone": "requester_phone",
            "address": None,
            "postcode": None,
            "date_of_birth": None,
        },
    },
}


###############################################################################
# 2. Validation functions
###############################################################################

def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    filename: str,
) -> None:
    # Confirm that all structurally required columns are present.

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        missing_values = ", ".join(missing_columns)
        raise ValueError(
            f"{filename} is missing required columns: {missing_values}."
        )


def validate_source_metadata(
    dataframe: pd.DataFrame,
    source_system: str,
    filename: str,
) -> None:
    # Confirm that source metadata remains populated and consistent.

    source_values = dataframe["source_system"]
    invalid_sources = source_values.ne(source_system)

    if invalid_sources.any():
        raise ValueError(
            f"{filename} contains inconsistent source_system values."
        )

    source_record_ids = dataframe["source_record_id"]
    missing_ids = source_record_ids.str.strip().eq("")

    if missing_ids.any():
        missing_count = int(missing_ids.sum())
        raise ValueError(
            f"{filename} contains {missing_count:,} missing "
            "source_record_id value(s)."
        )

    duplicate_ids = source_record_ids.duplicated(keep=False)

    if duplicate_ids.any():
        duplicate_count = int(duplicate_ids.sum())
        raise ValueError(
            f"{filename} contains {duplicate_count:,} records with "
            "duplicate source_record_id values."
        )


def validate_source_columns_preserved(
    source_dataframe: pd.DataFrame,
    standardised_dataframe: pd.DataFrame,
    filename: str,
) -> None:
    # Confirm that standardisation preserves every Bronze source column.

    missing_columns = [
        column
        for column in source_dataframe.columns
        if column not in standardised_dataframe.columns
    ]

    if missing_columns:
        missing_values = ", ".join(missing_columns)
        raise ValueError(
            f"{filename} lost source columns during standardisation: "
            f"{missing_values}."
        )


###############################################################################
# 3. Normalisation functions
###############################################################################

def convert_to_ascii(value: str) -> str:
    # Convert accented characters while retaining their base characters.

    normalised_value = unicodedata.normalize("NFKD", value)
    return normalised_value.encode("ascii", "ignore").decode("ascii")


def normalise_name(value: str) -> str:
    # Create a lowercase alphanumeric name for identity comparison.

    if not value.strip():
        return ""

    value = convert_to_ascii(value)
    value = value.lower()

    return re.sub(r"[^a-z0-9]", "", value)


def normalise_email(value: str) -> str:
    # Lowercase email values and remove surrounding or embedded whitespace.

    if not value.strip():
        return ""

    value = convert_to_ascii(value)
    value = value.lower()

    return re.sub(r"\s+", "", value)


def normalise_phone(value: str) -> str:
    # Convert UK telephone values to a digits-only national format.

    if not value.strip():
        return ""

    digits = re.sub(r"\D", "", value)

    if digits.startswith("0044"):
        digits = digits[4:]
    elif digits.startswith("44"):
        digits = digits[2:]

    if len(digits) == 10:
        digits = f"0{digits}"

    return digits


def normalise_address(value: str) -> str:
    # Lowercase addresses and standardise punctuation and whitespace.

    if not value.strip():
        return ""

    value = convert_to_ascii(value)
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return " ".join(value.split())


def normalise_postcode(value: str) -> str:
    # Convert postcodes to an uppercase format without spaces.

    if not value.strip():
        return ""

    value = convert_to_ascii(value)
    value = value.upper()

    return re.sub(r"[^A-Z0-9]", "", value)


def normalise_date_of_birth(value: str) -> str:
    # Convert recognised date formats to the ISO date format.

    if not value.strip():
        return ""

    date_formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
    ]

    for date_format in date_formats:
        try:
            parsed_date = datetime.strptime(value.strip(), date_format)
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return ""


###############################################################################
# 4. Standardisation functions
###############################################################################

def get_source_values(
    dataframe: pd.DataFrame,
    source_column: str | None,
) -> pd.Series:
    # Return source values or an empty series where the field is unavailable.

    if source_column is None:
        return pd.Series("", index=dataframe.index, dtype=str)

    return dataframe[source_column].copy()


def combine_name_values(
    first_names: pd.Series,
    surnames: pd.Series,
) -> pd.Series:
    # Combine separate source name fields without altering their values.

    combined_names = (
        first_names.str.strip() + " " + surnames.str.strip()
    )

    return combined_names.str.strip()


def add_common_identifiers(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    # Add empty common identifier fields that are absent from a source.

    standardised_dataframe = dataframe.copy()

    for column in COMMON_IDENTIFIER_COLUMNS:
        if column not in standardised_dataframe.columns:
            standardised_dataframe[column] = ""

    return standardised_dataframe


def add_identity_columns(
    dataframe: pd.DataFrame,
    identity_columns: dict[str, str | None],
) -> pd.DataFrame:
    # Map source identity fields and create normalised comparison values.

    standardised_dataframe = dataframe.copy()

    first_names = get_source_values(
        dataframe,
        identity_columns["first_name"],
    )
    surnames = get_source_values(
        dataframe,
        identity_columns["surname"],
    )
    full_names = get_source_values(
        dataframe,
        identity_columns["full_name"],
    )

    if identity_columns["full_name"] is None:
        full_names = combine_name_values(first_names, surnames)

    standardised_dataframe["first_name_raw"] = first_names
    standardised_dataframe["surname_raw"] = surnames
    standardised_dataframe["full_name_raw"] = full_names
    standardised_dataframe["email_raw"] = get_source_values(
        dataframe,
        identity_columns["email"],
    )
    standardised_dataframe["phone_raw"] = get_source_values(
        dataframe,
        identity_columns["phone"],
    )
    standardised_dataframe["address_raw"] = get_source_values(
        dataframe,
        identity_columns["address"],
    )
    standardised_dataframe["postcode_raw"] = get_source_values(
        dataframe,
        identity_columns["postcode"],
    )
    standardised_dataframe["date_of_birth_raw"] = get_source_values(
        dataframe,
        identity_columns["date_of_birth"],
    )

    standardised_dataframe["first_name_normalised"] = (
        standardised_dataframe["first_name_raw"].map(normalise_name)
    )
    standardised_dataframe["surname_normalised"] = (
        standardised_dataframe["surname_raw"].map(normalise_name)
    )
    standardised_dataframe["full_name_normalised"] = (
        standardised_dataframe["full_name_raw"].map(normalise_name)
    )
    standardised_dataframe["email_normalised"] = (
        standardised_dataframe["email_raw"].map(normalise_email)
    )
    standardised_dataframe["phone_normalised"] = (
        standardised_dataframe["phone_raw"].map(normalise_phone)
    )
    standardised_dataframe["address_normalised"] = (
        standardised_dataframe["address_raw"].map(normalise_address)
    )
    standardised_dataframe["postcode_normalised"] = (
        standardised_dataframe["postcode_raw"].map(normalise_postcode)
    )
    standardised_dataframe["date_of_birth_normalised"] = (
        standardised_dataframe["date_of_birth_raw"].map(
            normalise_date_of_birth
        )
    )

    return standardised_dataframe


def order_standardised_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    # Place common identity fields before retained source-specific columns.

    common_columns = (
        ["source_system", "source_record_id"]
        + COMMON_IDENTIFIER_COLUMNS
        + RAW_IDENTITY_COLUMNS
        + NORMALISED_IDENTITY_COLUMNS
    )
    remaining_columns = [
        column
        for column in dataframe.columns
        if column not in common_columns
    ]

    return dataframe[common_columns + remaining_columns]


def standardise_source(
    ingested_data_dir: Path,
    source_config: dict,
) -> pd.DataFrame:
    # Read and standardise one Bronze source dataset.

    filename = source_config["input_filename"]
    input_path = ingested_data_dir / filename

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Required Bronze source file was not found: {input_path}"
        )

    try:
        dataframe = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        raise ValueError(
            f"Unable to read Bronze source file {filename}: {error}"
        ) from error

    if dataframe.empty:
        raise ValueError(f"{filename} contains no records.")

    validate_required_columns(
        dataframe,
        source_config["required_columns"],
        filename,
    )
    validate_source_metadata(
        dataframe,
        source_config["source_system"],
        filename,
    )

    standardised_dataframe = add_common_identifiers(dataframe)
    standardised_dataframe = add_identity_columns(
        standardised_dataframe,
        source_config["identity_columns"],
    )

    standardised_dataframe = order_standardised_columns(
        standardised_dataframe
    )

    validate_source_columns_preserved(
        dataframe,
        standardised_dataframe,
        filename,
    )

    return order_standardised_columns(standardised_dataframe)


def standardise_all_sources(
    ingested_data_dir: Path = INGESTED_DATA_DIR,
    standardised_data_dir: Path = STANDARDISED_DATA_DIR,
) -> dict[str, int]:
    # Standardise all Bronze sources and write separate Silver CSV files.

    standardised_sources = {}

    print("Standardising Bronze source datasets...")

    for source_name, source_config in SOURCE_CONFIG.items():
        dataframe = standardise_source(
            ingested_data_dir,
            source_config,
        )
        standardised_sources[source_name] = dataframe

        print(
            f"  {source_config['source_system']:<11} "
            f"{len(dataframe):>7,} records standardised"
        )

    standardised_data_dir.mkdir(parents=True, exist_ok=True)
    row_counts = {}

    for source_name, source_config in SOURCE_CONFIG.items():
        dataframe = standardised_sources[source_name]
        output_path = (
            standardised_data_dir / source_config["output_filename"]
        )

        try:
            dataframe.to_csv(
                output_path,
                index=False,
                encoding="utf-8",
            )
        except OSError as error:
            raise OSError(
                f"Unable to write standardised source file {output_path}: "
                f"{error}"
            ) from error

        row_counts[source_name] = len(dataframe)

    print(f"\nStandardised files written to: {standardised_data_dir}")
    print(f"Total records standardised: {sum(row_counts.values()):,}")

    return row_counts


###############################################################################
# 5. Run standardisation pipeline
###############################################################################

def main() -> None:
    # Run the Bronze-to-Silver standardisation stage.

    try:
        standardise_all_sources()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"Standardisation failed: {error}") from error


if __name__ == "__main__":
    main()
