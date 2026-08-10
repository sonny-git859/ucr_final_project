###############################################################################
# Imports
###############################################################################

from pathlib import Path

import pandas as pd


###############################################################################
# 1. File paths and source configuration
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INGESTED_DATA_DIR = PROJECT_ROOT / "data" / "ingested_bronze"

SOURCE_CONFIG = {
    "crm": {
        "filename": "crm_customer_records.csv",
        "source_system": "CRM",
        "record_id_column": "crm_customer_id",
        "required_columns": [
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
    },
    "ecommerce": {
        "filename": "ecommerce_transactions.csv",
        "source_system": "ECOMMERCE",
        "record_id_column": "transaction_id",
        "required_columns": [
            "transaction_id",
            "portal_user_id",
            "billing_address",
            "billing_postcode",
            "email",
            "purchaser_first_name",
            "purchaser_surname",
            "telephone_number",
        ],
    },
    "marketing": {
        "filename": "marketing_contact_lists.csv",
        "source_system": "MARKETING",
        "record_id_column": "marketing_contact_id",
        "required_columns": [
            "marketing_contact_id",
            "contact_name",
            "email",
            "postcode",
        ],
    },
    "online": {
        "filename": "online_sessions.csv",
        "source_system": "ONLINE",
        "record_id_column": "session_id",
        "required_columns": [
            "session_id",
            "portal_user_id",
            "linked_transaction_id",
        ],
    },
    "support": {
        "filename": "support_ticket_logs.csv",
        "source_system": "SUPPORT",
        "record_id_column": "support_ticket_id",
        "required_columns": [
            "support_ticket_id",
            "requester_name",
            "requester_email",
            "requester_phone",
        ],
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


def validate_source_record_id(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    # Confirm that source record identifiers are populated and unique

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


###############################################################################
# 3. Ingestion functions
###############################################################################

def add_source_metadata(
    dataframe: pd.DataFrame,
    source_system: str,
    record_id_column: str,
    filename: str,
) -> pd.DataFrame:
    # Add common source metadata without changing original source values.

    ingested_dataframe = dataframe.copy()

    if "source_system" in ingested_dataframe.columns:
        source_values = ingested_dataframe["source_system"]
        invalid_values = source_values.ne(source_system)

        if invalid_values.any():
            raise ValueError(
                f"{filename} contains an inconsistent source_system column."
            )
    else:
        ingested_dataframe["source_system"] = source_system

    if "source_record_id" not in ingested_dataframe.columns:
        ingested_dataframe["source_record_id"] = (
            ingested_dataframe[record_id_column]
        )

    common_columns = ["source_system", "source_record_id"]
    remaining_columns = [
        column
        for column in ingested_dataframe.columns
        if column not in common_columns
    ]

    return ingested_dataframe[common_columns + remaining_columns]


def ingest_source(
    raw_data_dir: Path,
    source_config: dict,
) -> pd.DataFrame:
    # Read and structurally validate one raw source dataset.

    filename = source_config["filename"]
    input_path = raw_data_dir / filename

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Required raw source file was not found: {input_path}"
        )

    try:
        dataframe = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ValueError(
            f"Unable to read raw source file {filename}: {error}"
        ) from error

    if dataframe.empty:
        raise ValueError(f"{filename} contains no records.")

    validate_required_columns(
        dataframe,
        source_config["required_columns"],
        filename,
    )

    ingested_dataframe = add_source_metadata(
        dataframe,
        source_config["source_system"],
        source_config["record_id_column"],
        filename,
    )

    validate_source_record_id(ingested_dataframe, filename)

    return ingested_dataframe


def ingest_all_sources(
    raw_data_dir: Path = RAW_DATA_DIR,
    ingested_data_dir: Path = INGESTED_DATA_DIR,
) -> dict[str, int]:
    # Ingest all configured sources and write intermediate CSV files.

    ingested_sources = {}

    print("Ingesting raw source datasets...")

    for source_name, source_config in SOURCE_CONFIG.items():
        dataframe = ingest_source(raw_data_dir, source_config)
        ingested_sources[source_name] = dataframe

        print(
            f"  {source_config['source_system']:<11} "
            f"{len(dataframe):>7,} records validated"
        )

    ingested_data_dir.mkdir(parents=True, exist_ok=True)
    row_counts = {}

    for source_name, source_config in SOURCE_CONFIG.items():
        dataframe = ingested_sources[source_name]

        source_filename = Path(source_config["filename"])
        output_filename = (
            f"{source_filename.stem}_bronze{source_filename.suffix}"
        )
        output_path = ingested_data_dir / output_filename

        try:
            dataframe.to_csv(
                output_path,
                index=False,
                encoding="utf-8",
            )
        except OSError as error:
            raise OSError(
                f"Unable to write ingested source file {output_path}: "
                f"{error}"
            ) from error

        row_counts[source_name] = len(dataframe)

    print(f"\nIngested files written to: {ingested_data_dir}")
    print(f"Total records ingested: {sum(row_counts.values()):,}")

    return row_counts


###############################################################################
# 4. Run ingestion pipeline
###############################################################################

def main() -> None:
    # Run the raw source ingestion stage
    try:
        ingest_all_sources()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"Ingestion failed: {error}") from error


if __name__ == "__main__":
    main()
