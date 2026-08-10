###############################################################################
# Imports
###############################################################################

from pathlib import Path

import pandas as pd


###############################################################################
# 1. File paths and source configuration
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STANDARDISED_DATA_DIR = PROJECT_ROOT / "data" / "standardised_silver"
CONSOLIDATED_DATA_DIR = PROJECT_ROOT / "data" / "consolidated_silver"
OUTPUT_FILENAME = "consolidated_customer_records.csv"
SUMMARY_FILENAME = "consolidation_summary.csv"

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

EXPECTED_COMMON_COLUMNS = (
    ["source_system", "source_record_id"]
    + COMMON_IDENTIFIER_COLUMNS
    + RAW_IDENTITY_COLUMNS
    + NORMALISED_IDENTITY_COLUMNS
)

SOURCE_CONFIG = {
    "crm": {
        "filename": "crm_customer_records_silver.csv",
        "source_system": "CRM",
    },
    "ecommerce": {
        "filename": "ecommerce_transactions_silver.csv",
        "source_system": "ECOMMERCE",
    },
    "marketing": {
        "filename": "marketing_contact_lists_silver.csv",
        "source_system": "MARKETING",
    },
    "online": {
        "filename": "online_sessions_silver.csv",
        "source_system": "ONLINE",
    },
    "support": {
        "filename": "support_ticket_logs_silver.csv",
        "source_system": "SUPPORT",
    },
}


###############################################################################
# 2. Validation functions
###############################################################################

def validate_common_schema(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    # Confirm that every Silver source contains the common staging schema.

    missing_columns = [
        column
        for column in EXPECTED_COMMON_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        missing_values = ", ".join(missing_columns)
        raise ValueError(
            f"{filename} is missing common columns: {missing_values}."
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


def validate_consolidated_records(
    dataframe: pd.DataFrame,
    expected_record_count: int,
) -> None:
    # Confirm that consolidation preserves every source record exactly once.

    if len(dataframe) != expected_record_count:
        raise ValueError(
            "Consolidation produced an unexpected number of records: "
            f"expected {expected_record_count:,}, found {len(dataframe):,}."
        )

    duplicate_source_records = dataframe.duplicated(
        subset=["source_system", "source_record_id"],
        keep=False,
    )

    if duplicate_source_records.any():
        duplicate_count = int(duplicate_source_records.sum())
        raise ValueError(
            f"Consolidated data contains {duplicate_count:,} duplicate "
            "source record reference(s)."
        )

    staging_record_ids = dataframe["staging_record_id"]
    missing_staging_ids = staging_record_ids.str.strip().eq("")

    if missing_staging_ids.any():
        missing_count = int(missing_staging_ids.sum())
        raise ValueError(
            f"Consolidated data contains {missing_count:,} missing "
            "staging_record_id value(s)."
        )

    duplicate_staging_ids = staging_record_ids.duplicated(keep=False)

    if duplicate_staging_ids.any():
        duplicate_count = int(duplicate_staging_ids.sum())
        raise ValueError(
            f"Consolidated data contains {duplicate_count:,} duplicate "
            "staging_record_id value(s)."
        )


def validate_consolidated_schema(
    dataframe: pd.DataFrame,
    standardised_sources: dict[str, pd.DataFrame],
) -> None:
    # Confirm that consolidation preserves the complete Silver union schema.

    expected_columns = [
        "staging_record_id",
        *build_union_columns(standardised_sources),
    ]
    actual_columns = dataframe.columns.tolist()

    if actual_columns != expected_columns:
        missing_columns = [
            column
            for column in expected_columns
            if column not in actual_columns
        ]
        unexpected_columns = [
            column
            for column in actual_columns
            if column not in expected_columns
        ]

        details = []

        if missing_columns:
            details.append(
                f"missing columns: {', '.join(missing_columns)}"
            )

        if unexpected_columns:
            details.append(
                f"unexpected columns: {', '.join(unexpected_columns)}"
            )

        if not details:
            details.append("columns are in an unexpected order")

        raise ValueError(
            "Consolidated data does not contain the expected complete "
            f"Silver schema; {'; '.join(details)}."
        )

    if "ground_truth_id" in dataframe.columns:
        raise ValueError(
            "Consolidated data must not contain ground_truth_id."
        )


def validate_source_records_preserved(
    dataframe: pd.DataFrame,
    standardised_sources: dict[str, pd.DataFrame],
) -> None:
    # Confirm that every original Silver value remains unchanged.

    union_columns = build_union_columns(standardised_sources)
    expected_dataframes = [
        standardised_sources[source_name].reindex(
            columns=union_columns,
            fill_value="",
        )
        for source_name in SOURCE_CONFIG
    ]
    expected_dataframe = pd.concat(
        expected_dataframes,
        ignore_index=True,
    )
    actual_dataframe = dataframe[union_columns]

    if not actual_dataframe.equals(expected_dataframe):
        raise ValueError(
            "Consolidation changed or omitted one or more Silver "
            "source values."
        )


###############################################################################
# 3. Consolidation functions
###############################################################################

def read_standardised_source(
    standardised_data_dir: Path,
    source_config: dict,
) -> pd.DataFrame:
    # Read and structurally validate one standardised Silver source.

    filename = source_config["filename"]
    input_path = standardised_data_dir / filename

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Required Silver source file was not found: {input_path}"
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
            f"Unable to read Silver source file {filename}: {error}"
        ) from error

    if dataframe.empty:
        raise ValueError(f"{filename} contains no records.")

    validate_common_schema(dataframe, filename)
    validate_source_metadata(
        dataframe,
        source_config["source_system"],
        filename,
    )

    return dataframe.copy()


def build_union_columns(
    standardised_sources: dict[str, pd.DataFrame],
) -> list[str]:
    # Build an ordered schema containing every Silver source column.

    union_columns = EXPECTED_COMMON_COLUMNS.copy()

    for source_name in SOURCE_CONFIG:
        dataframe = standardised_sources[source_name]

        for column in dataframe.columns:
            if column not in union_columns:
                union_columns.append(column)

    return union_columns


def add_staging_record_id(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    # Assign a technical row identifier without resolving customer identities.

    consolidated_dataframe = dataframe.copy()
    record_numbers = range(1, len(consolidated_dataframe) + 1)
    consolidated_dataframe.insert(
        0,
        "staging_record_id",
        [f"STG{number:06d}" for number in record_numbers],
    )

    return consolidated_dataframe


def consolidate_sources(
    standardised_sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    # Vertically append all Silver records without merging duplicates.

    union_columns = build_union_columns(standardised_sources)

    aligned_dataframes = [
        standardised_sources[source_name].reindex(
            columns=union_columns,
            fill_value="",
        )
        for source_name in SOURCE_CONFIG
    ]

    consolidated_dataframe = pd.concat(
        aligned_dataframes,
        ignore_index=True,
    )

    return add_staging_record_id(consolidated_dataframe)


###############################################################################
# 4. Summary functions
###############################################################################

def create_consolidation_summary(
    dataframe: pd.DataFrame,
    standardised_sources: dict[str, pd.DataFrame],
    row_counts: dict[str, int],
) -> pd.DataFrame:
    # Create a metric-value report of consolidation validation results.

    expected_record_count = sum(row_counts.values())
    expected_columns = [
        "staging_record_id",
        *build_union_columns(standardised_sources),
    ]
    duplicate_source_records = dataframe.duplicated(
        subset=["source_system", "source_record_id"],
        keep=False,
    )
    staging_record_ids = dataframe["staging_record_id"]
    union_columns = build_union_columns(standardised_sources)
    expected_dataframes = [
        standardised_sources[source_name].reindex(
            columns=union_columns,
            fill_value="",
        )
        for source_name in SOURCE_CONFIG
    ]
    expected_dataframe = pd.concat(
        expected_dataframes,
        ignore_index=True,
    )

    common_schema_complete = all(
        all(
            column in source_dataframe.columns
            for column in EXPECTED_COMMON_COLUMNS
        )
        for source_dataframe in standardised_sources.values()
    )
    source_metadata_valid = all(
        source_dataframe["source_system"]
        .eq(SOURCE_CONFIG[source_name]["source_system"])
        .all()
        and not source_dataframe["source_record_id"]
        .str.strip()
        .eq("")
        .any()
        and not source_dataframe["source_record_id"]
        .duplicated(keep=False)
        .any()
        for source_name, source_dataframe in standardised_sources.items()
    )
    source_record_counts_preserved = all(
        int(
            dataframe["source_system"]
            .eq(source_config["source_system"])
            .sum()
        )
        == row_counts[source_name]
        for source_name, source_config in SOURCE_CONFIG.items()
    )

    summary_rows = [
        {"metric": "RECORD COUNTS", "value": ""},
    ]

    for source_name in SOURCE_CONFIG:
        source_metric = f"{source_name}_records"
        summary_rows.append(
            {
                "metric": source_metric,
                "value": row_counts[source_name],
            }
        )

    summary_rows.extend(
        [
            {
                "metric": "total_input_records",
                "value": expected_record_count,
            },
            {
                "metric": "consolidated_records",
                "value": len(dataframe),
            },
            {"metric": "", "value": ""},
            {"metric": "SCHEMA COUNTS", "value": ""},
            {
                "metric": "required_common_columns",
                "value": len(EXPECTED_COMMON_COLUMNS),
            },
            {
                "metric": "unique_silver_columns",
                "value": len(expected_columns) - 1,
            },
            {
                "metric": "consolidated_columns",
                "value": len(dataframe.columns),
            },
            {"metric": "", "value": ""},
            {"metric": "VALIDATION RESULTS", "value": ""},
            {
                "metric": "all_source_files_read",
                "value": len(standardised_sources) == len(SOURCE_CONFIG),
            },
            {
                "metric": "common_schema_complete",
                "value": common_schema_complete,
            },
            {
                "metric": "source_metadata_valid",
                "value": source_metadata_valid,
            },
            {
                "metric": "record_count_preserved",
                "value": len(dataframe) == expected_record_count,
            },
            {
                "metric": "source_record_counts_preserved",
                "value": source_record_counts_preserved,
            },
            {
                "metric": "source_record_references_unique",
                "value": not duplicate_source_records.any(),
            },
            {
                "metric": "staging_record_ids_complete",
                "value": not staging_record_ids.str.strip().eq("").any(),
            },
            {
                "metric": "staging_record_ids_unique",
                "value": not staging_record_ids.duplicated().any(),
            },
            {
                "metric": "complete_union_schema_preserved",
                "value": dataframe.columns.tolist() == expected_columns,
            },
            {
                "metric": "all_silver_values_preserved",
                "value": dataframe[union_columns].equals(
                    expected_dataframe
                ),
            },
            {
                "metric": "ground_truth_id_excluded",
                "value": "ground_truth_id" not in dataframe.columns,
            },
        ]
    )

    return pd.DataFrame(summary_rows, columns=["metric", "value"])


def print_source_summary(
    dataframe: pd.DataFrame,
) -> None:
    # Print record counts for each source in the consolidated dataset.

    source_counts = dataframe["source_system"].value_counts(sort=False)

    print("\nConsolidated record summary:")

    for source_config in SOURCE_CONFIG.values():
        source_system = source_config["source_system"]
        record_count = int(source_counts.get(source_system, 0))
        print(f"  {source_system:<11} {record_count:>7,} records")

    print(f"  {'TOTAL':<11} {len(dataframe):>7,} records")
    print(f"\nTotal consolidated columns: {len(dataframe.columns):,}")


###############################################################################
# 5. Pipeline orchestration
###############################################################################

def consolidate_all_sources(
    standardised_data_dir: Path = STANDARDISED_DATA_DIR,
    consolidated_data_dir: Path = CONSOLIDATED_DATA_DIR,
) -> dict[str, int]:
    # Consolidate all standardised Silver sources into one staging dataset.

    standardised_sources = {}
    row_counts = {}

    print("Reading standardised Silver source datasets...")

    for source_name, source_config in SOURCE_CONFIG.items():
        dataframe = read_standardised_source(
            standardised_data_dir,
            source_config,
        )
        standardised_sources[source_name] = dataframe
        row_counts[source_name] = len(dataframe)

        print(
            f"  {source_config['source_system']:<11} "
            f"{len(dataframe):>7,} records validated"
        )

    consolidated_dataframe = consolidate_sources(standardised_sources)
    expected_record_count = sum(row_counts.values())
    validate_consolidated_records(
        consolidated_dataframe,
        expected_record_count,
    )
    validate_consolidated_schema(
        consolidated_dataframe,
        standardised_sources,
    )
    validate_source_records_preserved(
        consolidated_dataframe,
        standardised_sources,
    )

    consolidation_summary = create_consolidation_summary(
        consolidated_dataframe,
        standardised_sources,
        row_counts,
    )
    consolidated_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = consolidated_data_dir / OUTPUT_FILENAME
    summary_path = consolidated_data_dir / SUMMARY_FILENAME

    try:
        consolidated_dataframe.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
        )
        consolidation_summary.to_csv(
            summary_path,
            index=False,
            encoding="utf-8",
        )
    except OSError as error:
        raise OSError(
            "Unable to write consolidated Silver outputs: "
            f"{error}"
        ) from error

    print_source_summary(consolidated_dataframe)
    print(f"\nConsolidated file written to: {output_path}")
    print(f"Consolidation summary written to: {summary_path}")

    return row_counts


###############################################################################
# 6. Run consolidation pipeline
###############################################################################

def main() -> None:
    # Run the Silver consolidation stage.

    consolidate_all_sources()


if __name__ == "__main__":
    main()