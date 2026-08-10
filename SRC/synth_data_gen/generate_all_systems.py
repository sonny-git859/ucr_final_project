###############################################################################
# Imports
###############################################################################

from pathlib import Path
import subprocess
import sys

import pandas as pd


###############################################################################
# 1. Orchestration configuration
###############################################################################

# Script location:
# UCR_PROJECT/SRC/synth_data_gen/generate_all_systems.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Generator scripts are executed in dependency order.
GENERATION_SCRIPTS = [
    (
        "Canonical",
        Path(
            "SRC/synth_data_gen/canonical_gen/"
            "canonical_customer_generation.py"
        ),
    ),
    (
        "Events",
        Path("SRC/synth_data_gen/events_gen/events_generation.py"),
    ),
    (
        "CRM",
        Path("SRC/synth_data_gen/crm_gen/crm_generation.py"),
    ),
    (
        "E-commerce",
        Path("SRC/synth_data_gen/ecommerce_gen/ecommerce_gen.py"),
    ),
    (
        "Online",
        Path(
            "SRC/synth_data_gen/online_sessions_gen/"
            "online_session_generation.py"
        ),
    ),
    (
        "Support",
        Path("SRC/synth_data_gen/support_gen/support_generation.py"),
    ),
    (
        "Marketing",
        Path(
            "SRC/synth_data_gen/marketing_gen/"
            "marketing_contacts_generation.py"
        ),
    ),
]

# system: (operational/reference file, unique ID, ground-truth mapping)
DATASET_CONFIG = {
    "Canonical": (
        Path("data/canonical/canonical_customers.csv"),
        "ground_truth_id",
        None,
    ),
    "Events": (
        Path("data/events/events.csv"),
        "event_id",
        None,
    ),
    "CRM": (
        Path("data/raw/crm_customer_records.csv"),
        "crm_customer_id",
        Path("data/reference/crm_ground_truth_mapping.csv"),
    ),
    "E-commerce": (
        Path("data/raw/ecommerce_transactions.csv"),
        "transaction_id",
        Path("data/reference/ecommerce_ground_truth_mapping.csv"),
    ),
    "Online": (
        Path("data/raw/online_sessions.csv"),
        "session_id",
        Path("data/reference/online_ground_truth_mapping.csv"),
    ),
    "Support": (
        Path("data/raw/support_ticket_logs.csv"),
        "support_ticket_id",
        Path("data/reference/support_ground_truth_mapping.csv"),
    ),
    "Marketing": (
        Path("data/raw/marketing_contact_lists.csv"),
        "marketing_contact_id",
        Path("data/reference/marketing_ground_truth_mapping.csv"),
    ),
}

PORTAL_MAPPING_PATH = Path(
    "data/reference/portal_account_mapping.csv"
)

VALIDATION_SUMMARY_PATHS = [
    Path("data/reference/crm_validation_summary.csv"),
    Path("data/reference/ecommerce_validation_summary.csv"),
    Path("data/reference/online_validation_summary.csv"),
    Path("data/reference/support_validation_summary.csv"),
    Path("data/reference/marketing_validation_summary.csv"),
]

ENVIRONMENT_VALIDATION_OUTPUT_PATH = Path(
    "data/reference/system_validation_summary.csv"
)


###############################################################################
# 2. Helper functions
###############################################################################

# Resolve a path relative to the project root.
def project_path(relative_path: Path) -> Path:
    return PROJECT_ROOT / relative_path


# Confirm a required generated file exists.
def validate_file_exists(relative_path: Path) -> None:
    absolute_path = project_path(relative_path)
    assert absolute_path.exists(), f"Required file not found: {absolute_path}"


# Load a project CSV.
def load_csv(relative_path: Path) -> pd.DataFrame:
    return pd.read_csv(project_path(relative_path))


# Convert a system name into a validation-metric prefix.
def metric_prefix(system_name: str) -> str:
    return (
        system_name.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


# Confirm every source record has exactly one mapping row.
def validate_mapping_complete(
    source_records: pd.DataFrame,
    mapping_records: pd.DataFrame,
    source_id_column: str,
) -> bool:
    if not mapping_records[source_id_column].is_unique:
        return False

    source_ids = set(source_records[source_id_column])
    mapping_ids = set(mapping_records[source_id_column])

    return source_ids == mapping_ids


# Add a labelled section to the validation summary.
def append_summary_section(
    summary_rows: list[dict],
    section_name: str,
    metrics: dict,
) -> None:
    if summary_rows:
        summary_rows.append({"metric": "", "value": ""})

    summary_rows.append({"metric": section_name, "value": ""})

    for metric, value in metrics.items():
        summary_rows.append({"metric": metric, "value": value})


###############################################################################
# 3. Execute generation scripts
###############################################################################

# Execute each generator as separate Python process.
def run_generation_scripts() -> None:
    print("\nStarting complete synthetic-data generation...")

    for system_name, relative_path in GENERATION_SCRIPTS:
        script_path = project_path(relative_path)
        assert script_path.exists(), f"Generator not found: {script_path}"

        print(f"\nRunning {system_name} generator...")

        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )

    print("\nAll generation scripts completed successfully.")


###############################################################################
# 4. Validate generated environment
###############################################################################

def validate_generated_environment() -> pd.DataFrame:

    ###########################################################################
    # Confirm required outputs exist
    ###########################################################################

    required_paths = []

    for data_path, _, mapping_path in DATASET_CONFIG.values():
        required_paths.append(data_path)

        if mapping_path is not None:
            required_paths.append(mapping_path)

    required_paths.append(PORTAL_MAPPING_PATH)
    required_paths.extend(VALIDATION_SUMMARY_PATHS)

    for relative_path in required_paths:
        validate_file_exists(relative_path)

    ###########################################################################
    # Load generated datasets and mappings
    ###########################################################################

    datasets = {
        name: load_csv(config[0])
        for name, config in DATASET_CONFIG.items()
    }

    mappings = {
        name: load_csv(config[2])
        for name, config in DATASET_CONFIG.items()
        if config[2] is not None
    }

    portal_mapping = load_csv(PORTAL_MAPPING_PATH)

    ###########################################################################
    # Validate record counts and unique source identifiers
    ###########################################################################

    record_counts = {
        f"{metric_prefix(name)}_records": len(records)
        for name, records in datasets.items()
    }

    unique_id_checks = {}

    for name, (_, id_column, _) in DATASET_CONFIG.items():
        metric = f"{metric_prefix(name)}_source_ids_unique"
        unique_id_checks[metric] = bool(
            datasets[name][id_column].is_unique
        )

    assert all(unique_id_checks.values())

    ###########################################################################
    # Validate ground-truth mapping completeness
    ###########################################################################

    mapping_checks = {}

    for name, mapping_records in mappings.items():
        id_column = DATASET_CONFIG[name][1]
        metric = f"{metric_prefix(name)}_mapping_complete"

        mapping_checks[metric] = validate_mapping_complete(
            source_records=datasets[name],
            mapping_records=mapping_records,
            source_id_column=id_column,
        )

    assert all(mapping_checks.values())

    canonical_ids = set(datasets["Canonical"]["ground_truth_id"])
    ground_truth_ids_valid = True

    for mapping_records in mappings.values():
        mapped_ids = mapping_records["ground_truth_id"]

        if mapped_ids.isna().any():
            ground_truth_ids_valid = False

        if not set(mapped_ids).issubset(canonical_ids):
            ground_truth_ids_valid = False

    assert ground_truth_ids_valid

    ###########################################################################
    # Validate cross-system references
    ###########################################################################

    ecommerce_event_references_valid = set(
        datasets["E-commerce"]["event_id"]
    ).issubset(
        set(datasets["Events"]["event_id"])
    )

    online_transaction_references_valid = set(
        datasets["Online"]["linked_transaction_id"].dropna()
    ).issubset(
        set(datasets["E-commerce"]["transaction_id"])
    )

    portal_mapping_valid = (
        portal_mapping["ground_truth_id"].is_unique
        and portal_mapping["portal_user_id"].is_unique
        and set(portal_mapping["ground_truth_id"]).issubset(canonical_ids)
    )

    valid_portal_ids = set(portal_mapping["portal_user_id"])
    portal_references_valid = True

    for system_name in ["CRM", "E-commerce", "Online"]:
        source_portal_ids = set(
            datasets[system_name]["portal_user_id"].dropna()
        )

        if not source_portal_ids.issubset(valid_portal_ids):
            portal_references_valid = False

    referential_checks = {
        "ecommerce_event_references_valid": (
            ecommerce_event_references_valid
        ),
        "online_transaction_references_valid": (
            online_transaction_references_valid
        ),
        "portal_mapping_valid": portal_mapping_valid,
        "portal_references_valid": portal_references_valid,
        "ground_truth_ids_valid": ground_truth_ids_valid,
        "validation_summaries_present": True,
    }

    assert all(referential_checks.values())

    ###########################################################################
    # Create and export environment validation summary
    ###########################################################################

    summary_rows = []

    append_summary_section(
        summary_rows,
        "RECORD COUNTS",
        record_counts,
    )

    append_summary_section(
        summary_rows,
        "UNIQUE SOURCE IDENTIFIERS",
        unique_id_checks,
    )

    append_summary_section(
        summary_rows,
        "GROUND TRUTH MAPPING COMPLETENESS",
        mapping_checks,
    )

    append_summary_section(
        summary_rows,
        "REFERENTIAL CONSISTENCY",
        referential_checks,
    )

    validation_summary = pd.DataFrame(summary_rows)
    output_path = project_path(ENVIRONMENT_VALIDATION_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_summary.to_csv(output_path, index=False)

    ###########################################################################
    # Visualise validation results
    ###########################################################################

    print("\nEnvironment validation completed successfully.")
    print("\nGenerated record counts:")

    for metric, value in record_counts.items():
        print(f"{metric}: {value:,}")

    print(
        "\nGround-truth mappings and cross-system references "
        "validated successfully."
    )

    return validation_summary


###############################################################################
# 5. Main function
###############################################################################

def main() -> None:
    # Execute all generators in dependency order.
    run_generation_scripts()

    # Perform lightweight environment-level validation.
    validate_generated_environment()

    output_path = project_path(ENVIRONMENT_VALIDATION_OUTPUT_PATH)

    print(
        f"\nSystem validation summary saved to: "
        f"{output_path.resolve()}"
    )

    print("\nComplete synthetic-data environment generated successfully.")


if __name__ == "__main__":
    main()
