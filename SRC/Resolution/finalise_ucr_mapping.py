# This script is the final identity-mapping stage of the Universal Customer
# Record (UCR) project. It converts eligible provisional PCL clusters into
# stable UCR identifiers without consulting protected ground truth.
#
# UCR identifiers are assigned in ascending order of each cluster's minimum
# staging_record_id. Eligible singleton clusters remain valid UCR profiles but
# retain the unresolved_singleton status. Anonymous records remain unassigned.

###############################################################################
# Imports
###############################################################################

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


###############################################################################
# 1. Configuration
###############################################################################

INPUT_DIRECTORY = (
    Path("identity_resolution") / "probabilistic_calibrated"
)
OUTPUT_DIRECTORY = Path("identity_resolution") / "final"

INPUT_FILENAME = "probabilistic_record_mapping.csv"
RECORD_MAPPING_FILENAME = "record_to_ucr_mapping.csv"
CLUSTER_SUMMARY_FILENAME = "ucr_cluster_summary.csv"
UNRESOLVED_FILENAME = "unresolved_singleton_records.csv"
ANONYMOUS_FILENAME = "anonymous_unresolvable_records.csv"
FINALISATION_SUMMARY_FILENAME = "finalisation_summary.csv"

UCR_PREFIX = "UCR"
UCR_DIGITS = 6

REQUIRED_COLUMNS = {
    "staging_record_id",
    "source_system",
    "source_record_id",
    "deterministic_cluster_id",
    "probabilistic_cluster_id",
    "resolution_status",
    "match_method",
    "cluster_size",
    "deterministic_cluster_count",
    "cluster_confidence",
    "cluster_confidence_type",
    "deterministic_linking_rules",
}

ELIGIBLE_STATUSES = {
    "deterministically_linked",
    "probabilistically_linked",
    "unresolved_singleton",
}

ALLOWED_STATUSES = ELIGIBLE_STATUSES | {"anonymous_unresolvable"}

EXPECTED_METHODS = {
    "deterministically_linked": "Deterministic",
    "probabilistically_linked": "Hybrid",
    "unresolved_singleton": "Unresolved",
    "anonymous_unresolvable": "Not applicable",
}

CORE_OUTPUT_COLUMNS = [
    "staging_record_id",
    "ucr_id",
    "source_system",
    "source_record_id",
    "deterministic_cluster_id",
    "probabilistic_cluster_id",
    "resolution_status",
    "match_method",
    "cluster_size",
    "deterministic_cluster_count",
    "cluster_confidence",
    "cluster_confidence_type",
    "deterministic_linking_rules",
]

CLUSTER_SUMMARY_COLUMNS = [
    "ucr_id",
    "probabilistic_cluster_id",
    "minimum_staging_record_id",
    "resolution_status",
    "match_method",
    "cluster_size",
    "deterministic_cluster_count",
    "deterministic_cluster_ids",
    "source_system_count",
    "source_systems",
    "cluster_confidence",
    "cluster_confidence_type",
    "deterministic_linking_rules",
]


###############################################################################
# 2. General helper functions
###############################################################################


def clean_value(value: object) -> str:
    # Return a stripped string while treating missing values as blank.
    if pd.isna(value):
        return ""

    return str(value).strip()


def serialise_values(values: Iterable[object]) -> str:
    # Serialise unique populated values in stable ascending order.
    cleaned = {
        clean_value(value)
        for value in values
        if clean_value(value)
    }
    return " | ".join(sorted(cleaned))


def serialise_linking_rules(values: Iterable[object]) -> str:
    # Combine pipe-delimited rule lists without retaining duplicates.
    rules: set[str] = set()
    for value in values:
        rules.update(
            rule.strip()
            for rule in clean_value(value).split("|")
            if rule.strip()
        )

    return " | ".join(sorted(rules))


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    description: str,
) -> None:
    # Confirm that a dataframe contains every required field.
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"{description} is missing required columns: "
            + ", ".join(missing_columns)
        )


def validate_ground_truth_exclusion(dataframe: pd.DataFrame) -> None:
    # Prohibit protected ground-truth fields from finalisation.
    ground_truth_columns = [
        column
        for column in dataframe.columns
        if "ground_truth" in column.casefold()
    ]
    if ground_truth_columns:
        raise ValueError(
            "Ground-truth columns must not enter UCR finalisation: "
            + ", ".join(ground_truth_columns)
        )


def add_summary_section(
    rows: list[dict[str, object]],
    section_name: str,
) -> None:
    # Add a readable section header to the two-column summary.
    if rows:
        rows.append({"metric": "", "value": ""})
    rows.append({"metric": section_name, "value": ""})


###############################################################################
# 3. Command-line configuration
###############################################################################


def parse_arguments() -> argparse.Namespace:
    # Parse command-line input and output paths.
    parser = argparse.ArgumentParser(
        description="Assign stable final UCR identifiers."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_DIRECTORY / INPUT_FILENAME,
        help=(
            "Calibrated probabilistic record mapping. Default: "
            f"{INPUT_DIRECTORY / INPUT_FILENAME}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help=f"Output directory. Default: {OUTPUT_DIRECTORY}",
    )
    return parser.parse_args()


###############################################################################
# 4. Input loading and validation
###############################################################################


def load_mapping(path: Path) -> pd.DataFrame:
    # Load the calibrated mapping without converting IDs to numbers.
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibrated probabilistic mapping not found: {path}"
        )

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def validate_identifiers(mapping: pd.DataFrame) -> None:
    # Validate record identifiers and source references.
    for column in (
        "staging_record_id",
        "source_system",
        "source_record_id",
    ):
        cleaned = mapping[column].map(clean_value)
        if cleaned.eq("").any():
            raise ValueError(f"The input contains blank {column} values.")

    if mapping["staging_record_id"].duplicated().any():
        raise ValueError("The input contains duplicate staging records.")

    duplicate_sources = mapping.duplicated(
        subset=["source_system", "source_record_id"]
    )
    if duplicate_sources.any():
        raise ValueError(
            "The input contains duplicate source-system references."
        )


def validate_statuses(mapping: pd.DataFrame) -> None:
    # Validate resolution statuses and their associated match methods.
    statuses = set(mapping["resolution_status"].map(clean_value))
    unexpected_statuses = sorted(statuses - ALLOWED_STATUSES)
    if unexpected_statuses:
        raise ValueError(
            "The input contains unexpected resolution statuses: "
            + ", ".join(unexpected_statuses)
        )

    if not statuses:
        raise ValueError("The input does not contain resolution statuses.")

    for status, expected_method in EXPECTED_METHODS.items():
        status_rows = mapping.loc[
            mapping["resolution_status"].eq(status)
        ]
        if status_rows.empty:
            continue

        invalid_methods = ~status_rows["match_method"].eq(
            expected_method
        )
        if invalid_methods.any():
            raise ValueError(
                f"Status {status} must use match method "
                f"{expected_method}."
            )


def validate_cluster_assignment(mapping: pd.DataFrame) -> None:
    # Validate eligible and anonymous provisional cluster assignments.
    eligible = mapping.loc[
        mapping["resolution_status"].isin(ELIGIBLE_STATUSES)
    ].copy()
    anonymous = mapping.loc[
        mapping["resolution_status"].eq("anonymous_unresolvable")
    ].copy()

    if eligible.empty:
        raise ValueError("No eligible provisional clusters were supplied.")
    if eligible["probabilistic_cluster_id"].map(
        clean_value
    ).eq("").any():
        raise ValueError(
            "An eligible record has no probabilistic cluster ID."
        )
    if anonymous["probabilistic_cluster_id"].map(
        clean_value
    ).ne("").any():
        raise ValueError(
            "An anonymous record has a probabilistic cluster ID."
        )

    actual_sizes = eligible["probabilistic_cluster_id"].map(
        eligible["probabilistic_cluster_id"].value_counts()
    )
    stated_sizes = pd.to_numeric(
        eligible["cluster_size"],
        errors="coerce",
    )
    if stated_sizes.isna().any() or not stated_sizes.eq(
        actual_sizes
    ).all():
        raise ValueError("The input cluster sizes are inconsistent.")

    actual_dcl_counts = eligible["probabilistic_cluster_id"].map(
        eligible.groupby("probabilistic_cluster_id")[
            "deterministic_cluster_id"
        ].nunique()
    )
    stated_dcl_counts = pd.to_numeric(
        eligible["deterministic_cluster_count"],
        errors="coerce",
    )
    if stated_dcl_counts.isna().any() or not stated_dcl_counts.eq(
        actual_dcl_counts
    ).all():
        raise ValueError(
            "The input deterministic-cluster counts are inconsistent."
        )


def validate_cluster_consistency(mapping: pd.DataFrame) -> None:
    # Confirm that cluster-level attributes are constant within each PCL.
    eligible = mapping.loc[
        mapping["resolution_status"].isin(ELIGIBLE_STATUSES)
    ]
    cluster_fields = (
        "resolution_status",
        "match_method",
        "cluster_size",
        "deterministic_cluster_count",
        "cluster_confidence",
        "cluster_confidence_type",
    )

    for field in cluster_fields:
        value_counts = eligible.groupby(
            "probabilistic_cluster_id"
        )[field].nunique(dropna=False)
        if value_counts.gt(1).any():
            raise ValueError(
                f"The field {field} is inconsistent within a PCL."
            )


def validate_special_populations(mapping: pd.DataFrame) -> None:
    # Validate singleton and anonymous population characteristics.
    unresolved = mapping.loc[
        mapping["resolution_status"].eq("unresolved_singleton")
    ]
    unresolved_sizes = pd.to_numeric(
        unresolved["cluster_size"],
        errors="coerce",
    )
    if unresolved_sizes.isna().any() or not unresolved_sizes.eq(1).all():
        raise ValueError(
            "Every unresolved singleton must have cluster size one."
        )

    unresolved_dcl_counts = pd.to_numeric(
        unresolved["deterministic_cluster_count"],
        errors="coerce",
    )
    if unresolved_dcl_counts.isna().any() or not (
        unresolved_dcl_counts.eq(1).all()
    ):
        raise ValueError(
            "Every unresolved singleton must contain one DCL cluster."
        )

    if unresolved["cluster_confidence"].map(
        clean_value
    ).ne("").any():
        raise ValueError(
            "Unresolved singletons must not have cluster confidence."
        )

    anonymous = mapping.loc[
        mapping["resolution_status"].eq("anonymous_unresolvable")
    ]
    anonymous_fields = (
        "deterministic_cluster_id",
        "probabilistic_cluster_id",
        "cluster_size",
        "deterministic_cluster_count",
        "cluster_confidence",
        "cluster_confidence_type",
    )
    for field in anonymous_fields:
        if anonymous[field].map(clean_value).ne("").any():
            raise ValueError(
                f"Anonymous records must have blank {field} values."
            )


def validate_input(mapping: pd.DataFrame) -> None:
    # Run every calibrated-mapping input validation.
    validate_ground_truth_exclusion(mapping)
    validate_required_columns(
        mapping,
        REQUIRED_COLUMNS,
        "Calibrated probabilistic mapping",
    )
    validate_identifiers(mapping)
    validate_statuses(mapping)
    validate_cluster_assignment(mapping)
    validate_cluster_consistency(mapping)
    validate_special_populations(mapping)


###############################################################################
# 5. Stable UCR assignment
###############################################################################


def build_ucr_lookup(mapping: pd.DataFrame) -> pd.DataFrame:
    # Order PCL clusters by minimum staging ID and assign stable UCR IDs.
    eligible = mapping.loc[
        mapping["resolution_status"].isin(ELIGIBLE_STATUSES)
    ]
    cluster_minima = (
        eligible.groupby(
            "probabilistic_cluster_id",
            as_index=False,
        )["staging_record_id"]
        .min()
        .rename(
            columns={
                "staging_record_id": "minimum_staging_record_id",
            }
        )
    )
    cluster_minima = cluster_minima.sort_values(
        by=[
            "minimum_staging_record_id",
            "probabilistic_cluster_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    cluster_count = len(cluster_minima)
    maximum_identifiers = (10**UCR_DIGITS) - 1
    if cluster_count > maximum_identifiers:
        raise ValueError(
            f"Six-digit UCR IDs support at most {maximum_identifiers:,} "
            "clusters."
        )

    cluster_minima["ucr_id"] = [
        f"{UCR_PREFIX}{position:0{UCR_DIGITS}d}"
        for position in range(1, cluster_count + 1)
    ]
    return cluster_minima[
        [
            "probabilistic_cluster_id",
            "minimum_staging_record_id",
            "ucr_id",
        ]
    ]


def build_record_mapping(
    mapping: pd.DataFrame,
    ucr_lookup: pd.DataFrame,
) -> pd.DataFrame:
    # Add final UCR IDs while preserving every operational mapping field.
    final_mapping = mapping.merge(
        ucr_lookup[["probabilistic_cluster_id", "ucr_id"]],
        on="probabilistic_cluster_id",
        how="left",
        validate="many_to_one",
    )
    final_mapping["ucr_id"] = final_mapping["ucr_id"].fillna("")
    final_mapping = final_mapping.sort_values(
        "staging_record_id",
        kind="stable",
    ).reset_index(drop=True)

    remaining_columns = [
        column
        for column in final_mapping.columns
        if column not in CORE_OUTPUT_COLUMNS
    ]
    return final_mapping[CORE_OUTPUT_COLUMNS + remaining_columns]


###############################################################################
# 6. UCR cluster summary
###############################################################################


def get_single_cluster_value(
    group: pd.DataFrame,
    column: str,
) -> str:
    # Return the single validated value for a cluster-level field.
    values = group[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(
            f"Cluster {group.name} has inconsistent {column} values."
        )

    return clean_value(values[0])


def build_cluster_summary(
    final_mapping: pd.DataFrame,
    ucr_lookup: pd.DataFrame,
) -> pd.DataFrame:
    # Build one transparent summary row for every assigned UCR profile.
    eligible = final_mapping.loc[
        final_mapping["ucr_id"].ne("")
    ].copy()
    minimum_lookup = ucr_lookup.set_index(
        "probabilistic_cluster_id"
    )["minimum_staging_record_id"].to_dict()
    rows: list[dict[str, object]] = []

    grouped = eligible.groupby(
        "probabilistic_cluster_id",
        sort=False,
    )
    for cluster_id, group in grouped:
        source_systems = sorted(
            set(group["source_system"].map(clean_value))
        )
        rows.append(
            {
                "ucr_id": get_single_cluster_value(group, "ucr_id"),
                "probabilistic_cluster_id": cluster_id,
                "minimum_staging_record_id": minimum_lookup[
                    cluster_id
                ],
                "resolution_status": get_single_cluster_value(
                    group,
                    "resolution_status",
                ),
                "match_method": get_single_cluster_value(
                    group,
                    "match_method",
                ),
                "cluster_size": int(len(group)),
                "deterministic_cluster_count": int(
                    group["deterministic_cluster_id"].nunique()
                ),
                "deterministic_cluster_ids": serialise_values(
                    group["deterministic_cluster_id"]
                ),
                "source_system_count": len(source_systems),
                "source_systems": " | ".join(source_systems),
                "cluster_confidence": get_single_cluster_value(
                    group,
                    "cluster_confidence",
                ),
                "cluster_confidence_type": get_single_cluster_value(
                    group,
                    "cluster_confidence_type",
                ),
                "deterministic_linking_rules": (
                    serialise_linking_rules(
                        group["deterministic_linking_rules"]
                    )
                ),
            }
        )

    summary = pd.DataFrame(rows, columns=CLUSTER_SUMMARY_COLUMNS)
    return summary.sort_values(
        "ucr_id",
        kind="stable",
    ).reset_index(drop=True)


###############################################################################
# 7. Special-population outputs
###############################################################################


def build_unresolved_output(
    final_mapping: pd.DataFrame,
) -> pd.DataFrame:
    # Retain standalone unresolved profiles with their assigned UCR IDs.
    unresolved = final_mapping.loc[
        final_mapping["resolution_status"].eq(
            "unresolved_singleton"
        )
    ].copy()
    return unresolved.reset_index(drop=True)


def build_anonymous_output(
    final_mapping: pd.DataFrame,
) -> pd.DataFrame:
    # Retain anonymous records without assigning a UCR identifier.
    anonymous = final_mapping.loc[
        final_mapping["resolution_status"].eq(
            "anonymous_unresolvable"
        )
    ].copy()
    return anonymous.reset_index(drop=True)


###############################################################################
# 8. Finalisation summary
###############################################################################


def build_finalisation_summary(
    input_mapping: pd.DataFrame,
    final_mapping: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    unresolved: pd.DataFrame,
    anonymous: pd.DataFrame,
) -> pd.DataFrame:
    # Build a two-column validation and final-UCR result report.
    rows: list[dict[str, object]] = []
    eligible = final_mapping.loc[final_mapping["ucr_id"].ne("")]
    status_counts = final_mapping["resolution_status"].value_counts()
    profile_status_counts = cluster_summary[
        "resolution_status"
    ].value_counts()

    add_summary_section(rows, "INPUT VALIDATION")
    rows.extend(
        [
            {
                "metric": "input_mapping_rows",
                "value": len(input_mapping),
            },
            {
                "metric": "unique_staging_records",
                "value": input_mapping["staging_record_id"].nunique(),
            },
            {
                "metric": "ground_truth_columns_excluded",
                "value": True,
            },
            {
                "metric": "eligible_input_records",
                "value": len(eligible),
            },
            {
                "metric": "eligible_pcl_clusters",
                "value": input_mapping.loc[
                    input_mapping["probabilistic_cluster_id"].ne("")
                ]["probabilistic_cluster_id"].nunique(),
            },
            {
                "metric": "anonymous_input_records",
                "value": int(
                    status_counts.get("anonymous_unresolvable", 0)
                ),
            },
        ]
    )

    add_summary_section(rows, "FINAL UCR ASSIGNMENT")
    rows.extend(
        [
            {
                "metric": "final_ucr_profiles",
                "value": cluster_summary["ucr_id"].nunique(),
            },
            {
                "metric": "records_assigned_to_ucr",
                "value": len(eligible),
            },
            {
                "metric": "deterministically_linked_only_records",
                "value": int(
                    status_counts.get("deterministically_linked", 0)
                ),
            },
            {
                "metric": "deterministically_linked_only_profiles",
                "value": int(
                    profile_status_counts.get(
                        "deterministically_linked",
                        0,
                    )
                ),
            },
            {
                "metric": "probabilistically_linked_records",
                "value": int(
                    status_counts.get("probabilistically_linked", 0)
                ),
            },
            {
                "metric": "probabilistically_linked_profiles",
                "value": int(
                    profile_status_counts.get(
                        "probabilistically_linked",
                        0,
                    )
                ),
            },
            {
                "metric": "unresolved_singleton_records",
                "value": len(unresolved),
            },
            {
                "metric": "unresolved_singleton_profiles",
                "value": int(
                    profile_status_counts.get(
                        "unresolved_singleton",
                        0,
                    )
                ),
            },
            {
                "metric": "anonymous_unresolvable_records",
                "value": len(anonymous),
            },
        ]
    )

    add_summary_section(rows, "IDENTIFIER STABILITY")
    rows.extend(
        [
            {
                "metric": "assignment_order",
                "value": "minimum_staging_record_id",
            },
            {
                "metric": "ucr_identifier_format",
                "value": f"{UCR_PREFIX}{'#' * UCR_DIGITS}",
            },
            {
                "metric": "first_ucr_id",
                "value": clean_value(
                    cluster_summary["ucr_id"].min()
                ),
            },
            {
                "metric": "last_ucr_id",
                "value": clean_value(
                    cluster_summary["ucr_id"].max()
                ),
            },
            {
                "metric": "one_ucr_per_pcl_cluster",
                "value": True,
            },
            {
                "metric": "anonymous_records_without_ucr_id",
                "value": anonymous["ucr_id"].eq("").all(),
            },
        ]
    )

    add_summary_section(rows, "OUTPUT VALIDATION")
    rows.extend(
        [
            {
                "metric": "record_mapping_rows",
                "value": len(final_mapping),
            },
            {
                "metric": "mapping_reconciles_to_input",
                "value": len(final_mapping) == len(input_mapping),
            },
            {
                "metric": "ucr_cluster_summary_rows",
                "value": len(cluster_summary),
            },
            {
                "metric": "cluster_summary_reconciles_to_ucrs",
                "value": len(cluster_summary)
                == eligible["ucr_id"].nunique(),
            },
            {
                "metric": "unresolved_output_rows",
                "value": len(unresolved),
            },
            {
                "metric": "anonymous_output_rows",
                "value": len(anonymous),
            },
            {
                "metric": "cluster_sizes_reconcile",
                "value": True,
            },
            {
                "metric": "stable_ucr_order_validated",
                "value": True,
            },
        ]
    )

    return pd.DataFrame(rows, columns=["metric", "value"])


###############################################################################
# 9. Output validation
###############################################################################


def validate_preserved_input(
    input_mapping: pd.DataFrame,
    final_mapping: pd.DataFrame,
) -> None:
    # Confirm that finalisation only adds the UCR identifier field.
    input_ordered = input_mapping.sort_values(
        "staging_record_id",
        kind="stable",
    ).reset_index(drop=True)
    preserved = final_mapping[input_mapping.columns]
    if not preserved.equals(input_ordered):
        raise ValueError(
            "Finalisation changed an existing operational mapping field."
        )


def validate_ucr_identifiers(final_mapping: pd.DataFrame) -> None:
    # Validate assignment coverage, format and cluster uniqueness.
    eligible = final_mapping.loc[
        final_mapping["resolution_status"].isin(ELIGIBLE_STATUSES)
    ]
    anonymous = final_mapping.loc[
        final_mapping["resolution_status"].eq("anonymous_unresolvable")
    ]

    if eligible["ucr_id"].eq("").any():
        raise ValueError("An eligible record did not receive a UCR ID.")
    if anonymous["ucr_id"].ne("").any():
        raise ValueError("An anonymous record received a UCR ID.")

    ucr_pattern = rf"{UCR_PREFIX}\d{{{UCR_DIGITS}}}"
    if not eligible["ucr_id"].str.fullmatch(ucr_pattern).all():
        raise ValueError("A final UCR identifier has an invalid format.")

    pcl_to_ucr_counts = eligible.groupby(
        "probabilistic_cluster_id"
    )["ucr_id"].nunique()
    if not pcl_to_ucr_counts.eq(1).all():
        raise ValueError("A PCL cluster maps to more than one UCR ID.")

    ucr_to_pcl_counts = eligible.groupby("ucr_id")[
        "probabilistic_cluster_id"
    ].nunique()
    if not ucr_to_pcl_counts.eq(1).all():
        raise ValueError("A UCR ID maps to more than one PCL cluster.")


def validate_stable_assignment(
    final_mapping: pd.DataFrame,
    ucr_lookup: pd.DataFrame,
) -> None:
    # Recalculate the expected sequential ID assigned to each PCL.
    ordered_lookup = ucr_lookup.sort_values(
        by=[
            "minimum_staging_record_id",
            "probabilistic_cluster_id",
        ],
        kind="stable",
    ).reset_index(drop=True)
    expected_ids = [
        f"{UCR_PREFIX}{position:0{UCR_DIGITS}d}"
        for position in range(1, len(ordered_lookup) + 1)
    ]
    if ordered_lookup["ucr_id"].tolist() != expected_ids:
        raise ValueError("UCR identifiers are not sequential and stable.")

    actual_minima = final_mapping.loc[
        final_mapping["ucr_id"].ne("")
    ].groupby("probabilistic_cluster_id")[
        "staging_record_id"
    ].min()
    stated_minima = ucr_lookup.set_index(
        "probabilistic_cluster_id"
    )["minimum_staging_record_id"]
    if not actual_minima.sort_index().equals(stated_minima.sort_index()):
        raise ValueError("Stored cluster minimum staging IDs are invalid.")


def validate_cluster_summary(
    final_mapping: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> None:
    # Reconcile cluster-summary rows to the record-level mapping.
    eligible = final_mapping.loc[final_mapping["ucr_id"].ne("")]
    if not cluster_summary["ucr_id"].is_unique:
        raise ValueError("The UCR cluster summary has duplicate UCR IDs.")
    if not cluster_summary[
        "probabilistic_cluster_id"
    ].is_unique:
        raise ValueError("The UCR cluster summary has duplicate PCL IDs.")

    expected_ucr_ids = set(eligible["ucr_id"])
    if set(cluster_summary["ucr_id"]) != expected_ucr_ids:
        raise ValueError(
            "The UCR cluster summary does not contain every assigned UCR."
        )

    actual_sizes = eligible["ucr_id"].value_counts()
    stated_sizes = cluster_summary.set_index("ucr_id")[
        "cluster_size"
    ]
    stated_sizes = pd.to_numeric(stated_sizes, errors="coerce")
    if stated_sizes.isna().any() or not actual_sizes.sort_index().eq(
        stated_sizes.sort_index()
    ).all():
        raise ValueError("The UCR cluster-summary sizes are inconsistent.")

    actual_dcl_counts = eligible.groupby("ucr_id")[
        "deterministic_cluster_id"
    ].nunique()
    stated_dcl_counts = cluster_summary.set_index("ucr_id")[
        "deterministic_cluster_count"
    ]
    stated_dcl_counts = pd.to_numeric(
        stated_dcl_counts,
        errors="coerce",
    )
    if stated_dcl_counts.isna().any() or not (
        actual_dcl_counts.sort_index().eq(
            stated_dcl_counts.sort_index()
        ).all()
    ):
        raise ValueError(
            "The UCR deterministic-cluster counts are inconsistent."
        )


def validate_special_outputs(
    final_mapping: pd.DataFrame,
    unresolved: pd.DataFrame,
    anonymous: pd.DataFrame,
) -> None:
    # Reconcile unresolved and anonymous extracts to the final mapping.
    expected_unresolved = set(
        final_mapping.loc[
            final_mapping["resolution_status"].eq(
                "unresolved_singleton"
            ),
            "staging_record_id",
        ]
    )
    if set(unresolved["staging_record_id"]) != expected_unresolved:
        raise ValueError(
            "The unresolved-singleton output does not reconcile."
        )
    if unresolved["ucr_id"].eq("").any():
        raise ValueError("An unresolved singleton has no final UCR ID.")

    expected_anonymous = set(
        final_mapping.loc[
            final_mapping["resolution_status"].eq(
                "anonymous_unresolvable"
            ),
            "staging_record_id",
        ]
    )
    if set(anonymous["staging_record_id"]) != expected_anonymous:
        raise ValueError("The anonymous output does not reconcile.")
    if anonymous["ucr_id"].ne("").any():
        raise ValueError("An anonymous output record received a UCR ID.")


def validate_outputs(
    input_mapping: pd.DataFrame,
    final_mapping: pd.DataFrame,
    ucr_lookup: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    unresolved: pd.DataFrame,
    anonymous: pd.DataFrame,
) -> None:
    # Run every final-UCR output validation.
    if len(final_mapping) != len(input_mapping):
        raise ValueError("The final mapping does not reconcile to input.")
    if not final_mapping["staging_record_id"].is_unique:
        raise ValueError("The final mapping has duplicate staging records.")

    validate_ground_truth_exclusion(final_mapping)
    validate_ground_truth_exclusion(cluster_summary)
    validate_ground_truth_exclusion(unresolved)
    validate_ground_truth_exclusion(anonymous)
    validate_preserved_input(input_mapping, final_mapping)
    validate_ucr_identifiers(final_mapping)
    validate_stable_assignment(final_mapping, ucr_lookup)
    validate_cluster_summary(final_mapping, cluster_summary)
    validate_special_outputs(final_mapping, unresolved, anonymous)


###############################################################################
# 10. Output writing
###############################################################################


def write_outputs(
    output_directory: Path,
    final_mapping: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    unresolved: pd.DataFrame,
    anonymous: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    # Write the five validated final-UCR CSV outputs.
    output_directory.mkdir(parents=True, exist_ok=True)
    final_mapping.to_csv(
        output_directory / RECORD_MAPPING_FILENAME,
        index=False,
    )
    cluster_summary.to_csv(
        output_directory / CLUSTER_SUMMARY_FILENAME,
        index=False,
    )
    unresolved.to_csv(
        output_directory / UNRESOLVED_FILENAME,
        index=False,
    )
    anonymous.to_csv(
        output_directory / ANONYMOUS_FILENAME,
        index=False,
    )
    summary.to_csv(
        output_directory / FINALISATION_SUMMARY_FILENAME,
        index=False,
    )


###############################################################################
# 11. Main finalisation workflow
###############################################################################


def run_finalisation(
    input_path: Path,
    output_directory: Path,
) -> dict[str, int]:
    # Assign stable UCR identifiers and write validated outputs.
    input_mapping = load_mapping(input_path)
    validate_input(input_mapping)

    ucr_lookup = build_ucr_lookup(input_mapping)
    final_mapping = build_record_mapping(input_mapping, ucr_lookup)
    cluster_summary = build_cluster_summary(
        final_mapping,
        ucr_lookup,
    )
    unresolved = build_unresolved_output(final_mapping)
    anonymous = build_anonymous_output(final_mapping)
    summary = build_finalisation_summary(
        input_mapping,
        final_mapping,
        cluster_summary,
        unresolved,
        anonymous,
    )

    validate_outputs(
        input_mapping,
        final_mapping,
        ucr_lookup,
        cluster_summary,
        unresolved,
        anonymous,
    )
    write_outputs(
        output_directory,
        final_mapping,
        cluster_summary,
        unresolved,
        anonymous,
        summary,
    )

    profile_counts = cluster_summary[
        "resolution_status"
    ].value_counts()
    return {
        "input_records": len(input_mapping),
        "assigned_records": int(final_mapping["ucr_id"].ne("").sum()),
        "final_ucr_profiles": len(cluster_summary),
        "deterministic_profiles": int(
            profile_counts.get("deterministically_linked", 0)
        ),
        "probabilistic_profiles": int(
            profile_counts.get("probabilistically_linked", 0)
        ),
        "unresolved_singletons": len(unresolved),
        "anonymous_unresolvable": len(anonymous),
    }


def main() -> None:
    # Run the command-line workflow and print a concise result.
    arguments = parse_arguments()
    results = run_finalisation(
        arguments.input,
        arguments.output_dir,
    )

    print("Final UCR assignment completed successfully.")
    print(f"Input: {arguments.input.resolve()}")
    print(f"Output directory: {arguments.output_dir.resolve()}")
    print(f"Input records: {results['input_records']:,}")
    print(f"Records assigned to a UCR: {results['assigned_records']:,}")
    print(f"Final UCR profiles: {results['final_ucr_profiles']:,}")
    print(
        "Deterministically linked-only profiles: "
        f"{results['deterministic_profiles']:,}"
    )
    print(
        "Probabilistically linked profiles: "
        f"{results['probabilistic_profiles']:,}"
    )
    print(
        "Unresolved singleton profiles: "
        f"{results['unresolved_singletons']:,}"
    )
    print(
        "Anonymous records without a UCR: "
        f"{results['anonymous_unresolvable']:,}"
    )


if __name__ == "__main__":
    main()
