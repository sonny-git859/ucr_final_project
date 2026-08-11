# This script joins the consolidated Silver records to the final operational
# UCR mapping and builds one transparent master profile per assigned UCR. It
# uses explicit survivorship rules, preserves record-level links, aggregates
# system interactions and records attribute-level provenance.
#
# Anonymous records are excluded because they do not have a final UCR ID.
# Protected ground truth and evaluation outputs are never accessed.

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


###############################################################################
# 1. Configuration
###############################################################################

SILVER_INPUT = (
    Path("data")
    / "consolidated_silver"
    / "consolidated_customer_records.csv"
)
FINAL_DIRECTORY = Path("identity_resolution") / "final"
RECORD_MAPPING_INPUT = FINAL_DIRECTORY / "record_to_ucr_mapping.csv"
CLUSTER_SUMMARY_INPUT = FINAL_DIRECTORY / "ucr_cluster_summary.csv"
OUTPUT_DIRECTORY = Path("data") / "gold"

MASTER_FILENAME = "ucr_master_records.csv"
RECORD_LINKS_FILENAME = "ucr_record_links.csv"
PROVENANCE_FILENAME = "ucr_attribute_provenance.csv"
INTERACTION_FILENAME = "ucr_interaction_summary.csv"
SUMMARY_FILENAME = "golden_record_summary.csv"

SOURCE_SYSTEMS = (
    "CRM",
    "ECOMMERCE",
    "MARKETING",
    "ONLINE",
    "SUPPORT",
)

SOURCE_PRIORITY = {
    "CRM": 1,
    "ECOMMERCE": 2,
    "SUPPORT": 3,
    "MARKETING": 4,
    "ONLINE": 5,
}

ASSIGNED_STATUSES = {
    "deterministically_linked",
    "probabilistically_linked",
    "unresolved_singleton",
}

IDENTITY_ATTRIBUTES = [
    "first_name",
    "surname",
    "full_name",
    "date_of_birth",
    "primary_email",
    "telephone_number",
    "address",
    "postcode",
    "preferred_contact_channel",
    "registration_date",
]

SILVER_REQUIRED_COLUMNS = {
    "staging_record_id",
    "source_system",
    "source_record_id",
    "first_name_raw",
    "surname_raw",
    "full_name_raw",
    "email_raw",
    "phone_raw",
    "address_raw",
    "postcode_raw",
    "date_of_birth_raw",
    "email_normalised",
    "phone_normalised",
    "address_normalised",
    "postcode_normalised",
    "date_of_birth_normalised",
    "record_created_date",
    "preferred_contact_channel",
    "transaction_id",
    "transaction_date_time",
    "transaction_total",
    "ticket_quantity",
    "event_id",
    "event_date",
    "session_id",
    "linked_transaction_id",
    "session_date_time_start",
    "session_duration_seconds",
    "basket_created",
    "basket_converted",
    "marketing_contact_id",
    "last_contact_date",
    "consent_status",
    "emails_opened_count",
    "links_clicked_count",
    "support_ticket_id",
    "support_ticket_created_date_time",
    "ticket_status",
    "resolution_time_hours_minutes",
}

MAPPING_REQUIRED_COLUMNS = {
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
}

CLUSTER_REQUIRED_COLUMNS = {
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
}

RECORD_LINK_COLUMNS = [
    "ucr_id",
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
]

MASTER_IDENTITY_COLUMNS = [
    "ucr_id",
    "resolution_status",
    "match_method",
    "cluster_size",
    "deterministic_cluster_count",
    "source_system_count",
    "source_systems",
    "cluster_confidence",
    "cluster_confidence_type",
    "first_name",
    "surname",
    "full_name",
    "date_of_birth",
    "primary_email",
    "telephone_number",
    "address",
    "postcode",
    "preferred_contact_channel",
    "registration_date",
    "identity_attributes_populated",
    "identity_attribute_completeness",
]

PROVENANCE_COLUMNS = [
    "ucr_id",
    "attribute_name",
    "selected_value",
    "source_system",
    "source_record_id",
    "staging_record_id",
    "selection_rule",
    "source_priority",
    "populated_candidate_records",
    "distinct_candidate_values",
]

NAME_RULE = (
    "Prefer a complete name pair; then use source priority "
    "CRM > ECOMMERCE > SUPPORT > MARKETING > ONLINE; then the "
    "most frequent normalised name, latest record and lowest staging ID."
)

DOB_RULE = (
    "Use source priority CRM > ECOMMERCE > SUPPORT > MARKETING > "
    "ONLINE; then the most frequent normalised DOB, latest record and "
    "lowest staging ID."
)

EMAIL_RULE = (
    "Use source priority CRM > ECOMMERCE > SUPPORT > MARKETING > "
    "ONLINE; then the most frequent normalised email, latest record and "
    "lowest staging ID. The selected value is normalised."
)

PHONE_RULE = (
    "Use source priority CRM > ECOMMERCE > SUPPORT > MARKETING > "
    "ONLINE; then the most frequent normalised phone, latest record and "
    "lowest staging ID. The selected display value remains source-based."
)

ADDRESS_RULE = (
    "Prefer a complete address and postcode from one source record; then "
    "use source priority CRM > ECOMMERCE > SUPPORT > MARKETING > ONLINE; "
    "then frequency, latest record and lowest staging ID."
)

CHANNEL_RULE = (
    "Use source priority CRM > ECOMMERCE > SUPPORT > MARKETING > "
    "ONLINE; then frequency, latest record and lowest staging ID."
)

REGISTRATION_RULE = (
    "Use source priority CRM > ECOMMERCE > SUPPORT > MARKETING > "
    "ONLINE; then the earliest populated registration date and lowest "
    "staging ID."
)

NO_CANDIDATE_RULE = "No populated candidate was available."


###############################################################################
# 2. General helper functions
###############################################################################


def clean_value(value: object) -> str:
    # Return a stripped string while treating missing values as blank.
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalise_key(value: object) -> str:
    # Build a conservative comparison key for survivorship voting.
    cleaned = clean_value(value).casefold()
    return re.sub(r"[^a-z0-9]", "", cleaned)


def clean_name(value: object) -> str:
    # Replace system separators and repeated whitespace in display names.
    cleaned = clean_value(value).replace("_", " ")
    return " ".join(cleaned.split())


def parse_full_name(value: object) -> tuple[str, str]:
    # Split a full name only when separate name components are unavailable.
    cleaned = clean_name(value)
    parts = cleaned.split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""

    return parts[0], " ".join(parts[1:])


def parse_boolean(value: object) -> bool:
    # Interpret common source-system boolean representations.
    return clean_value(value).casefold() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
    }


def parse_resolution_hours(value: object) -> float | None:
    # Convert an HH:MM support duration into decimal hours.
    cleaned = clean_value(value)
    if not cleaned:
        return None

    parts = cleaned.split(":")
    if len(parts) != 2:
        return None

    try:
        hours = int(parts[0])
        minutes = int(parts[1])
    except ValueError:
        return None

    if hours < 0 or minutes < 0 or minutes > 59:
        return None

    return hours + (minutes / 60)


def serialise_values(values: Iterable[object]) -> str:
    # Serialise unique populated values in stable ascending order.
    cleaned = {
        clean_value(value)
        for value in values
        if clean_value(value)
    }
    return " | ".join(sorted(cleaned))


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


def validate_protected_column_exclusion(
    dataframe: pd.DataFrame,
    description: str,
) -> None:
    # Prohibit protected labels from entering operational Gold processing.
    protected = [
        column
        for column in dataframe.columns
        if "ground_truth" in column.casefold()
        or "evaluation_split" in column.casefold()
    ]
    if protected:
        raise ValueError(
            f"{description} contains protected columns: "
            + ", ".join(protected)
        )


def add_summary_section(
    rows: list[dict[str, object]],
    section_name: str,
) -> None:
    # Add a readable section header to the two-column summary.
    if rows:
        rows.append({"metric": "", "value": ""})
    rows.append({"metric": section_name, "value": ""})


def safe_min(values: pd.Series) -> str:
    # Return the minimum populated ISO date or datetime value.
    populated = values.map(clean_value)
    populated = populated.loc[populated.ne("")]
    if populated.empty:
        return ""

    return populated.min()


def safe_max(values: pd.Series) -> str:
    # Return the maximum populated ISO date or datetime value.
    populated = values.map(clean_value)
    populated = populated.loc[populated.ne("")]
    if populated.empty:
        return ""

    return populated.max()


###############################################################################
# 3. Command-line configuration
###############################################################################


def parse_arguments() -> argparse.Namespace:
    # Parse command-line input and output paths.
    parser = argparse.ArgumentParser(
        description="Build governed UCR golden records."
    )
    parser.add_argument(
        "--silver-input",
        type=Path,
        default=SILVER_INPUT,
        help=f"Consolidated Silver CSV. Default: {SILVER_INPUT}",
    )
    parser.add_argument(
        "--record-mapping",
        type=Path,
        default=RECORD_MAPPING_INPUT,
        help=f"Final record mapping. Default: {RECORD_MAPPING_INPUT}",
    )
    parser.add_argument(
        "--cluster-summary",
        type=Path,
        default=CLUSTER_SUMMARY_INPUT,
        help=f"Final UCR cluster summary. Default: {CLUSTER_SUMMARY_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help=f"Gold output directory. Default: {OUTPUT_DIRECTORY}",
    )
    return parser.parse_args()


###############################################################################
# 4. Input loading and validation
###############################################################################


def read_csv(path: Path, description: str) -> pd.DataFrame:
    # Load a required CSV without converting identifiers to numbers.
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def validate_unique_references(
    dataframe: pd.DataFrame,
    description: str,
) -> None:
    # Validate record identifiers and source-system references.
    for column in (
        "staging_record_id",
        "source_system",
        "source_record_id",
    ):
        if dataframe[column].map(clean_value).eq("").any():
            raise ValueError(f"{description} has blank {column} values.")

    if dataframe["staging_record_id"].duplicated().any():
        raise ValueError(
            f"{description} contains duplicate staging records."
        )

    duplicate_sources = dataframe.duplicated(
        subset=["source_system", "source_record_id"]
    )
    if duplicate_sources.any():
        raise ValueError(
            f"{description} contains duplicate source references."
        )


def validate_source_systems(dataframe: pd.DataFrame) -> None:
    # Confirm that only the five controlled source systems are present.
    actual = set(dataframe["source_system"].map(clean_value))
    unexpected = sorted(actual - set(SOURCE_SYSTEMS))
    if unexpected:
        raise ValueError(
            "Silver input contains unexpected source systems: "
            + ", ".join(unexpected)
        )


def validate_record_reconciliation(
    silver: pd.DataFrame,
    mapping: pd.DataFrame,
) -> None:
    # Confirm exact one-to-one agreement between Silver and final mapping.
    silver_keys = silver[
        [
            "staging_record_id",
            "source_system",
            "source_record_id",
        ]
    ].sort_values("staging_record_id", kind="stable")
    mapping_keys = mapping[
        [
            "staging_record_id",
            "source_system",
            "source_record_id",
        ]
    ].sort_values("staging_record_id", kind="stable")

    silver_keys = silver_keys.reset_index(drop=True)
    mapping_keys = mapping_keys.reset_index(drop=True)
    if not silver_keys.equals(mapping_keys):
        raise ValueError(
            "Silver records do not reconcile exactly to the final mapping."
        )


def validate_assignment_population(mapping: pd.DataFrame) -> None:
    # Validate assigned and anonymous record treatment.
    assigned = mapping["ucr_id"].map(clean_value).ne("")
    assigned_status = mapping["resolution_status"].isin(
        ASSIGNED_STATUSES
    )
    if not assigned.eq(assigned_status).all():
        raise ValueError(
            "Final UCR assignment is inconsistent with resolution status."
        )

    if mapping.loc[assigned, "probabilistic_cluster_id"].map(
        clean_value
    ).eq("").any():
        raise ValueError("An assigned record has no PCL identifier.")

    anonymous = mapping.loc[~assigned]
    if not anonymous["resolution_status"].eq(
        "anonymous_unresolvable"
    ).all():
        raise ValueError("A non-anonymous record has no UCR assignment.")


def validate_cluster_input(
    mapping: pd.DataFrame,
    clusters: pd.DataFrame,
) -> None:
    # Reconcile cluster-summary profiles to assigned record memberships.
    if clusters["ucr_id"].map(clean_value).eq("").any():
        raise ValueError("The cluster summary contains blank UCR IDs.")
    if not clusters["ucr_id"].is_unique:
        raise ValueError("The cluster summary contains duplicate UCR IDs.")
    if not clusters["probabilistic_cluster_id"].is_unique:
        raise ValueError("The cluster summary contains duplicate PCL IDs.")

    assigned = mapping.loc[mapping["ucr_id"].ne("")]
    if set(clusters["ucr_id"]) != set(assigned["ucr_id"]):
        raise ValueError(
            "The cluster summary does not contain every assigned UCR."
        )

    actual_sizes = assigned["ucr_id"].value_counts().sort_index()
    stated_sizes = pd.to_numeric(
        clusters.set_index("ucr_id")["cluster_size"],
        errors="coerce",
    ).sort_index()
    if stated_sizes.isna().any() or not actual_sizes.eq(
        stated_sizes
    ).all():
        raise ValueError("Cluster-summary record counts are inconsistent.")

    mapping_pairs = assigned[
        ["ucr_id", "probabilistic_cluster_id"]
    ].drop_duplicates()
    cluster_pairs = clusters[
        ["ucr_id", "probabilistic_cluster_id"]
    ].drop_duplicates()
    mapping_pairs = mapping_pairs.sort_values("ucr_id").reset_index(
        drop=True
    )
    cluster_pairs = cluster_pairs.sort_values("ucr_id").reset_index(
        drop=True
    )
    if not mapping_pairs.equals(cluster_pairs):
        raise ValueError("UCR-to-PCL assignments do not reconcile.")


def validate_inputs(
    silver: pd.DataFrame,
    mapping: pd.DataFrame,
    clusters: pd.DataFrame,
) -> None:
    # Run all three operational-input validation groups.
    validate_required_columns(
        silver,
        SILVER_REQUIRED_COLUMNS,
        "Consolidated Silver input",
    )
    validate_required_columns(
        mapping,
        MAPPING_REQUIRED_COLUMNS,
        "Final record mapping",
    )
    validate_required_columns(
        clusters,
        CLUSTER_REQUIRED_COLUMNS,
        "Final UCR cluster summary",
    )

    validate_protected_column_exclusion(silver, "Silver input")
    validate_protected_column_exclusion(mapping, "Record mapping")
    validate_protected_column_exclusion(clusters, "Cluster summary")
    validate_unique_references(silver, "Silver input")
    validate_unique_references(mapping, "Final record mapping")
    validate_source_systems(silver)
    validate_record_reconciliation(silver, mapping)
    validate_assignment_population(mapping)
    validate_cluster_input(mapping, clusters)


###############################################################################
# 5. Operational record links
###############################################################################


def build_linked_records(
    silver: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    # Join mapped UCR identifiers to Silver records and exclude anonymous rows.
    linked = mapping.merge(
        silver,
        on=[
            "staging_record_id",
            "source_system",
            "source_record_id",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_silver"),
    )
    linked = linked.loc[linked["ucr_id"].ne("")].copy()
    return linked.sort_values(
        ["ucr_id", "staging_record_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_record_links(linked: pd.DataFrame) -> pd.DataFrame:
    # Create the auditable bridge from source records to final UCRs.
    return linked[RECORD_LINK_COLUMNS].copy().reset_index(drop=True)


###############################################################################
# 6. Survivorship candidate preparation
###############################################################################


def get_record_rank_date(row: pd.Series) -> str:
    # Select the system-specific timestamp used for stable tie-breaking.
    date_columns = {
        "CRM": "record_created_date",
        "ECOMMERCE": "transaction_date_time",
        "MARKETING": "last_contact_date",
        "ONLINE": "session_date_time_start",
        "SUPPORT": "support_ticket_created_date_time",
    }
    column = date_columns.get(clean_value(row["source_system"]), "")
    if not column:
        return ""

    return clean_value(row[column])


def derive_name_values(row: pd.Series) -> tuple[str, str]:
    # Prefer separate name fields and fall back to a source full name.
    first_name = clean_name(row["first_name_raw"])
    surname = clean_name(row["surname_raw"])
    if first_name or surname:
        return first_name, surname

    return parse_full_name(row["full_name_raw"])


def prepare_survivorship_candidates(
    linked: pd.DataFrame,
) -> pd.DataFrame:
    # Add temporary candidate values and keys used by explicit rules.
    candidates = linked.copy()
    names = candidates.apply(derive_name_values, axis=1)
    candidates["_name_first"] = [value[0] for value in names]
    candidates["_name_surname"] = [value[1] for value in names]
    candidates["_name_key"] = (
        candidates["_name_first"].map(normalise_key)
        + "|"
        + candidates["_name_surname"].map(normalise_key)
    )
    candidates["_name_key"] = candidates["_name_key"].where(
        candidates["_name_key"].ne("|"),
        "",
    )
    candidates["_name_completeness"] = (
        candidates["_name_first"].ne("").astype(int)
        + candidates["_name_surname"].ne("").astype(int)
    )

    candidates["_dob_value"] = candidates[
        "date_of_birth_normalised"
    ].map(clean_value)
    dob_fallback = candidates["date_of_birth_raw"].map(clean_value)
    candidates["_dob_value"] = candidates["_dob_value"].where(
        candidates["_dob_value"].ne(""),
        dob_fallback,
    )
    candidates["_dob_key"] = candidates["_dob_value"].map(
        normalise_key
    )

    candidates["_email_value"] = candidates[
        "email_normalised"
    ].map(clean_value)
    email_fallback = candidates["email_raw"].map(clean_value).str.lower()
    candidates["_email_value"] = candidates["_email_value"].where(
        candidates["_email_value"].ne(""),
        email_fallback,
    )
    candidates["_email_key"] = candidates["_email_value"].map(
        normalise_key
    )

    candidates["_phone_value"] = candidates["phone_raw"].map(
        clean_value
    )
    candidates["_phone_key"] = candidates[
        "phone_normalised"
    ].map(normalise_key)
    phone_fallback = candidates["_phone_value"].map(normalise_key)
    candidates["_phone_key"] = candidates["_phone_key"].where(
        candidates["_phone_key"].ne(""),
        phone_fallback,
    )

    candidates["_address_value"] = candidates["address_raw"].map(
        clean_value
    )
    candidates["_postcode_value"] = candidates["postcode_raw"].map(
        clean_value
    )
    address_key = candidates["address_normalised"].map(normalise_key)
    address_fallback = candidates["_address_value"].map(normalise_key)
    address_key = address_key.where(address_key.ne(""), address_fallback)
    postcode_key = candidates["postcode_normalised"].map(normalise_key)
    postcode_fallback = candidates["_postcode_value"].map(
        normalise_key
    )
    postcode_key = postcode_key.where(
        postcode_key.ne(""),
        postcode_fallback,
    )
    candidates["_address_key"] = address_key + "|" + postcode_key
    candidates["_address_key"] = candidates["_address_key"].where(
        candidates["_address_key"].ne("|"),
        "",
    )
    candidates["_address_completeness"] = (
        candidates["_address_value"].ne("").astype(int)
        + candidates["_postcode_value"].ne("").astype(int)
    )

    candidates["_channel_value"] = candidates[
        "preferred_contact_channel"
    ].map(clean_value)
    candidates["_channel_key"] = candidates["_channel_value"].map(
        normalise_key
    )

    candidates["_registration_value"] = candidates[
        "record_created_date"
    ].map(clean_value)
    candidates["_registration_key"] = candidates[
        "_registration_value"
    ].map(normalise_key)

    candidates["_source_priority"] = candidates["source_system"].map(
        SOURCE_PRIORITY
    )
    candidates["_record_rank_date"] = candidates.apply(
        get_record_rank_date,
        axis=1,
    )
    return candidates


###############################################################################
# 7. Attribute survivorship and provenance
###############################################################################


def select_candidate(
    group: pd.DataFrame,
    key_column: str,
    rule: str,
    completeness_column: str | None = None,
    earliest_value: bool = False,
) -> dict[str, object] | None:
    # Select one candidate through deterministic, documented tie-breaks.
    populated = group.loc[group[key_column].map(clean_value).ne("")].copy()
    if populated.empty:
        return None

    candidate_count = len(populated)
    distinct_count = populated[key_column].nunique()
    if completeness_column:
        maximum = pd.to_numeric(
            populated[completeness_column],
            errors="coerce",
        ).max()
        populated = populated.loc[
            pd.to_numeric(
                populated[completeness_column],
                errors="coerce",
            ).eq(maximum)
        ].copy()

    frequencies = populated[key_column].value_counts()
    populated["_candidate_frequency"] = populated[key_column].map(
        frequencies
    )
    populated["_earliest_sort"] = populated[key_column]
    populated["_date_sort"] = populated["_record_rank_date"]

    if earliest_value:
        sort_columns = [
            "_source_priority",
            "_earliest_sort",
            "staging_record_id",
        ]
        ascending = [True, True, True]
    else:
        sort_columns = [
            "_source_priority",
            "_candidate_frequency",
            "_date_sort",
            "staging_record_id",
        ]
        ascending = [True, False, False, True]

    selected = populated.sort_values(
        sort_columns,
        ascending=ascending,
        kind="stable",
    ).iloc[0]
    return {
        "record": selected,
        "rule": rule,
        "candidate_count": candidate_count,
        "distinct_count": distinct_count,
    }


def selected_value(
    selection: dict[str, object] | None,
    column: str,
) -> str:
    # Return a value from the selected source record or blank.
    if selection is None:
        return ""

    record = selection["record"]
    return clean_value(record[column])


def build_provenance_row(
    ucr_id: str,
    attribute_name: str,
    value: str,
    selection: dict[str, object] | None,
) -> dict[str, object]:
    # Record the source record and rule for one selected master attribute.
    if selection is None or not clean_value(value):
        return {
            "ucr_id": ucr_id,
            "attribute_name": attribute_name,
            "selected_value": "",
            "source_system": "",
            "source_record_id": "",
            "staging_record_id": "",
            "selection_rule": NO_CANDIDATE_RULE,
            "source_priority": "",
            "populated_candidate_records": 0,
            "distinct_candidate_values": 0,
        }

    record = selection["record"]
    return {
        "ucr_id": ucr_id,
        "attribute_name": attribute_name,
        "selected_value": value,
        "source_system": clean_value(record["source_system"]),
        "source_record_id": clean_value(record["source_record_id"]),
        "staging_record_id": clean_value(record["staging_record_id"]),
        "selection_rule": clean_value(selection["rule"]),
        "source_priority": int(record["_source_priority"]),
        "populated_candidate_records": int(
            selection["candidate_count"]
        ),
        "distinct_candidate_values": int(selection["distinct_count"]),
    }


def build_identity_profiles(
    linked: pd.DataFrame,
    clusters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Build selected identity attributes and long-form provenance.
    candidates = prepare_survivorship_candidates(linked)
    cluster_lookup = clusters.set_index("ucr_id")
    master_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []

    for ucr_id, group in candidates.groupby("ucr_id", sort=True):
        name = select_candidate(
            group,
            "_name_key",
            NAME_RULE,
            completeness_column="_name_completeness",
        )
        dob = select_candidate(group, "_dob_key", DOB_RULE)
        email = select_candidate(group, "_email_key", EMAIL_RULE)
        phone = select_candidate(group, "_phone_key", PHONE_RULE)
        address = select_candidate(
            group,
            "_address_key",
            ADDRESS_RULE,
            completeness_column="_address_completeness",
        )
        channel = select_candidate(
            group,
            "_channel_key",
            CHANNEL_RULE,
        )
        registration = select_candidate(
            group,
            "_registration_key",
            REGISTRATION_RULE,
            earliest_value=True,
        )

        first_name = selected_value(name, "_name_first")
        surname = selected_value(name, "_name_surname")
        full_name = " ".join(
            value for value in (first_name, surname) if value
        )
        values = {
            "first_name": first_name,
            "surname": surname,
            "full_name": full_name,
            "date_of_birth": selected_value(dob, "_dob_value"),
            "primary_email": selected_value(email, "_email_value"),
            "telephone_number": selected_value(phone, "_phone_value"),
            "address": selected_value(address, "_address_value"),
            "postcode": selected_value(address, "_postcode_value"),
            "preferred_contact_channel": selected_value(
                channel,
                "_channel_value",
            ),
            "registration_date": selected_value(
                registration,
                "_registration_value",
            ),
        }
        populated = sum(bool(value) for value in values.values())
        cluster = cluster_lookup.loc[ucr_id]
        master_rows.append(
            {
                "ucr_id": ucr_id,
                "resolution_status": cluster["resolution_status"],
                "match_method": cluster["match_method"],
                "cluster_size": int(cluster["cluster_size"]),
                "deterministic_cluster_count": int(
                    cluster["deterministic_cluster_count"]
                ),
                "source_system_count": int(
                    cluster["source_system_count"]
                ),
                "source_systems": cluster["source_systems"],
                "cluster_confidence": cluster["cluster_confidence"],
                "cluster_confidence_type": cluster[
                    "cluster_confidence_type"
                ],
                **values,
                "identity_attributes_populated": populated,
                "identity_attribute_completeness": round(
                    populated / len(IDENTITY_ATTRIBUTES),
                    4,
                ),
            }
        )

        selections = {
            "first_name": name,
            "surname": name,
            "full_name": name,
            "date_of_birth": dob,
            "primary_email": email,
            "telephone_number": phone,
            "address": address,
            "postcode": address,
            "preferred_contact_channel": channel,
            "registration_date": registration,
        }
        for attribute in IDENTITY_ATTRIBUTES:
            provenance_rows.append(
                build_provenance_row(
                    ucr_id,
                    attribute,
                    values[attribute],
                    selections[attribute],
                )
            )

    master = pd.DataFrame(master_rows, columns=MASTER_IDENTITY_COLUMNS)
    provenance = pd.DataFrame(
        provenance_rows,
        columns=PROVENANCE_COLUMNS,
    )
    return master, provenance


###############################################################################
# 8. Interaction aggregation
###############################################################################


def aggregate_ecommerce(linked: pd.DataFrame) -> pd.DataFrame:
    # Aggregate transaction, ticket and event activity by UCR.
    ecommerce = linked.loc[
        linked["source_system"].eq("ECOMMERCE")
    ].copy()
    if ecommerce.empty:
        return pd.DataFrame(columns=["ucr_id"])

    ecommerce["_transaction_total"] = pd.to_numeric(
        ecommerce["transaction_total"],
        errors="coerce",
    )
    ecommerce["_ticket_quantity"] = pd.to_numeric(
        ecommerce["ticket_quantity"],
        errors="coerce",
    )
    grouped = ecommerce.groupby("ucr_id", sort=False)
    output = grouped.agg(
        ecommerce_transaction_count=("staging_record_id", "size"),
        ecommerce_total_revenue=("_transaction_total", "sum"),
        ecommerce_total_tickets=("_ticket_quantity", "sum"),
        ecommerce_average_transaction_value=(
            "_transaction_total",
            "mean",
        ),
        ecommerce_distinct_events=("event_id", "nunique"),
        first_transaction_date_time=("transaction_date_time", safe_min),
        last_transaction_date_time=("transaction_date_time", safe_max),
        first_attended_event_date=("event_date", safe_min),
        last_attended_event_date=("event_date", safe_max),
    ).reset_index()
    output["ecommerce_total_revenue"] = output[
        "ecommerce_total_revenue"
    ].round(2)
    output["ecommerce_total_tickets"] = output[
        "ecommerce_total_tickets"
    ].round(0)
    output["ecommerce_average_transaction_value"] = output[
        "ecommerce_average_transaction_value"
    ].round(2)
    return output


def aggregate_online(linked: pd.DataFrame) -> pd.DataFrame:
    # Aggregate identified online browsing and conversion activity by UCR.
    online = linked.loc[linked["source_system"].eq("ONLINE")].copy()
    if online.empty:
        return pd.DataFrame(columns=["ucr_id"])

    online["_session_duration"] = pd.to_numeric(
        online["session_duration_seconds"],
        errors="coerce",
    )
    online["_basket_created"] = online["basket_created"].map(
        parse_boolean
    ).astype(int)
    online["_basket_converted"] = online["basket_converted"].map(
        parse_boolean
    ).astype(int)
    online["_linked_transaction"] = online[
        "linked_transaction_id"
    ].map(clean_value).ne("").astype(int)
    grouped = online.groupby("ucr_id", sort=False)
    output = grouped.agg(
        online_session_count=("staging_record_id", "size"),
        online_total_duration_seconds=("_session_duration", "sum"),
        online_average_duration_seconds=("_session_duration", "mean"),
        online_baskets_created=("_basket_created", "sum"),
        online_baskets_converted=("_basket_converted", "sum"),
        online_linked_transaction_count=("_linked_transaction", "sum"),
        first_online_session=("session_date_time_start", safe_min),
        last_online_session=("session_date_time_start", safe_max),
    ).reset_index()
    denominator = output["online_baskets_created"].astype(float)
    denominator = denominator.replace(0, float("nan"))
    output["online_basket_conversion_rate"] = (
        output["online_baskets_converted"] / denominator
    ).fillna(0).round(4)
    output["online_total_duration_seconds"] = output[
        "online_total_duration_seconds"
    ].round(0)
    output["online_average_duration_seconds"] = output[
        "online_average_duration_seconds"
    ].round(2)
    return output


def latest_marketing_status(marketing: pd.DataFrame) -> pd.DataFrame:
    # Select the consent state from the latest marketing record per UCR.
    ordered = marketing.sort_values(
        ["ucr_id", "last_contact_date", "staging_record_id"],
        ascending=[True, False, False],
        kind="stable",
    )
    latest = ordered.drop_duplicates("ucr_id", keep="first")
    return latest[["ucr_id", "consent_status"]].rename(
        columns={"consent_status": "latest_marketing_consent_status"}
    )


def aggregate_marketing(linked: pd.DataFrame) -> pd.DataFrame:
    # Aggregate marketing engagement and latest consent state by UCR.
    marketing = linked.loc[
        linked["source_system"].eq("MARKETING")
    ].copy()
    if marketing.empty:
        return pd.DataFrame(columns=["ucr_id"])

    marketing["_emails_opened"] = pd.to_numeric(
        marketing["emails_opened_count"],
        errors="coerce",
    )
    marketing["_links_clicked"] = pd.to_numeric(
        marketing["links_clicked_count"],
        errors="coerce",
    )
    grouped = marketing.groupby("ucr_id", sort=False)
    output = grouped.agg(
        marketing_contact_count=("staging_record_id", "size"),
        marketing_emails_opened=("_emails_opened", "sum"),
        marketing_links_clicked=("_links_clicked", "sum"),
        first_marketing_contact_date=("last_contact_date", safe_min),
        last_marketing_contact_date=("last_contact_date", safe_max),
    ).reset_index()
    latest = latest_marketing_status(marketing)
    return output.merge(latest, on="ucr_id", how="left")


def aggregate_support(linked: pd.DataFrame) -> pd.DataFrame:
    # Aggregate support volume, status and resolution duration by UCR.
    support = linked.loc[linked["source_system"].eq("SUPPORT")].copy()
    if support.empty:
        return pd.DataFrame(columns=["ucr_id"])

    support["_open_ticket"] = support["ticket_status"].map(
        clean_value
    ).str.casefold().eq("open").astype(int)
    support["_resolved_ticket"] = support["ticket_status"].map(
        clean_value
    ).str.casefold().eq("resolved").astype(int)
    support["_resolution_hours"] = support[
        "resolution_time_hours_minutes"
    ].map(parse_resolution_hours)
    grouped = support.groupby("ucr_id", sort=False)
    output = grouped.agg(
        support_ticket_count=("staging_record_id", "size"),
        support_open_ticket_count=("_open_ticket", "sum"),
        support_resolved_ticket_count=("_resolved_ticket", "sum"),
        support_average_resolution_hours=("_resolution_hours", "mean"),
        first_support_ticket=(
            "support_ticket_created_date_time",
            safe_min,
        ),
        last_support_ticket=(
            "support_ticket_created_date_time",
            safe_max,
        ),
    ).reset_index()
    output["support_average_resolution_hours"] = output[
        "support_average_resolution_hours"
    ].round(2)
    return output


def build_overall_activity_dates(linked: pd.DataFrame) -> pd.DataFrame:
    # Derive the first and last non-CRM interaction timestamps per UCR.
    activity = linked.copy()
    date_columns = {
        "ECOMMERCE": "transaction_date_time",
        "MARKETING": "last_contact_date",
        "ONLINE": "session_date_time_start",
        "SUPPORT": "support_ticket_created_date_time",
    }
    activity["_activity_date"] = ""
    for source_system, column in date_columns.items():
        source_rows = activity["source_system"].eq(source_system)
        activity.loc[source_rows, "_activity_date"] = activity.loc[
            source_rows,
            column,
        ].map(clean_value)

    grouped = activity.groupby("ucr_id", sort=False)
    return grouped.agg(
        first_recorded_interaction=("_activity_date", safe_min),
        last_recorded_interaction=("_activity_date", safe_max),
    ).reset_index()


def build_interaction_summary(
    linked: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    # Combine source counts and system-specific interaction aggregates.
    base = clusters[
        ["ucr_id", "cluster_size", "source_system_count", "source_systems"]
    ].copy()
    base = base.rename(
        columns={
            "cluster_size": "total_linked_records",
            "source_system_count": "total_source_systems",
        }
    )

    source_counts = pd.crosstab(
        linked["ucr_id"],
        linked["source_system"],
    ).reindex(columns=SOURCE_SYSTEMS, fill_value=0)
    source_counts = source_counts.rename(
        columns={
            "CRM": "crm_record_count",
            "ECOMMERCE": "ecommerce_record_count",
            "MARKETING": "marketing_record_count",
            "ONLINE": "online_record_count",
            "SUPPORT": "support_record_count",
        }
    ).reset_index()

    output = base.merge(source_counts, on="ucr_id", how="left")
    aggregates = [
        aggregate_ecommerce(linked),
        aggregate_online(linked),
        aggregate_marketing(linked),
        aggregate_support(linked),
        build_overall_activity_dates(linked),
    ]
    for aggregate in aggregates:
        output = output.merge(aggregate, on="ucr_id", how="left")

    count_columns = [
        column
        for column in output.columns
        if column.endswith("_count")
        or column
        in {
            "crm_record_count",
            "ecommerce_record_count",
            "marketing_record_count",
            "online_record_count",
            "support_record_count",
            "ecommerce_total_tickets",
            "online_baskets_created",
            "online_baskets_converted",
            "marketing_emails_opened",
            "marketing_links_clicked",
        }
    ]
    for column in count_columns:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        ).fillna(0).astype(int)

    zero_float_columns = [
        "ecommerce_total_revenue",
        "online_total_duration_seconds",
        "online_basket_conversion_rate",
    ]
    for column in zero_float_columns:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        ).fillna(0)

    text_columns = [
        "first_transaction_date_time",
        "last_transaction_date_time",
        "first_attended_event_date",
        "last_attended_event_date",
        "first_online_session",
        "last_online_session",
        "first_marketing_contact_date",
        "last_marketing_contact_date",
        "latest_marketing_consent_status",
        "first_support_ticket",
        "last_support_ticket",
        "first_recorded_interaction",
        "last_recorded_interaction",
    ]
    for column in text_columns:
        output[column] = output[column].fillna("")

    return output.sort_values("ucr_id", kind="stable").reset_index(
        drop=True
    )


###############################################################################
# 9. Master-record construction
###############################################################################


def build_master_records(
    identity_profiles: pd.DataFrame,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    # Combine selected identity attributes with aggregated UCR behaviour.
    interaction_fields = interactions.drop(columns=["source_systems"])
    master = identity_profiles.merge(
        interaction_fields,
        on="ucr_id",
        how="left",
        validate="one_to_one",
    )
    return master.sort_values("ucr_id", kind="stable").reset_index(
        drop=True
    )


###############################################################################
# 10. Output validation
###############################################################################


def validate_record_links(
    linked: pd.DataFrame,
    record_links: pd.DataFrame,
) -> None:
    # Confirm that links contain every assigned record exactly once.
    if len(record_links) != len(linked):
        raise ValueError("Record-link rows do not reconcile to input.")
    if not record_links["staging_record_id"].is_unique:
        raise ValueError("Record links contain duplicate staging IDs.")
    if record_links["ucr_id"].map(clean_value).eq("").any():
        raise ValueError("An anonymous record entered UCR record links.")
    if set(record_links["staging_record_id"]) != set(
        linked["staging_record_id"]
    ):
        raise ValueError("Record-link membership is inconsistent.")


def validate_master_profiles(
    master: pd.DataFrame,
    clusters: pd.DataFrame,
) -> None:
    # Confirm one complete master row per assigned UCR profile.
    missing_columns = sorted(
        set(MASTER_IDENTITY_COLUMNS) - set(master.columns)
    )
    if missing_columns:
        raise ValueError(
            "Golden master records are missing fields: "
            + ", ".join(missing_columns)
        )
    suffix_columns = [
        column
        for column in master.columns
        if column.endswith("_x") or column.endswith("_y")
    ]
    if suffix_columns:
        raise ValueError(
            "Golden master records contain ambiguous merge fields: "
            + ", ".join(suffix_columns)
        )
    if not master["ucr_id"].is_unique:
        raise ValueError("Golden master records contain duplicate UCR IDs.")
    if set(master["ucr_id"]) != set(clusters["ucr_id"]):
        raise ValueError("Golden master records omit assigned UCR IDs.")

    unresolved = master.loc[
        master["resolution_status"].eq("unresolved_singleton")
    ]
    unresolved_sizes = pd.to_numeric(
        unresolved["cluster_size"],
        errors="coerce",
    )
    if unresolved_sizes.isna().any() or not unresolved_sizes.eq(1).all():
        raise ValueError(
            "An unresolved singleton was not retained as one profile."
        )

    if master["identity_attribute_completeness"].lt(0).any():
        raise ValueError("An identity completeness score is below zero.")
    if master["identity_attribute_completeness"].gt(1).any():
        raise ValueError("An identity completeness score exceeds one.")


def validate_provenance(
    master: pd.DataFrame,
    provenance: pd.DataFrame,
    record_links: pd.DataFrame,
) -> None:
    # Validate complete, referential and coherent attribute provenance.
    expected_rows = len(master) * len(IDENTITY_ATTRIBUTES)
    if len(provenance) != expected_rows:
        raise ValueError("Attribute provenance has an incorrect row count.")
    if provenance.duplicated(
        subset=["ucr_id", "attribute_name"]
    ).any():
        raise ValueError("Attribute provenance contains duplicate fields.")

    expected_pairs = pd.MultiIndex.from_product(
        [master["ucr_id"], IDENTITY_ATTRIBUTES],
        names=["ucr_id", "attribute_name"],
    )
    actual_pairs = pd.MultiIndex.from_frame(
        provenance[["ucr_id", "attribute_name"]]
    )
    if set(expected_pairs) != set(actual_pairs):
        raise ValueError("Attribute provenance coverage is incomplete.")

    master_values = master.set_index("ucr_id")[IDENTITY_ATTRIBUTES]
    pivoted = provenance.pivot(
        index="ucr_id",
        columns="attribute_name",
        values="selected_value",
    ).reindex(columns=IDENTITY_ATTRIBUTES)
    master_values = master_values.fillna("").astype(str).sort_index()
    pivoted = pivoted.fillna("").astype(str).sort_index()
    if not master_values.equals(pivoted):
        raise ValueError("Provenance values do not match master values.")

    populated = provenance["selected_value"].map(clean_value).ne("")
    populated_provenance = provenance.loc[populated]
    links = record_links[
        [
            "ucr_id",
            "staging_record_id",
            "source_system",
            "source_record_id",
        ]
    ]
    checked = populated_provenance.merge(
        links,
        on=[
            "ucr_id",
            "staging_record_id",
            "source_system",
            "source_record_id",
        ],
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    if not checked["_merge"].eq("both").all():
        raise ValueError(
            "A provenance source record is not linked to its UCR."
        )

    for attributes in (
        ("first_name", "surname", "full_name"),
        ("address", "postcode"),
    ):
        subset = provenance.loc[
            provenance["attribute_name"].isin(attributes)
            & provenance["selected_value"].map(clean_value).ne("")
        ]
        counts = subset.groupby("ucr_id")["staging_record_id"].nunique()
        if counts.gt(1).any():
            raise ValueError(
                "A coherent attribute bundle uses multiple source records."
            )


def validate_interactions(
    linked: pd.DataFrame,
    interactions: pd.DataFrame,
) -> None:
    # Reconcile UCR activity counts and key numeric totals to source rows.
    if not interactions["ucr_id"].is_unique:
        raise ValueError("Interaction summary contains duplicate UCR IDs.")
    if set(interactions["ucr_id"]) != set(linked["ucr_id"]):
        raise ValueError("Interaction summary omits assigned UCR IDs.")

    count_map = {
        "CRM": "crm_record_count",
        "ECOMMERCE": "ecommerce_record_count",
        "MARKETING": "marketing_record_count",
        "ONLINE": "online_record_count",
        "SUPPORT": "support_record_count",
    }
    for source_system, column in count_map.items():
        expected = int(linked["source_system"].eq(source_system).sum())
        actual = int(interactions[column].sum())
        if actual != expected:
            raise ValueError(
                f"Interaction count does not reconcile for {source_system}."
            )

    source_count_columns = list(count_map.values())
    stated_total = pd.to_numeric(
        interactions["total_linked_records"],
        errors="coerce",
    )
    calculated_total = interactions[source_count_columns].sum(axis=1)
    if stated_total.isna().any() or not stated_total.eq(
        calculated_total
    ).all():
        raise ValueError("Per-UCR source counts do not reconcile.")

    expected_revenue = pd.to_numeric(
        linked.loc[
            linked["source_system"].eq("ECOMMERCE"),
            "transaction_total",
        ],
        errors="coerce",
    ).sum()
    actual_revenue = interactions["ecommerce_total_revenue"].sum()
    if round(expected_revenue, 2) != round(actual_revenue, 2):
        raise ValueError("E-commerce revenue does not reconcile.")


def validate_protected_outputs(
    outputs: Iterable[pd.DataFrame],
) -> None:
    # Confirm that no protected columns were introduced into Gold outputs.
    for dataframe in outputs:
        validate_protected_column_exclusion(dataframe, "Gold output")


def validate_outputs(
    linked: pd.DataFrame,
    clusters: pd.DataFrame,
    master: pd.DataFrame,
    record_links: pd.DataFrame,
    provenance: pd.DataFrame,
    interactions: pd.DataFrame,
) -> None:
    # Run every Gold-layer output validation.
    validate_record_links(linked, record_links)
    validate_master_profiles(master, clusters)
    validate_provenance(master, provenance, record_links)
    validate_interactions(linked, interactions)
    validate_protected_outputs(
        [master, record_links, provenance, interactions]
    )


###############################################################################
# 11. Golden-record summary
###############################################################################


def build_golden_summary(
    silver: pd.DataFrame,
    mapping: pd.DataFrame,
    clusters: pd.DataFrame,
    master: pd.DataFrame,
    record_links: pd.DataFrame,
    provenance: pd.DataFrame,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    # Build a two-column Gold construction and validation report.
    rows: list[dict[str, object]] = []
    anonymous_count = int(mapping["ucr_id"].eq("").sum())
    unresolved_count = int(
        master["resolution_status"].eq("unresolved_singleton").sum()
    )
    source_counts = record_links["source_system"].value_counts()

    add_summary_section(rows, "INPUT VALIDATION")
    rows.extend(
        [
            {"metric": "silver_input_records", "value": len(silver)},
            {
                "metric": "mapping_input_records",
                "value": len(mapping),
            },
            {
                "metric": "input_ucr_profiles",
                "value": len(clusters),
            },
            {
                "metric": "silver_mapping_reconciliation",
                "value": True,
            },
            {
                "metric": "protected_columns_excluded",
                "value": True,
            },
        ]
    )

    add_summary_section(rows, "GOLDEN RECORD CONSTRUCTION")
    rows.extend(
        [
            {
                "metric": "golden_master_records",
                "value": len(master),
            },
            {
                "metric": "record_links",
                "value": len(record_links),
            },
            {
                "metric": "attribute_provenance_rows",
                "value": len(provenance),
            },
            {
                "metric": "interaction_summary_rows",
                "value": len(interactions),
            },
            {
                "metric": "unresolved_singleton_profiles_retained",
                "value": unresolved_count,
            },
            {
                "metric": "anonymous_records_excluded",
                "value": anonymous_count,
            },
        ]
    )

    add_summary_section(rows, "ASSIGNED RECORDS BY SOURCE")
    for source_system in SOURCE_SYSTEMS:
        rows.append(
            {
                "metric": f"{source_system.casefold()}_assigned_records",
                "value": int(source_counts.get(source_system, 0)),
            }
        )

    add_summary_section(rows, "IDENTITY COMPLETENESS")
    rows.extend(
        [
            {
                "metric": "mean_identity_attribute_completeness",
                "value": round(
                    master["identity_attribute_completeness"].mean(),
                    4,
                ),
            },
            {
                "metric": "profiles_with_name",
                "value": int(master["full_name"].ne("").sum()),
            },
            {
                "metric": "profiles_with_email",
                "value": int(master["primary_email"].ne("").sum()),
            },
            {
                "metric": "profiles_with_phone",
                "value": int(master["telephone_number"].ne("").sum()),
            },
            {
                "metric": "profiles_with_address",
                "value": int(master["address"].ne("").sum()),
            },
        ]
    )

    add_summary_section(rows, "INTERACTION TOTALS")
    rows.extend(
        [
            {
                "metric": "ecommerce_transactions",
                "value": int(
                    interactions["ecommerce_transaction_count"].sum()
                ),
            },
            {
                "metric": "ecommerce_revenue",
                "value": round(
                    interactions["ecommerce_total_revenue"].sum(),
                    2,
                ),
            },
            {
                "metric": "identified_online_sessions",
                "value": int(interactions["online_session_count"].sum()),
            },
            {
                "metric": "marketing_contacts",
                "value": int(
                    interactions["marketing_contact_count"].sum()
                ),
            },
            {
                "metric": "support_tickets",
                "value": int(interactions["support_ticket_count"].sum()),
            },
        ]
    )

    add_summary_section(rows, "OUTPUT VALIDATION")
    rows.extend(
        [
            {
                "metric": "one_master_record_per_ucr",
                "value": True,
            },
            {
                "metric": "all_assigned_records_linked",
                "value": True,
            },
            {
                "metric": "all_master_attributes_have_provenance",
                "value": True,
            },
            {
                "metric": "name_bundles_are_coherent",
                "value": True,
            },
            {
                "metric": "address_bundles_are_coherent",
                "value": True,
            },
            {
                "metric": "interaction_counts_reconcile",
                "value": True,
            },
        ]
    )
    return pd.DataFrame(rows, columns=["metric", "value"])


###############################################################################
# 12. Output writing
###############################################################################


def write_outputs(
    output_directory: Path,
    master: pd.DataFrame,
    record_links: pd.DataFrame,
    provenance: pd.DataFrame,
    interactions: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    # Write all validated Gold-layer CSV files.
    output_directory.mkdir(parents=True, exist_ok=True)
    master.to_csv(
        output_directory / MASTER_FILENAME,
        index=False,
    )
    record_links.to_csv(
        output_directory / RECORD_LINKS_FILENAME,
        index=False,
    )
    provenance.to_csv(
        output_directory / PROVENANCE_FILENAME,
        index=False,
    )
    interactions.to_csv(
        output_directory / INTERACTION_FILENAME,
        index=False,
    )
    summary.to_csv(
        output_directory / SUMMARY_FILENAME,
        index=False,
    )


###############################################################################
# 13. Main execution
###############################################################################


def main() -> None:
    # Load inputs, build governed Gold outputs and report completion.
    arguments = parse_arguments()
    silver = read_csv(
        arguments.silver_input,
        "Consolidated Silver input",
    )
    mapping = read_csv(
        arguments.record_mapping,
        "Final record mapping",
    )
    clusters = read_csv(
        arguments.cluster_summary,
        "Final UCR cluster summary",
    )
    validate_inputs(silver, mapping, clusters)

    linked = build_linked_records(silver, mapping)
    record_links = build_record_links(linked)
    identity_profiles, provenance = build_identity_profiles(
        linked,
        clusters,
    )
    interactions = build_interaction_summary(linked, clusters)
    master = build_master_records(identity_profiles, interactions)
    validate_outputs(
        linked,
        clusters,
        master,
        record_links,
        provenance,
        interactions,
    )
    summary = build_golden_summary(
        silver,
        mapping,
        clusters,
        master,
        record_links,
        provenance,
        interactions,
    )
    write_outputs(
        arguments.output_dir,
        master,
        record_links,
        provenance,
        interactions,
        summary,
    )

    print("Golden UCR record construction completed successfully.")
    print(f"Silver input: {arguments.silver_input.resolve()}")
    print(f"Record mapping: {arguments.record_mapping.resolve()}")
    print(f"Cluster summary: {arguments.cluster_summary.resolve()}")
    print(f"Output directory: {arguments.output_dir.resolve()}")
    print(f"Input records: {len(silver):,}")
    print(f"Records linked to UCRs: {len(record_links):,}")
    print(f"Anonymous records excluded: {mapping['ucr_id'].eq('').sum():,}")
    print(f"Golden master records: {len(master):,}")
    print(
        "Unresolved singleton profiles retained: "
        f"{master['resolution_status'].eq('unresolved_singleton').sum():,}"
    )
    print(f"Attribute provenance rows: {len(provenance):,}")


if __name__ == "__main__":
    main()
