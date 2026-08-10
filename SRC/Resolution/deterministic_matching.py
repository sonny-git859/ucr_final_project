###############################################################################
# Imports
###############################################################################


from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


###############################################################################
# 1. Configuration
###############################################################################

INPUT_FILENAME = "consolidated_customer_records.csv"
OUTPUT_DIRECTORY = Path("identity_resolution") / "deterministic"

PAIRWISE_MATCHES_FILENAME = "deterministic_pairwise_matches.csv"
RECORD_MAPPING_FILENAME = "deterministic_record_mapping.csv"
REJECTED_MATCHES_FILENAME = "deterministic_rejected_matches.csv"
UNRESOLVED_RECORDS_FILENAME = "deterministic_unresolved_records.csv"
SUMMARY_FILENAME = "deterministic_matching_summary.csv"

EXPECTED_SOURCE_SYSTEMS = {
    "CRM",
    "ECOMMERCE",
    "MARKETING",
    "ONLINE",
    "SUPPORT",
}

IDENTITY_FIELDS = (
    "first_name_normalised",
    "surname_normalised",
    "full_name_normalised",
    "email_normalised",
    "phone_normalised",
    "address_normalised",
    "postcode_normalised",
    "date_of_birth_normalised",
)

RAW_NORMALISED_PAIRS = (
    ("first_name_raw", "first_name_normalised"),
    ("surname_raw", "surname_normalised"),
    ("full_name_raw", "full_name_normalised"),
    ("email_raw", "email_normalised"),
    ("phone_raw", "phone_normalised"),
    ("address_raw", "address_normalised"),
    ("postcode_raw", "postcode_normalised"),
    ("date_of_birth_raw", "date_of_birth_normalised"),
)

REQUIRED_COLUMNS = {
    "staging_record_id",
    "source_system",
    "source_record_id",
    "transaction_id",
    "portal_user_id",
    "linked_transaction_id",
    *(raw for raw, _ in RAW_NORMALISED_PAIRS),
    *IDENTITY_FIELDS,
}

PLACEHOLDER_VALUES = {
    "",
    "n/a",
    "na",
    "nan",
    "none",
    "not available",
    "null",
    "unknown",
}


###############################################################################
# 2. Deterministic rule definitions
###############################################################################


@dataclass(frozen=True)
class DeterministicRule:
    priority: int
    code: str
    name: str
    fields: tuple[str, ...]
    authoritative: bool = False
    relationship_rule: bool = False
    source_systems: tuple[str, ...] | None = None


RULES = (
    DeterministicRule(
        priority=1,
        code="D01",
        name="Linked transaction ID to transaction ID",
        fields=("linked_transaction_id", "transaction_id"),
        authoritative=True,
        relationship_rule=True,
    ),
    DeterministicRule(
        priority=2,
        code="D02",
        name="Exact portal user ID",
        fields=("portal_user_id",),
        authoritative=True,
    ),
    DeterministicRule(
        priority=3,
        code="D03",
        name="Exact email and full name",
        fields=("email_normalised", "full_name_normalised"),
    ),
    DeterministicRule(
        priority=4,
        code="D04",
        name="Exact email and phone",
        fields=("email_normalised", "phone_normalised"),
    ),
    DeterministicRule(
        priority=5,
        code="D05",
        name="Exact email and postcode",
        fields=("email_normalised", "postcode_normalised"),
    ),
    DeterministicRule(
        priority=6,
        code="D06",
        name="Exact phone and full name",
        fields=("phone_normalised", "full_name_normalised"),
    ),
    DeterministicRule(
        priority=7,
        code="D07",
        name="Exact full name and address",
        fields=("full_name_normalised", "address_normalised"),
    ),
    DeterministicRule(
        priority=8,
        code="D08",
        name="Exact full name and postcode",
        fields=("full_name_normalised", "postcode_normalised"),
    ),
    DeterministicRule(
        priority=9,
        code="D09",
        name="Exact full name and date of birth",
        fields=("full_name_normalised", "date_of_birth_normalised"),
        source_systems=("CRM",),
    ),
    DeterministicRule(
        priority=10,
        code="D10",
        name="Exact surname, postcode and date of birth",
        fields=(
            "surname_normalised",
            "postcode_normalised",
            "date_of_birth_normalised",
        ),
        source_systems=("CRM",),
    ),
)


###############################################################################
# 3. Union-find cluster structure
###############################################################################


class UnionFind:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        record_count = len(dataframe)
        self.parent = list(range(record_count))
        self.size = [1] * record_count
        self.portal_ids: list[set[str]] = []
        self.dates_of_birth: list[set[str]] = []

        for row in dataframe.itertuples(index=False):
            portal_id = clean_value(row.portal_user_id)
            date_of_birth = clean_value(row.date_of_birth_normalised)
            self.portal_ids.append({portal_id} if portal_id else set())
            self.dates_of_birth.append(
                {date_of_birth} if date_of_birth else set()
            )

    def find(self, item: int) -> int:
        # Return the root for an item using path compression.
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]

        return item

    def union(self, left: int, right: int) -> tuple[int, bool]:
        # Union two roots and return the resulting root and merge status.
        left_root = self.find(left)
        right_root = self.find(right)

        if left_root == right_root:
            return left_root, False

        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root

        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        self.portal_ids[left_root].update(self.portal_ids[right_root])
        self.dates_of_birth[left_root].update(
            self.dates_of_birth[right_root]
        )

        return left_root, True


###############################################################################
# 4. General helper functions
###############################################################################


def clean_value(value: object) -> str:
    # Return a stripped value, or an empty string for a placeholder.
    if value is None or pd.isna(value):
        return ""

    cleaned = str(value).strip()
    if cleaned.casefold() in PLACEHOLDER_VALUES:
        return ""

    return cleaned


def canonical_value(value: object) -> str:
    # Return a case-insensitive value suitable for an exact-match key.
    return clean_value(value).casefold()


def serialise_values(fields: Sequence[str], values: Sequence[str]) -> str:
    # Serialise matched fields and values for transparent CSV evidence.
    evidence = dict(zip(fields, values))
    return json.dumps(evidence, ensure_ascii=False, sort_keys=True)


def locate_default_input() -> Path:
    # Locate a single conventional consolidated input file.
    direct_candidates = (
        Path(INPUT_FILENAME),
        Path("consolidated_silver") / INPUT_FILENAME,
        Path("data") / "consolidated_silver" / INPUT_FILENAME,
        Path("data") / "processed" / "consolidated_silver"
        / INPUT_FILENAME,
    )

    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    discovered = sorted(Path.cwd().rglob(INPUT_FILENAME))
    if len(discovered) == 1:
        return discovered[0]

    if len(discovered) > 1:
        locations = "\n".join(f"  - {path}" for path in discovered)
        raise FileNotFoundError(
            "More than one consolidated input file was found. Supply the "
            f"correct file with --input:\n{locations}"
        )

    raise FileNotFoundError(
        f"Could not find {INPUT_FILENAME}. Supply its path with --input."
    )


def parse_arguments() -> argparse.Namespace:
    # Parse command-line paths.
    parser = argparse.ArgumentParser(
        description=(
            "Apply deterministic identity-resolution rules and create "
            "provisional customer clusters."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Consolidated Silver CSV. If omitted, the script searches "
            "conventional project locations."
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
# 5. Input loading and validation
###############################################################################


def load_input(input_path: Path) -> pd.DataFrame:
    # Load the consolidated CSV without converting identifiers to numbers.
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    dataframe = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    return dataframe


def validate_input(dataframe: pd.DataFrame) -> None:
    # Validate identifiers, source systems and ground-truth separation.
    missing_columns = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "The consolidated input is missing required columns: "
            + ", ".join(missing_columns)
        )

    ground_truth_columns = [
        column
        for column in dataframe.columns
        if "ground_truth" in column.casefold()
    ]
    if ground_truth_columns:
        raise ValueError(
            "Ground-truth columns must not enter identity resolution: "
            + ", ".join(ground_truth_columns)
        )

    for identifier in ("staging_record_id", "source_record_id"):
        cleaned = dataframe[identifier].map(clean_value)
        if cleaned.eq("").any():
            raise ValueError(f"{identifier} contains blank values.")
        if cleaned.duplicated().any():
            duplicate_count = int(cleaned.duplicated(keep=False).sum())
            raise ValueError(
                f"{identifier} contains {duplicate_count} duplicate rows."
            )

    observed_sources = set(dataframe["source_system"].map(clean_value))
    unexpected_sources = sorted(observed_sources - EXPECTED_SOURCE_SYSTEMS)
    missing_sources = sorted(EXPECTED_SOURCE_SYSTEMS - observed_sources)
    if unexpected_sources or missing_sources:
        raise ValueError(
            "Source-system validation failed. "
            f"Unexpected: {unexpected_sources or 'none'}; "
            f"missing: {missing_sources or 'none'}."
        )

    validate_transaction_references(dataframe)
    validate_normalised_identity_coverage(dataframe)


def validate_transaction_references(dataframe: pd.DataFrame) -> None:
    # Confirm linked online transactions resolve to one e-commerce row.
    ecommerce_mask = dataframe["source_system"].eq("ECOMMERCE")
    ecommerce_ids = dataframe.loc[ecommerce_mask, "transaction_id"].map(
        canonical_value
    )

    if ecommerce_ids.eq("").any():
        raise ValueError(
            "Every ECOMMERCE record must contain a transaction_id."
        )
    if ecommerce_ids.duplicated().any():
        raise ValueError(
            "ECOMMERCE transaction_id values must be unique."
        )

    linked_mask = dataframe["linked_transaction_id"].map(
        clean_value
    ).ne("")
    invalid_sources = dataframe.loc[
        linked_mask & ~dataframe["source_system"].eq("ONLINE")
    ]
    if not invalid_sources.empty:
        raise ValueError(
            "linked_transaction_id is populated outside ONLINE records."
        )

    valid_transaction_ids = set(ecommerce_ids)
    linked_ids = dataframe.loc[
        linked_mask, "linked_transaction_id"
    ].map(canonical_value)
    invalid_ids = sorted(set(linked_ids) - valid_transaction_ids)
    if invalid_ids:
        preview = ", ".join(invalid_ids[:10])
        raise ValueError(
            f"{len(invalid_ids)} linked transaction IDs do not resolve. "
            f"Examples: {preview}"
        )


def validate_normalised_identity_coverage(
    dataframe: pd.DataFrame,
) -> None:
    # Confirm every populated raw identity value has a normalised value.
    failures: list[str] = []
    for raw_field, normalised_field in RAW_NORMALISED_PAIRS:
        raw_present = dataframe[raw_field].map(clean_value).ne("")
        normalised_blank = dataframe[normalised_field].map(
            clean_value
        ).eq("")
        failure_count = int((raw_present & normalised_blank).sum())
        if failure_count:
            failures.append(
                f"{raw_field} -> {normalised_field}: {failure_count} rows"
            )

    if failures:
        raise ValueError(
            "Normalised identity coverage validation failed: "
            + "; ".join(failures)
        )


###############################################################################
# 6. Matching population and candidate generation
###############################################################################


def identify_anonymous_online_records(
    dataframe: pd.DataFrame,
) -> pd.Series:
    # Identify online records with no reliable identity evidence.
    online_mask = dataframe["source_system"].eq("ONLINE")
    no_portal = dataframe["portal_user_id"].map(clean_value).eq("")
    no_transaction = dataframe["linked_transaction_id"].map(
        clean_value
    ).eq("")

    no_identity = pd.Series(True, index=dataframe.index)
    for field in IDENTITY_FIELDS:
        no_identity &= dataframe[field].map(clean_value).eq("")

    return online_mask & no_portal & no_transaction & no_identity


def grouped_candidates(
    dataframe: pd.DataFrame,
    eligible_indices: set[int],
    fields: Sequence[str],
) -> Iterable[tuple[int, int, tuple[str, ...]]]:
    # Yield record pairs sharing complete, non-placeholder exact keys.
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)

    for index in sorted(eligible_indices):
        values = tuple(
            canonical_value(dataframe.at[index, field])
            for field in fields
        )
        if all(values):
            groups[values].append(index)

    staging_ids = dataframe["staging_record_id"].to_dict()
    for values in sorted(groups):
        indices = groups[values]
        if len(indices) < 2:
            continue

        indices.sort(key=lambda item: staging_ids[item])
        for left, right in combinations(indices, 2):
            yield left, right, values


def relationship_candidates(
    dataframe: pd.DataFrame,
    eligible_indices: set[int],
) -> Iterable[tuple[int, int, tuple[str, ...]]]:
    # Yield online-to-e-commerce explicit transaction relationships.
    ecommerce_lookup: dict[str, int] = {}
    ecommerce_mask = dataframe["source_system"].eq("ECOMMERCE")

    for index in dataframe.index[ecommerce_mask]:
        transaction_id = canonical_value(
            dataframe.at[index, "transaction_id"]
        )
        ecommerce_lookup[transaction_id] = index

    online_mask = dataframe["source_system"].eq("ONLINE")
    staging_ids = dataframe["staging_record_id"].to_dict()
    online_indices = sorted(
        dataframe.index[online_mask],
        key=lambda item: staging_ids[item],
    )

    for online_index in online_indices:
        linked_id = canonical_value(
            dataframe.at[online_index, "linked_transaction_id"]
        )
        if not linked_id:
            continue

        ecommerce_index = ecommerce_lookup[linked_id]
        if (
            online_index in eligible_indices
            and ecommerce_index in eligible_indices
        ):
            yield online_index, ecommerce_index, (linked_id, linked_id)


###############################################################################
# 7. Conflict checks and ordered deterministic matching
###############################################################################


def find_cluster_conflict(
    union_find: UnionFind,
    left: int,
    right: int,
    rule: DeterministicRule,
) -> str:
    # Return a hard-conflict reason, or an empty string if mergeable.
    left_root = union_find.find(left)
    right_root = union_find.find(right)

    left_portals = union_find.portal_ids[left_root]
    right_portals = union_find.portal_ids[right_root]
    if left_portals and right_portals:
        if left_portals.isdisjoint(right_portals):
            return "Conflicting trusted portal_user_id values"

    if rule.authoritative:
        return ""

    left_dobs = union_find.dates_of_birth[left_root]
    right_dobs = union_find.dates_of_birth[right_root]
    if left_dobs and right_dobs and left_dobs.isdisjoint(right_dobs):
        return "Conflicting known date_of_birth_normalised values"

    # Same-source membership is not itself a conflict. A customer may have
    # several transactions, sessions or support tickets, and duplicate CRM
    # accounts were introduced deliberately.

    # Portal and DOB contradictions provide the cluster-level hard safeguards.
    return ""


def apply_deterministic_rules(
    dataframe: pd.DataFrame,
    anonymous_mask: pd.Series,
) -> tuple[
    UnionFind,
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    # Apply rules in priority order and return decisions and statistics.
    union_find = UnionFind(dataframe)
    eligible_indices = set(dataframe.index[~anonymous_mask])
    accepted_matches: list[dict[str, object]] = []
    rejected_matches: list[dict[str, object]] = []
    rule_statistics: list[dict[str, object]] = []

    for rule in RULES:
        if rule.relationship_rule:
            candidates = relationship_candidates(
                dataframe,
                eligible_indices,
            )
            evidence_fields = (
                "linked_transaction_id",
                "transaction_id",
            )
        else:
            rule_eligible_indices = eligible_indices
            if rule.source_systems:
                source_mask = dataframe["source_system"].isin(
                    rule.source_systems
                )
                rule_eligible_indices = set(
                    dataframe.index[source_mask]
                ) & eligible_indices
            candidates = grouped_candidates(
                dataframe,
                rule_eligible_indices,
                rule.fields,
            )
            evidence_fields = rule.fields

        candidate_count = 0
        accepted_count = 0
        rejected_count = 0
        already_linked_count = 0

        for left, right, values in candidates:
            candidate_count += 1
            left_root = union_find.find(left)
            right_root = union_find.find(right)

            if left_root == right_root:
                already_linked_count += 1
                continue

            rejection_reason = find_cluster_conflict(
                union_find,
                left,
                right,
                rule,
            )
            evidence = serialise_values(evidence_fields, values)

            base_decision = {
                "record_id_1": dataframe.at[
                    left, "staging_record_id"
                ],
                "source_system_1": dataframe.at[left, "source_system"],
                "source_record_id_1": dataframe.at[
                    left, "source_record_id"
                ],
                "record_id_2": dataframe.at[
                    right, "staging_record_id"
                ],
                "source_system_2": dataframe.at[right, "source_system"],
                "source_record_id_2": dataframe.at[
                    right, "source_record_id"
                ],
                "match_method": "Deterministic",
                "rule_priority": rule.priority,
                "rule_code": rule.code,
                "match_rule": rule.name,
                "matched_fields": " | ".join(evidence_fields),
                "matched_values": evidence,
                "match_score": 1.0,
            }

            if rejection_reason:
                rejected_count += 1
                rejected_matches.append(
                    {
                        **base_decision,
                        "rejection_reason": rejection_reason,
                    }
                )
                continue

            union_find.union(left, right)
            accepted_count += 1
            accepted_matches.append(base_decision)

        rule_statistics.append(
            {
                "rule": rule,
                "candidate_pairs": candidate_count,
                "accepted_merges": accepted_count,
                "rejected_merges": rejected_count,
                "already_linked_pairs": already_linked_count,
            }
        )

    return (
        union_find,
        accepted_matches,
        rejected_matches,
        rule_statistics,
    )


###############################################################################
# 8. Provisional mapping and output construction
###############################################################################


def build_record_mapping(
    dataframe: pd.DataFrame,
    anonymous_mask: pd.Series,
    union_find: UnionFind,
    accepted_matches: list[dict[str, object]],
) -> pd.DataFrame:
    # Assign reproducible DCL identifiers and resolution statuses.
    eligible_indices = list(dataframe.index[~anonymous_mask])
    clusters: dict[int, list[int]] = defaultdict(list)
    for index in eligible_indices:
        clusters[union_find.find(index)].append(index)

    staging_ids = dataframe["staging_record_id"].to_dict()
    ordered_clusters = sorted(
        clusters.values(),
        key=lambda members: min(staging_ids[index] for index in members),
    )

    index_to_cluster: dict[int, str] = {}
    cluster_sizes: dict[str, int] = {}
    for cluster_number, members in enumerate(ordered_clusters, start=1):
        cluster_id = f"DCL{cluster_number:06d}"
        cluster_sizes[cluster_id] = len(members)
        for index in members:
            index_to_cluster[index] = cluster_id

    rules_by_cluster: dict[str, set[str]] = defaultdict(set)
    staging_to_index = {
        staging_id: index for index, staging_id in staging_ids.items()
    }
    for match in accepted_matches:
        left_index = staging_to_index[str(match["record_id_1"])]
        cluster_id = index_to_cluster[left_index]
        rules_by_cluster[cluster_id].add(str(match["rule_code"]))

    mapping_rows: list[dict[str, object]] = []
    for index in dataframe.index:
        if bool(anonymous_mask.at[index]):
            mapping_rows.append(
                {
                    "staging_record_id": dataframe.at[
                        index, "staging_record_id"
                    ],
                    "source_system": dataframe.at[index, "source_system"],
                    "source_record_id": dataframe.at[
                        index, "source_record_id"
                    ],
                    "provisional_cluster_id": "",
                    "resolution_status": "anonymous_unresolvable",
                    "match_method": "Not applicable",
                    "cluster_size": "",
                    "cluster_confidence": "",
                    "linking_rules": "",
                }
            )
            continue

        cluster_id = index_to_cluster[index]
        cluster_size = cluster_sizes[cluster_id]
        linked = cluster_size > 1
        mapping_rows.append(
            {
                "staging_record_id": dataframe.at[
                    index, "staging_record_id"
                ],
                "source_system": dataframe.at[index, "source_system"],
                "source_record_id": dataframe.at[
                    index, "source_record_id"
                ],
                "provisional_cluster_id": cluster_id,
                "resolution_status": (
                    "deterministically_linked"
                    if linked
                    else "eligible_singleton"
                ),
                "match_method": "Deterministic" if linked else "Unresolved",
                "cluster_size": cluster_size,
                "cluster_confidence": 1.0 if linked else "",
                "linking_rules": " | ".join(
                    sorted(rules_by_cluster[cluster_id])
                ),
            }
        )

    return pd.DataFrame(mapping_rows)


def add_final_cluster_ids(
    decisions: list[dict[str, object]],
    mapping: pd.DataFrame,
    include_rejection_reason: bool = False,
) -> pd.DataFrame:
    # Attach the final provisional cluster for both decision records.
    columns = [
        "record_id_1",
        "source_system_1",
        "source_record_id_1",
        "record_id_2",
        "source_system_2",
        "source_record_id_2",
        "match_method",
        "rule_priority",
        "rule_code",
        "match_rule",
        "matched_fields",
        "matched_values",
        "match_score",
    ]
    if include_rejection_reason:
        columns.append("rejection_reason")

    columns.extend(
        [
            "provisional_cluster_id_1",
            "provisional_cluster_id_2",
        ]
    )

    if not decisions:
        return pd.DataFrame(columns=columns)

    decision_frame = pd.DataFrame(decisions)
    cluster_lookup = mapping.set_index("staging_record_id")[
        "provisional_cluster_id"
    ].to_dict()
    decision_frame["provisional_cluster_id_1"] = decision_frame[
        "record_id_1"
    ].map(cluster_lookup)
    decision_frame["provisional_cluster_id_2"] = decision_frame[
        "record_id_2"
    ].map(cluster_lookup)

    return decision_frame[columns]


def build_unresolved_records(
    dataframe: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    # Return eligible singleton records for probabilistic matching.
    unresolved_mapping = mapping.loc[
        mapping["resolution_status"].eq("eligible_singleton")
    ].copy()
    source_columns = [
        column
        for column in dataframe.columns
        if column not in {
            "staging_record_id",
            "source_system",
            "source_record_id",
        }
    ]

    return unresolved_mapping.merge(
        dataframe[
            [
                "staging_record_id",
                "source_system",
                "source_record_id",
                *source_columns,
            ]
        ],
        on=[
            "staging_record_id",
            "source_system",
            "source_record_id",
        ],
        how="left",
        validate="one_to_one",
    )


###############################################################################
# 9. Summary and output validation
###############################################################################


def add_summary_section(
    rows: list[dict[str, object]],
    section_name: str,
) -> None:
    # Add a readable section header to the two-column summary.
    if rows:
        rows.append({"metric": "", "value": ""})
    rows.append({"metric": section_name, "value": ""})


def build_summary(
    dataframe: pd.DataFrame,
    anonymous_mask: pd.Series,
    mapping: pd.DataFrame,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    unresolved: pd.DataFrame,
    rule_statistics: list[dict[str, object]],
) -> pd.DataFrame:
    # Create the deterministic validation and result summary.
    rows: list[dict[str, object]] = []

    add_summary_section(rows, "INPUT VALIDATION")
    rows.extend(
        [
            {"metric": "input_records", "value": len(dataframe)},
            {"metric": "input_columns", "value": len(dataframe.columns)},
            {
                "metric": "source_systems",
                "value": dataframe["source_system"].nunique(),
            },
            {
                "metric": "staging_record_ids_unique",
                "value": dataframe["staging_record_id"].is_unique,
            },
            {
                "metric": "source_record_ids_unique",
                "value": dataframe["source_record_id"].is_unique,
            },
            {
                "metric": "ground_truth_columns_excluded",
                "value": not any(
                    "ground_truth" in column.casefold()
                    for column in dataframe.columns
                ),
            },
        ]
    )

    add_summary_section(rows, "MATCHING POPULATION")
    rows.extend(
        [
            {
                "metric": "eligible_records",
                "value": int((~anonymous_mask).sum()),
            },
            {
                "metric": "anonymous_unresolvable_records",
                "value": int(anonymous_mask.sum()),
            },
        ]
    )

    add_summary_section(rows, "RULE RESULTS")
    for statistics in rule_statistics:
        rule = statistics["rule"]
        for metric in (
            "candidate_pairs",
            "accepted_merges",
            "rejected_merges",
            "already_linked_pairs",
        ):
            rows.append(
                {
                    "metric": f"{rule.code}_{metric}",
                    "value": statistics[metric],
                }
            )

    status_counts = mapping["resolution_status"].value_counts()
    linked_mapping = mapping.loc[
        mapping["resolution_status"].eq("deterministically_linked")
    ]
    cluster_sizes = pd.to_numeric(
        linked_mapping["cluster_size"], errors="coerce"
    )

    add_summary_section(rows, "CLUSTER RESULTS")
    rows.extend(
        [
            {
                "metric": "accepted_deterministic_merges",
                "value": len(accepted),
            },
            {
                "metric": "rejected_deterministic_merges",
                "value": len(rejected),
            },
            {
                "metric": "deterministically_linked_records",
                "value": int(
                    status_counts.get("deterministically_linked", 0)
                ),
            },
            {
                "metric": "deterministic_clusters",
                "value": linked_mapping[
                    "provisional_cluster_id"
                ].nunique(),
            },
            {
                "metric": "eligible_singleton_records",
                "value": int(status_counts.get("eligible_singleton", 0)),
            },
            {
                "metric": "all_provisional_clusters",
                "value": mapping["provisional_cluster_id"].replace(
                    "", pd.NA
                ).nunique(),
            },
            {
                "metric": "largest_deterministic_cluster",
                "value": int(cluster_sizes.max())
                if not cluster_sizes.empty
                else 0,
            },
        ]
    )

    add_summary_section(rows, "OUTPUT VALIDATION")
    rows.extend(
        [
            {
                "metric": "record_mapping_rows",
                "value": len(mapping),
            },
            {
                "metric": "mapping_reconciles_to_input",
                "value": len(mapping) == len(dataframe),
            },
            {
                "metric": "pairwise_match_rows",
                "value": len(accepted),
            },
            {
                "metric": "rejected_match_rows",
                "value": len(rejected),
            },
            {
                "metric": "unresolved_output_rows",
                "value": len(unresolved),
            },
            {
                "metric": "unresolved_excludes_anonymous_records",
                "value": len(unresolved)
                == int(status_counts.get("eligible_singleton", 0)),
            },
        ]
    )

    return pd.DataFrame(rows, columns=["metric", "value"])


def validate_outputs(
    dataframe: pd.DataFrame,
    mapping: pd.DataFrame,
    accepted: pd.DataFrame,
    unresolved: pd.DataFrame,
) -> None:
    # Validate reconciliation, mapping uniqueness and accepted links.
    if len(mapping) != len(dataframe):
        raise ValueError("The deterministic mapping does not reconcile.")
    if not mapping["staging_record_id"].is_unique:
        raise ValueError("The deterministic mapping contains duplicate rows.")

    input_ids = set(dataframe["staging_record_id"])
    mapping_ids = set(mapping["staging_record_id"])
    if input_ids != mapping_ids:
        raise ValueError("The mapping does not contain every staging record.")

    anonymous_mapping = mapping.loc[
        mapping["resolution_status"].eq("anonymous_unresolvable")
    ]
    if anonymous_mapping["provisional_cluster_id"].ne("").any():
        raise ValueError("Anonymous records were assigned a cluster ID.")

    if not unresolved["resolution_status"].eq(
        "eligible_singleton"
    ).all():
        raise ValueError(
            "The unresolved output contains ineligible records."
        )

    if not accepted.empty:
        left_clusters = accepted["provisional_cluster_id_1"]
        right_clusters = accepted["provisional_cluster_id_2"]
        if not left_clusters.eq(right_clusters).all():
            raise ValueError(
                "An accepted pair was not assigned to one final cluster."
            )


def write_outputs(
    output_directory: Path,
    accepted: pd.DataFrame,
    mapping: pd.DataFrame,
    rejected: pd.DataFrame,
    unresolved: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    # Write the five deterministic-stage CSV outputs.
    output_directory.mkdir(parents=True, exist_ok=True)

    accepted.to_csv(
        output_directory / PAIRWISE_MATCHES_FILENAME,
        index=False,
    )
    mapping.to_csv(
        output_directory / RECORD_MAPPING_FILENAME,
        index=False,
    )
    rejected.to_csv(
        output_directory / REJECTED_MATCHES_FILENAME,
        index=False,
    )
    unresolved.to_csv(
        output_directory / UNRESOLVED_RECORDS_FILENAME,
        index=False,
    )
    summary.to_csv(
        output_directory / SUMMARY_FILENAME,
        index=False,
    )


###############################################################################
# 10. Main deterministic matching workflow
###############################################################################


def run_deterministic_matching(
    input_path: Path,
    output_directory: Path,
) -> dict[str, int]:
    # Run deterministic identity resolution and write validated outputs.
    dataframe = load_input(input_path)
    validate_input(dataframe)
    anonymous_mask = identify_anonymous_online_records(dataframe)

    (
        union_find,
        accepted_decisions,
        rejected_decisions,
        rule_statistics,
    ) = apply_deterministic_rules(dataframe, anonymous_mask)

    mapping = build_record_mapping(
        dataframe,
        anonymous_mask,
        union_find,
        accepted_decisions,
    )
    accepted = add_final_cluster_ids(accepted_decisions, mapping)
    rejected = add_final_cluster_ids(
        rejected_decisions,
        mapping,
        include_rejection_reason=True,
    )
    unresolved = build_unresolved_records(dataframe, mapping)
    summary = build_summary(
        dataframe,
        anonymous_mask,
        mapping,
        accepted,
        rejected,
        unresolved,
        rule_statistics,
    )

    validate_outputs(dataframe, mapping, accepted, unresolved)
    write_outputs(
        output_directory,
        accepted,
        mapping,
        rejected,
        unresolved,
        summary,
    )

    status_counts = mapping["resolution_status"].value_counts()
    return {
        "input_records": len(dataframe),
        "accepted_matches": len(accepted),
        "rejected_matches": len(rejected),
        "linked_records": int(
            status_counts.get("deterministically_linked", 0)
        ),
        "eligible_singletons": int(
            status_counts.get("eligible_singleton", 0)
        ),
        "anonymous_unresolvable": int(
            status_counts.get("anonymous_unresolvable", 0)
        ),
    }


def main() -> None:
    # Run the command-line workflow and print a concise result.
    arguments = parse_arguments()
    input_path = arguments.input or locate_default_input()
    output_directory = arguments.output_dir

    results = run_deterministic_matching(input_path, output_directory)

    print("Deterministic identity resolution completed successfully.")
    print(f"Input: {input_path.resolve()}")
    print(f"Output directory: {output_directory.resolve()}")
    print(f"Input records: {results['input_records']:,}")
    print(f"Accepted matches: {results['accepted_matches']:,}")
    print(f"Rejected matches: {results['rejected_matches']:,}")
    print(f"Deterministically linked records: {results['linked_records']:,}")
    print(f"Eligible singletons: {results['eligible_singletons']:,}")
    print(
        "Anonymous unresolvable records: "
        f"{results['anonymous_unresolvable']:,}"
    )


if __name__ == "__main__":
    main()
