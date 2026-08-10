# This script is a calibrated second stage of identity resolution for the
# Universal Customer Record (UCR) project. It uses the frozen configuration
# selected through ground-truth calibration and applies that configuration
# without accessing ground truth during operational matching.
#
# Frozen configuration:
# automatic match threshold = 0.91
# review match threshold = 0.70
# minimum automatic evidence coverage = 0.55
# maximum blocking-key size = 100
#
# Ground truth is deliberately prohibited from every input. Anonymous online
# sessions remain in the record inventory but are excluded from candidate
# generation, scoring and clustering. Final UCR identifiers are not assigned
# here; eligible records receive stable provisional PCL identifiers.

###############################################################################
# Imports
###############################################################################

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError as error:
    raise ImportError(
        "RapidFuzz is required. Install it with: pip install rapidfuzz"
    ) from error


###############################################################################
# 1. Configuration
###############################################################################

INPUT_FILENAME = "consolidated_customer_records.csv"
DETERMINISTIC_DIRECTORY = Path("identity_resolution") / "deterministic"
OUTPUT_DIRECTORY = (
    Path("identity_resolution") / "probabilistic_calibrated"
)

DETERMINISTIC_PAIRWISE_FILENAME = (
    "deterministic_pairwise_matches.csv"
)
DETERMINISTIC_MAPPING_FILENAME = "deterministic_record_mapping.csv"
DETERMINISTIC_REJECTED_FILENAME = (
    "deterministic_rejected_matches.csv"
)
DETERMINISTIC_UNRESOLVED_FILENAME = (
    "deterministic_unresolved_records.csv"
)
DETERMINISTIC_SUMMARY_FILENAME = "deterministic_matching_summary.csv"

CANDIDATE_PAIRS_FILENAME = "probabilistic_candidate_pairs.csv"
PAIRWISE_SCORES_FILENAME = "probabilistic_pairwise_scores.csv"
ACCEPTED_MATCHES_FILENAME = "probabilistic_accepted_matches.csv"
REVIEW_MATCHES_FILENAME = "probabilistic_review_matches.csv"
REJECTED_MATCHES_FILENAME = "probabilistic_rejected_matches.csv"
RECORD_MAPPING_FILENAME = "probabilistic_record_mapping.csv"
UNRESOLVED_RECORDS_FILENAME = "probabilistic_unresolved_records.csv"
SUMMARY_FILENAME = "probabilistic_matching_summary.csv"

AUTOMATIC_MATCH_THRESHOLD = 0.91
REVIEW_MATCH_THRESHOLD = 0.70
MINIMUM_AUTOMATIC_COVERAGE = 0.55
MAXIMUM_BLOCK_SIZE = 100

FEATURE_WEIGHTS = {
    "email": 0.30,
    "phone": 0.25,
    "name": 0.20,
    "address": 0.10,
    "postcode": 0.10,
    "date_of_birth": 0.05,
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

CONSOLIDATED_REQUIRED_COLUMNS = {
    "staging_record_id",
    "source_system",
    "source_record_id",
    "portal_user_id",
    *IDENTITY_FIELDS,
}

MAPPING_REQUIRED_COLUMNS = {
    "staging_record_id",
    "source_system",
    "source_record_id",
    "provisional_cluster_id",
    "resolution_status",
    "match_method",
    "cluster_size",
    "cluster_confidence",
    "linking_rules",
}

PAIRWISE_REQUIRED_COLUMNS = {
    "record_id_1",
    "record_id_2",
    "match_method",
    "rule_code",
    "provisional_cluster_id_1",
    "provisional_cluster_id_2",
}

REJECTED_REQUIRED_COLUMNS = {
    "record_id_1",
    "record_id_2",
    "rejection_reason",
    "provisional_cluster_id_1",
    "provisional_cluster_id_2",
}

UNRESOLVED_REQUIRED_COLUMNS = {
    "staging_record_id",
    "provisional_cluster_id",
    "resolution_status",
}

SUMMARY_REQUIRED_COLUMNS = {"metric", "value"}

ALLOWED_DETERMINISTIC_STATUSES = {
    "deterministically_linked",
    "eligible_singleton",
    "anonymous_unresolvable",
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
# 2. Blocking rule definitions
###############################################################################


@dataclass(frozen=True)
class BlockingRule:
    priority: int
    code: str
    name: str


BLOCKING_RULES = (
    BlockingRule(
        priority=1,
        code="B01",
        name="Surname prefix and exact postcode",
    ),
    BlockingRule(
        priority=2,
        code="B02",
        name="Surname initial and postcode outward code",
    ),
    BlockingRule(
        priority=3,
        code="B03",
        name="Phone suffix and surname initial",
    ),
    BlockingRule(
        priority=4,
        code="B04",
        name="Email domain and local-part prefix",
    ),
    BlockingRule(
        priority=5,
        code="B05",
        name="First-name initial and exact postcode",
    ),
    BlockingRule(
        priority=6,
        code="B06",
        name="Birth year and surname prefix",
    ),
)


###############################################################################
# 3. Probabilistic cluster structure
###############################################################################


class ClusterUnionFind:
    def __init__(
        self,
        cluster_ids: Sequence[str],
        cluster_profiles: dict[str, dict[str, object]],
    ) -> None:
        self.cluster_ids = list(cluster_ids)
        self.cluster_to_index = {
            cluster_id: index
            for index, cluster_id in enumerate(self.cluster_ids)
        }
        cluster_count = len(self.cluster_ids)
        self.parent = list(range(cluster_count))
        self.size = [1] * cluster_count
        self.record_count: list[int] = []
        self.portal_ids: list[set[str]] = []
        self.dates_of_birth: list[set[str]] = []
        self.deterministic_clusters: list[set[str]] = []
        self.accepted_scores: list[list[float]] = []

        for cluster_id in self.cluster_ids:
            profile = cluster_profiles[cluster_id]
            self.record_count.append(int(profile["record_count"]))
            self.portal_ids.append(set(profile["portal_ids"]))
            self.dates_of_birth.append(
                set(profile["dates_of_birth"])
            )
            self.deterministic_clusters.append({cluster_id})
            self.accepted_scores.append([])

    def find(self, item: int) -> int:
        # Return the root for an item using path compression.
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]

        return item

    def find_cluster(self, cluster_id: str) -> int:
        # Return the current root for a deterministic cluster ID.
        return self.find(self.cluster_to_index[cluster_id])

    def union(
        self,
        left_root: int,
        right_root: int,
        score: float,
    ) -> tuple[int, bool]:
        # Merge two roots and retain their evidence and safeguards.
        left_root = self.find(left_root)
        right_root = self.find(right_root)

        if left_root == right_root:
            return left_root, False

        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root

        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        self.record_count[left_root] += self.record_count[right_root]
        self.portal_ids[left_root].update(self.portal_ids[right_root])
        self.dates_of_birth[left_root].update(
            self.dates_of_birth[right_root]
        )
        self.deterministic_clusters[left_root].update(
            self.deterministic_clusters[right_root]
        )
        self.accepted_scores[left_root].extend(
            self.accepted_scores[right_root]
        )
        self.accepted_scores[left_root].append(score)

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
    # Return a case-insensitive value suitable for matching.
    return clean_value(value).casefold()


def clamp_similarity(value: float) -> float:
    # Constrain a similarity value to the inclusive range zero to one.
    return max(0.0, min(1.0, value))


def fuzzy_ratio(left: str, right: str) -> float:
    # Return a normalised RapidFuzz ratio for two populated strings.
    if not left or not right:
        return 0.0

    return clamp_similarity(fuzz.ratio(left, right) / 100.0)


def email_parts(value: str) -> tuple[str, str]:
    # Split a valid-looking email into local and domain components.
    if "@" not in value:
        return value, ""

    local_part, domain = value.rsplit("@", maxsplit=1)
    return local_part, domain


def postcode_outward(value: str) -> str:
    # Return the outward portion of a normalised UK postcode.
    if len(value) <= 3:
        return ""

    return value[:-3]


def birth_year(value: str) -> str:
    # Return a four-digit birth year from a normalised date.
    if len(value) >= 4 and value[:4].isdigit():
        return value[:4]

    return ""


def parse_float(value: object) -> float | None:
    # Convert a populated value to float without accepting placeholders.
    cleaned = clean_value(value)
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def serialise_list(values: Iterable[str]) -> str:
    # Serialise sorted values for transparent CSV evidence.
    return " | ".join(sorted(set(values)))


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
    # Parse command-line paths for the calibrated matching run.
    parser = argparse.ArgumentParser(
        description=(
            "Run calibrated probabilistic identity matching."
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
        "--deterministic-dir",
        type=Path,
        default=DETERMINISTIC_DIRECTORY,
        help=(
            "Directory containing the five deterministic outputs. "
            f"Default: {DETERMINISTIC_DIRECTORY}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help=f"Output directory. Default: {OUTPUT_DIRECTORY}",
    )
    arguments = parser.parse_args()
    validate_configuration()
    return arguments


def validate_configuration() -> None:
    # Validate the frozen thresholds and block-size safeguard.
    thresholds = {
        "automatic threshold": AUTOMATIC_MATCH_THRESHOLD,
        "review threshold": REVIEW_MATCH_THRESHOLD,
        "minimum coverage": MINIMUM_AUTOMATIC_COVERAGE,
    }
    for name, value in thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"The {name} must be between 0 and 1.")

    if REVIEW_MATCH_THRESHOLD > AUTOMATIC_MATCH_THRESHOLD:
        raise ValueError(
            "The review threshold cannot exceed the automatic threshold."
        )
    if MAXIMUM_BLOCK_SIZE < 2:
        raise ValueError("The maximum block size must be at least 2.")


###############################################################################
# 5. Input loading and validation
###############################################################################


def load_csv(path: Path, description: str) -> pd.DataFrame:
    # Load one CSV without converting identifiers to numbers.
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def load_inputs(
    input_path: Path,
    deterministic_directory: Path,
) -> dict[str, pd.DataFrame]:
    # Load the consolidated data and all deterministic-stage outputs.
    return {
        "records": load_csv(input_path, "Consolidated input"),
        "mapping": load_csv(
            deterministic_directory / DETERMINISTIC_MAPPING_FILENAME,
            "Deterministic record mapping",
        ),
        "pairwise": load_csv(
            deterministic_directory / DETERMINISTIC_PAIRWISE_FILENAME,
            "Deterministic pairwise matches",
        ),
        "rejected": load_csv(
            deterministic_directory / DETERMINISTIC_REJECTED_FILENAME,
            "Deterministic rejected matches",
        ),
        "unresolved": load_csv(
            deterministic_directory / DETERMINISTIC_UNRESOLVED_FILENAME,
            "Deterministic unresolved records",
        ),
        "summary": load_csv(
            deterministic_directory / DETERMINISTIC_SUMMARY_FILENAME,
            "Deterministic matching summary",
        ),
    }


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    description: str,
) -> None:
    # Confirm an input contains its required fields.
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"{description} is missing required columns: "
            + ", ".join(missing_columns)
        )


def validate_ground_truth_exclusion(
    inputs: dict[str, pd.DataFrame],
) -> None:
    # Prohibit ground truth from every identity-resolution input.
    failures: list[str] = []
    for name, dataframe in inputs.items():
        ground_truth_columns = [
            column
            for column in dataframe.columns
            if "ground_truth" in column.casefold()
        ]
        if ground_truth_columns:
            failures.append(
                f"{name}: {', '.join(ground_truth_columns)}"
            )

    if failures:
        raise ValueError(
            "Ground-truth columns must not enter probabilistic matching: "
            + "; ".join(failures)
        )


def validate_input_structure(
    records: pd.DataFrame,
    mapping: pd.DataFrame,
) -> None:
    # Validate consolidated identifiers and mapping reconciliation.
    for dataframe, name in (
        (records, "Consolidated input"),
        (mapping, "Deterministic mapping"),
    ):
        identifiers = dataframe["staging_record_id"].map(clean_value)
        if identifiers.eq("").any():
            raise ValueError(f"{name} contains blank staging IDs.")
        if identifiers.duplicated().any():
            raise ValueError(f"{name} contains duplicate staging IDs.")

    if len(records) != len(mapping):
        raise ValueError(
            "The deterministic mapping does not reconcile to the "
            "consolidated input."
        )

    record_ids = set(records["staging_record_id"])
    mapping_ids = set(mapping["staging_record_id"])
    if record_ids != mapping_ids:
        raise ValueError(
            "The deterministic mapping does not contain every input record."
        )

    comparison = mapping.merge(
        records[
            [
                "staging_record_id",
                "source_system",
                "source_record_id",
            ]
        ],
        on="staging_record_id",
        how="left",
        suffixes=("_mapping", "_input"),
        validate="one_to_one",
    )
    source_mismatch = ~comparison["source_system_mapping"].eq(
        comparison["source_system_input"]
    )
    record_mismatch = ~comparison["source_record_id_mapping"].eq(
        comparison["source_record_id_input"]
    )
    if (source_mismatch | record_mismatch).any():
        raise ValueError(
            "Source references differ between the input and mapping."
        )


def validate_mapping_statuses(mapping: pd.DataFrame) -> None:
    # Validate eligible and anonymous deterministic mapping states.
    statuses = set(mapping["resolution_status"].map(clean_value))
    unexpected_statuses = sorted(
        statuses - ALLOWED_DETERMINISTIC_STATUSES
    )
    if unexpected_statuses:
        raise ValueError(
            "Unexpected deterministic resolution statuses: "
            + ", ".join(unexpected_statuses)
        )

    anonymous_mask = mapping["resolution_status"].eq(
        "anonymous_unresolvable"
    )
    eligible_mask = ~anonymous_mask
    if mapping.loc[anonymous_mask, "provisional_cluster_id"].ne("").any():
        raise ValueError("Anonymous records have deterministic cluster IDs.")
    if mapping.loc[eligible_mask, "provisional_cluster_id"].eq("").any():
        raise ValueError("An eligible record has no deterministic cluster.")

    stated_sizes = pd.to_numeric(
        mapping.loc[eligible_mask, "cluster_size"],
        errors="coerce",
    )
    actual_sizes = mapping.loc[
        eligible_mask, "provisional_cluster_id"
    ].map(
        mapping.loc[
            eligible_mask, "provisional_cluster_id"
        ].value_counts()
    )
    if stated_sizes.isna().any() or not stated_sizes.eq(
        actual_sizes
    ).all():
        raise ValueError("Deterministic cluster sizes are inconsistent.")


def validate_pairwise_audit(
    pairwise: pd.DataFrame,
    mapping: pd.DataFrame,
) -> None:
    # Confirm every trusted deterministic link remains within one cluster.
    if not pairwise["match_method"].eq("Deterministic").all():
        raise ValueError(
            "The deterministic pairwise file contains another method."
        )

    mapping_lookup = mapping.set_index("staging_record_id")[
        "provisional_cluster_id"
    ].to_dict()
    known_ids = set(mapping_lookup)
    referenced_ids = set(pairwise["record_id_1"]) | set(
        pairwise["record_id_2"]
    )
    if not referenced_ids <= known_ids:
        raise ValueError(
            "The deterministic pairwise file references unknown records."
        )

    expected_left = pairwise["record_id_1"].map(mapping_lookup)
    expected_right = pairwise["record_id_2"].map(mapping_lookup)
    if not expected_left.eq(expected_right).all():
        raise ValueError(
            "A trusted deterministic pair spans different clusters."
        )
    if not pairwise["provisional_cluster_id_1"].eq(
        expected_left
    ).all():
        raise ValueError(
            "A deterministic pair contains an outdated left cluster ID."
        )
    if not pairwise["provisional_cluster_id_2"].eq(
        expected_right
    ).all():
        raise ValueError(
            "A deterministic pair contains an outdated right cluster ID."
        )


def validate_unresolved_input(
    unresolved: pd.DataFrame,
    mapping: pd.DataFrame,
) -> None:
    # Confirm the convenience subset contains exactly eligible singletons.
    if not unresolved["staging_record_id"].is_unique:
        raise ValueError(
            "The deterministic unresolved file contains duplicate records."
        )
    if not unresolved["resolution_status"].eq(
        "eligible_singleton"
    ).all():
        raise ValueError(
            "The deterministic unresolved file contains ineligible records."
        )

    expected_ids = set(
        mapping.loc[
            mapping["resolution_status"].eq("eligible_singleton"),
            "staging_record_id",
        ]
    )
    observed_ids = set(unresolved["staging_record_id"])
    if expected_ids != observed_ids:
        raise ValueError(
            "The deterministic unresolved file does not match the mapping."
        )


def validate_summary_input(
    summary: pd.DataFrame,
    records: pd.DataFrame,
    mapping: pd.DataFrame,
    pairwise: pd.DataFrame,
    rejected: pd.DataFrame,
    unresolved: pd.DataFrame,
) -> None:
    # Reconcile core deterministic summary metrics to the supplied files.
    metric_lookup = summary.loc[
        summary["metric"].map(clean_value).ne("")
    ].set_index("metric")["value"].to_dict()
    expected_metrics = {
        "input_records": len(records),
        "accepted_deterministic_merges": len(pairwise),
        "rejected_deterministic_merges": len(rejected),
        "eligible_singleton_records": len(unresolved),
        "anonymous_unresolvable_records": int(
            mapping["resolution_status"].eq(
                "anonymous_unresolvable"
            ).sum()
        ),
    }

    failures: list[str] = []
    for metric, expected_value in expected_metrics.items():
        observed = parse_float(metric_lookup.get(metric, ""))
        if observed is None or int(observed) != expected_value:
            failures.append(
                f"{metric}: expected {expected_value}, observed "
                f"{metric_lookup.get(metric, 'missing')}"
            )

    if failures:
        raise ValueError(
            "Deterministic summary reconciliation failed: "
            + "; ".join(failures)
        )


def validate_inputs(inputs: dict[str, pd.DataFrame]) -> None:
    # Run structural, audit and ground-truth validation checks.
    requirements = {
        "records": CONSOLIDATED_REQUIRED_COLUMNS,
        "mapping": MAPPING_REQUIRED_COLUMNS,
        "pairwise": PAIRWISE_REQUIRED_COLUMNS,
        "rejected": REJECTED_REQUIRED_COLUMNS,
        "unresolved": UNRESOLVED_REQUIRED_COLUMNS,
        "summary": SUMMARY_REQUIRED_COLUMNS,
    }
    descriptions = {
        "records": "Consolidated input",
        "mapping": "Deterministic mapping",
        "pairwise": "Deterministic pairwise matches",
        "rejected": "Deterministic rejected matches",
        "unresolved": "Deterministic unresolved records",
        "summary": "Deterministic summary",
    }
    for name, required_columns in requirements.items():
        validate_required_columns(
            inputs[name],
            required_columns,
            descriptions[name],
        )

    validate_ground_truth_exclusion(inputs)
    validate_input_structure(inputs["records"], inputs["mapping"])
    validate_mapping_statuses(inputs["mapping"])
    validate_pairwise_audit(inputs["pairwise"], inputs["mapping"])
    validate_unresolved_input(inputs["unresolved"], inputs["mapping"])
    validate_summary_input(
        inputs["summary"],
        inputs["records"],
        inputs["mapping"],
        inputs["pairwise"],
        inputs["rejected"],
        inputs["unresolved"],
    )


###############################################################################
# 6. Eligible population and deterministic cluster profiles
###############################################################################


def build_eligible_records(
    records: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    # Join identity fields to every non-anonymous deterministic cluster.
    eligible_mapping = mapping.loc[
        ~mapping["resolution_status"].eq("anonymous_unresolvable")
    ].copy()
    identity_columns = [
        "staging_record_id",
        "portal_user_id",
        *IDENTITY_FIELDS,
    ]
    eligible = eligible_mapping.merge(
        records[identity_columns],
        on="staging_record_id",
        how="left",
        validate="one_to_one",
    )
    eligible = eligible.rename(
        columns={"provisional_cluster_id": "deterministic_cluster_id"}
    )

    for field in ("portal_user_id", *IDENTITY_FIELDS):
        eligible[field] = eligible[field].map(canonical_value)

    return eligible


def build_cluster_members(
    eligible: pd.DataFrame,
) -> dict[str, list[int]]:
    # Map each deterministic cluster to its eligible dataframe rows.
    members: dict[str, list[int]] = defaultdict(list)
    for index, cluster_id in eligible[
        "deterministic_cluster_id"
    ].items():
        members[cluster_id].append(index)

    for cluster_id in members:
        members[cluster_id].sort(
            key=lambda index: eligible.at[index, "staging_record_id"]
        )

    return dict(members)


def build_cluster_profiles(
    eligible: pd.DataFrame,
    cluster_members: dict[str, list[int]],
) -> dict[str, dict[str, object]]:
    # Build cluster-level values used by conflict safeguards.
    profiles: dict[str, dict[str, object]] = {}
    for cluster_id, indices in cluster_members.items():
        portal_ids = {
            eligible.at[index, "portal_user_id"]
            for index in indices
            if eligible.at[index, "portal_user_id"]
        }
        dates_of_birth = {
            eligible.at[index, "date_of_birth_normalised"]
            for index in indices
            if eligible.at[index, "date_of_birth_normalised"]
        }
        profiles[cluster_id] = {
            "record_count": len(indices),
            "portal_ids": portal_ids,
            "dates_of_birth": dates_of_birth,
            "minimum_staging_id": min(
                eligible.at[index, "staging_record_id"]
                for index in indices
            ),
        }

    return profiles


def build_prohibited_cluster_pairs(
    rejected: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], set[str]]]:
    # Convert deterministic rejections into prohibited cluster constraints.
    prohibited_pairs: set[tuple[str, str]] = set()
    reasons: dict[tuple[str, str], set[str]] = defaultdict(set)
    mapping_lookup = mapping.set_index("staging_record_id")[
        "provisional_cluster_id"
    ].to_dict()

    for row in rejected.itertuples(index=False):
        left_cluster = clean_value(row.provisional_cluster_id_1)
        right_cluster = clean_value(row.provisional_cluster_id_2)
        if not left_cluster:
            left_cluster = clean_value(mapping_lookup.get(row.record_id_1))
        if not right_cluster:
            right_cluster = clean_value(
                mapping_lookup.get(row.record_id_2)
            )
        if not left_cluster or not right_cluster:
            raise ValueError(
                "A deterministic rejection cannot be mapped to clusters."
            )
        if left_cluster == right_cluster:
            raise ValueError(
                "A deterministic rejection occurs within one cluster."
            )

        pair = tuple(sorted((left_cluster, right_cluster)))
        prohibited_pairs.add(pair)
        reasons[pair].add(clean_value(row.rejection_reason))

    return prohibited_pairs, dict(reasons)


###############################################################################
# 7. Cluster-level blocking and candidate generation
###############################################################################


def blocking_key(
    row: object,
    rule: BlockingRule,
) -> tuple[str, ...] | None:
    # Return a complete blocking key for one eligible record.
    surname = canonical_value(row.surname_normalised)
    first_name = canonical_value(row.first_name_normalised)
    postcode = canonical_value(row.postcode_normalised)
    phone = canonical_value(row.phone_normalised)
    email = canonical_value(row.email_normalised)
    date_of_birth = canonical_value(row.date_of_birth_normalised)

    if rule.code == "B01":
        if len(surname) >= 3 and postcode:
            return surname[:3], postcode
    elif rule.code == "B02":
        outward = postcode_outward(postcode)
        if surname and outward:
            return surname[0], outward
    elif rule.code == "B03":
        if len(phone) >= 7 and surname:
            return phone[-7:], surname[0]
    elif rule.code == "B04":
        local_part, domain = email_parts(email)
        if len(local_part) >= 4 and domain:
            return domain, local_part[:4]
    elif rule.code == "B05":
        if first_name and postcode:
            return first_name[0], postcode
    elif rule.code == "B06":
        year = birth_year(date_of_birth)
        if year and len(surname) >= 3:
            return year, surname[:3]

    return None


def build_block_groups(
    eligible: pd.DataFrame,
    rule: BlockingRule,
) -> dict[tuple[str, ...], set[str]]:
    # Group deterministic clusters by one record-level blocking key.
    groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
    columns = [
        "deterministic_cluster_id",
        "first_name_normalised",
        "surname_normalised",
        "email_normalised",
        "phone_normalised",
        "postcode_normalised",
        "date_of_birth_normalised",
    ]
    for row in eligible[columns].itertuples(index=False):
        key = blocking_key(row, rule)
        if key:
            groups[key].add(row.deterministic_cluster_id)

    return dict(groups)


def generate_candidate_pairs(
    eligible: pd.DataFrame,
    cluster_profiles: dict[str, dict[str, object]],
    maximum_block_size: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    # Combine candidates produced by all conservative blocking rules.
    candidates: dict[
        tuple[str, str], dict[str, object]
    ] = defaultdict(
        lambda: {
            "blocking_rules": set(),
            "blocking_rule_names": set(),
            "blocking_hits": 0,
        }
    )
    statistics: list[dict[str, object]] = []

    for rule in BLOCKING_RULES:
        groups = build_block_groups(eligible, rule)
        used_keys = 0
        skipped_keys = 0
        rule_pairs: set[tuple[str, str]] = set()

        for key in sorted(groups):
            cluster_ids = sorted(groups[key])
            if len(cluster_ids) < 2:
                continue
            if len(cluster_ids) > maximum_block_size:
                skipped_keys += 1
                continue

            used_keys += 1
            for left_cluster, right_cluster in combinations(
                cluster_ids, 2
            ):
                pair = (left_cluster, right_cluster)
                rule_pairs.add(pair)
                candidates[pair]["blocking_rules"].add(rule.code)
                candidates[pair]["blocking_rule_names"].add(rule.name)
                candidates[pair]["blocking_hits"] += 1

        statistics.append(
            {
                "rule": rule,
                "populated_keys": len(groups),
                "used_multi_cluster_keys": used_keys,
                "skipped_oversized_keys": skipped_keys,
                "candidate_pairs_contributed": len(rule_pairs),
            }
        )

    rows: list[dict[str, object]] = []
    for left_cluster, right_cluster in sorted(candidates):
        evidence = candidates[(left_cluster, right_cluster)]
        left_profile = cluster_profiles[left_cluster]
        right_profile = cluster_profiles[right_cluster]
        rows.append(
            {
                "cluster_id_1": left_cluster,
                "cluster_id_2": right_cluster,
                "cluster_size_1": left_profile["record_count"],
                "cluster_size_2": right_profile["record_count"],
                "blocking_rules": serialise_list(
                    evidence["blocking_rules"]
                ),
                "blocking_rule_names": serialise_list(
                    evidence["blocking_rule_names"]
                ),
                "blocking_hits": evidence["blocking_hits"],
            }
        )

    columns = [
        "cluster_id_1",
        "cluster_id_2",
        "cluster_size_1",
        "cluster_size_2",
        "blocking_rules",
        "blocking_rule_names",
        "blocking_hits",
    ]
    return pd.DataFrame(rows, columns=columns), statistics


###############################################################################
# 8. Record-pair similarity features
###############################################################################


def email_similarity(left: str, right: str) -> float:
    # Compare complete emails while retaining local-part evidence.
    if not left or not right:
        return 0.0

    full_score = fuzzy_ratio(left, right)
    left_local, left_domain = email_parts(left)
    right_local, right_domain = email_parts(right)
    local_score = fuzzy_ratio(left_local, right_local)
    if left_domain and left_domain == right_domain:
        return max(full_score, local_score)

    return full_score


def phone_similarity(left: str, right: str) -> float:
    # Compare normalised phones with controlled suffix agreement.
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if len(left) >= 7 and len(right) >= 7:
        if left[-7:] == right[-7:]:
            return 0.90

    return fuzzy_ratio(left, right)


def name_similarity(left: object, right: object) -> float:
    # Compare one coherent pair of full or component names.
    left_full = canonical_value(left.full_name_normalised)
    right_full = canonical_value(right.full_name_normalised)
    if left_full and right_full:
        return fuzzy_ratio(left_full, right_full)

    component_scores: list[float] = []
    for field in (
        "first_name_normalised",
        "surname_normalised",
    ):
        left_value = canonical_value(getattr(left, field))
        right_value = canonical_value(getattr(right, field))
        if left_value and right_value:
            component_scores.append(
                fuzzy_ratio(left_value, right_value)
            )

    if not component_scores:
        return 0.0

    return sum(component_scores) / len(component_scores)


def name_available(left: object, right: object) -> bool:
    # Confirm at least one comparable name representation is populated.
    left_full = canonical_value(left.full_name_normalised)
    right_full = canonical_value(right.full_name_normalised)
    if left_full and right_full:
        return True

    for field in (
        "first_name_normalised",
        "surname_normalised",
    ):
        if canonical_value(getattr(left, field)) and canonical_value(
            getattr(right, field)
        ):
            return True

    return False


def address_similarity(left: str, right: str) -> float:
    # Compare address tokens without requiring identical token order.
    if not left or not right:
        return 0.0

    return clamp_similarity(fuzz.token_set_ratio(left, right) / 100.0)


def postcode_similarity(left: str, right: str) -> float:
    # Compare exact postcodes and controlled outward-code agreement.
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    left_outward = postcode_outward(left)
    right_outward = postcode_outward(right)
    if left_outward and left_outward == right_outward:
        return max(0.60, fuzzy_ratio(left, right))

    return fuzzy_ratio(left, right)


def calculate_record_pair_score(
    left: object,
    right: object,
) -> dict[str, object]:
    # Score a single coherent record pair across available identity fields.
    left_email = canonical_value(left.email_normalised)
    right_email = canonical_value(right.email_normalised)
    left_phone = canonical_value(left.phone_normalised)
    right_phone = canonical_value(right.phone_normalised)
    left_address = canonical_value(left.address_normalised)
    right_address = canonical_value(right.address_normalised)
    left_postcode = canonical_value(left.postcode_normalised)
    right_postcode = canonical_value(right.postcode_normalised)
    left_dob = canonical_value(left.date_of_birth_normalised)
    right_dob = canonical_value(right.date_of_birth_normalised)

    similarities = {
        "email": email_similarity(left_email, right_email),
        "phone": phone_similarity(left_phone, right_phone),
        "name": name_similarity(left, right),
        "address": address_similarity(left_address, right_address),
        "postcode": postcode_similarity(
            left_postcode, right_postcode
        ),
        "date_of_birth": 1.0
        if left_dob and left_dob == right_dob
        else 0.0,
    }
    available = {
        "email": bool(left_email and right_email),
        "phone": bool(left_phone and right_phone),
        "name": name_available(left, right),
        "address": bool(left_address and right_address),
        "postcode": bool(left_postcode and right_postcode),
        "date_of_birth": bool(left_dob and right_dob),
    }

    evidence_coverage = sum(
        FEATURE_WEIGHTS[feature]
        for feature, is_available in available.items()
        if is_available
    )
    weighted_total = sum(
        FEATURE_WEIGHTS[feature] * similarities[feature]
        for feature, is_available in available.items()
        if is_available
    )
    weighted_score = (
        weighted_total / evidence_coverage
        if evidence_coverage
        else 0.0
    )

    strong_identifiers = [
        feature
        for feature, threshold in (("email", 0.85), ("phone", 0.90))
        if available[feature] and similarities[feature] >= threshold
    ]
    supporting_features = [
        feature
        for feature, is_available in available.items()
        if is_available and similarities[feature] >= 0.75
    ]
    corroborating_features = [
        feature
        for feature, is_available in available.items()
        if is_available
        and similarities[feature] >= 0.70
        and feature not in strong_identifiers
    ]

    return {
        "record_id_1": left.staging_record_id,
        "source_system_1": left.source_system,
        "source_record_id_1": left.source_record_id,
        "record_id_2": right.staging_record_id,
        "source_system_2": right.source_system,
        "source_record_id_2": right.source_record_id,
        **{
            f"{feature}_available": available[feature]
            for feature in FEATURE_WEIGHTS
        },
        **{
            f"{feature}_similarity": round(
                similarities[feature], 4
            )
            for feature in FEATURE_WEIGHTS
        },
        "weighted_similarity_score": round(weighted_score, 4),
        "evidence_coverage": round(evidence_coverage, 4),
        "strong_identifiers": serialise_list(strong_identifiers),
        "supporting_features": serialise_list(supporting_features),
        "corroborating_features": serialise_list(
            corroborating_features
        ),
        "supporting_feature_count": len(supporting_features),
        "corroborating_feature_count": len(
            corroborating_features
        ),
    }


###############################################################################
# 9. Candidate-cluster scoring and preliminary classification
###############################################################################


def static_cluster_conflict(
    left_cluster: str,
    right_cluster: str,
    cluster_profiles: dict[str, dict[str, object]],
    prohibited_pairs: set[tuple[str, str]],
    prohibited_reasons: dict[tuple[str, str], set[str]],
) -> str:
    # Return a known pre-scoring cluster conflict, if present.
    pair = tuple(sorted((left_cluster, right_cluster)))
    if pair in prohibited_pairs:
        reasons = prohibited_reasons.get(pair, set())
        detail = serialise_list(reasons)
        return (
            "Prohibited by deterministic rejection"
            + (f": {detail}" if detail else "")
        )

    left_profile = cluster_profiles[left_cluster]
    right_profile = cluster_profiles[right_cluster]
    left_portals = set(left_profile["portal_ids"])
    right_portals = set(right_profile["portal_ids"])
    if left_portals and right_portals:
        if left_portals.isdisjoint(right_portals):
            return "Conflicting trusted portal_user_id values"

    left_dobs = set(left_profile["dates_of_birth"])
    right_dobs = set(right_profile["dates_of_birth"])
    if left_dobs and right_dobs and left_dobs.isdisjoint(right_dobs):
        return "Conflicting known date_of_birth_normalised values"

    return ""


def classify_score(
    score: dict[str, object],
    conflict_reason: str,
    automatic_threshold: float,
    review_threshold: float,
    minimum_coverage: float,
) -> tuple[str, str]:
    # Apply the frozen decision bands and evidence requirements.
    weighted_score = float(score["weighted_similarity_score"])
    coverage = float(score["evidence_coverage"])
    strong_identifiers = clean_value(score["strong_identifiers"])
    supporting_count = int(score["supporting_feature_count"])
    corroborating_count = int(score["corroborating_feature_count"])

    if conflict_reason:
        return "rejected", conflict_reason

    sufficient_support = (
        bool(strong_identifiers) and corroborating_count >= 1
    ) or supporting_count >= 3

    if weighted_score >= automatic_threshold:
        if coverage < minimum_coverage:
            return (
                "review",
                "Automatic score reached but evidence coverage is low",
            )
        if not sufficient_support:
            return (
                "review",
                "Automatic score reached without sufficient corroboration",
            )
        return "automatic_match", "Automatic match criteria satisfied"

    if weighted_score >= review_threshold:
        return "review", "Weighted score falls within the review band"

    return "rejected", "Weighted score is below the review threshold"


def pair_rank(score: dict[str, object]) -> tuple[object, ...]:
    # Rank coherent record pairs by score, coverage and stable IDs.
    return (
        float(score["weighted_similarity_score"]),
        float(score["evidence_coverage"]),
        int(score["supporting_feature_count"]),
        -len(str(score["record_id_1"])),
        str(score["record_id_1"]),
        str(score["record_id_2"]),
    )


def score_candidate_clusters(
    eligible: pd.DataFrame,
    cluster_members: dict[str, list[int]],
    cluster_profiles: dict[str, dict[str, object]],
    candidates: pd.DataFrame,
    prohibited_pairs: set[tuple[str, str]],
    prohibited_reasons: dict[tuple[str, str], set[str]],
    automatic_threshold: float,
    review_threshold: float,
    minimum_coverage: float,
) -> pd.DataFrame:
    # Score every candidate using its strongest coherent record pair.
    rows: list[dict[str, object]] = []
    record_rows = {
        index: row
        for index, row in zip(
            eligible.index,
            eligible.itertuples(index=False),
        )
    }

    for candidate in candidates.itertuples(index=False):
        left_cluster = candidate.cluster_id_1
        right_cluster = candidate.cluster_id_2
        best_score: dict[str, object] | None = None
        compared_pairs = 0

        for left_index in cluster_members[left_cluster]:
            left_record = record_rows[left_index]
            for right_index in cluster_members[right_cluster]:
                right_record = record_rows[right_index]
                score = calculate_record_pair_score(
                    left_record,
                    right_record,
                )
                compared_pairs += 1
                if best_score is None or pair_rank(score) > pair_rank(
                    best_score
                ):
                    best_score = score

        if best_score is None:
            raise ValueError(
                "A candidate cluster pair contains no comparable records."
            )

        conflict_reason = static_cluster_conflict(
            left_cluster,
            right_cluster,
            cluster_profiles,
            prohibited_pairs,
            prohibited_reasons,
        )
        preliminary_decision, reason = classify_score(
            best_score,
            conflict_reason,
            automatic_threshold,
            review_threshold,
            minimum_coverage,
        )
        rows.append(
            {
                "cluster_id_1": left_cluster,
                "cluster_id_2": right_cluster,
                "cluster_size_1": candidate.cluster_size_1,
                "cluster_size_2": candidate.cluster_size_2,
                "blocking_rules": candidate.blocking_rules,
                "blocking_rule_names": (
                    candidate.blocking_rule_names
                ),
                "blocking_hits": candidate.blocking_hits,
                "record_pairs_compared": compared_pairs,
                **best_score,
                "static_conflict": bool(conflict_reason),
                "preliminary_decision": preliminary_decision,
                "preliminary_reason": reason,
            }
        )

    columns = score_output_columns()
    return pd.DataFrame(rows, columns=columns)


def score_output_columns() -> list[str]:
    # Return the stable schema for candidate score outputs.
    feature_columns: list[str] = []
    for feature in FEATURE_WEIGHTS:
        feature_columns.extend(
            [
                f"{feature}_available",
                f"{feature}_similarity",
            ]
        )

    return [
        "cluster_id_1",
        "cluster_id_2",
        "cluster_size_1",
        "cluster_size_2",
        "blocking_rules",
        "blocking_rule_names",
        "blocking_hits",
        "record_pairs_compared",
        "record_id_1",
        "source_system_1",
        "source_record_id_1",
        "record_id_2",
        "source_system_2",
        "source_record_id_2",
        *feature_columns,
        "weighted_similarity_score",
        "evidence_coverage",
        "strong_identifiers",
        "supporting_features",
        "corroborating_features",
        "supporting_feature_count",
        "corroborating_feature_count",
        "static_conflict",
        "preliminary_decision",
        "preliminary_reason",
    ]


###############################################################################
# 10. Conflict-controlled probabilistic clustering
###############################################################################


def prohibited_root_conflict(
    left_clusters: set[str],
    right_clusters: set[str],
    prohibited_pairs: set[tuple[str, str]],
) -> bool:
    # Check whether two evolving roots contain a prohibited DCL pair.
    for left_cluster in left_clusters:
        for right_cluster in right_clusters:
            pair = tuple(sorted((left_cluster, right_cluster)))
            if pair in prohibited_pairs:
                return True

    return False


def dynamic_cluster_conflict(
    union_find: ClusterUnionFind,
    left_root: int,
    right_root: int,
    prohibited_pairs: set[tuple[str, str]],
) -> str:
    # Reapply hard safeguards before each ordered cluster merge.
    left_portals = union_find.portal_ids[left_root]
    right_portals = union_find.portal_ids[right_root]
    if left_portals and right_portals:
        if left_portals.isdisjoint(right_portals):
            return "Conflicting trusted portal_user_id values after merging"

    left_dobs = union_find.dates_of_birth[left_root]
    right_dobs = union_find.dates_of_birth[right_root]
    if left_dobs and right_dobs and left_dobs.isdisjoint(right_dobs):
        return (
            "Conflicting known date_of_birth_normalised values after "
            "merging"
        )

    if prohibited_root_conflict(
        union_find.deterministic_clusters[left_root],
        union_find.deterministic_clusters[right_root],
        prohibited_pairs,
    ):
        return "Merge would violate a deterministic rejection constraint"

    return ""


def process_probabilistic_matches(
    scores: pd.DataFrame,
    cluster_profiles: dict[str, dict[str, object]],
    prohibited_pairs: set[tuple[str, str]],
) -> tuple[ClusterUnionFind, pd.DataFrame]:
    # Process automatic candidates from strongest to weakest evidence.
    cluster_ids = sorted(cluster_profiles)
    union_find = ClusterUnionFind(cluster_ids, cluster_profiles)
    decisions = scores.copy()
    decisions["final_decision"] = decisions["preliminary_decision"]
    decisions["final_reason"] = decisions["preliminary_reason"]
    decisions["merge_performed"] = False

    automatic_indices = decisions.index[
        decisions["preliminary_decision"].eq("automatic_match")
    ].tolist()
    automatic_indices.sort(
        key=lambda index: (
            -float(
                decisions.at[index, "weighted_similarity_score"]
            ),
            -float(decisions.at[index, "evidence_coverage"]),
            str(decisions.at[index, "cluster_id_1"]),
            str(decisions.at[index, "cluster_id_2"]),
        )
    )

    for index in automatic_indices:
        left_cluster = str(decisions.at[index, "cluster_id_1"])
        right_cluster = str(decisions.at[index, "cluster_id_2"])
        left_root = union_find.find_cluster(left_cluster)
        right_root = union_find.find_cluster(right_cluster)

        if left_root == right_root:
            decisions.at[index, "final_decision"] = (
                "accepted_transitive_support"
            )
            decisions.at[index, "final_reason"] = (
                "Clusters were already connected by stronger accepted "
                "matches"
            )
            continue

        conflict_reason = dynamic_cluster_conflict(
            union_find,
            left_root,
            right_root,
            prohibited_pairs,
        )
        if conflict_reason:
            decisions.at[index, "final_decision"] = "rejected"
            decisions.at[index, "final_reason"] = conflict_reason
            continue

        score = float(
            decisions.at[index, "weighted_similarity_score"]
        )
        union_find.union(left_root, right_root, score)
        decisions.at[index, "final_decision"] = "accepted_merge"
        decisions.at[index, "final_reason"] = (
            "Automatic criteria and cluster safeguards satisfied"
        )
        decisions.at[index, "merge_performed"] = True

    return union_find, decisions


###############################################################################
# 11. Probabilistic mapping and decision outputs
###############################################################################


def assign_probabilistic_cluster_ids(
    union_find: ClusterUnionFind,
    cluster_profiles: dict[str, dict[str, object]],
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    # Assign stable PCL identifiers to the final eligible roots.
    roots: dict[int, set[str]] = defaultdict(set)
    for cluster_id in sorted(cluster_profiles):
        root = union_find.find_cluster(cluster_id)
        roots[root].add(cluster_id)

    ordered_roots = sorted(
        roots,
        key=lambda root: min(
            str(cluster_profiles[cluster_id]["minimum_staging_id"])
            for cluster_id in roots[root]
        ),
    )

    cluster_lookup: dict[str, str] = {}
    pcl_profiles: dict[str, dict[str, object]] = {}
    for number, root in enumerate(ordered_roots, start=1):
        probabilistic_cluster_id = f"PCL{number:06d}"
        deterministic_clusters = roots[root]
        for cluster_id in deterministic_clusters:
            cluster_lookup[cluster_id] = probabilistic_cluster_id

        scores = union_find.accepted_scores[root]
        pcl_profiles[probabilistic_cluster_id] = {
            "record_count": union_find.record_count[root],
            "deterministic_cluster_count": len(
                deterministic_clusters
            ),
            "cluster_confidence": min(scores) if scores else None,
            "deterministic_clusters": deterministic_clusters,
        }

    return cluster_lookup, pcl_profiles


def add_probabilistic_cluster_ids(
    decisions: pd.DataFrame,
    cluster_lookup: dict[str, str],
) -> pd.DataFrame:
    # Attach the final PCL identifiers to every candidate decision.
    output = decisions.copy()
    output["probabilistic_cluster_id_1"] = output["cluster_id_1"].map(
        cluster_lookup
    )
    output["probabilistic_cluster_id_2"] = output["cluster_id_2"].map(
        cluster_lookup
    )
    return output


def build_probabilistic_mapping(
    mapping: pd.DataFrame,
    cluster_lookup: dict[str, str],
    pcl_profiles: dict[str, dict[str, object]],
) -> pd.DataFrame:
    # Map every staging record to its post-probabilistic resolution state.
    rows: list[dict[str, object]] = []

    for row in mapping.itertuples(index=False):
        deterministic_cluster_id = clean_value(
            row.provisional_cluster_id
        )
        if row.resolution_status == "anonymous_unresolvable":
            rows.append(
                {
                    "staging_record_id": row.staging_record_id,
                    "source_system": row.source_system,
                    "source_record_id": row.source_record_id,
                    "deterministic_cluster_id": "",
                    "probabilistic_cluster_id": "",
                    "resolution_status": "anonymous_unresolvable",
                    "match_method": "Not applicable",
                    "cluster_size": "",
                    "deterministic_cluster_count": "",
                    "cluster_confidence": "",
                    "cluster_confidence_type": "",
                    "deterministic_linking_rules": "",
                }
            )
            continue

        probabilistic_cluster_id = cluster_lookup[
            deterministic_cluster_id
        ]
        profile = pcl_profiles[probabilistic_cluster_id]
        deterministic_cluster_count = int(
            profile["deterministic_cluster_count"]
        )

        if deterministic_cluster_count > 1:
            resolution_status = "probabilistically_linked"
            match_method = "Hybrid"
            confidence = profile["cluster_confidence"]
            confidence_type = "Minimum accepted probabilistic score"
        elif row.resolution_status == "deterministically_linked":
            resolution_status = "deterministically_linked"
            match_method = "Deterministic"
            confidence = 1.0
            confidence_type = "Exact deterministic-rule confidence"
        else:
            resolution_status = "unresolved_singleton"
            match_method = "Unresolved"
            confidence = ""
            confidence_type = ""

        rows.append(
            {
                "staging_record_id": row.staging_record_id,
                "source_system": row.source_system,
                "source_record_id": row.source_record_id,
                "deterministic_cluster_id": deterministic_cluster_id,
                "probabilistic_cluster_id": probabilistic_cluster_id,
                "resolution_status": resolution_status,
                "match_method": match_method,
                "cluster_size": profile["record_count"],
                "deterministic_cluster_count": (
                    deterministic_cluster_count
                ),
                "cluster_confidence": confidence,
                "cluster_confidence_type": confidence_type,
                "deterministic_linking_rules": row.linking_rules,
            }
        )

    return pd.DataFrame(rows)


def build_unresolved_records(
    records: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    # Return identifiable records still unresolved after matching.
    unresolved_mapping = mapping.loc[
        mapping["resolution_status"].eq("unresolved_singleton")
    ].copy()
    source_columns = [
        column
        for column in records.columns
        if column not in {
            "staging_record_id",
            "source_system",
            "source_record_id",
        }
    ]
    return unresolved_mapping.merge(
        records[
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


def split_decision_outputs(
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Split final candidate decisions into accepted, review and rejected.
    accepted_statuses = {
        "accepted_merge",
        "accepted_transitive_support",
    }
    accepted = decisions.loc[
        decisions["final_decision"].isin(accepted_statuses)
    ].copy()
    review = decisions.loc[
        decisions["final_decision"].eq("review")
    ].copy()
    rejected = decisions.loc[
        decisions["final_decision"].eq("rejected")
    ].copy()
    return accepted, review, rejected


###############################################################################
# 12. Summary and output validation
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
    records: pd.DataFrame,
    deterministic_mapping: pd.DataFrame,
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    accepted: pd.DataFrame,
    review: pd.DataFrame,
    rejected: pd.DataFrame,
    mapping: pd.DataFrame,
    unresolved: pd.DataFrame,
    blocking_statistics: list[dict[str, object]],
    automatic_threshold: float,
    review_threshold: float,
    minimum_coverage: float,
    maximum_block_size: int,
) -> pd.DataFrame:
    # Create the probabilistic validation and result summary.
    rows: list[dict[str, object]] = []
    deterministic_clusters = deterministic_mapping[
        "provisional_cluster_id"
    ].replace("", pd.NA).nunique()
    possible_cluster_pairs = (
        deterministic_clusters * (deterministic_clusters - 1) // 2
    )
    candidate_reduction = (
        1.0 - (len(candidates) / possible_cluster_pairs)
        if possible_cluster_pairs
        else 0.0
    )
    eligible_records = int(
        (~deterministic_mapping["resolution_status"].eq(
            "anonymous_unresolvable"
        )).sum()
    )

    add_summary_section(rows, "INPUT VALIDATION")
    rows.extend(
        [
            {"metric": "input_records", "value": len(records)},
            {
                "metric": "deterministic_mapping_rows",
                "value": len(deterministic_mapping),
            },
            {
                "metric": "ground_truth_columns_excluded",
                "value": True,
            },
            {
                "metric": "deterministic_clusters",
                "value": deterministic_clusters,
            },
            {
                "metric": "eligible_records",
                "value": eligible_records,
            },
            {
                "metric": "anonymous_unresolvable_records",
                "value": int(
                    deterministic_mapping["resolution_status"].eq(
                        "anonymous_unresolvable"
                    ).sum()
                ),
            },
        ]
    )

    add_summary_section(rows, "FROZEN MATCHING CONFIGURATION")
    rows.extend(
        [
            {
                "metric": "automatic_match_threshold",
                "value": automatic_threshold,
            },
            {
                "metric": "review_match_threshold",
                "value": review_threshold,
            },
            {
                "metric": "minimum_automatic_evidence_coverage",
                "value": minimum_coverage,
            },
            {
                "metric": "maximum_block_size",
                "value": maximum_block_size,
            },
            *[
                {
                    "metric": f"{feature}_weight",
                    "value": weight,
                }
                for feature, weight in FEATURE_WEIGHTS.items()
            ],
        ]
    )

    add_summary_section(rows, "BLOCKING RESULTS")
    for statistics in blocking_statistics:
        rule = statistics["rule"]
        for metric in (
            "populated_keys",
            "used_multi_cluster_keys",
            "skipped_oversized_keys",
            "candidate_pairs_contributed",
        ):
            rows.append(
                {
                    "metric": f"{rule.code}_{metric}",
                    "value": statistics[metric],
                }
            )
    rows.extend(
        [
            {
                "metric": "possible_cluster_pairs",
                "value": possible_cluster_pairs,
            },
            {"metric": "candidate_cluster_pairs", "value": len(candidates)},
            {
                "metric": "candidate_reduction_ratio",
                "value": round(candidate_reduction, 6),
            },
            {
                "metric": "record_pairs_scored",
                "value": int(scores["record_pairs_compared"].sum())
                if not scores.empty
                else 0,
            },
        ]
    )

    add_summary_section(rows, "MATCH DECISIONS")
    rows.extend(
        [
            {
                "metric": "accepted_probabilistic_relationships",
                "value": len(accepted),
            },
            {
                "metric": "accepted_probabilistic_merges",
                "value": int(accepted["merge_performed"].sum())
                if not accepted.empty
                else 0,
            },
            {
                "metric": "accepted_transitive_support",
                "value": int(
                    accepted["final_decision"].eq(
                        "accepted_transitive_support"
                    ).sum()
                )
                if not accepted.empty
                else 0,
            },
            {"metric": "review_matches", "value": len(review)},
            {"metric": "rejected_matches", "value": len(rejected)},
        ]
    )

    status_counts = mapping["resolution_status"].value_counts()
    eligible_mapping = mapping.loc[
        ~mapping["resolution_status"].eq("anonymous_unresolvable")
    ]
    cluster_sizes = pd.to_numeric(
        eligible_mapping["cluster_size"], errors="coerce"
    )

    add_summary_section(rows, "CLUSTER RESULTS")
    rows.extend(
        [
            {
                "metric": "probabilistically_linked_records",
                "value": int(
                    status_counts.get("probabilistically_linked", 0)
                ),
            },
            {
                "metric": "deterministically_linked_only_records",
                "value": int(
                    status_counts.get("deterministically_linked", 0)
                ),
            },
            {
                "metric": "unresolved_singleton_records",
                "value": int(
                    status_counts.get("unresolved_singleton", 0)
                ),
            },
            {
                "metric": "anonymous_unresolvable_records_final",
                "value": int(
                    status_counts.get("anonymous_unresolvable", 0)
                ),
            },
            {
                "metric": "probabilistic_clusters",
                "value": eligible_mapping[
                    "probabilistic_cluster_id"
                ].nunique(),
            },
            {
                "metric": "largest_probabilistic_cluster",
                "value": int(cluster_sizes.max())
                if not cluster_sizes.empty
                else 0,
            },
        ]
    )

    add_summary_section(rows, "OUTPUT VALIDATION")
    rows.extend(
        [
            {"metric": "record_mapping_rows", "value": len(mapping)},
            {
                "metric": "mapping_reconciles_to_input",
                "value": len(mapping) == len(records),
            },
            {
                "metric": "candidate_decisions_reconcile",
                "value": len(scores)
                == len(accepted) + len(review) + len(rejected),
            },
            {
                "metric": "unresolved_output_rows",
                "value": len(unresolved),
            },
            {
                "metric": "anonymous_records_remain_unclustered",
                "value": mapping.loc[
                    mapping["resolution_status"].eq(
                        "anonymous_unresolvable"
                    ),
                    "probabilistic_cluster_id",
                ].eq("").all(),
            },
        ]
    )

    return pd.DataFrame(rows, columns=["metric", "value"])


def validate_outputs(
    records: pd.DataFrame,
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    accepted: pd.DataFrame,
    review: pd.DataFrame,
    rejected: pd.DataFrame,
    mapping: pd.DataFrame,
    unresolved: pd.DataFrame,
) -> None:
    # Validate candidate uniqueness, decisions and final reconciliation.
    if len(mapping) != len(records):
        raise ValueError("The probabilistic mapping does not reconcile.")
    if not mapping["staging_record_id"].is_unique:
        raise ValueError("The probabilistic mapping has duplicate records.")
    if set(mapping["staging_record_id"]) != set(
        records["staging_record_id"]
    ):
        raise ValueError(
            "The probabilistic mapping does not contain every input record."
        )

    candidate_pairs = list(
        zip(candidates["cluster_id_1"], candidates["cluster_id_2"])
    )
    if len(candidate_pairs) != len(set(candidate_pairs)):
        raise ValueError("Candidate cluster pairs are not unique.")
    if any(left >= right for left, right in candidate_pairs):
        raise ValueError("Candidate cluster pair ordering is unstable.")

    if len(decisions) != len(candidates):
        raise ValueError("Not every candidate cluster pair was scored.")
    if len(decisions) != len(accepted) + len(review) + len(rejected):
        raise ValueError("Final candidate decisions do not reconcile.")

    accepted_merges = accepted.loc[accepted["merge_performed"]]
    if not accepted_merges.empty:
        same_final_cluster = accepted_merges[
            "probabilistic_cluster_id_1"
        ].eq(accepted_merges["probabilistic_cluster_id_2"])
        if not same_final_cluster.all():
            raise ValueError(
                "An accepted merge spans different final clusters."
            )

    anonymous = mapping.loc[
        mapping["resolution_status"].eq("anonymous_unresolvable")
    ]
    if anonymous["probabilistic_cluster_id"].ne("").any():
        raise ValueError("Anonymous records received probabilistic IDs.")

    if not unresolved["resolution_status"].eq(
        "unresolved_singleton"
    ).all():
        raise ValueError(
            "The probabilistic unresolved output contains linked records."
        )

    eligible = mapping.loc[
        ~mapping["resolution_status"].eq("anonymous_unresolvable")
    ].copy()
    if eligible.empty:
        raise ValueError("No eligible probabilistic clusters were produced.")

    actual_sizes = eligible["probabilistic_cluster_id"].map(
        eligible["probabilistic_cluster_id"].value_counts()
    )
    stated_sizes = pd.to_numeric(
        eligible["cluster_size"], errors="coerce"
    )
    if stated_sizes.isna().any() or not stated_sizes.eq(
        actual_sizes
    ).all():
        raise ValueError("Probabilistic cluster sizes are inconsistent.")

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
            "Probabilistic deterministic-cluster counts are inconsistent."
        )

    identity_checks = eligible[[
        "staging_record_id",
        "probabilistic_cluster_id",
    ]].merge(
        records[
            [
                "staging_record_id",
                "portal_user_id",
                "date_of_birth_normalised",
            ]
        ],
        on="staging_record_id",
        how="left",
        validate="one_to_one",
    )
    for field in ("portal_user_id", "date_of_birth_normalised"):
        identity_checks[field] = identity_checks[field].map(
            canonical_value
        )
        populated = identity_checks.loc[
            identity_checks[field].ne("")
        ]
        conflict_counts = populated.groupby(
            "probabilistic_cluster_id"
        )[field].nunique()
        if conflict_counts.gt(1).any():
            raise ValueError(
                f"A final cluster contains conflicting {field} values."
            )

    expected_unresolved_ids = set(
        mapping.loc[
            mapping["resolution_status"].eq("unresolved_singleton"),
            "staging_record_id",
        ]
    )
    if set(unresolved["staging_record_id"]) != expected_unresolved_ids:
        raise ValueError(
            "The probabilistic unresolved output does not match the mapping."
        )


def write_outputs(
    output_directory: Path,
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    accepted: pd.DataFrame,
    review: pd.DataFrame,
    rejected: pd.DataFrame,
    mapping: pd.DataFrame,
    unresolved: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    # Write the eight probabilistic-stage CSV outputs.
    output_directory.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(
        output_directory / CANDIDATE_PAIRS_FILENAME,
        index=False,
    )
    decisions.to_csv(
        output_directory / PAIRWISE_SCORES_FILENAME,
        index=False,
    )
    accepted.to_csv(
        output_directory / ACCEPTED_MATCHES_FILENAME,
        index=False,
    )
    review.to_csv(
        output_directory / REVIEW_MATCHES_FILENAME,
        index=False,
    )
    rejected.to_csv(
        output_directory / REJECTED_MATCHES_FILENAME,
        index=False,
    )
    mapping.to_csv(
        output_directory / RECORD_MAPPING_FILENAME,
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
# 13. Main probabilistic matching workflow
###############################################################################


def run_probabilistic_matching(
    input_path: Path,
    deterministic_directory: Path,
    output_directory: Path,
    automatic_threshold: float,
    review_threshold: float,
    minimum_coverage: float,
    maximum_block_size: int,
) -> dict[str, int]:
    # Run probabilistic matching and write validated outputs.
    inputs = load_inputs(input_path, deterministic_directory)
    validate_inputs(inputs)

    records = inputs["records"]
    deterministic_mapping = inputs["mapping"]
    eligible = build_eligible_records(records, deterministic_mapping)
    cluster_members = build_cluster_members(eligible)
    cluster_profiles = build_cluster_profiles(
        eligible,
        cluster_members,
    )
    prohibited_pairs, prohibited_reasons = (
        build_prohibited_cluster_pairs(
            inputs["rejected"],
            deterministic_mapping,
        )
    )

    candidates, blocking_statistics = generate_candidate_pairs(
        eligible,
        cluster_profiles,
        maximum_block_size,
    )
    scores = score_candidate_clusters(
        eligible,
        cluster_members,
        cluster_profiles,
        candidates,
        prohibited_pairs,
        prohibited_reasons,
        automatic_threshold,
        review_threshold,
        minimum_coverage,
    )
    union_find, decisions = process_probabilistic_matches(
        scores,
        cluster_profiles,
        prohibited_pairs,
    )
    cluster_lookup, pcl_profiles = assign_probabilistic_cluster_ids(
        union_find,
        cluster_profiles,
    )
    decisions = add_probabilistic_cluster_ids(
        decisions,
        cluster_lookup,
    )
    accepted, review, rejected = split_decision_outputs(decisions)
    mapping = build_probabilistic_mapping(
        deterministic_mapping,
        cluster_lookup,
        pcl_profiles,
    )
    unresolved = build_unresolved_records(records, mapping)
    summary = build_summary(
        records,
        deterministic_mapping,
        candidates,
        decisions,
        accepted,
        review,
        rejected,
        mapping,
        unresolved,
        blocking_statistics,
        automatic_threshold,
        review_threshold,
        minimum_coverage,
        maximum_block_size,
    )

    validate_outputs(
        records,
        candidates,
        decisions,
        accepted,
        review,
        rejected,
        mapping,
        unresolved,
    )
    write_outputs(
        output_directory,
        candidates,
        decisions,
        accepted,
        review,
        rejected,
        mapping,
        unresolved,
        summary,
    )

    status_counts = mapping["resolution_status"].value_counts()
    return {
        "input_records": len(records),
        "deterministic_clusters": len(cluster_profiles),
        "candidate_pairs": len(candidates),
        "accepted_relationships": len(accepted),
        "accepted_merges": int(accepted["merge_performed"].sum())
        if not accepted.empty
        else 0,
        "review_matches": len(review),
        "rejected_matches": len(rejected),
        "probabilistically_linked": int(
            status_counts.get("probabilistically_linked", 0)
        ),
        "unresolved_singletons": int(
            status_counts.get("unresolved_singleton", 0)
        ),
        "anonymous_unresolvable": int(
            status_counts.get("anonymous_unresolvable", 0)
        ),
    }


def main() -> None:
    # Run the command-line workflow and print a concise result.
    arguments = parse_arguments()
    input_path = arguments.input or locate_default_input()
    results = run_probabilistic_matching(
        input_path,
        arguments.deterministic_dir,
        arguments.output_dir,
        AUTOMATIC_MATCH_THRESHOLD,
        REVIEW_MATCH_THRESHOLD,
        MINIMUM_AUTOMATIC_COVERAGE,
        MAXIMUM_BLOCK_SIZE,
    )

    print(
        "Calibrated probabilistic identity resolution completed "
        "successfully."
    )
    print(f"Input: {input_path.resolve()}")
    print(
        "Deterministic directory: "
        f"{arguments.deterministic_dir.resolve()}"
    )
    print(f"Output directory: {arguments.output_dir.resolve()}")
    print(
        "Frozen automatic threshold: "
        f"{AUTOMATIC_MATCH_THRESHOLD:.2f}"
    )
    print(
        "Frozen minimum evidence coverage: "
        f"{MINIMUM_AUTOMATIC_COVERAGE:.2f}"
    )
    print(f"Input records: {results['input_records']:,}")
    print(
        "Eligible deterministic clusters: "
        f"{results['deterministic_clusters']:,}"
    )
    print(f"Candidate cluster pairs: {results['candidate_pairs']:,}")
    print(
        "Accepted probabilistic relationships: "
        f"{results['accepted_relationships']:,}"
    )
    print(f"Accepted cluster merges: {results['accepted_merges']:,}")
    print(f"Review matches: {results['review_matches']:,}")
    print(f"Rejected matches: {results['rejected_matches']:,}")
    print(
        "Probabilistically linked records: "
        f"{results['probabilistically_linked']:,}"
    )
    print(
        "Unresolved singleton records: "
        f"{results['unresolved_singletons']:,}"
    )
    print(
        "Anonymous unresolvable records: "
        f"{results['anonymous_unresolvable']:,}"
    )


if __name__ == "__main__":
    main()
