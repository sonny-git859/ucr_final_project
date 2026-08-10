# Evaluate deterministic and probabilistic identity-resolution results against
# protected synthetic ground truth.

###############################################################################
# Imports
###############################################################################

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Sequence

import pandas as pd


###############################################################################
# 1. Configuration
###############################################################################

INPUT_FILENAME = "consolidated_customer_records.csv"
GROUND_TRUTH_DIRECTORY = Path("data") / "reference"
DETERMINISTIC_DIRECTORY = Path("identity_resolution") / "deterministic"
PROBABILISTIC_DIRECTORY = Path("identity_resolution") / "probabilistic"
OUTPUT_DIRECTORY = Path("identity_resolution") / "evaluation"

DETERMINISTIC_PAIRWISE_FILENAME = (
    "deterministic_pairwise_matches.csv"
)
DETERMINISTIC_MAPPING_FILENAME = "deterministic_record_mapping.csv"
DETERMINISTIC_REJECTED_FILENAME = (
    "deterministic_rejected_matches.csv"
)

PROBABILISTIC_CANDIDATES_FILENAME = (
    "probabilistic_candidate_pairs.csv"
)
PROBABILISTIC_SCORES_FILENAME = "probabilistic_pairwise_scores.csv"
PROBABILISTIC_ACCEPTED_FILENAME = (
    "probabilistic_accepted_matches.csv"
)
PROBABILISTIC_MAPPING_FILENAME = "probabilistic_record_mapping.csv"
PROBABILISTIC_SUMMARY_FILENAME = "probabilistic_matching_summary.csv"

STAGING_GROUND_TRUTH_FILENAME = "staging_ground_truth_mapping.csv"
IDENTITY_SPLIT_FILENAME = "identity_split_mapping.csv"
CALIBRATION_RESULTS_FILENAME = "calibration_results.csv"
SELECTED_CONFIGURATION_FILENAME = (
    "selected_matching_configuration.csv"
)
DETERMINISTIC_EVALUATION_FILENAME = "deterministic_evaluation.csv"
BLOCKING_EVALUATION_FILENAME = "blocking_evaluation.csv"
PROBABILISTIC_EVALUATION_FILENAME = "probabilistic_evaluation.csv"
CLUSTER_EVALUATION_FILENAME = "cluster_evaluation.csv"
SELECTED_MATCHES_FILENAME = "selected_configuration_matches.csv"
SELECTED_MAPPING_FILENAME = (
    "selected_configuration_record_mapping.csv"
)
SUMMARY_FILENAME = "identity_resolution_evaluation_summary.csv"
FALSE_POSITIVES_FILENAME = "false_positive_matches.csv"
FALSE_NEGATIVES_FILENAME = "false_negative_matches.csv"
SPLIT_IDENTITIES_FILENAME = "split_ground_truth_identities.csv"
IMPURE_CLUSTERS_FILENAME = "impure_resolved_clusters.csv"

CALIBRATION_PROPORTION = 0.70
RANDOM_SEED = 42
MINIMUM_PRECISION = 0.99
MINIMUM_CALIBRATION_MATCHES = 10

AUTOMATIC_THRESHOLDS = tuple(
    round(value / 100, 2)
    for value in range(80, 99)
)
EVIDENCE_COVERAGE_THRESHOLDS = tuple(
    round(value / 100, 2)
    for value in range(50, 91, 5)
)

FEATURE_WEIGHTS = {
    "email": 0.30,
    "phone": 0.25,
    "name": 0.20,
    "address": 0.10,
    "postcode": 0.10,
    "date_of_birth": 0.05,
}

GROUND_TRUTH_SOURCES = {
    "CRM": (
        "crm_ground_truth_mapping.csv",
        "crm_customer_id",
    ),
    "ECOMMERCE": (
        "ecommerce_ground_truth_mapping.csv",
        "transaction_id",
    ),
    "MARKETING": (
        "marketing_ground_truth_mapping.csv",
        "marketing_contact_id",
    ),
    "ONLINE": (
        "online_ground_truth_mapping.csv",
        "session_id",
    ),
    "SUPPORT": (
        "support_ground_truth_mapping.csv",
        "support_ticket_id",
    ),
}

CONSOLIDATED_REQUIRED_COLUMNS = {
    "staging_record_id",
    "source_system",
    "source_record_id",
    "portal_user_id",
    "date_of_birth_normalised",
}

DETERMINISTIC_MAPPING_REQUIRED_COLUMNS = {
    "staging_record_id",
    "source_system",
    "source_record_id",
    "provisional_cluster_id",
    "resolution_status",
}

DETERMINISTIC_PAIRWISE_REQUIRED_COLUMNS = {
    "record_id_1",
    "record_id_2",
    "provisional_cluster_id_1",
    "provisional_cluster_id_2",
}

DETERMINISTIC_REJECTED_REQUIRED_COLUMNS = {
    "provisional_cluster_id_1",
    "provisional_cluster_id_2",
    "rejection_reason",
}

PROBABILISTIC_CANDIDATE_REQUIRED_COLUMNS = {
    "cluster_id_1",
    "cluster_id_2",
    "blocking_rules",
}

PROBABILISTIC_SCORE_REQUIRED_COLUMNS = {
    "cluster_id_1",
    "cluster_id_2",
    "record_id_1",
    "record_id_2",
    "weighted_similarity_score",
    "evidence_coverage",
    "strong_identifiers",
    "supporting_feature_count",
    "corroborating_feature_count",
    "static_conflict",
    "final_decision",
    "merge_performed",
}

PROBABILISTIC_MAPPING_REQUIRED_COLUMNS = {
    "staging_record_id",
    "source_system",
    "source_record_id",
    "deterministic_cluster_id",
    "probabilistic_cluster_id",
    "resolution_status",
}

SUMMARY_REQUIRED_COLUMNS = {"metric", "value"}

ANONYMOUS_STATUS = "anonymous_unresolvable"
ANONYMOUS_ONLY_SPLIT = "anonymous_only"
CALIBRATION_SPLIT = "calibration"
EVALUATION_SPLIT = "evaluation"


###############################################################################
# 2. Evaluation union-find structure
###############################################################################


class EvaluationUnionFind:
    def __init__(
        self,
        cluster_ids: Sequence[str],
        cluster_profiles: dict[str, dict[str, object]],
    ) -> None:
        self.parent = {
            cluster_id: cluster_id
            for cluster_id in cluster_ids
        }
        self.size = {
            cluster_id: 1
            for cluster_id in cluster_ids
        }
        self.record_count = {
            cluster_id: int(
                cluster_profiles[cluster_id]["record_count"]
            )
            for cluster_id in cluster_ids
        }
        self.portal_ids = {
            cluster_id: set(
                cluster_profiles[cluster_id]["portal_ids"]
            )
            for cluster_id in cluster_ids
        }
        self.dates_of_birth = {
            cluster_id: set(
                cluster_profiles[cluster_id]["dates_of_birth"]
            )
            for cluster_id in cluster_ids
        }
        self.deterministic_clusters = {
            cluster_id: {cluster_id}
            for cluster_id in cluster_ids
        }

    def find(self, cluster_id: str) -> str:
        # Return a root using path compression.
        root = cluster_id
        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[cluster_id] != cluster_id:
            parent = self.parent[cluster_id]
            self.parent[cluster_id] = root
            cluster_id = parent

        return root

    def union(self, left_root: str, right_root: str) -> str:
        # Merge roots while retaining stable cluster information.
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root

        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        self.record_count[left_root] += self.record_count[right_root]
        self.portal_ids[left_root].update(
            self.portal_ids[right_root]
        )
        self.dates_of_birth[left_root].update(
            self.dates_of_birth[right_root]
        )
        self.deterministic_clusters[left_root].update(
            self.deterministic_clusters[right_root]
        )

        return left_root


###############################################################################
# 3. General helper functions
###############################################################################


def clean_value(value: object) -> str:
    # Convert a value into a stripped string without placeholders.
    if value is None or pd.isna(value):
        return ""

    cleaned = str(value).strip()
    if cleaned.lower() in {"", "nan", "none", "null"}:
        return ""

    return cleaned


def parse_boolean(value: object) -> bool:
    # Convert common CSV boolean representations into a Boolean value.
    return clean_value(value).lower() in {"1", "true", "yes", "y"}


def serialise_values(values: object) -> str:
    # Convert a collection into a stable audit string.
    cleaned = sorted(
        {
            clean_value(value)
            for value in values
            if clean_value(value)
        }
    )
    return " | ".join(cleaned)


def safe_divide(numerator: float, denominator: float) -> float:
    # Divide safely when a metric has no possible observations.
    if denominator == 0:
        return 0.0

    return numerator / denominator


def harmonic_mean(precision: float, recall: float) -> float:
    # Calculate an F1 score from precision and recall.
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def pair_count(value: int) -> int:
    # Return the number of unordered pairs in a group.
    return value * (value - 1) // 2


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    # Return a stable unordered cluster-pair key.
    return tuple(sorted((clean_value(left), clean_value(right))))


def add_summary_section(
    rows: list[dict[str, object]],
    section_name: str,
) -> None:
    # Add a labelled section and separator to a summary file.
    if rows:
        rows.append({"metric": "", "value": ""})
    rows.append({"metric": section_name, "value": ""})


###############################################################################
# 4. Command-line configuration
###############################################################################


def parse_arguments() -> argparse.Namespace:
    # Parse paths and reproducible evaluation parameters.
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate identity-resolution outputs against protected "
            "synthetic ground truth."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data")
        / "consolidated_silver"
        / INPUT_FILENAME,
        help="Path to the consolidated Silver customer records.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=GROUND_TRUTH_DIRECTORY,
        help="Directory containing the five ground-truth mappings.",
    )
    parser.add_argument(
        "--deterministic-dir",
        type=Path,
        default=DETERMINISTIC_DIRECTORY,
        help="Directory containing deterministic matching outputs.",
    )
    parser.add_argument(
        "--probabilistic-dir",
        type=Path,
        default=PROBABILISTIC_DIRECTORY,
        help="Directory containing probabilistic matching outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--calibration-proportion",
        type=float,
        default=CALIBRATION_PROPORTION,
        help=(
            "Proportion of identities used for calibration. "
            f"Default: {CALIBRATION_PROPORTION}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for identity splitting. Default: {RANDOM_SEED}",
    )
    parser.add_argument(
        "--minimum-precision",
        type=float,
        default=MINIMUM_PRECISION,
        help=(
            "Minimum calibration precision used to select a frozen "
            f"configuration. Default: {MINIMUM_PRECISION}"
        ),
    )
    parser.add_argument(
        "--minimum-calibration-matches",
        type=int,
        default=MINIMUM_CALIBRATION_MATCHES,
        help=(
            "Minimum predicted calibration relationships required for "
            "configuration selection. "
            f"Default: {MINIMUM_CALIBRATION_MATCHES}"
        ),
    )

    return parser.parse_args()


def validate_parameters(arguments: argparse.Namespace) -> None:
    # Reject parameters that would invalidate the evaluation design.
    if not 0 < arguments.calibration_proportion < 1:
        raise ValueError(
            "The calibration proportion must be between 0 and 1."
        )
    if not 0 <= arguments.minimum_precision <= 1:
        raise ValueError(
            "The minimum precision must be between 0 and 1."
        )
    if arguments.minimum_calibration_matches < 1:
        raise ValueError(
            "Minimum calibration matches must be at least 1."
        )


###############################################################################
# 5. Input loading and validation
###############################################################################


def read_csv(path: Path) -> pd.DataFrame:
    # Read CSV values as strings so identifiers remain unchanged.
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    filename: str,
) -> None:
    # Confirm that an input contains its required schema.
    missing = sorted(required_columns - set(dataframe.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"{filename} is missing required columns: {missing_text}"
        )


def validate_matching_output_separation(
    inputs: dict[str, pd.DataFrame],
) -> None:
    # Ensure ground truth was not exposed to earlier matching stages.
    for filename, dataframe in inputs.items():
        ground_truth_columns = [
            column
            for column in dataframe.columns
            if "ground_truth" in column.lower()
        ]
        if ground_truth_columns:
            detail = ", ".join(ground_truth_columns)
            raise ValueError(
                f"{filename} contains prohibited ground-truth columns: "
                f"{detail}"
            )


def load_matching_inputs(
    arguments: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    # Load consolidated, deterministic and probabilistic results.
    paths = {
        "records": arguments.input,
        "deterministic_pairwise": (
            arguments.deterministic_dir
            / DETERMINISTIC_PAIRWISE_FILENAME
        ),
        "deterministic_mapping": (
            arguments.deterministic_dir
            / DETERMINISTIC_MAPPING_FILENAME
        ),
        "deterministic_rejected": (
            arguments.deterministic_dir
            / DETERMINISTIC_REJECTED_FILENAME
        ),
        "probabilistic_candidates": (
            arguments.probabilistic_dir
            / PROBABILISTIC_CANDIDATES_FILENAME
        ),
        "probabilistic_scores": (
            arguments.probabilistic_dir
            / PROBABILISTIC_SCORES_FILENAME
        ),
        "probabilistic_accepted": (
            arguments.probabilistic_dir
            / PROBABILISTIC_ACCEPTED_FILENAME
        ),
        "probabilistic_mapping": (
            arguments.probabilistic_dir
            / PROBABILISTIC_MAPPING_FILENAME
        ),
        "probabilistic_summary": (
            arguments.probabilistic_dir
            / PROBABILISTIC_SUMMARY_FILENAME
        ),
    }
    inputs = {
        name: read_csv(path)
        for name, path in paths.items()
    }

    validate_required_columns(
        inputs["records"],
        CONSOLIDATED_REQUIRED_COLUMNS,
        str(paths["records"]),
    )
    validate_required_columns(
        inputs["deterministic_pairwise"],
        DETERMINISTIC_PAIRWISE_REQUIRED_COLUMNS,
        DETERMINISTIC_PAIRWISE_FILENAME,
    )
    validate_required_columns(
        inputs["deterministic_mapping"],
        DETERMINISTIC_MAPPING_REQUIRED_COLUMNS,
        DETERMINISTIC_MAPPING_FILENAME,
    )
    validate_required_columns(
        inputs["deterministic_rejected"],
        DETERMINISTIC_REJECTED_REQUIRED_COLUMNS,
        DETERMINISTIC_REJECTED_FILENAME,
    )
    validate_required_columns(
        inputs["probabilistic_candidates"],
        PROBABILISTIC_CANDIDATE_REQUIRED_COLUMNS,
        PROBABILISTIC_CANDIDATES_FILENAME,
    )
    validate_required_columns(
        inputs["probabilistic_scores"],
        PROBABILISTIC_SCORE_REQUIRED_COLUMNS,
        PROBABILISTIC_SCORES_FILENAME,
    )
    validate_required_columns(
        inputs["probabilistic_accepted"],
        PROBABILISTIC_SCORE_REQUIRED_COLUMNS,
        PROBABILISTIC_ACCEPTED_FILENAME,
    )
    validate_required_columns(
        inputs["probabilistic_mapping"],
        PROBABILISTIC_MAPPING_REQUIRED_COLUMNS,
        PROBABILISTIC_MAPPING_FILENAME,
    )
    validate_required_columns(
        inputs["probabilistic_summary"],
        SUMMARY_REQUIRED_COLUMNS,
        PROBABILISTIC_SUMMARY_FILENAME,
    )
    validate_matching_output_separation(inputs)

    return inputs


def load_ground_truth_mappings(
    ground_truth_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load and standardise the five source-specific protected mappings.
    rows: list[pd.DataFrame] = []
    validation_rows: list[dict[str, object]] = []

    for source_system, source_config in GROUND_TRUTH_SOURCES.items():
        filename, source_id_column = source_config
        path = ground_truth_directory / filename
        dataframe = read_csv(path)
        validate_required_columns(
            dataframe,
            {source_id_column, "ground_truth_id"},
            filename,
        )

        source_ids = dataframe[source_id_column].map(clean_value)
        ground_truth_ids = dataframe["ground_truth_id"].map(
            clean_value
        )
        if source_ids.eq("").any():
            raise ValueError(f"{filename} contains blank source IDs.")
        if not source_ids.is_unique:
            raise ValueError(f"{filename} contains duplicate source IDs.")
        if ground_truth_ids.eq("").any():
            raise ValueError(
                f"{filename} contains blank ground-truth IDs."
            )

        standardised = pd.DataFrame(
            {
                "source_system": source_system,
                "source_record_id": source_ids,
                "ground_truth_id": ground_truth_ids,
            }
        )
        rows.append(standardised)
        validation_rows.append(
            {
                "source_system": source_system,
                "mapping_filename": filename,
                "mapping_records": len(standardised),
                "unique_source_records": (
                    standardised["source_record_id"].nunique()
                ),
                "unique_ground_truth_identities": (
                    standardised["ground_truth_id"].nunique()
                ),
            }
        )

    combined = pd.concat(rows, ignore_index=True)
    if combined.duplicated(
        subset=["source_system", "source_record_id"]
    ).any():
        raise ValueError(
            "The combined ground-truth mappings contain duplicate source "
            "references."
        )

    validation = pd.DataFrame(validation_rows)
    return combined, validation


def validate_matching_inputs(
    inputs: dict[str, pd.DataFrame],
) -> None:
    # Reconcile records, mappings, candidates and scored decisions.
    records = inputs["records"]
    deterministic_mapping = inputs["deterministic_mapping"]
    probabilistic_mapping = inputs["probabilistic_mapping"]
    candidates = inputs["probabilistic_candidates"]
    scores = inputs["probabilistic_scores"]
    accepted = inputs["probabilistic_accepted"]

    if not records["staging_record_id"].is_unique:
        raise ValueError("Consolidated staging IDs are not unique.")

    input_ids = set(records["staging_record_id"])
    for name, mapping in (
        ("deterministic", deterministic_mapping),
        ("probabilistic", probabilistic_mapping),
    ):
        if not mapping["staging_record_id"].is_unique:
            raise ValueError(
                f"The {name} mapping contains duplicate staging IDs."
            )
        if set(mapping["staging_record_id"]) != input_ids:
            raise ValueError(
                f"The {name} mapping does not reconcile to the input."
            )

    candidate_pairs = {
        canonical_pair(left, right)
        for left, right in zip(
            candidates["cluster_id_1"],
            candidates["cluster_id_2"],
        )
    }
    score_pairs = {
        canonical_pair(left, right)
        for left, right in zip(
            scores["cluster_id_1"],
            scores["cluster_id_2"],
        )
    }
    if len(candidate_pairs) != len(candidates):
        raise ValueError("Probabilistic candidate pairs are not unique.")
    if candidate_pairs != score_pairs:
        raise ValueError(
            "Probabilistic candidates and scored pairs do not reconcile."
        )

    accepted_pairs = {
        canonical_pair(left, right)
        for left, right in zip(
            accepted["cluster_id_1"],
            accepted["cluster_id_2"],
        )
    }
    if not accepted_pairs.issubset(score_pairs):
        raise ValueError(
            "Accepted probabilistic matches contain an unknown pair."
        )


###############################################################################
# 6. Protected staging mapping and identity split
###############################################################################


def build_staging_ground_truth_mapping(
    records: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    # Join protected source mappings to stable staging identifiers.
    columns = [
        "staging_record_id",
        "source_system",
        "source_record_id",
    ]
    staging = records[columns].copy()
    for column in columns:
        staging[column] = staging[column].map(clean_value)

    record_references = set(
        zip(staging["source_system"], staging["source_record_id"])
    )
    ground_truth_references = set(
        zip(
            ground_truth["source_system"],
            ground_truth["source_record_id"],
        )
    )
    if record_references != ground_truth_references:
        missing = len(record_references - ground_truth_references)
        extra = len(ground_truth_references - record_references)
        raise ValueError(
            "Ground-truth source references do not match consolidated "
            f"records. Missing: {missing}; extra: {extra}."
        )

    output = staging.merge(
        ground_truth,
        on=["source_system", "source_record_id"],
        how="left",
        validate="one_to_one",
    )
    if output["ground_truth_id"].isna().any():
        missing = output.loc[
            output["ground_truth_id"].isna(),
            ["source_system", "source_record_id"],
        ].head(10)
        raise ValueError(
            "Ground truth is missing for consolidated source records: "
            f"{missing.to_dict(orient='records')}"
        )
    if len(output) != len(records):
        raise ValueError(
            "The staging ground-truth mapping does not reconcile."
        )

    return output


def attach_ground_truth(
    mapping: pd.DataFrame,
    staging_ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    # Attach protected labels while validating source-reference agreement.
    truth_columns = [
        "staging_record_id",
        "source_system",
        "source_record_id",
        "ground_truth_id",
    ]
    output = mapping.merge(
        staging_ground_truth[truth_columns],
        on=[
            "staging_record_id",
            "source_system",
            "source_record_id",
        ],
        how="left",
        validate="one_to_one",
    )
    if output["ground_truth_id"].isna().any():
        raise ValueError(
            "A resolution mapping contains unmatched source references."
        )

    return output


def build_identity_split(
    deterministic_mapping: pd.DataFrame,
    calibration_proportion: float,
    random_seed: int,
) -> pd.DataFrame:
    # Split identities rather than individual records to prevent leakage.
    eligible = deterministic_mapping.loc[
        ~deterministic_mapping["resolution_status"].eq(
            ANONYMOUS_STATUS
        )
    ]
    eligible_identities = sorted(
        eligible["ground_truth_id"].unique()
    )
    all_identities = sorted(
        deterministic_mapping["ground_truth_id"].unique()
    )
    random_generator = random.Random(random_seed)
    random_generator.shuffle(eligible_identities)

    calibration_count = int(
        round(len(eligible_identities) * calibration_proportion)
    )
    calibration_ids = set(
        eligible_identities[:calibration_count]
    )
    eligible_ids = set(eligible_identities)
    rows = [
        {
            "ground_truth_id": ground_truth_id,
            "evaluation_split": (
                CALIBRATION_SPLIT
                if ground_truth_id in calibration_ids
                else EVALUATION_SPLIT
                if ground_truth_id in eligible_ids
                else ANONYMOUS_ONLY_SPLIT
            ),
        }
        for ground_truth_id in all_identities
    ]

    return pd.DataFrame(
        rows,
        columns=["ground_truth_id", "evaluation_split"],
    )


def attach_identity_split(
    dataframe: pd.DataFrame,
    identity_split: pd.DataFrame,
) -> pd.DataFrame:
    # Attach the reproducible identity-level evaluation split.
    output = dataframe.merge(
        identity_split,
        on="ground_truth_id",
        how="left",
        validate="many_to_one",
    )
    if output["evaluation_split"].isna().any():
        raise ValueError(
            "An eligible ground-truth identity has no evaluation split."
        )

    return output


###############################################################################
# 7. Deterministic-cluster truth profiles and candidate labels
###############################################################################


def build_cluster_truth_sets(
    deterministic_mapping: pd.DataFrame,
) -> dict[str, set[str]]:
    # Record every protected identity represented in each DCL cluster.
    eligible = deterministic_mapping.loc[
        ~deterministic_mapping["resolution_status"].eq(
            ANONYMOUS_STATUS
        )
    ]
    grouped = eligible.groupby("provisional_cluster_id")[
        "ground_truth_id"
    ].agg(set)

    return {
        clean_value(cluster_id): {
            clean_value(value)
            for value in values
            if clean_value(value)
        }
        for cluster_id, values in grouped.items()
    }


def build_cluster_split_sets(
    cluster_truth_sets: dict[str, set[str]],
    split_lookup: dict[str, str],
) -> dict[str, set[str]]:
    # Record which evaluation subsets are represented in each DCL cluster.
    return {
        cluster_id: {
            split_lookup[ground_truth_id]
            for ground_truth_id in ground_truth_ids
        }
        for cluster_id, ground_truth_ids in cluster_truth_sets.items()
    }


def candidate_evaluation_scope(
    left_splits: set[str],
    right_splits: set[str],
) -> str:
    # Assign candidates without using cross-split pairs for calibration.
    combined = left_splits | right_splits
    if combined == {CALIBRATION_SPLIT}:
        return CALIBRATION_SPLIT
    if EVALUATION_SPLIT in combined:
        return EVALUATION_SPLIT

    return "excluded"


def prepare_score_columns(scores: pd.DataFrame) -> pd.DataFrame:
    # Convert scoring fields into stable numeric and Boolean values.
    output = scores.copy()
    numeric_columns = [
        "weighted_similarity_score",
        "evidence_coverage",
        "supporting_feature_count",
        "corroborating_feature_count",
    ]
    for column in numeric_columns:
        output[column] = pd.to_numeric(
            output[column],
            errors="raise",
        )

    output["static_conflict"] = output["static_conflict"].map(
        parse_boolean
    )
    output["merge_performed"] = output["merge_performed"].map(
        parse_boolean
    )
    output["sufficient_support"] = (
        output["strong_identifiers"].map(clean_value).ne("")
        & output["corroborating_feature_count"].ge(1)
    ) | output["supporting_feature_count"].ge(3)

    return output


def annotate_candidate_truth(
    scores: pd.DataFrame,
    cluster_truth_sets: dict[str, set[str]],
    cluster_split_sets: dict[str, set[str]],
    staging_truth_lookup: dict[str, str],
) -> pd.DataFrame:
    # Attach protected labels after all matching scores have been frozen.
    output = scores.copy()
    truth_ids_1: list[str] = []
    truth_ids_2: list[str] = []
    shared_truth_ids: list[str] = []
    true_matches: list[bool] = []
    scopes: list[str] = []
    best_pair_labels: list[bool] = []

    for row in output.itertuples(index=False):
        left_cluster = clean_value(row.cluster_id_1)
        right_cluster = clean_value(row.cluster_id_2)
        if left_cluster not in cluster_truth_sets:
            raise ValueError(
                f"Unknown deterministic cluster: {left_cluster}"
            )
        if right_cluster not in cluster_truth_sets:
            raise ValueError(
                f"Unknown deterministic cluster: {right_cluster}"
            )

        left_truth = cluster_truth_sets[left_cluster]
        right_truth = cluster_truth_sets[right_cluster]
        shared = left_truth & right_truth
        left_record_truth = staging_truth_lookup[
            clean_value(row.record_id_1)
        ]
        right_record_truth = staging_truth_lookup[
            clean_value(row.record_id_2)
        ]

        truth_ids_1.append(serialise_values(left_truth))
        truth_ids_2.append(serialise_values(right_truth))
        shared_truth_ids.append(serialise_values(shared))
        true_matches.append(bool(shared))
        scopes.append(
            candidate_evaluation_scope(
                cluster_split_sets[left_cluster],
                cluster_split_sets[right_cluster],
            )
        )
        best_pair_labels.append(
            left_record_truth == right_record_truth
        )

    output["ground_truth_ids_1"] = truth_ids_1
    output["ground_truth_ids_2"] = truth_ids_2
    output["shared_ground_truth_ids"] = shared_truth_ids
    output["is_true_cluster_match"] = true_matches
    output["evaluation_scope"] = scopes
    output["best_record_pair_true_match"] = best_pair_labels

    return output


def annotate_relationship_truth(
    relationships: pd.DataFrame,
    annotated_scores: pd.DataFrame,
) -> pd.DataFrame:
    # Join candidate truth labels to a probabilistic relationship subset.
    truth_columns = [
        "cluster_id_1",
        "cluster_id_2",
        "ground_truth_ids_1",
        "ground_truth_ids_2",
        "shared_ground_truth_ids",
        "is_true_cluster_match",
        "evaluation_scope",
        "best_record_pair_true_match",
    ]
    output = relationships.merge(
        annotated_scores[truth_columns],
        on=["cluster_id_1", "cluster_id_2"],
        how="left",
        validate="one_to_one",
    )
    if output["is_true_cluster_match"].isna().any():
        raise ValueError(
            "A probabilistic relationship has no candidate truth label."
        )

    return output


###############################################################################
# 8. True cross-cluster pairs and blocking evaluation
###############################################################################


def build_true_cross_cluster_pairs(
    deterministic_mapping: pd.DataFrame,
) -> pd.DataFrame:
    # Enumerate the DCL pairs that should belong to one true identity.
    eligible = deterministic_mapping.loc[
        ~deterministic_mapping["resolution_status"].eq(
            ANONYMOUS_STATUS
        )
    ].copy()
    counts = (
        eligible.groupby(
            [
                "ground_truth_id",
                "evaluation_split",
                "provisional_cluster_id",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "identity_records_in_cluster"})
    )
    rows: list[dict[str, object]] = []

    for ground_truth_id, group in counts.groupby("ground_truth_id"):
        cluster_counts = {
            clean_value(row.provisional_cluster_id): int(
                row.identity_records_in_cluster
            )
            for row in group.itertuples(index=False)
        }
        split = clean_value(group["evaluation_split"].iloc[0])
        for left_cluster, right_cluster in combinations(
            sorted(cluster_counts),
            2,
        ):
            rows.append(
                {
                    "ground_truth_id": ground_truth_id,
                    "evaluation_split": split,
                    "cluster_id_1": left_cluster,
                    "cluster_id_2": right_cluster,
                    "identity_records_in_cluster_1": (
                        cluster_counts[left_cluster]
                    ),
                    "identity_records_in_cluster_2": (
                        cluster_counts[right_cluster]
                    ),
                    "true_record_pair_opportunities": (
                        cluster_counts[left_cluster]
                        * cluster_counts[right_cluster]
                    ),
                }
            )

    columns = [
        "ground_truth_id",
        "evaluation_split",
        "cluster_id_1",
        "cluster_id_2",
        "identity_records_in_cluster_1",
        "identity_records_in_cluster_2",
        "true_record_pair_opportunities",
    ]
    return pd.DataFrame(rows, columns=columns)


def add_candidate_generation_status(
    true_pairs: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    # Mark which genuine cross-cluster opportunities survived blocking.
    candidate_subset = candidates.copy()
    candidate_subset["candidate_generated"] = True
    output = true_pairs.merge(
        candidate_subset,
        on=["cluster_id_1", "cluster_id_2"],
        how="left",
        validate="many_to_one",
    )
    output["candidate_generated"] = output[
        "candidate_generated"
    ].map(parse_boolean)

    return output


def possible_pairs_for_scope(
    deterministic_mapping: pd.DataFrame,
    scope: str,
) -> int:
    # Calculate possible DCL comparisons for an evaluation subset.
    eligible = deterministic_mapping.loc[
        ~deterministic_mapping["resolution_status"].eq(
            ANONYMOUS_STATUS
        )
    ]
    total_clusters = eligible["provisional_cluster_id"].nunique()
    if scope == "all":
        return pair_count(total_clusters)

    calibration_clusters = eligible.loc[
        eligible["evaluation_split"].eq(CALIBRATION_SPLIT),
        "provisional_cluster_id",
    ].nunique()
    if scope == CALIBRATION_SPLIT:
        return pair_count(calibration_clusters)

    return pair_count(total_clusters) - pair_count(
        calibration_clusters
    )


def build_blocking_evaluation(
    true_pairs: pd.DataFrame,
    annotated_candidates: pd.DataFrame,
    deterministic_mapping: pd.DataFrame,
) -> pd.DataFrame:
    # Evaluate blocking recall alongside comparison reduction.
    rows: list[dict[str, object]] = []

    for scope in ("all", CALIBRATION_SPLIT, EVALUATION_SPLIT):
        if scope == "all":
            scoped_true = true_pairs
            scoped_candidates = annotated_candidates
        else:
            scoped_true = true_pairs.loc[
                true_pairs["evaluation_split"].eq(scope)
            ]
            scoped_candidates = annotated_candidates.loc[
                annotated_candidates["evaluation_scope"].eq(scope)
            ]

        generated = scoped_true.loc[
            scoped_true["candidate_generated"]
        ]
        total_true_cluster_pairs = len(scoped_true)
        generated_true_cluster_pairs = len(generated)
        total_true_record_pairs = int(
            scoped_true["true_record_pair_opportunities"].sum()
        )
        generated_true_record_pairs = int(
            generated["true_record_pair_opportunities"].sum()
        )
        possible_pairs = possible_pairs_for_scope(
            deterministic_mapping,
            scope,
        )
        candidate_pairs = len(scoped_candidates)

        rows.append(
            {
                "evaluation_scope": scope,
                "possible_cluster_pairs": possible_pairs,
                "candidate_cluster_pairs": candidate_pairs,
                "candidate_reduction_ratio": round(
                    1 - safe_divide(candidate_pairs, possible_pairs),
                    6,
                ),
                "true_cross_cluster_pairs": (
                    total_true_cluster_pairs
                ),
                "generated_true_cluster_pairs": (
                    generated_true_cluster_pairs
                ),
                "cluster_pair_blocking_recall": round(
                    safe_divide(
                        generated_true_cluster_pairs,
                        total_true_cluster_pairs,
                    ),
                    6,
                ),
                "true_cross_cluster_record_pairs": (
                    total_true_record_pairs
                ),
                "generated_true_cross_cluster_record_pairs": (
                    generated_true_record_pairs
                ),
                "record_pair_blocking_recall": round(
                    safe_divide(
                        generated_true_record_pairs,
                        total_true_record_pairs,
                    ),
                    6,
                ),
            }
        )

    return pd.DataFrame(rows)


###############################################################################
# 9. Threshold and evidence-coverage calibration
###############################################################################


def automatic_match_mask(
    scores: pd.DataFrame,
    automatic_threshold: float,
    minimum_coverage: float,
) -> pd.Series:
    # Reapply automatic criteria without consulting ground truth.
    return (
        ~scores["static_conflict"]
        & scores["sufficient_support"]
        & scores["weighted_similarity_score"].ge(
            automatic_threshold
        )
        & scores["evidence_coverage"].ge(minimum_coverage)
    )


def build_calibration_results(
    annotated_scores: pd.DataFrame,
    true_pairs: pd.DataFrame,
) -> pd.DataFrame:
    # Test reproducible threshold and evidence-coverage combinations.
    calibration_scores = annotated_scores.loc[
        annotated_scores["evaluation_scope"].eq(
            CALIBRATION_SPLIT
        )
    ]
    calibration_true_pairs = true_pairs.loc[
        true_pairs["evaluation_split"].eq(CALIBRATION_SPLIT)
    ]
    true_candidate_count = int(
        calibration_scores["is_true_cluster_match"].sum()
    )
    total_true_count = len(calibration_true_pairs)
    missed_by_blocking = total_true_count - true_candidate_count
    rows: list[dict[str, object]] = []

    for automatic_threshold in AUTOMATIC_THRESHOLDS:
        for minimum_coverage in EVIDENCE_COVERAGE_THRESHOLDS:
            predicted = automatic_match_mask(
                calibration_scores,
                automatic_threshold,
                minimum_coverage,
            )
            truth = calibration_scores["is_true_cluster_match"]
            true_positives = int((predicted & truth).sum())
            false_positives = int((predicted & ~truth).sum())
            predicted_matches = true_positives + false_positives
            scored_false_negatives = (
                true_candidate_count - true_positives
            )
            end_to_end_false_negatives = (
                total_true_count - true_positives
            )
            precision = safe_divide(
                true_positives,
                predicted_matches,
            )
            scored_recall = safe_divide(
                true_positives,
                true_candidate_count,
            )
            end_to_end_recall = safe_divide(
                true_positives,
                total_true_count,
            )

            rows.append(
                {
                    "automatic_match_threshold": (
                        automatic_threshold
                    ),
                    "minimum_evidence_coverage": minimum_coverage,
                    "predicted_matches": predicted_matches,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "scored_false_negatives": (
                        scored_false_negatives
                    ),
                    "missed_by_blocking": missed_by_blocking,
                    "end_to_end_false_negatives": (
                        end_to_end_false_negatives
                    ),
                    "precision": round(precision, 6),
                    "scored_candidate_recall": round(
                        scored_recall,
                        6,
                    ),
                    "end_to_end_recall": round(
                        end_to_end_recall,
                        6,
                    ),
                    "end_to_end_f1": round(
                        harmonic_mean(
                            precision,
                            end_to_end_recall,
                        ),
                        6,
                    ),
                }
            )

    return pd.DataFrame(rows)


def select_configuration(
    calibration_results: pd.DataFrame,
    minimum_precision: float,
    minimum_matches: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    # Maximise recall subject to the precision-first constraint.
    eligible = calibration_results.loc[
        calibration_results["precision"].ge(minimum_precision)
        & calibration_results["predicted_matches"].ge(
            minimum_matches
        )
    ].copy()

    if eligible.empty:
        candidates = calibration_results.loc[
            calibration_results["predicted_matches"].ge(
                minimum_matches
            )
        ].copy()
        if candidates.empty:
            raise ValueError(
                "No calibration configuration produced the minimum "
                "number of matches."
            )
        selection_status = (
            "Precision target not achieved; highest-precision "
            "configuration selected"
        )
        ordered = candidates.sort_values(
            by=[
                "precision",
                "end_to_end_recall",
                "end_to_end_f1",
                "minimum_evidence_coverage",
                "automatic_match_threshold",
            ],
            ascending=[False, False, False, False, False],
            kind="stable",
        )
    else:
        selection_status = (
            "Precision target achieved; recall maximised subject to "
            "the target"
        )
        ordered = eligible.sort_values(
            by=[
                "end_to_end_recall",
                "precision",
                "end_to_end_f1",
                "predicted_matches",
                "minimum_evidence_coverage",
                "automatic_match_threshold",
            ],
            ascending=[False, False, False, False, False, False],
            kind="stable",
        )

    selected = ordered.iloc[0].to_dict()
    selected["selection_status"] = selection_status
    selected["minimum_precision_target"] = minimum_precision
    selected["minimum_calibration_matches"] = minimum_matches

    configuration_rows: list[dict[str, object]] = [
        {
            "parameter": "selection_status",
            "value": selection_status,
        },
        {
            "parameter": "minimum_precision_target",
            "value": minimum_precision,
        },
        {
            "parameter": "minimum_calibration_matches",
            "value": minimum_matches,
        },
        {
            "parameter": "automatic_match_threshold",
            "value": selected["automatic_match_threshold"],
        },
        {
            "parameter": "minimum_evidence_coverage",
            "value": selected["minimum_evidence_coverage"],
        },
        {
            "parameter": "calibration_precision",
            "value": selected["precision"],
        },
        {
            "parameter": "calibration_end_to_end_recall",
            "value": selected["end_to_end_recall"],
        },
        {
            "parameter": "calibration_end_to_end_f1",
            "value": selected["end_to_end_f1"],
        },
    ]
    configuration_rows.extend(
        {
            "parameter": f"{feature}_weight",
            "value": weight,
        }
        for feature, weight in FEATURE_WEIGHTS.items()
    )

    return pd.DataFrame(configuration_rows), selected


###############################################################################
# 10. Frozen-configuration cluster simulation
###############################################################################


def populated_values(series: pd.Series) -> set[str]:
    # Return populated canonical values from a cluster field.
    return {
        clean_value(value)
        for value in series
        if clean_value(value)
    }


def build_cluster_profiles(
    records: pd.DataFrame,
    deterministic_mapping: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    # Build ground-truth-free profiles for dynamic merge safeguards.
    eligible_mapping = deterministic_mapping.loc[
        ~deterministic_mapping["resolution_status"].eq(
            ANONYMOUS_STATUS
        ),
        ["staging_record_id", "provisional_cluster_id"],
    ]
    profile_records = eligible_mapping.merge(
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
    profiles: dict[str, dict[str, object]] = {}

    for cluster_id, group in profile_records.groupby(
        "provisional_cluster_id",
        sort=True,
    ):
        profiles[clean_value(cluster_id)] = {
            "record_count": len(group),
            "minimum_staging_id": min(group["staging_record_id"]),
            "portal_ids": populated_values(group["portal_user_id"]),
            "dates_of_birth": populated_values(
                group["date_of_birth_normalised"]
            ),
        }

    return profiles


def build_prohibited_pairs(
    deterministic_rejected: pd.DataFrame,
) -> set[tuple[str, str]]:
    # Preserve every deterministic cluster rejection as a constraint.
    prohibited: set[tuple[str, str]] = set()
    for row in deterministic_rejected.itertuples(index=False):
        left_cluster = clean_value(row.provisional_cluster_id_1)
        right_cluster = clean_value(row.provisional_cluster_id_2)
        if left_cluster and right_cluster:
            prohibited.add(
                canonical_pair(left_cluster, right_cluster)
            )

    return prohibited


def roots_violate_prohibited_pair(
    left_clusters: set[str],
    right_clusters: set[str],
    prohibited_pairs: set[tuple[str, str]],
) -> bool:
    # Check every DCL pairing across two evolving roots.
    for left_cluster in left_clusters:
        for right_cluster in right_clusters:
            if canonical_pair(
                left_cluster,
                right_cluster,
            ) in prohibited_pairs:
                return True

    return False


def dynamic_conflict_reason(
    union_find: EvaluationUnionFind,
    left_root: str,
    right_root: str,
    prohibited_pairs: set[tuple[str, str]],
) -> str:
    # Reapply trusted conflict safeguards before each selected merge.
    left_portals = union_find.portal_ids[left_root]
    right_portals = union_find.portal_ids[right_root]
    if left_portals and right_portals:
        if left_portals.isdisjoint(right_portals):
            return (
                "Conflicting trusted portal_user_id values after "
                "merging"
            )

    left_dobs = union_find.dates_of_birth[left_root]
    right_dobs = union_find.dates_of_birth[right_root]
    if left_dobs and right_dobs and left_dobs.isdisjoint(right_dobs):
        return (
            "Conflicting known date_of_birth_normalised values after "
            "merging"
        )

    if roots_violate_prohibited_pair(
        union_find.deterministic_clusters[left_root],
        union_find.deterministic_clusters[right_root],
        prohibited_pairs,
    ):
        return "Merge would violate a deterministic rejection constraint"

    return ""


def simulate_selected_configuration(
    annotated_scores: pd.DataFrame,
    cluster_profiles: dict[str, dict[str, object]],
    prohibited_pairs: set[tuple[str, str]],
    automatic_threshold: float,
    minimum_coverage: float,
) -> tuple[dict[str, str], pd.DataFrame]:
    # Simulate frozen automatic links without using protected labels.
    cluster_ids = sorted(cluster_profiles)
    union_find = EvaluationUnionFind(
        cluster_ids,
        cluster_profiles,
    )
    selected_mask = automatic_match_mask(
        annotated_scores,
        automatic_threshold,
        minimum_coverage,
    )
    selected = annotated_scores.loc[selected_mask].copy()
    selected = selected.sort_values(
        by=[
            "weighted_similarity_score",
            "evidence_coverage",
            "cluster_id_1",
            "cluster_id_2",
        ],
        ascending=[False, False, True, True],
        kind="stable",
    )
    decisions: list[dict[str, object]] = []

    for row in selected.itertuples(index=False):
        left_cluster = clean_value(row.cluster_id_1)
        right_cluster = clean_value(row.cluster_id_2)
        left_root = union_find.find(left_cluster)
        right_root = union_find.find(right_cluster)
        decision = ""
        reason = ""
        merge_performed = False

        if left_root == right_root:
            decision = "accepted_transitive_support"
            reason = (
                "Clusters were already connected by stronger accepted "
                "matches"
            )
        else:
            conflict = dynamic_conflict_reason(
                union_find,
                left_root,
                right_root,
                prohibited_pairs,
            )
            if conflict:
                decision = "rejected_dynamic_conflict"
                reason = conflict
            else:
                union_find.union(left_root, right_root)
                decision = "accepted_merge"
                reason = (
                    "Frozen automatic criteria and cluster safeguards "
                    "satisfied"
                )
                merge_performed = True

        decisions.append(
            {
                "cluster_id_1": left_cluster,
                "cluster_id_2": right_cluster,
                "weighted_similarity_score": (
                    row.weighted_similarity_score
                ),
                "evidence_coverage": row.evidence_coverage,
                "blocking_rules": row.blocking_rules,
                "record_id_1": row.record_id_1,
                "record_id_2": row.record_id_2,
                "ground_truth_ids_1": row.ground_truth_ids_1,
                "ground_truth_ids_2": row.ground_truth_ids_2,
                "shared_ground_truth_ids": (
                    row.shared_ground_truth_ids
                ),
                "is_true_cluster_match": (
                    row.is_true_cluster_match
                ),
                "evaluation_scope": row.evaluation_scope,
                "best_record_pair_true_match": (
                    row.best_record_pair_true_match
                ),
                "selected_decision": decision,
                "selected_reason": reason,
                "merge_performed": merge_performed,
            }
        )

    roots: dict[str, set[str]] = defaultdict(set)
    for cluster_id in cluster_ids:
        roots[union_find.find(cluster_id)].add(cluster_id)

    ordered_roots = sorted(
        roots,
        key=lambda root: min(
            clean_value(
                cluster_profiles[cluster_id][
                    "minimum_staging_id"
                ]
            )
            for cluster_id in roots[root]
        ),
    )
    cluster_lookup: dict[str, str] = {}
    for number, root in enumerate(ordered_roots, start=1):
        evaluated_cluster_id = f"ECL{number:06d}"
        for cluster_id in roots[root]:
            cluster_lookup[cluster_id] = evaluated_cluster_id

    decision_columns = [
        "cluster_id_1",
        "cluster_id_2",
        "weighted_similarity_score",
        "evidence_coverage",
        "blocking_rules",
        "record_id_1",
        "record_id_2",
        "ground_truth_ids_1",
        "ground_truth_ids_2",
        "shared_ground_truth_ids",
        "is_true_cluster_match",
        "evaluation_scope",
        "best_record_pair_true_match",
        "selected_decision",
        "selected_reason",
        "merge_performed",
    ]
    decision_output = pd.DataFrame(
        decisions,
        columns=decision_columns,
    )
    if not decision_output.empty:
        decision_output["evaluated_cluster_id_1"] = (
            decision_output["cluster_id_1"].map(cluster_lookup)
        )
        decision_output["evaluated_cluster_id_2"] = (
            decision_output["cluster_id_2"].map(cluster_lookup)
        )
    else:
        decision_output["evaluated_cluster_id_1"] = pd.Series(
            dtype=str
        )
        decision_output["evaluated_cluster_id_2"] = pd.Series(
            dtype=str
        )

    return cluster_lookup, decision_output


def build_selected_record_mapping(
    deterministic_mapping: pd.DataFrame,
    cluster_lookup: dict[str, str],
) -> pd.DataFrame:
    # Build an evaluation-only mapping for the frozen configuration.
    output = deterministic_mapping[
        [
            "staging_record_id",
            "source_system",
            "source_record_id",
            "ground_truth_id",
            "evaluation_split",
            "provisional_cluster_id",
            "resolution_status",
        ]
    ].copy()
    output = output.rename(
        columns={
            "provisional_cluster_id": "deterministic_cluster_id",
            "resolution_status": "deterministic_resolution_status",
        }
    )
    anonymous = output["deterministic_resolution_status"].eq(
        ANONYMOUS_STATUS
    )
    output["evaluated_cluster_id"] = output[
        "deterministic_cluster_id"
    ].map(cluster_lookup)
    output.loc[anonymous, "evaluated_cluster_id"] = ""

    eligible = output.loc[~anonymous]
    cluster_sizes = eligible["evaluated_cluster_id"].value_counts()
    dcl_counts = eligible.groupby("evaluated_cluster_id")[
        "deterministic_cluster_id"
    ].nunique()
    output["cluster_size"] = output["evaluated_cluster_id"].map(
        lambda cluster_id: int(cluster_sizes[cluster_id])
        if clean_value(cluster_id)
        else ""
    )
    output["deterministic_cluster_count"] = output[
        "evaluated_cluster_id"
    ].map(
        lambda cluster_id: int(dcl_counts[cluster_id])
        if clean_value(cluster_id)
        else ""
    )

    output["evaluation_resolution_status"] = ""
    output.loc[anonymous, "evaluation_resolution_status"] = (
        ANONYMOUS_STATUS
    )
    probabilistically_linked = (
        ~anonymous
        & pd.to_numeric(
            output["deterministic_cluster_count"],
            errors="coerce",
        ).gt(1)
    )
    output.loc[
        probabilistically_linked,
        "evaluation_resolution_status",
    ] = "probabilistically_linked_selected"
    deterministic_only = (
        ~anonymous
        & ~probabilistically_linked
        & output["deterministic_resolution_status"].eq(
            "deterministically_linked"
        )
    )
    output.loc[
        deterministic_only,
        "evaluation_resolution_status",
    ] = "deterministically_linked"
    unresolved = (
        ~anonymous
        & output["evaluation_resolution_status"].eq("")
    )
    output.loc[
        unresolved,
        "evaluation_resolution_status",
    ] = "unresolved_singleton"

    return output


###############################################################################
# 11. Pairwise and cluster quality metrics
###############################################################################


def scoped_cluster_metrics(
    mapping: pd.DataFrame,
    predicted_cluster_column: str,
    scope: str,
) -> dict[str, object]:
    # Calculate pairwise, B-cubed and cluster-level quality measures.
    eligible = mapping.loc[
        mapping[predicted_cluster_column].map(clean_value).ne("")
    ].copy()
    if scope == "all":
        scoped_mask = pd.Series(True, index=eligible.index)
    else:
        scoped_mask = eligible["evaluation_split"].eq(scope)

    scoped = eligible.loc[scoped_mask]
    if scoped.empty:
        raise ValueError(f"No eligible records exist for scope: {scope}")

    total_predicted_sizes = eligible.groupby(
        predicted_cluster_column
    ).size()
    non_scope = eligible.loc[~scoped_mask]
    non_scope_sizes = non_scope.groupby(
        predicted_cluster_column
    ).size()
    predicted_pairs = sum(
        pair_count(int(size))
        - pair_count(int(non_scope_sizes.get(cluster_id, 0)))
        for cluster_id, size in total_predicted_sizes.items()
    )

    contingency = (
        scoped.groupby(
            [predicted_cluster_column, "ground_truth_id"]
        )
        .size()
        .rename("cell_size")
        .reset_index()
    )
    true_positive_pairs = int(
        contingency["cell_size"].map(
            lambda value: pair_count(int(value))
        ).sum()
    )
    truth_sizes = scoped.groupby("ground_truth_id").size()
    actual_pairs = int(
        truth_sizes.map(lambda value: pair_count(int(value))).sum()
    )
    false_positive_pairs = predicted_pairs - true_positive_pairs
    false_negative_pairs = actual_pairs - true_positive_pairs
    pairwise_precision = safe_divide(
        true_positive_pairs,
        predicted_pairs,
    )
    pairwise_recall = safe_divide(
        true_positive_pairs,
        actual_pairs,
    )

    contingency["predicted_cluster_size"] = contingency[
        predicted_cluster_column
    ].map(total_predicted_sizes)
    contingency["ground_truth_size"] = contingency[
        "ground_truth_id"
    ].map(truth_sizes)
    record_count = len(scoped)
    bcubed_precision = safe_divide(
        float(
            (
                contingency["cell_size"] ** 2
                / contingency["predicted_cluster_size"]
            ).sum()
        ),
        record_count,
    )
    bcubed_recall = safe_divide(
        float(
            (
                contingency["cell_size"] ** 2
                / contingency["ground_truth_size"]
            ).sum()
        ),
        record_count,
    )

    scoped_contingency = contingency.groupby(
        predicted_cluster_column
    )["cell_size"].max()
    weighted_purity = safe_divide(
        float(scoped_contingency.sum()),
        record_count,
    )
    touching_clusters = set(scoped[predicted_cluster_column])
    full_cluster_truth_counts = eligible.groupby(
        predicted_cluster_column
    )["ground_truth_id"].nunique()
    impure_clusters = int(
        full_cluster_truth_counts.loc[
            full_cluster_truth_counts.index.isin(touching_clusters)
        ].gt(1).sum()
    )
    split_identities = int(
        scoped.groupby("ground_truth_id")[
            predicted_cluster_column
        ].nunique().gt(1).sum()
    )

    return {
        "evaluation_scope": scope,
        "eligible_records": record_count,
        "ground_truth_identities": scoped[
            "ground_truth_id"
        ].nunique(),
        "predicted_clusters_touching_scope": len(touching_clusters),
        "true_positive_pairs": true_positive_pairs,
        "false_positive_pairs": false_positive_pairs,
        "false_negative_pairs": false_negative_pairs,
        "pairwise_precision": round(pairwise_precision, 6),
        "pairwise_recall": round(pairwise_recall, 6),
        "pairwise_f1": round(
            harmonic_mean(pairwise_precision, pairwise_recall),
            6,
        ),
        "bcubed_precision": round(bcubed_precision, 6),
        "bcubed_recall": round(bcubed_recall, 6),
        "bcubed_f1": round(
            harmonic_mean(bcubed_precision, bcubed_recall),
            6,
        ),
        "weighted_cluster_purity": round(weighted_purity, 6),
        "impure_clusters_touching_scope": impure_clusters,
        "split_ground_truth_identities": split_identities,
    }


def build_cluster_evaluation(
    deterministic_mapping: pd.DataFrame,
    probabilistic_mapping: pd.DataFrame,
    selected_mapping: pd.DataFrame,
) -> pd.DataFrame:
    # Compare deterministic, current PCL and frozen selected clusters.
    stages = (
        (
            "deterministic",
            deterministic_mapping,
            "provisional_cluster_id",
        ),
        (
            "current_probabilistic",
            probabilistic_mapping,
            "probabilistic_cluster_id",
        ),
        (
            "selected_configuration",
            selected_mapping,
            "evaluated_cluster_id",
        ),
    )
    rows: list[dict[str, object]] = []

    for stage, mapping, cluster_column in stages:
        for scope in ("all", CALIBRATION_SPLIT, EVALUATION_SPLIT):
            metrics = scoped_cluster_metrics(
                mapping,
                cluster_column,
                scope,
            )
            rows.append({"resolution_stage": stage, **metrics})

    return pd.DataFrame(rows)


def relationship_metrics(
    relationships: pd.DataFrame,
    stage: str,
    relationship_type: str,
) -> list[dict[str, object]]:
    # Calculate protected precision for accepted cluster relationships.
    rows: list[dict[str, object]] = []
    for scope in ("all", CALIBRATION_SPLIT, EVALUATION_SPLIT):
        if scope == "all":
            scoped = relationships
        else:
            scoped = relationships.loc[
                relationships["evaluation_scope"].eq(scope)
            ]
        true_matches = int(scoped["is_true_cluster_match"].sum())
        false_matches = len(scoped) - true_matches
        rows.append(
            {
                "resolution_stage": stage,
                "relationship_type": relationship_type,
                "evaluation_scope": scope,
                "accepted_relationships": len(scoped),
                "true_positive_relationships": true_matches,
                "false_positive_relationships": false_matches,
                "relationship_precision": round(
                    safe_divide(true_matches, len(scoped)),
                    6,
                ),
            }
        )

    return rows


def annotate_deterministic_pairwise_truth(
    pairwise: pd.DataFrame,
    staging_truth: pd.DataFrame,
    split_lookup: dict[str, str],
) -> pd.DataFrame:
    # Label deterministic audit edges using protected record identities.
    truth_lookup = staging_truth.set_index("staging_record_id")[
        "ground_truth_id"
    ].to_dict()
    output = pairwise.copy()
    output["ground_truth_id_1"] = output["record_id_1"].map(
        truth_lookup
    )
    output["ground_truth_id_2"] = output["record_id_2"].map(
        truth_lookup
    )
    if output[
        ["ground_truth_id_1", "ground_truth_id_2"]
    ].isna().any().any():
        raise ValueError(
            "A deterministic edge references an unknown staging record."
        )

    output["is_true_match"] = output["ground_truth_id_1"].eq(
        output["ground_truth_id_2"]
    )
    split_1 = output["ground_truth_id_1"].map(split_lookup)
    split_2 = output["ground_truth_id_2"].map(split_lookup)
    output["evaluation_scope"] = EVALUATION_SPLIT
    output.loc[
        split_1.eq(CALIBRATION_SPLIT)
        & split_2.eq(CALIBRATION_SPLIT),
        "evaluation_scope",
    ] = CALIBRATION_SPLIT

    return output


def build_deterministic_evaluation(
    deterministic_pairwise: pd.DataFrame,
    cluster_evaluation: pd.DataFrame,
) -> pd.DataFrame:
    # Combine deterministic relationship and cluster-quality results.
    deterministic_clusters = cluster_evaluation.loc[
        cluster_evaluation["resolution_stage"].eq("deterministic")
    ].copy()
    relationship_rows: list[dict[str, object]] = []

    for scope in ("all", CALIBRATION_SPLIT, EVALUATION_SPLIT):
        if scope == "all":
            scoped = deterministic_pairwise
        else:
            scoped = deterministic_pairwise.loc[
                deterministic_pairwise["evaluation_scope"].eq(scope)
            ]
        true_edges = int(scoped["is_true_match"].sum())
        relationship_rows.append(
            {
                "evaluation_scope": scope,
                "accepted_deterministic_edges": len(scoped),
                "true_deterministic_edges": true_edges,
                "false_deterministic_edges": len(scoped) - true_edges,
                "deterministic_edge_precision": round(
                    safe_divide(true_edges, len(scoped)),
                    6,
                ),
            }
        )

    relationships = pd.DataFrame(relationship_rows)
    output = deterministic_clusters.merge(
        relationships,
        on="evaluation_scope",
        how="left",
        validate="one_to_one",
    )
    return output.drop(columns="resolution_stage")


def selected_candidate_evaluation(
    annotated_scores: pd.DataFrame,
    true_pairs: pd.DataFrame,
    automatic_threshold: float,
    minimum_coverage: float,
) -> list[dict[str, object]]:
    # Evaluate frozen automatic criteria on calibration and held-out pairs.
    rows: list[dict[str, object]] = []
    selected_mask = automatic_match_mask(
        annotated_scores,
        automatic_threshold,
        minimum_coverage,
    )

    for scope in (CALIBRATION_SPLIT, EVALUATION_SPLIT):
        scoped_scores = annotated_scores.loc[
            annotated_scores["evaluation_scope"].eq(scope)
        ]
        scoped_selected = selected_mask.loc[scoped_scores.index]
        truth = scoped_scores["is_true_cluster_match"]
        total_true_pairs = int(
            true_pairs["evaluation_split"].eq(scope).sum()
        )
        true_candidate_pairs = int(truth.sum())
        true_positives = int((scoped_selected & truth).sum())
        false_positives = int((scoped_selected & ~truth).sum())
        predicted = true_positives + false_positives
        precision = safe_divide(true_positives, predicted)
        scored_recall = safe_divide(
            true_positives,
            true_candidate_pairs,
        )
        end_to_end_recall = safe_divide(
            true_positives,
            total_true_pairs,
        )

        rows.append(
            {
                "resolution_stage": "selected_configuration",
                "relationship_type": "threshold_classification",
                "evaluation_scope": scope,
                "accepted_relationships": predicted,
                "true_positive_relationships": true_positives,
                "false_positive_relationships": false_positives,
                "relationship_precision": round(precision, 6),
                "scored_candidate_recall": round(
                    scored_recall,
                    6,
                ),
                "end_to_end_recall": round(
                    end_to_end_recall,
                    6,
                ),
                "end_to_end_f1": round(
                    harmonic_mean(precision, end_to_end_recall),
                    6,
                ),
            }
        )

    return rows


def build_probabilistic_evaluation(
    current_accepted: pd.DataFrame,
    selected_matches: pd.DataFrame,
    annotated_scores: pd.DataFrame,
    true_pairs: pd.DataFrame,
    automatic_threshold: float,
    minimum_coverage: float,
) -> pd.DataFrame:
    # Compare current provisional and frozen selected relationships.
    rows: list[dict[str, object]] = []
    rows.extend(
        relationship_metrics(
            current_accepted,
            "current_probabilistic",
            "accepted_relationships",
        )
    )
    current_merges = current_accepted.loc[
        current_accepted["merge_performed"].map(parse_boolean)
    ]
    rows.extend(
        relationship_metrics(
            current_merges,
            "current_probabilistic",
            "performed_merges",
        )
    )

    selected_accepted = selected_matches.loc[
        selected_matches["selected_decision"].isin(
            {"accepted_merge", "accepted_transitive_support"}
        )
    ]
    rows.extend(
        relationship_metrics(
            selected_accepted,
            "selected_configuration",
            "accepted_relationships",
        )
    )
    selected_merges = selected_matches.loc[
        selected_matches["merge_performed"]
    ]
    rows.extend(
        relationship_metrics(
            selected_merges,
            "selected_configuration",
            "performed_merges",
        )
    )
    rows.extend(
        selected_candidate_evaluation(
            annotated_scores,
            true_pairs,
            automatic_threshold,
            minimum_coverage,
        )
    )

    output = pd.DataFrame(rows)
    for column in ("scored_candidate_recall", "end_to_end_recall"):
        if column not in output.columns:
            output[column] = ""
    if "end_to_end_f1" not in output.columns:
        output["end_to_end_f1"] = ""

    return output


###############################################################################
# 12. Error-analysis outputs
###############################################################################


def build_false_positive_matches(
    selected_matches: pd.DataFrame,
) -> pd.DataFrame:
    # Retain accepted selected relationships joining different identities.
    accepted = selected_matches["selected_decision"].isin(
        {"accepted_merge", "accepted_transitive_support"}
    )
    false_positive = ~selected_matches["is_true_cluster_match"]
    return selected_matches.loc[accepted & false_positive].copy()


def false_negative_reason(row: object) -> str:
    # Explain why a true DCL pair did not join under the frozen setting.
    if not bool(row.candidate_generated):
        return "True pair was not generated by blocking"
    if parse_boolean(row.static_conflict):
        return "True pair was withheld by a static conflict safeguard"
    if not bool(row.sufficient_support):
        return "True pair lacked the required corroborating evidence"
    if float(row.weighted_similarity_score) < float(
        row.selected_automatic_threshold
    ):
        return "True pair scored below the selected threshold"
    if float(row.evidence_coverage) < float(
        row.selected_minimum_coverage
    ):
        return "True pair had insufficient evidence coverage"
    if clean_value(row.selected_decision) == (
        "rejected_dynamic_conflict"
    ):
        return clean_value(row.selected_reason)

    return "True clusters remained separate after selected clustering"


def build_false_negative_matches(
    true_pairs: pd.DataFrame,
    annotated_scores: pd.DataFrame,
    selected_matches: pd.DataFrame,
    cluster_lookup: dict[str, str],
    automatic_threshold: float,
    minimum_coverage: float,
) -> pd.DataFrame:
    # Retain true DCL pairs that remain split after selected clustering.
    output = true_pairs.copy()
    output["evaluated_cluster_id_1"] = output["cluster_id_1"].map(
        cluster_lookup
    )
    output["evaluated_cluster_id_2"] = output["cluster_id_2"].map(
        cluster_lookup
    )
    output = output.loc[
        output["evaluated_cluster_id_1"].ne(
            output["evaluated_cluster_id_2"]
        )
    ].copy()

    score_columns = [
        "cluster_id_1",
        "cluster_id_2",
        "weighted_similarity_score",
        "evidence_coverage",
        "sufficient_support",
        "static_conflict",
    ]
    output = output.merge(
        annotated_scores[score_columns],
        on=["cluster_id_1", "cluster_id_2"],
        how="left",
        validate="many_to_one",
    )
    decision_columns = [
        "cluster_id_1",
        "cluster_id_2",
        "selected_decision",
        "selected_reason",
    ]
    output = output.merge(
        selected_matches[decision_columns],
        on=["cluster_id_1", "cluster_id_2"],
        how="left",
        validate="many_to_one",
    )
    output["selected_automatic_threshold"] = automatic_threshold
    output["selected_minimum_coverage"] = minimum_coverage
    output["weighted_similarity_score"] = output[
        "weighted_similarity_score"
    ].fillna(0.0)
    output["evidence_coverage"] = output[
        "evidence_coverage"
    ].fillna(0.0)
    output["sufficient_support"] = output[
        "sufficient_support"
    ].map(parse_boolean)
    output["static_conflict"] = output[
        "static_conflict"
    ].map(parse_boolean)
    output["selected_decision"] = output[
        "selected_decision"
    ].fillna("")
    output["selected_reason"] = output[
        "selected_reason"
    ].fillna("")
    output["false_negative_reason"] = [
        false_negative_reason(row)
        for row in output.itertuples(index=False)
    ]

    return output


def build_split_identities(
    selected_mapping: pd.DataFrame,
) -> pd.DataFrame:
    # Report true identities distributed across selected clusters.
    eligible = selected_mapping.loc[
        selected_mapping["evaluated_cluster_id"].ne("")
    ]
    rows: list[dict[str, object]] = []

    for ground_truth_id, group in eligible.groupby(
        "ground_truth_id",
        sort=True,
    ):
        evaluated_clusters = set(group["evaluated_cluster_id"])
        if len(evaluated_clusters) <= 1:
            continue
        rows.append(
            {
                "ground_truth_id": ground_truth_id,
                "evaluation_split": group[
                    "evaluation_split"
                ].iloc[0],
                "record_count": len(group),
                "source_systems": serialise_values(
                    group["source_system"]
                ),
                "deterministic_cluster_count": group[
                    "deterministic_cluster_id"
                ].nunique(),
                "evaluated_cluster_count": len(evaluated_clusters),
                "evaluated_cluster_ids": serialise_values(
                    evaluated_clusters
                ),
            }
        )

    columns = [
        "ground_truth_id",
        "evaluation_split",
        "record_count",
        "source_systems",
        "deterministic_cluster_count",
        "evaluated_cluster_count",
        "evaluated_cluster_ids",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_impure_clusters(
    selected_mapping: pd.DataFrame,
) -> pd.DataFrame:
    # Report selected clusters containing multiple true identities.
    eligible = selected_mapping.loc[
        selected_mapping["evaluated_cluster_id"].ne("")
    ]
    rows: list[dict[str, object]] = []

    for cluster_id, group in eligible.groupby(
        "evaluated_cluster_id",
        sort=True,
    ):
        truth_ids = set(group["ground_truth_id"])
        if len(truth_ids) <= 1:
            continue
        rows.append(
            {
                "evaluated_cluster_id": cluster_id,
                "record_count": len(group),
                "ground_truth_identity_count": len(truth_ids),
                "ground_truth_ids": serialise_values(truth_ids),
                "evaluation_splits": serialise_values(
                    group["evaluation_split"]
                ),
                "source_systems": serialise_values(
                    group["source_system"]
                ),
                "deterministic_cluster_count": group[
                    "deterministic_cluster_id"
                ].nunique(),
                "deterministic_cluster_ids": serialise_values(
                    group["deterministic_cluster_id"]
                ),
            }
        )

    columns = [
        "evaluated_cluster_id",
        "record_count",
        "ground_truth_identity_count",
        "ground_truth_ids",
        "evaluation_splits",
        "source_systems",
        "deterministic_cluster_count",
        "deterministic_cluster_ids",
    ]
    return pd.DataFrame(rows, columns=columns)


###############################################################################
# 13. Evaluation summary
###############################################################################


def selected_metric_row(
    dataframe: pd.DataFrame,
    filters: dict[str, str],
) -> pd.Series:
    # Return one validated metric row using exact filter values.
    mask = pd.Series(True, index=dataframe.index)
    for column, value in filters.items():
        mask &= dataframe[column].eq(value)

    selected = dataframe.loc[mask]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one metric row for filters: {filters}"
        )

    return selected.iloc[0]


def summary_value_lookup(
    summary: pd.DataFrame,
) -> dict[str, str]:
    # Convert a sectioned metric file into a lookup dictionary.
    return {
        clean_value(row.metric): clean_value(row.value)
        for row in summary.itertuples(index=False)
        if clean_value(row.metric) and clean_value(row.value)
    }


def build_evaluation_summary(
    records: pd.DataFrame,
    source_validation: pd.DataFrame,
    deterministic_mapping: pd.DataFrame,
    probabilistic_mapping: pd.DataFrame,
    selected_mapping: pd.DataFrame,
    identity_split: pd.DataFrame,
    selected_configuration: dict[str, object],
    probabilistic_summary: pd.DataFrame,
    blocking_evaluation: pd.DataFrame,
    deterministic_evaluation: pd.DataFrame,
    probabilistic_evaluation: pd.DataFrame,
    cluster_evaluation: pd.DataFrame,
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
    split_identities: pd.DataFrame,
    impure_clusters: pd.DataFrame,
) -> pd.DataFrame:
    # Create a concise, sectioned report for academic interpretation.
    rows: list[dict[str, object]] = []
    current_parameters = summary_value_lookup(probabilistic_summary)
    heldout_blocking = selected_metric_row(
        blocking_evaluation,
        {"evaluation_scope": EVALUATION_SPLIT},
    )
    heldout_deterministic = selected_metric_row(
        deterministic_evaluation,
        {"evaluation_scope": EVALUATION_SPLIT},
    )
    heldout_selected_relationships = selected_metric_row(
        probabilistic_evaluation,
        {
            "resolution_stage": "selected_configuration",
            "relationship_type": "threshold_classification",
            "evaluation_scope": EVALUATION_SPLIT,
        },
    )
    heldout_selected_clusters = selected_metric_row(
        cluster_evaluation,
        {
            "resolution_stage": "selected_configuration",
            "evaluation_scope": EVALUATION_SPLIT,
        },
    )
    current_clusters = selected_metric_row(
        cluster_evaluation,
        {
            "resolution_stage": "current_probabilistic",
            "evaluation_scope": "all",
        },
    )
    selected_clusters = selected_metric_row(
        cluster_evaluation,
        {
            "resolution_stage": "selected_configuration",
            "evaluation_scope": "all",
        },
    )

    add_summary_section(rows, "INPUT VALIDATION")
    rows.extend(
        [
            {"metric": "input_records", "value": len(records)},
            {
                "metric": "ground_truth_mapping_records",
                "value": int(source_validation["mapping_records"].sum()),
            },
            {
                "metric": "ground_truth_sources",
                "value": len(source_validation),
            },
            {
                "metric": "eligible_records",
                "value": int(
                    deterministic_mapping["resolution_status"]
                    .ne(ANONYMOUS_STATUS)
                    .sum()
                ),
            },
            {
                "metric": "anonymous_unresolvable_records",
                "value": int(
                    deterministic_mapping["resolution_status"]
                    .eq(ANONYMOUS_STATUS)
                    .sum()
                ),
            },
            {
                "metric": "ground_truth_excluded_from_matching",
                "value": True,
            },
        ]
    )

    split_counts = identity_split["evaluation_split"].value_counts()
    add_summary_section(rows, "IDENTITY-LEVEL SPLIT")
    rows.extend(
        [
            {
                "metric": "ground_truth_identities",
                "value": len(identity_split),
            },
            {
                "metric": "calibration_identities",
                "value": int(
                    split_counts.get(CALIBRATION_SPLIT, 0)
                ),
            },
            {
                "metric": "held_out_evaluation_identities",
                "value": int(
                    split_counts.get(EVALUATION_SPLIT, 0)
                ),
            },
            {
                "metric": "anonymous_only_identities",
                "value": int(
                    split_counts.get(ANONYMOUS_ONLY_SPLIT, 0)
                ),
            },
        ]
    )

    add_summary_section(rows, "CURRENT PROVISIONAL CONFIGURATION")
    rows.extend(
        [
            {
                "metric": "current_automatic_match_threshold",
                "value": current_parameters.get(
                    "automatic_match_threshold",
                    "",
                ),
            },
            {
                "metric": "current_minimum_evidence_coverage",
                "value": current_parameters.get(
                    "minimum_automatic_evidence_coverage",
                    "",
                ),
            },
            {
                "metric": "current_probabilistic_clusters",
                "value": probabilistic_mapping.loc[
                    probabilistic_mapping[
                        "probabilistic_cluster_id"
                    ].ne("")
                ]["probabilistic_cluster_id"].nunique(),
            },
            {
                "metric": "current_pairwise_precision_all_records",
                "value": current_clusters["pairwise_precision"],
            },
            {
                "metric": "current_impure_clusters",
                "value": current_clusters[
                    "impure_clusters_touching_scope"
                ],
            },
        ]
    )

    add_summary_section(rows, "SELECTED FROZEN CONFIGURATION")
    rows.extend(
        [
            {
                "metric": "selection_status",
                "value": selected_configuration[
                    "selection_status"
                ],
            },
            {
                "metric": "selected_automatic_match_threshold",
                "value": selected_configuration[
                    "automatic_match_threshold"
                ],
            },
            {
                "metric": "selected_minimum_evidence_coverage",
                "value": selected_configuration[
                    "minimum_evidence_coverage"
                ],
            },
            {
                "metric": "calibration_precision",
                "value": selected_configuration["precision"],
            },
            {
                "metric": "calibration_end_to_end_recall",
                "value": selected_configuration[
                    "end_to_end_recall"
                ],
            },
            {
                "metric": "selected_evaluated_clusters",
                "value": selected_mapping.loc[
                    selected_mapping["evaluated_cluster_id"].ne("")
                ]["evaluated_cluster_id"].nunique(),
            },
            {
                "metric": "selected_pairwise_precision_all_records",
                "value": selected_clusters["pairwise_precision"],
            },
            {
                "metric": "selected_impure_clusters",
                "value": selected_clusters[
                    "impure_clusters_touching_scope"
                ],
            },
        ]
    )

    add_summary_section(rows, "HELD-OUT EVALUATION")
    rows.extend(
        [
            {
                "metric": "held_out_blocking_recall",
                "value": heldout_blocking[
                    "cluster_pair_blocking_recall"
                ],
            },
            {
                "metric": "held_out_candidate_reduction_ratio",
                "value": heldout_blocking[
                    "candidate_reduction_ratio"
                ],
            },
            {
                "metric": "held_out_deterministic_edge_precision",
                "value": heldout_deterministic[
                    "deterministic_edge_precision"
                ],
            },
            {
                "metric": "held_out_selected_relationship_precision",
                "value": heldout_selected_relationships[
                    "relationship_precision"
                ],
            },
            {
                "metric": "held_out_selected_end_to_end_recall",
                "value": heldout_selected_relationships[
                    "end_to_end_recall"
                ],
            },
            {
                "metric": "held_out_selected_end_to_end_f1",
                "value": heldout_selected_relationships[
                    "end_to_end_f1"
                ],
            },
            {
                "metric": "held_out_pairwise_precision",
                "value": heldout_selected_clusters[
                    "pairwise_precision"
                ],
            },
            {
                "metric": "held_out_pairwise_recall",
                "value": heldout_selected_clusters[
                    "pairwise_recall"
                ],
            },
            {
                "metric": "held_out_pairwise_f1",
                "value": heldout_selected_clusters["pairwise_f1"],
            },
            {
                "metric": "held_out_bcubed_precision",
                "value": heldout_selected_clusters[
                    "bcubed_precision"
                ],
            },
            {
                "metric": "held_out_bcubed_recall",
                "value": heldout_selected_clusters["bcubed_recall"],
            },
            {
                "metric": "held_out_bcubed_f1",
                "value": heldout_selected_clusters["bcubed_f1"],
            },
        ]
    )

    add_summary_section(rows, "ERROR ANALYSIS")
    rows.extend(
        [
            {
                "metric": "selected_false_positive_relationships",
                "value": len(false_positives),
            },
            {
                "metric": "selected_false_negative_cluster_pairs",
                "value": len(false_negatives),
            },
            {
                "metric": "selected_split_ground_truth_identities",
                "value": len(split_identities),
            },
            {
                "metric": "selected_impure_resolved_clusters",
                "value": len(impure_clusters),
            },
        ]
    )

    add_summary_section(rows, "ABSTENTION VALIDATION")
    anonymous = selected_mapping.loc[
        selected_mapping["evaluation_resolution_status"].eq(
            ANONYMOUS_STATUS
        )
    ]
    rows.extend(
        [
            {
                "metric": "anonymous_records_evaluated_separately",
                "value": len(anonymous),
            },
            {
                "metric": "anonymous_records_remain_unclustered",
                "value": anonymous["evaluated_cluster_id"].eq("").all(),
            },
        ]
    )

    return pd.DataFrame(rows, columns=["metric", "value"])


###############################################################################
# 14. Output validation and writing
###############################################################################


def validate_selected_cluster_conflicts(
    records: pd.DataFrame,
    selected_mapping: pd.DataFrame,
) -> None:
    # Confirm selected clusters preserve trusted portal and DOB safeguards.
    eligible = selected_mapping.loc[
        selected_mapping["evaluated_cluster_id"].ne(""),
        ["staging_record_id", "evaluated_cluster_id"],
    ].merge(
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
        eligible[field] = eligible[field].map(clean_value)
        populated = eligible.loc[eligible[field].ne("")]
        conflicts = populated.groupby("evaluated_cluster_id")[
            field
        ].nunique()
        if conflicts.gt(1).any():
            raise ValueError(
                f"A selected cluster contains conflicting {field} values."
            )


def validate_evaluation_outputs(
    records: pd.DataFrame,
    staging_ground_truth: pd.DataFrame,
    identity_split: pd.DataFrame,
    calibration_results: pd.DataFrame,
    selected_configuration: dict[str, object],
    selected_matches: pd.DataFrame,
    selected_mapping: pd.DataFrame,
    cluster_evaluation: pd.DataFrame,
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
) -> None:
    # Reconcile protected mappings, selected clusters and error outputs.
    if len(staging_ground_truth) != len(records):
        raise ValueError(
            "The staging ground-truth output does not reconcile."
        )
    if not staging_ground_truth["staging_record_id"].is_unique:
        raise ValueError(
            "The staging ground-truth output contains duplicate IDs."
        )
    if staging_ground_truth["ground_truth_id"].map(
        clean_value
    ).eq("").any():
        raise ValueError(
            "The staging ground-truth output contains blank identities."
        )
    if not identity_split["ground_truth_id"].is_unique:
        raise ValueError("The identity split contains duplicate identities.")
    if len(selected_mapping) != len(records):
        raise ValueError(
            "The selected evaluation mapping does not reconcile."
        )
    if not selected_mapping["staging_record_id"].is_unique:
        raise ValueError(
            "The selected evaluation mapping contains duplicate records."
        )

    anonymous = selected_mapping[
        "evaluation_resolution_status"
    ].eq(ANONYMOUS_STATUS)
    if selected_mapping.loc[
        anonymous,
        "evaluated_cluster_id",
    ].ne("").any():
        raise ValueError("Anonymous records received evaluated cluster IDs.")
    if selected_mapping.loc[
        ~anonymous,
        "evaluated_cluster_id",
    ].eq("").any():
        raise ValueError("An eligible record has no evaluated cluster ID.")

    threshold = selected_configuration["automatic_match_threshold"]
    coverage = selected_configuration["minimum_evidence_coverage"]
    selected_grid_row = calibration_results.loc[
        calibration_results["automatic_match_threshold"].eq(threshold)
        & calibration_results["minimum_evidence_coverage"].eq(
            coverage
        )
    ]
    if len(selected_grid_row) != 1:
        raise ValueError(
            "The selected configuration is not present in calibration."
        )

    accepted = selected_matches["selected_decision"].isin(
        {"accepted_merge", "accepted_transitive_support"}
    )
    accepted_rows = selected_matches.loc[accepted]
    if not accepted_rows.empty:
        same_final_cluster = accepted_rows[
            "evaluated_cluster_id_1"
        ].eq(accepted_rows["evaluated_cluster_id_2"])
        if not same_final_cluster.all():
            raise ValueError(
                "An accepted selected relationship spans final clusters."
            )

    if not false_positives.empty:
        if false_positives["is_true_cluster_match"].any():
            raise ValueError(
                "The false-positive output contains a true relationship."
            )
    if not false_negatives.empty:
        same_cluster = false_negatives[
            "evaluated_cluster_id_1"
        ].eq(false_negatives["evaluated_cluster_id_2"])
        if same_cluster.any():
            raise ValueError(
                "The false-negative output contains a resolved pair."
            )

    metric_columns = [
        "pairwise_precision",
        "pairwise_recall",
        "pairwise_f1",
        "bcubed_precision",
        "bcubed_recall",
        "bcubed_f1",
        "weighted_cluster_purity",
    ]
    for column in metric_columns:
        values = pd.to_numeric(
            cluster_evaluation[column],
            errors="raise",
        )
        if not values.between(0, 1).all():
            raise ValueError(f"Cluster metric is outside 0-1: {column}")

    validate_selected_cluster_conflicts(records, selected_mapping)


def write_outputs(
    output_directory: Path,
    outputs: dict[str, pd.DataFrame],
) -> None:
    # Write every evaluation and audit output with stable filenames.
    output_directory.mkdir(parents=True, exist_ok=True)
    for filename, dataframe in outputs.items():
        dataframe.to_csv(
            output_directory / filename,
            index=False,
        )


###############################################################################
# 15. Main evaluation workflow
###############################################################################


def main() -> None:
    # Run protected calibration, held-out evaluation and error analysis.
    arguments = parse_arguments()
    validate_parameters(arguments)

    inputs = load_matching_inputs(arguments)
    validate_matching_inputs(inputs)
    ground_truth, source_validation = load_ground_truth_mappings(
        arguments.ground_truth_dir
    )
    staging_ground_truth = build_staging_ground_truth_mapping(
        inputs["records"],
        ground_truth,
    )

    deterministic_mapping = attach_ground_truth(
        inputs["deterministic_mapping"],
        staging_ground_truth,
    )
    identity_split = build_identity_split(
        deterministic_mapping,
        arguments.calibration_proportion,
        arguments.seed,
    )
    deterministic_mapping = attach_identity_split(
        deterministic_mapping,
        identity_split,
    )
    probabilistic_mapping = attach_ground_truth(
        inputs["probabilistic_mapping"],
        staging_ground_truth,
    )
    probabilistic_mapping = attach_identity_split(
        probabilistic_mapping,
        identity_split,
    )
    staging_ground_truth = attach_identity_split(
        staging_ground_truth,
        identity_split,
    )

    cluster_truth_sets = build_cluster_truth_sets(
        deterministic_mapping
    )
    split_lookup = identity_split.set_index("ground_truth_id")[
        "evaluation_split"
    ].to_dict()
    cluster_split_sets = build_cluster_split_sets(
        cluster_truth_sets,
        split_lookup,
    )
    staging_truth_lookup = staging_ground_truth.set_index(
        "staging_record_id"
    )["ground_truth_id"].to_dict()

    scores = prepare_score_columns(inputs["probabilistic_scores"])
    annotated_scores = annotate_candidate_truth(
        scores,
        cluster_truth_sets,
        cluster_split_sets,
        staging_truth_lookup,
    )
    current_accepted = annotate_relationship_truth(
        inputs["probabilistic_accepted"],
        annotated_scores,
    )

    true_pairs = build_true_cross_cluster_pairs(
        deterministic_mapping
    )
    true_pairs = add_candidate_generation_status(
        true_pairs,
        inputs["probabilistic_candidates"],
    )
    blocking_evaluation = build_blocking_evaluation(
        true_pairs,
        annotated_scores,
        deterministic_mapping,
    )
    calibration_results = build_calibration_results(
        annotated_scores,
        true_pairs,
    )
    selected_configuration_output, selected_configuration = (
        select_configuration(
            calibration_results,
            arguments.minimum_precision,
            arguments.minimum_calibration_matches,
        )
    )

    automatic_threshold = float(
        selected_configuration["automatic_match_threshold"]
    )
    minimum_coverage = float(
        selected_configuration["minimum_evidence_coverage"]
    )
    cluster_profiles = build_cluster_profiles(
        inputs["records"],
        deterministic_mapping,
    )
    prohibited_pairs = build_prohibited_pairs(
        inputs["deterministic_rejected"]
    )
    cluster_lookup, selected_matches = (
        simulate_selected_configuration(
            annotated_scores,
            cluster_profiles,
            prohibited_pairs,
            automatic_threshold,
            minimum_coverage,
        )
    )
    selected_mapping = build_selected_record_mapping(
        deterministic_mapping,
        cluster_lookup,
    )

    cluster_evaluation = build_cluster_evaluation(
        deterministic_mapping,
        probabilistic_mapping,
        selected_mapping,
    )
    deterministic_pairwise = annotate_deterministic_pairwise_truth(
        inputs["deterministic_pairwise"],
        staging_ground_truth,
        split_lookup,
    )
    deterministic_evaluation = build_deterministic_evaluation(
        deterministic_pairwise,
        cluster_evaluation,
    )
    probabilistic_evaluation = build_probabilistic_evaluation(
        current_accepted,
        selected_matches,
        annotated_scores,
        true_pairs,
        automatic_threshold,
        minimum_coverage,
    )

    false_positives = build_false_positive_matches(
        selected_matches
    )
    false_negatives = build_false_negative_matches(
        true_pairs,
        annotated_scores,
        selected_matches,
        cluster_lookup,
        automatic_threshold,
        minimum_coverage,
    )
    split_identities = build_split_identities(selected_mapping)
    impure_clusters = build_impure_clusters(selected_mapping)
    summary = build_evaluation_summary(
        inputs["records"],
        source_validation,
        deterministic_mapping,
        probabilistic_mapping,
        selected_mapping,
        identity_split,
        selected_configuration,
        inputs["probabilistic_summary"],
        blocking_evaluation,
        deterministic_evaluation,
        probabilistic_evaluation,
        cluster_evaluation,
        false_positives,
        false_negatives,
        split_identities,
        impure_clusters,
    )

    validate_evaluation_outputs(
        inputs["records"],
        staging_ground_truth,
        identity_split,
        calibration_results,
        selected_configuration,
        selected_matches,
        selected_mapping,
        cluster_evaluation,
        false_positives,
        false_negatives,
    )

    outputs = {
        STAGING_GROUND_TRUTH_FILENAME: staging_ground_truth,
        IDENTITY_SPLIT_FILENAME: identity_split,
        CALIBRATION_RESULTS_FILENAME: calibration_results,
        SELECTED_CONFIGURATION_FILENAME: (
            selected_configuration_output
        ),
        DETERMINISTIC_EVALUATION_FILENAME: (
            deterministic_evaluation
        ),
        BLOCKING_EVALUATION_FILENAME: blocking_evaluation,
        PROBABILISTIC_EVALUATION_FILENAME: (
            probabilistic_evaluation
        ),
        CLUSTER_EVALUATION_FILENAME: cluster_evaluation,
        SELECTED_MATCHES_FILENAME: selected_matches,
        SELECTED_MAPPING_FILENAME: selected_mapping,
        SUMMARY_FILENAME: summary,
        FALSE_POSITIVES_FILENAME: false_positives,
        FALSE_NEGATIVES_FILENAME: false_negatives,
        SPLIT_IDENTITIES_FILENAME: split_identities,
        IMPURE_CLUSTERS_FILENAME: impure_clusters,
    }
    write_outputs(arguments.output_dir, outputs)

    heldout_relationships = selected_metric_row(
        probabilistic_evaluation,
        {
            "resolution_stage": "selected_configuration",
            "relationship_type": "threshold_classification",
            "evaluation_scope": EVALUATION_SPLIT,
        },
    )
    heldout_clusters = selected_metric_row(
        cluster_evaluation,
        {
            "resolution_stage": "selected_configuration",
            "evaluation_scope": EVALUATION_SPLIT,
        },
    )

    print("Identity-resolution evaluation completed successfully.")
    print(f"Input: {arguments.input.resolve()}")
    print(
        "Ground-truth directory: "
        f"{arguments.ground_truth_dir.resolve()}"
    )
    print(f"Output directory: {arguments.output_dir.resolve()}")
    print(f"Input records: {len(inputs['records']):,}")
    print(f"Ground-truth identities: {len(identity_split):,}")
    print(
        "Selected automatic threshold: "
        f"{automatic_threshold:.2f}"
    )
    print(
        "Selected minimum evidence coverage: "
        f"{minimum_coverage:.2f}"
    )
    print(
        "Held-out relationship precision: "
        f"{float(heldout_relationships['relationship_precision']):.4f}"
    )
    print(
        "Held-out relationship end-to-end recall: "
        f"{float(heldout_relationships['end_to_end_recall']):.4f}"
    )
    print(
        "Held-out B-cubed F1: "
        f"{float(heldout_clusters['bcubed_f1']):.4f}"
    )
    print(f"False-positive selected links: {len(false_positives):,}")
    print(f"Split ground-truth identities: {len(split_identities):,}")
    print(f"Impure selected clusters: {len(impure_clusters):,}")


if __name__ == "__main__":
    main()
