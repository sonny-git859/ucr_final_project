# This script runs the complete UCR project workflow from synthetic-data
# generation through governed Gold-layer construction.
#
# Three modes are available:
#
# 1. The default mode rebuilds the synthetic environment, runs the complete
#    Bronze and Silver data pipeline, and executes the operational,
#    ground-truth-free identity-resolution workflow.
# 2. --include-evaluation runs the same generation and pipeline stages, then
#    asks the identity-resolution orchestrator to reproduce the original
#    calibration and ground-truth evaluation before constructing the final
#    operational UCR and Gold outputs.
# 3. --dry-run checks that the orchestration and component scripts exist and
#    displays their execution order. It does not generate data, process
#    records, inspect existing CSV contents, or write outputs.
#
# The two options can be combined. --include-evaluation --dry-run displays the
# complete research workflow without executing any processing stage.
#
# The Streamlit application is intentionally excluded. It is an interactive
# server rather than a terminating batch-processing stage and should be
# launched separately after the Gold outputs have been constructed.

###############################################################################
# Imports
###############################################################################

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


###############################################################################
# 1. Workflow configuration
###############################################################################


PROJECT_ROOT = Path(__file__).resolve().parent

GENERATION_SCRIPT = (
    Path("SRC")
    / "synth_data_gen"
    / "generate_all_systems.py"
)
PIPELINE_SCRIPT = Path("SRC") / "pipeline" / "run_pipeline.py"
IDENTITY_SCRIPT = (
    Path("SRC")
    / "Resolution"
    / "run_identity_resolution.py"
)


@dataclass(frozen=True)
class ProjectStage:
    name: str
    script_path: Path
    supporting_scripts: tuple[Path, ...]
    expected_outputs: tuple[Path, ...]


GENERATION_STAGE = ProjectStage(
    name="Synthetic-data generation and validation",
    script_path=GENERATION_SCRIPT,
    supporting_scripts=(
        Path("SRC")
        / "synth_data_gen"
        / "canonical_gen"
        / "canonical_customer_generation.py",
        Path("SRC")
        / "synth_data_gen"
        / "events_gen"
        / "events_generation.py",
        Path("SRC")
        / "synth_data_gen"
        / "crm_gen"
        / "crm_generation.py",
        Path("SRC")
        / "synth_data_gen"
        / "ecommerce_gen"
        / "ecommerce_gen.py",
        Path("SRC")
        / "synth_data_gen"
        / "online_sessions_gen"
        / "online_session_generation.py",
        Path("SRC")
        / "synth_data_gen"
        / "support_gen"
        / "support_generation.py",
        Path("SRC")
        / "synth_data_gen"
        / "marketing_gen"
        / "marketing_contacts_generation.py",
    ),
    expected_outputs=(
        Path("data") / "canonical" / "canonical_customers.csv",
        Path("data") / "events" / "events.csv",
        Path("data") / "raw" / "crm_customer_records.csv",
        Path("data") / "raw" / "ecommerce_transactions.csv",
        Path("data") / "raw" / "online_sessions.csv",
        Path("data") / "raw" / "support_ticket_logs.csv",
        Path("data") / "raw" / "marketing_contact_lists.csv",
        Path("data")
        / "reference"
        / "system_validation_summary.csv",
    ),
)

PIPELINE_STAGE = ProjectStage(
    name="Bronze and Silver data pipeline",
    script_path=PIPELINE_SCRIPT,
    supporting_scripts=(
        Path("SRC") / "pipeline" / "ingest_raw_sources.py",
        Path("SRC") / "pipeline" / "standardise_bronze_sources.py",
        Path("SRC") / "pipeline" / "consolidate_silver_sources.py",
    ),
    expected_outputs=(
        Path("data")
        / "consolidated_silver"
        / "consolidated_customer_records.csv",
        Path("data")
        / "consolidated_silver"
        / "consolidation_summary.csv",
    ),
)

IDENTITY_STAGE = ProjectStage(
    name="Identity resolution and Gold construction",
    script_path=IDENTITY_SCRIPT,
    supporting_scripts=(
        Path("SRC") / "Resolution" / "deterministic_matching.py",
        Path("SRC") / "Resolution" / "probabilistic_matching.py",
        Path("SRC")
        / "Resolution"
        / "evaluate_identity_resolution.py",
        Path("SRC")
        / "Resolution"
        / "probabilistic_matching_calibrated.py",
        Path("SRC") / "Resolution" / "finalise_ucr_mapping.py",
        Path("SRC") / "Resolution" / "build_golden_records.py",
    ),
    expected_outputs=(
        Path("identity_resolution")
        / "final"
        / "record_to_ucr_mapping.csv",
        Path("identity_resolution")
        / "final"
        / "ucr_cluster_summary.csv",
        Path("data") / "gold" / "ucr_master_records.csv",
        Path("data") / "gold" / "ucr_record_links.csv",
        Path("data") / "gold" / "ucr_attribute_provenance.csv",
        Path("data") / "gold" / "ucr_interaction_summary.csv",
        Path("data") / "gold" / "golden_record_summary.csv",
        Path("data") / "gold" / "golden_record_summary.csv",
        Path("data")
        / "gold"
        / "evaluation"
        / "ucr_attribute_candidates.csv",
        Path("data")
        / "gold"
        / "evaluation"
        / "ucr_interaction_details.csv",
    ),
)

PROJECT_STAGES = (
    GENERATION_STAGE,
    PIPELINE_STAGE,
    IDENTITY_STAGE,
)

EVALUATION_OUTPUTS = (
    Path("identity_resolution")
    / "evaluation"
    / "identity_resolution_evaluation_summary.csv",
    Path("identity_resolution")
    / "evaluation"
    / "selected_matching_configuration.csv",
)


###############################################################################
# 2. Command-line configuration
###############################################################################


def parse_arguments() -> argparse.Namespace:
    # Parse optional end-to-end workflow controls.

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete UCR project from synthetic-data generation "
            "through final Gold-layer construction."
        ),
        epilog="\n".join(
            (
                "Examples:",
                "  Run the normal end-to-end operational workflow:",
                "    python run_project.py",
                "",
                "  Reproduce calibration and evaluation as well:",
                "    python run_project.py --include-evaluation",
                "",
                "  Check the end-to-end structure without processing:",
                "    python run_project.py --dry-run",
                "",
                "  Check the full research structure without processing:",
                "    python run_project.py --include-evaluation --dry-run",
                "",
                "--include-evaluation affects only the identity-resolution "
                "stage. Synthetic-data generation and the Bronze/Silver "
                "pipeline are identical in both modes.",
                "",
                "The evaluator may read protected ground truth. Final UCR "
                "assignment and Gold construction remain ground-truth free.",
            )
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--include-evaluation",
        action="store_true",
        help=(
            "Pass the research-reproduction option to the identity-resolution "
            "orchestrator. This adds the experimental probabilistic matcher "
            "and ground-truth evaluation before the frozen operational "
            "identity-resolution stages."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Check the orchestration and component scripts, then display the "
            "selected end-to-end plan without running processing stages or "
            "writing outputs."
        ),
    )
    return parser.parse_args()


###############################################################################
# 3. Workflow validation
###############################################################################


def validate_project_structure() -> None:
    # Confirm that every orchestration and component script exists.

    required_scripts = []
    for stage in PROJECT_STAGES:
        required_scripts.append(stage.script_path)
        required_scripts.extend(stage.supporting_scripts)

    missing_scripts = [
        str(script_path)
        for script_path in required_scripts
        if not (PROJECT_ROOT / script_path).is_file()
    ]
    if missing_scripts:
        raise FileNotFoundError(
            "Required project scripts not found: "
            + ", ".join(missing_scripts)
        )


def validate_stage_outputs(
    stage: ProjectStage,
    include_evaluation: bool,
) -> None:
    # Confirm that a completed project stage created its required outputs.

    expected_outputs = list(stage.expected_outputs)
    if stage == IDENTITY_STAGE and include_evaluation:
        expected_outputs.extend(EVALUATION_OUTPUTS)

    missing_outputs = [
        str(output)
        for output in expected_outputs
        if not (PROJECT_ROOT / output).is_file()
    ]
    if missing_outputs:
        raise FileNotFoundError(
            f"{stage.name} did not create required outputs: "
            + ", ".join(missing_outputs)
        )


###############################################################################
# 4. Console reporting
###############################################################################


def print_stage_header(
    stage_number: int,
    total_stages: int,
    stage: ProjectStage,
) -> None:
    # Print a clear divider before each project stage.

    print("\n" + "=" * 79)
    print(f"Project stage {stage_number} of {total_stages}: {stage.name}")
    print("=" * 79 + "\n")


def print_workflow_plan(include_evaluation: bool) -> None:
    # Display the selected end-to-end workflow in execution order.

    identity_mode = (
        "full research reproduction"
        if include_evaluation
        else "operational, ground-truth-free workflow"
    )

    print("Selected end-to-end UCR workflow:")
    for stage_number, stage in enumerate(PROJECT_STAGES, start=1):
        print(
            f"  {stage_number}. {stage.script_path}"
        )
        if stage == IDENTITY_STAGE:
            print(f"     Identity mode: {identity_mode}")


def print_mode_explanation(
    include_evaluation: bool,
    dry_run: bool,
) -> None:
    # Explain how the selected options affect the complete project run.

    if include_evaluation:
        print("Mode: complete research reproduction")
        print(
            "Generation and data preparation run normally. The identity "
            "stage also reproduces calibration and evaluation before using "
            "the frozen configuration for operational outputs."
        )
    else:
        print("Mode: complete operational UCR construction")
        print(
            "Generation, data preparation, frozen identity resolution and "
            "Gold construction are included. Evaluation-only stages are "
            "excluded."
        )

    if dry_run:
        print(
            "Dry run: orchestration script paths and execution order will be "
            "checked, but no project stage will run and no output will be "
            "written."
        )


def print_workflow_summary(
    stage_timings: dict[str, float],
    elapsed_seconds: float,
) -> None:
    # Print successful project completion and elapsed times.

    print("\n" + "=" * 79)
    print("Complete end-to-end UCR project finished successfully")
    print("=" * 79)
    print("\nCompleted project stages:")

    for stage_name, stage_seconds in stage_timings.items():
        print(f"  {stage_name}: {stage_seconds:.2f} seconds")

    print(f"\nProject stages completed: {len(stage_timings)}")
    print(f"Total elapsed time: {elapsed_seconds:.2f} seconds")
    print(
        "Gold outputs are ready. Launch the Streamlit application "
        "separately when it is available."
    )


###############################################################################
# 5. Workflow orchestration
###############################################################################


def build_stage_command(
    stage: ProjectStage,
    include_evaluation: bool,
) -> list[str]:
    # Build one stage command using the active Python interpreter.

    command = [
        sys.executable,
        str(PROJECT_ROOT / stage.script_path),
    ]
    if stage == IDENTITY_STAGE and include_evaluation:
        command.append("--include-evaluation")

    return command


def run_stage(
    stage: ProjectStage,
    include_evaluation: bool,
) -> float:
    # Run one project stage and stop immediately if it fails.

    command = build_stage_command(stage, include_evaluation)
    start_time = perf_counter()

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{stage.name} failed with exit code "
            f"{completed.returncode}."
        )

    validate_stage_outputs(stage, include_evaluation)
    return perf_counter() - start_time


def run_complete_project(
    include_evaluation: bool = False,
    dry_run: bool = False,
) -> dict[str, float]:
    # Run the selected complete project workflow in dependency order.

    validate_project_structure()
    print_mode_explanation(include_evaluation, dry_run)
    print()
    print_workflow_plan(include_evaluation)

    if dry_run:
        print("\nDry run completed successfully.")
        print("All required project workflow scripts were found.")
        print("No data was processed and no outputs were created or changed.")
        return {}

    print("\nStarting complete end-to-end UCR project workflow...")

    workflow_start = perf_counter()
    stage_timings = {}
    total_stages = len(PROJECT_STAGES)

    for stage_number, stage in enumerate(PROJECT_STAGES, start=1):
        print_stage_header(
            stage_number,
            total_stages,
            stage,
        )
        stage_timings[stage.name] = run_stage(
            stage,
            include_evaluation,
        )

    elapsed_seconds = perf_counter() - workflow_start
    print_workflow_summary(stage_timings, elapsed_seconds)

    return stage_timings


###############################################################################
# 6. Run complete project
###############################################################################


def main() -> None:
    # Run the complete project and report any failed stage clearly.

    arguments = parse_arguments()

    try:
        run_complete_project(
            include_evaluation=arguments.include_evaluation,
            dry_run=arguments.dry_run,
        )
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        raise SystemExit(
            f"Complete project workflow failed: {error}"
        ) from error


if __name__ == "__main__":
    main()
