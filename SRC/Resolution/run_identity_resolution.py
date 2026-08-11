# This script runs the identity-resolution stages in their required order.
#
# Three modes are available:
#
# 1. The default mode runs the operational, ground-truth-free workflow. This
#    is the normal mode for rebuilding the final UCR and Gold-layer outputs.
# 2. --include-evaluation reproduces the earlier calibration and evaluation
#    experiment before continuing through the operational workflow. It adds
#    the original probabilistic matcher and ground-truth evaluation. This mode
#    is intended for research reproducibility, not normal UCR construction.
# 3. --dry-run checks that the consolidated input and selected scripts exist,
#    and prints their execution order. It does not process data, inspect CSV
#    contents, validate existing outputs, or write any files.
#
# The two options can be combined. --include-evaluation --dry-run displays and
# validates the structure of the full research workflow without executing it.

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


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parents[1]

SILVER_INPUT = (
    Path("data")
    / "consolidated_silver"
    / "consolidated_customer_records.csv"
)


@dataclass(frozen=True)
class WorkflowStage:
    name: str
    script_name: str
    expected_outputs: tuple[Path, ...]
    evaluation_only: bool = False


DETERMINISTIC_STAGE = WorkflowStage(
    name="Deterministic identity resolution",
    script_name="deterministic_matching.py",
    expected_outputs=(
        Path("identity_resolution")
        / "deterministic"
        / "deterministic_record_mapping.csv",
        Path("identity_resolution")
        / "deterministic"
        / "deterministic_matching_summary.csv",
    ),
)

EXPERIMENTAL_PROBABILISTIC_STAGE = WorkflowStage(
    name="Experimental probabilistic matching",
    script_name="probabilistic_matching.py",
    expected_outputs=(
        Path("identity_resolution")
        / "probabilistic"
        / "probabilistic_record_mapping.csv",
        Path("identity_resolution")
        / "probabilistic"
        / "probabilistic_matching_summary.csv",
    ),
    evaluation_only=True,
)

EVALUATION_STAGE = WorkflowStage(
    name="Identity-resolution evaluation",
    script_name="evaluate_identity_resolution.py",
    expected_outputs=(
        Path("identity_resolution")
        / "evaluation"
        / "identity_resolution_evaluation_summary.csv",
        Path("identity_resolution")
        / "evaluation"
        / "selected_matching_configuration.csv",
    ),
    evaluation_only=True,
)

CALIBRATED_PROBABILISTIC_STAGE = WorkflowStage(
    name="Calibrated probabilistic matching",
    script_name="probabilistic_matching_calibrated.py",
    expected_outputs=(
        Path("identity_resolution")
        / "probabilistic_calibrated"
        / "probabilistic_record_mapping.csv",
        Path("identity_resolution")
        / "probabilistic_calibrated"
        / "probabilistic_matching_summary.csv",
    ),
)

FINALISATION_STAGE = WorkflowStage(
    name="Final UCR assignment",
    script_name="finalise_ucr_mapping.py",
    expected_outputs=(
        Path("identity_resolution")
        / "final"
        / "record_to_ucr_mapping.csv",
        Path("identity_resolution")
        / "final"
        / "ucr_cluster_summary.csv",
        Path("identity_resolution")
        / "final"
        / "finalisation_summary.csv",
    ),
)

GOLDEN_RECORD_STAGE = WorkflowStage(
    name="Golden UCR record construction",
    script_name="build_golden_records.py",
    expected_outputs=(
        Path("data") / "gold" / "ucr_master_records.csv",
        Path("data") / "gold" / "ucr_record_links.csv",
        Path("data") / "gold" / "ucr_attribute_provenance.csv",
        Path("data") / "gold" / "ucr_interaction_summary.csv",
        Path("data") / "gold" / "golden_record_summary.csv",
    ),
)

OPERATIONAL_STAGES = (
    DETERMINISTIC_STAGE,
    CALIBRATED_PROBABILISTIC_STAGE,
    FINALISATION_STAGE,
    GOLDEN_RECORD_STAGE,
)

EVALUATION_STAGES = (
    DETERMINISTIC_STAGE,
    EXPERIMENTAL_PROBABILISTIC_STAGE,
    EVALUATION_STAGE,
    CALIBRATED_PROBABILISTIC_STAGE,
    FINALISATION_STAGE,
    GOLDEN_RECORD_STAGE,
)


###############################################################################
# 2. Command-line configuration
###############################################################################


def parse_arguments() -> argparse.Namespace:
    # Parse optional workflow controls.

    parser = argparse.ArgumentParser(
        description=(
            "Run the UCR identity-resolution workflow. The default mode "
            "uses only operational, ground-truth-free stages."
        ),
        epilog="\n".join(
            (
                "Examples:",
                "  Run the normal operational workflow:",
                "    python SRC\\Resolution\\run_identity_resolution.py",
                "",
                "  Reproduce calibration and evaluation as well:",
                "    python SRC\\Resolution\\run_identity_resolution.py "
                "--include-evaluation",
                "",
                "  Check the operational structure without processing:",
                "    python SRC\\Resolution\\run_identity_resolution.py "
                "--dry-run",
                "",
                "  Check the full research workflow without processing:",
                "    python SRC\\Resolution\\run_identity_resolution.py "
                "--include-evaluation --dry-run",
                "",
                "The evaluation mode uses protected ground truth only in "
                "evaluate_identity_resolution.py. Operational Gold outputs "
                "remain ground-truth free.",
            )
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--include-evaluation",
        action="store_true",
        help=(
            "Reproduce the research calibration and evaluation process. "
            "Adds probabilistic_matching.py and "
            "evaluate_identity_resolution.py before the frozen calibrated "
            "workflow. Not required for normal operational runs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Check the required input and scripts, then display the selected "
            "execution order without running stages or writing outputs."
        ),
    )
    return parser.parse_args()


###############################################################################
# 3. Workflow validation
###############################################################################


def select_stages(
    include_evaluation: bool,
) -> tuple[WorkflowStage, ...]:
    # Select the clean operational or full evaluation workflow.

    if include_evaluation:
        return EVALUATION_STAGES

    return OPERATIONAL_STAGES


def validate_project_structure(
    stages: tuple[WorkflowStage, ...],
) -> None:
    # Confirm the consolidated input and all selected scripts exist.

    silver_path = PROJECT_ROOT / SILVER_INPUT
    if not silver_path.is_file():
        raise FileNotFoundError(
            "Consolidated Silver input not found: "
            f"{silver_path}"
        )

    missing_scripts = [
        stage.script_name
        for stage in stages
        if not (SCRIPT_DIRECTORY / stage.script_name).is_file()
    ]
    if missing_scripts:
        raise FileNotFoundError(
            "Workflow scripts not found in "
            f"{SCRIPT_DIRECTORY}: {', '.join(missing_scripts)}"
        )


def validate_stage_outputs(stage: WorkflowStage) -> None:
    # Confirm that a completed stage created its required outputs.

    missing_outputs = [
        str(output)
        for output in stage.expected_outputs
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
    stage: WorkflowStage,
) -> None:
    # Print a clear divider before each workflow stage.

    stage_type = "evaluation-only" if stage.evaluation_only else "operational"

    print("\n" + "=" * 79)
    print(
        f"Stage {stage_number} of {total_stages}: "
        f"{stage.name} ({stage_type})"
    )
    print("=" * 79 + "\n")


def print_workflow_plan(stages: tuple[WorkflowStage, ...]) -> None:
    # Display the selected scripts in execution order.

    print("Selected identity-resolution workflow:")
    for stage_number, stage in enumerate(stages, start=1):
        label = "evaluation-only" if stage.evaluation_only else "operational"
        print(
            f"  {stage_number}. {stage.script_name} "
            f"[{label}]"
        )


def print_mode_explanation(
    include_evaluation: bool,
    dry_run: bool,
) -> None:
    # Explain the selected mode before showing or executing the workflow.

    if include_evaluation:
        print("Mode: full research reproduction")
        print(
            "The original probabilistic matcher and protected ground-truth "
            "evaluation are included before the operational stages."
        )
    else:
        print("Mode: operational UCR construction")
        print(
            "Only the frozen, ground-truth-free operational stages are "
            "included."
        )

    if dry_run:
        print(
            "Dry run: required input and script paths will be checked, but "
            "no stage will run and no output will be written."
        )


def print_workflow_summary(
    stage_timings: dict[str, float],
    elapsed_seconds: float,
) -> None:
    # Print successful stage completion and elapsed times.

    print("\n" + "=" * 79)
    print("Complete UCR identity-resolution workflow finished successfully")
    print("=" * 79)
    print("\nCompleted stages:")

    for stage_name, stage_seconds in stage_timings.items():
        print(f"  {stage_name}: {stage_seconds:.2f} seconds")

    print(f"\nStages completed: {len(stage_timings)}")
    print(f"Total elapsed time: {elapsed_seconds:.2f} seconds")


###############################################################################
# 5. Workflow orchestration
###############################################################################


def run_stage(stage: WorkflowStage) -> float:
    # Run one script using the active Python interpreter and fail immediately.

    script_path = SCRIPT_DIRECTORY / stage.script_name
    command = [sys.executable, str(script_path)]
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

    validate_stage_outputs(stage)
    return perf_counter() - start_time


def run_identity_resolution(
    include_evaluation: bool = False,
    dry_run: bool = False,
) -> dict[str, float]:
    # Run the selected workflow in its required sequence.

    stages = select_stages(include_evaluation)
    validate_project_structure(stages)
    print_mode_explanation(include_evaluation, dry_run)
    print()
    print_workflow_plan(stages)

    if dry_run:
        print("\nDry run completed successfully.")
        print("Required input and selected workflow scripts were found.")
        print("No data was processed and no outputs were created or changed.")
        return {}

    print("\nStarting complete UCR identity-resolution workflow...")

    workflow_start = perf_counter()
    stage_timings = {}
    total_stages = len(stages)

    for stage_number, stage in enumerate(stages, start=1):
        print_stage_header(
            stage_number,
            total_stages,
            stage,
        )
        stage_timings[stage.name] = run_stage(stage)

    elapsed_seconds = perf_counter() - workflow_start
    print_workflow_summary(stage_timings, elapsed_seconds)

    return stage_timings


###############################################################################
# 6. Run complete workflow
###############################################################################


def main() -> None:
    # Run the workflow and report any failed stage clearly.

    arguments = parse_arguments()

    try:
        run_identity_resolution(
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
            f"Identity-resolution workflow failed: {error}"
        ) from error


if __name__ == "__main__":
    main()
