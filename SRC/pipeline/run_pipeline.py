###############################################################################
# Imports
###############################################################################

from collections.abc import Callable
from time import perf_counter

from ingest_raw_sources import ingest_all_sources
from standardise_bronze_sources import standardise_all_sources
from consolidate_silver_sources import consolidate_all_sources


###############################################################################
# 1. Pipeline configuration
###############################################################################

PipelineFunction = Callable[[], dict[str, int]]

PIPELINE_STAGES: list[tuple[str, PipelineFunction]] = [
    ("Raw-to-Bronze ingestion", ingest_all_sources),
    ("Bronze-to-Silver standardisation", standardise_all_sources),
    ("Silver source consolidation", consolidate_all_sources),
]


###############################################################################
# 2. Validation functions
###############################################################################

def validate_stage_record_counts(
    stage_results: dict[str, dict[str, int]],
) -> None:
    # Confirm that every pipeline stage preserves the source record counts.

    result_items = list(stage_results.items())
    reference_stage, reference_counts = result_items[0]

    for stage_name, row_counts in result_items[1:]:
        if row_counts != reference_counts:
            raise ValueError(
                f"Record counts after {stage_name} do not match "
                f"{reference_stage}."
            )


###############################################################################
# 3. Summary functions
###############################################################################

def print_stage_header(
    stage_number: int,
    total_stages: int,
    stage_name: str,
) -> None:
    # Print a clear divider before each pipeline stage.

    print("\n" + "=" * 79)
    print(f"Stage {stage_number} of {total_stages}: {stage_name}")
    print("=" * 79 + "\n")


def print_pipeline_summary(
    stage_results: dict[str, dict[str, int]],
    elapsed_seconds: float,
) -> None:
    # Print final record counts and successful stage completion details.

    final_counts = list(stage_results.values())[-1]

    print("\n" + "=" * 79)
    print("Complete UCR data pipeline finished successfully")
    print("=" * 79)
    print("\nFinal record summary:")

    for source_name, record_count in final_counts.items():
        print(f"  {source_name.upper():<11} {record_count:>7,} records")

    print(f"  {'TOTAL':<11} {sum(final_counts.values()):>7,} records")
    print(f"\nPipeline stages completed: {len(stage_results)}")
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")


###############################################################################
# 4. Pipeline orchestration
###############################################################################

def run_complete_pipeline() -> dict[str, dict[str, int]]:
    # Run all UCR data pipeline stages in their required sequence.

    print("Starting complete UCR data pipeline...")

    start_time = perf_counter()
    stage_results = {}
    total_stages = len(PIPELINE_STAGES)

    for stage_number, (stage_name, stage_function) in enumerate(
        PIPELINE_STAGES,
        start=1,
    ):
        print_stage_header(
            stage_number,
            total_stages,
            stage_name,
        )
        stage_results[stage_name] = stage_function()

    validate_stage_record_counts(stage_results)

    elapsed_seconds = perf_counter() - start_time
    print_pipeline_summary(stage_results, elapsed_seconds)

    return stage_results


###############################################################################
# 5. Run complete pipeline
###############################################################################

def main() -> None:
    # Run the complete pipeline and report any failed stage clearly.

    try:
        run_complete_pipeline()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"Complete pipeline failed: {error}") from error


if __name__ == "__main__":
    main()
