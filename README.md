# Universal Customer Record Proof-of-Concept Platform

This repository contains the implementation of a Universal Customer Record
(UCR) proof-of-concept platform developed as part of the IOT635W final
project. The project investigates whether fragmented customer information can
be integrated and reconciled through a transparent and reproducible process to
form a unified master record and present a user-facing Single Customer View
(SCV).

The platform represents a fictional UK events organisation operating a single
venue. Customers interact with the organisation through five synthetic
enterprise systems: CRM, e-commerce, online sessions, customer support and
marketing. A canonical population of 10,000 synthetic customers provides the
hidden source of truth from which these operational records are generated.
Controlled inconsistencies, including missing values, duplicate records and
inconsistent formats, are introduced to emulate customer data fragmentation.

No authentic customer data or personally identifiable information is used.

## Project scope

The repository implements a controlled, batch-based reconstruction process. It
does not attempt to reproduce the scale or functionality of a commercial
Customer Data Platform. Real-time ingestion, production deployment,
enterprise-scale performance testing, user authentication and integration with
live organisational systems are outside the project scope.

The implemented process consists of four stages:

1. Synthetic data generation creates the canonical population, events data and
   five fragmented operational datasets.
2. A staged data pipeline ingests, standardises and consolidates the source
   records while preserving duplicate representations of the same individual.
3. Deterministic and probabilistic identity-resolution processes reconcile
   source records and assign stable UCR identifiers.
4. Gold-layer construction applies attribute survivorship, retains source
   provenance and produces the datasets used by the Streamlit SCV.

Protected ground-truth mappings are used only for controlled evaluation. They
are not accessed during final UCR assignment, Gold-layer construction or SCV
presentation.

## Repository structure

```text
UCR_project/
|-- data/                         Generated data-engineering outputs
|   |-- canonical/                Hidden canonical customer population
|   |-- raw/                      Fragmented operational source datasets
|   |-- bronze/                   Ingested source records
|   |-- silver/                   Standardised source records
|   |-- consolidated_silver/      Consolidated identity-resolution input
|   `-- gold/                     Unified records and SCV datasets
|       `-- evaluation/           Presentation and transparency details
|-- identity_resolution/         Resolution and evaluation outputs
|-- SRC/
|   |-- synth_data_gen/           Synthetic-data generation scripts
|   |-- pipeline/                 Bronze and Silver pipeline scripts
|   `-- Resolution/               Identity-resolution and Gold scripts
|-- docs/evaluation/              Preserved qualitative acceptance evidence
|-- run_project.py                Complete project orchestrator
|-- streamlit_app.py              Read-only Single Customer View
`-- requirements.txt              Direct Python dependencies
```

Generated directories are created by the relevant scripts when required.

## Environment setup

Python 3.12 is recommended. From the repository root, create and activate a
virtual environment before installing the required dependencies.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, enable scripts for the
current terminal session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Running the project

The complete operational workflow can be checked without processing data:

```powershell
python run_project.py --dry-run
```

The complete research workflow, including its evaluation stages, can also be
checked without processing data:

```powershell
python run_project.py --include-evaluation --dry-run
```

Run the complete operational reconstruction from synthetic-data generation to
Gold-layer construction with:

```powershell
python run_project.py
```

To reproduce the protected calibration and ground-truth evaluation stages as
well as the operational workflow, use:

```powershell
python run_project.py --include-evaluation
```

The full evaluation mode is not required for normal construction of the UCR.
It is retained to support research reproducibility.

## Single Customer View

After the Gold layer has been constructed, launch the Streamlit application
from the repository root:

```powershell
python -m streamlit run streamlit_app.py
```

The SCV is read-only and consumes governed Gold-layer outputs. It supports
profile retrieval, presentation of the unified master record, resolution
status and confidence, attribute provenance, alternative source values,
interaction history, linked-record inspection and a profile-specific cluster
construction map. The application does not read protected ground truth or
modify identity-resolution outcomes.

## Principal Gold outputs

The Gold construction stage creates:

```text
data/gold/
    ucr_master_records.csv
    ucr_record_links.csv
    ucr_attribute_provenance.csv
    ucr_interaction_summary.csv
    golden_record_summary.csv

data/gold/evaluation/
    ucr_attribute_candidates.csv
    ucr_interaction_details.csv
```

These outputs separate the operational unified record from the additional
detail required to demonstrate provenance, survivorship and interaction-level
traceability within the SCV.

## Validation and evaluation

Each principal stage performs automated validation before reporting successful
completion. The project additionally evaluates identity-resolution outcomes
using protected synthetic ground truth and preserves manually inspected Gold
profiles as qualitative acceptance evidence.

The acceptance audit is stored at:

```text
docs/evaluation/gold_layer_acceptance_audit.md
```

The audit complements the complete automated checks and quantitative
identity-resolution evaluation. It should not be interpreted as a statistical
estimate of error across the complete UCR population.

## Reproducibility

Reproducibility is treated as a project requirement. Synthetic-data generation
uses controlled seeds and defined configurations; pipeline and
identity-resolution stages are implemented as repeatable scripts; and the
orchestrators preserve the required execution order and validate the presence
of expected outputs. A clean-clone test should be completed before submission
to confirm that the documented process can be repeated outside the development
working copy.

## Ethical statement

The project uses synthetic data to avoid the ethical and legal constraints
associated with authentic customer information. Although the data represents
plausible customer attributes and interactions, it does not describe real
individuals. The controlled ground truth is retained solely to evaluate the
technical artefact.
