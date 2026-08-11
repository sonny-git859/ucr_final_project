# Run this file from the project root with:
#
#     streamlit run streamlit_app.py
#
# Install the two application dependencies in the active environment with:
#
#     python -m pip install streamlit pandas
#
# The application is read-only and loads only governed Gold-layer outputs.
# It never accesses ground truth, calibration results or evaluation mappings.
#
###############################################################################
# Imports
###############################################################################

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st


###############################################################################
# 1. Configuration
###############################################################################

APP_TITLE = "Universal Customer Record Visualisation"
APP_SUBTITLE = "Supporting Final Project Report"

MASTER_ATTRIBUTES = [
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

ATTRIBUTE_LABELS = {
    "first_name": "First name",
    "surname": "Surname",
    "full_name": "Full name",
    "date_of_birth": "Date of birth",
    "primary_email": "Primary email",
    "telephone_number": "Telephone number",
    "address": "Address",
    "postcode": "Postcode",
    "preferred_contact_channel": "Preferred contact channel",
    "registration_date": "Registration date",
}

STATUS_LABELS = {
    "deterministically_linked": "Deterministically resolved",
    "probabilistically_linked": "Probabilistically resolved",
    "unresolved_singleton": "Unresolved singleton profile",
}

STATUS_HELP = {
    "deterministically_linked": (
        "The source records in this UCR were linked by exact or composite "
        "deterministic matching rules."
    ),
    "probabilistically_linked": (
        "At least one relationship in this UCR was accepted using weighted "
        "attribute similarity and the calibrated decision threshold."
    ),
    "unresolved_singleton": (
        "This UCR contains one unresolved source record. No sufficiently "
        "reliable relationship to another record was identified."
    ),
}

SOURCE_LABELS = {
    "CRM": "CRM profile",
    "ECOMMERCE": "E-commerce transaction",
    "MARKETING": "Marketing contact",
    "ONLINE": "Online session",
    "SUPPORT": "Support ticket",
}

SOURCE_COLOURS = {
    "CRM": "#dbeafe",
    "ECOMMERCE": "#dcfce7",
    "MARKETING": "#fef3c7",
    "ONLINE": "#ede9fe",
    "SUPPORT": "#fee2e2",
}

REQUIRED_COLUMNS = {
    "master": {
        "ucr_id",
        "resolution_status",
        "match_method",
        "cluster_size",
        "source_system_count",
        "source_systems",
        "cluster_confidence",
        "cluster_confidence_type",
        *MASTER_ATTRIBUTES,
        "identity_attribute_completeness",
        "total_linked_records",
    },
    "links": {
        "ucr_id",
        "staging_record_id",
        "source_system",
        "source_record_id",
        "deterministic_cluster_id",
        "probabilistic_cluster_id",
        "resolution_status",
        "match_method",
        "cluster_size",
        "cluster_confidence",
        "cluster_confidence_type",
        "deterministic_linking_rules",
    },
    "provenance": {
        "ucr_id",
        "attribute_name",
        "selected_value",
        "source_system",
        "source_record_id",
        "staging_record_id",
        "selection_rule",
        "populated_candidate_records",
        "distinct_candidate_values",
    },
    "interactions": {
        "ucr_id",
        "ecommerce_transaction_count",
        "ecommerce_total_revenue",
        "ecommerce_total_tickets",
        "ecommerce_distinct_events",
        "online_session_count",
        "marketing_contact_count",
        "support_ticket_count",
        "first_recorded_interaction",
        "last_recorded_interaction",
    },
    "candidates": {
        "ucr_id",
        "attribute_name",
        "candidate_value",
        "source_system",
        "source_record_id",
        "staging_record_id",
        "is_selected_source",
        "matches_selected_value",
        "is_alternative_value",
        "selection_rule",
    },
    "details": {
        "ucr_id",
        "staging_record_id",
        "source_system",
        "source_record_id",
        "interaction_type",
        "interaction_id",
        "interaction_date_time",
    },
    "summary": {"metric", "value"},
}


###############################################################################
# 2. Paths and data loading
###############################################################################


def find_project_root() -> Path:
    # Locate the project from either the script path or working directory.
    start_points = [Path.cwd(), Path(__file__).resolve().parent]
    checked: set[Path] = set()

    for start in start_points:
        for candidate in [start, *start.parents]:
            resolved = candidate.resolve()
            if resolved in checked:
                continue
            checked.add(resolved)
            if (resolved / "data" / "gold").is_dir():
                return resolved

    return Path(__file__).resolve().parent


PROJECT_ROOT = find_project_root()
GOLD_DIRECTORY = PROJECT_ROOT / "data" / "gold"
GOLD_EVALUATION_DIRECTORY = GOLD_DIRECTORY / "evaluation"

DATA_PATHS = {
    "master": GOLD_DIRECTORY / "ucr_master_records.csv",
    "links": GOLD_DIRECTORY / "ucr_record_links.csv",
    "provenance": GOLD_DIRECTORY / "ucr_attribute_provenance.csv",
    "interactions": GOLD_DIRECTORY / "ucr_interaction_summary.csv",
    "summary": GOLD_DIRECTORY / "golden_record_summary.csv",
    "candidates": (
        GOLD_EVALUATION_DIRECTORY / "ucr_attribute_candidates.csv"
    ),
    "details": GOLD_EVALUATION_DIRECTORY / "ucr_interaction_details.csv",
}


@st.cache_data(show_spinner="Loading governed Gold-layer data...")
def read_gold_csv(path_text: str) -> pd.DataFrame:
    # Load identifiers and display values without automatic type coercion.
    return pd.read_csv(
        path_text,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


@st.cache_data(show_spinner="Validating the Single Customer View inputs...")
def load_data(
    path_items: tuple[tuple[str, str], ...],
) -> dict[str, pd.DataFrame]:
    # Load every required table and enforce cross-file integrity.
    missing_files = [
        path
        for _, path in path_items
        if not Path(path).is_file()
    ]
    if missing_files:
        missing_text = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Required Gold-layer files are missing:\n"
            f"{missing_text}\n\n"
            "Run build_golden_records.py before starting the app."
        )

    tables = {
        name: read_gold_csv(path)
        for name, path in path_items
    }
    validate_data(tables)
    return tables


def validate_required_columns(
    table_name: str,
    dataframe: pd.DataFrame,
) -> None:
    # Reject incomplete schemas before the interface tries to use them.
    missing = REQUIRED_COLUMNS[table_name] - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: "
            + ", ".join(sorted(missing))
        )


def validate_data(tables: dict[str, pd.DataFrame]) -> None:
    # Enforce structural relationships relied on by the interface.
    for name, dataframe in tables.items():
        validate_required_columns(name, dataframe)

    master = tables["master"]
    links = tables["links"]
    provenance = tables["provenance"]
    interactions = tables["interactions"]
    candidates = tables["candidates"]
    details = tables["details"]

    if master.empty:
        raise ValueError("The Gold master table contains no UCR profiles.")
    if master["ucr_id"].eq("").any():
        raise ValueError("The Gold master table contains a blank UCR ID.")
    if master["ucr_id"].duplicated().any():
        raise ValueError("The Gold master table contains duplicate UCR IDs.")

    master_ids = set(master["ucr_id"])
    for name in [
        "links",
        "provenance",
        "interactions",
        "candidates",
        "details",
    ]:
        unknown = set(tables[name]["ucr_id"]) - master_ids
        if unknown:
            raise ValueError(
                f"{name} contains UCR IDs absent from the master table."
            )

    if interactions["ucr_id"].duplicated().any():
        raise ValueError("Interaction summaries must be unique by UCR ID.")
    if set(interactions["ucr_id"]) != master_ids:
        raise ValueError(
            "Every UCR must have exactly one interaction summary."
        )
    if links["staging_record_id"].duplicated().any():
        raise ValueError(
            "A staging record has been assigned to more than one UCR."
        )

    expected_provenance = len(master) * len(MASTER_ATTRIBUTES)
    if len(provenance) != expected_provenance:
        raise ValueError(
            "The provenance table must contain one row for every master "
            "attribute and UCR."
        )
    provenance_keys = provenance[["ucr_id", "attribute_name"]]
    if provenance_keys.duplicated().any():
        raise ValueError(
            "The provenance table contains duplicate UCR-attribute rows."
        )

    if set(details["staging_record_id"]) != set(
        links["staging_record_id"]
    ):
        raise ValueError(
            "Interaction details do not reconcile to the linked records."
        )


###############################################################################
# 3. General display helpers
###############################################################################


def clean_value(value: object) -> str:
    # Convert missing and whitespace-only display values to a safe blank.
    if pd.isna(value):
        return ""
    return str(value).strip()


def display_value(value: object) -> str:
    # Present unavailable values consistently throughout the interface.
    cleaned = clean_value(value)
    return cleaned if cleaned else "Not available"


def safe_html(value: object) -> str:
    # Escape source values before inserting them into styled HTML blocks.
    return html.escape(display_value(value))


def to_number(value: object, default: float = 0.0) -> float:
    # Convert a Gold numeric value while handling an empty source field.
    try:
        return float(clean_value(value).replace(",", ""))
    except ValueError:
        return default


def to_integer(value: object) -> int:
    # Convert count fields that may have been stored as decimal strings.
    return int(round(to_number(value)))


def to_boolean(value: object) -> bool:
    # Interpret the boolean representation written to Gold CSV files.
    return clean_value(value).casefold() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
    }


def format_currency(value: object) -> str:
    # Format synthetic transaction values in pounds sterling.
    return f"GBP {to_number(value):,.2f}"


def format_percentage(value: object) -> str:
    # Format stored proportions as percentages.
    return f"{to_number(value) * 100:.1f}%"


def format_count(value: object) -> str:
    # Format a count for a metric card.
    return f"{to_integer(value):,}"


def normalise_search(value: object) -> str:
    # Build a conservative key for case- and punctuation-insensitive search.
    cleaned = clean_value(value).casefold()
    return re.sub(r"[^a-z0-9]", "", cleaned)


def pretty_label(value: object) -> str:
    # Convert snake-case technical labels into readable interface text.
    return clean_value(value).replace("_", " ").strip().title()


def metric_from_summary(
    summary: pd.DataFrame,
    metric_name: str,
    default: str = "0",
) -> str:
    # Retrieve one metric from the long-form Gold build summary.
    matches = summary.loc[summary["metric"].eq(metric_name), "value"]
    if matches.empty:
        return default
    return clean_value(matches.iloc[0]) or default


def render_field(label: str, value: object) -> None:
    # Render one master identity attribute in a compact card.
    st.markdown(
        "<div class='field-card'>"
        f"<div class='field-label'>{html.escape(label)}</div>"
        f"<div class='field-value'>{safe_html(value)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_status(status: str) -> None:
    # Display resolution certainty honestly and consistently.
    label = STATUS_LABELS.get(status, pretty_label(status))
    if status == "unresolved_singleton":
        st.warning(f"**{label}.** {STATUS_HELP.get(status, '')}")
    elif status == "probabilistically_linked":
        st.info(f"**{label}.** {STATUS_HELP.get(status, '')}")
    else:
        st.success(f"**{label}.** {STATUS_HELP.get(status, '')}")


###############################################################################
# 4. Search and profile preparation
###############################################################################


@st.cache_data(show_spinner=False)
def prepare_search_index(master: pd.DataFrame) -> pd.DataFrame:
    # Precompute searchable identity text without exposing hidden data.
    indexed = master.copy()
    search_columns = [
        "ucr_id",
        "full_name",
        "primary_email",
        "telephone_number",
        "postcode",
    ]
    indexed["_search_text"] = indexed[search_columns].apply(
        lambda row: " ".join(
            clean_value(value).casefold()
            for value in row
            if clean_value(value)
        ),
        axis=1,
    )
    indexed["_search_key"] = indexed[search_columns].apply(
        lambda row: " ".join(normalise_search(value) for value in row),
        axis=1,
    )
    return indexed


def filter_profiles(
    indexed: pd.DataFrame,
    query: str,
    status_filter: str,
) -> pd.DataFrame:
    # Apply the resolution filter and an identity-only search query.
    results = indexed
    if status_filter != "All profile types":
        selected_status = next(
            status
            for status, label in STATUS_LABELS.items()
            if label == status_filter
        )
        results = results.loc[
            results["resolution_status"].eq(selected_status)
        ]

    cleaned_query = clean_value(query).casefold()
    search_key = normalise_search(query)
    if cleaned_query:
        literal_match = results["_search_text"].str.contains(
            re.escape(cleaned_query),
            na=False,
        )
        key_match = results["_search_key"].str.contains(
            re.escape(search_key),
            na=False,
        )
        results = results.loc[literal_match | key_match]

    return results.sort_values(
        ["full_name", "ucr_id"],
        kind="stable",
    ).head(50)


def build_result_label(row: pd.Series) -> str:
    # Build a compact, unambiguous option for profile selection.
    name = display_value(row.get("full_name", ""))
    email = display_value(row.get("primary_email", ""))
    postcode = display_value(row.get("postcode", ""))
    return f"{row['ucr_id']} | {name} | {email} | {postcode}"


def select_profile(master: pd.DataFrame) -> str | None:
    # Search and select one profile from a bounded result list.
    indexed = prepare_search_index(master)
    st.sidebar.markdown("### Profile search")
    query = st.sidebar.text_input(
        "Search profiles",
        placeholder="UCR ID, name, email, phone or postcode",
        help=(
            "Search is case-insensitive and ignores common punctuation in "
            "telephone numbers and postcodes."
        ),
    )
    status_options = ["All profile types", *STATUS_LABELS.values()]
    status_filter = st.sidebar.selectbox(
        "Resolution type",
        status_options,
    )
    results = filter_profiles(indexed, query, status_filter)

    if results.empty:
        st.sidebar.warning("No profiles match the current search.")
        return None

    labels = {
        build_result_label(row): clean_value(row["ucr_id"])
        for _, row in results.iterrows()
    }
    st.sidebar.caption(
        f"Showing {len(results):,} result(s), limited to 50."
    )
    selected_label = st.sidebar.selectbox(
        "Select a UCR profile",
        list(labels),
    )
    return labels[selected_label]


def profile_tables(
    tables: dict[str, pd.DataFrame],
    ucr_id: str,
) -> dict[str, pd.DataFrame]:
    # Slice all long-form Gold tables to the selected profile.
    return {
        name: dataframe.loc[dataframe["ucr_id"].eq(ucr_id)].copy()
        if "ucr_id" in dataframe.columns
        else dataframe.copy()
        for name, dataframe in tables.items()
    }


###############################################################################
# 5. Overview page
###############################################################################


def render_overview(tables: dict[str, pd.DataFrame]) -> None:
    # Explain the artefact and summarise the governed UCR population.
    master = tables["master"]
    links = tables["links"]
    summary = tables["summary"]

    st.title(APP_TITLE)
    st.markdown(f"### {APP_SUBTITLE}")
    st.write(
        "This read-only proof-of-concept platform integrates fragmented "
        "and inconsistent customer information from five synthetic "
        "enterprise datasets into unified master records. The resulting "
        "Single Customer View (SCV) presents the selected master attributes, "
        "identity-resolution method and source provenance used to construct "
        "each Universal Customer Record (UCR)."
    )
    st.info(
        "All people, contact details and interactions shown in this "
        "application are synthetic. No authentic personal data is used."
    )

    status_counts = master["resolution_status"].value_counts()
    columns = st.columns(3)
    columns[0].metric("UCR profiles", f"{len(master):,}")
    columns[1].metric("Linked source records", f"{len(links):,}")
    anonymous_count = metric_from_summary(
        summary,
        "anonymous_records_excluded",
    )
    columns[2].metric(
        "Anonymous records excluded",
        format_count(anonymous_count),
    )

    columns = st.columns(3)
    columns[0].metric(
        "Deterministic profiles",
        f"{status_counts.get('deterministically_linked', 0):,}",
    )
    columns[1].metric(
        "Probabilistic profiles",
        f"{status_counts.get('probabilistically_linked', 0):,}",
    )
    columns[2].metric(
        "Unresolved singleton profiles",
        f"{status_counts.get('unresolved_singleton', 0):,}",
    )

    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown("#### How to interpret the platform")
        st.markdown(
            "- A **source record** is one fragmented representation from "
            "CRM, e-commerce, marketing, online or support.\n"
            "- A **UCR** groups source records resolved as representations "
            "of one customer identity.\n"
            "- A **golden record** applies explicit survivorship rules to "
            "select master attributes while retaining source provenance.\n"
            "- An **unresolved singleton profile** is retained when no "
            "sufficiently reliable link to another record was identified."
        )
    with right:
        distribution = (
            master["resolution_status"]
            .map(STATUS_LABELS)
            .fillna(master["resolution_status"].map(pretty_label))
            .value_counts()
            .rename_axis("Resolution type")
            .to_frame("Profiles")
        )
        st.markdown("#### Resolution composition")
        st.bar_chart(distribution, horizontal=True)

    st.markdown("#### Proof-of-concept disclaimer")
    st.write(
        "This interface demonstrates identity consolidation, provenance, "
        "matching transparency and cross-system interaction history. It is "
        "not a production CRM or commercial CDP: it has no live ingestion, "
        "record editing, authentication, operational reporting, continuous "
        "learning or human-in-the-loop verification, as defined in the "
        "project scope."
    )


###############################################################################
# 6. Master identity and resolution transparency
###############################################################################


def render_profile_header(master_row: pd.Series) -> None:
    # Introduce the profile before its detailed evidence tabs.
    status = clean_value(master_row["resolution_status"])
    st.title(display_value(master_row["full_name"]))
    st.caption(
        f"{master_row['ucr_id']} | "
        f"{STATUS_LABELS.get(status, pretty_label(status))}"
    )
    render_status(status)

    columns = st.columns(4)
    columns[0].metric(
        "Linked records",
        format_count(master_row["total_linked_records"]),
    )
    columns[1].metric(
        "Source systems",
        format_count(master_row["source_system_count"]),
    )
    columns[2].metric(
        "Identity completeness",
        format_percentage(
            master_row["identity_attribute_completeness"]
        ),
    )
    columns[3].metric(
        "Cluster confidence",
        format_percentage(master_row["cluster_confidence"]),
        help=display_value(master_row["cluster_confidence_type"]),
    )


def render_master_identity(master_row: pd.Series) -> None:
    # Display the surviving identity without implying missing data is zero.
    st.markdown("### Master identity")
    st.caption(
        "These master values were selected by the governed Gold-layer "
        "survivorship rules. Their source provenance is available in the "
        "Provenance tab."
    )

    rows = [
        [
            ("Full name", master_row["full_name"]),
            ("Date of birth", master_row["date_of_birth"]),
        ],
        [
            ("Primary email", master_row["primary_email"]),
            ("Telephone number", master_row["telephone_number"]),
        ],
        [
            ("Address", master_row["address"]),
            ("Postcode", master_row["postcode"]),
        ],
        [
            (
                "Preferred contact channel",
                master_row["preferred_contact_channel"],
            ),
            ("Registration date", master_row["registration_date"]),
        ],
    ]
    for field_row in rows:
        columns = st.columns(2)
        for column, (label, value) in zip(columns, field_row):
            with column:
                render_field(label, value)

    st.markdown("#### Contributing systems")
    st.write(display_value(master_row["source_systems"]))


def render_resolution_transparency(
    master_row: pd.Series,
    links: pd.DataFrame,
) -> None:
    # Explain how the records were assigned without exposing ground truth.
    st.markdown("### Resolution transparency")
    status = clean_value(master_row["resolution_status"])
    render_status(status)

    fields = pd.DataFrame(
        [
            {
                "Resolution property": "Method",
                "Value": display_value(master_row["match_method"]),
            },
            {
                "Resolution property": "Status",
                "Value": STATUS_LABELS.get(
                    status,
                    pretty_label(status),
                ),
            },
            {
                "Resolution property": "Cluster size",
                "Value": format_count(master_row["cluster_size"]),
            },
            {
                "Resolution property": "Confidence",
                "Value": format_percentage(
                    master_row["cluster_confidence"]
                ),
            },
            {
                "Resolution property": "Confidence classification",
                "Value": display_value(
                    master_row["cluster_confidence_type"]
                ),
            },
            {
                "Resolution property": "Source systems",
                "Value": display_value(master_row["source_systems"]),
            },
        ]
    )
    st.dataframe(fields, hide_index=True, use_container_width=True)

    deterministic_ids = sorted(
        value
        for value in links["deterministic_cluster_id"].unique()
        if clean_value(value)
    )
    probabilistic_ids = sorted(
        value
        for value in links["probabilistic_cluster_id"].unique()
        if clean_value(value)
    )
    linking_rules = sorted(
        value
        for value in links["deterministic_linking_rules"].unique()
        if clean_value(value)
    )

    with st.expander("View cluster and matching evidence"):
        st.markdown(
            f"**Deterministic cluster identifiers:** "
            f"{', '.join(deterministic_ids) or 'Not applicable'}"
        )
        st.markdown(
            f"**Probabilistic cluster identifiers:** "
            f"{', '.join(probabilistic_ids) or 'Not applicable'}"
        )
        st.markdown(
            f"**Deterministic linking rules:** "
            f"{', '.join(linking_rules) or 'No linking rule applied'}"
        )
        st.caption(
            "Confidence describes the match evidence available to the "
            "identity-resolution process. It does not establish that the "
            "resolved identity is certainly correct."
        )


def dot_text(value: object) -> str:
    # Escape text before inserting it into a Graphviz DOT label.
    return (
        clean_value(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
    )


def dot_identifier(prefix: str, position: int) -> str:
    # Create a safe internal node identifier independent of source values.
    return f"{prefix}_{position}"


def build_cluster_map(links: pd.DataFrame, ucr_id: str) -> str:
    # Map cluster membership without implying unrecorded pairwise matches.
    lines = [
        "digraph UCRCluster {",
        "rankdir=LR;",
        'graph [bgcolor="transparent", ranksep="0.65", nodesep="0.3"];',
        'node [fontname="Arial", fontsize="10", style="rounded,filled"];',
        'edge [color="#8292a3", arrowsize="0.7"];',
        (
            'ucr [label="UCR\\n'
            f'{dot_text(ucr_id)}", shape="doubleoctagon", '
            'fillcolor="#bfdbfe", color="#1d4ed8"];'
        ),
    ]
    deterministic_nodes: dict[str, str] = {}
    probabilistic_nodes: dict[str, str] = {}

    for position, row in links.reset_index(drop=True).iterrows():
        record_node = dot_identifier("record", position)
        source = clean_value(row.get("source_system", ""))
        source_label = SOURCE_LABELS.get(source, pretty_label(source))
        record_label = (
            f"{source_label}\\n"
            f"{dot_text(row.get('source_record_id', ''))}"
        )
        colour = SOURCE_COLOURS.get(source, "#f1f5f9")
        lines.append(
            f'{record_node} [label="{record_label}", shape="box", '
            f'fillcolor="{colour}", color="#64748b"];'
        )

        deterministic_id = clean_value(
            row.get("deterministic_cluster_id", "")
        )
        probabilistic_id = clean_value(
            row.get("probabilistic_cluster_id", "")
        )

        if deterministic_id:
            if deterministic_id not in deterministic_nodes:
                node_id = dot_identifier(
                    "deterministic",
                    len(deterministic_nodes),
                )
                deterministic_nodes[deterministic_id] = node_id
                lines.append(
                    f'{node_id} [label="Deterministic cluster\\n'
                    f'{dot_text(deterministic_id)}", shape="ellipse", '
                    'fillcolor="#e0f2fe", color="#0284c7"];'
                )
            origin_node = deterministic_nodes[deterministic_id]
            lines.append(f"{record_node} -> {origin_node};")
        else:
            origin_node = record_node

        if probabilistic_id:
            if probabilistic_id not in probabilistic_nodes:
                node_id = dot_identifier(
                    "probabilistic",
                    len(probabilistic_nodes),
                )
                probabilistic_nodes[probabilistic_id] = node_id
                lines.append(
                    f'{node_id} [label="Probabilistic cluster\\n'
                    f'{dot_text(probabilistic_id)}", shape="ellipse", '
                    'fillcolor="#fef3c7", color="#d97706"];'
                )
            target_node = probabilistic_nodes[probabilistic_id]
            lines.append(f"{origin_node} -> {target_node};")
        else:
            lines.append(f"{origin_node} -> ucr;")

    for node_id in probabilistic_nodes.values():
        lines.append(f"{node_id} -> ucr;")

    lines.append("}")
    return "\n".join(dict.fromkeys(lines))


def render_cluster_map(links: pd.DataFrame, ucr_id: str) -> None:
    # Visualise how source records were grouped into the selected UCR.
    st.markdown("### Cluster construction map")
    st.caption(
        "Follow the source records through deterministic and, where "
        "applicable, probabilistic clustering to the selected UCR. Colours "
        "identify the contributing source systems."
    )
    st.graphviz_chart(
        build_cluster_map(links, ucr_id),
        use_container_width=True,
    )
    st.info(
        "This diagram shows cluster membership and the resolution sequence. "
        "It does not represent every candidate-pair comparison performed "
        "during identity resolution."
    )


###############################################################################
# 7. Attribute provenance and alternatives
###############################################################################


def provenance_display(provenance: pd.DataFrame) -> pd.DataFrame:
    # Prepare the winning source evidence for presentation.
    output = provenance.copy()
    output["Attribute"] = output["attribute_name"].map(
        lambda value: ATTRIBUTE_LABELS.get(value, pretty_label(value))
    )
    output["Selected value"] = output["selected_value"].map(
        display_value
    )
    output["Source"] = output["source_system"].map(display_value)
    output["Source record"] = output["source_record_id"].map(
        display_value
    )
    output["Staging record"] = output["staging_record_id"].map(
        display_value
    )
    output["Candidate values"] = output["distinct_candidate_values"].map(
        format_count
    )
    return output[
        [
            "Attribute",
            "Selected value",
            "Source",
            "Source record",
            "Staging record",
            "Candidate values",
        ]
    ]


def render_attribute_provenance(
    provenance: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    # Present the selected source and every alternative considered.
    st.markdown("### Attribute provenance")
    st.caption(
        "Each selected value can be traced to the source record and explicit "
        "survivorship rule that supplied it."
    )
    st.dataframe(
        provenance_display(provenance),
        hide_index=True,
        use_container_width=True,
    )

    available_attributes = [
        attribute
        for attribute in MASTER_ATTRIBUTES
        if attribute in set(candidates["attribute_name"])
    ]
    if not available_attributes:
        st.info("No populated source candidates exist for this profile.")
        return

    selected_attribute = st.selectbox(
        "Inspect source candidates for",
        available_attributes,
        format_func=lambda value: ATTRIBUTE_LABELS.get(
            value,
            pretty_label(value),
        ),
    )
    candidate_rows = candidates.loc[
        candidates["attribute_name"].eq(selected_attribute)
    ].copy()
    candidate_rows["Selected source"] = candidate_rows[
        "is_selected_source"
    ].map(to_boolean)
    candidate_rows["Same surviving value"] = candidate_rows[
        "matches_selected_value"
    ].map(to_boolean)
    candidate_rows["Alternative value"] = candidate_rows[
        "is_alternative_value"
    ].map(to_boolean)
    candidate_rows = candidate_rows.rename(
        columns={
            "candidate_value": "Candidate value",
            "source_system": "Source",
            "source_record_id": "Source record",
            "staging_record_id": "Staging record",
        }
    )
    st.dataframe(
        candidate_rows[
            [
                "Candidate value",
                "Source",
                "Source record",
                "Staging record",
                "Selected source",
                "Same surviving value",
                "Alternative value",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    selected_provenance = provenance.loc[
        provenance["attribute_name"].eq(selected_attribute)
    ]
    if not selected_provenance.empty:
        st.markdown("#### Applied survivorship rule")
        st.write(
            display_value(selected_provenance.iloc[0]["selection_rule"])
        )

    if selected_attribute in {"first_name", "surname", "full_name"}:
        st.info(
            "Name components are selected as one coherent bundle from a "
            "single source record."
        )
    if selected_attribute in {"address", "postcode"}:
        st.info(
            "Address and postcode are selected as one coherent bundle from "
            "a single source record wherever both are available."
        )


###############################################################################
# 8. Customer interactions
###############################################################################


def render_interaction_summary(interaction_row: pd.Series) -> None:
    # Summarise cross-system activity without adding business analytics.
    st.markdown("### Customer interactions")
    columns = st.columns(4)
    columns[0].metric(
        "Transactions",
        format_count(interaction_row["ecommerce_transaction_count"]),
    )
    columns[1].metric(
        "Transaction revenue",
        format_currency(interaction_row["ecommerce_total_revenue"]),
    )
    columns[2].metric(
        "Tickets",
        format_count(interaction_row["ecommerce_total_tickets"]),
    )
    columns[3].metric(
        "Events",
        format_count(interaction_row["ecommerce_distinct_events"]),
    )

    columns = st.columns(3)
    columns[0].metric(
        "Online sessions",
        format_count(interaction_row["online_session_count"]),
    )
    columns[1].metric(
        "Marketing contacts",
        format_count(interaction_row["marketing_contact_count"]),
    )
    columns[2].metric(
        "Support tickets",
        format_count(interaction_row["support_ticket_count"]),
    )

    st.caption(
        "Recorded interaction range: "
        f"{display_value(interaction_row['first_recorded_interaction'])} "
        "to "
        f"{display_value(interaction_row['last_recorded_interaction'])}."
    )


def interaction_display_columns(details: pd.DataFrame) -> list[str]:
    # Choose useful columns while tolerating source-specific blanks.
    preferred = [
        "interaction_date_time",
        "source_system",
        "interaction_type",
        "interaction_id",
        "event_name",
        "event_date",
        "transaction_total",
        "ticket_quantity",
        "session_duration_seconds",
        "device_type",
        "consent_status",
        "issue_category",
        "ticket_status",
        "source_record_id",
        "staging_record_id",
    ]
    return [column for column in preferred if column in details.columns]


def render_interaction_details(details: pd.DataFrame) -> None:
    # Allow a user to inspect the record-level history behind aggregates.
    st.markdown("#### Interaction history")
    source_options = [
        source
        for source in SOURCE_LABELS
        if source in set(details["source_system"])
    ]
    selected_sources = st.multiselect(
        "Filter by source system",
        source_options,
        default=source_options,
        format_func=lambda value: SOURCE_LABELS.get(value, value),
    )
    filtered = details.loc[
        details["source_system"].isin(selected_sources)
    ].copy()
    filtered = filtered.sort_values(
        ["interaction_date_time", "source_system"],
        ascending=[False, True],
        kind="stable",
    )
    display_columns = interaction_display_columns(filtered)
    st.dataframe(
        filtered[display_columns].rename(
            columns={
                column: pretty_label(column)
                for column in display_columns
            }
        ),
        hide_index=True,
        use_container_width=True,
        height=410,
    )

    event_columns = [
        column
        for column in [
            "event_id",
            "event_name",
            "event_category",
            "event_date",
            "ticket_type",
            "ticket_quantity",
            "transaction_total",
        ]
        if column in details.columns
    ]
    if event_columns:
        events = details.loc[
            details.get("event_id", pd.Series(index=details.index)).ne("")
        ].copy()
        if not events.empty:
            with st.expander("View relevant event history"):
                st.dataframe(
                    events[event_columns].drop_duplicates().rename(
                        columns={
                            column: pretty_label(column)
                            for column in event_columns
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )


###############################################################################
# 9. Linked source records
###############################################################################


def build_linked_record_view(
    links: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    # Add source identity values to each record-link audit row.
    identity = candidates.pivot_table(
        index="staging_record_id",
        columns="attribute_name",
        values="candidate_value",
        aggfunc="first",
        fill_value="",
    ).reset_index()
    output = links.merge(
        identity,
        on="staging_record_id",
        how="left",
        validate="one_to_one",
    )
    preferred = [
        "source_system",
        "source_record_id",
        "staging_record_id",
        "full_name",
        "primary_email",
        "telephone_number",
        "address",
        "postcode",
        "match_method",
        "cluster_confidence",
        "deterministic_linking_rules",
    ]
    return output[
        [column for column in preferred if column in output.columns]
    ]


def render_linked_records(
    links: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    # Trace each contributing source record through to the selected UCR.
    st.markdown("### Linked source records")
    st.caption(
        "This table provides the traceability path from each fragmented "
        "source record to the selected UCR."
    )
    linked_view = build_linked_record_view(links, candidates)
    st.dataframe(
        linked_view.rename(
            columns={column: pretty_label(column) for column in linked_view}
        ),
        hide_index=True,
        use_container_width=True,
        height=480,
    )


###############################################################################
# 10. Single Customer View page
###############################################################################


def render_single_customer_view(
    tables: dict[str, pd.DataFrame],
) -> None:
    # Coordinate profile selection and the five transparency sections.
    selected_ucr_id = select_profile(tables["master"])
    if selected_ucr_id is None:
        st.title("Single Customer View")
        st.write("Change the profile search to select a UCR.")
        render_user_guide()
        return

    profile = profile_tables(tables, selected_ucr_id)
    master_row = profile["master"].iloc[0]
    interaction_row = profile["interactions"].iloc[0]
    render_profile_header(master_row)

    tabs = st.tabs(
        [
            "Master identity",
            "Resolution",
            "Provenance",
            "Interactions",
            "Linked records",
        ]
    )
    with tabs[0]:
        render_master_identity(master_row)
    with tabs[1]:
        render_resolution_transparency(master_row, profile["links"])
        with st.expander("View cluster construction map"):
            render_cluster_map(
                profile["links"],
                clean_value(master_row["ucr_id"]),
            )
    with tabs[2]:
        render_attribute_provenance(
            profile["provenance"],
            profile["candidates"],
        )
    with tabs[3]:
        render_interaction_summary(interaction_row)
        render_interaction_details(profile["details"])
    with tabs[4]:
        render_linked_records(
            profile["links"],
            profile["candidates"],
        )

    render_user_guide()


###############################################################################
# 11. Educational guidance and styling
###############################################################################


def render_glossary() -> None:
    # Explain the central terms used to communicate the artefact.
    st.sidebar.markdown("---")
    with st.sidebar.expander("Identity-resolution glossary"):
        st.markdown(
            "**Golden record**  \n"
            "The governed master representation selected for one UCR.\n\n"
            "**Survivorship**  \n"
            "Rules used to select a master value from competing sources.\n\n"
            "**Deterministic matching**  \n"
            "Linking based on exact or composite governed rules.\n\n"
            "**Probabilistic matching**  \n"
            "Linking based on weighted similarities and a calibrated "
            "threshold.\n\n"
            "**Attribute provenance**  \n"
            "Evidence identifying where a selected value originated.\n\n"
            "**Unresolved singleton profile**  \n"
            "An unresolved source record retained as its own UCR when no "
            "sufficiently reliable link was identified."
        )


def render_user_guide() -> None:
    # Give first-time users a short workflow for exploring the SCV.
    st.markdown("---")
    st.markdown("## How to use the Single Customer View")
    st.caption(
        "Use this guide to retrieve a customer and inspect how the profile "
        "was constructed. The application is read-only."
    )

    with st.expander("Open the step-by-step user guide"):
        st.markdown(
            "**1. Find a profile**  \n"
            "Use **Profile search** in the left sidebar. Enter a UCR ID, "
            "customer name, email address, telephone number or postcode. "
            "Search is case-insensitive and ignores common punctuation in "
            "telephone numbers and postcodes.\n\n"
            "**2. Refine the results**  \n"
            "Use **Resolution type** to show all profiles or restrict the "
            "results to deterministic, probabilistic or unresolved "
            "singleton profiles. Up to 50 results are displayed.\n\n"
            "**3. Select a customer**  \n"
            "Choose a result from **Select a UCR profile**. The customer "
            "header and all five SCV tabs will update automatically.\n\n"
            "**4. Explore the profile**  \n"
            "Open **Master identity** for the selected master values; "
            "**Resolution** for status, confidence and the cluster map; "
            "**Provenance** for selected and alternative attribute values; "
            "**Interactions** for cross-system activity; and **Linked "
            "records** for the source-to-UCR audit trail.\n\n"
            "**5. Interpret uncertainty carefully**  \n"
            "A probabilistic confidence value describes the available match "
            "evidence, not absolute certainty. An unresolved singleton "
            "profile was deliberately retained because no sufficiently "
            "reliable link was identified."
        )

        st.markdown("#### Suggested demonstration")
        st.markdown(
            "1. Select one deterministic profile and inspect its linking "
            "rules.\n"
            "2. Select one probabilistic profile and review its confidence, "
            "cluster map and source alternatives.\n"
            "3. Select one unresolved singleton profile and confirm that "
            "the interface displays an uncertainty warning."
        )


def apply_styles() -> None:
    # Apply restrained styling suitable for a dissertation demonstration.
    styles = (
        "<style>"
        ":root {"
        "--ucr-ink: #17324d;"
        "--ucr-muted: #5f6f7f;"
        "--ucr-line: #dce5ec;"
        "--ucr-soft: #f5f8fa;"
        "}"
        ".stApp {background: #fbfcfd;}"
        "h1, h2, h3, h4 {color: var(--ucr-ink);}"
        ".field-card {"
        "min-height: 92px;"
        "margin: 0 0 0.8rem 0;"
        "padding: 0.9rem 1rem;"
        "background: white;"
        "border: 1px solid var(--ucr-line);"
        "border-radius: 0.65rem;"
        "box-shadow: 0 1px 2px rgba(23, 50, 77, 0.04);"
        "}"
        ".field-label {"
        "margin-bottom: 0.35rem;"
        "color: var(--ucr-muted);"
        "font-size: 0.78rem;"
        "font-weight: 700;"
        "letter-spacing: 0.04em;"
        "text-transform: uppercase;"
        "}"
        ".field-value {"
        "color: var(--ucr-ink);"
        "font-size: 1rem;"
        "overflow-wrap: anywhere;"
        "}"
        "[data-testid='stMetric'] {"
        "padding: 0.8rem;"
        "background: white;"
        "border: 1px solid var(--ucr-line);"
        "border-radius: 0.65rem;"
        "}"
        "[data-testid='stSidebar'] {"
        "border-right: 1px solid var(--ucr-line);"
        "}"
        "</style>"
    )
    st.markdown(
        styles,
        unsafe_allow_html=True,
    )


###############################################################################
# 12. Main execution
###############################################################################


def main() -> None:
    # Configure the application, load Gold data and render the selected page.
    st.set_page_config(
        page_title=f"{APP_TITLE} | {APP_SUBTITLE}",
        page_icon="\U0001F517",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    try:
        path_items = tuple(
            (name, str(path))
            for name, path in DATA_PATHS.items()
        )
        tables = load_data(path_items)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        st.error("The Single Customer View could not load its Gold data.")
        st.code(str(error), language="text")
        st.info(
            "From the project root, rebuild the Gold layer and then restart "
            "this application. No ground-truth or evaluation mapping should "
            "be supplied to the app."
        )
        st.stop()

    st.sidebar.markdown(f"## {APP_TITLE}")
    st.sidebar.caption(APP_SUBTITLE)
    page = st.sidebar.radio(
        "Navigation",
        ["Platform overview", "Single Customer View"],
    )
    render_glossary()

    if page == "Platform overview":
        render_overview(tables)
    else:
        render_single_customer_view(tables)

    st.markdown("---")
    st.caption(
        "Controlled synthetic-data research artefact | Read-only Gold layer "
        "| No authentic customer data"
    )


if __name__ == "__main__":
    main()
