"""
Streamlit MVP for running a Tidepool Loop risk assessment without a terminal.

View layer only -- all simulator/validation logic lives in gui_runner.py.
Library browsing (listing config collections, resolving a chosen name to a
path) lives here, not in gui_runner.py, per the locked extensibility
constraint (design doc, 2026-07-21) that keeps the door open for a future
"configure parameters directly" mode to hand gui_runner a freshly-written
temp directory with no changes required there.
"""

import os
import threading

import pandas as pd
import streamlit as st

from tidepool_data_science_simulator.utils import PROJECT_ROOT_DIR
from tidepool_data_science_simulator.projects.risk.gui_runner import (
    run_risk_assessment,
    validate_config_dir,
)

LIBRARY_ROOT = os.path.join(PROJECT_ROOT_DIR, "scenario_configs", "tidepool_risk_v2", "loop_risk_v2_0")

STAGE_ORDER = ["pre", "no_loop", "post"]
STAGE_DISPLAY = {"pre": "Pre-mitigation", "no_loop": "No Loop", "post": "Post-mitigation"}


def _list_collections():
    if not os.path.isdir(LIBRARY_ROOT):
        return []
    return sorted(d for d in os.listdir(LIBRARY_ROOT) if os.path.isdir(os.path.join(LIBRARY_ROOT, d)))


def _list_tlr_dirs(collection_dir):
    return sorted(
        d for d in os.listdir(collection_dir)
        if os.path.isdir(os.path.join(collection_dir, d)) and "TLR-" in d
    )


def _init_session_state():
    defaults = {
        "cancel_event": None,
        "run_thread": None,
        "progress": None,  # (completed, total, risk_dir_name)
        "run_result": None,
        "run_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _start_run(config_dir, target_risk_dir):
    cancel_event = threading.Event()
    st.session_state.cancel_event = cancel_event
    st.session_state.progress = None
    st.session_state.run_result = None
    st.session_state.run_error = None

    def _progress_callback(completed, total, risk_dir_name):
        st.session_state.progress = (completed, total, risk_dir_name)

    def _target():
        try:
            result = run_risk_assessment(
                config_dir,
                target_risk_dir=target_risk_dir,
                progress_callback=_progress_callback,
                cancel_event=cancel_event,
            )
            st.session_state.run_result = result
        except Exception as exc:  # surfaced in the UI -- never swallowed
            st.session_state.run_error = str(exc)

    thread = threading.Thread(target=_target, daemon=True)
    st.session_state.run_thread = thread
    thread.start()


def _render_stage_table(assessment):
    rows = []
    for stage in STAGE_ORDER:
        stage_result = assessment.stages.get(stage)
        if stage_result is None:
            continue
        rows.append({
            "Stage": STAGE_DISPLAY[stage],
            "Harm type": stage_result.harm_type,
            "Severity": stage_result.severity,
            "TIR %": stage_result.tir,
            "TBR %": stage_result.tbr,
            "TAR %": stage_result.tar,
            "N sims": stage_result.n_sims,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def _render_risk_dir_result(result):
    with st.expander(result.risk_dir_name, expanded=True):
        if result.assessment is None:
            st.warning(
                "No usable data was found for this directory "
                "(no summary_results_Simulation-Configuration-TLR*.csv files)."
            )
            return

        assessment = result.assessment
        st.caption(f"{assessment.profile_count} profile(s) · timestamp {assessment.timestamp}")
        _render_stage_table(assessment)

        st.selectbox(
            "Which stage is the applicable pre-mitigation figure for this report? "
            "(TWI-0006 §2.g.ii.1: normally 'Pre-mitigation', but 'No Loop' "
            "if Loop wasn't automating during that period)",
            options=["Pre-mitigation", "No Loop"],
            index=None,
            placeholder="Select one -- required before reporting a pre-mitigation figure",
            key=f"premit_choice_{result.risk_dir_name}",
        )

        if assessment.catastrophic_findings:
            st.markdown("**Catastrophic findings (severity 4→5):**")
            st.dataframe(
                pd.DataFrame([f.to_dict() for f in assessment.catastrophic_findings]),
                hide_index=True,
            )

        if assessment.outlier_status != "ok":
            st.caption(f"Outlier detection: {assessment.outlier_status}")
        elif assessment.outlier_findings:
            st.markdown("**Outlier findings:**")
            st.dataframe(
                pd.DataFrame([f.to_dict() for f in assessment.outlier_findings]),
                hide_index=True,
            )

        for png_path in result.png_paths:
            if os.path.exists(png_path):
                st.image(png_path)


@st.fragment(run_every=1)
def _render_progress_fragment():
    thread = st.session_state.run_thread
    if thread is None:
        return

    if thread.is_alive():
        progress = st.session_state.progress
        if progress is None:
            st.info("Starting...")
        else:
            completed, total, risk_dir_name = progress
            st.progress(completed / total if total else 0, text=f"Run {completed} of {total}: {risk_dir_name}")
        if st.button("Cancel", key="cancel_run_button"):
            st.session_state.cancel_event.set()
        return

    # Thread finished -- clear it so this fragment stops polling.
    st.session_state.run_thread = None
    st.rerun()


def main():
    st.set_page_config(page_title="Tidepool Loop Risk Assessment", layout="wide")
    _init_session_state()
    st.title("Tidepool Loop Risk Assessment")

    collections = _list_collections()
    if not collections:
        st.error(f"No scenario config collections found under {LIBRARY_ROOT}")
        return

    collection = st.selectbox("Config collection", options=collections)
    config_dir = os.path.join(LIBRARY_ROOT, collection)

    tlr_dirs = _list_tlr_dirs(config_dir)
    scope_choice = st.radio(
        "Run scope", options=["All directories in this collection", "One specific directory"], horizontal=True
    )
    target_risk_dir = None
    if scope_choice == "One specific directory":
        target_risk_dir = st.selectbox("TLR-* directory", options=tlr_dirs)

    validation_result = validate_config_dir(config_dir, target_risk_dir)
    if validation_result.errors_by_file:
        st.error(f"{len(validation_result.errors_by_file)} config file(s) have validation errors:")
        for path, errors in validation_result.errors_by_file.items():
            for error in errors:
                st.write(f"- `{os.path.basename(path)}`: {error.error_message}")
    if validation_result.warnings_by_file:
        with st.expander(f"{len(validation_result.warnings_by_file)} config file(s) have warnings"):
            for path, warnings in validation_result.warnings_by_file.items():
                for warning in warnings:
                    st.write(f"- `{os.path.basename(path)}`: {warning.warning_message}")

    run_in_progress = st.session_state.run_thread is not None and st.session_state.run_thread.is_alive()

    if st.button("Run assessment", disabled=bool(validation_result.errors_by_file) or run_in_progress):
        _start_run(config_dir, target_risk_dir)
        st.rerun()

    if st.session_state.run_thread is not None:
        _render_progress_fragment()

    if st.session_state.run_error is not None:
        st.error(f"Run failed: {st.session_state.run_error}")

    result = st.session_state.run_result
    if result is not None:
        if result.cancelled:
            st.warning(f"Run cancelled. {len(result.risk_dir_results)} director(y/ies) completed before cancellation.")
        for risk_dir_result in result.risk_dir_results:
            _render_risk_dir_result(risk_dir_result)


if __name__ == "__main__":
    main()
