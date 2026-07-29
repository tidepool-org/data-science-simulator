"""Trace-tier data layer for simulator GUI work (TRSET-21).

Pure read-and-shape layer: parses a single simulator run's ``<sim_id>.tsv``
output into a typed :class:`SimulationTrace`. No plotting, no Streamlit, no file
writes -- consumed by the TRSET-22 renderer and TRSET-23 Streamlit integration.
"""
from tidepool_data_science_simulator.trace.reader import (
    SimulationTrace,
    TraceReadError,
    read_trace,
)

__all__ = ["SimulationTrace", "TraceReadError", "read_trace"]
