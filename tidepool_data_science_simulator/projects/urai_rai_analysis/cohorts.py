"""Cohort filtering and sampling (TRSET-74, TRSET-77).

Cohorts are filtered from the eligible pool and sampled independently. Each cohort has a seeded ordering of its eligible
patients; the first n are the cohort and the rest are the replacement candidates, in order.
"""
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from tidepool_data_science_simulator.projects.urai_rai_analysis.config import CohortSpec, RunConfig

logger = logging.getLogger(__name__)

AGE_BANDS = ((18, "<18"), (60, "18-59"))
AGE_BAND_OLDEST = "60+"


def derive_seed(master_seed: int, *parts: str) -> int:
    """Stable 63-bit seed derived from the master seed and names (not Python's salted ``hash``)."""
    text = "|".join([str(master_seed), *parts])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") >> 1


def age_band(age: float) -> str:
    """Age band label used in the manifest."""
    for upper, label in AGE_BANDS:
        if age < upper:
            return label
    return AGE_BAND_OLDEST


@dataclass(frozen=True)
class CohortDraw:
    """Patients drawn for one cohort."""
    spec: CohortSpec
    patient_ids: List[str]
    eligible: int
    shortfall_note: Optional[str]


def apply_filter(table: pd.DataFrame, spec: CohortSpec) -> pd.DataFrame:
    """Rows of ``table`` matching the cohort filter (vectorised)."""
    mask = pd.Series(True, index=table.index)
    f = spec.filter
    if "age_min" in f:
        mask &= table["age"] >= f["age_min"]
    if "age_max" in f:
        mask &= table["age"] < f["age_max"]
    if "isf_min" in f:
        mask &= table["isf"] >= f["isf_min"]
    if "isf_max" in f:
        mask &= table["isf"] <= f["isf_max"]
    return table[mask]


def cohort_order(table: pd.DataFrame, spec: CohortSpec, master_seed: int) -> List[str]:
    """The cohort's eligible patients in seeded order (seed derived from the run seed and the cohort name).

    ``table`` holds eligible patients only. The same table, spec and seed always give the same order.
    """
    ids = sorted(apply_filter(table, spec)["patient_id"])
    seed = derive_seed(master_seed, "cohort", spec.name)
    logger.info("Cohort %s: %d eligible, ordering seed %d", spec.name, len(ids), seed)
    return np.random.default_rng(seed).permutation(np.array(ids, dtype=object)).tolist() if ids else []


def draw_cohort(table: pd.DataFrame, spec: CohortSpec, config: RunConfig) -> CohortDraw:
    """Draw one cohort: the first n patients of its seeded order.

    Raises
    ------
    ValueError
        If fewer than ``spec.n`` patients are eligible and the config does not accept a smaller group.
    """
    ids = cohort_order(table, spec, config.master_seed)
    note = None
    n = spec.n
    if len(ids) < n:
        note = f"cohort {spec.name}: requested n={n}, only {len(ids)} eligible"
        if not config.accept_smaller_group:
            raise ValueError(note + " (set accept_smaller_group to allow)")
        logger.warning("Accepted shortfall: %s", note)
        n = len(ids)
    return CohortDraw(spec=spec, patient_ids=ids[:n], eligible=len(ids), shortfall_note=note)
