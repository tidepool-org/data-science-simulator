"""Cohort filtering and sampling (TRSET-74). Cohorts are filtered from the full pool and sampled independently."""
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from tidepool_data_science_simulator.projects.urai_rai_analysis.config import CohortSpec, RunConfig

logger = logging.getLogger(__name__)

AGE_BANDS = ((18, "<18"), (55, "18-54"))
AGE_BAND_OLDEST = "55+"


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


def eligible_pool(table: pd.DataFrame, min_age: float) -> pd.DataFrame:
    """Drop patients younger than the age floor. Done before any cohort filter."""
    return table[table["age"] >= min_age]


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


def draw_cohort(table: pd.DataFrame, spec: CohortSpec, config: RunConfig) -> CohortDraw:
    """Sample one cohort from the eligible pool.

    Raises
    ------
    ValueError
        If fewer than ``spec.n`` patients are eligible and the config does not accept a smaller group.
    """
    eligible = apply_filter(eligible_pool(table, config.min_age), spec)
    ids = sorted(eligible["patient_id"])
    note = None
    n = spec.n
    if len(ids) < n:
        note = f"cohort {spec.name}: requested n={n}, only {len(ids)} eligible"
        if not config.accept_smaller_group:
            raise ValueError(note + " (set accept_smaller_group to allow)")
        logger.warning("Accepted shortfall: %s", note)
        n = len(ids)
    rng = np.random.default_rng(derive_seed(config.master_seed, "cohort", spec.name))
    chosen = sorted(rng.choice(ids, size=n, replace=False).tolist()) if n else []
    return CohortDraw(spec=spec, patient_ids=chosen, eligible=len(ids), shortfall_note=note)
