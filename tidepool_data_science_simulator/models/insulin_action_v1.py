"""Insulin-action curves v1 for the simulated person.

Insulin passes through three stages in sequence: absorption from under the skin, insulin in the blood, and a lag
before glucose falls. Each stage is a first-order compartment with one time constant (minutes). The glucose-lowering
rate per unit dose is the output of the last stage, a curve with area 1, so a dose D lowers glucose by
D x ISF in total.

Selected through the patient's insulin type (scenario setting
``patient_model.metabolism_settings.patient_insulin_type``):

    rapid_acting_v1   NovoLog, Humalog   time constants 55 / 70 / 19.8 min, peak effect 86 min after the dose
    ultra_rapid_v1    Fiasp, Lyumjev     time constants 44 / 70 / 19.8 min, peak effect 79 min after the dose

and, for variation between people, ``patient_model.metabolism_settings.insulin_timing``:

    main (default) x 1.000, faster x 0.750, slower x 1.334 on all three time constants
    (one standard deviation either side of the main curve for a 29.4% between-person coefficient of variation).

Existing insulin types (rapid_acting_adult, fiasp, ...) are unchanged and keep the data-science-models curves.

Source: "Simulated patient insulin action" white paper, internal release v1.0 (frozen 2026-11-16): absorption and
blood time constants from Palerm 2009; the lag fitted to Heise et al. 2017 (trial NN1218-3887), NovoLog 0.1 U/kg
glucose infusion rate; Fiasp absorption fitted to the same study; between-person variation from the FDA Fiasp
clinical pharmacology review (NDA 208751). The reference implementation is scripts/model.py in that release.

Insulin on board is the share of the dose's glucose-lowering effect still to come, as for PalermInsulinModel (where
glucose lowering follows blood insulin) and in Loop.
"""

import numpy as np

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_models.models.treatment_models import PalermInsulinModel
from tidepool_data_science_models.utils import get_timeseries

V1_TIME_CONSTANTS_MIN = {
    "rapid_acting_v1": (55.0, 70.0, 19.8),
    "ultra_rapid_v1": (44.0, 70.0, 19.8),
}

# Median of the 12 per-person timing CVs (SD / mean) for trial NN1218-3887 (FDA Fiasp review, NDA 208751, PDF p25);
# lognormal sigma = sqrt(ln(1 + CV^2)); factors exp(-sigma), exp(+sigma).
TIMING_CV = 0.2939303904923599
TIMING_SIGMA = float(np.sqrt(np.log(1 + TIMING_CV ** 2)))
INSULIN_TIMING_FACTORS = {
    "main": 1.0,
    "faster": float(np.exp(-TIMING_SIGMA)),
    "slower": float(np.exp(TIMING_SIGMA)),
}


# The simulator applies each dose's glucose effect over a fixed window. Existing insulin types use 8 hours. For the v1
# curves the window is the shortest whole number of hours (at least 8) after which at most this share of the dose's
# effect is still to come, so at most 0.1% of each dose's effect falls outside the window.
DEFAULT_PREDICTION_HORIZON_HOURS = 8
MAX_EFFECT_OUTSIDE_WINDOW = 0.001


def is_v1_insulin_type(insulin_type):
    return insulin_type in V1_TIME_CONSTANTS_MIN


def v1_time_constants(insulin_type, insulin_timing="main"):
    if not is_v1_insulin_type(insulin_type):
        raise ValueError(f"{insulin_type!r} is not a v1 insulin type; expected one of {sorted(V1_TIME_CONSTANTS_MIN)}")
    if insulin_timing not in INSULIN_TIMING_FACTORS:
        raise ValueError(f"insulin_timing must be one of {sorted(INSULIN_TIMING_FACTORS)}, got {insulin_timing!r}")
    return tuple(tau * INSULIN_TIMING_FACTORS[insulin_timing] for tau in V1_TIME_CONSTANTS_MIN[insulin_type])


def prediction_horizon_hours(insulin_type, insulin_timing="main"):
    """Hours over which the simulator applies each dose's effect: 8 for existing insulin types; for v1 curves, long
    enough that at most MAX_EFFECT_OUTSIDE_WINDOW of the effect falls outside."""
    if not is_v1_insulin_type(insulin_type):
        return DEFAULT_PREDICTION_HORIZON_HOURS
    taus = v1_time_constants(insulin_type, insulin_timing)
    hours = DEFAULT_PREDICTION_HORIZON_HOURS
    while 1 - effect_fraction_delivered(taus, np.array([hours * 60.0]))[0] > MAX_EFFECT_OUTSIDE_WINDOW:
        hours += 1
    return hours


def effect_fraction_delivered(time_constants, t_min):
    """Cumulative share of the total glucose-lowering effect delivered by time t (minutes), exact closed form.

    For first-order stages in series with distinct time constants tau_i, the remaining share is
    sum_i A_i exp(-t / tau_i) with A_i = tau_i^(n-1) / prod_{j != i} (tau_i - tau_j).
    """
    taus = np.array(time_constants, dtype=float)
    for i in range(1, len(taus)):  # nudge exact ties so the closed form is defined
        while np.min(np.abs(taus[i] - taus[:i]) / taus[i]) < 1e-6:
            taus[i] *= 1.000001
    n = len(taus)
    t = np.asarray(t_min, dtype=float)
    remaining = np.zeros_like(t)
    for i, tau_i in enumerate(taus):
        coefficient = tau_i ** (n - 1) / np.prod([tau_i - tau_j for j, tau_j in enumerate(taus) if j != i])
        remaining += coefficient * np.exp(-t / tau_i)
    return np.clip(1.0 - remaining, 0.0, 1.0)


class InsulinActionV1Model(PalermInsulinModel):
    """Three-stage insulin action with the same interface as PalermInsulinModel.

    ``run`` returns (t, bg_delta, bg, iob) like PalermInsulinModel.run; ``iob`` is the effect still to come.
    """

    def __init__(self, insulin_type, insulin_timing="main", **kwargs):
        tau1, tau2, tau3 = v1_time_constants(insulin_type, insulin_timing)
        super().__init__(tau1=tau1, tau2=tau2, **kwargs)
        self.name = "InsulinActionV1"
        self._tau3 = tau3
        self.insulin_type = insulin_type
        self.insulin_timing = insulin_timing

    @property
    def time_constants(self):
        return (self._tau1, self._tau2, self._tau3)

    def run(self, num_hours, insulin_amount, five_min=True):
        t_min = get_timeseries(num_hours, five_min=five_min)
        delivered = insulin_amount * effect_fraction_delivered(self.time_constants, t_min) / self._Kcl
        iob = insulin_amount - delivered
        bg = -1 * self._isf * delivered
        bg_delta = np.append(0, bg[1:] - bg[:-1])
        return t_min, bg_delta, bg, iob


class SimpleMetabolismModelV1(SimpleMetabolismModel):
    """SimpleMetabolismModel that also accepts the v1 insulin types.

    For a v1 insulin type the insulin model is InsulinActionV1Model and ``run`` and ``get_iob_from_sbr`` use the
    curve's window (prediction_horizon_hours) instead of 8 hours. Every other insulin type behaves exactly like
    SimpleMetabolismModel (insulin_timing must then be "main").
    """

    def __init__(self, *args, patient_insulin_type="rapid_acting_adult", insulin_timing="main", **kwargs):
        v1 = is_v1_insulin_type(patient_insulin_type)
        if not v1 and insulin_timing != "main":
            raise ValueError(f"insulin_timing {insulin_timing!r} applies only to v1 insulin types "
                             f"{sorted(V1_TIME_CONSTANTS_MIN)}, not {patient_insulin_type!r}")
        # Build the base model with a known type so it doesn't warn, then replace the insulin model.
        super().__init__(*args, patient_insulin_type="rapid_acting_adult" if v1 else patient_insulin_type, **kwargs)
        self.prediction_horizon_hours = prediction_horizon_hours(patient_insulin_type, insulin_timing)
        if v1:
            self.insulin_model = InsulinActionV1Model(
                patient_insulin_type, insulin_timing, isf=self._isf, cir=self._cir)
        self._v1 = v1

    def run(self, *args, num_hours=None, **kwargs):
        if num_hours is None:
            num_hours = self.prediction_horizon_hours
        return super().run(*args, num_hours=num_hours, **kwargs)

    def get_iob_from_sbr(self, sbr_actual):
        """Insulin on board over the window from a basal rate held constant for the window before t=0.

        Same calculation as SimpleMetabolismModel.get_iob_from_sbr (which fixes 8 hours before and after t=0), with the
        curve's window; for existing insulin types it is the parent's result."""
        if not self._v1:
            return super().get_iob_from_sbr(sbr_actual)
        steps = int(self.prediction_horizon_hours * 12)  # 5-min steps
        _, _, _, unit_iob = self.insulin_model.run(self.prediction_horizon_hours, insulin_amount=sbr_actual / 12,
                                                   five_min=True)
        padded = np.append(unit_iob, np.zeros(steps))
        # Pulses k = 0 ... steps-1 steps before t=0 contribute padded[j + k] at step j (same indexing as the parent).
        return np.array([padded[j:j + steps].sum() for j in range(steps)])
