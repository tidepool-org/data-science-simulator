"""
Palerm insulin model plus a third step, the action delay, so glucose lowering lags blood insulin.
Fitted to Heise 2017 (study NN1218-3887) mean GIR curves, insulin aspart 0.1 and 0.2 U/kg.
Exploratory: one curve shape for all doses.

Person-to-person spread: each person gets one timing factor (median 1) that multiplies all three
time constants (>1 slower, <1 faster). The action delay is refitted for each spread size so the
AVERAGE of the people matches Heise's mean curves (ACTION_DELAY_BY_SPREAD).

Use in place of SimpleMetabolismModel, one person per patient:
    k = draw_timing_factors(1, spread, random_state)[0]  # spread = 0.26 proposed
    metabolism_model = functools.partial(EffectDelayMetabolismModel, timing_factor=k,
                                         tau3=ACTION_DELAY_BY_SPREAD[spread])
"""

import numpy as np

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_models.models.treatment_models import PalermInsulinModel
from tidepool_data_science_models.utils import get_timeseries

# Spread (CV of the timing factor) -> fitted action delay (min); spread 0 = everyone the same.
# Proposed: 0.26 (study 3887 reported CV of individual time to peak, NovoLog 0.2 U/kg); 0.11, 0.17, 0.39 for sensitivity.
ACTION_DELAY_BY_SPREAD = {0.0: 31.6, 0.11: 32.7, 0.17: 34.2, 0.26: 37.8, 0.39: 45.8}


def draw_timing_factors(n, spread, random_state):
    """Lognormal person timing factors, median 1, CV = spread."""
    return np.exp(random_state.normal(0, np.sqrt(np.log(1 + spread ** 2)), n))


def chain_density(t1, t2, t3, t_min):
    """Unit-area output rate of three exponential steps in series (distinct time constants)."""
    return (t1 * np.exp(-t_min / t1) / ((t1 - t2) * (t1 - t3))
            + t2 * np.exp(-t_min / t2) / ((t2 - t1) * (t2 - t3))
            + t3 * np.exp(-t_min / t3) / ((t3 - t1) * (t3 - t2)))


class EffectDelayInsulinModel(PalermInsulinModel):

    def __init__(self, tau3=31.6, timing_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self._tau3 = tau3
        self._timing_factor = timing_factor

    def run(self, num_hours, insulin_amount, five_min=True):
        """Same as PalermInsulinModel.run with the third step added."""
        t_min = get_timeseries(num_hours, five_min=False)
        k = self._timing_factor
        rate = chain_density(self._tau1 * k, self._tau2 * k, self._tau3 * k, t_min)
        insulin_cleared = np.cumsum(insulin_amount * rate / self._Kcl)
        iob = insulin_amount - insulin_cleared
        bg = -1 * self._isf * insulin_cleared

        if five_min:
            t_min = get_timeseries(num_hours, five_min=True)
            bg = bg[t_min]
            iob = iob[t_min]

        return t_min, np.append(0, np.diff(bg)), bg, iob


class EffectDelayMetabolismModel(SimpleMetabolismModel):

    def __init__(self, *args, tau3=31.6, timing_factor=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        m = self.insulin_model
        self.insulin_model = EffectDelayInsulinModel(isf=m._isf, cir=m._cir, tau1=m._tau1, tau2=m._tau2,
                                                     tau3=tau3, timing_factor=timing_factor)
