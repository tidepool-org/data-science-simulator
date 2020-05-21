import numpy as np


from tdsm.models.simple_metabolism_model import SimpleMetabolismModel


def get_bgri(bg_df):
    # Calculate LBGI and HBGI using equation from
    # Clarke, W., & Kovatchev, B. (2009)
    bgs = bg_df.copy()
    bgs[bgs < 1] = 1  # this is added to take care of edge case BG <= 0
    transformed_bg = 1.509 * ((np.log(bgs) ** 1.084) - 5.381)
    risk_power = 10 * (transformed_bg) ** 2
    low_risk_bool = transformed_bg < 0
    high_risk_bool = transformed_bg > 0
    rlBG = risk_power * low_risk_bool
    rhBG = risk_power * high_risk_bool
    LBGI = np.mean(rlBG)
    HBGI = np.mean(rhBG)
    BGRI = LBGI + HBGI

    return LBGI, HBGI, BGRI


def lbgi_risk_score(lbgi):
    if lbgi > 10:
        risk = 4
    elif lbgi > 5:
        risk = 3
    elif lbgi > 2.5:
        risk = 2
    elif lbgi > 0:
        risk = 1
    else:
        risk = 0
    return risk


def hbgi_risk_score(hbgi):
    if hbgi > 18:
        risk = 4
    elif hbgi > 9:
        risk = 3
    elif hbgi > 4.5:
        risk = 2
    elif hbgi > 0:
        risk = 1
    else:
        risk = 0
    return risk


def get_dka_risk_hours(temp_basals, iob_array, sbr):

    # Use refactor of metabolism model
    metab_model = SimpleMetabolismModel(
        insulin_sensitivity_factor=0, carb_insulin_ratio=0
    )
    steady_state_iob = metab_model.get_steady_state_iob_from_sbr(
        sbr, use_fda_submission_constant=True
    )

    fifty_percent_steady_state_iob = steady_state_iob / 2

    indices_with_less_50percent_sbr_iob = iob_array < fifty_percent_steady_state_iob

    hours_with_less_50percent_sbr_iob = (
        np.sum(indices_with_less_50percent_sbr_iob) * 5 / 60
    )
    return hours_with_less_50percent_sbr_iob


def dka_risk_score(hours_with_less_50percent_sbr_iob):
    if hours_with_less_50percent_sbr_iob >= 16:
        risk = 4
    elif hours_with_less_50percent_sbr_iob >= 12:
        risk = 3
    elif hours_with_less_50percent_sbr_iob >= 8:
        risk = 2
    elif hours_with_less_50percent_sbr_iob >= 4:
        risk = 1
    else:
        risk = 0
    return risk


def suspend_risk_score(minutes_of_suspend):
    if minutes_of_suspend >= 8 * 60:
        risk = 4
    elif minutes_of_suspend >= 5 * 60:
        risk = 3
    elif minutes_of_suspend >= 2 * 60:
        risk = 2
    elif minutes_of_suspend >= 1 * 60:
        risk = 1
    else:
        risk = 0
    return risk
