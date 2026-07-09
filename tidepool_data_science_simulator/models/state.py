__author__ = "Cameron Summers"


class VirtualPatientState(object):
    """
    A class of instantaneous patient information.
    """

    def __init__(self, **kwargs):

        self.bg = kwargs.get("bg")
        self.sensor_bg = kwargs.get("sensor_bg")
        self.bg_prediction = kwargs.get("bg_prediction")
        self.sensor_bg_prediction = kwargs.get("sensor_bg_prediction")
        self.iob = kwargs.get("iob")
        self.iob_prediction = kwargs.get("iob_prediction")
        self.ei = kwargs.get("ei")
        self.pump_state = kwargs.get("pump_state")
        self.sbr = kwargs.get("sbr")
        self.isf = kwargs.get("isf")
        self.cir = kwargs.get("cir")
        self.bolus = kwargs.get("bolus")
        self.carb = kwargs.get("carb")
        self.actions = kwargs.get("actions")
        self.heart_rate = kwargs.get("heart_rate")

    def get_carb_value(self):
        value = None
        if self.carb is not None:
            value = self.carb.get_value()
        return value

    def get_bolus_value(self):
        value = None
        if self.bolus is not None:
            value = self.bolus.get_value()
        return value

    def get_carb_duration(self):
        value = None
        if self.carb is not None:
            value = self.carb.get_duration()
        return value

    def get_sbr_value(self):
        value = None
        if self.sbr is not None:
            value = self.sbr.get_value()
        return value

    def get_isf_value(self):
        value = None
        if self.isf is not None:
            value = self.isf.get_value()
        return value

    def get_cir_value(self):
        value = None
        if self.cir is not None:
            value = self.cir.get_value()
        return value


class PumpState(object):
    """
    A class to house the state information for a pump
    """

    def __init__(self, **kwargs):

        self.scheduled_basal_rate = kwargs.get("scheduled_basal_rate")
        self.scheduled_carb_insulin_ratio = kwargs.get("scheduled_cir")
        self.scheduled_insulin_sensitivity_factor = kwargs.get("schedule_isf")
        self.temp_basal_rate = kwargs.get("temp_basal_rate")
        self.delivered_basal_insulin = kwargs.get("delivered_basal_insulin")
        self.undelivered_basal_insulin = kwargs.get("undelivered_basal_insulin", 0)
        self.bolus = kwargs.get("bolus")
        self.carb = kwargs.get("carb")

    def get_schedule_basal_rate_value(self):
        value = None
        if self.scheduled_basal_rate is not None:
            value = self.scheduled_basal_rate.get_value()
        return value

    def get_scheduled_insulin_sensitivity_factor_value(self):
        value = None
        if self.scheduled_insulin_sensitivity_factor is not None:
            value = self.scheduled_insulin_sensitivity_factor.get_value()
        return value

    def get_scheduled_carb_insulin_ratio_value(self):
        value = None
        if self.scheduled_carb_insulin_ratio is not None:
            value = self.scheduled_carb_insulin_ratio.get_value()
        return value

    def get_carb_value(self):
        value = None
        if self.carb is not None:
            value = self.carb.get_value()
        return value

    def get_bolus_value(self):
        value = None
        if self.bolus is not None:
            value = self.bolus.get_value()
        return value

    def get_carb_duration(self):
        value = None
        if self.carb is not None:
            value = self.carb.get_duration()
        return value

    def get_temp_basal_rate_value(self, default=None):
        """
        Get the value of the temp basal and return None if one is not set.

        Returns
        -------
        float
        """
        value = default
        if self.temp_basal_rate is not None:
            value = self.temp_basal_rate.value

        return value

    def get_temp_basal_minutes_left(self, time, default=None):

        value = default
        if self.temp_basal_rate is not None:
            value = self.temp_basal_rate.get_minutes_remaining(time)
        return value