__author__ = "Jason Meno"

from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.measures import Carb


class VirtualPatientISA(VirtualPatient):
    """
    A Virtual Patient for iCGM Sensitivity Analysis (ISA)

    Unique Parameters
    -----------------
    t0 : datetime
        The time = 0 evaluation datetime for the simulation.
    analysis_type : str ('temp_basal_only', 'correction_bolus', 'meal_bolus')
        The sensitivity analysis type.
            - 'temp_basal_only' will only issue Loop temp basals.
            - 'correction_bolus' will accept the t0 bolus recommendation and issue temp basals for the remaining time
            - 'meal_bolus' will create a carb event at t0, accept the t0 bolus recommendation, and issue temp basals

    """

    def __init__(self, time, pump, sensor, metabolism_model, patient_config, t0, analysis_type):

        super().__init__(time, pump, sensor, metabolism_model, patient_config)

        self.t0 = t0
        self.patient_config.recommendation_accept_prob = 0
        self.analysis_type = analysis_type

        # Set a 30g carb entry at t0 for the meal_bolus analysis type
        if analysis_type == "meal_bolus":
            carb_entry = Carb(30, "g", 180)
            self.carb_event_timeline.events.update({t0: carb_entry})
            self.pump.carb_event_timeline.events.update({t0: carb_entry})

    def does_accept_bolus_recommendation(self, bolus):
        """
        This override sets the patient to accept the recommended bolus only at t0
        if the iCGM Sensitivity analysis_type is meal_bolus or correction_bolus

        Parameters
        ----------
        bolus: Bolus
            The recommended bolus

        Returns
        -------
        bool
            True if patient accepts
        """

        currently_t0 = self.time == self.t0
        analysis_type_uses_bolus = self.analysis_type in ["meal_bolus", "correction_bolus"]

        if currently_t0 and analysis_type_uses_bolus:
            does_accept = True
        else:
            does_accept = False

        return does_accept
