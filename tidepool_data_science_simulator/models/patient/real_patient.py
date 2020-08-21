__author__ = "Cameron Summers"

from tidepool_data_science_simulator.models.patient.virtual_patient import (
    VirtualPatientState, VirtualPatient
)


class RealPatient(VirtualPatient):

    def __init__(self, time, pump, sensor, patient_config):

        super().__init__(time, pump, sensor, metabolism_model=None, patient_config=patient_config)

    def init(self):
        pass

    def get_state(self):
        """
        Get the current state of the patient.

        Returns
        -------
        VirtualPatientState
        """
        sensor_state = self.sensor.get_state()  # todo: put whole state below

        pump_state = None
        if self.pump is not None:
            pump_state = self.pump.get_state()

        patient_state = VirtualPatientState(
            bg=None,
            bg_prediction=None,
            sensor_bg=sensor_state.sensor_bg,
            sensor_bg_prediction=sensor_state.sensor_bg_prediction,
            iob=None,
            iob_prediction=None,
            sbr=None,
            isf=None,
            cir=None,
            pump_state=pump_state,
            bolus=None,
            carb=None,
            actions=None
        )

        return patient_state

    def update(self, time, **kwargs):
        """
        Move the state forward in time.
        """
        self.time = time
        self.sensor.update(time)
        self.pump.update(time)

    def add_event(self, time_of_event, event):
        pass


class RealPatientReplay(RealPatient):
    def update(self, time, **kwargs):
        self.time = time

        if self.pump is not None:
            self.pump.update(time)

        self.sensor.update(time)
