"""
Value validators for individual configuration fields.

Contains validation functions for all configuration parameters including
newer features like physical activity and max active insulin.
"""

import datetime
from typing import List, Any, Optional


class ValidationError:
    """Container for a validation error"""
    
    def __init__(self, field_path: str, error_message: str, value: Any = None):
        """
        Initialize a validation error.
        
        Parameters
        ----------
        field_path : str
            Dot-notation path to the field with the error (e.g., "patient.pump.basal_rate")
        error_message : str
            Description of the validation error
        value : Any, optional
            The problematic value
        """
        self.field_path = field_path
        self.error_message = error_message
        self.value = value
    
    def __str__(self):
        if self.value is not None:
            return f"❌ {self.field_path}: {self.error_message} (value: {self.value})"
        return f"❌ {self.field_path}: {self.error_message}"
    
    def __repr__(self):
        return f"ValidationError(field_path='{self.field_path}', error_message='{self.error_message}', value={self.value})"


class ValueValidators:
    """Collection of field-level validators for configuration values"""
    
    DATETIME_FORMAT = "%m/%d/%Y %H:%M:%S"
    
    @staticmethod
    def validate_basal_rate(value: float, field_path: str = "") -> List[ValidationError]:
        """
        Validate a basal rate value.
        
        Parameters
        ----------
        value : float
            Basal rate in U/hr
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        try:
            value = float(value)
            if not 0 <= value <= 100:
                errors.append(ValidationError(
                    field_path, 
                    f"Basal rate outside valid range [0, 100] U/hr",
                    value
                ))
        except (ValueError, TypeError):
            errors.append(ValidationError(
                field_path,
                "Basal rate must be numeric",
                value
            ))
        return errors
    
    @staticmethod
    def validate_carb_ratio(value: float, field_path: str = "") -> List[ValidationError]:
        """
        Validate a carb ratio value.
        
        Parameters
        ----------
        value : float
            Carb ratio in g/U
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        try:
            value = float(value)
            if not 0 < value <= 231:
                errors.append(ValidationError(
                    field_path,
                    f"Carb ratio outside valid range (0, 231] g/U",
                    value
                ))
        except (ValueError, TypeError):
            errors.append(ValidationError(
                field_path,
                "Carb ratio must be numeric",
                value
            ))
        return errors
    
    @staticmethod
    def validate_insulin_sensitivity(value: float, field_path: str = "") -> List[ValidationError]:
        """
        Validate an insulin sensitivity factor value.
        
        Parameters
        ----------
        value : float
            Insulin sensitivity in mg/dL/U
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        try:
            value = float(value)
            if not 0 < value <= 1200:
                errors.append(ValidationError(
                    field_path,
                    f"Insulin sensitivity outside valid range (0, 1200] mg/dL/U",
                    value
                ))
        except (ValueError, TypeError):
            errors.append(ValidationError(
                field_path,
                "Insulin sensitivity must be numeric",
                value
            ))
        return errors
    
    @staticmethod
    def validate_target_range(lower: float, upper: float, field_path: str = "") -> List[ValidationError]:
        """
        Validate a target glucose range.
        
        Parameters
        ----------
        lower : float
            Lower bound in mg/dL
        upper : float
            Upper bound in mg/dL
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        try:
            lower = float(lower)
            upper = float(upper)
            
            if lower > upper:
                errors.append(ValidationError(
                    field_path,
                    f"Lower value {lower} exceeds upper value {upper}"
                ))
            
            if lower < 0 or upper < 0:
                errors.append(ValidationError(
                    field_path,
                    "Target range values must be non-negative"
                ))
                
        except (ValueError, TypeError):
            errors.append(ValidationError(
                field_path,
                "Target range values must be numeric"
            ))
        return errors
    
    @staticmethod
    def validate_glucose_sensitivity_factor(value: float, field_path: str = "") -> List[ValidationError]:
        """
        Validate a glucose sensitivity factor (Type 2 model parameter).
        
        Parameters
        ----------
        value : float
            Glucose sensitivity factor in U/mg/dL
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        try:
            value = float(value)
            if not 0 <= value <= 500:
                errors.append(ValidationError(
                    field_path,
                    f"Glucose sensitivity factor outside valid range [0, 500] U/mg/dL",
                    value
                ))
        except (ValueError, TypeError):
            errors.append(ValidationError(
                field_path,
                "Glucose sensitivity factor must be numeric",
                value
            ))
        return errors
    
    @staticmethod
    def validate_basal_blood_glucose(value: float, field_path: str = "") -> List[ValidationError]:
        """
        Validate a basal blood glucose value (Type 2 model parameter).
        
        Parameters
        ----------
        value : float
            Basal blood glucose in mg/dL
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        try:
            value = float(value)
            if not 0 <= value <= 500:
                errors.append(ValidationError(
                    field_path,
                    f"Basal blood glucose outside valid range [0, 500] mg/dL",
                    value
                ))
        except (ValueError, TypeError):
            errors.append(ValidationError(
                field_path,
                "Basal blood glucose must be numeric",
                value
            ))
        return errors
    
    @staticmethod
    def validate_insulin_production_rate(value: float, field_path: str = "") -> List[ValidationError]:
        """
        Validate an insulin production rate (Type 2 model parameter).
        
        Parameters
        ----------
        value : float
            Insulin production rate in U/min
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        try:
            value = float(value)
            if not 0 <= value <= 5:
                errors.append(ValidationError(
                    field_path,
                    f"Insulin production rate outside valid range [0, 5] U/min",
                    value
                ))
        except (ValueError, TypeError):
            errors.append(ValidationError(
                field_path,
                "Insulin production rate must be numeric",
                value
            ))
        return errors
    
    @staticmethod
    def validate_max_active_insulin_multiplier(value: float, field_path: str = "") -> List[ValidationError]:
        """
        Validate max active insulin multiplier (Loop algorithm parameter).
        
        Parameters
        ----------
        value : float
            Multiplier value (typically 2.0)
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        try:
            value = float(value)
            if not 0.0 < value <= 10.0:
                errors.append(ValidationError(
                    field_path,
                    f"max_active_insulin_multiplier outside valid range (0, 10]",
                    value
                ))
            if value != 2.0:
                # This is a warning, not an error - print but don't add to errors list
                print(f"⚠️  {field_path}: Using non-standard max_active_insulin_multiplier: {value} (standard is 2.0)")
        except (ValueError, TypeError):
            errors.append(ValidationError(
                field_path,
                "max_active_insulin_multiplier must be numeric",
                value
            ))
        return errors
    
    @staticmethod
    def validate_metabolism_parameters(params: dict, field_path: str = "") -> List[ValidationError]:
        """
        Validate physical activity metabolism parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary containing metabolism parameters (w_hr, a, tau, n)
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        
        if 'w_hr' in params:
            try:
                w_hr = float(params['w_hr'])
                if not -10.0 <= w_hr <= 10.0:
                    errors.append(ValidationError(
                        f"{field_path}.w_hr",
                        f"w_hr outside valid range [-10, 10]",
                        w_hr
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    f"{field_path}.w_hr",
                    "w_hr must be numeric",
                    params.get('w_hr')
                ))
        
        if 'a' in params:
            try:
                a = float(params['a'])
                if not -1.0 <= a <= 1.0:
                    errors.append(ValidationError(
                        f"{field_path}.a",
                        f"a outside valid range [-1, 1]",
                        a
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    f"{field_path}.a",
                    "a must be numeric",
                    params.get('a')
                ))
        
        if 'tau' in params:
            try:
                tau = float(params['tau'])
                if not 0.0 < tau <= 1000.0:
                    errors.append(ValidationError(
                        f"{field_path}.tau",
                        f"tau outside valid range (0, 1000]",
                        tau
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    f"{field_path}.tau",
                    "tau must be numeric",
                    params.get('tau')
                ))
        
        if 'n' in params:
            try:
                n = float(params['n'])
                if not 0.0 < n <= 100.0:
                    errors.append(ValidationError(
                        f"{field_path}.n",
                        f"n outside valid range (0, 100]",
                        n
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    f"{field_path}.n",
                    "n must be numeric",
                    params.get('n')
                ))
        
        return errors
    
    @staticmethod
    def validate_physical_activity_entry(entry: dict, field_path: str = "") -> List[ValidationError]:
        """
        Validate a physical activity entry.
        
        Parameters
        ----------
        entry : dict
            Physical activity entry dictionary
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Required fields
        if 'start_time' not in entry:
            errors.append(ValidationError(
                field_path,
                "Missing required field 'start_time'"
            ))
        else:
            try:
                datetime.datetime.strptime(entry['start_time'], ValueValidators.DATETIME_FORMAT)
            except ValueError:
                errors.append(ValidationError(
                    f"{field_path}.start_time",
                    f"Invalid datetime format. Expected {ValueValidators.DATETIME_FORMAT}",
                    entry['start_time']
                ))
        
        # Must have either 'activity' or 'activity_ref'
        if 'activity' not in entry and 'activity_ref' not in entry:
            errors.append(ValidationError(
                field_path,
                "Must have either 'activity' or 'activity_ref'"
            ))
        
        # Validate activity reference format
        if 'activity_ref' in entry:
            ref = entry['activity_ref']
            if not isinstance(ref, str) or not ref.startswith('reusable.'):
                errors.append(ValidationError(
                    f"{field_path}.activity_ref",
                    "Invalid activity_ref format. Must start with 'reusable.'",
                    ref
                ))
        
        # Validate optional fields
        if 'duration' in entry:
            try:
                duration = float(entry['duration'])
                if not 0 < duration <= 480:
                    errors.append(ValidationError(
                        f"{field_path}.duration",
                        "Duration must be between 0 and 480 minutes",
                        duration
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    f"{field_path}.duration",
                    "Duration must be numeric",
                    entry.get('duration')
                ))
        
        if 'intensity' in entry:
            valid_intensities = ['light', 'moderate', 'high']
            if entry['intensity'] not in valid_intensities:
                errors.append(ValidationError(
                    f"{field_path}.intensity",
                    f"Invalid intensity. Must be one of: {valid_intensities}",
                    entry['intensity']
                ))
        
        if 'expected_hr' in entry:
            try:
                hr = float(entry['expected_hr'])
                if not 40 <= hr <= 220:
                    errors.append(ValidationError(
                        f"{field_path}.expected_hr",
                        "Heart rate must be between 40 and 220 bpm",
                        hr
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    f"{field_path}.expected_hr",
                    "Heart rate must be numeric",
                    entry.get('expected_hr')
                ))
        
        return errors
    
    @staticmethod
    def validate_datetime_format(value: str, field_path: str = "") -> List[ValidationError]:
        """
        Validate a datetime string format.
        
        Parameters
        ----------
        value : str
            Datetime string
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        try:
            datetime.datetime.strptime(value, ValueValidators.DATETIME_FORMAT)
        except ValueError:
            errors.append(ValidationError(
                field_path,
                f"Invalid datetime format. Expected {ValueValidators.DATETIME_FORMAT}",
                value
            ))
        except (TypeError, AttributeError):
            errors.append(ValidationError(
                field_path,
                "Datetime must be a string",
                value
            ))
        return errors
    
    @staticmethod
    def validate_time_format(value: str, field_path: str = "") -> List[ValidationError]:
        """
        Validate a time string format (HH:MM:SS).
        
        Parameters
        ----------
        value : str
            Time string
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        try:
            datetime.datetime.strptime(value, '%H:%M:%S').time()
        except ValueError:
            errors.append(ValidationError(
                field_path,
                "Invalid time format. Expected HH:MM:SS",
                value
            ))
        except (TypeError, AttributeError):
            errors.append(ValidationError(
                field_path,
                "Time must be a string",
                value
            ))
        return errors
    
    @staticmethod
    def validate_carb_entry(entry: dict, field_path: str = "") -> List[ValidationError]:
        """
        Validate a carb entry.
        
        Parameters
        ----------
        entry : dict
            Carb entry dictionary
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Required fields
        required_fields = ['start_time', 'value']
        for field in required_fields:
            if field not in entry:
                errors.append(ValidationError(
                    field_path,
                    f"Missing required field '{field}'"
                ))
        
        # Validate start_time format
        if 'start_time' in entry:
            errors.extend(ValueValidators.validate_datetime_format(
                entry['start_time'], 
                f"{field_path}.start_time"
            ))
        
        # Validate carb value
        if 'value' in entry:
            try:
                value = float(entry['value'])
                if not 0 < value <= 500:
                    errors.append(ValidationError(
                        f"{field_path}.value",
                        "Carb value must be between 0 and 500 grams",
                        value
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    f"{field_path}.value",
                    "Carb value must be numeric",
                    entry.get('value')
                ))
        
        # Validate optional duration
        if 'duration' in entry:
            try:
                duration = float(entry['duration'])
                if not 0 < duration <= 600:
                    errors.append(ValidationError(
                        f"{field_path}.duration",
                        "Carb absorption duration must be between 0 and 600 minutes",
                        duration
                    ))
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    f"{field_path}.duration",
                    "Carb duration must be numeric",
                    entry.get('duration')
                ))
        
        return errors
    
    # Valid non-numeric sentinel values for bolus entries.
    # These instruct the simulator to derive the bolus dose dynamically
    # rather than using a fixed numeric amount.
    VALID_BOLUS_SENTINELS = frozenset({"accept_recommendation"})

    @staticmethod
    def validate_bolus_entry(entry: dict, field_path: str = "") -> List[ValidationError]:
        """
        Validate a bolus entry.

        Bolus values may be either a numeric dose (0 < value <= 50 units) or a
        recognised sentinel string such as ``"accept_recommendation"`` which
        instructs the simulator to accept the Loop algorithm's bolus suggestion.
        
        Parameters
        ----------
        entry : dict
            Bolus entry dictionary
        field_path : str
            Path to this field in the configuration
            
        Returns
        -------
        List[ValidationError]
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Required fields
        required_fields = ['time', 'value']
        for field in required_fields:
            if field not in entry:
                errors.append(ValidationError(
                    field_path,
                    f"Missing required field '{field}'"
                ))
        
        # Validate time format
        if 'time' in entry:
            errors.extend(ValueValidators.validate_datetime_format(
                entry['time'], 
                f"{field_path}.time"
            ))
        
        # Validate bolus value: accept known sentinel strings or a numeric dose
        if 'value' in entry:
            raw = entry['value']
            if isinstance(raw, str) and raw in ValueValidators.VALID_BOLUS_SENTINELS:
                pass  # Valid sentinel — no further checks needed
            else:
                try:
                    value = float(raw)
                    if not 0 <= value <= 50:
                        errors.append(ValidationError(
                            f"{field_path}.value",
                            "Bolus value must be between 0 and 50 units",
                            value
                        ))
                except (ValueError, TypeError):
                    valid_sentinels = ", ".join(sorted(ValueValidators.VALID_BOLUS_SENTINELS))
                    errors.append(ValidationError(
                        f"{field_path}.value",
                        f"Bolus value must be numeric or one of: {valid_sentinels}",
                        raw
                    ))
        
        return errors
