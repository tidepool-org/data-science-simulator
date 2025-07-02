"""
Data loader for insulin algorithm testing.

This module handles loading patient configurations and other data sources
for the insulin algorithm testing framework.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np

from tidepool_data_science_simulator.makedata.make_icgm_patients import transform_icgm_json_to_v2_parser
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.config.experiment_config import (
    ExperimentConfig, ScenarioConfig
)

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading of patient configurations and other data sources.
    
    Supports loading from:
    - Built-in iCGM patient configurations
    - CSV files with patient parameters
    - JSON files with patient configurations
    - Custom data sources
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the data loader.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.scenario_config = config.get_scenario_config()
        
        logger.info(f"Initialized DataLoader with config: {config}")
    
    def load_patient_configs(self, max_patients: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load patient configurations with optional limit.
        
        Args:
            max_patients: Maximum number of patients to load (optional)
            
        Returns:
            List of patient configuration dictionaries
        """
        # Temporarily override the num_patients setting if max_patients is provided
        original_num_patients = self.scenario_config.num_patients
        if max_patients is not None:
            self.scenario_config.num_patients = max_patients
        
        try:
            patient_configs = self.load_patient_configurations()
        finally:
            # Restore original setting
            self.scenario_config.num_patients = original_num_patients
        
        return patient_configs
    
    def load_patient_configurations(self) -> List[Dict[str, Any]]:
        """
        Load patient configurations based on configuration settings.
        
        Returns:
            List of patient configuration dictionaries
        """
        source = self.scenario_config.patient_source
        num_patients = self.scenario_config.num_patients
        
        logger.info(f"Loading patient configurations from: {source}")
        
        if source == "icgm_patients":
            patient_configs = self._load_icgm_patients()
        elif source.endswith('.csv'):
            patient_configs = self._load_from_csv(source)
        elif source.endswith('.json'):
            patient_configs = self._load_from_json(source)
        else:
            # Try to interpret as a directory path
            patient_configs = self._load_from_directory(source)
        
        # Limit number of patients if specified
        if num_patients is not None and len(patient_configs) > num_patients:
            patient_configs = patient_configs[:num_patients]
            logger.info(f"Limited to {num_patients} patients")
        
        logger.info(f"Loaded {len(patient_configs)} patient configurations")
        return patient_configs
    
    def _load_icgm_patients(self) -> List[Dict[str, Any]]:
        """Load built-in iCGM patient configurations."""
        
        try:
            # Use the existing function to load iCGM patients
            patient_configs = transform_icgm_json_to_v2_parser()
            logger.info(f"Loaded {len(patient_configs)} iCGM patient configurations")
            return patient_configs
        
        except Exception as e:
            logger.error(f"Error loading iCGM patients: {e}")
            raise
    
    def _load_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """
        Load patient configurations from CSV file.
        
        Expected CSV format:
        patient_id,isf,cir,basal_rate,target_min,target_max,...
        """
        
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            patient_configs = []
            
            for _, row in df.iterrows():
                config = self._create_patient_config_from_row(row)
                patient_configs.append(config)
            
            logger.info(f"Loaded {len(patient_configs)} patients from CSV: {csv_path}")
            return patient_configs
        
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}")
            raise
    
    def _load_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """Load patient configurations from JSON file."""
        
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                patient_configs = data
            elif isinstance(data, dict) and 'patients' in data:
                patient_configs = data['patients']
            else:
                # Assume single patient configuration
                patient_configs = [data]
            
            logger.info(f"Loaded {len(patient_configs)} patients from JSON: {json_path}")
            return patient_configs
        
        except Exception as e:
            logger.error(f"Error loading JSON file {json_path}: {e}")
            raise
    
    def _load_from_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Load patient configurations from directory of JSON files."""
        
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        patient_configs = []
        
        # Look for JSON files in the directory
        json_files = list(dir_path.glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in directory: {dir_path}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    config = json.load(f)
                patient_configs.append(config)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded {len(patient_configs)} patients from directory: {dir_path}")
        return patient_configs
    
    def _create_patient_config_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Create a patient configuration dictionary from a CSV row.
        
        This creates a minimal configuration that can be used with the simulator.
        """
        
        # Extract basic parameters
        patient_id = row.get('patient_id', f'patient_{row.name}')
        isf = row.get('isf', 50.0)  # mg/dL per unit
        cir = row.get('cir', 15.0)  # grams per unit
        basal_rate = row.get('basal_rate', 1.0)  # units per hour
        target_min = row.get('target_min', 100.0)  # mg/dL
        target_max = row.get('target_max', 120.0)  # mg/dL
        
        # Create a basic patient configuration
        # This is a simplified version - you may need to expand based on your CSV format
        config = {
            "patient_id": patient_id,
            "time_to_calculate_at": "8/15/2019 12:00:00",
            "patient": {
                "patient_model": {
                    "glucose_history": {
                        "value": {i: 100.0 for i in range(12)}  # 12 historical values
                    },
                    "metabolism_settings": {
                        "insulin_sensitivity_factor": {
                            "start_times": ["00:00:00"],
                            "values": [isf],
                            "duration_minutes": [1440]
                        },
                        "carb_insulin_ratio": {
                            "start_times": ["00:00:00"],
                            "values": [cir],
                            "duration_minutes": [1440]
                        },
                        "basal_rate": {
                            "start_times": ["00:00:00"],
                            "values": [basal_rate],
                            "duration_minutes": [1440]
                        }
                    }
                },
                "sensor": {
                    "glucose_history": {
                        "value": {i: 100.0 for i in range(12)}  # 12 historical values
                    }
                }
            },
            "controller": {
                "id": "swift",
                "settings": {
                    "target_range": {
                        "start_times": ["00:00:00"],
                        "end_times": ["23:59:59"],
                        "min_values": [target_min],
                        "max_values": [target_max]
                    }
                }
            }
        }
        
        # Add any additional parameters from the CSV
        for col in row.index:
            if col not in ['patient_id', 'isf', 'cir', 'basal_rate', 'target_min', 'target_max']:
                # Store additional parameters for potential use
                if 'additional_params' not in config:
                    config['additional_params'] = {}
                config['additional_params'][col] = row[col]
        
        return config
    
    def load_population_weights(self, weights_path: Optional[str] = None) -> Dict[float, float]:
        """
        Load population weights for initial blood glucose values.
        
        Args:
            weights_path: Path to CSV file with weights (optional)
            
        Returns:
            Dictionary mapping initial_bg -> weight
        """
        if weights_path is None:
            # Return uniform weights
            bg_values = range(
                self.scenario_config.initial_bg_range[0],
                self.scenario_config.initial_bg_range[1] + 1,
                self.scenario_config.initial_bg_step
            )
            return {float(bg): 1.0 for bg in bg_values}
        
        weights_path = Path(weights_path)
        if not weights_path.exists():
            logger.warning(f"Weights file not found: {weights_path}. Using uniform weights.")
            return self.load_population_weights(None)
        
        try:
            df = pd.read_csv(weights_path)
            
            # Expected columns: 'ibg' (initial blood glucose) and 'proportion' or 'weight'
            weight_col = 'proportion' if 'proportion' in df.columns else 'weight'
            
            if 'ibg' not in df.columns or weight_col not in df.columns:
                raise ValueError(f"Expected columns 'ibg' and '{weight_col}' in weights file")
            
            weights = dict(zip(df['ibg'], df[weight_col]))
            logger.info(f"Loaded population weights for {len(weights)} BG values")
            
            return weights
        
        except Exception as e:
            logger.error(f"Error loading weights file {weights_path}: {e}")
            return self.load_population_weights(None)
    
    def validate_patient_configurations(self, patient_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate patient configurations and filter out invalid ones.
        
        Args:
            patient_configs: List of patient configuration dictionaries
            
        Returns:
            List of validated patient configurations
        """
        valid_configs = []
        
        for i, config in enumerate(patient_configs):
            try:
                # Basic validation
                if not self._validate_patient_config(config):
                    logger.warning(f"Invalid patient configuration at index {i}")
                    continue
                
                valid_configs.append(config)
            
            except Exception as e:
                logger.warning(f"Error validating patient config at index {i}: {e}")
        
        logger.info(f"Validated {len(valid_configs)} out of {len(patient_configs)} patient configurations")
        return valid_configs
    
    def _validate_patient_config(self, config: Dict[str, Any]) -> bool:
        """Validate a single patient configuration."""
        
        required_keys = ['patient', 'controller']
        
        for key in required_keys:
            if key not in config:
                logger.warning(f"Missing required key: {key}")
                return False
        
        # Validate patient model
        patient = config['patient']
        if 'patient_model' not in patient:
            logger.warning("Missing patient_model in patient configuration")
            return False
        
        patient_model = patient['patient_model']
        if 'metabolism_settings' not in patient_model:
            logger.warning("Missing metabolism_settings in patient model")
            return False
        
        # Validate required metabolism settings
        metabolism = patient_model['metabolism_settings']
        required_metabolism = ['insulin_sensitivity_factor', 'carb_insulin_ratio', 'basal_rate']
        
        for setting in required_metabolism:
            if setting not in metabolism:
                logger.warning(f"Missing metabolism setting: {setting}")
                return False
        
        return True
    
    def get_patient_summary(self, patient_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics for loaded patient configurations.
        
        Args:
            patient_configs: List of patient configuration dictionaries
            
        Returns:
            Dictionary with summary statistics
        """
        if not patient_configs:
            return {'num_patients': 0}
        
        # Extract parameter values
        isf_values = []
        cir_values = []
        basal_values = []
        
        for config in patient_configs:
            try:
                metabolism = config['patient']['patient_model']['metabolism_settings']
                
                # Get first value from each setting (assuming single value for simplicity)
                isf = metabolism['insulin_sensitivity_factor']['values'][0]
                cir = metabolism['carb_insulin_ratio']['values'][0]
                basal = metabolism['basal_rate']['values'][0]
                
                isf_values.append(isf)
                cir_values.append(cir)
                basal_values.append(basal)
            
            except (KeyError, IndexError) as e:
                logger.warning(f"Could not extract parameters from patient config: {e}")
        
        summary = {
            'num_patients': len(patient_configs),
            'isf_stats': {
                'mean': np.mean(isf_values) if isf_values else 0,
                'std': np.std(isf_values) if isf_values else 0,
                'min': np.min(isf_values) if isf_values else 0,
                'max': np.max(isf_values) if isf_values else 0
            },
            'cir_stats': {
                'mean': np.mean(cir_values) if cir_values else 0,
                'std': np.std(cir_values) if cir_values else 0,
                'min': np.min(cir_values) if cir_values else 0,
                'max': np.max(cir_values) if cir_values else 0
            },
            'basal_stats': {
                'mean': np.mean(basal_values) if basal_values else 0,
                'std': np.std(basal_values) if basal_values else 0,
                'min': np.min(basal_values) if basal_values else 0,
                'max': np.max(basal_values) if basal_values else 0
            }
        }
        
        return summary
    
    def save_patient_configurations(
        self,
        patient_configs: List[Dict[str, Any]],
        output_path: str,
        format: str = 'json'
    ) -> None:
        """
        Save patient configurations to file.
        
        Args:
            patient_configs: List of patient configuration dictionaries
            output_path: Output file path
            format: Output format ('json' or 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(patient_configs, f, indent=2)
        
        elif format == 'csv':
            # Extract key parameters for CSV
            data = []
            for config in patient_configs:
                try:
                    metabolism = config['patient']['patient_model']['metabolism_settings']
                    
                    row = {
                        'patient_id': config.get('patient_id', 'unknown'),
                        'isf': metabolism['insulin_sensitivity_factor']['values'][0],
                        'cir': metabolism['carb_insulin_ratio']['values'][0],
                        'basal_rate': metabolism['basal_rate']['values'][0]
                    }
                    
                    # Add target range if available
                    try:
                        target_range = config['controller']['settings']['target_range']
                        row['target_min'] = target_range['min_values'][0]
                        row['target_max'] = target_range['max_values'][0]
                    except KeyError:
                        pass
                    
                    data.append(row)
                
                except (KeyError, IndexError) as e:
                    logger.warning(f"Could not extract data for CSV: {e}")
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(patient_configs)} patient configurations to {output_path}")
