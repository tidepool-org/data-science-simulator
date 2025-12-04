"""
Core simulation runner for insulin algorithm testing.

This module provides the execution layer for running pre-built Simulation objects
using the Tidepool simulator.

Responsibilities:
- Parallel execution of simulation batches
- Progress logging and timing estimation
- Extraction of results (glucose traces, insulin delivery)

Note: This module only handles execution. Simulation building is done externally
via the simulation_builder module.
"""

import logging
import os
import time
from typing import Dict, Any, Optional

import numexpr
import pandas as pd
import numpy as np
from numpy.random import RandomState

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.config.experiment_config import (
    ExperimentConfig
)
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.utils import format_duration

logger = logging.getLogger(__name__)


class SimulationRunner:
    """
    Main class for running insulin algorithm simulations.
    
    Integrates with the Tidepool simulator to run comparisons between
    temp basal and autobolus algorithms across different scenarios.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the simulation runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.sim_config = config.get_simulation_config()
        self.processing_config = config.get_processing_config()
        self.random_state = RandomState(config.random_seed)
        
        # Set up parallel processing environment
        num_parallel_processes = self.processing_config.parallel_processes
        os.environ['NUMEXPR_MAX_THREADS'] = str(num_parallel_processes)
        numexpr.set_num_threads(num_parallel_processes)
        logger.info(f"Set NUMEXPR_MAX_THREADS to {num_parallel_processes}")
        
        logger.info(f"Initialized SimulationRunner with config: {config}")
    
    def run_simulations(
        self,
        simulations: Dict[str, Simulation],
        save_dir: Optional[str] = None
    ) -> None:
        """
        Run pre-built simulations in parallel.
        
        This is the main entry point for running simulations.
        
        Args:
            simulations: Dictionary of simulation_id -> Simulation objects
            save_dir: Directory to save results (defaults to config.output_dir)
        """
        save_path = save_dir or self.config.output_dir
        
        logger.info(f"Running {len(simulations)} simulations...")
        start_time = time.time()
        
        _, _ = run_simulations(
            simulations,
            save_dir=save_path,
            save_results=self.processing_config.save_individual_results,
            compute_summary_metrics=False,
            num_procs=self.processing_config.parallel_processes
        )
        
        duration = time.time() - start_time
        logger.info(f"Completed {len(simulations)} simulations in {format_duration(duration)}")
    
    def run_parallel_batch_simulations(
        self,
        simulations: Dict[str, Simulation],
        save_dir: Optional[str] = None,
        total_scenarios: Optional[int] = None,
        total_start_time: Optional[float] = None,
        num_estimated_scenarios: Optional[int] = None,
        is_final_batch: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Run a batch of simulations in parallel.
        
        Args:
            simulations: Dictionary of simulation_id -> Simulation objects
            save_dir: Optional directory to save results
            total_scenarios: Total number of scenarios processed so far
            total_start_time: Start time of the entire batch operation
            is_final_batch: Whether this is the final batch
            
        Returns:
            Dictionary of simulation_id -> results DataFrame
        """
        batch_start_time = time.time()
        
        _, _ = run_simulations(
            simulations,
            save_dir=save_dir or self.config.output_dir,
            save_results=self.processing_config.save_individual_results,
            compute_summary_metrics=False,  # Handled separately
            num_procs=self.processing_config.parallel_processes
        )
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        
        # Log timing information if tracking parameters are provided
        if total_scenarios is not None and total_start_time is not None:
            total_elapsed = time.time() - total_start_time
            batch_type = "final batch" if is_final_batch else "batch"
            logger.info(f"Completed {batch_type} of {len(simulations)} simulations in {format_duration(batch_duration)} "
                       f"(total: {total_scenarios} scenarios, elapsed: {format_duration(total_elapsed)})")
        
    def extract_glucose_trace(
        self,
        results_df: pd.DataFrame,
        start_hours: float = 0,
        duration_hours: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract glucose trace from simulation results.
        
        Args:
            results_df: Simulation results DataFrame
            start_hours: Start time in hours from simulation start
            duration_hours: Duration to extract (None for all remaining)
            
        Returns:
            Glucose trace as numpy array
        """
        # Calculate indices
        start_idx = int(start_hours * 12)  # 5-minute intervals
        
        if duration_hours is not None:
            end_idx = start_idx + int(duration_hours * 12)
        else:
            end_idx = len(results_df)
        
        # Extract active data only
        active_data = results_df[results_df['active'] == 1]
        
        if len(active_data) == 0:
            logger.warning("No active data found in results")
            return np.array([])
        
        # Slice the data
        sliced_data = active_data.iloc[start_idx:end_idx]
        
        return sliced_data['bg'].values
    
    def extract_insulin_delivery(
        self,
        results_df: pd.DataFrame,
        start_hours: float = 0,
        duration_hours: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Extract insulin delivery data from simulation results.

        Args:
            results_df: Simulation results DataFrame
            start_hours: Start time in hours from simulation start
            duration_hours: Duration to extract (None for all remaining)

        Returns:
            Dictionary with insulin delivery metrics
        """
        # Calculate indices
        start_idx = int(start_hours * 12)  # 5-minute intervals
        
        if duration_hours is not None:
            end_idx = start_idx + int(duration_hours * 12)
        else:
            end_idx = len(results_df)
        
        # Extract active data only
        active_data = results_df[results_df['active'] == 1]
        
        if len(active_data) == 0:
            logger.warning("No active data found in results")
            return {'basal': 0.0, 'bolus': 0.0, 'total': 0.0}
        
        # Slice the data
        sliced_data = active_data.iloc[start_idx:end_idx]
        
        # Calculate insulin delivery
        basal_delivered = sliced_data['delivered_basal_insulin'].sum()
        bolus_delivered = sliced_data['true_bolus'].sum()
        total_delivered = basal_delivered + bolus_delivered
        
        return {
            'basal': basal_delivered,
            'bolus': bolus_delivered,
            'total': total_delivered
        }
