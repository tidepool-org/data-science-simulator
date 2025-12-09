__author__ = "Cameron Summers"

import pdb
import os
import datetime as dt
import time
import numpy as np

import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
DATA_DIR = os.path.join(os.path.expanduser("~"), "data/simulator/")


def get_bernoulli_trial_uniform_step_prob(num_trials, prob_of_occurring):
    """
    Given an event has a probability P of happening in a set number of trials,
    what is the trial bias B that the coin should have to yield P
    on average of events occurring?

    For meals:
    P(meal=False) = (1 - B) ^ (num_trials)
    P(meal=True) = 1 - P(meal=False) = 1 - (1 - B) ^ num_trials = prob_of_occuring

    1 - prob_of_occurring = (1 - B) ^ num_trials
    (1 - prob_of_occurring) ^ -num_trials = 1 - B
    B = 1 - (1 - prob_of_occurring) ^ -num_trials

    Parameters
    ----------
    num_trials: int
        How many trials are happening

    prob_of_occurring: float
        Probability of event

    Returns
    -------
    float
        Bias
    """

    return 1.0 - np.power(1 - prob_of_occurring, 1.0 / num_trials)


def findDiff(d1, d2, path=""):
    """
    Utility function for debugging that prints the difference between two nested dictionaries.

    Parameters
    ----------
    d1: dict
    d2
    path

    Returns
    -------

    """
    for k in d1:
        if k not in d2:
            print(path, ":")
            print(k + " as key not in d2", "\n")
        else:
            if type(d1[k]) is dict:
                if path == "":
                    path = k
                else:
                    path = path + "->" + k
                findDiff(d1[k], d2[k], path)
            else:
                if d1[k] != d2[k]:
                    print(path, ":")
                    print(" - ", k, " : ", d1[k])
                    print(" + ", k, " : ", d2[k])


def get_equivalent_isf(total_delta_bg, basal_rates=None):
    """
    For a given change in bg over some time and list of basal rates, return the complementary
     ISFs that achieve it.

    Parameters
    ----------
    total_delta_bg
    basal_rates

    Returns
    -------
    list

    """
    if basal_rates is None:
        basal_rates = np.arange(0.1, 1.0, 0.1)

    isfs = []
    for br in basal_rates:
        isfs.append(total_delta_bg / (br + 1.0))

    return isfs


def timing(f):
    """
    Util decorator for timing functions
    """
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        logger.debug(
            "{:s} function took {:.3f} ms".format(f.__name__, (time2 - time1) * 1000.0)
        )

        return ret

    return wrap


def save_df(df_results, analysis_name, save_dir, save_type="tsv"):
    utc_string = dt.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
    filename = "{}".format(analysis_name, utc_string)
    path = os.path.join(save_dir, filename)
    if "tsv" in save_type:
        df_results.to_csv("{}.tsv".format(path), sep="\t")
    else:
        df_results.to_csv("{}.csv".format(path))
    logger.debug("Saving sim to {}...".format(path))


class StreamingParquetWriter:
    """
    A streaming Parquet writer that appends DataFrames incrementally to avoid memory issues.
    
    Uses PyArrow's ParquetWriter to write row groups as simulations complete,
    allowing for arbitrarily large result sets without holding all data in memory.
    Metadata is stored in a separate parquet file that is also streamed incrementally,
    keeping data and metadata in the same format and allowing both to be updated during runtime.
    
    Output Files:
        - combined_results.parquet - Main simulation data with sim_id column
        - combined_results_metadata.parquet - Simulation metadata (sim_id, metadata_json)
    
    Example
    -------
    >>> writer = StreamingParquetWriter(save_dir)
    >>> for sim_id, df in simulation_results:
    ...     writer.write_batch(df, sim_id, sim_info)
    >>> writer.close()
    """
    
    def __init__(self, save_dir, filename="combined_results.parquet"):
        """
        Initialize the streaming writer.
        
        Parameters
        ----------
        save_dir: str
            Directory to save the parquet files
        filename: str
            Name of the main parquet file (default: combined_results.parquet)
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd
        
        self.save_dir = save_dir
        self.parquet_path = os.path.join(save_dir, filename)
        self.metadata_parquet_path = os.path.join(save_dir, filename.replace('.parquet', '_metadata.parquet'))
        self.data_writer = None
        self.metadata_writer = None
        self.data_schema = None
        self.metadata_schema = None
        self._pa = pa
        self._pq = pq
        self._pd = pd
        
        # Pre-define metadata schema for consistency
        self.metadata_schema = pa.schema([
            ('sim_id', pa.string()),
            ('metadata_json', pa.string())
        ])
    
    def write_batch(self, df, sim_id, sim_info=None):
        """
        Write a simulation result batch to both parquet files.
        
        Parameters
        ----------
        df: pd.DataFrame
            Simulation results DataFrame
        sim_id: str
            Simulation identifier (will be added as a column)
        sim_info: dict, optional
            Simulation metadata to store in metadata parquet file
        """
        import json
        
        # Add sim_id column to data
        df_with_id = df.copy()
        df_with_id['sim_id'] = sim_id
        
        # Convert data to PyArrow Table
        data_table = self._pa.Table.from_pandas(df_with_id)
        
        # Initialize data writer on first batch
        if self.data_writer is None:
            self.data_schema = data_table.schema
            self.data_writer = self._pq.ParquetWriter(
                self.parquet_path, 
                self.data_schema,
                compression='zstd'
            )
            logger.debug(f"Initialized streaming data parquet writer: {self.parquet_path}")
        
        # Write data batch as a row group
        self.data_writer.write_table(data_table)
        
        # Write metadata to separate parquet file
        if sim_info:
            metadata_df = self._pd.DataFrame([{
                'sim_id': sim_id,
                'metadata_json': json.dumps(sim_info)
            }])
            metadata_table = self._pa.Table.from_pandas(metadata_df, schema=self.metadata_schema)
            
            # Initialize metadata writer on first metadata write
            if self.metadata_writer is None:
                self.metadata_writer = self._pq.ParquetWriter(
                    self.metadata_parquet_path,
                    self.metadata_schema,
                    compression='zstd'
                )
                logger.debug(f"Initialized streaming metadata parquet writer: {self.metadata_parquet_path}")
            
            self.metadata_writer.write_table(metadata_table)
    
    def close(self):
        """
        Finalize both parquet files.
        """
        if self.data_writer is not None:
            self.data_writer.close()
            logger.debug(f"Closed streaming data parquet writer: {self.parquet_path}")
        
        if self.metadata_writer is not None:
            self.metadata_writer.close()
            logger.debug(f"Closed streaming metadata parquet writer: {self.metadata_parquet_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def load_streaming_parquet_with_metadata(parquet_path):
    """
    Load a streaming parquet file and its associated metadata parquet file.
    
    This is the companion function to StreamingParquetWriter for reading results.
    Both data and metadata are stored in parquet format for consistency.
    
    Parameters
    ----------
    parquet_path: str
        Path to the main Parquet file (combined_results.parquet)
        
    Returns
    -------
    tuple
        (df, metadata) where:
        - df: pd.DataFrame with all simulation results (has sim_id column)
        - metadata: dict with simulation info keyed by sim_id (or None if not present)
    
    Example
    -------
    >>> df, info = load_streaming_parquet_with_metadata('results/combined_results.parquet')
    >>> sim_001_info = info['sim_001']
    >>> print(sim_001_info['patient_id'])
    """
    import pyarrow.parquet as pq
    import json
    
    # Load main data parquet
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    # Load metadata parquet if it exists
    metadata_path = parquet_path.replace('.parquet', '_metadata.parquet')
    metadata = None
    if os.path.exists(metadata_path):
        metadata_table = pq.read_table(metadata_path)
        metadata_df = metadata_table.to_pandas()
        
        # Convert to dict keyed by sim_id
        metadata = {}
        for _, row in metadata_df.iterrows():
            sim_id = row['sim_id']
            metadata_json = row['metadata_json']
            metadata[sim_id] = json.loads(metadata_json)
    
    return df, metadata


def load_parquet_with_metadata(path):
    """
    Load a Parquet file and extract embedded simulation metadata.
    
    This is the companion function to save_df_parquet() for reading results
    that have embedded simulation info metadata.
    
    Parameters
    ----------
    path: str
        Path to the Parquet file
        
    Returns
    -------
    tuple
        (df, metadata) where:
        - df: pd.DataFrame with the simulation results
        - metadata: dict with simulation info (or None if not present)
    
    Example
    -------
    >>> df, info = load_parquet_with_metadata('results/sim_001.parquet')
    >>> print(info['sim_id'])
    'sim_001'
    """
    import pyarrow.parquet as pq
    import json
    
    table = pq.read_table(path)
    
    # Extract simulation info from metadata if present
    metadata = None
    if table.schema.metadata:
        sim_info_bytes = table.schema.metadata.get(b'simulation_info')
        if sim_info_bytes:
            metadata = json.loads(sim_info_bytes.decode('utf-8'))
    
    return table.to_pandas(), metadata


def get_sim_results_save_dir(description):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    utc_string = dt.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = "../data/results/simulations/{}/{}".format(description, utc_string)
    abs_path = os.path.join(this_dir, results_dir)
    os.makedirs(abs_path)
    return abs_path
