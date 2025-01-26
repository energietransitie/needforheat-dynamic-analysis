from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Set, Callable
from enum import Enum
import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline
import math
from gekko import GEKKO
from tqdm.notebook import tqdm
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue, Process
import time

import numbers
import logging

from pythermalcomfort.models import pmv_ppd

from nfh_utils import *



class BoilerEfficiency:
    def __init__(self, file_path):
        """
        Initializes the BoilerEfficiency class with a path to the boiler efficiency data file.

        Parameters:
            file_path (str): Path to the Parquet file containing boiler efficiency data.
        """
        self.file_path = file_path
        self.df_boiler_efficiency = None
        self.efficiency_hhv_interpolators = {}

    def load_boiler_hhv_efficiency(self):
        """
        Lazy loading of the boiler HHV efficiency data from a Parquet file.

        Returns:
            DataFrame: Loaded DataFrame containing boiler HHV efficiency data.
        """
        if self.df_boiler_efficiency is None:
            try:
                self.df_boiler_efficiency = pd.read_parquet(
                    self.file_path,
                    engine='pyarrow',
                    dtype_backend='numpy_nullable'
                )
            except Exception as e:
                raise IOError(f"Error reading Parquet file: {e}")
        return self.df_boiler_efficiency

    def get_efficiency_hhv_interpolator(self, brand_model):
        """
        Retrieves or creates an efficiency HHV interpolator for the specified brand_model,
        with strict range enforcement based on valid ranges in the dataset.

        Parameters:
            brand_model (str): The specific boiler brand_model.

        Returns:
            function: A callable function for interpolating efficiency HHV with strict range checks.
        """
        df = self.load_boiler_hhv_efficiency()

        # Check if interpolator for this brand_model already exists
        if brand_model not in self.efficiency_hhv_interpolators:
            group = df.loc[brand_model]
            load_points = group.index.get_level_values('rounded_load__pct').unique()
            temp_points = group.index.get_level_values('rounded_temp_ret__degC').unique()
            efficiency_values = group.unstack(level='rounded_temp_ret__degC').values

            # Create interpolator
            interpolator = RectBivariateSpline(load_points, temp_points, efficiency_values)

            # Determine valid ranges
            min_load__pct = load_points.min()
            max_load__pct = load_points.max()
            min_temp_ret__degC = temp_points.min()
            max_temp_ret__degC = temp_points.max()

            # Store interpolator and ranges
            self.efficiency_hhv_interpolators[brand_model] = {
                "interpolator": interpolator,
                "min_load__pct": min_load__pct,
                "max_load__pct": max_load__pct,
                "min_temp_ret__degC": min_temp_ret__degC,
                "max_temp_ret__degC": max_temp_ret__degC,
            }

        # Retrieve stored data
        data = self.efficiency_hhv_interpolators[brand_model]
        interpolator = data["interpolator"]
        min_load__pct = data["min_load__pct"]
        max_load__pct = data["max_load__pct"]
        min_temp_ret__degC = data["min_temp_ret__degC"]
        max_temp_ret__degC = data["max_temp_ret__degC"]

        # Define strict interpolator
        def boiler_efficiency_hhv(
            load__pct=None,
            temp_ret__degC=None,
        ):
            """
            Calculate the boiler efficiency on a higher heating value (HHV) basis 
            using an interpolator. 
            
            If `load__pct` or `temp_ret__degC` is None, the function returns NaN. 
            If `load__pct` or `temp_ret__degC` falls outside the defined min and max ranges, 
            the values are clipped to the nearest bound, and a result is still calculated.
        
            Parameters:
            - load__pct (float): Boiler load as a percentage. Expected range: [min_load__pct, max_load__pct].
            - temp_ret__degC (float): Return temperature in degrees Celsius. 
                                      Expected range: [min_temp_ret__degC, max_temp_ret__degC].
        
            Returns:
            - float: The interpolated boiler efficiency (HHV basis) or NaN if inputs are invalid.
            """
            if load__pct is None or temp_ret__degC is None:
                return np.nan
        
            # Clip values to their respective ranges
            load__pct_clipped = np.clip(load__pct, min_load__pct, max_load__pct)
            temp_ret__degC_clipped = np.clip(temp_ret__degC, min_temp_ret__degC, max_temp_ret__degC)
        
            # Perform interpolation with clipped values
            return interpolator(load__pct_clipped, temp_ret__degC_clipped, grid=False).item()

        return boiler_efficiency_hhv


class LearnError(Exception):
    def __init__(self, message):
        self.message = message
        
class Learner():

    
    def get_longest_sane_streak(df_data:pd.DataFrame,
                                id,
                                learn_period_start,
                                learn_period_end,
                                duration_threshold:timedelta=timedelta(hours=24)) -> pd.DataFrame:
        
        df_learn_period  = df_data.loc[(df_data.index.get_level_values('id') == id) & 
                                       (df_data.index.get_level_values('timestamp') >= learn_period_start) & 
                                       (df_data.index.get_level_values('timestamp') < learn_period_end)]
        
        learn_period_len = len(df_learn_period)

        # Check for enough values
        if learn_period_len <=1:
            logging.info(f'No values for id: {id} between {learn_period_start} and {learn_period_end}; skipping...')
            return None

        # Check for at least two sane values
        if (df_learn_period['sanity'].sum()) <=1: #counts the number of sane rows, since True values will be counted as 1 in suming
            logging.info(f'Less than two sane values for id: {id} between {learn_period_start} and {learn_period_start}; skipping...')
            return None
            
        actual_start = df_learn_period.index.min()
        actual_end = df_learn_period.index.max()
        logging.info(f'id: {id}; actual_start: {actual_start}')
        logging.info(f'id: {id}; actual_end: {actual_end}')

        logging.info(f'before longest streak analysis')
        logging.info(f'#rows in learning period before longest streak analysis: {len(df_learn_period)}')

        # restrict the dataframe to the longest streak of sane data
        ## give each streak a separate id
        df_data.loc[(id,learn_period_start):(id,learn_period_end), 'streak_id'] =  np.asarray(
            df_data.loc[(id,learn_period_start):(id,learn_period_end)].sanity
            .ne(df_data.loc[(id,learn_period_start):(id,learn_period_end)].sanity.shift())
            .cumsum()
        )

        # Calculate timedelta for each interval (suitable for unevenly spaced measurementes)
        df_data.loc[id, 'interval__s'] = (df_data.loc[id]
                                                .index.to_series()
                                                .diff()
                                                .shift(-1)
                                                .apply(lambda x: x.total_seconds())
                                                .fillna(0)
                                                .astype(int)
                                                .to_numpy()
                                                )

        df_data.loc[(id,learn_period_start):(id,learn_period_end), 'streak_cumulative_duration__s'] = np.asarray(
            df_data.loc[(id,learn_period_start):(id,learn_period_end)]
            .groupby('streak_id')
            .interval__s
            .cumsum()
        )

        # Ensure that insane streaks are not selected as longest streak
        df_data.loc[(df_data.index.get_level_values('id') == id) & (df_data['sanity'] == False), 'streak_cumulative_duration__s'] = np.nan

        # Get the longest streak: find the streak_id with the longest cumulative duration
        longest_streak_idxmax = df_data.loc[(id,learn_period_start):(id,learn_period_end)].streak_cumulative_duration__s.idxmax()
        logging.info(f'longest_streak_idxmax: {longest_streak_idxmax}') 
        
        longest_streak_query = 'streak_id == ' + str(df_data.loc[longest_streak_idxmax].streak_id)
        logging.info(f'longest_streak_query: {longest_streak_query}') 
        
        # Filter to the longest streak
        df_longest_streak  = df_data.loc[(id,learn_period_start):(id,learn_period_end)].query(longest_streak_query)
        timestamps = df_longest_streak.index.get_level_values('timestamp')

        # Log details about the filtered data
        if learn_period_len != len(df_longest_streak):
            logging.info(f'id: {id}; {learn_period_len} rows between {learn_period_start} and {learn_period_end}')
            logging.info(f'id: {id}; {len(df_longest_streak)} rows between {timestamps.min()} and {timestamps.max()} in the longest streak')
        
        # Check if streak duration is long enough
        if ((timestamps.max() - timestamps.min()) < duration_threshold):
            logging.info(f'Longest streak duration to short for id: {id}; shorter than {duration_threshold} between {timestamps.min()} and {timestamps.max()}; skipping...')
            return None

        return df_longest_streak

    
    def periodic_learn_list(
            df_data: pd.DataFrame,
            req_props: Set[str],
            property_sources: Dict,
            duration_threshold: timedelta = timedelta(hours=24),
            learn_period__d: int = None,
            max_len: int = None,
            ) -> pd.DataFrame:
        """
        Create a DataFrame with a list of jobs with MultiIndex (id, start, end) to be processed.

        Parameters:
            df_data (pd.DataFrame): Input data with a MultiIndex ('id', 'timestamp').
            req_props (set): Set of required properties to check for non-NaN/NA values.
            property_sources: Dictionary with mapping of properties to column names in df_data.
            duration_threshold (timedelta): Minimum duration for a streak to be included.
            learn_period__d (int): period length (in days) to use; hence also the maximum job length.
            max_len: length of list (to reduce calculation time during tests); if needed a random subsample will be taken
    
        Returns:
            pd.DataFrame: A DataFrame containing job list with MultiIndex ['id', 'start', 'end'].
        """
        jobs = []
        ids = df_data.index.unique('id').dropna()
        start_analysis_period = df_data.index.unique('timestamp').min().to_pydatetime()
        end_analysis_period = df_data.index.unique('timestamp').max().to_pydatetime()
        daterange_frequency = str(learn_period__d) + 'D'
        learn_period_starts = pd.date_range(start=start_analysis_period, end=end_analysis_period, inclusive='both', freq=daterange_frequency)
    
        # Perform sanity check
        req_cols = (
            set(property_sources.values())
            if req_props is None
            else {property_sources[prop] for prop in req_props & property_sources.keys()}
        )
        if req_cols:
            df_data['sanity'] = ~df_data[list(req_cols)].isna().any(axis="columns")
        else:
            df_data['sanity'] = True  # No required columns, mark all as sane
        
        for id in tqdm(ids, desc=f"Identifying {f'at most {max_len} ' if max_len is not None and max_len > 0 else ''} periodic sane streaks of at least of at least {duration_threshold} and at  most {learn_period__d} days"):
            for learn_period_start in learn_period_starts:
                learn_period_end = min(learn_period_start + timedelta(days=learn_period__d), end_analysis_period)
    
                # Learn only for the longest streak of sane data
                #TODO: remove df_learn slicing, directly deliver start & end time per period
                df_learn = Learner.get_longest_sane_streak(df_data, id, learn_period_start, learn_period_end, duration_threshold)
                if df_learn is None:
                    continue
                    
                start_time = df_learn.index.unique('timestamp').min().to_pydatetime()
                end_time = df_learn.index.unique('timestamp').max().to_pydatetime()
                streak_duration = end_time - start_time

                jobs.append((id, start_time, end_time, streak_duration))
    
        df_data.drop(columns=['sanity'], inplace=True)
    
        # Convert jobs to DataFrame
        df_jobs = pd.DataFrame(jobs, columns=['id', 'start', 'end', 'duration'])

        # if max_len is not None take a random subsample to limit calculation time
        if max_len is not None and max_len < len(df_jobs):
            df_jobs = df_jobs.sample(n=max_len, random_state=42)
        
        return df_jobs.set_index(['id', 'start', 'end', 'duration'])

    
    def valid_learn_list(
        df_data: pd.DataFrame,
        req_props: Set[str],
        property_sources: Dict,
        duration_threshold: timedelta = timedelta(minutes=30),
        learn_period__d: int = None,
        max_len: int = None,
        ) -> pd.DataFrame:
        """
        Create a list of jobs (id, start, end) based on consecutive streaks of data
        with non-NaN values in required columns and duration above a threshold.
    
        Parameters:
            df_data (pd.DataFrame): Input data with a MultiIndex ('id', 'timestamp').
            req_props (set): Set of required properties to check for non-NaN/NA values.
            property_sources: Dictionary with mapping of properties to column names in df_data.
            duration_threshold (timedelta): Minimum duration for a streak to be included.
            learn_period__d (int): not supported, but required to be passed as a Callable
            max_len: length of list (to reduce calculation time during tests); if needed a random subsample will be taken
    
        Returns:
            pd.DataFrame: A DataFrame containing job list with MultiIndex ['id', 'start', 'end'].
        """
        # Ensure required columns are specified
        req_cols = (
            set(property_sources.values())
            if req_props is None
            else {property_sources[prop] for prop in req_props & property_sources.keys()}
        )    

        # Initialize job list
        jobs = []
        
        # Iterate over each unique id
        for id_, group in tqdm(df_data.groupby(level='id'), desc=f"Identifying {f'at most {max_len} ' if max_len is not None and max_len > 0 else ''} valid streaks of at least {duration_threshold}"):            # Filter for rows where all required columns are not NaN
            group = group.droplevel('id')  # Drop 'id' level for easier handling
            group = group.loc[group[list(req_cols)].notna().all(axis=1)]
    
            if group.empty:
                continue
    
            # Compute time differences to detect breaks in streaks
            time_diff = group.index.to_series().diff()
    
            # Mark new streaks where the time gap is larger than a threshold
            streak_ids = (time_diff > timedelta(minutes=1)).cumsum()
    
            # Group by streaks to find start and end timestamps
            for streak_id, streak_group in group.groupby(streak_ids):
                start_time = streak_group.index.min()
                end_time = streak_group.index.max()
                streak_duration = end_time - start_time
    
                # Only include streaks that meet the duration threshold
                if streak_duration >= duration_threshold:
                    jobs.append((id_, start_time, end_time, streak_duration))
    
        # Convert jobs to DataFrame
        df_jobs = pd.DataFrame(jobs, columns=['id', 'start', 'end', 'duration'])

        # if max_len is not None take a random subsample to limit calculation time
        if max_len is not None and max_len < len(df_jobs):
            df_jobs = df_jobs.sample(n=max_len, random_state=42)

        return df_jobs.set_index(['id', 'start', 'end', 'duration'])
    

    def get_time_info(df_learn):
        """
        Extracts time-related information from a DataFrame with a MultiIndex (id, timestamp).
        
        Parameters:
        df_learn (pandas.DataFrame): DataFrame with a MultiIndex ('id', 'timestamp').
        
        Returns:
        tuple: A tuple containing:
            - id (str or int): The unique identifier from the 'id' level of the index.
            - start (datetime): The earliest timestamp in the DataFrame.
            - end (datetime): The latest timestamp in the DataFrame.
            - step__s (float): The time interval between timestamps in seconds.
            - duration__s (float): The total duration in seconds (step * number of rows).
            
        Raises:
        ValueError: If 'df_learn' contains multiple IDs or inconsistent time intervals.
        """
        
        # Extract unique ID(s)
        ids = df_learn.index.get_level_values('id').unique()
        if len(ids) != 1:
            raise ValueError("df_learn contains multiple IDs.")
        
        id = ids[0]
        start = df_learn.index.get_level_values('timestamp').min()
        end = df_learn.index.get_level_values('timestamp').max()
        
        # Extract timestamps and check if they are sorted
        timestamps = df_learn.index.get_level_values('timestamp')
        
        if not timestamps.is_monotonic_increasing:
            df_learn = df_learn.sort_index(level='timestamp')
            logging.info("Timestamps were not sorted. Sorting performed.")
        else:
            logging.info("Timestamps are already sorted.")
        
        # Calculate time differences between consecutive timestamps
        time_diffs = df_learn.index.get_level_values('timestamp').diff().dropna()
        
        # Check for consistent time intervals
        if time_diffs.nunique() != 1:
            raise ValueError("The intervals between timestamps are not consistent.")
        
        # Calculate step (interval in seconds) and total duration in seconds
        step__s = time_diffs[0].total_seconds()

        duration__s = step__s * len(df_learn)
        
        return id, start, end, step__s, duration__s



    # Function to create a new results directory
    def create_results_directory(base_dir='results'):
        timestamp = datetime.now().isoformat()
        results_dir = os.path.join(base_dir, f'results-{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

    
    def get_actual_parameter_values(id, aperture_inf_avg__cm2, heat_tr_dstr_avg__W_K_1, th_mass_dstr_avg__Wh_K_1):
        """
        Calculate actual thermal parameter values based on the given 'id' and return them in a dictionary.
    
        Args:
            id: The unique identifier for which to calculate the actual values.
            aperture_inf_avg__cm2: Average aperture for infiltration in cm².
            heat_tr_dstr_avg__W_K_1: Average heat transfer capacity of the distribution system in W/K.
            th_mass_dstr_avg__Wh_K_1: Average thermal mass of the distribution system in Wh/K.
    
        Returns:
            dict: A dictionary containing the actual parameter values.
        """
        actual_params = {
            'heat_tr_bldng_cond__W_K_1': np.nan,
            'th_inert_bldng__h': np.nan,
            'aperture_sol__m2': np.nan,
            'th_mass_bldng__Wh_K_1': np.nan,
            'aperture_inf__cm2': aperture_inf_avg__cm2,  # Use average value as provided
            'heat_tr_dstr__W_K_1': heat_tr_dstr_avg__W_K_1,  # Use average value
            'th_mass_dstr__Wh_K_1': th_mass_dstr_avg__Wh_K_1  # Use average value
        }
    
        # Calculate specific actual values based on the 'id'
        if id is not None:
            actual_params['heat_tr_bldng_cond__W_K_1'] = id // 1e5
            actual_params['th_inert_bldng__h'] = (id % 1e5) // 1e2
            actual_params['aperture_sol__m2'] = id % 1e2
            actual_params['th_mass_bldng__Wh_K_1'] = (
                actual_params['heat_tr_bldng_cond__W_K_1'] *
                actual_params['th_inert_bldng__h']
            )
    
        return actual_params
 

    @staticmethod
    def learn_system_parameters(
        df_data: pd.DataFrame,
        df_bldng_data: pd.DataFrame,
        system_model_fn: Callable,
        job_identification_fn: Callable,
        property_sources: Dict = None,
        param_hints: Dict = None,
        learn_params: Dict = None,
        actual_params: Set[str] = None,
        req_props: Set[str] = None,
        helper_props: Set[str] = None,
        predict_props: Set[str] = None,
        duration_threshold: timedelta = None,
        learn_period__d: int = None,
        max_periods: int = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generalized function to learn system parameters.

        Parameters:
            df_data (pd.DataFrame): Main data.
            df_bldng_data (pd.DataFrame): Building-specific data.
            system_model_fn (Callable): Function to call for learning system parameters.
            job_identification_fn (Callable): Function to identify learning jobs.
            property_sources (Dict): Mapping of properties to column names.
            param_hints (Dict): Parameter hints for the learning function.
            learn_params (Dict): Learning-specific parameters.
            actual_params (Set[str]): Set of actual parameters to consider.
            req_props (Set[str]): Required properties for learning.
            helper_props (Set[str]): Properties not required all the time, but should be passes on to the model.
            predict_props (Set[str]): Properties to predict.
            duration_threshold (timedelta): Minimum duration for jobs.
            learn_period__d (int): Maximum duration of data for jobs (in days)
            max_periods (int): Maximum periods for learning jobs.
            **kwargs: Additional arguments passed to the system_model_fn.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Learned parameters and predicted properties.
        """
        # Determine required columns
        if req_props is None:
            req_cols = set(property_sources.values())
        else:
            req_cols = {property_sources[prop] for prop in req_props & property_sources.keys()}

        if helper_props is None:
            helper_cols = set()
        else:
            helper_cols = {property_sources[prop] for prop in helper_props & property_sources.keys()}

        # Focus on the required columns + columns for which an average needs to be calculated
        df_learn_all = df_data[list(req_cols | helper_cols)]

        # Identify analysis jobs using the provided function
        df_analysis_jobs = job_identification_fn(
            df_data,
            req_props=req_props,
            property_sources=property_sources,
            duration_threshold=duration_threshold,
            learn_period__d=learn_period__d,
            max_len=max_periods,
        )

        # Initialize result lists
        aggregated_predicted_job_properties = []
        aggregated_learned_job_parameters = []

        num_jobs = df_analysis_jobs.shape[0]
        num_workers = min(num_jobs, os.cpu_count())

        max_iter = kwargs.get('max_iter')
        max_iter_desc = f"with at most {max_iter} iterations " if max_iter is not None else "without iteration limit "
        
        mode = kwargs.get('mode')
        mode_desc = f"using mode {mode.value} " if mode is not None else ""

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            learned_jobs = {}

            for id, start, end, duration in tqdm(
                df_analysis_jobs.index,
                desc=f"Submitting {system_model_fn.__name__} jobs {mode_desc}{max_iter_desc}to {num_workers} processes",
            ):
                # Create df_learn for the current job
                df_learn = df_learn_all.loc[
                    (df_learn_all.index.get_level_values("id") == id)
                    & (df_learn_all.index.get_level_values("timestamp") >= start)
                    & (df_learn_all.index.get_level_values("timestamp") < end)
                ]

                # Get building-specific data for each job
                bldng_data = df_bldng_data.loc[id].to_dict()

                # Submit the system model function to the executor
                future = executor.submit(
                    system_model_fn,
                    df_learn,
                    bldng_data=bldng_data,
                    property_sources=property_sources,
                    param_hints=param_hints,
                    learn_params=learn_params,
                    actual_params=actual_params,
                    predict_props=predict_props,
                    **kwargs,
                )

                futures.append(future)
                learned_jobs[(id, start, end, duration)] = future

            # Collect results as they complete
            with tqdm(total=len(futures), desc=f"Collecting results from {num_workers} processes") as pbar:
                for future in as_completed(futures):
                    try:
                        df_learned_parameters, df_predicted_properties = future.result()
                        aggregated_predicted_job_properties.append(df_predicted_properties)
                        aggregated_learned_job_parameters.append(df_learned_parameters)
                    except Exception as e:
                        if "Solution Not Found" in str(e):
                            for (id, start, end, duration), job_future in learned_jobs.items():
                                if job_future == future:
                                    logging.warning(f"Solution Not Found for job (id: {id}, start: {start}, end: {end}, duration: {duration}). Skipping.")
                                    break
                            continue
                        elif "Maximum solver iterations" in str(e):
                            print("")
                            for (id, start, end, duration), job_future in learned_jobs.items():
                                if job_future == future:
                                    logging.warning(f"Solution exceeded maximum iterations for job (id: {id}, start: {start}, end: {end}, duration: {duration}). Skipping.")
                                    break
                            continue
                        else:
                            raise
                    finally:
                        pbar.update(1)

        if aggregated_predicted_job_properties:
            # Combine results into a cumulative DataFrame for predicted properties
            df_predicted_properties = (
                pd.concat(aggregated_predicted_job_properties, axis=0)
                .drop_duplicates()
                .sort_index()
            )
        else:
            df_predicted_properties = pd.DataFrame()
        
        if aggregated_learned_job_parameters:
            # Combine results into a cumulative DataFrame for learned parameters
            df_learned_parameters = (
                pd.concat(aggregated_learned_job_parameters, axis=0)
                .drop_duplicates()
                .sort_index()
            )
        else:
            df_learned_parameters = pd.DataFrame()

        return df_learned_parameters, df_predicted_properties
        

class Model():
    
    def ventilation(
        df_learn: pd.DataFrame,
        bldng_data: Dict = None,
        property_sources: Dict = None,
        param_hints: Dict = None,
        learn_params: Set[str] = None,
        actual_params: Dict = None,
        predict_props: Set[str] = {'ventilation__dm3_s_1'},
        learn_change_interval: pd.Timedelta = pd.Timedelta(minutes=30)
     ) -> Tuple[pd.DataFrame, pd.DataFrame]:


        id, start, end, step__s, duration__s  = Learner.get_time_info(df_learn) 
        duration = timedelta(seconds=duration__s)
        
        bldng__m3 = bldng_data['bldng__m3']
        usable_area__m2 = bldng_data['usable_area__m2']
        
        logging.info(f"learn ventilation rate for id {df_learn.index.get_level_values('id')[0]}, from  {df_learn.index.get_level_values('timestamp').min()} to {df_learn.index.get_level_values('timestamp').max()}")

        ##################################################################################################################
        # GEKKO Model - Initialize
        ##################################################################################################################
        m = GEKKO(remote=False)
        m.time = np.arange(0, duration__s, step__s)

        ##################################################################################################################
        ## Use measured CO₂ concentration indoors
        ##################################################################################################################
        co2_indoor__ppm = m.CV(value=df_learn[property_sources['co2_indoor__ppm']].values, name='co2_indoor__ppm')
        co2_indoor__ppm.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
        co2_indoor__ppm.FSTATUS = 1 # Use the measured values
        
        ##################################################################################################################
        ## CO₂ concentration gain indoors
        ##################################################################################################################

        # Use measured occupancy
        occupancy__p = m.MV(value = df_learn[property_sources['occupancy__p']].astype('float32').values, name='occupancy__p')
        occupancy__p.STATUS = 0  # No optimization
        occupancy__p.FSTATUS = 1 # Use the measured values

        co2_indoor_gain__ppm_s_1 = m.Intermediate(occupancy__p * co2_exhale_sedentary__umol_p_1_s_1 / 
                                                  (bldng__m3 * gas_room__mol_m_3),
                                                  name='co2_indoor_gain__ppm_s_1')
        
        ##################################################################################################################
        ## CO₂ concentration loss indoors
        ##################################################################################################################

        # Ventilation-induced CO₂ concentration loss indoors
        ventilation__dm3_s_1 = m.MV(value=param_hints['ventilation_default__dm3_s_1'],
                                    lb=0.0, 
                                    ub=param_hints['ventilation_max__dm3_s_1_m_2'] * usable_area__m2,
                                    name='ventilation__dm3_s_1'
                                   )
        ventilation__dm3_s_1.STATUS = 1  # Allow optimization
        ventilation__dm3_s_1.FSTATUS = 1 # Use the measured values
        
        if learn_change_interval is not None:
            update_interval_steps = int(np.ceil(learn_change_interval.total_seconds() / step__s))
            ventilation__dm3_s_1.MV_STEP_HOR = update_interval_steps
        
        # Wind-induced (infiltration) CO₂ concentration loss indoors
        wind__m_s_1 = m.MV(value=df_learn[property_sources['wind__m_s_1']].astype('float32').values, name='wind__m_s_1')
        wind__m_s_1.STATUS = 0  # No optimization
        wind__m_s_1.FSTATUS = 1 # Use the measured values
    
        if 'aperture_inf_vent__cm2' in learn_params:
            aperture_inf_vent__cm2 = m.FV(value=param_hints['aperture_inf__cm2'], lb=0, ub=100000.0, name='aperture_inf_vent__cm2')
            aperture_inf_vent__cm2.STATUS = 1  # Allow optimization
            aperture_inf_vent__cm2.FSTATUS = 1 # Use the initial value as a hint for the solver
        else:
            aperture_inf_vent__cm2 = m.Param(value=param_hints['aperture_inf__cm2'], name='aperture_inf_vent__cm2')

        air_changes_vent__s_1 = m.Intermediate(ventilation__dm3_s_1 / (bldng__m3 * dm3_m_3), name='air_changes_vent__s_1')
        
        air_inf__m3_s_1 = m.Intermediate(wind__m_s_1 * aperture_inf_vent__cm2 / cm2_m_2, name='air_inf__m3_s_1')        
        air_changes_inf__s_1 = m.Intermediate(air_inf__m3_s_1 / bldng__m3, name='air_changes_inf__s_1')

        # Total losses of CO₂ concentration indoors
        air_changes_total__s_1 = m.Intermediate(air_changes_vent__s_1 + air_changes_inf__s_1, name='air_changes_total__s_1')
        co2_elevation__ppm = m.Intermediate(co2_indoor__ppm - param_hints['co2_outdoor__ppm'], name='co2_elevation__ppm')
        co2_indoor_loss__ppm_s_1 = m.Intermediate(air_changes_total__s_1 * co2_elevation__ppm, name='co2_indoor_loss__ppm_s_1')
        
        ##################################################################################################################
        # CO₂ concentration balance equation:  
        ##################################################################################################################
        m.Equation(co2_indoor__ppm.dt() == co2_indoor_gain__ppm_s_1 - co2_indoor_loss__ppm_s_1)
        
        ##################################################################################################################
        # Solve the model to start the learning process
        ##################################################################################################################
        m.options.IMODE = 5        # Simultaneous Estimation 
        m.options.SOLVER = 3       # based on Brains4Building using IPOPT is recommended for ventilation learning
        m.options.EV_TYPE = 2      # RMSE
        m.solve(disp=False)

        ##################################################################################################################
        # Store results of the learning process
        ##################################################################################################################

        if m.options.APPSTATUS == 1:
            
            # print(f"A solution was found for id {id} from {start} to {end} with duration {duration}")

            # Load results
            try:
                results = m.load_results()
                # DEBUG Save results to the local directory
                filename = 'gekko_results_vent.json'
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=4)
            except AttributeError:
                results = None
                # print("load_results() not available.")
            
            if any(item is not None for item in [learn_params, predict_props]):
                # Initialize DataFrame for learned thermal parameters (only for learning mode)
                df_learned_parameters = pd.DataFrame({
                    'id': id, 
                    'start': start,
                    'end': end,
                    'duration': duration,
                }, index=[0])
            
            # Loop over the learn_params set and store learned values and calculate MAE if actual value is available
            for param in (learn_params - (predict_props or set())):
                learned_value = results.get(param.lower(), [np.nan])[0]
                df_learned_parameters.loc[0, f'learned_{param}'] = learned_value
                # If actual value exists, compute MAE
                if actual_params is not None and param in actual_params:
                    df_learned_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_params[param])
    
            if predict_props is not None:
                # Initialize a DataFrame for learned time-varying properties
                df_predicted_properties = pd.DataFrame(index=df_learn.index)
            
                # Store learned time-varying data in DataFrame and calculate MAE and RMSE
                current_locals = locals() # current_locals is valid in list comprehensions and for loops, locals() is not. 
                for prop in (predict_props or set()) & set(current_locals.keys()):
                    predicted_prop = f'predicted_{prop}'
                    df_predicted_properties.loc[:,predicted_prop] = np.asarray(current_locals[prop].value)


                    ##### additional debug 
                    # Count rows in df_results_per_period where all values in the subset of columns are non-NaN
                    valid_source_rows = array_length = len(np.asarray(current_locals[prop].value))
                    
                    # Count non-NaN values in the target column of df_predicted_properties
                    non_nan_target_values = df_predicted_properties[predicted_prop].notna().sum()
                    
                    # Compare the counts
                    if valid_source_rows != non_nan_target_values:
                        print("Counts do not match:")
                        print(f"Valid source rows: {valid_source_rows}, Target non-NaN values: {non_nan_target_values}")
                    # else:
                        # print("Counts match: Target non-NaN values correspond to valid source rows.")
                    #####
                    
                    # If the property was measured, calculate and store MAE and RMSE
                    if prop in property_sources.keys() and property_sources[prop] in set(df_learn.columns):
                        df_learned_parameters.loc[0, f'mae_{prop}'] = mae(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
                        df_learned_parameters.loc[0, f'rmse_{prop}'] = rmse(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
            
            if any(item is not None for item in [learn_params, predict_props]):
                # Set MultiIndex on the DataFrame (id, start, end)
                df_learned_parameters.set_index(['id', 'start', 'end', 'duration'], inplace=True)

        else:
            df_learned_parameters = pd.DataFrame()
            # print(f"NO Solution was found for id {id} from {start} to {end} with duration {duration}")
        
        m.cleanup()

        return df_learned_parameters, df_predicted_properties

    
    def heat_distribution(
        df_learn,
        bldng_data: Dict = None,
        property_sources: Dict = None,
        param_hints: Dict = None,
        learn_params: Set[str] = {'heat_tr_dstr__W_K_1',
                                  'th_mass_dstr__Wh_K_1',
                                  'th_inert_dstr__h',
                                  'flow_dstr_capacity__dm3_s_1', 
                                 },
        actual_params: Dict = None,
        predict_props: Set[str] = {'temp_ret_ch__degC',
                                   'temp_dstr__degC',
                                   'heat_dstr__W',
                                  }        
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        logging.info(f"learn heat distribution for id {df_learn.index.get_level_values('id')[0]}, from  {df_learn.index.get_level_values('timestamp').min()} to {df_learn.index.get_level_values('timestamp').max()}")


 
        id, start, end, step__s, duration__s  = Learner.get_time_info(df_learn) 
        duration = timedelta(seconds=duration__s)

        ##################################################################################################################
        # GEKKO Model - Initialize
        ##################################################################################################################
        m = GEKKO(remote=False)
        m.time = np.arange(0, duration__s, step__s)

        ##################################################################################################################
        # Central heating gains
        ##################################################################################################################
        heat_ch__W = m.MV(value=df_learn[property_sources['heat_ch__W']].astype('float32').values, name='heat_ch__W')
        heat_ch__W.STATUS = 0  # No optimization
        heat_ch__W.FSTATUS = 1 # Use the measured values
    
        ##################################################################################################################
        # Heat distribution system parameters to learn
        ##################################################################################################################
        # Effective heat transfer capacity of the heat distribution system
        heat_tr_dstr__W_K_1 = m.FV(value=param_hints['heat_tr_dstr__W_K_1'], lb=50, ub=1000, name='heat_tr_dstr__W_K_1')
        heat_tr_dstr__W_K_1.STATUS = 1  # Allow optimization
        heat_tr_dstr__W_K_1.FSTATUS = 1 # Use the initial value as a hint for the solver

        # Effective thermal mass of the heat distribution system
        th_mass_dstr__Wh_K_1 = m.FV(value=param_hints['th_mass_dstr__Wh_K_1'], lb=50, ub=5000, name='th_mass_dstr__Wh_K_1')
        th_mass_dstr__Wh_K_1.STATUS = 1  # Allow optimization
        th_mass_dstr__Wh_K_1.FSTATUS = 1 # Use the initial value as a hint for the solver

        # Effective thermal inertia (a.k.a. thermal time constant) of the heat distribution system
        if 'th_inert_dstr__h' in learn_params:
            th_inert_dstr__h = m.Intermediate(th_mass_dstr__Wh_K_1 / heat_tr_dstr__W_K_1, name='th_inert_dstr__h')
    
        ##################################################################################################################
        # Flow and indoor temperature  
        ##################################################################################################################
        temp_flow_ch__degC = m.MV(value=df_learn[property_sources['temp_flow_ch__degC']].astype('float32').values, name='temp_flow_ch__degC')
        temp_flow_ch__degC.STATUS = 0  # No optimization
        temp_flow_ch__degC.FSTATUS = 1 # Use the measured values

        temp_indoor__degC = m.MV(value=df_learn[property_sources['temp_indoor__degC']].astype('float32').values, name='temp_indoor__degC')
        temp_indoor__degC.STATUS = 0  # No optimization
        temp_indoor__degC.FSTATUS = 1 # Use the measured values

        # ##################################################################################################################
        # # Alternative way: fit on distribution temperature
        # ##################################################################################################################
        # temp_dstr__degC = m.CV(value=((df_learn[property_sources['temp_flow_ch__degC']] 
        #                                + df_learn[property_sources['temp_ret_ch__degC']]) / 2).astype('float32').values, name='temp_dstr__degC')
        # temp_dstr__degC.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
        # temp_dstr__degC.FSTATUS = 1 # Use the measured values
        # temp_ret_ch__degC = m.Var(value=df_learn[property_sources['temp_ret_ch__degC']].iloc[0], name='temp_ret_ch__degC')  # Initial guesss
        
        ##################################################################################################################
        # Fit on return temperature
        ##################################################################################################################
        if learn_params is None:
            # Simulation mode: Use initial measured value as the starting point
            temp_ret_ch__degC = m.Var(value=df_learn[property_sources['temp_ret_ch__degC']].iloc[0], name='temp_ret_ch__degC')
        else:
            temp_ret_ch__degC = m.CV(value=df_learn[property_sources['temp_ret_ch__degC']].astype('float32').values, name='temp_ret_ch__degC')
            temp_ret_ch__degC.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
            temp_ret_ch__degC.FSTATUS = 1 # Use the measured values

        temp_dstr__degC = m.Var(value=(df_learn[property_sources['temp_flow_ch__degC']].iloc[0] 
                                       + df_learn[property_sources['temp_ret_ch__degC']].iloc[0]) / 2, name='temp_dstr__degC')  # Initial guesss

        ##################################################################################################################
        # Pump speed flow ratio (and heat distribution flow resistance if pump head is known)
        ##################################################################################################################

        if learn_params is None:
            flow_dstr_capacity__dm3_s_1 = m.Param(value=bldng_data['flow_dstr_capacity__dm3_s_1'], name='flow_dstr_capacity__dm3_s_1')
        else:
            if 'flow_dstr_capacity__dm3_s_1' in learn_params:
                # Flow distribution capacity
                flow_dstr_capacity__dm3_s_1 = m.FV(value=param_hints['flow_dstr_capacity__dm3_s_1'], name='flow_dstr_capacity__dm3_s_1')
                flow_dstr_capacity__dm3_s_1.STATUS = 1  # Allow optimization
                flow_dstr_capacity__dm3_s_1.FSTATUS = 1 # Use the initial value as a hint for the solver

                # Flow rate in the distribution system
                flow_dstr__dm3_s_1 = m.MV(value=df_learn[property_sources['flow_dstr__dm3_s_1']].astype('float32').values, name='flow_dstr__dm3_s_1')
                flow_dstr__dm3_s_1.STATUS = 0  # No optimization
                flow_dstr__dm3_s_1.FSTATUS = 1 # Use the measured values

                # Pump speed that drives the flow in the heat distribution system
                flow_dstr_pump_speed__pct = m.MV(value=df_learn[property_sources['flow_dstr_pump_speed__pct']].astype('float32').values, name='flow_dstr_pump_speed__pct')
                flow_dstr_pump_speed__pct.STATUS = 0  # No optimization
                flow_dstr_pump_speed__pct.FSTATUS = 1 # Use the measured values
        
                # Flow equation
                m.Equation(flow_dstr__dm3_s_1 == flow_dstr_capacity__dm3_s_1 * flow_dstr_pump_speed__pct/100)
            else:
                flow_dstr_capacity__dm3_s_1 = m.Param(value=param_hints['flow_dstr_capacity__dm3_s_1'], name='flow_dstr_capacity__dm3_s_1')



        ##################################################################################################################
        # Dynamic model of the heat distribution system
        ##################################################################################################################
        m.Equation(temp_dstr__degC == (temp_flow_ch__degC + temp_ret_ch__degC) / 2)
        heat_dstr__W = m.Intermediate(heat_tr_dstr__W_K_1 * (temp_dstr__degC - temp_indoor__degC), name='heat_dstr__W')
        th_mass_dstr__J_K_1 = m.Intermediate(th_mass_dstr__Wh_K_1 * s_h_1,  name='th_mass_dstr__J_K_1') 
        m.Equation(temp_dstr__degC.dt() == (heat_ch__W - heat_dstr__W) / th_mass_dstr__J_K_1)

        
        ##################################################################################################################
        # Solve the model to start the learning process
        ##################################################################################################################
        if learn_params is None:
            m.options.IMODE = 4    # Do not learn, but only simulate using learned parameters passed via bldng_data
        else:
            m.options.IMODE = 5    # Learn one or more parameter values using Simultaneous Estimation 
        m.options.EV_TYPE = 2      # RMSE
        m.solve(disp=False)

        ##################################################################################################################
        # Store results of the learning process
        ##################################################################################################################
        
        if m.options.APPSTATUS == 1:
            
            # Load results
            try:
                results = m.load_results()
                # DEBUG Save results to the local directory
                filename = 'gekko_results_dstr.json'
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=4)
            except AttributeError:
                results = None
                print("load_results() not available.")
            
            if any(item is not None for item in [learn_params, predict_props]):
                # Initialize DataFrame for learned thermal parameters (only for learning mode)
                df_learned_parameters = pd.DataFrame({
                    'id': id, 
                    'start': start,
                    'end': end,
                    'duration': duration,
                }, index=[0])
            
            # Loop over the learn_params set and store learned values and calculate MAE if actual value is available
            for param in (learn_params - (predict_props or set())):
                learned_value = results.get(param.lower(), [np.nan])[0]
                df_learned_parameters.loc[0, f'learned_{param}'] = learned_value
                # If actual value exists, compute MAE
                if actual_params is not None and param in actual_params:
                    df_learned_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_params[param])

            if predict_props is not None:
                # Initialize a DataFrame for learned time-varying properties
                df_predicted_properties = pd.DataFrame(index=df_learn.index)
            
                # Store learned time-varying data in DataFrame and calculate MAE and RMSE
                current_locals = locals() # current_locals is valid in list comprehensions and for loops, locals() is not. 
                for prop in (predict_props or set()) & set(current_locals.keys()):
                    predicted_prop = f'predicted_{prop}'
                    df_predicted_properties.loc[:,predicted_prop] = np.asarray(current_locals[prop].value)
            
                    # If the property was measured, calculate and store MAE and RMSE
                    if prop in property_sources.keys() and property_sources[prop] in set(df_learn.columns):
                        df_learned_parameters.loc[0, f'mae_{prop}'] = mae(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
                        df_learned_parameters.loc[0, f'rmse_{prop}'] = rmse(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
            
            if any(item is not None for item in [learn_params, predict_props]):
                # Set MultiIndex on the DataFrame (id, start, end)
                df_learned_parameters.set_index(['id', 'start', 'end', 'duration'], inplace=True)

        else:
            df_learned_parameters = pd.DataFrame()
        
        m.cleanup()

        return df_learned_parameters, df_predicted_properties

    
    def add_building_model(
        m: GEKKO,
        df_learn: pd.DataFrame,
        bldng_data: Dict = None,
        property_sources: Dict = None,
        param_hints: Dict = None,
        learn_params: Set[str] = None,
        
    ) -> GEKKO:
        """
        Adds the building submodel to the given GEKKO model.
        
        Parameters:
        - m (GEKKO): The GEKKO model instance to add variables and equations to.
        - df_learn (pd.DataFrame): Dataframe with time series data.
        - bldng_data (dict): Building-specific data (e.g., volume, learned parameters).
        - property_sources (dict): Mapping of property names to DataFrame columns.
        - param_hints (dict): Default parameter values.
        - learn_params (set): Parameters to learn (optional).
        
        Returns:
        - GEKKO: The updated GEKKO model with added submodel.
        """

        bldng__m3 = bldng_data['bldng__m3']
        
        ##################################################################################################################
        # Heat gains
        ##################################################################################################################
    
        # Central heating gains
        g_use_ch_hhv__W = m.MV(value=df_learn[property_sources['g_use_ch_hhv__W']].astype('float32').values, name='g_use_ch_hhv__W')
        g_use_ch_hhv__W.STATUS = 0  # No optimization
        g_use_ch_hhv__W.FSTATUS = 1 # Use the measured values

        e_use_ch__W = 0.0  # TODO: add electricity use from heat pump here when hybrid or all-electic heat pumps must be simulated
    
        eta_ch_hhv__W0 = m.MV(value=df_learn[property_sources['eta_ch_hhv__W0']].astype('float32').values, name='eta_ch_hhv__W0')
        eta_ch_hhv__W0.STATUS = 0  # No optimization
        eta_ch_hhv__W0.FSTATUS = 1 # Use the measured values
    
        heat_g_ch__W = m.Intermediate(g_use_ch_hhv__W * eta_ch_hhv__W0, name='heat_g_ch__W')

        # TODO: add heat gains from heat pump here when hybrid or all-electic heat pumps must be simulated
        cop_ch__W0 = 1.0
        heat_e_ch__W = e_use_ch__W * cop_ch__W0
        
        # Heat generation power input from gas (and electricity)
        power_input_ch__W = m.Intermediate(g_use_ch_hhv__W + e_use_ch__W, name='power_input_ch__W')

        # Heating power output to heat distribution system
        heat_ch__W = m.Intermediate(heat_g_ch__W + heat_e_ch__W, name='heat_ch__W')
    
        if learn_params is None:
            # Simulation mode: Use initial measured value as the starting point
            temp_indoor__degC = m.Var(value=df_learn[property_sources['temp_indoor__degC']].iloc[0], name='temp_indoor__degC')
        else:
            # Learning mode: optimize indoor temperature
            temp_indoor__degC = m.CV(value=df_learn[property_sources['temp_indoor__degC']].astype('float32').values, name='temp_indoor__degC')
            temp_indoor__degC.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
            temp_indoor__degC.FSTATUS = 1  # Use the measured values
            
        ##################################################################################################################
        # Heat gains from heat distribution system
        ##################################################################################################################
        # TO DO: merge code with heat distribution model, making the code usable both in training and testing
        
        # If possible, use the learned heat transfer capacity of the heat distribution system
        if (pd.notna(bldng_data.get('learned_heat_tr_dstr__W_K_1')) and
            pd.notna(bldng_data.get('learned_th_mass_dstr__Wh_K_1'))
        ):
            # Use learned parameters
            heat_tr_dstr__W_K_1 =  m.Param(value=bldng_data['learned_heat_tr_dstr__W_K_1'], name='heat_tr_dstr__W_K_1')
            th_mass_dstr__Wh_K_1 =  m.Param(value=bldng_data['learned_th_mass_dstr__Wh_K_1'], name='th_mass_dstr__Wh_K_1')
    
            # Estimate initial temperature for the distribution system
            if (pd.notna(df_learn[property_sources['temp_flow_ch__degC']].iloc[0]) and 
                pd.notna(df_learn[property_sources['temp_ret_ch__degC']].iloc[0])
            ):
                # Estimate based on initial supply and return temperature
                initial_temp_dstr__degC = (
                    df_learn[property_sources['temp_flow_ch__degC']].iloc[0]
                    + df_learn[property_sources['temp_ret_ch__degC']].iloc[0]
                ) / 2
            else:
                # We're not starting in the middle of a heat generation streak, so we estimate based on indoor temperature
                initial_temp_dstr__degC = df_learn[property_sources['temp_indoor__degC']].iloc[0] 
                
            # Define variables for the dynamic model
            temp_dstr__degC = m.Var(value=initial_temp_dstr__degC, name='temp_dstr__degC')

            # Define equations for the dynamic model
            heat_dstr__W = m.Intermediate(heat_tr_dstr__W_K_1 * (temp_dstr__degC - temp_indoor__degC), name='heat_dstr__W')
            th_mass_dstr__J_K_1 = m.Intermediate(th_mass_dstr__Wh_K_1 * s_h_1, name='th_mass_dstr__J_K_1')
            m.Equation(temp_dstr__degC.dt() == (heat_ch__W - heat_dstr__W) / th_mass_dstr__J_K_1)

        else:
            # Simplistic model for heat distribution: immediate and full heat distribution
            heat_dstr__W =  m.Intermediate(heat_ch__W, name='heat_dstr__W')
    
        ##################################################################################################################
        # Solar heat gains
        ##################################################################################################################
    
        if learn_params is None:
            # Simulation mode: use the value from bldng_data
            aperture_sol__m2 = m.Param(value=bldng_data['learned_aperture_sol__m2'], name='aperture_sol__m2')
        else:
            # Learning mode: decide based on presence in learn_params
            if 'aperture_sol__m2' in learn_params:
                aperture_sol__m2 = m.FV(value=param_hints['aperture_sol__m2'], lb=1, ub=100, name='aperture_sol__m2')
                aperture_sol__m2.STATUS = 1  # Allow optimization
                aperture_sol__m2.FSTATUS = 1 # Use the initial value as a hint for the solver
            else:
                aperture_sol__m2 = m.Param(value=param_hints['aperture_sol__m2'], name='aperture_sol__m2')
    
        sol_ghi__W_m_2 = m.MV(value=df_learn[property_sources['sol_ghi__W_m_2']].astype('float32').values, name='sol_ghi__W_m_2')
        sol_ghi__W_m_2.STATUS = 0  # No optimization
        sol_ghi__W_m_2.FSTATUS = 1 # Use the measured values
    
        heat_sol__W = m.Intermediate(sol_ghi__W_m_2 * aperture_sol__m2, name='heat_sol__W')
    
        ##################################################################################################################
        ## Internal heat gains ##
        ##################################################################################################################

        # Heat gains from domestic hot water

        g_use_dhw_hhv__W = m.MV(value = df_learn[property_sources['g_use_dhw_hhv__W']].astype('float32').values, name='g_use_dhw_hhv__W')
        g_use_dhw_hhv__W.STATUS = 0  # No optimization 
        g_use_dhw_hhv__W.FSTATUS = 1 # Use the measured values
        heat_g_dhw__W = m.Intermediate(g_use_dhw_hhv__W * param_hints['eta_dhw_hhv__W0'] * param_hints['frac_remain_dhw__0'], name='heat_g_dhw__W')

        # Heat gains from cooking
        heat_g_cooking__W = m.Param(param_hints['g_use_cooking_hhv__W'] * param_hints['eta_cooking_hhv__W0'] * param_hints['frac_remain_cooking__0'], name='heat_g_cooking__W')

        # Heat gains from electricity
        # we assume all electricity is used indoors and turned into heat
        heat_e__W = m.MV(value = df_learn[property_sources['e__W']].astype('float32').values, name='heat_e__W')
        heat_e__W.STATUS = 0  # No optimization
        heat_e__W.FSTATUS = 1 # Use the measured values

        # Heat gains from occupants
        occupancy__p = m.MV(value = df_learn[property_sources['occupancy__p']].astype('float32').values, name='occupancy__p')
        occupancy__p.STATUS = 0  # No optimization
        occupancy__p.FSTATUS = 1 # Use the measured values
        heat_int_occupancy__W = m.Intermediate(occupancy__p * param_hints['heat_int__W_p_1'], name='heat_int_occupancy__W')

        # Sum of all 'internal' heat gains 
        heat_int__W = m.Intermediate(heat_g_dhw__W + heat_g_cooking__W + heat_e__W + heat_int_occupancy__W, name='heat_int__W')
        
        ##################################################################################################################
        # Conductive heat losses
        ##################################################################################################################
    
        if learn_params is None:
            # Simulation mode: use the value from bldng_data
            heat_tr_bldng_cond__W_K_1 = m.Param(value=bldng_data['learned_heat_tr_bldng_cond__W_K_1'], name='heat_tr_bldng_cond__W_K_1')
        else:
            # Learning mode: decide based on presence in learn_params
            if 'heat_tr_bldng_cond__W_K_1' in learn_params:
                heat_tr_bldng_cond__W_K_1 = m.FV(value=param_hints['heat_tr_bldng_cond__W_K_1'], lb=0, ub=1000, name='heat_tr_bldng_cond__W_K_1')
                heat_tr_bldng_cond__W_K_1.STATUS = 1  # Allow optimization
                heat_tr_bldng_cond__W_K_1.FSTATUS = 1 # Use the initial value as a hint for the solver
            else:
                heat_tr_bldng_cond__W_K_1 = m.Param(param_hints['heat_tr_bldng_cond__W_K_1'], name='heat_tr_bldng_cond__W_K_1')
    
        temp_outdoor__degC = m.MV(value=df_learn[property_sources['temp_outdoor__degC']].astype('float32').values, name='temp_outdoor__degC')
        temp_outdoor__degC.STATUS = 0  # No optimization
        temp_outdoor__degC.FSTATUS = 1 # Use the measured values
    
        delta_t_indoor_outdoor__K = m.Intermediate(temp_indoor__degC - temp_outdoor__degC, name='delta_t_indoor_outdoor__K')
    
        heat_loss_bldng_cond__W = m.Intermediate(heat_tr_bldng_cond__W_K_1 * delta_t_indoor_outdoor__K, name='heat_loss_bldng_cond__W')
    
        ##################################################################################################################
        # Infiltration and ventilation heat losses
        ##################################################################################################################
    
        wind__m_s_1 = m.MV(value=df_learn[property_sources['wind__m_s_1']].astype('float32').values, name='wind__m_s_1')
        wind__m_s_1.STATUS = 0  # No optimization
        wind__m_s_1.FSTATUS = 1 # Use the measured values
    
        if learn_params is None:
            # Simulation mode: use the value from bldng_data
            aperture_inf__cm2 = m.Param(value=bldng_data['learned_aperture_inf__cm2'], name='aperture_inf__cm2')
        else:
            # Learning mode: decide based on presence in learn_params
            if 'aperture_inf__cm2' in learn_params:
                aperture_inf__cm2 = m.FV(value=param_hints['aperture_inf__cm2'], lb=0, ub=100000.0, name='aperture_inf__cm2')
                aperture_inf__cm2.STATUS = 1  # Allow optimization
                aperture_inf__cm2.FSTATUS = 1 # Use the initial value as a hint for the solver
            else:
                aperture_inf__cm2 = m.Param(value=param_hints['aperture_inf__cm2'], name='aperture_inf__cm2')
    
        air_inf__m3_s_1 = m.Intermediate(wind__m_s_1 * aperture_inf__cm2 / cm2_m_2, name='air_inf__m3_s_1')
        heat_tr_bldng_inf__W_K_1 = m.Intermediate(air_inf__m3_s_1 * air_room__J_m_3_K_1, name='heat_tr_bldng_inf__W_K_1')
        heat_loss_bldng_inf__W = m.Intermediate(heat_tr_bldng_inf__W_K_1 * delta_t_indoor_outdoor__K, name='heat_loss_bldng_inf__W')
    
        if learn_params is None or (property_sources['ventilation__dm3_s_1'] in df_learn.columns and df_learn[property_sources['ventilation__dm3_s_1']].notna().all()):
            ventilation__dm3_s_1 = m.MV(value=df_learn[property_sources['ventilation__dm3_s_1']].astype('float32').values, name='ventilation__dm3_s_1')
            ventilation__dm3_s_1.STATUS = 0  # No optimization
            ventilation__dm3_s_1.FSTATUS = 1  # Use the measured values
            
            air_changes_vent__s_1 = m.Intermediate(ventilation__dm3_s_1 / (bldng__m3 * dm3_m_3), name='air_changes_vent__s_1')
            heat_tr_bldng_vent__W_K_1 = m.Intermediate(air_changes_vent__s_1 * bldng__m3 * air_room__J_m_3_K_1, name='heat_tr_bldng_vent__W_K_1')
            heat_loss_bldng_vent__W = m.Intermediate(heat_tr_bldng_vent__W_K_1 * delta_t_indoor_outdoor__K, name='heat_loss_bldng_vent__W')
        else:
            heat_tr_bldng_vent__W_K_1 = m.Var(0, name='heat_tr_bldng_vent__W_K_1')
            heat_loss_bldng_vent__W = m.Var(0, name='heat_loss_bldng_vent__W')

        ##################################################################################################################
        ## Thermal inertia ##
        ##################################################################################################################
                    
        if learn_params is None:
            # Simulation mode: use the value from bldng_data
            th_inert_bldng__h = m.Param(value=bldng_data['learned_th_inert_bldng__h'], name='th_inert_bldng__h')
        else:
            # Learning mode: decide based on presence in learn_params
            if 'th_inert_bldng__h' in learn_params:
                # Learn thermal inertia
                th_inert_bldng__h = m.FV(value = param_hints['th_inert_bldng__h'], lb=(10), ub=(1000), name='th_inert_bldng__h')
                th_inert_bldng__h.STATUS = 1  # Allow optimization
                th_inert_bldng__h.FSTATUS = 1 # Use the initial value as a hint for the solver
            else:
                # Do not learn thermal inertia of the building, but use a fixed value based on hint
                th_inert_bldng__h = m.Param(value = param_hints['th_inert_bldng__h'], name='th_inert_bldng__h')
                # TO DO: check whether we indeed can remove the line below
                # learned_th_inert_bldng__h = np.nan
        
        ##################################################################################################################
        ### Heat balance ###
        ##################################################################################################################

        heat_gain_bldng__W = m.Intermediate(heat_dstr__W + heat_sol__W + heat_int__W, name='heat_gain_bldng__W')
        heat_loss_bldng__W = m.Intermediate(heat_loss_bldng_cond__W + heat_loss_bldng_inf__W + heat_loss_bldng_vent__W, name='heat_loss_bldng__W')
        heat_tr_bldng__W_K_1 = m.Intermediate(heat_tr_bldng_cond__W_K_1 + heat_tr_bldng_inf__W_K_1 + heat_tr_bldng_vent__W_K_1, name='heat_tr_bldng__W_K_1')
        th_mass_bldng__Wh_K_1  = m.Intermediate(heat_tr_bldng__W_K_1 * th_inert_bldng__h, name='th_mass_bldng__Wh_K_1') 
        m.Equation(temp_indoor__degC.dt() == ((heat_gain_bldng__W - heat_loss_bldng__W)  / (th_mass_bldng__Wh_K_1 * s_h_1)))

        return m

    def building(
        df_learn: pd.DataFrame,
        bldng_data: Dict = None,
        property_sources: Dict = None,
        param_hints: Dict = None,
        learn_params: Set[str] = None,
        actual_params: Dict = None,
        predict_props: Set[str] = None,
        max_iter=None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Learn thermal parameters for a building's heating system using GEKKO.
        
        Parameters:
        df_learn (pd.DataFrame): DataFrame containing the time series data to be used for learning.
        property_sources (dict): Dictionary mapping property names to their corresponding columns in df_learn.
        param_hints (dict): Dictionary containing default values for the various parameters.
        learn_params (dict): Dictionary of parameters to be learned.
        actual_params (dict, optional): Dictionary of actual values of the parameters to be learned.
        bldng_data: dictionary containing at least:
        - bldng__m3 (float): Volume of the building in m3.
        """
        
        # Periodic averages to calculate, which include Energy Case metrics (as far as available in the df_learn columns)
        properties_mean = {
            'temp_set__degC',
            'temp_flow_ch__degC',
            'temp_ret_ch__degC',
            'comfortable__bool',
            'temp_indoor__degC',
            'temp_outdoor__degC',
            'temp_flow_ch_max__degC',
            'heat_ch__W',
        }
            
        id, start, end, step__s, duration__s  = Learner.get_time_info(df_learn) 
        duration = timedelta(seconds=duration__s)

        logging.info(f"learn_thermal_parameters for id {df_learn.index.get_level_values('id')[0]}, from  {df_learn.index.get_level_values('timestamp').min()} to {df_learn.index.get_level_values('timestamp').max()}")

        ##################################################################################################################
        # GEKKO Model - Initialize
        ##################################################################################################################

        m = GEKKO(remote=False)
        m.time = np.arange(0, duration__s, step__s)

        bldng__m3 = bldng_data['bldng__m3']
        
        ##################################################################################################################
        # Heat gains
        ##################################################################################################################
    
        # Central heating gains
        heat_ch__W = m.MV(value=df_learn[property_sources['heat_ch__W']].astype('float32').values, name='heat_ch__W')
        heat_ch__W.STATUS = 0  # No optimization
        heat_ch__W.FSTATUS = 1 # Use the measured values

        if learn_params is None:
            # Simulation mode: Use initial measured value as the starting point
            temp_indoor__degC = m.Var(value=df_learn[property_sources['temp_indoor__degC']].iloc[0], name='temp_indoor__degC')
        else:
            # Learning mode: optimize indoor temperature
            temp_indoor__degC = m.CV(value=df_learn[property_sources['temp_indoor__degC']].astype('float32').values, name='temp_indoor__degC')
            temp_indoor__degC.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
            temp_indoor__degC.FSTATUS = 1  # Use the measured values
            
        ##################################################################################################################
        # Heat gains from heat distribution system
        ##################################################################################################################
        # TO DO: merge code with heat distribution model, making the code usable both in training and testing
        
        # If possible, use the learned heat transfer capacity of the heat distribution system
        if (pd.notna(bldng_data.get('learned_heat_tr_dstr__W_K_1')) and
            pd.notna(bldng_data.get('learned_th_mass_dstr__Wh_K_1'))
        ):
            # Use learned parameters
            heat_tr_dstr__W_K_1 =  m.Param(value=bldng_data['learned_heat_tr_dstr__W_K_1'], name='heat_tr_dstr__W_K_1')
            th_mass_dstr__Wh_K_1 =  m.Param(value=bldng_data['learned_th_mass_dstr__Wh_K_1'], name='th_mass_dstr__Wh_K_1')
    
            # Estimate initial temperature for the distribution system
            if (pd.notna(df_learn[property_sources['temp_flow_ch__degC']].iloc[0]) and 
                pd.notna(df_learn[property_sources['temp_ret_ch__degC']].iloc[0])
            ):
                # Estimate based on initial supply and return temperature
                initial_temp_dstr__degC = (
                    df_learn[property_sources['temp_flow_ch__degC']].iloc[0]
                    + df_learn[property_sources['temp_ret_ch__degC']].iloc[0]
                ) / 2
            else:
                # We're not starting in the middle of a heat generation streak, so we estimate based on indoor temperature
                initial_temp_dstr__degC = df_learn[property_sources['temp_indoor__degC']].iloc[0] 
                
            # Define variables for the dynamic model
            temp_dstr__degC = m.Var(value=initial_temp_dstr__degC, name='temp_dstr__degC')

            # Define equations for the dynamic model
            heat_dstr__W = m.Intermediate(heat_tr_dstr__W_K_1 * (temp_dstr__degC - temp_indoor__degC), name='heat_dstr__W')
            m.Equation(temp_dstr__degC.dt() == (heat_ch__W - heat_dstr__W) / (th_mass_dstr__Wh_K_1 * s_h_1))

        else:
            # Simplistic model for heat distribution: immediate and full heat distribution
            heat_dstr__W =  m.Intermediate(heat_ch__W, name='heat_dstr__W')
    
        ##################################################################################################################
        # Solar heat gains
        ##################################################################################################################
    
        if learn_params is None:
            # Simulation mode: use the value from bldng_data
            aperture_sol__m2 = m.Param(value=bldng_data['learned_aperture_sol__m2'], name='aperture_sol__m2')
        else:
            # Learning mode: decide based on presence in learn_params
            if 'aperture_sol__m2' in learn_params:
                aperture_sol__m2 = m.FV(value=param_hints['aperture_sol__m2'], lb=1, ub=100, name='aperture_sol__m2')
                aperture_sol__m2.STATUS = 1  # Allow optimization
                aperture_sol__m2.FSTATUS = 1 # Use the initial value as a hint for the solver
            else:
                aperture_sol__m2 = m.Param(value=param_hints['aperture_sol__m2'], name='aperture_sol__m2')
    
        sol_ghi__W_m_2 = m.MV(value=df_learn[property_sources['sol_ghi__W_m_2']].astype('float32').values, name='sol_ghi__W_m_2')
        sol_ghi__W_m_2.STATUS = 0  # No optimization
        sol_ghi__W_m_2.FSTATUS = 1 # Use the measured values
    
        heat_sol__W = m.Intermediate(sol_ghi__W_m_2 * aperture_sol__m2, name='heat_sol__W')
    
        ##################################################################################################################
        ## Internal heat gains ##
        ##################################################################################################################

        # Heat gains from domestic hot water

        g_use_dhw_hhv__W = m.MV(value = df_learn[property_sources['g_use_dhw_hhv__W']].astype('float32').values, name='g_use_dhw_hhv__W')
        g_use_dhw_hhv__W.STATUS = 0  # No optimization 
        g_use_dhw_hhv__W.FSTATUS = 1 # Use the measured values
        heat_g_dhw__W = m.Intermediate(g_use_dhw_hhv__W * param_hints['eta_dhw_hhv__W0'] * param_hints['frac_remain_dhw__0'], name='heat_g_dhw__W')

        # Heat gains from cooking
        heat_g_cooking__W = m.Param(param_hints['g_use_cooking_hhv__W'] * param_hints['eta_cooking_hhv__W0'] * param_hints['frac_remain_cooking__0'], name='heat_g_cooking__W')

        # Heat gains from electricity
        # we assume all electricity is used indoors and turned into heat
        heat_e__W = m.MV(value = df_learn[property_sources['e__W']].astype('float32').values, name='heat_e__W')
        heat_e__W.STATUS = 0  # No optimization
        heat_e__W.FSTATUS = 1 # Use the measured values

        # Heat gains from occupants
        occupancy__p = m.MV(value = df_learn[property_sources['occupancy__p']].astype('float32').values, name='occupancy__p')
        occupancy__p.STATUS = 0  # No optimization
        occupancy__p.FSTATUS = 1 # Use the measured values
        heat_int_occupancy__W = m.Intermediate(occupancy__p * param_hints['heat_int__W_p_1'], name='heat_int_occupancy__W')

        # Sum of all 'internal' heat gains 
        heat_int__W = m.Intermediate(heat_g_dhw__W + heat_g_cooking__W + heat_e__W + heat_int_occupancy__W, name='heat_int__W')
        
        ##################################################################################################################
        # Conductive heat losses
        ##################################################################################################################
    
        if learn_params is None:
            # Simulation mode: use the value from bldng_data
            heat_tr_bldng_cond__W_K_1 = m.Param(value=bldng_data['learned_heat_tr_bldng_cond__W_K_1'], name='heat_tr_bldng_cond__W_K_1')
        else:
            # Learning mode: decide based on presence in learn_params
            if 'heat_tr_bldng_cond__W_K_1' in learn_params:
                heat_tr_bldng_cond__W_K_1 = m.FV(value=param_hints['heat_tr_bldng_cond__W_K_1'], lb=0, ub=1000, name='heat_tr_bldng_cond__W_K_1')
                heat_tr_bldng_cond__W_K_1.STATUS = 1  # Allow optimization
                heat_tr_bldng_cond__W_K_1.FSTATUS = 1 # Use the initial value as a hint for the solver
            else:
                heat_tr_bldng_cond__W_K_1 = m.Param(param_hints['heat_tr_bldng_cond__W_K_1'], name='heat_tr_bldng_cond__W_K_1')
    
        temp_outdoor__degC = m.MV(value=df_learn[property_sources['temp_outdoor__degC']].astype('float32').values, name='temp_outdoor__degC')
        temp_outdoor__degC.STATUS = 0  # No optimization
        temp_outdoor__degC.FSTATUS = 1 # Use the measured values
    
        delta_t_indoor_outdoor__K = m.Intermediate(temp_indoor__degC - temp_outdoor__degC, name='delta_t_indoor_outdoor__K')
    
        heat_loss_bldng_cond__W = m.Intermediate(heat_tr_bldng_cond__W_K_1 * delta_t_indoor_outdoor__K, name='heat_loss_bldng_cond__W')
    
        ##################################################################################################################
        # Infiltration and ventilation heat losses
        ##################################################################################################################
    
        wind__m_s_1 = m.MV(value=df_learn[property_sources['wind__m_s_1']].astype('float32').values, name='wind__m_s_1')
        wind__m_s_1.STATUS = 0  # No optimization
        wind__m_s_1.FSTATUS = 1 # Use the measured values
    
        if learn_params is None:
            # Simulation mode: use the value from bldng_data
            aperture_inf__cm2 = m.Param(value=bldng_data['learned_aperture_inf__cm2'], name='aperture_inf__cm2')
        else:
            # Learning mode: decide based on presence in learn_params
            if 'aperture_inf__cm2' in learn_params:
                aperture_inf__cm2 = m.FV(value=param_hints['aperture_inf__cm2'], lb=0, ub=100000.0, name='aperture_inf__cm2')
                aperture_inf__cm2.STATUS = 1  # Allow optimization
                aperture_inf__cm2.FSTATUS = 1 # Use the initial value as a hint for the solver
            else:
                aperture_inf__cm2 = m.Param(value=param_hints['aperture_inf__cm2'], name='aperture_inf__cm2')
    
        air_inf__m3_s_1 = m.Intermediate(wind__m_s_1 * aperture_inf__cm2 / cm2_m_2, name='air_inf__m3_s_1')
        heat_tr_bldng_inf__W_K_1 = m.Intermediate(air_inf__m3_s_1 * air_room__J_m_3_K_1, name='heat_tr_bldng_inf__W_K_1')
        heat_loss_bldng_inf__W = m.Intermediate(heat_tr_bldng_inf__W_K_1 * delta_t_indoor_outdoor__K, name='heat_loss_bldng_inf__W')
    
        if learn_params is None or (property_sources['ventilation__dm3_s_1'] in df_learn.columns and df_learn[property_sources['ventilation__dm3_s_1']].notna().all()):
            ventilation__dm3_s_1 = m.MV(value=df_learn[property_sources['ventilation__dm3_s_1']].astype('float32').values, name='ventilation__dm3_s_1')
            ventilation__dm3_s_1.STATUS = 0  # No optimization
            ventilation__dm3_s_1.FSTATUS = 1  # Use the measured values
            
            air_changes_vent__s_1 = m.Intermediate(ventilation__dm3_s_1 / (bldng__m3 * dm3_m_3), name='air_changes_vent__s_1')
            heat_tr_bldng_vent__W_K_1 = m.Intermediate(air_changes_vent__s_1 * bldng__m3 * air_room__J_m_3_K_1, name='heat_tr_bldng_vent__W_K_1')
            heat_loss_bldng_vent__W = m.Intermediate(heat_tr_bldng_vent__W_K_1 * delta_t_indoor_outdoor__K, name='heat_loss_bldng_vent__W')
        else:
            heat_tr_bldng_vent__W_K_1 = 0
            heat_loss_bldng_vent__W = 0

        ##################################################################################################################
        ## Thermal inertia ##
        ##################################################################################################################
                    
        if learn_params is None:
            # Simulation mode: use the value from bldng_data
            th_inert_bldng__h = m.Param(value=bldng_data['learned_th_inert_bldng__h'], name='th_inert_bldng__h')
        else:
            # Learning mode: decide based on presence in learn_params
            if 'th_inert_bldng__h' in learn_params:
                # Learn thermal inertia
                th_inert_bldng__h = m.FV(value = param_hints['th_inert_bldng__h'], lb=(10), ub=(1000), name='th_inert_bldng__h')
                th_inert_bldng__h.STATUS = 1  # Allow optimization
                th_inert_bldng__h.FSTATUS = 1 # Use the initial value as a hint for the solver
            else:
                # Do not learn thermal inertia of the building, but use a fixed value based on hint
                th_inert_bldng__h = m.Param(value = param_hints['th_inert_bldng__h'], name='th_inert_bldng__h')
        
        ##################################################################################################################
        ### Heat balance ###
        ##################################################################################################################

        heat_gain_bldng__W = m.Intermediate(heat_dstr__W + heat_sol__W + heat_int__W, name='heat_gain_bldng__W')
        heat_loss_bldng__W = m.Intermediate(heat_loss_bldng_cond__W + heat_loss_bldng_inf__W + heat_loss_bldng_vent__W, name='heat_loss_bldng__W')
        heat_tr_bldng__W_K_1 = m.Intermediate(heat_tr_bldng_cond__W_K_1 + heat_tr_bldng_inf__W_K_1 + heat_tr_bldng_vent__W_K_1, name='heat_tr_bldng__W_K_1')
        th_mass_bldng__Wh_K_1  = m.Intermediate(heat_tr_bldng__W_K_1 * th_inert_bldng__h, name='th_mass_bldng__Wh_K_1') 
        m.Equation(temp_indoor__degC.dt() == ((heat_gain_bldng__W - heat_loss_bldng__W)  / (th_mass_bldng__Wh_K_1 * s_h_1)))

        ##################################################################################################################
        # Solve the model to start the learning process
        ##################################################################################################################
        
        if learn_params is None:
            m.options.IMODE = 4    # Do not learn, but only simulate using learned parameters passed via bldng_data
        else:
            m.options.IMODE = 5    # Learn one or more parameter values using Simultaneous Estimation 
        m.options.EV_TYPE = 2      # RMSE
        if max_iter is not None:   # retrict if needed to avoid waiting an eternity for unsolvable learning scenarios
            m.options.MAX_ITER = max_iter
            print(f"Solving restricted to at most {max_iter} iterations")
        # print(f"Start learning building model parameters for id {id} from {start} to {end} with duration {duration}")
        m.solve(disp=False)

        if m.options.APPSTATUS == 1:
            # print(f"A solution was found for id {id} from {start} to {end} with duration {duration}")
            
            ##################################################################################################################
            # Store results of the learning process
            ##################################################################################################################

            # Load results
            try:
                results = m.load_results()
                # DEBUG Save results to the local directory
                filename = 'gekko_results_building.json'
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=4)
                # print(f"Loaded results saved to {filename}")
            except AttributeError:
                results = None
                print("load_results() not available.")
            
            sim_arrays_mean = [
                'g_use_ch_hhv__W',
                'eta_ch_hhv__W0',
                'e_use_ch__W',
                'cop_ch__W0',
                'power_input_ch__W',
                'heat_sol__W',
                'heat_int__W',
                'heat_dstr__W',
                'heat_loss_bldng_cond__W', 
                'heat_loss_bldng_inf__W', 
                'heat_loss_bldng_vent__W',
                'delta_t_indoor_outdoor__K'
            ]
    
            if any(item is not None for item in [learn_params, predict_props, properties_mean, sim_arrays_mean]):
                # Initialize DataFrame for learned thermal parameters (only for learning mode)
                df_learned_parameters = pd.DataFrame({
                    'id': id, 
                    'start': start,
                    'end': end,
                    'duration': duration,
                }, index=[0])
            
                # Loop over the learn_params set and store learned values and calculate MAE if actual value is available
                for param in (learn_params - (predict_props or set())):
                    learned_value = results.get(param.lower(), [np.nan])[0]
                    df_learned_parameters.loc[0, f'learned_{param}'] = learned_value
                    # If actual value exists, compute MAE
                    if actual_params is not None and param in actual_params:
                        df_learned_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_params[param])
        
            if predict_props is not None:
                # Initialize a DataFrame for learned time-varying properties
                df_predicted_properties = pd.DataFrame(index=df_learn.index)
            
                # Store learned time-varying data in DataFrame and calculate MAE and RMSE
                current_locals = locals() # current_locals is valid in list comprehensions and for loops, locals() is not. 
                for prop in (predict_props or set()) & set(current_locals.keys()):
                    predicted_prop = f'predicted_{prop}'
                    df_predicted_properties.loc[:,predicted_prop] = np.asarray(current_locals[prop].value)
            
                    # If the property was measured, calculate and store MAE and RMSE
                    if prop in property_sources.keys() and property_sources[prop] in set(df_learn.columns):
                        df_learned_parameters.loc[0, f'mae_{prop}'] = mae(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
                        df_learned_parameters.loc[0, f'rmse_{prop}'] = rmse(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
                
            for prop in properties_mean:
                if property_sources[prop] in set(df_learn.columns):
                    # Determine the result column name based on whether the property ends with '__bool'
                    if prop.endswith('__bool'):
                        result_col = f"avg_{prop[:-6]}__0"  # Remove '__bool' and add '__0'
                        temp_column = df_learn[property_sources[prop]].fillna(False)  # Handle NA as False
                    else:
                        result_col = f"avg_{prop}"
                        temp_column = df_learn[property_sources[prop]]  # Use column directly
            
                    df_learned_parameters.loc[0, result_col] = temp_column.mean()
    
            current_locals = locals() # current_locals is valid in list comprehensions and for loops, locals() is not. 
            for prop in sim_arrays_mean:
                # Create variable names dynamically
                result_col = f"avg_{prop}"
                mean_value = np.asarray(results.get(prop.lower(), [np.nan])).mean()
                df_learned_parameters.loc[0, result_col] = mean_value
    
            # Calculate Carbon Case metrics
            df_learned_parameters.loc[0, 'avg_co2_ch__g_s_1'] = (
                (df_learned_parameters.loc[0, 'avg_g_use_ch_hhv__W'] 
                 * 
                 (co2_wtw_groningen_gas_std_nl_avg_2024__g__m_3 / gas_groningen_nl_avg_std_hhv__J_m_3)
                )
                +
                (df_learned_parameters.loc[0, 'avg_e_use_ch__W'] 
                 * 
                 co2_wtw_e_onbekend_nl_avg_2024__g__kWh_1
                )
            )
            
            if any(item is not None for item in [learn_params, predict_props, properties_mean, sim_arrays_mean]):
                # Set MultiIndex on the DataFrame (id, start, end)
                df_learned_parameters.set_index(['id', 'start', 'end', 'duration'], inplace=True)    

        else:
            df_learned_parameters = pd.DataFrame()
            print(f"NO Solution was found for id {id} from {start} to {end} with duration {duration}")


        m.cleanup()
    
        # Return both DataFrames: learned time-varying properties and learned fixed parameters
        return df_learned_parameters, df_predicted_properties


    # Define the modes as an enumeration
    class ControlMode(Enum):
        ALGORITHMIC = "alg"
        PID = "pid"   
        

    def boiler(
            df_learn,
            bldng_data: Dict = None,
            property_sources: Dict = None,
            param_hints: Dict = None,
            learn_params: Set[str] = {'fan_rotations_max_gain__pct_min_1',
                                      'error_threshold_delta_t_flow_flowset__K',
                                      'flow_dstr_pump_speed_max_gain__pct_min_1',
                                      'error_threshold_delta_t_flow_ret__K',
                                     },
            actual_params: Dict = None,
            predict_props: Set[str] = {'fan_speed__pct', 'flow_dstr_pump_speed__pct'},
            mode: ControlMode = ControlMode.ALGORITHMIC,
            max_iter=10,
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        
        id, start, end, step__s, duration__s  = Learner.get_time_info(df_learn) 
        duration = timedelta(seconds=duration__s)

        # Max fan gain in in %
        fan_scale = bldng_data['fan_max_ch_rotations__min_1'] - bldng_data['fan_min_ch_rotations__min_1']

        ##################################################################################################################
        # Initialize GEKKO model
        ##################################################################################################################
        m = GEKKO(remote=False)
        m.time = np.arange(0, duration__s, step__s)

        ##################################################################################################################
        # Flow setpoint
        ##################################################################################################################

        temp_flow_ch_max__degC  = m.MV(value=df_learn[property_sources['temp_flow_ch_max__degC']].astype('float32').values, name='temp_flow_ch_max__degC')
        temp_flow_ch_max__degC .STATUS = 0  # No optimization
        temp_flow_ch_max__degC .FSTATUS = 1 # Use the measured values
        
        # Initial assumption: fixed setpoint; will be relaxed later
        temp_flow_ch_set__degC = m.MV(value=df_learn[property_sources['temp_flow_ch_set__degC']].astype('float32').values, name='temp_flow_ch_set__degC')
        temp_flow_ch_set__degC.lower = 0  # Minimum value
        m.Equation(temp_flow_ch_set__degC <= temp_flow_ch_max__degC) # constraint to enforce the maximum limit dynamically

        ##################################################################################################################
        # Flow and return temperature
        ##################################################################################################################
        temp_flow_ch__degC = m.MV(value=df_learn[property_sources['temp_flow_ch__degC']].astype('float32').values, name='temp_flow_ch__degC')
        temp_flow_ch__degC.STATUS = 0  # No optimization
        temp_flow_ch__degC.FSTATUS = 1 # Use the measured values
        
        temp_ret_ch__degC = m.MV(value=df_learn[property_sources['temp_ret_ch__degC']].astype('float32').values, name='temp_ret_ch__degC')
        temp_ret_ch__degC.STATUS = 0  # No optimization
        temp_ret_ch__degC.FSTATUS = 1 # Use the measured values

        ##################################################################################################################
        # Fan speed and pump speed
        ##################################################################################################################

        # calculated fan speed percentage between min (0 %) and max (100 %)
        fan_speed__pct = m.CV(value=df_learn[property_sources['fan_speed__pct']].astype('float32').values, name='fan_speed__pct')
        fan_speed__pct.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
        fan_speed__pct.FSTATUS = 1 # Use the measured values
						
        # hydronic pump speed in % of maximum pump speed         
        flow_dstr_pump_speed__pct = m.CV(value=df_learn[property_sources['flow_dstr_pump_speed__pct']].astype('float32').values, name='flow_dstr_pump_speed__pct') 
        flow_dstr_pump_speed__pct.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
        flow_dstr_pump_speed__pct.FSTATUS = 1 # Use the measured values

        ##################################################################################################################
        # Control targets: flow temperature and 'delta-T': difference between flow and return temperature
        ##################################################################################################################

        # Error between supply temperature and setpoint fo the supply temperature
        error_delta_t_flow_flowset__K = m.Var(value=0.0, name='error_delta_t_flow_flowset__K')  # Initialize with a default value
        m.Equation(error_delta_t_flow_flowset__K == temp_flow_ch_set__degC - temp_flow_ch__degC)

        # Error in 'delta-T' (difference between supply and return temperature)
        desired_delta_t_flow_ret__K = m.Param(value=bldng_data['desired_delta_t_flow_ret__K'], name='desired_delta_t_flow_ret__K') 
        error_delta_t_flow_ret__K = m.Var(value=0.0, name='error_delta_t_flow_ret__K')  # Initialize with a default value
        m.Equation(error_delta_t_flow_ret__K == desired_delta_t_flow_ret__K - (temp_flow_ch__degC - temp_ret_ch__degC))
    
        ##################################################################################################################
        # Control  algorithm 
        ##################################################################################################################

        match mode:
            case Model.ControlMode.ALGORITHMIC:

                # TO DO: consider accepting param_hints for parameters that don't need do be learned
                
                # Define variables to hold the rate of fan and pump speed changes
                fan_rotations_gain__pct_min_1 = m.Var(value=0, name='fan_rotations_gain__pct_min_1')    # Rate of change for fan speed
                flow_dstr_pump_speed_gain__pct_min_1 = m.Var(value=0, name='flow_dstr_pump_speed_gain__pct_min_1')  # Rate of change for pump speed

                ##################################################################################################################
                # Algorithmic control 
                ##################################################################################################################
                
                ##################################################################################################################
                # Cooldown mode definitions 
                ##################################################################################################################

                # Temperature margins for cooldown hysteresis
                overheat_upper_margin_temp_flow__K = m.Param(value=5, name='overheat_upper_margin_temp_flow__K')                                        # Default overheating margin in K
                overheat_hysteresis__K = m.Param(value=bldng_data['overheat_hysteresis__K'], name='overheat_hysteresis__K')                 # Hysteresis, which might be boiler-specific
                cooldown_margin_temp_flow__K = overheat_hysteresis__K - overheat_upper_margin_temp_flow__K   # Default cooldown margin in K

                # Cooldown hysteresis: starts at crossing overheating margin, ends at crossing cooldown margin
                cooldown_condition = m.Var(value=0, name='cooldown_condition')  # Initialize hysteresis state variable

                
                # Define the overheating and cooldown conditions
                overheat_condition = temp_flow_ch__degC - (temp_flow_ch_set__degC + overheat_upper_margin_temp_flow__K)
                cooldown_exit_condition = (temp_flow_ch_set__degC + cooldown_margin_temp_flow__K) - temp_flow_ch__degC
                no_heat_demand_condition = 0.5 - temp_flow_ch_set__degC
                
                # Cooldown state transitions
                m.Equation(
                    cooldown_condition == m.if3(
                        overheat_condition,  # Enter cooldown mode if overheat condition is positive
                        1,                   # Cooldown mode active
                        m.if3(
                            cooldown_exit_condition, # Exit cooldown mode if this condition is positive
                            0,                       # Cooldown mode inactive
                            cooldown_condition       # Maintain current state (hysteresis)
                        )
                    )
                )

                ##################################################################################################################
                # Post-pump run definitions 
                ##################################################################################################################

                # Define post-pump run duration default: 3 minutes
                post_pump_run_duration__s = 3 * s_min_1
        
                # Boolean state for post-pump run condition
                in_post_pump_run_condition = m.Var(value=0, integer=True, name='in_post_pump_run_condition') 
        
                # Define a memory variable to store the previous value
                temp_flow_ch_set_prev__degC = m.Var(value=0, name='temp_flow_ch_set_prev__degC')
                
                # Differential equation to update the memory variable each timestep
                m.Equation(temp_flow_ch_set_prev__degC.dt() == temp_flow_ch_set__degC - temp_flow_ch_set_prev__degC)
        
                # Define the post-pump run entry condition
                post_pump_run_entry_condition = m.Var(value=0, name='post_pump_run_entry_condition')
        
                # Logical condition: (current == 0) & (delayed > 0)
                m.Equation(
                    post_pump_run_entry_condition == 
                    (temp_flow_ch_set__degC == 0) * (temp_flow_ch_set_prev__degC > 0)
                )

                # Create a post-pump run timer
                post_pump_run_timer = m.Var(value=0, name='post_pump_run_timer')  # Start at time 0
                m.Equation(post_pump_run_timer.dt() == in_post_pump_run_condition * step__s)  # Increment post_pump_run_timer by step__s seconds

                # Start post pump run timer (by calculating exporation time) whenever heat demand ends
                post_pump_run_expiration__s = m.Var(value=0, name='post_pump_run_expiration__s') # Expiration time
                
                m.Equation(
                    post_pump_run_expiration__s.dt() == m.if3(
                        post_pump_run_entry_condition,
                        post_pump_run_timer + post_pump_run_duration__s,    # Start expiration timer
                        0                                                   # Else: retain current expiration time
                    )
                )

                timer_not_expired_condition = m.Intermediate(
                    post_pump_run_timer - post_pump_run_expiration__s,
                    name='timer_not_expired_condition'
                )
                
                # Update in_post_pump_run_condition
                m.Equation(
                    in_post_pump_run_condition == m.if3(
                        timer_not_expired_condition,             # Timer not expired
                        1,                                       # Active
                        0                                        # Inactive
                    )
                )

                
                ##################################################################################################################
                # Fan speed definitions 
                ##################################################################################################################
                
                # Conditional fan speed gain based on flow error threshold, with an enforced maximum
                fan_rotations_gain__pct_min_1 = m.Var(name='fan_rotations_gain__pct_min_1', 
                                                      value=0)
                fan_rotations_gain__pct_min_1.upper=fan_rotations_max_gain__pct_min_1
                
                m.Equation(fan_rotations_gain__pct_min_1 == 
                           error_delta_t_flow_flowset__K / error_threshold_delta_t_flow_flowset__K * fan_rotations_max_gain__pct_min_1)
        
                m.Equation(fan_speed__pct.dt() == fan_rotations_gain__pct_min_1)
                
                # Override calculated fan speeds with 0 if in cooldown or when temp_flow_ch_set__degC is set to 0 
                m.Equation(
                    fan_speed__pct == m.if3(
                        no_heat_demand_condition,     # Condition: temp_flow_ch_set__degC == 0 (< 0.5)
                        0,                            # Action: set fan speed to 0
                        m.if3(                        # Else:
                            cooldown_condition,            # Condition: cooldown_condition == True
                            0,                             # Action: set fan speed also to 0
                            fan_speed__pct                 # Else: Keep the calculated fan speed
                        )
                    )
                )
                
                ##################################################################################################################
                # Pump speed definitions 
                ##################################################################################################################
        
                # Conditional pump speed gain based on error threshold, with en enforced maximum
                flow_dstr_pump_speed_gain__pct_min_1 = m.Var(
                    name='flow_dstr_pump_speed_gain__pct_min_1',
                    value=0)
                flow_dstr_pump_speed_gain__pct_min_1.upper=flow_dstr_pump_speed_max_gain__pct_min_1  # Enforce the max gain
                
                m.Equation(
                    flow_dstr_pump_speed_gain__pct_min_1 == 
                    error_delta_t_flow_ret__K / error_threshold_delta_t_flow_ret__K * flow_dstr_pump_speed_max_gain__pct_min_1
                )
        
                m.Equation(flow_dstr_pump_speed__pct.dt() == flow_dstr_pump_speed_gain__pct_min_1)
                
                m.Equation(
                    flow_dstr_pump_speed__pct == m.if3(
                        cooldown_condition,          # Condition: cooldown_condition == True
                        100,                         # Action: set pump speed to 100
                        m.if3(                       # Else:
                            in_post_pump_run_condition, # Condition: in_post_pump_run_condition == True
                            post_pump_speed__pct,       # Action:  use building-specific post-pump speed
                            m.if3(                      # Else:
                                no_heat_demand_condition,      # Condition: temp_flow_ch_set__degC == 0 (< 0.5)
                                0,                             # Action: set pump speed set to 0
                                flow_dstr_pump_speed__pct      # Else: retain the current pump speed
                            )
                        )
                    )
                )
        

        
            case Model.ControlMode.PID:
                ##################################################################################################################
                # PID control 
                ##################################################################################################################
                # Container to store the PID variables for fan and pump
                pid_parameters = {}
                
                # Loop over the components 
                for component in param_hints:
                    # Extract the bounds and param_hints for the current component
                    component_hints = param_hints[component]
                    pid_parameters[component] = {}  # Initialize section for this component
            
                    # Default values for PID terms
                    default_values = {'p': 1.0, 'i': 0.1, 'd': 0.05}
            
                    # Loop over the PID terms (p, i, d)
                    for term, default in default_values.items():
                        param_name = f'K{term}_{component}'
            
                        if term in component_hints:
                            # Create an adjustable FV for this term
                            param = m.FV(
                                value=component_hints[term]['initial_guess'],
                                lb=component_hints[term].get('lower_bound', None),
                                ub=component_hints[term].get('upper_bound', None),
                                name=param_name)
                            param.STATUS = 1  # Allow optimization
                            param.FSTATUS = 1  # Use the initial value as a hint for the solver
                        else:
                            # Create a fixed parameter for this term
                            param = m.Param(value=default, name=param_name)
            
                        # Store the parameter in the structured dictionary
                        pid_parameters[component][term] = param
            
               # PID control equations for fan speed
                m.Equation(
                    fan_speed__pct.dt() == (
                        pid_parameters['fan']['p'] * error_delta_t_flow_flowset__K +                   # Proportional term
                        pid_parameters['fan']['i'] * m.integral(error_delta_t_flow_flowset__K) +       # Integral term
                        pid_parameters['fan']['d'] * error_delta_t_flow_flowset__K.dt()                # Derivative term
                    )
                )

                # PID control equations for pump speed
                m.Equation(
                    flow_dstr_pump_speed__pct.dt() == (
                        pid_parameters['pump']['p'] * error_delta_t_flow_ret__K +                      # Proportional term
                        pid_parameters['pump']['i'] * m.integral(error_delta_t_flow_ret__K) +          # Integral term
                        pid_parameters['pump']['d'] * error_delta_t_flow_ret__K.dt()                   # Derivative term
                    )
                )

            case _:
                raise ValueError(f"Invalid ControlMode: {mode}")
    
        ##################################################################################################################
        # Solve the model to start the learning process
        ##################################################################################################################
        if learn_params is None:
            m.options.IMODE = 4    # Do not learn, but only simulate using learned parameters passed via bldng_data
        else:
            m.options.IMODE = 5    # Learn one or more parameter values using Simultaneous Estimation 
        m.options.EV_TYPE = 2      # RMSE
        if max_iter is not None:   # retrict if needed to avoid waiting an eternity for unsolvable learning scenarios
            m.options.MAX_ITER = max_iter
        m.solve(disp=False)

        ##################################################################################################################
        # Store results of the learning process
        ##################################################################################################################
        
        if m.options.APPSTATUS == 1:
            
            # print(f"A solution was found for id {id} from {start} to {end} with duration {duration}")

            # Load results
            try:
                results = m.load_results()
                # DEBUG Save results to the local directory
                filename = 'gekko_results_boiler.json'
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=4)
            except AttributeError:
                results = None
                # print("load_results() not available.")
            
            if any(item is not None for item in [learn_params, predict_props]):
                # Initialize DataFrame for learned thermal parameters (only for learning mode)
                df_learned_parameters = pd.DataFrame({
                    'id': id, 
                    'start': start,
                    'end': end,
                    'duration': duration,
                }, index=[0])
            
            # Loop over the learn_params set and store learned values and calculate MAE if actual value is available
            for param in (learn_params - (predict_props or set())):
                learned_value = results.get(param.lower(), [np.nan])[0]
                print(f"results.get{param.lower()}, [np.nan])[0]: {learned_value}")
                df_learned_parameters.loc[0, f'learned_{param}'] = learned_value
                # If actual value exists, compute MAE
                if actual_params is not None and param in actual_params:
                    df_learned_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_params[param])
    
            if predict_props is not None:
                # Initialize a DataFrame for learned time-varying properties
                df_predicted_properties = pd.DataFrame(index=df_learn.index)
            
                # Store learned time-varying data in DataFrame and calculate MAE and RMSE
                for prop in (predict_props or set()):
                    predicted_prop = f'predicted_{mode.value}_{prop}'
                    df_predicted_properties.loc[:,predicted_prop] = results.get(prop.lower(), [np.nan])
                    
                    # If the property was measured, calculate and store MAE and RMSE
                    if prop in property_sources.keys() and property_sources[prop] in set(df_learn.columns):
                        df_learned_parameters.loc[0, f'mae_{mode.value}_{prop}'] = mae(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
                        df_learned_parameters.loc[0, f'rmse_{mode.value}_{prop}'] = rmse(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
                    
                
                    # Loop over the learn_params set and store learned values and calculate MAE if actual value is available
                    for param in (learn_params - (predict_props or set())):
                        learned_value = results.get(param.lower(), [np.nan])[0]
                        df_learned_parameters.loc[0, f'learned_{param}'] = learned_value
                        # If actual value exists, compute MAE
                        if actual_params is not None and param in actual_params:
                            df_learned_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_params[param])
        
                    # If the property was measured, calculate and store MAE and RMSE
                    if prop in property_sources.keys() and property_sources[prop] in set(df_learn.columns):
                        df_learned_parameters.loc[0, f'mae_{prop}'] = mae(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
                        df_learned_parameters.loc[0, f'rmse_{prop}'] = rmse(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
            
            if any(item is not None for item in [learn_params, predict_props]):
                # Set MultiIndex on the DataFrame (id, start, end)
                df_learned_parameters.set_index(['id', 'start', 'end', 'duration'], inplace=True)

        else:
            df_learned_parameters = pd.DataFrame()
            print(f"NO Solution was found for id {id} from {start} to {end} with duration {duration}")
        
        m.cleanup()

        return df_learned_parameters, df_predicted_properties
        

    def thermostat(
            df_learn,
            bldng_data: Dict = None,
            property_sources: Dict = None,
            param_hints: Dict = None,
            learn_params: Set[str] = {'thermostat_hysteresis__K'},
            actual_params: Dict = None,
            predict_props: Set[str] = {'temp_flow_ch_set__degC'},
            mode: ControlMode = ControlMode.ALGORITHMIC,
            max_iter=10,
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        id, start, end, step__s, duration__s  = Learner.get_time_info(df_learn) 
        duration = timedelta(seconds=duration__s)

        # TO DO: Check whether we need to use something from bldng_data

        ##################################################################################################################
        # Initialize GEKKO model
        ##################################################################################################################
        m = GEKKO(remote=False)
        m.time = np.arange(0, duration__s, step__s)

        ##################################################################################################################
        # Flow setpoint
        ##################################################################################################################

        temp_flow_ch_max__degC  = m.MV(value=df_learn[property_sources['temp_flow_ch_max__degC']].astype('float32').values, name='temp_flow_ch_max__degC')
        temp_flow_ch_max__degC .STATUS = 0  # No optimization
        temp_flow_ch_max__degC .FSTATUS = 1 # Use the measured values
        
        # Initial assumption: fixed setpoint; will be relaxed later
        temp_flow_ch_set__degC = m.Var(value=0, name='temp_flow_ch_set__degC')
        temp_flow_ch_set__degC.lower = 0  # Minimum value
        m.Equation(temp_flow_ch_set__degC <= temp_flow_ch_max__degC) # constraint to enforce the maximum limit dynamically

        ##################################################################################################################
        # Setpoint and indoor temperature
        ##################################################################################################################
        temp_set__degC = m.MV(value=df_learn[property_sources['temp_set__degC']].astype('float32').values, name='temp_set__degC')
        temp_set__degC.STATUS = 0  # No optimization
        temp_set__degC.FSTATUS = 1 # Use the measured values
        
        temp_indoor__degC = m.MV(value=df_learn[property_sources['temp_indoor__degC']].astype('float32').values, name='temp_indoor__degC')
        temp_indoor__degC.STATUS = 0  # No optimization
        temp_indoor__degC.FSTATUS = 1 # Use the measured values

        ##################################################################################################################
        # Control targets: 'delta-T': difference between indoor setpoint and indoor temperature
        ##################################################################################################################

        # Error between thermostat setpoint and indoor temperature
        error_delta_t_indoor_set__K  = m.Var(value=0.0, name='error_delta_t_indoor_set__K')  # Initialize with a default value
        m.Equation(error_delta_t_indoor_set__K == temp_set__degC - temp_indoor__degC)

        ##################################################################################################################
        # Control  algorithm 
        ##################################################################################################################

        match mode:
            case Model.ControlMode.ALGORITHMIC:

                ##################################################################################################################
                # Algorithmic control; this implements a simple ON/OFF thermostat with hysteresis
                ##################################################################################################################
        
                # Thermostat hysteresis
                thermostat_hysteresis__K = m.FV(value=0.1,
                                               lb=0.0,
                                               ub=2.0, name='thermostat_hysteresis__K')         # Thermostat hysteresis, which might be boiler-specific
                thermostat_hysteresis__K.STATUS = 1            # Allow optimization
                thermostat_hysteresis__K.FSTATUS = 1           # Use the initial value as a hint for the solver
                
                hysteresis_upper_margin__K = m.Intermediate(temp_set__degC + thermostat_hysteresis__K/2, name='hysteresis_upper_margin__K')
                hysteresis_lower_margin__K = m.Intermediate(temp_set__degC - thermostat_hysteresis__K/2, name='hysteresis_lower_margin__K')
        
                m.Equation(
                    temp_flow_ch_set__degC == m.if3(
                        temp_indoor__degC - hysteresis_upper_margin__K,     # Positive if above upper margin (OFF)
                        0,                                                  # turn heating OFF
                        m.if3(
                            hysteresis_lower_margin__K - temp_indoor__degC, # Positive if below lower margin (ON)
                            temp_flow_ch_max__degC,                         # turn heating ON
                            temp_flow_ch_set__degC                          # Maintain current state
                        )
                    )
                )

            case Model.ControlMode.PID:
                ##################################################################################################################
                # PID control 
                ##################################################################################################################
                # Container to store the PID variables
                pid_parameters = {}
                
                # Define default PID terms
                default_pid_values = {'p': 1.0, 'i': 0.1, 'd': 0.05}
                
                # PID parameter initialization
                for component, component_hints in param_hints.items():
                    pid_parameters[component] = {}
                    for term, default in default_pid_values.items():
                        param_name = f'K{term}_{component}'
                        bounds = component_hints.get(term, {})
                        pid_parameters[component][term] = m.FV(
                            value=bounds.get('initial_guess', default),
                            lb=bounds.get('lower_bound', None),
                            ub=bounds.get('upper_bound', None),
                            name='pid_parameters')
                        pid_parameters[component][term].STATUS = 1
                        pid_parameters[component][term].FSTATUS = 1
            
                # PID control equations for flow temperature setpoint 
                m.Equation(
                    temp_flow_ch_set__degC.dt() == (
                        pid_parameters['thermostat']['p'] * error_delta_t_indoor_set__K +              # Proportional term
                        pid_parameters['thermostat']['i'] * m.integral(error_delta_t_indoor_set__K) +  # Integral term
                        pid_parameters['thermostat']['d'] * error_delta_t_indoor_set__K.dt()           # Derivative term
                    )
                )

            case _:
                raise ValueError(f"Invalid ControlMode: {mode}")
    
        ##################################################################################################################
        # Solve the model to start the learning process
        ##################################################################################################################
        if learn_params is None:
            m.options.IMODE = 4    # Do not learn, but only simulate using learned parameters passed via bldng_data
        else:
            m.options.IMODE = 5    # Learn one or more parameter values using Simultaneous Estimation 
        m.options.EV_TYPE = 2      # RMSE
        if max_iter is not None:   # retrict if needed to avoid waiting an eternity for unsolvable learning scenarios
            m.options.MAX_ITER = max_iter
        m.solve(disp=False)

        ##################################################################################################################
        # Store results of the learning process
        ##################################################################################################################
        
        if m.options.APPSTATUS == 1:
            
            # print(f"A solution was found for id {id} from {start} to {end} with duration {duration}")

            # Load results
            try:
                results = m.load_results()
                # DEBUG Save results to the local directory
                filename = 'gekko_results_therm.json'
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=4)
            except AttributeError:
                results = None
                print("load_results() not available.")
            
            if any(item is not None for item in [learn_params, predict_props]):
                # Initialize DataFrame for learned thermal parameters (only for learning mode)
                df_learned_parameters = pd.DataFrame({
                    'id': id, 
                    'start': start,
                    'end': end,
                    'duration': duration,
                }, index=[0])
            
            # Loop over the learn_params set and store learned values and calculate MAE if actual value is available
            for param in (learn_params - (predict_props or set())):
                learned_value = results.get(param.lower(), [np.nan])[0]
                print(f"results.get{param.lower()}, [np.nan])[0]: {learned_value}")
                df_learned_parameters.loc[0, f'learned_{param}'] = learned_value
                # If actual value exists, compute MAE
                if actual_params is not None and param in actual_params:
                    df_learned_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_params[param])
    
            if predict_props is not None:
                # Initialize a DataFrame for learned time-varying properties
                df_predicted_properties = pd.DataFrame(index=df_learn.index)
            
                # Store learned time-varying data in DataFrame and calculate MAE and RMSE
                for prop in (predict_props or set()):
                    predicted_prop = f'predicted_{mode.value}_{prop}'
                    # print(f"for {predicted_prop}: results.get({prop.lower()}, [np.nan]) = {results.get(prop.lower(), [np.nan])}")

                    df_predicted_properties.loc[:,predicted_prop] = results.get(prop.lower(), [np.nan])
                    
                    # If the property was measured, calculate and store MAE and RMSE
                    if prop in property_sources.keys() and property_sources[prop] in set(df_learn.columns):
                        df_learned_parameters.loc[0, f'mae_{mode.value}_{prop}'] = mae(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
                        df_learned_parameters.loc[0, f'rmse_{mode.value}_{prop}'] = rmse(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
                    
                
                    # Loop over the learn_params set and store learned values and calculate MAE if actual value is available
                    for param in (learn_params - (predict_props or set())):
                        learned_value = results.get(param.lower(), [np.nan])[0]
                        df_learned_parameters.loc[0, f'learned_{param}'] = learned_value
                        # If actual value exists, compute MAE
                        if actual_params is not None and param in actual_params:
                            df_learned_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_params[param])
        
                    # If the property was measured, calculate and store MAE and RMSE
                    if prop in property_sources.keys() and property_sources[prop] in set(df_learn.columns):
                        df_learned_parameters.loc[0, f'mae_{prop}'] = mae(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
                        df_learned_parameters.loc[0, f'rmse_{prop}'] = rmse(
                            df_learn[property_sources[prop]],  # Measured values
                            df_predicted_properties[predicted_prop]  # Predicted values
                        )
            
            if any(item is not None for item in [learn_params, predict_props]):
                # Set MultiIndex on the DataFrame (id, start, end)
                df_learned_parameters.set_index(['id', 'start', 'end', 'duration'], inplace=True)

        else:
            df_learned_parameters = pd.DataFrame()
            print(f"NO Solution was found for id {id} from {start} to {end} with duration {duration}")
        
        m.cleanup()

        return df_learned_parameters, df_predicted_properties



class Comfort():

    
    # Function to find the comfort zone
    def comfort_zone(
        rel_humidity__pct: float = 50,                     # Relative humidity [%]
        airflow__m_s_1: float = 0.1,                       # Airflow [m/s]
        metabolic_rate__MET: float = sedentary__MET,       # Metabolic rate [MET]
        clothing_insulation__clo: float = 1.0,             # Clothing insulation [clo]
        target_ppd__pct: float = 10                        # Target PPD [%]
    ) -> tuple[float, float, float]:
        """
        Calculate the comfort zone based on the PMV (Predicted Mean Vote) and PPD (Predicted Percentage Dissatisfied) indices.
    
        This function finds the range of indoor temperatures that maintain a PPD below or equal to the specified target
        percentage, while also determining the neutral temperature where the PMV is closest to zero.
    
        Parameters:
        -----------
        rel_humidity__pct : float
            Relative humidity in percentage (default is 50%).
        airflow__m_s_1 : float
            Airflow in meters per second (default is 0.1 m/s).
        metabolic_rate__MET : float
            Metabolic rate in met (default is sedentary__MET, which corresponds to 1.2 MET).
        clothing_insulation__clo : float
            Clothing insulation in clo (default is 1.0 clo).
        target_ppd__pct : float
            Target PPD percentage (default is 10%).
    
        Returns:
        --------
        tuple[float, float, float]
            A tuple containing:
            - lower_bound__degC: The lower temperature bound where PPD is acceptable.
            - upper_bound__degC: The upper temperature bound where PPD is acceptable.
            - neutral__degC: The temperature where PMV is closest to zero.
        """
        
        lower_bound__degC = None
        upper_bound__degC = None
        neutral__degC = None
    
        # Assume dry-bulb temperature = mean radiant temperature and loop over 15°C to 30°C range with 0.1°C precision
        for temp_indoor__degC in np.arange(15.0, 30.0, 0.1):
            result = pmv_ppd(
                tdb=temp_indoor__degC,
                tr=temp_indoor__degC,
                vr=airflow__m_s_1,
                rh=rel_humidity__pct,
                met=metabolic_rate__MET,
                clo=clothing_insulation__clo,
                wme=0  # External work, typically 0
            )
    
            pmv__0 = result['pmv']
            ppd__pct = result['ppd']
    
            # Track the neutral temperature where PMV is closest to 0
            if neutral__degC is None and -0.05 <= pmv__0 <= 0.05:  # PMV ~ 0, small tolerance
                neutral__degC = temp_indoor__degC
    
            # Find the bounds where PPD is within the target (i.e., less than or equal to 10%)
            if ppd__pct <= target_ppd__pct:
                if lower_bound__degC is None:
                    lower_bound__degC = temp_indoor__degC  # First temp where PPD is acceptable
                upper_bound__degC = temp_indoor__degC  # Last temp where PPD is still acceptable
    
        return lower_bound__degC, upper_bound__degC, neutral__degC    

    
    def comfort_margins(target_ppd__pct: float = 10) -> tuple[float, float]:
        """
        Calculate the overheating and underheating margins based on the comfort zone.
        
        Parameters:
        -----------
        target_ppd__pct : float
            Target PPD percentage (default is 10%).
        
        Returns:
        --------
        tuple[float, float]
            A tuple containing:
            - underheating_margin__K: The underheating margin in degrees Celsius.
            - overheating_margin__K: The overheating margin in degrees Celsius.
        """
        comfortable_temp_indoor_min__degC, comfortable_temp_indoor_max__degC, comfortable_temp_indoor_ideal__degC = Comfort.comfort_zone(target_ppd__pct)
    
        # Calculate margins
        if comfortable_temp_indoor_ideal__degC is not None and comfortable_temp_indoor_min__degC is not None and comfortable_temp_indoor_max__degC is not None:
            overheating_margin__K = comfortable_temp_indoor_max__degC - comfortable_temp_indoor_ideal__degC
            underheating_margin__K = comfortable_temp_indoor_ideal__degC - comfortable_temp_indoor_min__degC
            return underheating_margin__K, overheating_margin__K
        else:
            return None, None
   

    def is_comfortable(
        temp_indoor__degC,  
        temp_set__degC: float = None,  
        target_ppd__pct: float = 10,
        occupancy__bool: bool = True  # This can be a bool or pd.Series
    ) -> pd.Series | bool:
        """
        Check if the indoor temperature is comfortable based on setpoint and comfort margins,
        and whether occupancy is True or False. Works with both single values and Series.
        """
        
        # Convert single values to Series for uniform processing
        if isinstance(temp_indoor__degC, (float, int)):
            temp_indoor__degC = pd.Series([temp_indoor__degC])
        if isinstance(temp_set__degC, (float, int)):
            temp_set__degC = pd.Series([temp_set__degC])
        if isinstance(occupancy__bool, bool):
            occupancy__bool = pd.Series([occupancy__bool])
    
        # Ensure all Series have the same index
        temp_indoor__degC = temp_indoor__degC.reindex(temp_set__degC.index)
        occupancy__bool = occupancy__bool.reindex(temp_set__degC.index)
    
        # Handle the case where no setpoint is provided
        if temp_set__degC is not None:
            # Calculate margins based on the provided setpoint
            underheating_margin__K, overheating_margin__K = Comfort.comfort_margins(target_ppd__pct)
    
            if underheating_margin__K is not None and overheating_margin__K is not None:
                lower_bound__degC = temp_set__degC - underheating_margin__K
                upper_bound__degC = temp_set__degC + overheating_margin__K
                
                # Initialize the result to True for all rows (assuming comfort)
                result = pd.Series(True, index=temp_indoor__degC.index)
                
                # Apply the comfort logic only where occupancy__bool is True (someone is home)
                result = result.where(occupancy__bool == False,  # If unoccupied (False), keep True
                                      (temp_indoor__degC >= lower_bound__degC) & (temp_indoor__degC <= upper_bound__degC))  # If occupied (True), check temperature
                return result.where(~(temp_indoor__degC.isna() | temp_set__degC.isna()), pd.NA)  # Handle NaNs
    
        # If no setpoint is provided, use comfort_zone to check comfort
        comfortable_temp_indoor_min__degC, comfortable_temp_indoor_max__degC, _ = Comfort.comfort_zone(target_ppd__pct)
        
        if comfortable_temp_indoor_min__degC is not None and comfortable_temp_indoor_max__degC is not None:
            result = pd.Series(True, index=temp_indoor__degC.index)  # Start with all True for unoccupied
            result = result.where(occupancy__bool == False,  # If unoccupied (False), keep True
                                  (temp_indoor__degC >= comfortable_temp_indoor_min__degC) & 
                                  (temp_indoor__degC <= comfortable_temp_indoor_max__degC))  # If occupied (True), check temperature
            return result.where(~temp_indoor__degC.isna(), pd.NA)  # Handle NaNs
        
        return pd.Series(pd.NA, index=temp_indoor__degC.index)  # Return <NA> Series if comfort cannot be determined

    
class Simulator():
    def integrated_model(
            df_learn,
            bldng_data: Dict = None,
            boiler_efficiency_hhv: Callable[[float, float], float] = None, 
            property_sources: Dict = None,
            param_hints: Dict = None,
            learn_params: Set[str] = None,
            actual_params: Dict = None,
            predict_props: Set[str] = {'temp_indoor__degC'},
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        # retrieve building-specific constants
        bldng__m3 = bldng_data['bldng__m3']
        usable_area__m2 = bldng_data['usable_area__m2']

        # retrieve thermostat-specific constants
        thermostat_hysteresis__K = bldng_data['thermostat_hysteresis__K']
    
        # retrieve boiler-specific constants
        fan_min_ch_rotations__min_1 = bldng_data['fan_min_ch_rotations__min_1']
        fan_max_ch_rotations__min_1 = bldng_data['fan_max_ch_rotations__min_1']
        Qnh_min_lhv__kW = bldng_data['Qnh_min_lhv__kW']
        Qnh_max_lhv__kW = bldng_data['Qnh_max_lhv__kW']
        overheat_hysteresis__K = bldng_data['overheat_hysteresis__K']
        desired_delta_t_flow_ret__K = bldng_data['desired_delta_t_flow_ret__K']
        post_pump_speed__pct = bldng_data['post_pump_run__pct']
        fan_rotations_max_gain__pct_min_1 = bldng_data['fan_rotations_max_gain__pct_min_1']
        error_threshold_delta_t_flow_flowset__K = bldng_data['error_threshold_delta_t_flow_flowset__K']
        flow_dstr_pump_speed_max_gain__pct_min_1 = bldng_data['flow_dstr_pump_speed_max_gain__pct_min_1']
        error_threshold_delta_t_flow_ret__K = bldng_data['error_threshold_delta_t_flow_ret__K']
    

        used_params = {
            'bldng__m3',
            'usable_area__m2',
            'thermostat_hysteresis__K',
            'fan_max_ch_rotations__min_1',
            'fan_min_ch_rotations__min_1', 
            'Qnh_min_lhv__kW',
            'Qnh_max_lhv__kW',
            'overheat_hysteresis__K', 
            'desired_delta_t_flow_ret__K',
            'post_pump_speed__pct',
            'error_threshold_delta_t_flow_flowset__K',
            'flow_dstr_pump_speed_max_gain__pct_min_1',
            'error_threshold_delta_t_flow_ret__K',
            'learned_heat_tr_dstr__W_K_1',
            'learned_th_mass_dstr__Wh_K_1',
            'learned_heat_tr_bldng_cond__W_K_1',
            'learned_th_inert_bldng__h',
            'learned_aperture_sol__m2',
            'learned_aperture_inf__cm2',
            'flow_dstr_slope__dm3_s_1_pct_1',
            'flow_dstr_intercept__dm3_s_1',
        }

        ##################################################################################################################
        # Initialize GEKKO model
        ##################################################################################################################
        m = GEKKO(remote=False)
        id, start, end, step__s, duration__s  = Learner.get_time_info(df_learn) 
        m.time = np.arange(0, duration__s, step__s)

        ##################################################################################################################
        # Indoor Temperature
        ##################################################################################################################

        # Initial temperature equal to measured temperature
        temp_indoor__degC = m.Var(value=np.float32(df_learn[property_sources['temp_indoor__degC']].iloc[0]), name='temp_indoor__degC')

        ##################################################################################################################
        # Indoor Temperature Setpoint
        ##################################################################################################################
        temp_set__degC = m.MV(value=df_learn[property_sources['temp_set__degC']].astype('float32').values, name='temp_set__degC')
        temp_set__degC.STATUS = 0  # No optimization
        temp_set__degC.FSTATUS = 1 # Use the measured values

        ##################################################################################################################
        # Flow Temperature Setpoint
        ##################################################################################################################

        temp_flow_ch_max__degC  = m.MV(value=df_learn[property_sources['temp_flow_ch_max__degC']].astype('float32').values, name='temp_flow_ch_max__degC')
        temp_flow_ch_max__degC.STATUS = 0  # No optimization
        temp_flow_ch_max__degC.FSTATUS = 1 # Use the measured values
        
        temp_flow_ch_set__degC = m.Var(value=np.float32(0.0), name='temp_flow_ch_set__degC') 
        temp_flow_ch_set__degC.lower = 0  # Minimum value
        m.Equation(temp_flow_ch_set__degC <= temp_flow_ch_max__degC) # constraint to enforce the maximum limit dynamically

        ##################################################################################################################
        # Control targets: 'delta-T': difference between indoor setpoint and indoor temperature
        ##################################################################################################################

        # Error between thermostat setpoint and indoor temperature
        error_delta_t_indoor_set__K = m.Var(value=np.float32(0.0), name='error_delta_t_indoor_set__K')  # Initialize with a default value
        m.Equation(error_delta_t_indoor_set__K == temp_set__degC - temp_indoor__degC)

        ##################################################################################################################
        # Algorithmic control; this implements a simple ON/OFF thermostat with hysteresis
        ##################################################################################################################

        hysteresis_upper_margin__K = m.Intermediate(temp_set__degC + thermostat_hysteresis__K/2, name='hysteresis_upper_margin__K')
        hysteresis_lower_margin__K = m.Intermediate(temp_set__degC - thermostat_hysteresis__K/2, name='hysteresis_lower_margin__K')

        m.Equation(
            temp_flow_ch_set__degC == m.if3(
                temp_indoor__degC - hysteresis_upper_margin__K,     # Positive if above upper margin (OFF)
                0,                                                  # turn heating OFF
                m.if3(
                    hysteresis_lower_margin__K - temp_indoor__degC, # Positive if below lower margin (ON)
                    temp_flow_ch_max__degC,                         # turn heating ON
                    temp_flow_ch_set__degC                          # Maintain current state
                )
            )
        )

        ##################################################################################################################
        # Fan speed and pump speed
        ##################################################################################################################

        # calculated fan speed percentage between min (0 %) and max (100 %); use initial value during simulations
        fan_speed__pct = m.Var(value=np.float32((df_learn[property_sources['fan_rotations__min_1']].iloc[0] - fan_min_ch_rotations__min_1)
                                                /(fan_max_ch_rotations__min_1 - fan_min_ch_rotations__min_1)
                                                * 100),
                               name='fan_speed__pct')

        # hydronic pump speed in % of maximum pump speed; use initial value during simulations         
        flow_dstr_pump_speed__pct = m.Var(value=np.float32(df_learn[property_sources['flow_dstr_pump_speed__pct']].iloc[0]), name='flow_dstr_pump_speed__pct') 

        ##################################################################################################################
        # Flow and return temperature
        ##################################################################################################################
        if (pd.notna(df_learn[property_sources['temp_flow_ch__degC']].iloc[0]) and 
            pd.notna(df_learn[property_sources['temp_ret_ch__degC']].iloc[0])
        ):
            # Estimate based on initial supply and return temperature
            initial_temp_flow_ch__degC = np.float32(df_learn[property_sources['temp_flow_ch__degC']].iloc[0])
            initial_temp_ret_ch__degC = np.float32(df_learn[property_sources['temp_ret_ch__degC']].iloc[0])
        else:
            # We're not in the middle of a heat generation streak, so we estimate based on indoor temperature
            initial_temp_flow_ch__degC = np.float32(df_learn[property_sources['temp_indoor__degC']].iloc[0])
            initial_temp_ret_ch__degC = initial_temp_flow_ch__degC

        initial_temp_dstr__degC = (initial_temp_flow_ch__degC + initial_temp_ret_ch__degC) / 2
            
        # Define variables for the dynamic heat distribution model
        temp_flow_ch__degC = m.Var(value=initial_temp_flow_ch__degC, lb=0.0, ub=100.0, name='temp_flow_ch__degC')
        temp_ret_ch__degC = m.Var(value=initial_temp_ret_ch__degC, lb=0.0, ub=100.0, name='temp_ret_ch__degC')
        temp_dstr__degC = m.Var(value=initial_temp_dstr__degC, lb=0.0, ub=100.0, name='temp_dstr__degC')
            
        ##################################################################################################################
        # Control targets: flow temperature and 'delta-T': difference between flow and return temperature
        ##################################################################################################################

        # Error between supply temperature and setpoint fo the supply temperature
        error_delta_t_flow_flowset__K = m.Var(value=np.float32(0.0), name='error_delta_t_flow_flowset__K')  # Initialize with a default value
        m.Equation(error_delta_t_flow_flowset__K == temp_flow_ch_set__degC - temp_flow_ch__degC)

        # Error in 'delta-T' (difference between supply and return temperature)
        error_delta_t_flow_ret__K = m.Var(value=np.float32(0.0), name='error_delta_t_flow_ret__K')  # Initialize with a default value
        m.Equation(error_delta_t_flow_ret__K == desired_delta_t_flow_ret__K - (temp_flow_ch__degC - temp_ret_ch__degC))
    
        ##################################################################################################################
        # Boiler Control algorithm 
        ##################################################################################################################

        # Define variables to hold the rate of fan and pump speed changes
        # Rate of change for fan speed
        fan_rotations_gain__pct_min_1 = m.Var(value=np.float32(0.0), name='fan_rotations_gain__pct_min_1')
        # Rate of change for pump speed
        flow_dstr_pump_speed_gain__pct_min_1 = m.Var(value=np.float32(0.0),name='flow_dstr_pump_speed_gain__pct_min_1')

        ##################################################################################################################
        # Cooldown mode definitions 
        ##################################################################################################################

        # Temperature margins for cooldown hysteresis
        overheat_upper_margin_temp_flow__K = np.float32(5.0)                                         # Overheating margin in K
        cooldown_margin_temp_flow__K = overheat_hysteresis__K - overheat_upper_margin_temp_flow__K   # Default cooldown margin in K

        # Cooldown hysteresis: starts at crossing overheating margin, ends at crossing cooldown margin
        cooldown_condition = m.Var(value=0, name='cooldown_condition')  # Initialize hysteresis state variable
        
        # Define the overheating and cooldown conditions
        overheat_condition = temp_flow_ch__degC - (temp_flow_ch_set__degC + overheat_upper_margin_temp_flow__K)
        cooldown_exit_condition = (temp_flow_ch_set__degC + cooldown_margin_temp_flow__K) - temp_flow_ch__degC
        no_heat_demand_condition = 0.5 - temp_flow_ch_set__degC
        
        # Cooldown state transitions
        m.Equation(
            cooldown_condition == m.if3(
                overheat_condition,  # Enter cooldown mode if overheat condition is positive
                1,                # Cooldown mode active
                m.if3(
                    cooldown_exit_condition,  # Exit cooldown mode if this condition is positive
                    0,                   # Cooldown mode inactive
                    cooldown_condition       # Maintain current state (hysteresis)
                )
            )
        )

        ##################################################################################################################
        # Post-pump run definitions 
        ##################################################################################################################
        
        # Define post-pump run duration
        post_pump_run_duration__s = 3 * s_min_1                # Post pump run duration (in minutes); default to 3 minutes

        # Boolean state for post-pump run condition
        in_post_pump_run_condition = m.Var(value=0, integer=True, name='in_post_pump_run_condition') 

        # Define a memory variable to store the previous value
        temp_flow_ch_set_prev__degC = m.Var(value=0, name='temp_flow_ch_set_prev__degC')
        
        # Differential equation to update the memory variable each timestep
        m.Equation(temp_flow_ch_set_prev__degC.dt() == temp_flow_ch_set__degC - temp_flow_ch_set_prev__degC)

        # Define the post-pump run entry condition
        post_pump_run_entry_condition = m.Var(value=0, name='post_pump_run_entry_condition')

        # Logical condition: (current == 0) & (delayed > 0)
        m.Equation(
            post_pump_run_entry_condition == 
            (temp_flow_ch_set__degC == 0) * (temp_flow_ch_set_prev__degC > 0)
        )

        # Timer variable to track post-pump run duration
        post_pump_run_timer = m.Var(value=0, name='post_pump_run_timer')
        
        # Scale the timer increment by step__s to match the custom time steps
        m.Equation(post_pump_run_timer.dt() == in_post_pump_run_condition * step__s)

        # Start post pump run timer (by calculating exporation time) whenever heat demand ends
        post_pump_run_expiration__s = m.Var(value=0, name='post_pump_run_expiration__s') # Expiration time
        m.Equation(
            post_pump_run_expiration__s.dt() == m.if3(
                post_pump_run_entry_condition,
                post_pump_run_timer + post_pump_run_duration__s,    # Start expiration timer
                0                                                   # Else: retain current expiration time
            )
        )

        timer_not_expired_condition = m.Intermediate(
            post_pump_run_timer - post_pump_run_expiration__s,
            name='timer_not_expired_condition'
        )
        
        # Update in_post_pump_run_condition
        m.Equation(
            in_post_pump_run_condition == m.if3(
                timer_not_expired_condition,        # Timer not expired
                1,                                  # Active
                0                                   # Inactive
            )
        )
        
        ##################################################################################################################
        # Fan speed definitions 
        ##################################################################################################################
        
        # Conditional fan speed gain based on flow error threshold, with an enforced maximum
        fan_rotations_gain__pct_min_1 = m.Var(name='fan_rotations_gain__pct_min_1', 
                                              value=0)
        fan_rotations_gain__pct_min_1.upper=fan_rotations_max_gain__pct_min_1
        
        m.Equation(fan_rotations_gain__pct_min_1 == 
                   error_delta_t_flow_flowset__K / error_threshold_delta_t_flow_flowset__K * fan_rotations_max_gain__pct_min_1)

        m.Equation(fan_speed__pct.dt() == fan_rotations_gain__pct_min_1)
        
        # Override calculated fan speeds with 0 if in cooldown or when temp_flow_ch_set__degC is set to 0 
        m.Equation(
            fan_speed__pct == m.if3(
                no_heat_demand_condition,     # Condition: temp_flow_ch_set__degC == 0 (< 0.5)
                0,                            # Action: set fan speed to 0
                m.if3(                        # Else:
                    cooldown_condition,            # Condition: cooldown_condition == True
                    0,                             # Action: set fan speed also to 0
                    fan_speed__pct                 # Else: Keep the calculated fan speed
                )
            )
        )
        
        ##################################################################################################################
        # Pump speed definitions 
        ##################################################################################################################

        # Conditional pump speed gain based on error threshold, with en enforced maximum
        flow_dstr_pump_speed_gain__pct_min_1 = m.Var(
            name='flow_dstr_pump_speed_gain__pct_min_1',
            value=0)
        flow_dstr_pump_speed_gain__pct_min_1.upper=flow_dstr_pump_speed_max_gain__pct_min_1  # Enforce the max gain
        
        m.Equation(
            flow_dstr_pump_speed_gain__pct_min_1 == 
            error_delta_t_flow_ret__K / error_threshold_delta_t_flow_ret__K * flow_dstr_pump_speed_max_gain__pct_min_1
        )

        m.Equation(flow_dstr_pump_speed__pct.dt() == flow_dstr_pump_speed_gain__pct_min_1)
        
        m.Equation(
            flow_dstr_pump_speed__pct == m.if3(
                cooldown_condition,          # Condition: cooldown_condition == True
                100,                         # Action: set pump speed to 100
                m.if3(                       # Else:
                    in_post_pump_run_condition, # Condition: in_post_pump_run_condition == True
                    post_pump_speed__pct,       # Action:  use building-specific post-pump speed
                    m.if3(                      # Else:
                        no_heat_demand_condition,      # Condition: temp_flow_ch_set__degC == 0 (< 0.5)
                        0,                             # Action: set pump speed set to 0
                        flow_dstr_pump_speed__pct      # Else: retain the current pump speed
                    )
                )
            )
        )

        ##################################################################################################################
        # Heat gains from the heat generation system(s)
        ##################################################################################################################
    
        # Calculate g25_3_use_fan_lhv__W based on fan_speed__pct
        g25_3_use_fan_lhv__W = m.Var(value=0, name='g25_3_use_fan_lhv__W')
        m.Equation(g25_3_use_fan_lhv__W == 
                   (
                       Qnh_min_lhv__kW +
                       (fan_speed__pct / 100) * (Qnh_max_lhv__kW - Qnh_min_lhv__kW)
                   ) * W_kW_1
                  )

        # Calculate g_use_fan_load__pct
        g_use_fan_load__pct = m.Var(value=0, name='g_use_fan_load__pct')
        m.Equation(g_use_fan_load__pct == 
                   (
                       g25_3_use_fan_lhv__W /
                       (Qnh_max_lhv__kW * W_kW_1) *
                       100
                   )
                  )

        # Gas calorific conversion factor
        gas_std_hhv__J_m_3 = m.MV(
            value=df_learn[property_sources['gas_std_hhv__J_m_3']].astype('float32').values,
            name='gas_std_hhv__J_m_3'
            )
        gas_std_hhv__J_m_3.STATUS = 0  # No optimization
        gas_std_hhv__J_m_3.FSTATUS = 1 # Use the measured values

        gas_calorific_factor_g25_3_lhv_to_actual_hhv__J0 = m.Intermediate(
            gas_std_hhv__J_m_3 / gas_g25_3_ref_lhv__J_m_3,
            name='gas_calorific_factor_g25_3_lhv_to_actual_hhv__J0'
        )

        # Gas pressure conversion factor
        air_outdoor__Pa = m.MV(
            value=df_learn[property_sources['air_outdoor__Pa']].astype('float32').values,
            name='air_outdoor__Pa'
            )
        air_outdoor__Pa.STATUS = 0  # No optimization
        air_outdoor__Pa.FSTATUS = 1 # Use the measured values

        gas_pressure_factor_ref_to_actual__J0 = m.Intermediate(
            (air_outdoor__Pa + overpressure_gas_nl_avg__Pa) / 
            (P_std__Pa + overpressure_gas_nl_avg__Pa),
            name='gas_pressure_factor_ref_to_actual__J0'
        )

        # Gas temperature conversion factor
        gas_temp_factor_ref_to_actual__J0 = m.Intermediate(
            temp_gas_ref__K / temp_gas_nl_avg__K,
            name='gas_temp_factor_ref_to_actual__J0'
        )

        # Boiler gas input power at actual conditions
        g_use_ch_hhv__W = m.Var(value=0, name='g_use_ch_hhv__W')
        m.Equation(g_use_ch_hhv__W == g25_3_use_fan_lhv__W * 
                gas_calorific_factor_g25_3_lhv_to_actual_hhv__J0 * 
                gas_pressure_factor_ref_to_actual__J0 * 
                gas_temp_factor_ref_to_actual__J0)

        # Boiler efficiency central heating
        eta_ch_hhv__W0 = m.Var(value=0, name='eta_ch_hhv__W0')
        m.Equation(eta_ch_hhv__W0 == boiler_efficiency_hhv(g_use_fan_load__pct, temp_ret_ch__degC))

        heat_g_ch__W = m.Intermediate(g_use_ch_hhv__W * eta_ch_hhv__W0, name='heat_g_ch__W')

        # TODO: add heat gains from heat pump here when hybrid or all-electic heat pumps must be simulated
        e_use_ch__W = 0.0
        cop_ch__W0 = 1.0
        heat_e_ch__W = e_use_ch__W * cop_ch__W0
        
        # Heat generation power input from gas (and electricity)
        power_input_ch__W = m.Intermediate(g_use_ch_hhv__W + e_use_ch__W, name='power_input_ch__W')

        # Heating power output to heat distribution system
        heat_ch__W = m.Intermediate(heat_g_ch__W + heat_e_ch__W, name='heat_ch__W')
            
        ##################################################################################################################
        # Heat gains from the heat distribution system
        ##################################################################################################################
        
        # Use learned parameters
        heat_tr_dstr__W_K_1 =  m.Param(value=bldng_data['learned_heat_tr_dstr__W_K_1'], name='heat_tr_dstr__W_K_1')
        th_mass_dstr__Wh_K_1 =  m.Param(value=bldng_data['learned_th_mass_dstr__Wh_K_1'], name='th_mass_dstr__Wh_K_1')
        flow_dstr_slope__dm3_s_1_pct_1 =  m.Param(value=bldng_data['flow_dstr_slope__dm3_s_1_pct_1'], name='flow_dstr_slope__dm3_s_1_pct_1')
        flow_dstr_intercept__dm3_s_1 =  m.Param(value=bldng_data['flow_dstr_intercept__dm3_s_1'], name='flow_dstr_intercept__dm3_s_1')

        # Equations for heat distribution flow
        flow_dstr__dm3_s_1 = m.Intermediate(flow_dstr_intercept__dm3_s_1 + (flow_dstr_pump_speed__pct/100) * flow_dstr_slope__dm3_s_1_pct_1, name='flow_dstr__dm3_s_1')
        delta_t_flow_ret__K = m.Intermdiate(temp_flow_ch__degC - temp_ret_ch__degC, lb=0.0, name='delta_t_flow_ret__K')
        # vectorized_water_volumetric_heat_capacity__J_dm_3_K_1 = np.vectorize(water_volumetric_heat_capacity__J_dm_3_K_1)
        flow_dstr_J_dm_3_K_1= m.Intermdiate(water_volumetric_heat_capacity__J_dm_3_K_1(temp_dstr__degC, heat_dstr_nl_avg_abs__Pa), name='flow_dstr_J_dm_3_K_1')
        m.Equation(flow_dstr__dm3_s_1 == heat_ch__W / (flow_dstr_J_dm_3_K_1 * delta_t_flow_ret__K))

        # Define equations for heat distribution system
        m.Equation(temp_dstr__degC == (temp_flow_ch__degC + temp_ret_ch__degC) / 2)
        heat_dstr__W = m.Intermediate(heat_tr_dstr__W_K_1 * (temp_dstr__degC - temp_indoor__degC), name='heat_dstr__W')
        th_mass_dstr__J_K_1 = m.Intermediate(th_mass_dstr__Wh_K_1 * s_h_1, name='th_mass_dstr__J_K_1')
        m.Equation(temp_dstr__degC.dt() == (heat_ch__W - heat_dstr__W) / th_mass_dstr__J_K_1)
        
        ##################################################################################################################
        # Solar heat gains
        ##################################################################################################################
    
        aperture_sol__m2 = m.Param(value=bldng_data['learned_aperture_sol__m2'], name='aperture_sol__m2')
    
        sol_ghi__W_m_2 = m.MV(value=df_learn[property_sources['sol_ghi__W_m_2']].astype('float32').values, name='sol_ghi__W_m_2')
        sol_ghi__W_m_2.STATUS = 0  # No optimization
        sol_ghi__W_m_2.FSTATUS = 1 # Use the measured values
    
        heat_sol__W = m.Intermediate(sol_ghi__W_m_2 * aperture_sol__m2, name='heat_sol__W')
    
        ##################################################################################################################
        ## Internal heat gains ##
        ##################################################################################################################

        # Heat gains from domestic hot water

        g_use_dhw_hhv__W = m.MV(value = df_learn[property_sources['g_use_dhw_hhv__W']].astype('float32').values, name='g_use_dhw_hhv__W')
        g_use_dhw_hhv__W.STATUS = 0  # No optimization 
        g_use_dhw_hhv__W.FSTATUS = 1 # Use the measured values
        heat_g_dhw__W = m.Intermediate(g_use_dhw_hhv__W * param_hints['eta_dhw_hhv__W0'] * param_hints['frac_remain_dhw__0'], name='heat_g_dhw__W')

        # Heat gains from cooking
        heat_g_cooking__W = m.Param(
            value=param_hints['g_use_cooking_hhv__W'] * param_hints['eta_cooking_hhv__W0'] * param_hints['frac_remain_cooking__0'],
            name='heat_g_cooking__W'
        )

        # Heat gains from electricity
        # we assume all electricity is used indoors and turned into heat
        heat_e__W = m.MV(value = df_learn[property_sources['e__W']].astype('float32').values, name='heat_e__W')
        heat_e__W.STATUS = 0  # No optimization
        heat_e__W.FSTATUS = 1 # Use the measured values

        # Heat gains from occupants
        occupancy__p = m.MV(value = df_learn[property_sources['occupancy__p']].astype('float32').values, name='occupancy__p')
        occupancy__p.STATUS = 0  # No optimization
        occupancy__p.FSTATUS = 1 # Use the measured values
        heat_int_occupancy__W = m.Intermediate(occupancy__p * param_hints['heat_int__W_p_1'], name='heat_int_occupancy__W')

        # Sum of all 'internal' heat gains 
        heat_int__W = m.Intermediate(heat_g_dhw__W + heat_g_cooking__W + heat_e__W + heat_int_occupancy__W, name='heat_int__W')
        
        ##################################################################################################################
        # Conductive heat losses
        ##################################################################################################################
    
        heat_tr_bldng_cond__W_K_1 = m.Param(value=bldng_data['learned_heat_tr_bldng_cond__W_K_1'], name='heat_tr_bldng_cond__W_K_1')
    
        temp_outdoor__degC = m.MV(value=df_learn[property_sources['temp_outdoor__degC']].astype('float32').values, name='temp_outdoor__degC')
        temp_outdoor__degC.STATUS = 0  # No optimization
        temp_outdoor__degC.FSTATUS = 1 # Use the measured values
    
        delta_t_indoor_outdoor__K = m.Intermediate(temp_indoor__degC - temp_outdoor__degC, name='delta_t_indoor_outdoor__K')
    
        heat_loss_bldng_cond__W = m.Intermediate(heat_tr_bldng_cond__W_K_1 * delta_t_indoor_outdoor__K, name='heat_loss_bldng_cond__W')
    
        ##################################################################################################################
        # Infiltration and ventilation heat losses
        ##################################################################################################################
    
        wind__m_s_1 = m.MV(value=df_learn[property_sources['wind__m_s_1']].astype('float32').values, name='wind__m_s_1')
        wind__m_s_1.STATUS = 0  # No optimization
        wind__m_s_1.FSTATUS = 1 # Use the measured values
    
        aperture_inf__cm2 = m.Param(value=bldng_data['learned_aperture_inf__cm2'], name='aperture_inf__cm2')
    
        air_inf__m3_s_1 = m.Intermediate(wind__m_s_1 * aperture_inf__cm2 / cm2_m_2, name='air_inf__m3_s_1')
        heat_tr_bldng_inf__W_K_1 = m.Intermediate(air_inf__m3_s_1 * air_room__J_m_3_K_1, name='heat_tr_bldng_inf__W_K_1')
        heat_loss_bldng_inf__W = m.Intermediate(heat_tr_bldng_inf__W_K_1 * delta_t_indoor_outdoor__K, name='heat_loss_bldng_inf__W')

        if property_sources['ventilation__dm3_s_1'] in df_learn.columns:
            ventilation__dm3_s_1 = m.MV(value=df_learn[property_sources['ventilation__dm3_s_1']].astype('float32').values, name='ventilation__dm3_s_1')
            ventilation__dm3_s_1.STATUS = 0  # No optimization
            ventilation__dm3_s_1.FSTATUS = 1  # Use the measured values
            
            air_changes_vent__s_1 = m.Intermediate(ventilation__dm3_s_1 / (bldng__m3 * dm3_m_3), name='air_changes_vent__s_1')
            heat_tr_bldng_vent__W_K_1 = m.Intermediate(air_changes_vent__s_1 * bldng__m3 * air_room__J_m_3_K_1, name='heat_tr_bldng_vent__W_K_1')
            heat_loss_bldng_vent__W = m.Intermediate(heat_tr_bldng_vent__W_K_1 * delta_t_indoor_outdoor__K, name='heat_loss_bldng_vent__W')
        else:
            heat_loss_bldng_vent__W = 0

        ##################################################################################################################
        ## Thermal inertia ##
        ##################################################################################################################
                    
        th_inert_bldng__h = m.Param(value=bldng_data['learned_th_inert_bldng__h'], name='th_inert_bldng__h')
        
        ##################################################################################################################
        ### Heat balance ###
        ##################################################################################################################

        heat_gain_bldng__W = m.Intermediate(heat_dstr__W + heat_sol__W + heat_int__W, name='heat_gain_bldng__W')
        heat_loss_bldng__W = m.Intermediate(heat_loss_bldng_cond__W + heat_loss_bldng_inf__W + heat_loss_bldng_vent__W, name='heat_loss_bldng__W')
        heat_tr_bldng__W_K_1 = m.Intermediate(heat_tr_bldng_cond__W_K_1 + heat_tr_bldng_inf__W_K_1 + heat_tr_bldng_vent__W_K_1, name='heat_tr_bldng__W_K_1')
        th_mass_bldng__Wh_K_1  = m.Intermediate(heat_tr_bldng__W_K_1 * th_inert_bldng__h, name='th_mass_bldng__Wh_K_1') 
        m.Equation(temp_indoor__degC.dt() == ((heat_gain_bldng__W - heat_loss_bldng__W)  / (th_mass_bldng__Wh_K_1 * s_h_1)))

    
        ##################################################################################################################
        # Solve the model to start the simulation process
        ##################################################################################################################
        m.options.IMODE = 4        # Do not learn, but only simulate using learned parameters passed via bldng_data
        m.options.EV_TYPE = 2      # RMSE
        m.solve(disp=False)

        ##################################################################################################################
        # Store results of the simulation process
        ##################################################################################################################
        

        # Initialize a DataFrame for learned time-varying properties
        df_predicted_properties = pd.DataFrame(index=df_learn.index)
        
        # Initialize a DataFrame for used parameters (single row for metadata)
        df_learned_parameters = pd.DataFrame({         #TO DO: change to df_learned_parameters later if working
            'id': id, 
            'start': start,
            'end': end,
            'duration': timedelta(seconds=duration__s),
        }, index=[0])

        for param in used_params & current_locals.keys():
            used_value = current_locals[param].value[0]
            df_learned_parameters.loc[0, f'learned_{param}'] = used_value
            # If actual value exists, compute MAE
            if actual_params is not None and param in actual_params:
                df_learned_parameters.loc[0, f'mae_{param}'] = abs(used_value - actual_params[param])
        
        # Store learned time-varying data in DataFrame and calculate MAE and RMSE
        current_locals = locals() # current_locals is valid in list comprehensions and for loops, locals() is not. 
        for prop in (predict_props or set()) & set(current_locals.keys()):
            predicted_prop = f'predicted_{prop}'
            df_predicted_properties.loc[:,predicted_prop] = np.asarray(current_locals[prop].value)
            
            # If the property was measured, calculate and store MAE and RMSE
            if prop in property_sources.keys() and property_sources[prop] in df_learn.columns:
                df_learned_parameters.loc[0, f'mae_{prop}'] = mae(
                    df_learn[property_sources[prop]],  # Measured values
                    df_predicted_properties[predicted_prop]  # Predicted values
                )
                df_learned_parameters.loc[0, f'rmse_{prop}'] = rmse(
                    df_learn[property_sources[prop]],  # Measured values
                    df_predicted_properties[predicted_prop]  # Predicted values
                )

        # Set MultiIndex on the DataFrame (id, start, end)
        df_learned_parameters.set_index(['id', 'start', 'end', 'duration'], inplace=True)

        m.cleanup()

        return df_learned_parameters, df_predicted_properties        