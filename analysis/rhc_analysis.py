from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Set, Callable
from enum import Enum
import pandas as pd
import numpy as np
import math
from gekko import GEKKO
from tqdm.notebook import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue, Process
import time

import numbers
import logging

from pythermalcomfort.models import pmv_ppd

from nfh_utils import *

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
    
 
    def save_job_results_to_parquet(id, start, stop, df_learned_parameters, df_predicted_properties, results_dir):
        """Save the learned parameters and properties for a specific job."""
        
        # Format start and stop times as strings
        start_str = start.strftime('%Y%m%d_%H%M%S')
        stop_str = stop.strftime('%Y%m%d_%H%M%S')
    
        # File paths for learned parameters and learned properties
        learned_job_parameters_file_path = os.path.join(results_dir, f'learned-parameters-job-{id}-{start_str}-{stop_str}.parquet')
        predicted_job_properties_file_path = os.path.join(results_dir, f'predicted-properties-job-{id}-{start_str}-{stop_str}.parquet')
    
        # Save learned parameters
        if df_learned_parameters is not None:
            if os.path.exists(learned_job_parameters_file_path):
                # Read existing learned parameters
                df_existing_params = pd.read_parquet(learned_job_parameters_file_path)
    
                # Log info for debugging
                logging.info(f"Parameters already learned: {df_existing_params.columns}; to add {df_learned_parameters.columns}")
                logging.info(f"Shape already learned: {df_existing_params.shape}; shape to add {df_learned_parameters.shape}")
        
                # Concatenate new learned parameters horizontally
                # Aligns on index and adds new columns as needed
                df_existing_params = pd.concat([df_existing_params, df_learned_parameters], axis=1)
        
                # Ensure no duplicate columns
                df_existing_params = df_existing_params.loc[:,~df_existing_params.columns.duplicated()]
            else:
                # If no existing parameters, simply use the learned ones
                df_existing_params = df_learned_parameters
    
                # Save the updated learned parameters
                df_existing_params.to_parquet(learned_job_parameters_file_path)
                logging.info(f"Updated learned parameters for job ID {id} (from {start} to {stop}) in {learned_job_parameters_file_path}")
        
        # Save learned properties
        if df_predicted_properties is not None:
            if os.path.exists(predicted_job_properties_file_path):
                # Read existing learned properties
                df_existing_props = pd.read_parquet(predicted_job_properties_file_path)
        
                # Log info for debugging
                logging.info(f"Properties already learned: {df_existing_props.columns}; to add {df_predicted_properties.columns}")
                logging.info(f"Shape already learned: {df_existing_props.shape}; shape to add {df_predicted_properties.shape}")
        
                # Concatenate new and existing properties, avoiding duplicate columns
                # Use `join='outer'` to ensure new columns are added, but duplicates are avoided
                df_combined_props = pd.concat([df_existing_props, df_predicted_properties], axis=0, join='outer')
        
                # Ensure no duplicate columns by dropping columns with the same name that might have been reintroduced
                df_combined_props = df_combined_props.loc[:, ~df_combined_props.columns.duplicated()]
        
                # Save the updated learned properties
                df_combined_props.to_parquet(predicted_job_properties_file_path)
                logging.info(f"Updated learned properties for job ID {id} (from {start} to {stop}) in {predicted_job_properties_file_path}")
            else:
                # If the file does not exist, save the new learned properties
                df_predicted_properties.to_parquet(predicted_job_properties_file_path)
                logging.info(f"Saved new learned properties for job ID {id} (from {start} to {stop}) to {predicted_job_properties_file_path}")
    
    def save_to_parquet(id, df_learned_parameters, df_predicted_properties, df_data, results_dir):
        """Save the learned parameters and properties for a specific id to Parquet."""
        df_learned_parameters.to_parquet(os.path.join(results_dir, f'learned-parameters-per-period-{id}.parquet'))
        if df_predicted_properties is not None:
            df_predicted_properties.to_parquet(os.path.join(results_dir, f'learned-properties-{id}.parquet'))
        logging.info(f'Saved results for ID {id} to {results_dir}')
        
        # Save df_data if needed (incremental saving)
        df_data.to_parquet(os.path.join(results_dir, f'df_data_{id}.parquet'))
    
    def final_save_to_parquet(df_learned_parameters, df_data, results_dir):
        """Final save of all aggregated results after processing all ids."""
        df_learned_parameters.to_parquet(os.path.join(results_dir, 'results_per_period_final.parquet'))
        logging.info(f'Final results per period saved to {results_dir}/results_per_period_final.parquet')
        
        df_data.to_parquet(os.path.join(results_dir, 'results_final.parquet'))
        logging.info(f'Final df_data saved to {results_dir}/results_final.parquet')    

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
            predict_props (Set[str]): Properties to predict.
            duration_threshold (timedelta): Minimum duration for jobs.
            learn_period__d (int): Maximum duration of data for jobs (in days)
            max_periods (int): Maximum periods for learning jobs.
            **kwargs: Additional arguments passed to the system_model_fn.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Learned parameters and updated data.
        """
        # Determine required columns
        req_cols = (
            set(property_sources.values())
            if req_props is None
            else {property_sources[prop] for prop in req_props & property_sources.keys()}
        )

        # Focus only on the required columns
        df_learn_all = df_data[list(req_cols)]

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
            # Merge predicted properties back into the main DataFrame
            df_data = df_data.drop(columns=df_data.columns.intersection(df_predicted_properties.columns))
            df_data = df_data.merge(df_predicted_properties, left_index=True, right_index=True, how="left")
        
        if aggregated_learned_job_parameters:
            # Combine results into a cumulative DataFrame for learned parameters
            df_learned_parameters = (
                pd.concat(aggregated_learned_job_parameters, axis=0)
                .drop_duplicates()
                .sort_index()
            )
        else:
            df_learned_parameters = pd.DataFrame()

        return df_learned_parameters, df_data
        

    def learn_ventilation(
        df_learn: pd.DataFrame,
        bldng_data: Dict = None,
        property_sources: Dict = None,
        param_hints: Dict = None,
        learn_params: Set[str] = {'aperture_inf__cm2'},
        actual_params: Dict = None,
        predict_props: Set[str] = {'ventilation__dm3_s_1'},
        learn_change_interval: timedelta = timedelta(minutes=30)
     ) -> Tuple[pd.DataFrame, pd.DataFrame]:


        id, start, end, step__s, duration__s  = Learner.get_time_info(df_learn) 
        
        bldng__m3 = bldng_data['bldng__m3']
        floors__m2 = bldng_data['floors__m2']
        
        logging.info(f"learn ventilation rate for id {df_learn.index.get_level_values('id')[0]}, from  {df_learn.index.get_level_values('timestamp').min()} to {df_learn.index.get_level_values('timestamp').max()}")

        ##################################################################################################################
        # GEKKO Model - Initialize
        ##################################################################################################################
        m = GEKKO(remote=False)
        m.time = np.arange(0, duration__s, step__s)

        ##################################################################################################################
        ## Use measured CO₂ concentration indoors
        ##################################################################################################################
        co2_indoor__ppm = m.CV(value=df_learn[property_sources['co2_indoor__ppm']].values)
        co2_indoor__ppm.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
        co2_indoor__ppm.FSTATUS = 1 # Use the measured values
        
        ##################################################################################################################
        ## CO₂ concentration gain indoors
        ##################################################################################################################

        # Use measured occupancy
        occupancy__p = m.MV(value = df_learn[property_sources['occupancy__p']].astype('float32').values)
        occupancy__p.STATUS = 0  # No optimization
        occupancy__p.FSTATUS = 1 # Use the measured values

        co2_indoor_gain__ppm_s_1 = m.Intermediate(occupancy__p * co2_exhale_sedentary__umol_p_1_s_1 / 
                                                  (bldng__m3 * gas_room__mol_m_3))
        
        ##################################################################################################################
        ## CO₂ concentration loss indoors
        ##################################################################################################################

        # Ventilation-induced CO₂ concentration loss indoors
        ventilation__dm3_s_1 = m.MV(value=param_hints['ventilation_default__dm3_s_1'],
                                    lb=0.0, 
                                    ub=param_hints['ventilation_max__dm3_s_1_m_2'] * floors__m2)
        ventilation__dm3_s_1.STATUS = 1  # Allow optimization
        ventilation__dm3_s_1.FSTATUS = 1 # Use the measured values
        
        if learn_change_interval is not None:
            update_interval_steps = int(np.ceil(learn_change_interval.total_seconds() / step__s))
            ventilation__dm3_s_1.MV_STEP_HOR = update_interval_steps
        
        air_changes_vent__s_1 = m.Intermediate(ventilation__dm3_s_1 / (bldng__m3 * dm3_m_3))

        # Wind-induced (infiltration) CO₂ concentration loss indoors
        wind__m_s_1 = m.MV(value=df_learn[property_sources['wind__m_s_1']].astype('float32').values)
        wind__m_s_1.STATUS = 0  # No optimization
        wind__m_s_1.FSTATUS = 1 # Use the measured values
    
        if 'aperture_inf__cm2' in learn_params:
            aperture_inf__cm2 = m.FV(value=param_hints['aperture_inf__cm2'], lb=0, ub=100000.0)
            aperture_inf__cm2.STATUS = 1  # Allow optimization
            aperture_inf__cm2.FSTATUS = 1 # Use the initial value as a hint for the solver
        else:
            aperture_inf__cm2 = m.Param(value=param_hints['aperture_inf__cm2'])

        air_inf__m3_s_1 = m.Intermediate(wind__m_s_1 * aperture_inf__cm2 / cm2_m_2)        
        air_changes_inf__s_1 = m.Intermediate(air_inf__m3_s_1 / bldng__m3)

        # Total losses of CO₂ concentration indoors
        air_changes_total__s_1 = m.Intermediate(air_changes_vent__s_1 + air_changes_inf__s_1)
        co2_elevation__ppm = m.Intermediate(co2_indoor__ppm - param_hints['co2_outdoor__ppm'])
        co2_indoor_loss__ppm_s_1 = m.Intermediate(air_changes_total__s_1 * co2_elevation__ppm)
        
        ##################################################################################################################
        # CO₂ concentration balance equation:  
        ##################################################################################################################
        m.Equation(co2_indoor__ppm.dt() == co2_indoor_gain__ppm_s_1 - co2_indoor_loss__ppm_s_1)
        
        ##################################################################################################################
        # Solve the model to start the learning process
        ##################################################################################################################
        m.options.IMODE = 5        # Simultaneous Estimation 
        m.options.EV_TYPE = 2      # RMSE
        m.solve(disp=False)

        ##################################################################################################################
        # Store results of the learning process
        ##################################################################################################################
        
        # Initialize a DataFrame, even for a single learned parameter (one row with id, start, end), for consistency
        df_learned_parameters = pd.DataFrame({
            'id': id, 
            'start': start,
            'end': end,
            'duration': timedelta(seconds=duration__s),
        }, index=[0])
        
        # Loop over the learn_params set and store learned values and calculate MAE if actual value is available
        current_locals = locals() # current_locals is valid in list comprehensions and for loops, locals() is not. 
        if learn_params: 
            for param in learn_params & current_locals.keys():
                learned_value = current_locals[param].value[0]
                df_learned_parameters.loc[0, f'learned_{param}'] = learned_value
                # If actual value exists, compute MAE
                if actual_params is not None and param in actual_params:
                    df_learned_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_params[param])

        # Initialize a DataFrame for learned time-varying properties
        df_predicted_properties = pd.DataFrame(index=df_learn.index)

        # Store learned time-varying data in DataFrame and calculate MAE and RMSE
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

    
    def learn_heat_distribution(
        df_learn,
        bldng_data: Dict = None,
        property_sources: Dict = None,
        param_hints: Dict = None,
        learn_params: Set[str] = {'heat_tr_dstr__W_K_1',
                                  'th_mass_dstr__Wh_K_1',
                                  'th_inert_dstr__h',
                                 },
        actual_params: Dict = None,
        predict_props: Set[str] = {'temp_ret_ch__degC',
                                   'temp_dstr__degC',
                                   'heat_dstr__W',
                                  }        
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        logging.info(f"learn heat distribution for id {df_learn.index.get_level_values('id')[0]}, from  {df_learn.index.get_level_values('timestamp').min()} to {df_learn.index.get_level_values('timestamp').max()}")


 
        id, start, end, step__s, duration__s  = Learner.get_time_info(df_learn) 

        ##################################################################################################################
        # GEKKO Model - Initialize
        ##################################################################################################################
        m = GEKKO(remote=False)
        m.time = np.arange(0, duration__s, step__s)

        ##################################################################################################################
        # Central heating gains
        ##################################################################################################################
        g_use_ch_hhv__W = m.MV(value=df_learn[property_sources['g_use_ch_hhv__W']].astype('float32').values)
        g_use_ch_hhv__W.STATUS = 0  # No optimization
        g_use_ch_hhv__W.FSTATUS = 1 # Use the measured values

        e_use_ch__W = 0  # TODO: add electricity use from heat pump here when hybrid or all-electic heat pumps must be simulated
        energy_ch__W = m.Intermediate(g_use_ch_hhv__W + e_use_ch__W)
    
        eta_ch_hhv__W0 = m.MV(value=df_learn[property_sources['eta_ch_hhv__W0']].astype('float32').values)
        eta_ch_hhv__W0.STATUS = 0  # No optimization
        eta_ch_hhv__W0.FSTATUS = 1 # Use the measured values
    
        heat_g_ch__W = m.Intermediate(g_use_ch_hhv__W * eta_ch_hhv__W0)

        # TODO: add heat gains from heat pump here when hybrid or all-electic heat pumps must be simulated
        cop_ch__W0 = 1.0
        heat_e_ch__W = e_use_ch__W * cop_ch__W0
        
        heat_ch__W = m.Intermediate(heat_g_ch__W + heat_e_ch__W)
    
        ##################################################################################################################
        # Heat distribution system parameters to learn
        ##################################################################################################################
        # Effective heat transfer capacity of the heat distribution system
        heat_tr_dstr__W_K_1 = m.FV(value=param_hints['heat_tr_dstr__W_K_1'], lb=50, ub=1000)
        heat_tr_dstr__W_K_1.STATUS = 1  # Allow optimization
        heat_tr_dstr__W_K_1.FSTATUS = 1 # Use the initial value as a hint for the solver

        # Effective thermal mass of the heat distribution system
        th_mass_dstr__Wh_K_1 = m.FV(value=param_hints['th_mass_dstr__Wh_K_1'], lb=50, ub=5000)
        th_mass_dstr__Wh_K_1.STATUS = 1  # Allow optimization
        th_mass_dstr__Wh_K_1.FSTATUS = 1 # Use the initial value as a hint for the solver

        # Effective thermal inertia (a.k.a. thermal time constant) of the heat distribution system
        if 'th_inert_dstr__h' in learn_params:
            th_inert_dstr__h = m.Intermediate(th_mass_dstr__Wh_K_1 / heat_tr_dstr__W_K_1)
    
        ##################################################################################################################
        # Flow and indoor temperature  
        ##################################################################################################################
        temp_flow_ch__degC = m.MV(value=df_learn[property_sources['temp_flow_ch__degC']].astype('float32').values)
        temp_flow_ch__degC.STATUS = 0  # No optimization
        temp_flow_ch__degC.FSTATUS = 1 # Use the measured values

        temp_indoor__degC = m.MV(value=df_learn[property_sources['temp_indoor__degC']].astype('float32').values)
        temp_indoor__degC.STATUS = 0  # No optimization
        temp_indoor__degC.FSTATUS = 1 # Use the measured values

        # ##################################################################################################################
        # # Alternative way: fit on distribution temperature
        # ##################################################################################################################
        # temp_dstr__degC = m.CV(value=((df_learn[property_sources['temp_flow_ch__degC']] 
        #                                + df_learn[property_sources['temp_ret_ch__degC']]) / 2).astype('float32').values)
        # temp_dstr__degC.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
        # temp_dstr__degC.FSTATUS = 1 # Use the measured values
        # temp_ret_ch__degC = m.Var(value=df_learn[property_sources['temp_ret_ch__degC']].iloc[0])  # Initial guesss
        
        ##################################################################################################################
        # Fit on return temperature
        ##################################################################################################################
        temp_ret_ch__degC = m.CV(value=df_learn[property_sources['temp_ret_ch__degC']].astype('float32').values)
        temp_ret_ch__degC.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
        temp_ret_ch__degC.FSTATUS = 1 # Use the measured values
        temp_dstr__degC = m.Var(value=(df_learn[property_sources['temp_flow_ch__degC']].iloc[0] 
                                       + df_learn[property_sources['temp_ret_ch__degC']].iloc[0]) / 2)  # Initial guesss

        ##################################################################################################################
        # Dynamic model of the heat distribution system
        ##################################################################################################################
        m.Equation(temp_dstr__degC == (temp_flow_ch__degC + temp_ret_ch__degC) / 2)
        heat_dstr__W = m.Intermediate(heat_tr_dstr__W_K_1 * (temp_dstr__degC - temp_indoor__degC))
        th_mass_dstr__J_K_1 = m.Intermediate(th_mass_dstr__Wh_K_1 * s_h_1)
        m.Equation(temp_dstr__degC.dt() == (heat_ch__W - heat_dstr__W) / th_mass_dstr__J_K_1)


        
        ##################################################################################################################
        # Solve the model to start the learning process
        ##################################################################################################################
        m.options.IMODE = 5        # Simultaneous Estimation 
        m.options.EV_TYPE = 2      # RMSE
        m.solve(disp=False)

        ##################################################################################################################
        # Store results of the learning process
        ##################################################################################################################
        
        # Initialize a DataFrame for learned parameters (single row for metadata)
        df_learned_parameters = pd.DataFrame({
            'id': id, 
            'start': start,
            'end': end,
            'duration': timedelta(seconds=duration__s),
        }, index=[0])
        
        # Loop over the learn_params and store learned values and calculate MAE if actual value is available
        current_locals = locals() # current_locals is valid in list comprehensions and for loops, locals() is not. 
        if learn_params: 
            for param in learn_params & current_locals.keys():
                learned_value = current_locals[param].value[0]
                df_learned_parameters.loc[0, f'learned_{param}'] = learned_value
                # If actual value exists, compute MAE
                if actual_params is not None and param in actual_params:
                    df_learned_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_params[param])
       
        # Initialize a DataFrame for learned time-varying properties
        df_predicted_properties = pd.DataFrame(index=df_learn.index)
        
        # Store learned time-varying data in DataFrame and calculate MAE and RMSE
        predicted_props = (predict_props or set()) & set(current_locals.keys())
        for prop in predicted_props:
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
        
        
        
    def learn_thermal_parameters(
        df_learn: pd.DataFrame,
        bldng_data: Dict = None,
        property_sources: Dict = None,
        param_hints: Dict = None,
        learn_params: Set[str] = None,
        actual_params: Dict = None,
        predict_props: Set[str] = None,
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
        
        # Periodic averages to calculate, which include Energy Case metrics
        properties_mean = [
            'temp_set__degC',
            'temp_flow__degC',
            'temp_ret__degC',
            'comfortable__bool',
            'temp_indoor__degC',
            'temp_outdoor__degC',
            'temp_flow_ch_max__degC',
            'heat_ch__W',
        ]
            
        id, start, end, step__s, duration__s  = Learner.get_time_info(df_learn) 

        bldng__m3 = bldng_data['bldng__m3']
        
        logging.info(f"learn_thermal_parameters for id {df_learn.index.get_level_values('id')[0]}, from  {df_learn.index.get_level_values('timestamp').min()} to {df_learn.index.get_level_values('timestamp').max()}")

        ##################################################################################################################
        # GEKKO Model - Initialize
        ##################################################################################################################

        m = GEKKO(remote=False)
        m.time = np.arange(0, duration__s, step__s)

        ##################################################################################################################
        # Heat gains
        ##################################################################################################################
    
        # Central heating gains
        g_use_ch_hhv__W = m.MV(value=df_learn[property_sources['g_use_ch_hhv__W']].astype('float32').values)
        g_use_ch_hhv__W.STATUS = 0  # No optimization
        g_use_ch_hhv__W.FSTATUS = 1 # Use the measured values

        e_use_ch__W = 0  # TODO: add electricity use from heat pump here when hybrid or all-electic heat pumps must be simulated
        energy_ch__W = m.Intermediate(g_use_ch_hhv__W + e_use_ch__W)
    
        eta_ch_hhv__W0 = m.MV(value=df_learn[property_sources['eta_ch_hhv__W0']].astype('float32').values)
        eta_ch_hhv__W0.STATUS = 0  # No optimization
        eta_ch_hhv__W0.FSTATUS = 1 # Use the measured values
    
        heat_g_ch__W = m.Intermediate(g_use_ch_hhv__W * eta_ch_hhv__W0)

        # TODO: add heat gains from heat pump here when hybrid or all-electic heat pumps must be simulated
        cop_ch__W0 = 1.0
        heat_e_ch__W = e_use_ch__W * cop_ch__W0
        
        heat_ch__W = m.Intermediate(heat_g_ch__W + heat_e_ch__W)
    
        # Heat distribution system
        temp_flow__degC = m.MV(value=df_learn[property_sources['temp_flow__degC']].astype('float32').values)
        temp_flow__degC.STATUS = 0  # No optimization
        temp_flow__degC.FSTATUS = 1 # Use the measured values

        #TO DO: calculate return temperatures for what-if simulations
        temp_ret__degC = m.MV(value=df_learn[property_sources['temp_ret__degC']].astype('float32').values)
        temp_ret__degC.STATUS = 0  # No optimization
        temp_ret__degC.FSTATUS = 1 # Use the measured values
        
        temp_indoor__degC = m.CV(value=df_learn[property_sources['temp_indoor__degC']].astype('float32').values)
        temp_indoor__degC.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
        temp_indoor__degC.FSTATUS = 1 # Use the measured values

        # assume immediate and full heat distribution
        heat_dstr__W = heat_ch__W

        # # If possible, use the learned heat transfer capacity of the heat distribution system, otherwise use a generic value
        # if pd.notna(bldng_data['learned_heat_tr_dstr__W_K_1']) & pd.notna(bldng_data['learned_th_mass_dstr__Wh_K_1']):
        #     # Heat distribution characteristics were learned; use them:
        #     heat_tr_dstr__W_K_1 =  m.Param(value=bldng_data['learned_heat_tr_dstr__W_K_1'])
        #     th_mass_dstr__Wh_K_1 =  m.Param(value=bldng_data['learned_th_mass_dstr__Wh_K_1'])
        #     #TO DO: calculate return temperatures for what-if simulations
        #     temp_dstr__degC = m.Intermediate((temp_flow__degC + temp_ret__degC) / 2)
        #     heat_dstr__W = m.Intermediate(heat_tr_dstr__W_K_1 * (temp_dstr__degC - temp_indoor__degC))
        # else:
        #     # If the heat distribution system parameters were not learned, assume immediate and full heat distribution 
        #     heat_dstr__W = heat_ch__W
    
        ##################################################################################################################
        # Solar heat gains
        ##################################################################################################################
    
        if 'aperture_sol__m2' in learn_params:
            aperture_sol__m2 = m.FV(value=param_hints['aperture_sol__m2'], lb=1, ub=100)
            aperture_sol__m2.STATUS = 1  # Allow optimization
            aperture_sol__m2.FSTATUS = 1 # Use the initial value as a hint for the solver
        else:
            aperture_sol__m2 = m.Param(value=param_hints['aperture_sol__m2'])
    
        sol_ghi__W_m_2 = m.MV(value=df_learn[property_sources['sol_ghi__W_m_2']].astype('float32').values)
        sol_ghi__W_m_2.STATUS = 0  # No optimization
        sol_ghi__W_m_2.FSTATUS = 1 # Use the measured values
    
        heat_sol__W = m.Intermediate(sol_ghi__W_m_2 * aperture_sol__m2)
    
        ##################################################################################################################
        ## Internal heat gains ##
        ##################################################################################################################

        # Heat gains from domestic hot water

        g_use_dhw_hhv__W = m.MV(value = df_learn[property_sources['g_use_dhw_hhv__W']].astype('float32').values)
        g_use_dhw_hhv__W.STATUS = 0  # No optimization 
        g_use_dhw_hhv__W.FSTATUS = 1 # Use the measured values
        heat_g_dhw__W = m.Intermediate(g_use_dhw_hhv__W * param_hints['eta_dhw_hhv__W0'] * param_hints['frac_remain_dhw__0'])

        # Heat gains from cooking
        heat_g_cooking__W = m.Param(param_hints['g_use_cooking_hhv__W'] * param_hints['eta_cooking_hhv__W0'] * param_hints['frac_remain_cooking__0'])

        # Heat gains from electricity
        # we assume all electricity is used indoors and turned into heat
        heat_e__W = m.MV(value = df_learn[property_sources['e__W']].astype('float32').values)
        heat_e__W.STATUS = 0  # No optimization
        heat_e__W.FSTATUS = 1 # Use the measured values

        # Heat gains from occupants
        occupancy__p = m.MV(value = df_learn[property_sources['occupancy__p']].astype('float32').values)
        occupancy__p.STATUS = 0  # No optimization
        occupancy__p.FSTATUS = 1 # Use the measured values
        heat_int_occupancy__W = m.Intermediate(occupancy__p * param_hints['heat_int__W_p_1'])

        # Sum of all 'internal' heat gains 
        heat_int__W = m.Intermediate(heat_g_dhw__W + heat_g_cooking__W + heat_e__W + heat_int_occupancy__W)
        
        ##################################################################################################################
        # Conductive heat losses
        ##################################################################################################################
    
        if 'heat_tr_bldng_cond__W_K_1' in learn_params:
            heat_tr_bldng_cond__W_K_1 = m.FV(value=param_hints['heat_tr_bldng_cond__W_K_1'], lb=0, ub=1000)
            heat_tr_bldng_cond__W_K_1.STATUS = 1  # Allow optimization
            heat_tr_bldng_cond__W_K_1.FSTATUS = 1 # Use the initial value as a hint for the solver
        else:
            heat_tr_bldng_cond__W_K_1 = param_hints['heat_tr_bldng_cond__W_K_1']
    
        temp_outdoor__degC = m.MV(value=df_learn[property_sources['temp_outdoor__degC']].astype('float32').values)
        temp_outdoor__degC.STATUS = 0  # No optimization
        temp_outdoor__degC.FSTATUS = 1 # Use the measured values
    
        indoor_outdoor_delta__K = m.Intermediate(temp_indoor__degC - temp_outdoor__degC)
    
        heat_loss_bldng_cond__W = m.Intermediate(heat_tr_bldng_cond__W_K_1 * indoor_outdoor_delta__K)
    
        ##################################################################################################################
        # Infiltration and ventilation heat losses
        ##################################################################################################################
    
        wind__m_s_1 = m.MV(value=df_learn[property_sources['wind__m_s_1']].astype('float32').values)
        wind__m_s_1.STATUS = 0  # No optimization
        wind__m_s_1.FSTATUS = 1 # Use the measured values
    
        if 'aperture_inf__cm2' in learn_params:
            aperture_inf__cm2 = m.FV(value=param_hints['aperture_inf__cm2'], lb=0, ub=100000.0)
            aperture_inf__cm2.STATUS = 1  # Allow optimization
            aperture_inf__cm2.FSTATUS = 1 # Use the initial value as a hint for the solver
        else:
            aperture_inf__cm2 = m.Param(value=param_hints['aperture_inf__cm2'])
    
        air_inf__m3_s_1 = m.Intermediate(wind__m_s_1 * aperture_inf__cm2 / cm2_m_2)
        heat_tr_bldng_inf__W_K_1 = m.Intermediate(air_inf__m3_s_1 * air_room__J_m_3_K_1)
        heat_loss_bldng_inf__W = m.Intermediate(heat_tr_bldng_inf__W_K_1 * indoor_outdoor_delta__K)
    
        if property_sources['ventilation__dm3_s_1'] in df_learn.columns and df_learn[property_sources['ventilation__dm3_s_1']].notna().all():
            ventilation__dm3_s_1 = m.MV(value=df_learn[property_sources['ventilation__dm3_s_1']].astype('float32').values)
            ventilation__dm3_s_1.STATUS = 0  # No optimization
            ventilation__dm3_s_1.FSTATUS = 1  # Use the measured values
            
            air_changes_vent__s_1 = m.Intermediate(ventilation__dm3_s_1 / (bldng__m3 * dm3_m_3))
            heat_tr_bldng_vent__W_K_1 = m.Intermediate(air_changes_vent__s_1 * bldng__m3 * air_room__J_m_3_K_1)
            heat_loss_bldng_vent__W = m.Intermediate(heat_tr_bldng_vent__W_K_1 * indoor_outdoor_delta__K)
        else:
            heat_tr_bldng_vent__W_K_1 = 0
            heat_loss_bldng_vent__W = 0

        ##################################################################################################################
        ## Thermal inertia ##
        ##################################################################################################################
                    
        # Thermal inertia (a.k.a. thermal time constant) of the building
        if 'th_inert_bldng__h' in learn_params:
            # Learn thermal inertia
            th_inert_bldng__h = m.FV(value = param_hints['th_inert_bldng__h'], lb=(10), ub=(1000))
            th_inert_bldng__h.STATUS = 1  # Allow optimization
            th_inert_bldng__h.FSTATUS = 1 # Use the initial value as a hint for the solver
        else:
            # Do not learn thermal inertia of the building, but use a fixed value based on hint
            th_inert_bldng__h = m.Param(value = param_hints['th_inert_bldng__h'])
            learned_th_inert_bldng__h = np.nan
        
        ##################################################################################################################
        ### Heat balance ###
        ##################################################################################################################

        heat_gain_bldng__W = m.Intermediate(heat_dstr__W + heat_sol__W + heat_int__W)
        heat_loss_bldng__W = m.Intermediate(heat_loss_bldng_cond__W + heat_loss_bldng_inf__W + heat_loss_bldng_vent__W)
        heat_tr_bldng__W_K_1 = m.Intermediate(heat_tr_bldng_cond__W_K_1 + heat_tr_bldng_inf__W_K_1 + heat_tr_bldng_vent__W_K_1)
        th_mass_bldng__Wh_K_1  = m.Intermediate(heat_tr_bldng__W_K_1 * th_inert_bldng__h) 
        m.Equation(temp_indoor__degC.dt() == ((heat_gain_bldng__W - heat_loss_bldng__W)  / (th_mass_bldng__Wh_K_1 * s_h_1)))

        ##################################################################################################################
        # Solve the model to start the learning process
        ##################################################################################################################
        
        m.options.IMODE = 5        # Simultaneous Estimation 
        m.options.EV_TYPE = 2      # RMSE
        m.solve(disp=False)
    
        ##################################################################################################################
        # Store results of the learning process
        ##################################################################################################################

        # Initialize a DataFrame for learned thermal parameters (one row with id, start, end)
        df_learned_parameters = pd.DataFrame({
            'id': id, 
            'start': start,
            'end': end,
            'duration': timedelta(seconds=duration__s),
        }, index=[0])
    
        # Loop over the learn_params set and store learned values and calculate MAE if actual value is available
        current_locals = locals() # current_locals is valid in list comprehensions and for loops, locals() is not. 
        for param in [param for param in (learn_params or []) if param in current_locals and param != 'ventilation__dm3_s_1']:
            learned_value = current_locals[param].value[0]
            df_learned_parameters.loc[0, f'learned_{param}'] = learned_value
            # If actual value exists, compute MAE
            if actual_params is not None and param in actual_params:
                df_learned_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_params[param])
    
        # Initialize a DataFrame for learned time-varying properties
        df_predicted_properties = pd.DataFrame(index=df_learn.index)
    
        # Store learned time-varying data in DataFrame and calculate MAE and RMSE
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
            
        for prop in properties_mean:
            # Create variable names dynamically
            # Determine the result column name based on whether the property ends with '__bool'
            if prop.endswith('__bool'):
                result_col = f"avg_{prop[:-6]}__0"  # Remove '__bool' and add '__0'
            else:
                result_col = f"avg_{prop}"
    
            # Use prop directly if it starts with 'calculated_'
            source_col = prop if prop.startswith('calculated_') else property_sources[prop]
            mean_value = df_learn[source_col].mean()
            df_learned_parameters.loc[0, result_col] = mean_value

        sim_arrays_mean = [
            'g_use_ch_hhv__W',
            'eta_ch_hhv__W0',
            'e_use_ch__W',
            'cop_ch__W0',
            'energy_ch__W',
            'heat_sol__W',
            'heat_int__W',
            'heat_dstr__W',
            'heat_loss_bldng_cond__W', 
            'heat_loss_bldng_inf__W', 
            'heat_loss_bldng_vent__W',
            'indoor_outdoor_delta__K'
        ]

        current_locals = locals() # current_locals is valid in list comprehensions and for loops, locals() is not. 
        for var in sim_arrays_mean:
            # Create variable names dynamically
            result_col = f"avg_{var}"
            mean_value = np.asarray(current_locals[var]).mean()
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
        
        # Set MultiIndex on the DataFrame (id, start, end)
        df_learned_parameters.set_index(['id', 'start', 'end', 'duration'], inplace=True)    

        m.cleanup()
    
        # Return both DataFrames: learned time-varying properties and learned fixed parameters
        return df_learned_parameters, df_predicted_properties
        

    def merge_learned(df1: pd.DataFrame, df2: pd.DataFrame, index_columns: list) -> pd.DataFrame:
        """
        Merges two multi-index DataFrames on specified index columns and any other common columns,
        avoiding duplicate columns from the second DataFrame. Handles cases where one DataFrame is empty.
        
        Parameters:
        - df1: First DataFrame (e.g., learned job parameters or properties)
        - df2: Second DataFrame (e.g., learned thermal job parameters or properties)
        - index_columns: List of columns to merge on (e.g., ['id', 'start', 'end'] or ['id', 'timestamp'])
        
        Returns:
        - Merged DataFrame with combined parameters or properties, or the non-empty DataFrame.
        """

        # Reset index to bring index columns into the DataFrame as regular columns
        df1_reset = df1.reset_index()
        df2_reset = df2.reset_index()
    
        if df1.empty and df2.empty:
            raise ValueError("Both DataFrames are empty.")
        
        # If one DataFrame is empty, return the other after validating index columns
        if df1.empty:
            missing_cols_df2 = [col for col in index_columns if col not in df2_reset.columns]
            if missing_cols_df2:
                raise KeyError(f"Index columns missing in df2: {missing_cols_df2}")
            return df2
        
        if df2.empty:
            missing_cols_df1 = [col for col in index_columns if col not in df1_reset.columns]
            if missing_cols_df1:
                raise KeyError(f"Index columns missing in df1: {missing_cols_df1}")
            return df1
        
        # Check if all index_columns are present in both DataFrames
        missing_cols_df1 = [col for col in index_columns if col not in df1_reset.columns]
        missing_cols_df2 = [col for col in index_columns if col not in df2_reset.columns]
    
        if missing_cols_df1:
            logging.warning(f"Columns in df1: {df1_reset.columns}; missing: {missing_cols_df1}")
        if missing_cols_df2:
            logging.warning(f"Columns in df2: {df2_reset.columns}; missing: {missing_cols_df2}")
    
        # Proceed only if all index_columns are present
        if not missing_cols_df1 and not missing_cols_df2:
            # Perform merge, dropping duplicates
            df_merged = pd.merge(df1_reset, df2_reset, on=index_columns, how='outer', suffixes=('', '_new'))
            
            # Drop duplicate columns
            df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]
    
            # Restore the original index
            df_merged.set_index(index_columns, inplace=True)
            
            return df_merged
        else:
            raise KeyError("Not all index_columns are present in both DataFrames.")

    
    def analyze_job(
            df_learn: pd.DataFrame,
            bldng_data: Dict = None,
            property_sources: Dict = None,
            param_hints: Dict = None,
            learn_params: Set[str] = None,
            actual_params: Dict = None,
            predict_props: Set[str] = None,
            learn_change_interval=timedelta(minutes=30),
            results_dir=None
            ):
        
        id, start, end, step__s, duration__s  = Learner.get_time_info(df_learn) 
   
        logging.info(f'Analyzing job for ID: {id}, Building data: {bldng_data}')

        if df_learn is None or df_learn.empty:
            logging.warning(f"No data available for job ID: {id}. Skipping job.")
            return None, None
        
        df_learned_parameters = pd.DataFrame()
        df_predicted_properties = pd.DataFrame()
        
        learned_job_parameters_file_path = os.path.join(results_dir, 
            f'learned-parameters-job-{id}-{start.strftime("%Y%m%d_%H%M%S")}-{end.strftime("%Y%m%d_%H%M%S")}.parquet')
        predicted_job_properties_file_path = os.path.join(results_dir, 
            f'predicted-properties-job-{id}-{start.strftime("%Y%m%d_%H%M%S")}-{end.strftime("%Y%m%d_%H%M%S")}.parquet')

        learn_thrm_param = 'thermostat_hysteresis__K'
        mode= Learner.ControlMode.LEARN_ALGORITHMIC
        learned_param = 'learned_'+ mode.value + "_" + learn_thrm_param
        
        # Check if thermostat algorithmic learning is needed
        if learn_thrm_param in learn_params:
            thermostat_alg_learned = False
    
            # Check if thermostat algoritmic results already exist
            if os.path.exists(learned_job_parameters_file_path) and os.path.exists(predicted_job_properties_file_path):
                df_learned_thrm_alg_job_parameters = pd.read_parquet(learned_job_parameters_file_path)
                df_predicted_thrm_alg_job_properties = pd.read_parquet(predicted_job_properties_file_path)
                if learn_thrm_param in df_predicted_thrm_alg_job_properties.columns:
                    logging.info(f"thermostat algoritmic control already learned for job ID {id} (from {start} to {end}).")
                    thermostat_alg_learned = True
    
            if not thermostat_alg_learned:
                # Learn thermostat algoritmic rates
                logging.info(f"Analyzing thermostat algoritmic control for job ID {id} (from {start} to {end})...")
                
                df_learned_thrm_alg_job_parameters, df_predicted_thrm_alg_job_properties = Learner.learn_thermostat_control(
                    df_learn,
                    bldng_data=bldng_data,
                    property_sources=property_sources,
                    param_hints=param_hints,
                    learn_params={learn_thrm_param},
                    actual_params=actual_params,
                    predict_props={'temp_flow_ch_set__degC'},
                    mode=mode, 
                )
    
                df_learned_parameters = df_learned_thrm_alg_job_parameters
                df_predicted_properties = df_predicted_thrm_alg_job_properties

                # Storing learned thermostat algoritmic flow setpoints in df_learn
                df_learn.loc[:,learned_param] = df_predicted_thrm_alg_job_properties[learned_param].values
                logging.info(f"Wrote {learned_param} values to df_learn for {id} from {start} to {end}")

                # Save results for thermostat algoritmic
                Learner.save_job_results_to_parquet(id, start, end, df_learned_parameters, df_predicted_properties, results_dir)
                
        # Check if ventilation learning is needed
        ventilation_prop = 'ventilation__dm3_s_1'
        predicted_ventilation_prop = property_sources[ventilation_prop]
        if ventilation_prop in predict_props:
            ventilation_learned = False
    
            # Check if ventilation results already exist
            if os.path.exists(learned_job_parameters_file_path) and os.path.exists(predicted_job_properties_file_path):
                df_learned_vent_job_parameters = pd.read_parquet(learned_job_parameters_file_path)
                df_predicted_vent_job_properties = pd.read_parquet(predicted_job_properties_file_path)
                if predicted_ventilation_prop in df_predicted_vent_job_properties.columns:
                    logging.info(f"Ventilation results already learned for job ID {id} (from {start} to {end}).")
                    ventilation_learned = True
    
            if not ventilation_learned:
                # Learn ventilation rates
                logging.info(f"Analyzing ventilation rates for job ID {id} (from {start} to {end})...")
                df_learned_vent_job_parameters, df_predicted_vent_job_properties = Learner.learn_ventilation(
                    df_learn,
                    bldng_data=bldng_data,
                    property_sources=property_sources,
                    param_hints=param_hints,
                    learn_params={'aperture_inf__cm2'},
                    actual_params=actual_params,
                    predict_props={ventilation_prop},
                    learn_change_interval=learn_change_interval,
                )
    
                df_learned_parameters = df_learned_vent_job_parameters
                df_predicted_properties = df_predicted_vent_job_properties

                # Storing learned ventilation rates in df_learn
                df_learn.loc[:,predicted_ventilation_prop] = df_predicted_vent_job_properties[predicted_ventilation_prop].values
                logging.info(f"Wrote ventilation rates to df_learn for {id} from {start} to {end}")

                # Save results for ventilation
                Learner.save_job_results_to_parquet(id, start, end, df_learned_parameters, df_predicted_properties, results_dir)
                
        # Check if thermal parameters need to be learned
        thermal_learned = False
    
        if os.path.exists(learned_job_parameters_file_path) and os.path.exists(predicted_job_properties_file_path):
            df_learned_thermal_job_parameters = pd.read_parquet(learned_job_parameters_file_path)
            df_predicted_thermal_job_properties = pd.read_parquet(predicted_job_properties_file_path)
            if 'predicted_temp_indoor__degC' in df_predicted_thermal_job_properties.columns:
                logging.info(f"Thermal parameters already learned for job ID {id} (from {start} to {end}).")
                thermal_learned = True
    
        if not thermal_learned:
            # Learn thermal parameters
            logging.info(f"Analyzing thermal properties for job ID {id} (from {start} to {end})...")
            df_learned_thermal_job_parameters, df_predicted_thermal_job_properties = Learner.learn_thermal_parameters(
                df_learn,
                bldng_data=bldng_data,
                property_sources=property_sources,
                param_hints=param_hints,
                learn_params=learn_params,
                actual_params=actual_params,
                predict_props=predict_props,
            )
    
            # Add newly learned to already learned job parameters
            df_learned_parameters = Learner.merge_learned(df_learned_parameters, df_learned_thermal_job_parameters, ['id', 'start', 'end', 'duration'])
            
            # Add newly learned to already learned job properties
            df_predicted_properties = Learner.merge_learned(df_predicted_properties, df_predicted_thermal_job_properties, ['id', 'timestamp'])
   
            # Save results for thermal parameters
            Learner.save_job_results_to_parquet(id, start, end, df_learned_parameters, df_predicted_properties, results_dir)
            
        return df_learned_parameters, df_predicted_properties

    
    @staticmethod
    def learn_heat_performance_signature(
        df_data: pd.DataFrame,
        df_bldng_data: pd.DataFrame = None,
        property_sources: Dict = None,
        param_hints: Dict = None,
        learn_params: Set[str] = None,
        actual_params: Set[str] = None,
        req_props: Set[str] = None,
        predict_props: Set[str] = None,
        duration_threshold: timedelta = None,
        learn_period__d: int = None,
        learn_change_interval: timedelta = None,
        complete_most_recent_analysis: bool = False,
        max_periods: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Input:  
        - a preprocessed pandas DataFrame with
            - a MultiIndex ['id', 'timestamp'], where
                - the column 'timestamp' is timezone-aware
                - time intervals between consecutive measurements are constant
                - but there may be gaps of multiple intervals with no measurements
                - multiple sources for the same property are already dealth with in preprocessing
            - columns:
              - property_sources['temp_indoor__degC']: indoor temperature
              - property_sources['temp_outdoor__degC']: outdoor temperature 
              - property_sources['temp_set__degC']: indoor temperature
              - property_sources['wind__m_s_1']: outdoor wind speed
              - property_sources['sol_ghi__W_m_2']: global horizontal irradiation
              - property_sources['g_use_ch_hhv__W']: gas input power (using higher heating value) used for central heating
              - property_sources['eta_dhw_hhv__W0']: efficiency (against higher heating value) of turning gas power into heat
              - property_sources['g_use_dhw_hhv__W']: gas input power (using higher heating value) used for domestic hot water
              - property_sources['e__W']: electricity power used indoors
              - property_sources['temp_flow__degC']: Temperture of hot water flow temperature, measured inside the boiler
              - property_sources['temp_ret__degC']: Temperture of hot water return temperature, measured inside the boiler
              - property_sources['temp_flow_ch__degC']: Temperture of hot water flow temperature, filtered to represent the supply temperature to the heat distributon system
              - property_sources['temp_ret_ch__degC']: Temperture of hot water return temperature, filtered to represent only the return temperature from the heat distributon system
        - df_home_bldng_data: a DataFrame with index id and columns that contain metadata per id, e.g.
            - 'floors__m2': usable floor area of a dwelling in whole square meters according to NEN 2580:2007.
            - 'bldng__m3': (an estimate of) the building volume, e.g. 3D-BAG attribute b3_volume_lod22 (https://docs.3dbag.nl/en/schema/attributes/#b3_volume_lod22) 
            - (optionally) 'building_floors__0': the number of floors, e.g. 3D-BAG attribute b3_bouwlagen (https://docs.3dbag.nl/en/schema/attributes/#b3_bouwlagen)
        - 'property_sources': a dictionary that maps key listed above to actual column names in df_data
        - 'learn_params': a set of parameters to be learned 
        - 'learn_props': a set of properties to be learned 
        - 'req_props': a set of properties, occuring as keys in property_sources: 
            - If any of the values in this column are NaN, the interval is not considered 'sane'.
            - If you do not specify a value for req_props or specify req_props = None, then all properties from the property_sources dictionary are considered required
            - to speficy NO columns are required, specify property_sources = []
        - param_hints: a dictionary that maps keys to fixed values to be used for analysis (set value for None to learn it):
            - 'aperture_sol__m2':             apparent solar aperture
            - 'eta_ch_hhv__W0':               higher heating value efficiency of the heating system 
            - 'g_not_ch_hhv__W':              average yearly gas power (higher heating value)  for other purposes than heating 
            - 'eta_not_ch_hhv__W0':           superior efficiency of heating the home indirectly using gas
            - 'wind_chill__K_s_m_1':          wind chill factor
            - 'aperture_inf__cm2':            effective infiltration area
            - 'heat_tr_bldng_cond__W_K_1':    heat transfer capacity of the building (a.k.a. specific heat loss of the building)
            - 'eta_dhw_hhv__W0':              domestic hot water efficiency
            - 'frac_remain_dhw__0':           fraction of domestic hot water heat contributing to heating the home
            - 'g_use_cooking_hhv__W':         average gas power (higher heating value) for cooking
            - 'eta_cooking_hhv__W0':          cooking efficiency
            - 'frac_remain_cooking__0':       fraction of cooking heat contributing to heating the home
            - 'heat_tr_dstr__W_K_1':          heat transfer capacity of the heat distribution system
            - 'th_mass_dstr__Wh_K_1':         thermal mass of the heat distribution system
            - 'ventilation_default__dm3_s_1': default ventilation rate for for the learning process for the entire home
            - 'ventilation_max__dm3_s_1_m_2': maximum ventilation rate relative to the total floor area of the home
            - 'co2_outdoor__ppm':             average CO₂ outdoor concentration
        and optionally,
        - 'learn_period__d': the number of days to use as learn period in the analysis
        - 'learn_change_interval': timedelta with the minimum interval that any time-varying-parameter may change
        
        Output:
        - a dataframe with per id the learned parameters and error metrics
        - a dataframe with additional column(s):
            - 'predicted_temp_indoor__degC' best fitting indoor temperatures
            - 'predicted_temp_dstr__degC' best fitting heat distribution system temperatures (if learned)
            - 'predicted_ventilation__dm3_s_1' best fitting ventilation rates (if learned)

        """
        
        # check presence of param_hints
        mandatory_hints = {'aperture_sol__m2',
                           'occupancy__p',
                           'heat_int__W_p_1',
                           'wind_chill__K_s_m_1',
                           'aperture_inf__cm2',
                           'heat_tr_bldng_cond__W_K_1', 
                           'eta_ch_hhv__W0',
                           'eta_dhw_hhv__W0',
                           'frac_remain_dhw__0',
                           'g_use_cooking_hhv__W', 
                           'eta_cooking_hhv__W0',
                           'frac_remain_cooking__0',
                           'ventilation_default__dm3_s_1',
                           'ventilation_max__dm3_s_1_m_2',
                          }
        
        for hint in mandatory_hints:
            if not (hint in param_hints or isinstance(param_hints[hint], numbers.Number)):
                raise TypeError(f'param_hints[{hint}] parameter must be a number')

        # check for unlearnable parameters
        not_learnable =   {'eta_not_ch_hhv__W0',
                           'eta_dhw_hhv__W0',
                           'frac_remain_dhw__0',
                           'g_use_cooking_hhv__W', 
                           'eta_cooking_hhv__W0',
                           'frac_remain_cooking__0',
                           'heat_int__W_p_1'
                          }
        
        # Find parameters that are both in learn_params and not_learnable
        unlearnable_params = learn_params & not_learnable
        
        if unlearnable_params:
            raise LearnError(f'No support for learning {", ".join(unlearnable_params)} (yet).')
     
        # ensure that dataframe is sorted
        if not df_data.index.is_monotonic_increasing:
            df_data = df_data.sort_index()  

        # Check for the most recent results directory if complete_most_recent_analysis is True
        base_dir = 'results'
        if complete_most_recent_analysis:
            results_dirs = [d for d in os.listdir(base_dir) if d.startswith('results-')]
            if results_dirs:
                most_recent_dir = sorted(results_dirs)[-1]  # Assuming sorted alphabetically gives the most recent
                logging.info(f'Using most recent results directory: {most_recent_dir}')
                # Load existing results into a DataFrame
                existing_results = pd.read_parquet(os.path.join(base_dir, most_recent_dir))
                logging.info(f'Loaded existing results from {most_recent_dir}')
                results_dir = most_recent_dir
        else:
            results_dir = Learner.create_results_directory()

        df_analysis_jobs = Learner.periodic_learn_list(
            df_data,
            req_props=req_props,
            property_sources=property_sources,
            learn_period__d=learn_period__d,
            duration_threshold=duration_threshold,
            max_len=max_periods
            )

        # Initialize lists to store learned properties and parameters
        aggregated_predicted_job_properties = []
        aggregated_learned_job_parameters = []

        num_jobs = df_analysis_jobs.shape[0]  # Get the number of jobs
        num_workers = min(num_jobs, os.cpu_count())  # Use the lesser of number of jobs or 16 (or any other upper limit)

        with ProcessPoolExecutor() as executor:
            # Submit tasks for each job
            print(f"Processing {num_jobs} learning jobs using {num_workers} processes")

            # Create a list to store futures for later result retrieval
            futures = []
            learned_jobs = {}
            
            with tqdm(total=num_jobs) as pbar:
                for id, start, end, duration in df_analysis_jobs.index:
                    # Create df_learn for the current job
                    df_learn = df_data.loc[(df_data.index.get_level_values('id') == id) & 
                                            (df_data.index.get_level_values('timestamp') >= start) & 
                                            (df_data.index.get_level_values('timestamp') < end)]
                    # Extracting building-specific data for each job
                    bldng_data = df_bldng_data.loc[id].to_dict()

                    if any(df_data.columns.str.startswith('model_')): 
                        # Get actual values of parameters of this id (if available)
                        actual_params = Learner.get_actual_parameter_values(
                            id, 
                            aperture_inf_nl_avg__cm2,
                            heat_tr_dstr_nl_avg__W_K_1,
                            th_mass_dstr_nl_avg__Wh_K_1
                            )
                    else: 
                        actual_params = None

                    # Submit the analyze_job function to the executor
                    future = executor.submit(Learner.analyze_job,
                                             df_learn,
                                             bldng_data=bldng_data,
                                             property_sources=property_sources, 
                                             learn_params=learn_params, 
                                             param_hints=param_hints, 
                                             actual_params=actual_params,
                                             predict_props=predict_props,
                                             learn_change_interval=learn_change_interval,
                                             results_dir=results_dir)

                    futures.append(future)
                    learned_jobs[(id, start, end, duration)] = future
    
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        df_learned_parameters, df_predicted_properties = future.result()
                        aggregated_predicted_job_properties.append(df_predicted_properties)
                        aggregated_learned_job_parameters.append(df_learned_parameters)
                    except Exception as e:
                        # Handle only the specific "Solution Not Found" error
                        if "Solution Not Found" in str(e):
                            # Find which job caused the error
                            for (id, start, end, duration), job_future in learned_jobs.items():
                                if job_future == future:
                                    logging.warning(f"Solution Not Found for job (id: {id}, start: {start}, end: {end}). Skipping.")
                                    break
                            continue  # Skip this job and move on to the next one
                        else:
                            # Reraise other exceptions to stop execution
                            raise
                    finally:
                        pbar.update(1)  # Ensure progress bar updates even if there's an exception

        # Now merge all learned job properties and parameters into cumulative DataFrames
        if aggregated_predicted_job_properties:
            df_predicted_properties = pd.concat(aggregated_predicted_job_properties, axis=0).drop_duplicates()
        else: 
            df_predicted_properties = pd.DataFrame()

        if aggregated_learned_job_parameters:
            df_learned_parameters = pd.concat(aggregated_learned_job_parameters, axis=0).drop_duplicates()
        else:
            df_learned_parameters = pd.DataFrame()
            
        # Merging all learned time series data into df_data, making sure that columns from df_predicted_properties take precedende
        df_data = df_data.drop(columns=df_data.columns.intersection(df_predicted_properties.columns))
        df_data = df_data.merge(df_predicted_properties, left_index=True, right_index=True, how='left')    

        # After all IDs, save final results
        Learner.final_save_to_parquet(df_learned_parameters, df_data, results_dir)
        
        return df_learned_parameters.sort_index(), df_data.sort_index()


    # Define the modes as an enumeration
    class ControlMode(Enum):
        LEARN_ALGORITHMIC = "alg"
        LEARN_PID = "pid"   
        

    def learn_boiler_control(
            df_learn,
            bldng_data: Dict = None,
            property_sources: Dict = None,
            param_hints: Dict = None,
            learn_params: Set[str] = {'fan_rotations_max_gain__pct_min_1',
                                      'error_threshold_temp_delta_flow_flowset__K',
                                      'flow_dstr_pump_speed_max_gain__pct_min_1',
                                      'error_threshold_temp_delta_flow_ret__K',
                                     },
            actual_params: Dict = None,
            predict_props: Set[str] = {'fan_speed__pct', 'flow_dstr_pump_speed__pct'},
            mode: ControlMode = ControlMode.LEARN_ALGORITHMIC,
            max_iter=10,
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        
        id, start, end, step__s, duration__s  = Learner.get_time_info(df_learn) 

        ##################################################################################################################
        # Initialize GEKKO model
        ##################################################################################################################
        m = GEKKO(remote=False)
        m.time = np.arange(0, duration__s, step__s)

        ##################################################################################################################
        # Flow setpoint
        ##################################################################################################################

        temp_flow_ch_max__degC  = m.MV(value=df_learn[property_sources['temp_flow_ch_max__degC']].astype('float32').values)
        temp_flow_ch_max__degC .STATUS = 0  # No optimization
        temp_flow_ch_max__degC .FSTATUS = 1 # Use the measured values
        
        # Initial assumption: fixed setpoint; will be relaxed later
        temp_flow_ch_set__degC = m.MV(value=df_learn[property_sources['temp_flow_ch_set__degC']].astype('float32').values)
        temp_flow_ch_set__degC.lower = 0  # Minimum value
        m.Equation(temp_flow_ch_set__degC <= temp_flow_ch_max__degC) # constraint to enforce the maximum limit dynamically

        ##################################################################################################################
        # Flow and return temperature
        ##################################################################################################################
        temp_flow_ch__degC = m.MV(value=df_learn[property_sources['temp_flow_ch__degC']].astype('float32').values)
        temp_flow_ch__degC.STATUS = 0  # No optimization
        temp_flow_ch__degC.FSTATUS = 1 # Use the measured values
        
        temp_ret_ch__degC = m.MV(value=df_learn[property_sources['temp_ret_ch__degC']].astype('float32').values)
        temp_ret_ch__degC.STATUS = 0  # No optimization
        temp_ret_ch__degC.FSTATUS = 1 # Use the measured values

        ##################################################################################################################
        # Fan speed and pump speed
        ##################################################################################################################

        # calculated fan speed percentage between min (0 %) and max (100 %)
        fan_speed__pct = m.CV(value=df_learn[property_sources['fan_speed__pct']].astype('float32').values)
        fan_speed__pct.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
        fan_speed__pct.FSTATUS = 1 # Use the measured values
						
        # hydronic pump speed in % of maximum pump speed         
        flow_dstr_pump_speed__pct = m.CV(value=df_learn[property_sources['flow_dstr_pump_speed__pct']].astype('float32').values) 
        flow_dstr_pump_speed__pct.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
        flow_dstr_pump_speed__pct.FSTATUS = 1 # Use the measured values

        ##################################################################################################################
        # Control targets: flow temperature and 'delta-T': difference between flow and return temperature
        ##################################################################################################################

        # Error between supply temperature and setpoint fo the supply temperature
        error_temp_delta_flow_flowset__K = m.Var(value=0.0)  # Initialize with a default value
        m.Equation(error_temp_delta_flow_flowset__K == temp_flow_ch_set__degC - temp_flow_ch__degC)

        # Error in 'delta-T' (difference between supply and return temperature)
        desired_temp_delta_flow_ret__K = m.Param(value=bldng_data['desired_temp_delta_flow_ret__K']) 
        error_temp_delta_flow_ret__K = m.Var(value=0.0)  # Initialize with a default value
        m.Equation(error_temp_delta_flow_ret__K == desired_temp_delta_flow_ret__K - (temp_flow_ch__degC - temp_ret_ch__degC))
    
        ##################################################################################################################
        # Control  algorithm 
        ##################################################################################################################

        match mode:
            case Learner.ControlMode.LEARN_ALGORITHMIC:

                # TO DO: consider accepting param_hints for parameters that don't need do be learned
                
                # Define variables to hold the rate of fan and pump speed changes
                # fan_rotations_gain__min_1 = m.Var(value=0)    # Rate of change for fan speed
                fan_rotations_gain__pct_min_1 = m.Var(value=0)    # Rate of change for fan speed
                flow_dstr_pump_speed_gain__pct_min_1 = m.Var(value=0)  # Rate of change for pump speed

                ##################################################################################################################
                # Algorithmic control 
                ##################################################################################################################
        
                # GEKKO constants for better code readability 
                TRUE = 1
                FALSE = 0
                
                ##################################################################################################################
                # Cooldown mode definitions 
                ##################################################################################################################

                # Temperature margins for cooldown hysteresis
                overheat_upper_margin_temp_flow__K = m.Param(value=5)                                        # Default overheating margin in K
                overheat_hysteresis__K = m.Param(value=bldng_data['overheat_hysteresis__K'])                 # Hysteresis, which might be boiler-specific
                cooldown_margin_temp_flow__K = overheat_hysteresis__K - overheat_upper_margin_temp_flow__K   # Default cooldown margin in K

                # Cooldown hysteresis: starts at crossing overheating margin, ends at crossing cooldown margin
                cooldown_condition = m.Var(value=FALSE)  # Initialize hysteresis state variable

                
                # Define the overheating and cooldown conditions
                overheat_condition = temp_flow_ch__degC - (temp_flow_ch_set__degC + overheat_upper_margin_temp_flow__K)
                cooldown_exit_condition = (temp_flow_ch_set__degC + cooldown_margin_temp_flow__K) - temp_flow_ch__degC
                no_heat_demand_condition = 0.5 - temp_flow_ch_set__degC
                
                # Cooldown state transitions
                m.Equation(
                    cooldown_condition == m.if3(
                        overheat_condition,  # Enter cooldown mode if overheat condition is positive
                        TRUE,                # Cooldown mode active
                        m.if3(
                            cooldown_exit_condition,  # Exit cooldown mode if this condition is positive
                            FALSE,                   # Cooldown mode inactive
                            cooldown_condition       # Maintain current state (hysteresis)
                        )
                    )
                )

                ##################################################################################################################
                # Post-pump run definitions 
                ##################################################################################################################

                
                # Define variables
                in_post_pump_run_condition = m.Var(value=0, integer=True)               # Boolean state variable
                post_pump_run_duration__min = 3                                         # Post pump run duration (in minutes); default to 3 minutes
                post_pump_run_duration__s = post_pump_run_duration__min * s_min_1       # Convert duration to seconds (add half a minute to make soure counter does not end at 0)
                post_pump_speed__pct = m.Param(value=bldng_data['post_pump_run__pct'])  # Post pump run speeds percentage, may be boiler-specific
                post_pump_run_expiration__s = m.Var(value=0)                            # Expiration time
                
                # Create a variable to represent the current simulation time
                current_time = m.Var(value=0)  # Start at time 0
                m.Equation(current_time.dt() == step__s)  # Increment current_time by step__s seconds

                # Define the post-pump run entry condition
                # Create post_pump_run_entry_condition in a single line using pandas operations
                post_pump_run_entry_condition = m.MV(value=(
                    (df_learn[property_sources['temp_flow_ch_set__degC']].shift(1, fill_value=0) > 0.5) &
                    (df_learn[property_sources['temp_flow_ch_set__degC']] == 0)
                ).astype(int).values)
                post_pump_run_entry_condition.STATUS = 0  # No optimization
                post_pump_run_entry_condition.FSTATUS = 1 # Use the measured values

                # Start post pump run timer (by calculating exporation time) whenever heat demand ends
                m.Equation(
                    post_pump_run_expiration__s.dt() == m.if3(
                        post_pump_run_entry_condition,
                        current_time + post_pump_run_duration__s,           # Start expiration timer
                        0                                                   # Else: retain current expiration time
                    )
                )

                timer_not_expired_condition = current_time - post_pump_run_expiration__s
                # Update in_post_pump_run_condition
                m.Equation(
                    in_post_pump_run_condition == m.if3(
                        timer_not_expired_condition,                 # Timer not expired
                        TRUE,                                        # Active
                        FALSE                                        # Inactive
                    )
                )

                
                ##################################################################################################################
                # Fan speed definitions 
                ##################################################################################################################
                
                # Max fan gain in in %
                fan_scale = bldng_data['fan_max_ch_rotations__min_1'] - bldng_data['fan_min_ch_rotations__min_1']
                fan_rotations_max_gain__pct_min_1 = m.FV(value=1500/fan_scale * 100,
                                                         lb=0,
                                                         ub=100,
                                                         # lb=100/fan_scale * 100, 
                                                         # ub=2000/fan_scale * 100
                                                        )                                # Initialize with value and bounds
                fan_rotations_max_gain__pct_min_1.STATUS = 1                             # Allow optimization
                fan_rotations_max_gain__pct_min_1.FSTATUS = 1                            # Use the initial value as a hint for the solver

                # Fan error threshold in K
                error_threshold_temp_delta_flow_flowset__K = m.FV(value=5,
                                                                  lb=0,
                                                                  ub=100,
                                                                  # lb=2,
                                                                  # ub=10
                                                                 )                       # Initialize with value and bounds
                error_threshold_temp_delta_flow_flowset__K.STATUS = 1                    # Allow optimization
                error_threshold_temp_delta_flow_flowset__K.FSTATUS = 1                   # Use the initial value as a hint for the solver
                
                # Conditional fan speed gain based on flow error threshold, with an enforced maximum
                fan_rotations_gain__pct_min_1 = m.Intermediate(
                    m.min2(
                        error_temp_delta_flow_flowset__K / error_threshold_temp_delta_flow_flowset__K * fan_rotations_max_gain__pct_min_1, 
                        fan_rotations_max_gain__pct_min_1  # max gain
                    )
                )
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

                # Max pump gain in % per minute
                flow_dstr_pump_speed_max_gain__pct_min_1 = m.FV(value=3,
                                                                lb=0,
                                                                ub=100,
                                                                # lb=1,
                                                                # ub=5
                                                               )                         # Initialize with value and bounds
                flow_dstr_pump_speed_max_gain__pct_min_1.STATUS = 1                      # Allow optimization
                flow_dstr_pump_speed_max_gain__pct_min_1.FSTATUS = 1                     # Use the initial value as a hint for the solver
                
                # Pump error threshold in K
                error_threshold_temp_delta_flow_ret__K = m.FV(value=2,
                                                              lb=0,
                                                              ub=100,
                                                              # lb=1,
                                                              # ub=5
                                                             )                           # Initialize with value and bounds
                error_threshold_temp_delta_flow_ret__K.STATUS = 1                        # Allow optimization
                error_threshold_temp_delta_flow_ret__K.FSTATUS = 1                       # Use the initial value as a hint for the solver

                # Conditional pump speed gain based on error threshold, with en enforced maximum
                flow_dstr_pump_speed_gain__pct_min_1 = m.Intermediate(
                    m.min2(
                        error_temp_delta_flow_ret__K / error_threshold_temp_delta_flow_ret__K * flow_dstr_pump_speed_max_gain__pct_min_1,
                        flow_dstr_pump_speed_max_gain__pct_min_1 # max gain
                    )
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


        
            case Learner.ControlMode.LEARN_PID:
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
                                ub=component_hints[term].get('upper_bound', None)
                            )
                            param.STATUS = 1  # Allow optimization
                            param.FSTATUS = 1  # Use the initial value as a hint for the solver
                        else:
                            # Create a fixed parameter for this term
                            param = m.Param(value=default)
            
                        # Store the parameter in the structured dictionary
                        pid_parameters[component][term] = param
            
               # PID control equations for fan speed
                m.Equation(
                    fan_speed__pct.dt() == (
                        pid_parameters['fan']['p'] * error_temp_delta_flow_flowset__K +                   # Proportional term
                        pid_parameters['fan']['i'] * m.integral(error_temp_delta_flow_flowset__K) +       # Integral term
                        pid_parameters['fan']['d'] * error_temp_delta_flow_flowset__K.dt()                # Derivative term
                    )
                )

                # PID control equations for pump speed
                m.Equation(
                    flow_dstr_pump_speed__pct.dt() == (
                        pid_parameters['pump']['p'] * error_temp_delta_flow_ret__K +                      # Proportional term
                        pid_parameters['pump']['i'] * m.integral(error_temp_delta_flow_ret__K) +          # Integral term
                        pid_parameters['pump']['d'] * error_temp_delta_flow_ret__K.dt()                   # Derivative term
                    )
                )

            case _:
                raise ValueError(f"Invalid ControlMode: {mode}")
    
        ##################################################################################################################
        # Solve the model to start the learning process
        ##################################################################################################################
        m.options.IMODE = 5        # Simultaneous Estimation 
        m.options.EV_TYPE = 2      # RMSE
        if max_iter is not None:   # retrict if needed to avoid waiting an eternity for unsolvable learning scenarios
            m.options.MAX_ITER = max_iter
        m.solve(disp=False)

        ##################################################################################################################
        # Store results of the learning process
        ##################################################################################################################
        
        # Initialize a DataFrame for learned time-varying properties
        df_predicted_properties = pd.DataFrame(index=df_learn.index)
        
        # Initialize a DataFrame for learned parameters (single row for metadata)
        df_learned_parameters = pd.DataFrame({
            'id': id, 
            'start': start,
            'end': end,
            'duration': timedelta(seconds=duration__s),
        }, index=[0])
        
        # Store learned time-varying data in DataFrame and calculate MAE and RMSE
        current_locals = locals() # current_locals is valid in list comprehensions and for loops, locals() is not. 
        for prop in (predict_props or set()) & set(current_locals.keys()):
            predicted_prop = f'predicted_{mode.value}_{prop}'
            df_predicted_properties.loc[:,predicted_prop] = np.asarray(current_locals[prop].value)
            
            # If the property was measured, calculate and store MAE and RMSE
            if prop in property_sources.keys() and property_sources[prop] in df_learn.columns:
                df_learned_parameters.loc[0, f'mae_{mode.value}_{prop}'] = mae(
                    df_learn[property_sources[prop]],  # Measured values
                    df_predicted_properties[predicted_prop]  # Predicted values
                )
                df_learned_parameters.loc[0, f'rmse_{mode.value}_{prop}'] = rmse(
                    df_learn[property_sources[prop]],  # Measured values
                    df_predicted_properties[predicted_prop]  # Predicted values
                )
                
        match mode:
            case Learner.ControlMode.LEARN_ALGORITHMIC:
                if learn_params: 
                    for param in learn_params & current_locals.keys():
                        learned_value = current_locals[param].value[0]
                        df_learned_parameters.loc[0, f'learned_{param}'] = learned_value
                        # If actual value exists, compute MAE
                        if actual_params is not None and param in actual_params:
                            df_learned_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_params[param])
            case Learner.ControlMode.LEARN_PID:
                for component, params in pid_parameters.items():
                    for param_name, value in params.items():
                        df_learned_parameters.loc[0, f'learned_{component}_K{param_name}'] = value[0]
                        #TO DO: support actual_params?
            case _:
                raise ValueError(f"Invalid ControlMode: {mode}")
            

        # Set MultiIndex on the DataFrame (id, start, end)
        df_learned_parameters.set_index(['id', 'start', 'end', 'duration'], inplace=True)

        m.cleanup()

        return df_learned_parameters, df_predicted_properties
        

    def learn_thermostat_control(
            df_learn,
            bldng_data: Dict = None,
            property_sources: Dict = None,
            param_hints: Dict = None,
            learn_params: Set[str] = {'thermostat_hysteresis__K'},
            actual_params: Dict = None,
            predict_props: Set[str] = {'temp_flow_ch_set__degC'},
            mode: ControlMode = ControlMode.LEARN_ALGORITHMIC,
            max_iter=10,
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        id, start, end, step__s, duration__s  = Learner.get_time_info(df_learn) 

        # TO DO: Check whether we need to use something from bldng_data

        ##################################################################################################################
        # Initialize GEKKO model
        ##################################################################################################################
        m = GEKKO(remote=False)
        m.time = np.arange(0, duration__s, step__s)

        ##################################################################################################################
        # Flow setpoint
        ##################################################################################################################

        temp_flow_ch_max__degC  = m.MV(value=df_learn[property_sources['temp_flow_ch_max__degC']].astype('float32').values)
        temp_flow_ch_max__degC .STATUS = 0  # No optimization
        temp_flow_ch_max__degC .FSTATUS = 1 # Use the measured values
        
        # Initial assumption: fixed setpoint; will be relaxed later
        temp_flow_ch_set__degC = m.Var(value=0)
        temp_flow_ch_set__degC.lower = 0  # Minimum value
        m.Equation(temp_flow_ch_set__degC <= temp_flow_ch_max__degC) # constraint to enforce the maximum limit dynamically

        ##################################################################################################################
        # Setpoint and indoor temperature
        ##################################################################################################################
        temp_set__degC = m.MV(value=df_learn[property_sources['temp_set__degC']].astype('float32').values)
        temp_set__degC.STATUS = 0  # No optimization
        temp_set__degC.FSTATUS = 1 # Use the measured values
        
        temp_indoor__degC = m.MV(value=df_learn[property_sources['temp_indoor__degC']].astype('float32').values)
        temp_indoor__degC.STATUS = 0  # No optimization
        temp_indoor__degC.FSTATUS = 1 # Use the measured values

        ##################################################################################################################
        # Control targets: 'delta-T': difference between indoor setpoint and indoor temperature
        ##################################################################################################################

        # Error between thermostat setpoint and indoor temperature
        error_temp_delta_indoor_set__K  = m.Var(value=0.0)  # Initialize with a default value
        m.Equation(error_temp_delta_indoor_set__K == temp_set__degC - temp_indoor__degC)

        ##################################################################################################################
        # Control  algorithm 
        ##################################################################################################################

        match mode:
            case Learner.ControlMode.LEARN_ALGORITHMIC:

                ##################################################################################################################
                # Algorithmic control; this implements a simple ON/OFF thermostat with hysteresis
                ##################################################################################################################
        
                # Thermostat hysteresis
                thermostat_hysteresis__K = m.FV(value=0.1,
                                               lb=0.0,
                                               ub=2.0)         # Thermostat hysteresis, which might be boiler-specific
                thermostat_hysteresis__K.STATUS = 1            # Allow optimization
                thermostat_hysteresis__K.FSTATUS = 1           # Use the initial value as a hint for the solver
                
                hysteresis_upper_margin__K = m.Intermediate(temp_set__degC + thermostat_hysteresis__K/2)
                hysteresis_lower_margin__K = m.Intermediate(temp_set__degC - thermostat_hysteresis__K/2)
        
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

            case Learner.ControlMode.LEARN_PID:
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
                            ub=bounds.get('upper_bound', None)
                        )
                        pid_parameters[component][term].STATUS = 1
                        pid_parameters[component][term].FSTATUS = 1
            
                # PID control equations for flow temperature setpoint 
                m.Equation(
                    temp_flow_ch_set__degC.dt() == (
                        pid_parameters['thermostat']['p'] * error_temp_delta_indoor_set__K +              # Proportional term
                        pid_parameters['thermostat']['i'] * m.integral(error_temp_delta_indoor_set__K) +  # Integral term
                        pid_parameters['thermostat']['d'] * error_temp_delta_indoor_set__K.dt()           # Derivative term
                    )
                )

            case _:
                raise ValueError(f"Invalid ControlMode: {mode}")
    
        ##################################################################################################################
        # Solve the model to start the learning process
        ##################################################################################################################
        m.options.IMODE = 5        # Simultaneous Estimation 
        m.options.EV_TYPE = 2      # RMSE
        if max_iter is not None:   # retrict if needed to avoid waiting an eternity for unsolvable learning scenarios
            m.options.MAX_ITER = max_iter
        m.solve(disp=False)

        ##################################################################################################################
        # Store results of the learning process
        ##################################################################################################################
        

        # Initialize a DataFrame for learned time-varying properties
        df_predicted_properties = pd.DataFrame(index=df_learn.index)
        
        # Initialize a DataFrame for learned parameters (single row for metadata)
        df_learned_parameters = pd.DataFrame({
            'id': id, 
            'start': start,
            'end': end,
            'duration': timedelta(seconds=duration__s),
        }, index=[0])
        
        # Store learned time-varying data in DataFrame and calculate MAE and RMSE
        current_locals = locals() # current_locals is valid in list comprehensions and for loops, locals() is not. 
        for prop in (predict_props or set()) & set(current_locals.keys()):
            predicted_prop = f'predicted_{mode.value}_{prop}'
            df_predicted_properties.loc[:,predicted_prop] = np.asarray(current_locals[prop].value)
            
            # If the property was measured, calculate and store MAE and RMSE
            if prop in property_sources.keys() and property_sources[prop] in df_learn.columns:
                df_learned_parameters.loc[0, f'mae_{mode.value}_{prop}'] = mae(
                    df_learn[property_sources[prop]],  # Measured values
                    df_predicted_properties[predicted_prop]  # Predicted values
                )
                df_learned_parameters.loc[0, f'rmse_{mode.value}_{prop}'] = rmse(
                    df_learn[property_sources[prop]],  # Measured values
                    df_predicted_properties[predicted_prop]  # Predicted values
                )

        if learn_params: 
            match mode:
                case Learner.ControlMode.LEARN_ALGORITHMIC:
                    for param in learn_params & current_locals.keys():
                        learned_value = current_locals[param].value[0]
                        df_learned_parameters.loc[0, f'learned_{param}'] = learned_value
                        # If actual value exists, compute MAE
                        if actual_params is not None and param in actual_params:
                            df_learned_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_params[param])
    
                case Learner.ControlMode.LEARN_PID:
                    for component, params in pid_parameters.items():
                        for param_name, value in params.items():
                            df_learned_parameters.loc[0, f'learned_{component}_K{param_name}'] = value[0]
                case _:
                    raise ValueError(f"Invalid ControlMode: {mode}")
            

        # Set MultiIndex on the DataFrame (id, start, end)
        df_learned_parameters.set_index(['id', 'start', 'end', 'duration'], inplace=True)

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