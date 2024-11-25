from datetime import datetime, timedelta
from typing import List, Tuple
import pandas as pd
import numpy as np
import math
from gekko import GEKKO
from tqdm.notebook import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numbers
import logging

from pythermalcomfort.models import pmv_ppd

from nfh_utils import *

# TEST: this line should only be visible to accounts with explicit access to this private repo

class LearnError(Exception):
    def __init__(self, message):
        self.message = message
        
class Learner():

    
    def get_longest_sane_streak(df_data:pd.DataFrame,
                                id,
                                learn_period_start,
                                learn_period_end,
                                sanity_threshold_timedelta:timedelta=timedelta(hours=24)) -> pd.DataFrame:
        
        df_learn_period  = df_data.loc[(df_data.index.get_level_values('id') == id) & 
                                       (df_data.index.get_level_values('timestamp') >= learn_period_start) & 
                                       (df_data.index.get_level_values('timestamp') < learn_period_end)]
        
        learn_period_len = len(df_learn_period)

        # Check for enough values
        if learn_period_len <=1:
            logging.info(f'No values for id: {id} between {learn_period_start} and {learn_period_end}; skipping...')
            return None

        # Check for at least two sane values
        if (df_learn_period['sanity'].sum()) <=1: #counts the number of sane rows, since True values will be coutnd as 1 in suming
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
        if ((timestamps.max() - timestamps.min()) < sanity_threshold_timedelta):
            logging.info(f'Longest streak duration to short for id: {id}; shorter than {sanity_threshold_timedelta} between {timestamps.min()} and {timestamps.max()}; skipping...')
            return None

        return df_longest_streak

    
    def create_job_list(df_data: pd.DataFrame,
                        learn_period__d: int,
                        req_props: list,
                        property_sources: dict,
                        sanity_threshold_timedelta:timedelta=timedelta(hours=24)) -> pd.DataFrame:
        """
        Create a list of jobs (id, learn_period_start, learn_period_end) to be processed.
        """
        jobs = []
        ids = df_data.index.unique('id').dropna()
        start_analysis_period = df_data.index.unique('timestamp').min().to_pydatetime()
        end_analysis_period = df_data.index.unique('timestamp').max().to_pydatetime()
        daterange_frequency = str(learn_period__d) + 'D'
        learn_period_starts = pd.date_range(start=start_analysis_period, end=end_analysis_period, inclusive='both', freq=daterange_frequency)
    
        # Perform sanity check
        if req_props is None:  # If req_col not set, use all property sources
            req_col = list(property_sources.values())
        else:
            req_col = [property_sources[prop] for prop in req_props if prop in property_sources]
        if not req_col:
            df_data['sanity'] = True  # No required columns, mark all as sane
        else:
            df_data['sanity'] = ~df_data[req_col].isna().any(axis="columns")
        
        for id in tqdm(ids, desc="Identifying learning jobs"):
            for learn_period_start in learn_period_starts:
                learn_period_end = min(learn_period_start + timedelta(days=learn_period__d), end_analysis_period)
    
                # Learn only for the longest streak of sane data
                #TODO: remove df_learn slicing, directly deliver start & end time per period
                df_learn = Learner.get_longest_sane_streak(df_data, id, learn_period_start, learn_period_end, sanity_threshold_timedelta)
                if df_learn is None:
                    continue
                
                jobs.append((
                    id,
                    df_learn.index.unique('timestamp').min().to_pydatetime(),
                    df_learn.index.unique('timestamp').max().to_pydatetime()
                ))
    
        # Drop the columns created during the process
        # df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'], inplace=True)
        df_data.drop(columns=['sanity'], inplace=True)
    
        # Return jobs as a DataFrame
        return pd.DataFrame(jobs, columns=['id', 'start', 'end']).set_index(['id', 'start', 'end'])

    
    def create_streak_job_list(
        df_data: pd.DataFrame,
        req_props: list,
        property_sources: dict,
        duration_threshold_timedelta: timedelta = timedelta(minutes=30)) -> pd.DataFrame:
        """
        Create a list of jobs (id, start, end) based on consecutive streaks of data
        with non-NaN values in required columns and duration above a threshold.
    
        Parameters:
            df_data (pd.DataFrame): Input data with a MultiIndex ('id', 'timestamp').
            req_col (list): List of required properties to check for non-NaN/NA values.
            duration_threshold (timedelta): Minimum duration for a streak to be included.
    
        Returns:
            pd.DataFrame: A DataFrame containing job list with columns ['id', 'start', 'end'].
        """
        # Ensure required columns are specified
        if req_props is None:  # If req_col not set, use all property sources
            req_col = list(property_sources.values())
        else:
            req_col = [property_sources[prop] for prop in req_props if prop in property_sources]
    
        # Initialize job list
        jobs = []
        
        # Iterate over each unique id
        for id_, group in tqdm(df_data.groupby(level='id'), desc="Identifying learning jobs"):
            # Filter for rows where all required columns are not NaN
            group = group.droplevel('id')  # Drop 'id' level for easier handling
            group = group.loc[group[req_col].notna().all(axis=1)]
    
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
                if streak_duration >= duration_threshold_timedelta:
                    jobs.append((id_, start_time, end_time))
    
        # Convert jobs to DataFrame
        job_df = pd.DataFrame(jobs, columns=['id', 'start', 'end'])
        job_df['duration'] =  job_df['end'] - job_df['start']
        job_df['duration__min'] = job_df['duration'].dt.total_seconds() / 60  # Converts to minutes
        
        return job_df.set_index(['id', 'start', 'end'])
    

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
            heat_tr_dstr_avg__W_K_1: Average heat transfer coefficient of the distribution system in W/K.
            th_mass_dstr_avg__Wh_K_1: Average thermal mass of the distribution system in Wh/K.
    
        Returns:
            dict: A dictionary containing the actual parameter values.
        """
        actual_parameter_values = {
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
            actual_parameter_values['heat_tr_bldng_cond__W_K_1'] = id // 1e5
            actual_parameter_values['th_inert_bldng__h'] = (id % 1e5) // 1e2
            actual_parameter_values['aperture_sol__m2'] = id % 1e2
            actual_parameter_values['th_mass_bldng__Wh_K_1'] = (
                actual_parameter_values['heat_tr_bldng_cond__W_K_1'] *
                actual_parameter_values['th_inert_bldng__h']
            )
    
        return actual_parameter_values
    
 
    def save_job_results_to_parquet(id, start, stop, df_learned_job_parameters, df_learned_job_properties, results_dir):
        """Save the learned parameters and properties for a specific job."""
        
        # Format start and stop times as strings
        start_str = start.strftime('%Y%m%d_%H%M%S')
        stop_str = stop.strftime('%Y%m%d_%H%M%S')
    
        # File paths for learned parameters and learned properties
        learned_job_parameters_file_path = os.path.join(results_dir, f'learned-parameters-job-{id}-{start_str}-{stop_str}.parquet')
        learned_job_properties_file_path = os.path.join(results_dir, f'learned-properties-job-{id}-{start_str}-{stop_str}.parquet')
    
        # Save learned parameters
        if df_learned_job_parameters is not None:
            if os.path.exists(learned_job_parameters_file_path):
                # Read existing learned parameters
                df_existing_params = pd.read_parquet(learned_job_parameters_file_path)
    
                # Log info for debugging
                logging.info(f"Parameters already learned: {df_existing_params.columns}; to add {df_learned_job_parameters.columns}")
                logging.info(f"Shape already learned: {df_existing_params.shape}; shape to add {df_learned_job_parameters.shape}")
        
                # Concatenate new learned parameters horizontally
                # Aligns on index and adds new columns as needed
                df_existing_params = pd.concat([df_existing_params, df_learned_job_parameters], axis=1)
        
                # Ensure no duplicate columns
                df_existing_params = df_existing_params.loc[:,~df_existing_params.columns.duplicated()]
            else:
                # If no existing parameters, simply use the learned ones
                df_existing_params = df_learned_job_parameters
    
                # Save the updated learned parameters
                df_existing_params.to_parquet(learned_job_parameters_file_path)
                logging.info(f"Updated learned parameters for job ID {id} (from {start} to {stop}) in {learned_job_parameters_file_path}")
        
        # Save learned properties
        if df_learned_job_properties is not None:
            if os.path.exists(learned_job_properties_file_path):
                # Read existing learned properties
                df_existing_props = pd.read_parquet(learned_job_properties_file_path)
        
                # Log info for debugging
                logging.info(f"Properties already learned: {df_existing_props.columns}; to add {df_learned_job_properties.columns}")
                logging.info(f"Shape already learned: {df_existing_props.shape}; shape to add {df_learned_job_properties.shape}")
        
                # Concatenate new and existing properties, avoiding duplicate columns
                # Use `join='outer'` to ensure new columns are added, but duplicates are avoided
                df_combined_props = pd.concat([df_existing_props, df_learned_job_properties], axis=0, join='outer')
        
                # Ensure no duplicate columns by dropping columns with the same name that might have been reintroduced
                df_combined_props = df_combined_props.loc[:, ~df_combined_props.columns.duplicated()]
        
                # Save the updated learned properties
                df_combined_props.to_parquet(learned_job_properties_file_path)
                logging.info(f"Updated learned properties for job ID {id} (from {start} to {stop}) in {learned_job_properties_file_path}")
            else:
                # If the file does not exist, save the new learned properties
                df_learned_job_properties.to_parquet(learned_job_properties_file_path)
                logging.info(f"Saved new learned properties for job ID {id} (from {start} to {stop}) to {learned_job_properties_file_path}")
    
    def save_to_parquet(id, df_learned_job_parameters, df_learned_job_properties, df_data, results_dir):
        """Save the learned parameters and properties for a specific id to Parquet."""
        df_learned_job_parameters.to_parquet(os.path.join(results_dir, f'learned-parameters-per-period-{id}.parquet'))
        if df_learned_job_properties is not None:
            df_learned_job_properties.to_parquet(os.path.join(results_dir, f'learned-properties-{id}.parquet'))
        logging.info(f'Saved results for ID {id} to {results_dir}')
        
        # Save df_data if needed (incremental saving)
        df_data.to_parquet(os.path.join(results_dir, f'df_data_{id}.parquet'))
    
    def final_save_to_parquet(df_learned_parameters, df_data, results_dir):
        """Final save of all aggregated results after processing all ids."""
        df_learned_parameters.to_parquet(os.path.join(results_dir, 'results_per_period_final.parquet'))
        logging.info(f'Final results per period saved to {results_dir}/results_per_period_final.parquet')
        
        df_data.to_parquet(os.path.join(results_dir, 'results_final.parquet'))
        logging.info(f'Final df_data saved to {results_dir}/results_final.parquet')    

    def learn_ventilation(df_learn,
                          id, start, end,
                          duration__s, step__s,
                          property_sources, hints, learn, learn_change_interval__min,
                          bldng_data,
                          actual_parameter_values
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:

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
        ventilation__dm3_s_1 = m.MV(value=hints['ventilation_default__dm3_s_1'],
                                    lb=0.0, 
                                    ub=hints['ventilation_max__dm3_s_1_m_2'] * floors__m2)
        ventilation__dm3_s_1.STATUS = 1  # Allow optimization
        ventilation__dm3_s_1.FSTATUS = 1 # Use the measured values
        
        if learn_change_interval__min is not None:
            ventilation__dm3_s_1.MV_STEP_HOR = learn_change_interval__min
        
        air_changes_vent__s_1 = m.Intermediate(ventilation__dm3_s_1 / (bldng__m3 * dm3_m_3))

        # Wind-induced (infiltration) CO₂ concentration loss indoors
        wind__m_s_1 = m.MV(value=df_learn[property_sources['wind__m_s_1']].astype('float32').values)
        wind__m_s_1.STATUS = 0  # No optimization
        wind__m_s_1.FSTATUS = 1 # Use the measured values
    
        if 'aperture_inf__cm2' in learn:
            aperture_inf__cm2 = m.FV(value=hints['aperture_inf__cm2'], lb=0, ub=100000.0)
            aperture_inf__cm2.STATUS = 1  # Allow optimization
            aperture_inf__cm2.FSTATUS = 1 # Use the initial value as a hint for the solver
        else:
            aperture_inf__cm2 = m.Param(value=hints['aperture_inf__cm2'])

        air_inf__m3_s_1 = m.Intermediate(wind__m_s_1 * aperture_inf__cm2 / cm2_m_2)        
        air_changes_inf__s_1 = m.Intermediate(air_inf__m3_s_1 / bldng__m3)

        # Total losses of CO₂ concentration indoors
        air_changes_total__s_1 = m.Intermediate(air_changes_vent__s_1 + air_changes_inf__s_1)
        co2_elevation__ppm = m.Intermediate(co2_indoor__ppm - hints['co2_outdoor__ppm'])
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
        
        # Initialize a DataFrame for learned time-varying properties
        df_learned_job_properties = pd.DataFrame(index=df_learn.index)

        # Store learned time-varying data in DataFrame
        df_learned_job_properties.loc[:,'learned_ventilation__dm3_s_1'] = np.asarray(ventilation__dm3_s_1)

        # Initialize a DataFrame, even for a single learned parameter (one row with id, start, end), for consistency
        df_learned_job_parameters = pd.DataFrame({
            'id': id, 
            'start': start,
            'end': end
        }, index=[0])
        
        param = 'aperture_inf__cm2'
    
        df_learned_job_parameters.loc[0, f'learned_co2_{param}'] = aperture_inf__cm2.value[0]
    
        # If actual value exists, compute MAE
        if actual_parameter_values is not None and param in actual_parameter_values:
            df_learned_job_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_parameter_values[param])

        # Set MultiIndex on the DataFrame (id, start, end)
        df_learned_job_parameters.set_index(['id', 'start', 'end'], inplace=True)

        m.cleanup()

        return df_learned_job_parameters, df_learned_job_properties

    
    def learn_heat_distribution(id, start, end,
                                df_learn,
                                duration__s, step__s,
                                property_sources, hints) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info(f"learn heat distribution for id {df_learn.index.get_level_values('id')[0]}, from  {df_learn.index.get_level_values('timestamp').min()} to {df_learn.index.get_level_values('timestamp').max()}")

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
    
        # Learn heat distribution system parameters
        heat_tr_dstr__W_K_1 = m.FV(value=hints['heat_tr_dstr__W_K_1'], lb=0, ub=1000)
        heat_tr_dstr__W_K_1.STATUS = 1  # Allow optimization
        heat_tr_dstr__W_K_1.FSTATUS = 1 # Use the initial value as a hint for the solver

        th_mass_dstr__Wh_K_1 = m.FV(value=hints['th_mass_dstr__Wh_K_1'], lb=0, ub=10000)
        th_mass_dstr__Wh_K_1.STATUS = 1  # Allow optimization
        th_mass_dstr__Wh_K_1.FSTATUS = 1 # Use the initial value as a hint for the solver
    
        temp_ret_ch__degC = m.CV(value=df_learn[property_sources['temp_ret_ch__degC']].astype('float32').values)
        temp_ret_ch__degC.STATUS = 1  # Include this variable in the optimization (enabled for fitting)
        temp_ret_ch__degC.FSTATUS = 1 # Use the measured values

        temp_indoor__degC = m.MV(value=df_learn[property_sources['temp_indoor__degC']].astype('float32').values)
        temp_indoor__degC.STATUS = 0  # No optimization
        temp_indoor__degC.FSTATUS = 1 # Use the measured values
    
        # Define an initial value for temp_dstr__degC
        temp_dstr__degC = m.Var(value=(df_learn[property_sources['temp_flow_ch__degC']].iloc[0] + 
                                       df_learn[property_sources['temp_ret_ch__degC']].iloc[0]) / 2)
        
        heat_dstr__W = m.Intermediate(heat_tr_dstr__W_K_1 * (temp_dstr__degC - temp_indoor__degC))

        m.Equation(temp_dstr__degC.dt() == (heat_ch__W - heat_dstr__W) / (th_mass_dstr__Wh_K_1 * s_h_1))
        
        ##################################################################################################################
        # Solve the model to start the learning process
        ##################################################################################################################
        m.options.IMODE = 5        # Simultaneous Estimation 
        m.options.EV_TYPE = 2      # RMSE
        m.solve(disp=False)

        ##################################################################################################################
        # Store results of the learning process
        ##################################################################################################################
        
        # Initialize a DataFrame for learned time-varying properties
        df_learned_job_properties = pd.DataFrame(index=df_learn.index)

        # Initialize a DataFrame, even for a single learned parameter (one row with id, start, end), for consistency
        df_learned_job_parameters = pd.DataFrame({
            'id': id, 
            'start': start,
            'end': end
        }, index=[0])

        # Store learned heat transmissivity of the heat distribution system
        param = 'heat_tr_dstr__W_K_1'
        df_learned_job_parameters.loc[0, f'learned_{param}'] = heat_tr_dstr__W_K_1.value[0]

        # Store learned thermal mass of the heat distribution system
        param = 'th_mass_dstr__Wh_K_1'
        df_learned_job_parameters.loc[0, f'learned_{param}'] = th_mass_dstr__Wh_K_1.value[0]

        # Set MultiIndex on the DataFrame (id, start, end)
        df_learned_job_parameters.set_index(['id', 'start', 'end'], inplace=True)

        m.cleanup()

        return df_learned_job_parameters, df_learned_job_properties
        
        
        
    def learn_thermal_parameters(df_learn,
                                 id, start, end,
                                 duration__s, step__s,
                                 property_sources, hints, learn,
                                 bldng_data,
                                 actual_parameter_values) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Learn thermal parameters for a building's heating system using GEKKO.
        
        Parameters:
        df_learn (pd.DataFrame): DataFrame containing the time series data to be used for learning.
        property_sources (dict): Dictionary mapping property names to their corresponding columns in df_learn.
        hints (dict): Dictionary containing default values for the various parameters.
        learn (list): List of parameters to be learned.
        bldng_data: dictionary containing at least:
        - bldng__m3 (float): Volume of the building in m3.
        """
        
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

        # # If possible, use the learned heat transmissivity of the heat distribution system, otherwise use a generic value
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
    
        if 'aperture_sol__m2' in learn:
            aperture_sol__m2 = m.FV(value=hints['aperture_sol__m2'], lb=1, ub=100)
            aperture_sol__m2.STATUS = 1  # Allow optimization
            aperture_sol__m2.FSTATUS = 1 # Use the initial value as a hint for the solver
        else:
            aperture_sol__m2 = m.Param(value=hints['aperture_sol__m2'])
    
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
        heat_g_dhw__W = m.Intermediate(g_use_dhw_hhv__W * hints['eta_dhw_hhv__W0'] * hints['frac_remain_dhw__0'])

        # Heat gains from cooking
        heat_g_cooking__W = m.Param(hints['g_use_cooking_hhv__W'] * hints['eta_cooking_hhv__W0'] * hints['frac_remain_cooking__0'])

        # Heat gains from electricity
        # we assume all electricity is used indoors and turned into heat
        heat_e__W = m.MV(value = df_learn[property_sources['e__W']].astype('float32').values)
        heat_e__W.STATUS = 0  # No optimization
        heat_e__W.FSTATUS = 1 # Use the measured values

        # Heat gains from occupants
        if 'ventilation__dm3_s_1' in learn:
            # calculate using actual occupancy and average heat gain per occupant
            occupancy__p = m.MV(value = df_learn[property_sources['occupancy__p']].astype('float32').values)
            occupancy__p.STATUS = 0  # No optimization
            occupancy__p.FSTATUS = 1 # Use the measured values
            heat_int_occupancy__W = m.Intermediate(occupancy__p * hints['heat_int__W_p_1'])
        else:
            # calculate using average occupancy and average heat gain per occupant
            heat_int_occupancy__W = m.Param(hints['occupancy__p'] * hints['heat_int__W_p_1'])

        # Sum of all 'internal' heat gains 
        heat_int__W = m.Intermediate(heat_g_dhw__W + heat_g_cooking__W + heat_e__W + heat_int_occupancy__W)
        
        ##################################################################################################################
        # Conductive heat losses
        ##################################################################################################################
    
        if 'heat_tr_bldng_cond__W_K_1' in learn:
            heat_tr_bldng_cond__W_K_1 = m.FV(value=hints['heat_tr_bldng_cond__W_K_1'], lb=0, ub=1000)
            heat_tr_bldng_cond__W_K_1.STATUS = 1
            heat_tr_bldng_cond__W_K_1.FSTATUS = 1 # Use the initial value as a hint for the solver
        else:
            heat_tr_bldng_cond__W_K_1 = hints['heat_tr_bldng_cond__W_K_1']
    
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
    
        if 'aperture_inf__cm2' in learn:
            aperture_inf__cm2 = m.FV(value=hints['aperture_inf__cm2'], lb=0, ub=100000.0)
            aperture_inf__cm2.STATUS = 1  # Allow optimization
            aperture_inf__cm2.FSTATUS = 1 # Use the initial value as a hint for the solver
        else:
            aperture_inf__cm2 = m.Param(value=hints['aperture_inf__cm2'])
    
        air_inf__m3_s_1 = m.Intermediate(wind__m_s_1 * aperture_inf__cm2 / cm2_m_2)
        heat_tr_bldng_inf__W_K_1 = m.Intermediate(air_inf__m3_s_1 * air_room__J_m_3_K_1)
        heat_loss_bldng_inf__W = m.Intermediate(heat_tr_bldng_inf__W_K_1 * indoor_outdoor_delta__K)
    
        if 'ventilation__dm3_s_1' in learn:
            # in this model, we treat ventilation__dm3_s_1 as if it were measured, but it was learned earlier learn_property_ventilation_rate()  
            ventilation__dm3_s_1 = m.MV(value=df_learn['learned_ventilation__dm3_s_1'].astype('float32').values)
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
                    
        # Thermal inertia of the building
        if 'th_inert_bldng__h' in learn:
            # Learn thermal inertia
            th_inert_bldng__h = m.FV(value = hints['th_inert_bldng__h'], lb=(10), ub=(1000))
            th_inert_bldng__h.STATUS = 1  # Allow optimization
            th_inert_bldng__h.FSTATUS = 1 # Use the initial value as a hint for the solver
        else:
            # Do not learn thermal inertia of the building, but use a fixed value based on hint
            th_inert_bldng__h = m.Param(value = hints['th_inert_bldng__h'])
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

        # Initialize a DataFrame for learned time-varying properties
        df_learned_job_properties = pd.DataFrame(index=df_learn.index)
    
        # Store learned time-varying data in DataFrame
        df_learned_job_properties.loc[:,'learned_temp_indoor__degC'] = np.asarray(temp_indoor__degC)
    
        # If 'learned_temp_dstr__degC' is computed, include it as well
        if 'heat_tr_dstr__W_K_1' in learn or 'th_mass_dstr__J_K_1' in learn:
            df_learned_job_properties.loc[:,'learned_temp_dstr__degC'] = np.asarray(temp_dstr__degC)
    
        # Initialize a DataFrame for learned thermal parameters (one row with id, start, end)
        df_learned_job_parameters = pd.DataFrame({
            'id': id, 
            'start': start,
            'end': end
        }, index=[0])
    
        # Loop over the learn list and store learned values and calculate MAE if actual value is available
        for param in learn: 
            if param != 'ventilation__dm3_s_1':
                if param in locals():
                    learned_value = locals()[param].value[0]
                    df_learned_job_parameters.loc[0, f'learned_{param}'] = learned_value
    
                    # If actual value exists, compute MAE
                    if actual_parameter_values is not None and param in actual_parameter_values:
                        df_learned_job_parameters.loc[0, f'mae_{param}'] = abs(learned_value - actual_parameter_values[param])

        # Calculate MAE and RMSE for indoor temperature
        if 'temp_indoor__degC' in property_sources and 'learned_temp_indoor__degC' in df_learned_job_properties.columns:
            df_learned_job_parameters.loc[0, 'mae_temp_indoor__degC'] = mae(
                df_learn[property_sources['temp_indoor__degC']],  # the measured indoor temperatures 
                df_learned_job_properties['learned_temp_indoor__degC']  # the predicted indoor temperatures
            )
    
            df_learned_job_parameters.loc[0, 'rmse_temp_indoor__degC'] = rmse(
                df_learn[property_sources['temp_indoor__degC']],  # the measured indoor temperatures
                df_learned_job_properties['learned_temp_indoor__degC']  # the predicted indoor temperatures
            )
    
        # Calculate periodical averages, which include Energy Case metrics
        properties_mean = [
            'temp_set__degC',
            'temp_flow__degC',
            'temp_ret__degC',
            'comfortable__bool',
            'temp_indoor__degC',
            'temp_outdoor__degC',
            'temp_flow_ch_max__degC'
        ]
    
        for prop in properties_mean:
            # Create variable names dynamically
            # Determine the result column name based on whether the property ends with '__bool'
            if prop.endswith('__bool'):
                result_col = f"learned_avg_{prop[:-6]}__0"  # Remove '__bool' and add '__0'
            else:
                result_col = f"learned_avg_{prop}"
    
            # Use prop directly if it starts with 'calculated_'
            source_col = prop if prop.startswith('calculated_') else property_sources[prop]
            mean_value = df_learn[source_col].mean()
            df_learned_job_parameters.loc[0, result_col] = mean_value

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

        for var in sim_arrays_mean:
            # Create variable names dynamically
            result_col = f"learned_avg_{var}"
            mean_value = np.asarray(locals()[var]).mean()
            df_learned_job_parameters.loc[0, result_col] = mean_value

        # Calculate Carbon Case metrics
        df_learned_job_parameters.loc[0, 'learned_avg_co2_ch__g_s_1'] = (
            (df_learned_job_parameters.loc[0, 'learned_avg_g_use_ch_hhv__W'] 
             * 
             (co2_wtw_groningen_gas_std_nl_avg_2024__g__m_3 / gas_groningen_nl_avg_std_hhv__J_m_3)
            )
            +
            (df_learned_job_parameters.loc[0, 'learned_avg_e_use_ch__W'] 
             * 
             co2_wtw_e_onbekend_nl_avg_2024__g__kWh_1
            )
        )
        
        # Set MultiIndex on the DataFrame (id, start, end)
        df_learned_job_parameters.set_index(['id', 'start', 'end'], inplace=True)    

        m.cleanup()
    
        # Return both DataFrames: learned time-varying properties and learned fixed parameters
        return df_learned_job_parameters, df_learned_job_properties

    def merge_learned(df1: pd.DataFrame, df2: pd.DataFrame, index_columns: list) -> pd.DataFrame:
        """
        Merges two multi-index DataFrames on specified index columns and any other common columns,
        avoiding duplicate columns from the second DataFrame.
        
        Parameters:
        - df1: First DataFrame (e.g., learned job parameters or properties)
        - df2: Second DataFrame (e.g., learned thermal job parameters or properties)
        - index_columns: List of columns to merge on (e.g., ['id', 'start', 'end'] or ['id', 'timestamp'])
        
        Returns:
        - Merged DataFrame with combined parameters or properties.
        """
        # Reset index to bring index columns into the DataFrame as regular columns
        df1_reset = df1.reset_index()
        df2_reset = df2.reset_index()
    
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
            df_merged = df_merged.loc[:,~df_merged.columns.duplicated()]
    
            # Restore the original index
            df_merged.set_index(index_columns, inplace=True)
            
            return df_merged
        else:
            raise KeyError("Not all index_columns are present in both DataFrames.")

    
    def analyze_job(id, start, end, 
                    df_learn, 
                    property_sources, hints, learn, learn_change_interval__min,
                    bldng_data,
                    actual_parameter_values,
                    results_dir):
        logging.info(f'Analyzing job for ID: {id}, Building data: {bldng_data}')

        if df_learn is None or df_learn.empty:
            logging.warning(f"No data available for job ID: {id}. Skipping job.")
            return None, None
    
        # Determine the learning period
        learn_streak_period_start = df_learn.index.get_level_values('timestamp').min()
        learn_streak_period_end = df_learn.index.get_level_values('timestamp').max()
        learn_streak_period_len = len(df_learn)
    
        # Calculate step__s and MV_STEP_HOR
        step__s = ((learn_streak_period_end - learn_streak_period_start).total_seconds() / 
                    (learn_streak_period_len - 1))
    
        if learn_change_interval__min is None:
            learn_change_interval__min = np.nan
            MV_STEP_HOR = 1
        else:
            # Ceiling integer division
            MV_STEP_HOR = -((learn_change_interval__min * 60) // -step__s)
    
        logging.info(f'MV_STEP_HOR for ID {id}: {MV_STEP_HOR}')
    
        duration__s = step__s * learn_streak_period_len
        
        df_learned_job_parameters = pd.DataFrame()
        df_learned_job_properties = pd.DataFrame()
        
        learned_job_parameters_file_path = os.path.join(results_dir, 
            f'learned-parameters-job-{id}-{start.strftime("%Y%m%d_%H%M%S")}-{end.strftime("%Y%m%d_%H%M%S")}.parquet')
        learned_job_properties_file_path = os.path.join(results_dir, 
            f'learned-properties-job-{id}-{start.strftime("%Y%m%d_%H%M%S")}-{end.strftime("%Y%m%d_%H%M%S")}.parquet')

        # Check if ventilation learning is needed
        if 'ventilation__dm3_s_1' in learn:
            ventilation_learned = False
    
            # Check if ventilation results already exist
            if os.path.exists(learned_job_parameters_file_path) and os.path.exists(learned_job_properties_file_path):
                df_learned_vent_job_parameters = pd.read_parquet(learned_job_parameters_file_path)
                df_learned_vent_job_properties = pd.read_parquet(learned_job_properties_file_path)
                if 'learned_ventilation__dm3_s_1' in df_learned_vent_job_properties.columns:
                    logging.info(f"Ventilation results already learned for job ID {id} (from {start} to {end}).")
                    ventilation_learned = True
    
            if not ventilation_learned:
                # Learn ventilation rates
                logging.info(f"Analyzing ventilation rates for job ID {id} (from {start} to {end})...")
                df_learned_vent_job_parameters, df_learned_vent_job_properties = Learner.learn_ventilation(
                    df_learn,
                    id, start, end,
                    duration__s,
                    step__s,
                    property_sources,
                    hints,
                    learn,
                    learn_change_interval__min,
                    bldng_data,
                    actual_parameter_values=actual_parameter_values
                )
    
                df_learned_job_parameters = df_learned_vent_job_parameters
                df_learned_job_properties = df_learned_vent_job_properties

                # Storing learned ventilation rates in df_learn
                df_learn.loc[:,'learned_ventilation__dm3_s_1'] = df_learned_vent_job_properties['learned_ventilation__dm3_s_1'].values
                logging.info(f"Wrote ventilation rates to df_learn for {id} from {learn_streak_period_start} to {learn_streak_period_end}")

                # Save results for ventilation
                Learner.save_job_results_to_parquet(id, start, end, df_learned_job_parameters, df_learned_job_properties, results_dir)
                
        # Check if thermal parameters need to be learned
        thermal_learned = False
    
        if os.path.exists(learned_job_parameters_file_path) and os.path.exists(learned_job_properties_file_path):
            df_learned_thermal_job_parameters = pd.read_parquet(learned_job_parameters_file_path)
            df_learned_thermal_job_properties = pd.read_parquet(learned_job_properties_file_path)
            if 'learned_temp_indoor__degC' in df_learned_thermal_job_properties.columns:
                logging.info(f"Thermal parameters already learned for job ID {id} (from {start} to {end}).")
                thermal_learned = True
    
        if not thermal_learned:
            # Learn thermal parameters
            logging.info(f"Analyzing thermal properties for job ID {id} (from {start} to {end})...")
            df_learned_thermal_job_parameters, df_learned_thermal_job_properties = Learner.learn_thermal_parameters(
                df_learn,
                id, start, end,
                duration__s,
                step__s,
                property_sources,
                hints,
                learn,
                bldng_data,
                actual_parameter_values=actual_parameter_values
            )
    
            # Add newly learned to already learned job parameters
            df_learned_job_parameters = Learner.merge_learned(df_learned_job_parameters, df_learned_thermal_job_parameters, ['id', 'start', 'end'])
            
            # Add newly learned to already learned job properties
            df_learned_job_properties = Learner.merge_learned(df_learned_job_properties, df_learned_thermal_job_properties, ['id', 'timestamp'])
   
            # Save results for thermal parameters
            Learner.save_job_results_to_parquet(id, start, end, df_learned_job_parameters, df_learned_job_properties, results_dir)
            
        return df_learned_job_parameters, df_learned_job_properties

    
    @staticmethod
    def learn_heat_performance_signature(df_data:pd.DataFrame,
                                         df_bldng_data:pd.DataFrame=None,
                                         property_sources = None,
                                         df_metadata:pd.DataFrame=None,
                                         hints:dict = None,
                                         learn:List[str] = None,
                                         learn_period__d=7, 
                                         learn_change_interval__min = None,
                                         req_props:list = None,
                                         sanity_threshold_timedelta:timedelta=timedelta(hours=24),
                                         complete_most_recent_analysis=False,
                                         parallel=False
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
        - 'property_sources', a dictionary that maps key listed above to actual column names in df_data
        - 'req_props' list: a list of properties, occuring as keys in property_sources: 
            - If any of the values in this column are NaN, the interval is not considered 'sane'.
            - If you do not specify a value for req_props or specify req_props = None, then all properties from the property_sources dictionary are considered required
            - to speficy NO columns are required, specify property_sources = []
        - a df_metadata with index 'id' and columns:
            - none (this feature is not used in the current implementation yet, but added here for consistentcy with the learn_room_parameters() function)
        - hints: a dictionary that maps keys to fixed values to be used for analysis (set value for None to learn it):
            - 'aperture_sol__m2':             apparent solar aperture
            - 'eta_ch_hhv__W0':               higher heating value efficiency of the heating system 
            - 'g_not_ch_hhv__W':              average yearly gas power (higher heating value)  for other purposes than heating 
            - 'eta_not_ch_hhv__W0':           superior efficiency of heating the home indirectly using gas
            - 'wind_chill__K_s_m_1':          wind chill factor
            - 'aperture_inf__cm2':            effective infiltration area
            - 'heat_tr_bldng_cond__W_K_1':    specific heat loss
            - 'eta_dhw_hhv__W0':              domestic hot water efficiency
            - 'frac_remain_dhw__0':           fraction of domestic hot water heat contributing to heating the home
            - 'g_use_cooking_hhv__W':         average gas power (higher heating value) for cooking
            - 'eta_cooking_hhv__W0':          cooking efficiency
            - 'frac_remain_cooking__0':       fraction of cooking heat contributing to heating the home
            - 'heat_tr_dstr__W_K_1':          heat transmissivity of the heat distribution system
            - 'th_mass_dstr__Wh_K_1':         thermal mass of the heat distribution system
            - 'ventilation_default__dm3_s_1': default ventilation rate for for the learning process for the entire home
            - 'ventilation_max__dm3_s_1_m_2': maximum ventilation rate relative to the total floor area of the home
            - 'co2_outdoor__ppm':             average CO₂ outdoor concentration
        - df_home_bldng_data: a DataFrame with index id and columns
            - 'floors__m2': usable floor area of a dwelling in whole square meters according to NEN 2580:2007.
            - 'bldng__m3': (an estimate of) the building volume, e.g. 3D-BAG attribute b3_volume_lod22 (https://docs.3dbag.nl/en/schema/attributes/#b3_volume_lod22) 
            - (optionally) 'building_floors__0': the number of floors, e.g. 3D-BAG attribute b3_bouwlagen (https://docs.3dbag.nl/en/schema/attributes/#b3_bouwlagen)
        and optionally,
        - 'learn_period__d': the number of days to use as learn period in the analysis
        - 'learn_change_interval__min': the minimum interval (in minutes) that any time-varying-parameter may change
        
        Output:
        - a dataframe with per id the learned parameters and error metrics
        - a dataframe with additional column(s):
            - 'learned_temp_indoor__degC' best fitting indoor temperatures
            - 'learned_temp_dstr__degC' best fitting heat distribution system temperatures (if learned)
            - 'learned_ventilation__dm3_s_1' best fitting ventilation rates (if learned)

        """
        
        # check presence of hints
        mandatory_hints = ['aperture_sol__m2',
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
                          ]
        
        for hint in mandatory_hints:
            if not (hint in hints or isinstance(hints[hint], numbers.Number)):
                raise TypeError(f'hints[{hint}] parameter must be a number')

        # check for unlearnable parameters
        not_learnable =   ['eta_not_ch_hhv__W0',
                           'eta_dhw_hhv__W0',
                           'frac_remain_dhw__0',
                           'g_use_cooking_hhv__W', 
                           'eta_cooking_hhv__W0',
                           'frac_remain_cooking__0',
                           'heat_int__W_p_1'
                          ]
        
        for param in learn:
            if param in not_learnable:
                raise LearnError(f'No support for learning {param} (yet).')
     
        # Initialize DataFrames to store cumulative results
        df_learned_parameters = pd.DataFrame()
        df_learned_properties = pd.DataFrame()
        
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

        if parallel:
            df_analysis_jobs = Learner.create_job_list(df_data,
                                          learn_period__d=learn_period__d,
                                          req_props=req_props,
                                          property_sources=property_sources,
                                          sanity_threshold_timedelta=sanity_threshold_timedelta
                                         )
            
   
            # Initialize lists to store learned properties and parameters
            all_learned_job_properties = []
            all_learned_job_parameters = []
    
            num_jobs = df_analysis_jobs.shape[0]  # Get the number of jobs
            num_workers = min(num_jobs, os.cpu_count())  # Use the lesser of number of jobs or 16 (or any other upper limit)

            with ProcessPoolExecutor() as executor:
                # Submit tasks for each job
                print(f"Processing {num_jobs} learning jobs using {num_workers} processes")

                # Create a list to store futures for later result retrieval
                futures = []
                learned_jobs = {}
                
                with tqdm(total=num_jobs) as pbar:
                    for id, start, end in df_analysis_jobs.index:
                        # Create df_learn for the current job
                        df_learn = df_data.loc[(df_data.index.get_level_values('id') == id) & 
                                               (df_data.index.get_level_values('timestamp') >= start) & 
                                               (df_data.index.get_level_values('timestamp') < end)]
                        # Extracting building-specific data for each job
                        bldng_data = df_bldng_data.loc[id].to_dict()

                        if any(df_data.columns.str.startswith('model_')): 
                            # Get actual values of parameters of this id (if available)
                            actual_parameter_values = Learner.get_actual_parameter_values(id, 
                                                                                          aperture_inf_nl_avg__cm2,
                                                                                          heat_tr_dstr_nl_avg__W_K_1,
                                                                                          th_mass_dstr_nl_avg__Wh_K_1
                                                                                         )
                        else: 
                            actual_parameter_values = None
    
                        # Submit the analyze_job function to the executor
                        future = executor.submit(Learner.analyze_job, 
                                                 id, start, end, 
                                                 df_learn, 
                                                 property_sources, hints, learn, learn_change_interval__min,
                                                 bldng_data,
                                                 actual_parameter_values,
                                                 results_dir)
    
                        futures.append(future)
                        learned_jobs[(id, start, end)] = future
        
                    # Collect results as they complete
                    for future in as_completed(futures):
                        try:
                            df_learned_job_parameters, df_learned_job_properties = future.result()
                            all_learned_job_properties.append(df_learned_job_properties)
                            all_learned_job_parameters.append(df_learned_job_parameters)
                        except Exception as e:
                            # Handle only the specific "Solution Not Found" error
                            if "Solution Not Found" in str(e):
                                # Find which job caused the error
                                for (id, start, end), job_future in learned_jobs.items():
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
            df_learned_properties = pd.concat(all_learned_job_properties, axis=0).drop_duplicates()
            df_learned_parameters = pd.concat(all_learned_job_parameters, axis=0).drop_duplicates()

            # Merging all learned time series data into df_data
            df_data = df_data.merge(df_learned_properties, left_index=True, right_index=True, how='left')

    
        else: 
            # (old) non-parallel processing logic here
            ids = df_data.index.unique('id').dropna()
            logging.info(f'ids to analyze: {ids}')
    
            start_analysis_period = df_data.index.unique('timestamp').min().to_pydatetime()
            end_analysis_period = df_data.index.unique('timestamp').max().to_pydatetime()
            logging.info(f'Start of analyses: {start_analysis_period}')
            logging.info(f'End of analyses: {end_analysis_period}')
    
            daterange_frequency = str(learn_period__d) + 'D'
            logging.info(f'learn period: {daterange_frequency}')
    
                
           
            # perform sanity check; not any of the required column values may be missing a value
            if req_props is None: # then we assume all properties from property_sources are required
                req_col = list(property_sources.values())
            else:
                req_col = [property_sources[prop] for prop in req_props if prop in property_sources]

            if not req_col: # then the caller explicitly set the list to be empty
                df_data.loc[:,'sanity'] = True
            else:
                df_data.loc[:,'sanity'] = ~df_data[req_col].isna().any(axis="columns")
    
            # iterate over ids
            for id in tqdm(ids):
                
                # Check if results for this ID are already present
                if complete_most_recent_analysis and id in existing_results['id'].values:
                    logging.info(f'Results for ID {id} already exist. Skipping analysis.')
                    continue  # Skip to the next ID
    
                if any(df_data.columns.str.startswith('model_')): 
                    # Get actual values of parameters of this id (if available)
                    actual_parameter_values = Learner.get_actual_parameter_values(id, aperture_inf_nl_avg__cm2, heat_tr_dstr_nl_avg__W_K_1, th_mass_dstr_nl_avg__Wh_K_1)
                else: 
                    actual_parameter_values = None
    
                # Get building specific data for each job
                bldng_data = df_bldng_data.loc[id].to_dict()

    
                learn_period_starts = pd.date_range(start=start_analysis_period, end=end_analysis_period, inclusive='both', freq=daterange_frequency)
    
                learn_period_iterator = tqdm(learn_period_starts)
    
                try:
                    # iterate over learn periods
                    for learn_period_start in learn_period_iterator:
        
                        learn_period_end = min(learn_period_start + timedelta(days=learn_period__d), end_analysis_period)
         
                        # learn only for the longest streak of sane data 
                        df_learn = Learner.get_longest_sane_streak(df_data, id, learn_period_start, learn_period_end, sanity_threshold_timedelta)
                        if df_learn is None:
                            continue
                        learn_streak_period_start = df_learn.index.get_level_values('timestamp').min()
                        learn_streak_period_end = df_learn.index.get_level_values('timestamp').max()
                        learn_streak_period_len = len(df_learn)
                        
                        step__s = ((learn_streak_period_end - learn_streak_period_start).total_seconds()
                                  /
                                  (learn_streak_period_len-1)
                                 )
                        logging.info(f'longest sane streak: {learn_streak_period_start} - {learn_streak_period_end}: {learn_streak_period_len} steps of {step__s} s')
        
                        if learn_change_interval__min is None:
                            learn_change_interval__min = np.nan
                            MV_STEP_HOR =  1
                        else:
                            # implement ceiling integer division by 'upside down' floor integer division
                            MV_STEP_HOR =  -((learn_change_interval__min * 60) // -step__s)
        
                        logging.info(f'MV_STEP_HOR: {MV_STEP_HOR}')
        
                        duration__s = step__s * learn_streak_period_len
        
                        # setup learned_ dataframes with proper index
                        df_learned_job_properties = pd.DataFrame()
                        df_learned_job_parameters = pd.DataFrame()
        
        
                        # Learn varying ventilation rates if applicable
                        try:
                            logging.info(f"Analyzing ventilation rates for {id} from {learn_streak_period_start} to {learn_streak_period_end}...")
                            if 'ventilation__dm3_s_1' in learn:
                                df_learned_ventilation__dm3_s_1, df_learned_aperture_inf_co2__cm2 = Learner.learn_property_ventilation_rate(
                                    df_learn,
                                    duration__s,
                                    step__s,
                                    property_sources, 
                                    hints,
                                    learn,
                                    learn_change_interval__min,
                                    bldng_data,
                                    actual_parameter_values = actual_parameter_values
                                )
                                logging.info(f"Learned ventilation rates for {id} from {learn_streak_period_start} to {learn_streak_period_end}")
                    
                                # Merging the results of ventilation learning
                                
                                if df_learned_job_parameters.empty:
                                    df_learned_job_parameters = df_learned_aperture_inf_co2__cm2
                                else:
                                    df_learned_job_parameters = df_learned_job_parameters.merge(df_learned_aperture_inf_co2__cm2, left_index=True, right_index=True, how='outer')
    
                                logging.info(f"Stored learned ventilation properties for {id} from {learn_streak_period_start} to {learn_streak_period_end}")
                                   
                        
                                # Storing learned ventilation rates in df_learn; this will also store them in df_data in the right place
                                df_learn.loc[:,'learned_ventilation__dm3_s_1'] = df_learned_ventilation__dm3_s_1['learned_ventilation__dm3_s_1'].values
                                logging.info(f"Wrote ventilation rates to df_learn for {id} from {learn_streak_period_start} to {learn_streak_period_end}")
                            
        
                            # Learn fixed model parameters
                            logging.info(f"Analyzing thermal properties for {id} from {learn_streak_period_start} to {learn_streak_period_end}...")
                            df_learned_thermal_properties, df_learned_thermal_parameters = Learner.learn_thermal_parameters(
                                df_learn,
                                duration__s,
                                step__s,
                                property_sources, 
                                hints, 
                                learn, 
                                bldng_data,
                                actual_parameter_values = actual_parameter_values
                            )
    
                            # Merging thermal properties (learned time series data) into df_learned_job_properties
                            if df_learned_job_properties.empty:
                                df_learned_job_properties = df_learned_thermal_properties
                            else:
                                df_learned_job_properties = df_learned_job_properties.merge(df_learned_thermal_properties, left_index=True, right_index=True, how='outer')
    
                            logging.info(f"Stored learned thermal properties for {id} from {learn_streak_period_start} to {learn_streak_period_end}")
        
                            # Merging all learned time series data into df_data
                            df_data = df_data.merge(df_learned_job_properties, left_index=True, right_index=True, how='left')
                            # Combine and give preference to values from later periods
                            if 'learned_temp_indoor__degC_x' in df_data.columns and 'learned_temp_indoor__degC_y' in df_data.columns:
                                # Combine the columns and give preference to values from the later period
                                df_data['learned_temp_indoor__degC'] = df_data['learned_temp_indoor__degC_y'].combine_first(df_data['learned_temp_indoor__degC_x'])
                            
                                # Safely drop the _x and _y columns
                                df_data = df_data.drop(columns=['learned_temp_indoor__degC_x', 'learned_temp_indoor__degC_y'], errors='ignore')
                            
                            # Merging thermal parameters into df_learned_job_properties
                            if df_learned_job_parameters.empty:
                                df_learned_job_parameters = df_learned_thermal_parameters
                            else:
                                df_learned_job_parameters = df_learned_job_parameters.merge(df_learned_thermal_parameters, left_index=True, right_index=True, how='outer')

                            # Append learned parameters for this period to the cumulative DataFrame
                            df_learned_parameters = pd.concat([df_learned_parameters, df_learned_job_parameters])
    
                        except KeyboardInterrupt:    
                            logging.error(f'KeyboardInterrupt; home analysis {id} not complete; saving results so far then will exit...')
                            # Save progress and exit
                            Learner.save_to_parquet(id, df_learned_job_parameters, df_learned_job_properties, df_data, results_dir)
                            return df_learned_parameters, df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])
        
                        except Exception as e:
                            logging.error(f'Exception {e} for home {id} in period from {learn_streak_period_start} to {learn_streak_period_end}; skipping...')
        
                    # After learning all periods for this ID, save results for the ID
                    Learner.save_to_parquet(id, df_learned_parameters, df_learned_job_properties, df_data, results_dir)
               
                except KeyboardInterrupt:
                    logging.error(f'KeyboardInterrupt; exiting without completing all ids...')
                    return df_learned_parameters, df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])

            # when using the old non-parallel version, make sure that the additional columns are dropped
            df_data = df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])
    
        # After all IDs, save final results
        Learner.final_save_to_parquet(df_learned_parameters, df_data, results_dir)
        
        return df_learned_parameters.sort_index(), df_data.sort_index()

    @staticmethod
    def learn_heat_distribution_parameters(df_data:pd.DataFrame,
                                           property_sources = None,
                                           hints:dict = None,
                                           req_props:list = None,
                                           duration_threshold_timedelta:timedelta=timedelta(minutes=15),
                                           parallel=False
                                          ) -> pd.DataFrame:

        if req_props is None:  # If req_col not set, use all property sources
            req_col = list(property_sources.values())
        else:
            req_col = [property_sources[prop] for prop in req_props if prop in property_sources]

        # focus only on the required columns
        df_learn_all = df_data[req_col]
        df_dstr_analysis_jobs = Learner.create_streak_job_list(df_data,
                                                               req_props=req_props,
                                                               property_sources= property_sources,
                                                               duration_threshold_timedelta=duration_threshold_timedelta
                                                              )
        
        # Initialize result lists
        all_learned_dstr_job_properties = []
        all_learned_dstr_job_parameters = []
        
        if parallel:
            
            num_jobs = df_dstr_analysis_jobs.shape[0]  # Get the number of jobs
            num_workers = min(num_jobs, os.cpu_count())  # Use the lesser of number of jobs or 16 (or any other upper limit)
            
            with ProcessPoolExecutor() as executor:
                # Submit tasks for each job
            
                # Create a list to store futures for later result retrieval
                futures = []
                learned_dstr_jobs = {}
                
                for id, start, end in tqdm(df_dstr_analysis_jobs.index, desc=f"Submitting learning jobs to {num_workers} processes"):
                    # Create df_learn for the current job
                    df_learn = df_learn_all.loc[(df_learn_all.index.get_level_values('id') == id) & 
                                                (df_learn_all.index.get_level_values('timestamp') >= start) & 
                                                (df_learn_all.index.get_level_values('timestamp') < end)]
                
                    duration__min = (df_dstr_analysis_jobs.loc[(id, start, end)]['duration__min']*60)
                  
                
                    # Submit the analyze_job function to the executor
                    future = executor.submit(Learner.learn_heat_distribution,
                                             id, start, end,
                                             df_learn,
                                             duration__min, 60,
                                             property_sources, hints)
        
                    futures.append(future)
                    learned_dstr_jobs[(id, start, end)] = future
        
            
                # Collect results as they complete
                with tqdm(total=len(futures), desc=f"Collecting results from {num_workers} processes") as pbar:
                    for future in as_completed(futures):
                        try:
                            df_learned_dstr_job_parameters, df_learned_dstr_job_properties = future.result()
                            all_learned_dstr_job_properties.append(df_learned_dstr_job_properties)
                            all_learned_dstr_job_parameters.append(df_learned_dstr_job_parameters)
                        except Exception as e:
                            # Handle only the specific "Solution Not Found" error
                            if "Solution Not Found" in str(e):
                                # Find which job caused the error
                                for (id, start, end), job_future in learned_dstr_jobs.items():
                                    if job_future == future:
                                        logging.warning(f"Solution Not Found for job (id: {id}, start: {start}, end: {end}). Skipping.")
                                        break
                                continue  # Skip this job and move on to the next one
                            else:
                                # Reraise other exceptions to stop execution
                                raise
                        finally:
                            pbar.update(1)
                
        else:
            for id, start, end in tqdm(df_dstr_analysis_jobs.index, desc=f"Analyzing heat distribution using 1 process"):
                # Create df_learn for the current job
                df_learn = df_learn_all.loc[(df_learn_all.index.get_level_values('id') == id) & 
                                            (df_learn_all.index.get_level_values('timestamp') >= start) & 
                                            (df_learn_all.index.get_level_values('timestamp') < end)]
            
                duration__min = (df_dstr_analysis_jobs.loc[(id, start, end)]['duration__min']*60)
    
                df_learned_dstr_job_parameters, df_learned_dstr_job_properties = Learner.learn_heat_distribution(id, start, end,
                                                                                                                 df_learn,
                                                                                                                 duration__min, 60,
                                                                                                                 property_sources, hints)
                all_learned_dstr_job_properties.append(df_learned_dstr_job_properties)
                all_learned_dstr_job_parameters.append(df_learned_dstr_job_parameters)    
        
        # Now merge all learned job properties and parameters into cumulative DataFrames
        all_learned_dstr_job_parameters= pd.concat(all_learned_dstr_job_parameters, axis=0).drop_duplicates()
        all_learned_dstr_job_properties= pd.concat(all_learned_dstr_job_properties, axis=0).drop_duplicates()

        return all_learned_dstr_job_parameters.sort_index()    


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