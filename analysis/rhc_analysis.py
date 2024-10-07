from datetime import datetime, timedelta
from typing import List, Tuple
import pandas as pd
import numpy as np
import math
from gekko import GEKKO
from tqdm.notebook import tqdm
import os

import numbers
import logging

from nfh_utils import *

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
        
    # Function to create a new results directory
    def create_results_directory(base_dir='results'):
        timestamp = datetime.now().isoformat()
        results_dir = os.path.join(base_dir, f'results-{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        return results_dir
    
    def get_actual_parameter_values(id, aperture_inf_avg__cm2, heat_tr_dist_avg__W_K_1, th_mass_dist_avg__Wh_K_1):
        """
        Calculate actual thermal parameter values based on the given 'id' and return them in a dictionary.
    
        Args:
            id: The unique identifier for which to calculate the actual values.
            aperture_inf_avg__cm2: Average aperture for infiltration in cm².
            heat_tr_dist_avg__W_K_1: Average heat transfer coefficient of the distribution system in W/K.
            th_mass_dist_avg__Wh_K_1: Average thermal mass of the distribution system in Wh/K.
    
        Returns:
            dict: A dictionary containing the actual parameter values.
        """
        actual_parameter_values = {
            'heat_tr_bldng_cond__W_K_1': np.nan,
            'th_inert_bldng__h': np.nan,
            'aperture_sol__m2': np.nan,
            'th_mass_bldng__Wh_K_1': np.nan,
            'aperture_inf__cm2': aperture_inf_avg__cm2,  # Use average value as provided
            'heat_tr_dist__W_K_1': heat_tr_dist_avg__W_K_1,  # Use average value
            'th_mass_dist__Wh_K_1': th_mass_dist_avg__Wh_K_1  # Use average value
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

    
    def learn_property_ventilation_rate(df_learn,
                                        property_sources,
                                        hints, learn,
                                        learn_change_interval__min,
                                        building_volume__m3,
                                        building_floor_area__m2) -> Tuple[np.ndarray, float]:

        m = GEKKO(remote=False)
        m.time = np.arange(0, len(df_learn), 1)

        ## Use measured CO₂ concentration indoors
        co2_indoor__ppm = m.CV(value=df_learn[property_sources['co2_indoor__ppm']].values)
        co2_indoor__ppm.STATUS = 1
        co2_indoor__ppm.FSTATUS = 1
        
        ## CO₂ concentration gain indoors

        # Use measured occupancy
        occupancy__p = m.MV(value = df_learn[property_sources['occupancy__p']].astype('float32').values)
        occupancy__p.STATUS = 0
        occupancy__p.FSTATUS = 1

        co2_indoor_gain__ppm_s_1 = m.Intermediate(occupancy__p * co2_exhale_desk_work__umol_p_1_s_1 / 
                                                  (building_volume__m3 * gas_room__mol_m_3))
        
        ## CO₂ concentration loss indoors

        # Ventilation-induced CO₂ concentration loss indoors
        ventilation__dm3_s_1 = m.MV(value=hints['ventilation_default__dm3_s_1'],
                                    lb=0.0, 
                                    ub=hints['ventilation_max__dm3_s_1_m_2'] * building_floor_area__m2)
        ventilation__dm3_s_1.STATUS = 1
        ventilation__dm3_s_1.FSTATUS = 1
        
        if learn_change_interval__min is not None:
            ventilation__dm3_s_1.MV_STEP_HOR = learn_change_interval__min
        
        air_changes_vent__s_1 = m.Intermediate(ventilation__dm3_s_1 / (building_volume__m3 * 1000))  # dm3 to m3

        # Wind-induced (infiltration) CO₂ concentration loss indoors
        wind__m_s_1 = m.MV(value=df_learn[property_sources['wind__m_s_1']].astype('float32').values)
        wind__m_s_1.STATUS = 0
        wind__m_s_1.FSTATUS = 1
    
        if 'aperture_inf__cm2' in learn:
            aperture_inf__cm2 = m.FV(value=hints['aperture_inf__cm2'], lb=0, ub=100000.0)
            aperture_inf__cm2.STATUS = 1
            aperture_inf__cm2.FSTATUS = 0
        else:
            aperture_inf__cm2 = m.Param(value=hints['aperture_inf__cm2'])

        air_inf__m3_s_1 = m.Intermediate(wind__m_s_1 * aperture_inf__cm2 / cm2_m_2)        
        air_changes_inf__s_1 = m.Intermediate(air_inf__m3_s_1 / building_volume__m3)

        # Total losses of CO₂ concentration indoors
        air_changes_total__s_1 = m.Intermediate(air_changes_vent__s_1 + air_changes_inf__s_1)
        co2_elevation__ppm = m.Intermediate(co2_indoor__ppm - hints['co2_outdoor__ppm'])
        co2_indoor_loss__ppm_s_1 = m.Intermediate(air_changes_total__s_1 * co2_elevation__ppm)
        
        # CO₂ concentration balance equation:  
        m.Equation(co2_indoor__ppm.dt() == co2_indoor_gain__ppm_s_1 - co2_indoor_loss__ppm_s_1)
        
        # Solve the model to start the learning process
        m.options.IMODE = 5        # Simultaneous Estimation 
        m.options.EV_TYPE = 2      # RMSE
        m.solve(disp=False)

        # Store results of the learning process
        co2_model_learned_ventilation__dm3_s_1 = np.asarray(ventilation__dm3_s_1)
        co2_model_learned_aperture_inf__cm2 = aperture_inf__cm2.value[0]

        m.cleanup()

        return co2_model_learned_ventilation__dm3_s_1, co2_model_learned_aperture_inf__cm2

    
    def learn_thermal_parameters(df_learn,
                                 property_sources,
                                 hints,
                                 learn,
                                 building_volume__m3,
                                 actual_parameter_values) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Learn thermal parameters for a building's heating system using GEKKO.
        
        Parameters:
        df_learn (pd.DataFrame): DataFrame containing the time series data to be used for learning.
        property_sources (dict): Dictionary mapping property names to their corresponding columns in df_learn.
        hints (dict): Dictionary containing default values for the various parameters.
        learn (list): List of parameters to be learned.
        building_volume__m3 (float): Volume of the building in m3.
        """
        
        ##################################################################################################################
        # GEKKO Model - Initialize
        ##################################################################################################################

        m = GEKKO(remote=False)
        m.time = np.arange(0, len(df_learn), 1)

        ##################################################################################################################
        # Heat gains
        ##################################################################################################################
    
        # Central heating gains
        g_use_ch_hhv_W = m.MV(value=df_learn[property_sources['g_use_ch_hhv__W']].astype('float32').values)
        g_use_ch_hhv_W.STATUS = 0
        g_use_ch_hhv_W.FSTATUS = 1
    
        eta_ch_hhv__W0 = m.MV(value=df_learn[property_sources['eta_ch_hhv__W0']].astype('float32').values)
        eta_ch_hhv__W0.STATUS = 0
        eta_ch_hhv__W0.FSTATUS = 1
    
        heat_g_ch__W = m.Intermediate(g_use_ch_hhv_W * eta_ch_hhv__W0)
    
        # Optionally learn heat distribution system parameters
        if 'heat_tr_dist__W_K_1' in learn:
            heat_tr_dist__W_K_1 = m.FV(value=hints['heat_tr_dist__W_K_1'], lb=0, ub=1000)
            heat_tr_dist__W_K_1.STATUS = 1
            heat_tr_dist__W_K_1.FSTATUS = 0
        else:
            heat_tr_dist__W_K_1 = hints['heat_tr_dist__W_K_1']
        
        if 'th_mass_dist__Wh_K_1' in learn:
            th_mass_dist__Wh_K_1 = m.FV(value=hints['th_mass_dist__Wh_K_1'], lb=0, ub=10000)
            th_mass_dist__Wh_K_1.STATUS = 1
            th_mass_dist__Wh_K_1.FSTATUS = 0
        else:
            th_mass_dist__Wh_K_1 = hints['th_mass_dist__Wh_K_1']
    
        # Central heating temperature
        if 'heat_tr_dist__W_K_1' in learn or 'th_mass_dist__J_K_1' in learn:
            temp_sup_ch__degC = m.MV(value=df_learn[property_sources['temp_sup_ch__degC']].astype('float32').values)
            temp_sup_ch__degC.STATUS = 0
            temp_sup_ch__degC.FSTATUS = 1
    
            temp_ret_ch__degC = m.MV(value=df_learn[property_sources['temp_ret_ch__degC']].astype('float32').values)
            temp_ret_ch__degC.STATUS = 0
            temp_ret_ch__degC.FSTATUS = 1
    
            temp_dist__degC = m.Intermediate((temp_sup_ch__degC + temp_ret_ch__degC) / 2)
            heat_dist__W = m.Intermediate(heat_tr_dist__W_K_1 * (temp_dist__degC - temp_indoor__degC))
    
            m.Equation(temp_dist__degC.dt() == (heat_g_ch__W - heat_dist__W) / (th_mass_dist__Wh_K_1 * s_h_1))
        else:
            heat_dist__W = heat_g_ch__W
    
        ##################################################################################################################
        # Solar heat gains
        ##################################################################################################################
    
        if 'aperture_sol__m2' in learn:
            aperture_sol__m2 = m.FV(value=hints['aperture_sol__m2'], lb=1, ub=100)
            aperture_sol__m2.STATUS = 1
            aperture_sol__m2.FSTATUS = 0
        else:
            aperture_sol__m2 = m.Param(value=hints['aperture_sol__m2'])
    
        ghi__W_m_2 = m.MV(value=df_learn[property_sources['ghi__W_m_2']].astype('float32').values)
        ghi__W_m_2.STATUS = 0
        ghi__W_m_2.FSTATUS = 1
    
        heat_sol__W = m.Intermediate(ghi__W_m_2 * aperture_sol__m2)
    
        ##################################################################################################################
        # Conductive heat losses
        ##################################################################################################################
    
        if 'heat_tr_bldng_cond__W_K_1' in learn:
            heat_tr_bldng_cond__W_K_1 = m.FV(value=hints['heat_tr_bldng_cond__W_K_1'], lb=0, ub=1000)
            heat_tr_bldng_cond__W_K_1.STATUS = 1
            heat_tr_bldng_cond__W_K_1.FSTATUS = 0
        else:
            heat_tr_bldng_cond__W_K_1 = hints['heat_tr_bldng_cond__W_K_1']
    
        temp_indoor__degC = m.CV(value=df_learn[property_sources['temp_indoor__degC']].astype('float32').values)
        temp_indoor__degC.STATUS = 1
        temp_indoor__degC.FSTATUS = 1
    
        temp_outdoor__degC = m.MV(value=df_learn[property_sources['temp_outdoor__degC']].astype('float32').values)
        temp_outdoor__degC.STATUS = 0
        temp_outdoor__degC.FSTATUS = 1
    
        indoor_outdoor_delta__K = m.Intermediate(temp_indoor__degC - temp_outdoor__degC)
    
        heat_loss_bldng_cond__W = m.Intermediate(heat_tr_bldng_cond__W_K_1 * indoor_outdoor_delta__K)
    
        ##################################################################################################################
        # Infiltration and ventilation heat losses
        ##################################################################################################################
    
        wind__m_s_1 = m.MV(value=df_learn[property_sources['wind__m_s_1']].astype('float32').values)
        wind__m_s_1.STATUS = 0
        wind__m_s_1.FSTATUS = 1
    
        if 'aperture_inf__cm2' in learn:
            aperture_inf__cm2 = m.FV(value=hints['aperture_inf__cm2'], lb=0, ub=100000.0)
            aperture_inf__cm2.STATUS = 1
            aperture_inf__cm2.FSTATUS = 0
        else:
            aperture_inf__cm2 = m.Param(value=hints['aperture_inf__cm2'])
    
        air_inf__m3_s_1 = m.Intermediate(wind__m_s_1 * aperture_inf__cm2 / cm2_m_2)
        heat_tr_bldng_inf__W_K_1 = m.Intermediate(air_inf__m3_s_1 * air_room__J_m_3_K_1)
        heat_loss_bldng_inf__W = m.Intermediate(heat_tr_bldng_inf__W_K_1 * indoor_outdoor_delta__K)
    
        if 'ventilation__dm3_s_1' in learn:
            ventilation__dm3_s_1 = m.MV(value=df_learn['learned_ventilation__dm3_s_1'].astype('float32').values)
            air_changes_vent__s_1 = m.Intermediate(ventilation__dm3_s_1 / (building_volume__m3 * dm3_m_3))
            heat_tr_bldng_vent__W_K_1 = m.Intermediate(air_changes_vent__s_1 * building_volume__m3 * air_room__J_m_3_K_1)
            heat_loss_bldng_vent__W = m.Intermediate(heat_tr_bldng_vent__W_K_1 * indoor_outdoor_delta__K)
        else:
            heat_tr_bldng_vent__W_K_1 = 0
            heat_loss_bldng_vent__W = 0
    
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
        df_learned_properties = pd.DataFrame(index=df_learn.index)
    
        # Store learned time-varying data in DataFrame
        df_learned_properties['sim_temp_indoor__degC'] = np.asarray(temp_indoor__degC)
    
        # If 'sim_temp_dist__degC' is computed, include it as well
        if 'heat_tr_dist__W_K_1' in learn or 'th_mass_dist__J_K_1' in learn:
            df_learned_properties['sim_temp_dist__degC'] = np.asarray(temp_dist__degC)
    
        # Initialize a DataFrame for learned thermal parameters (one row with id, start, end)
        df_learned_parameters = pd.DataFrame({
            # TO DO: check whether the id index leval is actually retained in df_learn (needed for parallelization)
            'id': [df_learn.index.get_level_values('id')[0]], 
            'start': [df_learn.index.get_level_values('timestamp').min()],
            'end': [df_learn.index.get_level_values('timestamp').max()]
        })
    
        # Loop over the learn list and store learned values and calculate MAE if actual value is available
        for param in learn:
            if param in locals():  # Check if the variable is defined
                learned_value = locals()[param].value[0]
                df_learned_parameters[f'learned_{param}'] = learned_value
    
                # If actual value exists, compute MAE
                if param in actual_parameter_values:
                    df_learned_parameters[f'mae_{param}'] = abs(learned_value - actual_parameter_values[param])
                    
        # Set MultiIndex on the DataFrame (id, start, end)
        df_learned_parameters.set_index(['id', 'start', 'end'], inplace=True)    

        m.cleanup()
    
        # Return both DataFrames: learned time-varying properties and learned fixed parameters
        return df_learned_properties, df_learned_parameters
        
    
    @staticmethod
    def learn_heat_performance_signature(df_data:pd.DataFrame,
                                         df_bldng_data:pd.DataFrame=None,
                                         property_sources = None,
                                         df_metadata:pd.DataFrame=None,
                                         hints:dict = None,
                                         learn:List[str] = None,
                                         learn_period__d=7, 
                                         learn_change_interval__min = None,
                                         req_col:list = None,
                                         sanity_threshold_timedelta:timedelta=timedelta(hours=24),
                                         complete_most_recent_analysis=False
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
              - property_sources['wind__m_s_1']: outdoor wind speed
              - property_sources['ghi__W_m_2']: global horizontal irradiation
              - property_sources['g_use_ch_hhv__W']: gas input power (using higher heating value) used for central heating
              - property_sources['eta_dhw_hhv__W0']: efficiency (against higher heating value) of turning gas power into heat
              - property_sources['g_use_dhw_hhv__W']: gas input power (using higher heating value) used for domestic hot water
              - property_sources['e__W']: electricity power used indoors
              - property_sources['temp_sup_ch__degC']: Temperture of hot water supplied by the heat generation system to the heat distributon system
              - property_sources['temp_ret_ch__degC']: Temperture of hot water returned to the heat generation system from the heat distributon system
        - 'property_sources', a dictionary that maps key listed above to actual column names in df_data
        - 'req_col' list: a list of column names: 
            - If any of the values in this column are NaN, the interval is not considered 'sane'.
            - If you do not specify a value for req_col or specify req_col = None, then all properties from the property_sources dictionary are considered required
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
            - 'heat_tr_bldng_cond__W_K_1': specific heat loss
            - 'eta_dhw_hhv__W0':              domestic hot water efficiency
            - 'frac_remain_dhw__0':           fraction of domestic hot water heat contributing to heating the home
            - 'g_use_cooking_hhv__W':         average gas power (higher heating value) for cooking
            - 'eta_cooking_hhv__W0':          cooking efficiency
            - 'frac_remain_cooking__0':       fraction of cooking heat contributing to heating the home
            - 'heat_tr_dist__W_K_1':          heat dissipation capacity of the heat distribution system
            - 'th_mass_dist__Wh_K_1':         thermal mass of the heat distribution system
            - 'ventilation_default__dm3_s_1': default ventilation rate for for the learning process for the entire home
            - 'ventilation_max__dm3_s_1_m_2': maximum ventilation rate relative to the total floor area of the home
            - 'co2_outdoor__ppm':             average CO₂ outdoor concentration
        - df_home_bldng_data: a DataFrame with index id and columns
            - 'building_floor_area__m2': usable floor area of a dwelling in whole square meters according to NEN 2580:2007.
            - 'building_volume__m3': (an estimate of) the building volume, e.g. 3D-BAG attribute b3_volume_lod22 (https://docs.3dbag.nl/en/schema/attributes/#b3_volume_lod22) 
            - (optionally) 'building_floors__0': the number of floors, e.g. 3D-BAG attribute b3_bouwlagen (https://docs.3dbag.nl/en/schema/attributes/#b3_bouwlagen)
        and optionally,
        - 'learn_period__d': the number of days to use as learn period in the analysis
        - 'learn_change_interval__min': the minimum interval (in minutes) that any time-varying-parameter may change
        
        Output:
        - a dataframe with per id the learned parameters and error metrics
        - a dataframe with additional column(s):
            - 'sim_temp_indoor__degC' best fitting indoor temperatures
            - 'sim_temp_dist__degC' best fitting heat distribution system temperatures (if learned)

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
     
        # create empty dataframe for results of all homes
        df_results_per_period = pd.DataFrame()

        # ensure that dataframe is sorted
        if not df_data.index.is_monotonic_increasing:
            df_data = df_data.sort_index()  
        
        # add empty columns to store fitting and learning results for time-varying 
        df_data.loc[:,'sim_temp_indoor__degC'] = np.nan

        ids = df_data.index.unique('id').dropna()
        logging.info(f'ids to analyze: {ids}')

        start_analysis_period = df_data.index.unique('timestamp').min().to_pydatetime()
        end_analysis_period = df_data.index.unique('timestamp').max().to_pydatetime()
        logging.info(f'Start of analyses: {start_analysis_period}')
        logging.info(f'End of analyses: {end_analysis_period}')

        daterange_frequency = str(learn_period__d) + 'D'
        logging.info(f'learn period: {daterange_frequency}')

        # Check for the most recent results directory if complete_most_recent_analysis is True
        if complete_most_recent_analysis:
            results_dirs = [d for d in os.listdir(results_dir) if d.startswith('results-')]
            if results_dirs:
                most_recent_dir = sorted(results_dirs)[-1]  # Assuming sorted alphabetically gives the most recent
                logging.info(f'Using most recent results directory: {most_recent_dir}')
                # Load existing results into a DataFrame
                existing_results = pd.read_parquet(os.path.join(results_dir, most_recent_dir))
                logging.info(f'Loaded existing results from {most_recent_dir}')
                results_dir = d
        else:
            results_dir = Learner.create_results_directory()
            
       
        # perform sanity check; not any of the required column values may be missing a value
        if req_col is None: # then we assume all properties from property_sources are required
            req_col = list(property_sources.values())
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

            # Get actual values of parameters of this id (if available)
            actual_parameter_values = Learner.get_actual_parameter_values(id, aperture_inf_nl_avg__cm2, heat_tr_dist_nl_avg__W_K_1, th_mass_dist_nl_avg__Wh_K_1)

            # Get building_volume__m3 and building_floor_area__m2 from building-specific table
            building_volume__m3 = df_bldng_data.loc[id]['building_volume__m3']
            building_floor_area__m2 = df_bldng_data.loc[id]['building_floor_area__m2']

            learn_period_starts = pd.date_range(start=start_analysis_period, end=end_analysis_period, inclusive='both', freq=daterange_frequency)

            learn_period_iterator = tqdm(learn_period_starts)

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

                # setup learned_ and mae_ variables
                df_learned_parameters = pd.DataFrame
                mae_temp_indoor__degC = np.nan
                rmse_temp_indoor__degC = np.nan
                learned_aperture_inf_co2__cm2 = np.nan

                # Learn varying ventilation rates if applicable
                try:
                    if 'ventilation__dm3_s_1' in learn:
                        learned_ventilation__dm3_s_1, learned_aperture_inf_co2__cm2 = Learner.learn_property_ventilation_rate(
                            df_learn,
                            property_sources, 
                            hints,
                            learn,
                            learn_change_interval__min,
                            building_volume__m3,
                            building_floor_area__m2
                        )
                        logging.info(f"Learned ventilation rates for {id} from {learn_streak_period_start} to {learn_streak_period_end}")
                        # Update df_learn with the learned ventilation rates using .loc
                        df_learn.loc[:,'learned_ventilation__dm3_s_1'] = learned_ventilation__dm3_s_1
                        logging.info(f"Wrote ventilation rates to df_learn for {id} from {learn_streak_period_start} to {learn_streak_period_end}")
            
                    # Learn fixed model parameters
                    df_learned_properties, df_learned_parameters = Learner.learn_thermal_parameters(
                        df_learn, 
                        property_sources, 
                        hints, 
                        learn, 
                        building_volume__m3,
                        actual_parameter_values = actual_parameter_values
                    )
    
                    # Adding learned properties to df_learn
                    for column in df_learned_properties.columns:
                        df_learn.loc[:,column] = df_learned_properties[column].values

                    if learned_aperture_inf_co2__cm2 is not None:
                        df_learned_parameters['learned_aperture_inf_co2__cm2'] = learned_aperture_inf_co2__cm2
                        
                    # Calculate MAE and RMSE for indoor temperature and update learned_parameters
                    mae_temp_indoor__degC = mae(df_learn[property_sources['temp_indoor__degC']], df_learned_properties['sim_temp_indoor__degC'])
                    logging.info(f'mae_temp_indoor__degC: {mae_temp_indoor__degC}')
                    df_learned_parameters['mae_temp_indoor__degC'] = mae_temp_indoor__degC  # Store MAE for later
    
                    rmse_temp_indoor__degC = rmse(df_learn[property_sources['temp_indoor__degC']], df_learned_properties['sim_temp_indoor__degC'])
                    logging.info(f'rmse_temp_indoor__degC: {rmse_temp_indoor__degC}')
                    df_learned_parameters['rmse_temp_indoor__degC'] = rmse_temp_indoor__degC  # Store RMSE for later
                
                except KeyboardInterrupt:    
                    logging.error(f'KeyboardInterrupt; home analysis {id} not complete; saving results so far then will exit...')
                    # only then exit the function and return to caller
                    return df_results_per_period, df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])

                except Exception as e:
                    logging.error(f'Exception {e} for home {id} in period from {learn_streak_period_start} to {learn_streak_period_end}; skipping...')
                
                finally:
                    # Convert result_row to DataFrame and append it to df_results_per_period
                    df_results_per_period = pd.concat([df_results_per_period, df_learned_parameters])

            # after all learn periods of a single id
            # Write results to Parquet file for this ID
            df_results_per_period.to_parquet(os.path.join(results_dir, f'results-per-period-{id}.parquet'), index=False)
            df_learn.to_parquet(os.path.join(results_dir, f'results-{id}.parquet'), index=False)
            logging.info(f'Saved results for ID {id} to {results_dir}')
            
        # after all ids
        # Final save for df_results_per_period after all analyses
        df_results_per_period.to_parquet(os.path.join(results_dir, 'results_per_period_final.parquet'), index=False)
        logging.info(f'Final results per period saved to {results_dir}/results_per_period_final.parquet')
        df_data.to_parquet(os.path.join(results_dir, 'results_final.parquet'), index=False)
        logging.info(f'Final results saved to {results_dir}/results_per_period_final.parquet')

        return df_results_per_period.set_index('id'), df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])
        # return df_results_per_period.set_index('id'), df_data.drop(columns=['interval__s', 'sanity'])
 
    

