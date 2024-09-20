from datetime import datetime, timedelta
from typing import List
import pandas as pd
import numpy as np
import math
from gekko import GEKKO
from tqdm.notebook import tqdm

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
        
        df_learn = df_data.loc[id].loc[(df_data.loc[id].index >= learn_period_start) & (df_data.loc[id].index < learn_period_end)]
        learn_period_len = len(df_learn)
        #check for enough values
        if learn_period_len <=1:
            logging.info(f'No values for id: {id} between {learn_period_start} and {learn_period_end}; skipping...')
            return None
        #also check whether there are at least two sane values
        if (df_learn['sanity'].sum()) <=1: #counts the number of sane rows, since True values will be coutnd as 1 in suming
            logging.info(f'Less than two sane values for id: {id} between {learn_period_start} and {learn_period_start}; skipping...')
            return None                       
        learn_period_start = df_learn.index.min()
        learn_period_end = df_learn.index.max()
        logging.info(f'id: {id}; learn_period_start: {learn_period_start}')
        logging.info(f'id: {id}; learn_period_end: {learn_period_end}')

        logging.info(f'before longest streak analysis')
        logging.info(f'#rows in learning period before longest streak analysis: {len(df_learn)}')

        # restrict the dataframe to the longest streak of sane data
        ## give each streak a separate id
        df_data.loc[(id,learn_period_start):(id,learn_period_end), 'streak_id'] =  np.asarray(
            df_data.loc[(id,learn_period_start):(id,learn_period_end)].sanity
            .ne(df_data.loc[(id,learn_period_start):(id,learn_period_end)].sanity.shift())
            .cumsum()
        )

        # calculate timedelta for each interval (this code is suitable for unevenly spaced measurementes)
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

        # ensure that insane streaks are not selected as longest streak
        df_data.loc[(df_data.index.get_level_values('id') == id) & (df_data['sanity'] == False), 'streak_cumulative_duration__s'] = np.nan

        ## get the longest streak: the part of the dataframe where the streak_id matches the (first) streak_id that has the longest cumulative duration
        longest_streak_idxmax = df_data.loc[(id,learn_period_start):(id,learn_period_end)].streak_cumulative_duration__s.idxmax()
        logging.info(f'longest_streak_idxmax: {longest_streak_idxmax}') 
        longest_streak_query = 'streak_id == ' + str(df_data.loc[longest_streak_idxmax].streak_id)
        logging.info(f'longest_streak_query: {longest_streak_query}') 
        
        df_learn = df_data.loc[(id,learn_period_start):(id,learn_period_end)].query(longest_streak_query)
        timestamps = df_learn.index.get_level_values('timestamp')

        if learn_period_len != len(df_learn):
            logging.info(f'id: {id}; {learn_period_len} rows between {learn_period_start} and {learn_period_end}')
            logging.info(f'id: {id}; {len(df_learn)} rows between {timestamps.min()} and {timestamps.max()} in the longest streak')
        
        # also check whether streak duration is long enough
        if ((timestamps.max() - timestamps.min()) < sanity_threshold_timedelta):
            logging.info(f'Longest streak duration to short for id: {id}; shorter than {sanity_threshold_timedelta} between {timestamps.min()} and {timestamps.max()}; skipping...')
            return None

        return df_learn
    

    def gas_split_simple(df_data_id_g_use__W:pd.DataFrame,
                         g_not_ch_hhv__W:float) -> pd.DataFrame:
        """
        Input:  
        - df_data_id_g_use__W: DataFrae of a single id with only g_use__W values; e.g. df_data.loc[id][property_sources['g_use__W']]
        - 'g_not_ch_hhv__W': average gas power based on higher heating value for other purposes than heating
          (be sure to use upper heating value of on the conversion of [m^3/a] to [J/s])
        
        Output:
        - a dataframe with the same length os df_data_id with two column(s):
            - 'g_use_not_ch_hhv_W' estimate of gas power NOT used as input for other purposes than central heatig (e.g. DHW, cooking)
            - 'g_use_ch_hhv_W' estimate of gas power used as input for central heating
        """
        # create empty dataframe for the results
        df_result = pd.DataFrame()

        #substract fixed amount of noCH use from gas use
        df_result['g_use_ch_hhv_W'] =  df_data_id_g_use__W
        df_result['g_use_ch_hhv_W'] = df_result['g_use_ch_hhv_W'] - g_not_ch_hhv__W
        #prevent negative gas use
        df_result[df_result < 0] = 0

        # Compensate for missed gas use by scaling down the remaining g_use_ch_hhv_W 
        avg_g_use__W = df_data_id_g_use__W.mean()

        uncorrected_g_use_ch_hhv_W = df_result['g_use_ch_hhv_W'].mean()
        scaling_factor =  (avg_g_use__W - g_not_ch_hhv__W) / uncorrected_g_use_ch_hhv_W  
        
        df_result['g_use_ch_hhv_W'] = df_result['g_use_ch_hhv_W'] * scaling_factor

        df_result['g_use_not_ch_hhv_W'] = g_not_ch_hhv__W

        # check whether split was done connectly in the sense that no gas use is lost
        if not math.isclose(df_data_id_g_use__W.mean(), (df_result['g_use_not_ch_hhv_W'].mean() + df_result['g_use_ch_hhv_W'].mean())):
            logging.error(f'ERROR splitting gas: before {avg_g_use__W:.2f} W, split: {df_result.g_use_not_ch_hhv_W.mean():.2f} + {df_result.g_use_ch_hhv_W.mean():2f} = {df_result.g_use_not_ch_hhv_W.mean() + df_result.g_use_ch_hhv_W.mean():.2f}')
        else:
            logging.info(f'correct gas split: before {avg_g_use__W:.2f} W, split: {df_result.g_use_not_ch_hhv_W.mean():.2f} + {df_result.g_use_ch_hhv_W.mean():2f} = {df_result.g_use_not_ch_hhv_W.mean() + df_result.g_use_ch_hhv_W.mean():.2f}')

        return df_result

    
    @staticmethod
    def learn_energy_profile(df_data:pd.DataFrame,
                             property_sources = None,
                             df_metadata:pd.DataFrame=None,
                             hints:dict = None,
                             learn:List[str] = None,
                             learn_period__d=7, 
                             req_col:list = None,
                             sanity_threshold_timedelta:timedelta=timedelta(hours=24),
                             ev_type=2) -> pd.DataFrame:
        """
        Input:  
        - a preprocessed pandas DataFrame with
            - a MultiIndex ['id', 'timestamp'], where
                - the column 'timestamp' is timezone-aware
                - time intervals between consecutive measurements are constant
                - but there may be gaps of multiple intervals with no measurements
                - multiple sources for the same property are already dealth with in preprocessing
            - columns:
              - property_sources['temp_in__degC']: indoor temperature
              - property_sources['temp_out__degC']: outdoor temperature 
              - property_sources['wind__m_s_1']: outdoor wind speed
              - property_sources['ghi__W_m_2']: global horizontal irradiation
              - property_sources['g_use_ch_hhv__W']: gas input power (using higher heating value) used for central heating
              - property_sources['eta_dhw_hhv__W0']: efficiency (against higher heating value) of turning gas power into heat
              - property_sources['g_use_dhw_hhv__W']: gas input power (using higher heating value) used for domestic hot water
              - property_sources['e__W']: electricity power used indoors
        - 'property_sources', a dictionary that maps key listed above to actual column names in df_data
        - 'req_col' list: a list of column names: 
            - If any of the values in this column are NaN, the interval is not considered 'sane'.
            - If you do not specify a value for req_col or specify req_col = None, then all properties from the property_sources dictionary are considered required
            - to speficy NO columns are required, specify property_sources = []
        - a df_metadata with index 'id' and columns:
            - none (this feature is not used in the current implementation yet, but added here for consistentcy with the learn_room_parameters() function)
        - hints: a dictionary that maps keys to fixed values to be used for analysis (set value for None to learn it):
            - 'A_sol__m2': apparent solar aperture [m^2]
            - 'eta_ch_hhv__W0': higher heating value efficiency [-] of the heating system 
              In the Netherlands, eta_ch_nl_avg_hhv__W0 = 0.963 from nfh_utils is a reasonable hint
            - 'g_not_ch_hhv__W': average yearly gas power (higher heating value)  for other purposes than heating 
              In the Netherlands, g_not_ch_nl_avg_hhv__W = 377 from nfh_utils is a reasonable hint
            - 'eta_not_ch_hhv__W0': superior efficiency [-] of heating the home indirectly using gas
              I the Netherlands, 0.34 is a reasonable hint
            - 'wind_chill__K_s_m_1': wind chill factor (in NL: 0.67 is a reasonable hint)
            - 'A_inf__cm2': effective infiltration area (in NL, 108 is a reasonable hint)
            - 'H_cond__W_K_1': specific heat loss (in NL, 250 is a reasonable hint)
            - 'eta_dhw_hhv__W0': domestic hot water efficiency (in NL, 0.716 is a reasonable hint)
            - 'frac_remain_dhw__0': fraction of domestic hot water heat contributing to heating the home (in NL, 0.500 is a reasonable hint)
            - 'g_use_cooking_hhv__W': average gas power (higher heating value) for cooking (in NL, 72 is a reasonable hint)
            - 'eta_cooking_hhv__W0': cooking efficiency (in NL, 0.444 is a reasonable hint)
            - 'frac_remain_cooking__0': fraction of cooking heat contributing to heating the home (in NL, 0.460 is a reasonable hint)
         and optionally,
        - the number of days to use as learn period in the analysis
        - 'ev_type': type 2 is usually recommended, since this is typically more than 50 times faster
        
        Output:
        - a dataframe with per id the learned parameters and error metrics
        - a dataframe with additional column(s):
            - 'sim_temp_in__degC' best fiting indoor temperatures

        """
        
        # check presence of hints
        mandatory_hints = ['A_sol__m2',
                           'occupancy__p',
                           'Q_gain_int__W_p_1',
                           'wind_chill__K_s_m_1',
                           'A_inf__cm2',
                           'H_cond__W_K_1', 
                           'eta_ch_hhv__W0',
                           'eta_dhw_hhv__W0',
                           'frac_remain_dhw__0',
                           'g_use_cooking_hhv__W', 
                           'eta_cooking_hhv__W0',
                           'frac_remain_cooking__0',
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
                           'Q_gain_int__W_p_1'
                          ]
        
        for param in learn:
            if param in not_learnable:
                raise LearnError(f'No support for learning {param} (yet).')


        # Use National averages, depending on hints provided
        
        Q_gain_int_occup__W = hints['Q_gain_int__W_p_1'] * hints['occupancy__p']    # average heat gain per occupant
      
        # create empty dataframe for results of all homes
        df_results_per_period = pd.DataFrame()

        # ensure that dataframe is sorted
        if not df_data.index.is_monotonic_increasing:
            df_data = df_data.sort_index()  
        
        # add empty columns to store fitting and learning results for time-varying 
        df_data.loc[:,'sim_temp_in__degC'] = np.nan

        ids = df_data.index.unique('id').dropna()
        logging.info(f'ids to analyze: {ids}')

        start_analysis_period = df_data.index.unique('timestamp').min().to_pydatetime()
        end_analysis_period = df_data.index.unique('timestamp').max().to_pydatetime()
        logging.info(f'Start of analyses: {start_analysis_period}')
        logging.info(f'End of analyses: {end_analysis_period}')

        daterange_frequency = str(learn_period__d) + 'D'
        logging.info(f'learn period: {daterange_frequency}')
       
        # perform sanity check; not any of the required column values may be missing a value
        if req_col is None: # then we assume all properties from property_sources are required
            req_col = list(property_sources.values())
        if not req_col: # then the caller explicitly set the list to be empty
            df_data.loc[:,'sanity'] = True
        else:
            df_data.loc[:,'sanity'] = ~df_data[req_col].isna().any(axis="columns")

        # iterate over ids
        for id in tqdm(ids):
            
            if any(df_data.columns.str.startswith('model_')): 
                # calculate values from synthetic home based on id 
                actual_H_cond__W_K_1 = id // 1e5
                actual_tau__h = (id % 1e5) // 1e2
                actual_A_sol__m2 = id % 1e2
                actual_C__kWh_K_1 = actual_H_cond__W_K_1 * actual_tau__h / 1e3
                actual_eta_ch_hhv__W0 = eta_ch_nl_avg_hhv__J0 
                actual_A_inf__cm2 = A_inf_nl_avg__m2 * cm2_m_2
            else:
                actual_H_cond__W_K_1 = np.nan
                actual_tau__h = np.nan
                actual_A_sol__m2 = np.nan
                actual_C__kWh_K_1 = np.nan
                actual_eta_ch_hhv__W0 = np.nan
                actual_A_inf__cm2 = np.nan

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

                duration__s = step__s * learn_streak_period_len

                # setup learned_ and mae_ variables
                mae_temp_in__degC = np.nan
                rmse_temp_in__degC = np.nan

                # TODO loop over learn list

                learned_H_cond__W_K_1 = np.nan
                mae_H_cond__W_K_1 = np.nan

                learned_tau__h = np.nan
                mae_tau__h = np.nan

                learned_C__kWh_K_1 = np.nan
                mae_C__kWh_K_1 = np.nan

                learned_A_sol__m2 = np.nan
                mae_A_sol__m2 = np.nan

                learned_A_inf__cm2 = np.nan
                mae_A_inf__cm2 = np.nan
                
                ##################################################################################################################
                # GEKKO code

                try:
            
                    # GEKKO Model - Initialize
                    m = GEKKO(remote=False)
                    m.time = np.arange(0, duration__s, step__s)

                    # Model parameter: H [W/K]: specific heat loss
                    if 'H_cond__W_K_1' in learn:
                        # set this parameter up so it can be learnt
                        H_cond__W_K_1 = m.FV(value=hints['H_cond__W_K_1'], lb=0, ub=1000)
                        H_cond__W_K_1.STATUS = 1; H_cond__W_K_1.FSTATUS = 0
                    else:
                        # do not learn this parameter, but use a fixed value based on hint
                        H_cond__W_K_1 = m.Param(value = hints['H_cond__W_K_1'])
                        learned_H_cond__W_K_1 = np.nan
                    
                    # Model parameter: tau [s]: effective thermal inertia
                    hint_tau__s = hints['tau__h'] * s_h_1
                    if 'tau__h' in learn:
                        # set this parameter up so it can be learnt
                        tau__s = m.FV(value = hint_tau__s, lb=(10 * s_h_1), ub=(1000 * s_h_1))
                        tau__s.STATUS = 1; tau__s.FSTATUS = 0
                    else:
                        # do not learn this parameter, but use a fixed value based on hint
                        tau__s = m.Param(value = hint_tau__s)
                        learned_tau__h = np.nan

                    # g_use_ch_hhv_W [-]: higher heating value of gas input to the boiler for central heating purposes
                    g_use_ch_hhv_W = m.MV(value = df_learn[property_sources['g_use_ch_hhv__W']].astype('float32').values)
                    g_use_ch_hhv_W.STATUS = 0; g_use_ch_hhv_W.FSTATUS = 1

                    # eta_ch_hhv__W0 [-]: efficiency (relative to higher heating value) of the boiler for central heating
                    eta_ch_hhv__W0 = m.MV(value = df_learn[property_sources['eta_ch_hhv__W0']].astype('float32').values)
                    eta_ch_hhv__W0.STATUS = 0; eta_ch_hhv__W0.FSTATUS = 1

                    # Q_gain_ch [W]: heat gain from natural gas used for central heating
                    Q_gain_g_ch__W = m.Intermediate(g_use_ch_hhv_W * eta_ch_hhv__W0)
                
                    g_use_dhw_hhv__W = m.MV(value = df_learn[property_sources['g_use_dhw_hhv__W']].astype('float32').values)
                    g_use_dhw_hhv__W.STATUS = 0; g_use_dhw_hhv__W.FSTATUS = 1

                    # Q_gain_not_ch [W]: heat gain from natural gas NOT used for central heating c.q. dhw + cooking
                    Q_gain_g_not_ch__W = m.Intermediate(
                        g_use_dhw_hhv__W * hints['eta_dhw_hhv__W0'] * hints['frac_remain_dhw__0']
                        + hints['g_use_cooking_hhv__W'] * hints['eta_cooking_hhv__W0'] + hints['frac_remain_cooking__0']
                    )

                    # e [W] : internal heat gain from internally used electricity
                    e__W = m.MV(value = df_learn[property_sources['e__W']].astype('float32').values)
                    e__W.STATUS = 0; e__W.FSTATUS = 1

                    # Q_gain_int [W]: calculated heat gain from internal sources
                    Q_gain_int__W = m.Intermediate(e__W + Q_gain_int_occup__W + Q_gain_g_not_ch__W)

                    # A_sol__m2 [m^2]: calculated heat gain from internal sources
                    if 'A_sol__m2' in learn:
                        # set this parameter up so it can be learnt
                        A_sol__m2 = m.FV(value = hints['A_sol__m2'], lb=1, ub=100); A_sol__m2.STATUS = 1; A_sol__m2.FSTATUS = 0
                    else:
                        # do not learn this parameter, but use a fixed value based on hint
                        A_sol__m2 = m.Param(value = hints['A_sol__m2'])
                        learned_A_sol__m2 = np.nan

                    # ghi [W/m^2]: measured global horizontal irradiation
                    ghi__W_m_2 = m.MV(value = df_learn[property_sources['ghi__W_m_2']].astype('float32').values)
                    ghi__W_m_2.STATUS = 0; ghi__W_m_2.FSTATUS = 1

                    # Q_gain_sol [W]: calculated heat gain from solar irradiation
                    Q_gain_sol__W = m.Intermediate(ghi__W_m_2 * A_sol__m2)
                    
                    # temp_in [°C]: Indoor temperature; objective (Control Variable)
                    temp_in__degC = m.CV(value = df_learn[property_sources['temp_in__degC']].astype('float32').values)
                    temp_in__degC.STATUS = 1; temp_in__degC.FSTATUS = 1
                    # temp_in__degC.MEAS_GAP= 0.25

                    # temp_out [°C]: measured outdoor temperature
                    temp_out__degC = m.MV(value = df_learn[property_sources['temp_out__degC']].astype('float32').values)
                    temp_out__degC.STATUS = 0; temp_out__degC.FSTATUS = 1

                    # Indoor-outdoor temperature difference (K)
                    indoor_outdoor_delta__K = m.Intermediate(temp_in__degC - temp_out__degC)
                    
                    # Heat loss due to conduction
                    Q_loss_cond__W = m.Intermediate(H_cond__W_K_1 * indoor_outdoor_delta__K) 

                    # wind [m/s]: measured wind speed
                    wind__m_s_1 = m.MV(value = df_learn[property_sources['wind__m_s_1']].astype('float32').values)
                    wind__m_s_1.STATUS = 0; wind__m_s_1.FSTATUS = 1
                    
                    # Infiltration area (m^2): set this parameter up so it can be learnt
                    if 'A_inf__cm2' in learn:
                        A_inf__cm2 = m.FV(value=hints['A_inf__cm2'], lb=0, ub=1.0)
                        A_inf__cm2.STATUS = 1; A_inf__cm2.FSTATUS = 0
                    else:
                        A_inf__cm2 = m.Param(value=hints['A_inf__cm2'])
                        learned_A_inf__cm2 = np.nan  
                    
                    # Heat loss due to infiltration
                    air__J_m_3_K_1 = air_room__J_m_3_K_1 # if needed, the volumetric heat capacity can be made specific for pressure and temperature
                    H_inf__W_K_1 = m.Intermediate((A_inf__cm2 / cm2_m_2) * wind__m_s_1 * air__J_m_3_K_1) 
                    Q_loss_inf__W = m.Intermediate(H_inf__W_K_1 * indoor_outdoor_delta__K)
                  
                    # Heat loss due to ventilation
                    H_vent__W_K_1 = 0    # initial simplification, could be improved based on insight into CO₂ levels and number of people present
                    Q_loss_vent__W = 0   # initial simplification, could be improved based on insight into CO₂ levels and number of people present
                    
                    # Main Equations 
                    Q_gain__W = m.Intermediate(Q_gain_g_ch__W + Q_gain_sol__W + Q_gain_int__W)
                    Q_loss__W = m.Intermediate(Q_loss_cond__W + Q_loss_inf__W + Q_loss_vent__W)
                    H__W_K_1 = m.Intermediate(H_cond__W_K_1 + H_inf__W_K_1 + H_vent__W_K_1)
                    C__J_K_1  = m.Intermediate(H__W_K_1 * tau__s) 
                    m.Equation(temp_in__degC.dt() == ((Q_gain__W - Q_loss__W) / C__J_K_1))
                    
                    # GEKKO - Solver setting
                    m.options.IMODE = 5
                    m.options.EV_TYPE = ev_type # specific objective function (1 = MAE; 2 = RMSE)
                    m.solve(disp = False)      

                    # Write best fitting temperatures into df_data
                    df_data.loc[(id,learn_streak_period_start):(id,learn_streak_period_end), 'sim_temp_in__degC'] = np.asarray(temp_in__degC)

                    # set learned variables and calculate error metrics: 
                    # mean absolute error (mae) for all learned parameters; 
                    # root mean squared error (rmse) only for predicted time series
                    
                    mae_temp_in__degC = mae(temp_in__degC, df_learn[property_sources['temp_in__degC']])
                    logging.info(f'mae_temp_in__degC: {mae_temp_in__degC}')
                    rmse_temp_in__degC = rmse(temp_in__degC, df_learn[property_sources['temp_in__degC']])
                    logging.info(f'rmse_temp_in__degC: {rmse_temp_in__degC}')

                    # TODO loop over learn list
                    if 'H_cond__W_K_1' in learn:
                        learned_H_cond__W_K_1 = H_cond__W_K_1.value[0]
                        mae_H_cond__W_K_1 = abs(learned_H_cond__W_K_1  - actual_H_cond__W_K_1)                  # evaluates to np.nan if no actual value
                    if 'tau__h' in learn:
                        learned_tau__h = tau__s.value[0] / s_h_1
                        mae_tau__h = abs(learned_tau__h - actual_tau__h)                                        # evaluates to np.nan if no actual value
                    if 'H_cond__W_K_1' in learn or 'tau__h' in learn :
                        learned_C__kWh_K_1 = learned_H_cond__W_K_1 * learned_tau__h / 1e3
                        mae_C__kWh_K_1 = abs(learned_C__kWh_K_1 - actual_C__kWh_K_1)                            # evaluates to np.nan if no actual value
                    if 'A_sol__m2' in learn:
                        learned_A_sol__m2 = A_sol__m2.value[0]
                        mae_A_sol__m2 = abs(learned_A_sol__m2 - actual_A_sol__m2)                               # evaluates to np.nan if no actual value
                    if 'A_inf__cm2' in learn:
                        learned_A_inf__cm2 = A_inf__cm2.value[0]
                        mae_A_inf__cm2 = abs(learned_A_inf__cm2 - actual_A_inf__cm2)                               # evaluates to np.nan if no actual value


                except KeyboardInterrupt:    
                    logging.error(f'KeyboardInterrupt; home analysis {id} not complete; saving results so far then will exit...')
                    # only then exit the function and return to caller
                    return df_results_per_period.set_index('id'), df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])

                except Exception as e:
                    logging.error(f'Exception {e} for home {id} in period from {learn_streak_period_start} to {learn_streak_period_end}; skipping...')
                
                finally:
                    # create a results row and add to results per period dataframe
                    # TODO use learn array more
                    # TODO log more metadata such that we can compare results from different runs / learning strategy more easily
                    df_results_per_period = pd.concat(
                        [
                            df_results_per_period,
                            pd.DataFrame(
                                {
                                    'id': [id],
                                    'learn_streak_period_start': [learn_streak_period_start],
                                    'learn_streak_period_end': [learn_streak_period_end],
                                    'step__s': [step__s],
                                    'duration__s': [duration__s],
                                    'EV_TYPE': [m.options.EV_TYPE],
                                    'mae_temp_in__degC': [mae_temp_in__degC],
                                    'rmse_temp_in__degC': [rmse_temp_in__degC],
                                    'learned_H_cond__W_K_1': [learned_H_cond__W_K_1],
                                    'actual_H_cond__W_K_1': [actual_H_cond__W_K_1],
                                    'mae_H_cond__W_K_1': [mae_H_cond__W_K_1],
                                    'learned_tau__h': [learned_tau__h],
                                    'actual_tau__h': [actual_tau__h], 
                                    'mae_tau__h': [mae_tau__h], 
                                    'learned_C__kWh_K_1':[learned_C__kWh_K_1],
                                    'actual_C__kWh_K_1':[actual_C__kWh_K_1],
                                    'mae_C__kWh_K_1': [mae_C__kWh_K_1],
                                    'learned_A_sol__m2': [learned_A_sol__m2],
                                    'actual_A_sol__m2': [actual_A_sol__m2],
                                    'mae_A_sol__m2': [mae_A_sol__m2],
                                    'learned_A_inf__cm2': [learned_A_inf__cm2],
                                    'mae_A_inf__cm2': [mae_A_inf__cm2]
                                }
                            )
                        ]
                    )

                    m.cleanup()
                ##################################################################################################################

            # after all learn periods of a single id

            # TODO write results to excel file (incrementally update) after each id, to make sure partial results are not lost
            
        # after all ids

        return df_results_per_period.set_index('id'), df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])
        # return df_results_per_period.set_index('id'), df_data.drop(columns=['interval__s', 'sanity'])
 
    

