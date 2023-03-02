from datetime import datetime, timedelta
from typing import List
import pandas as pd
import numpy as np
import math
from gekko import GEKKO
from tqdm.notebook import tqdm

import numbers
import logging

class LearnError(Exception):
    def __init__(self, message):
        self.message = message
        
class Learner():

    def mae(predicted, actual) -> float:
        arr_predicted = np.asarray(predicted)
        arr_actual = np.asarray(actual)
        return np.mean(abs(arr_predicted - arr_actual))
        
    def rmse(predicted, actual) -> float:
        arr_predicted = np.asarray(predicted)
        arr_actual = np.asarray(actual)
        return np.sqrt(((arr_predicted - arr_actual)**2).mean())
    
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
        if len(df_learn.query('sanity == True')) <=1:
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
                         avg_g_use_noCH__W:float) -> pd.DataFrame:
        """
        Input:  
        - df_data_id_g_use__W: DataFrae of a single id with only g_use__W values; e.g. df_data.loc[id][property_sources['g_use__W']]
        - 'avg_g_use_noCH__W': average gas power flow for other purposes than heating
          (be sure to use upper heating value of on the conversion of [m^3/a] to [J/s])
        
        Output:
        - a dataframe with the same length os df_data_id with two column(s):
            - 'g_use_noCH__W' estimate of gas power NOT used as input for other purposes than central heatig (e.g. DHW, cooking)
            - 'g_use_CH__W' estimate of gas power used as input for central heating
        """
        # create empty dataframe for the results
        df_result = pd.DataFrame()

        #substract fixed amount of noCH use from gas use
        df_result['g_use_CH__W'] =  df_data_id_g_use__W
        df_result['g_use_CH__W'] = df_result['g_use_CH__W'] - avg_g_use_noCH__W
        #prevent negative gas use
        df_result[df_result < 0] = 0

        # Compensate for missed gas use by scaling down the remaining g_use_CH__W 
        avg_g_use__W = df_data_id_g_use__W.mean()

        uncorrected_g_use_CH__W = df_result['g_use_CH__W'].mean()
        scaling_factor =  (avg_g_use__W - avg_g_use_noCH__W) / uncorrected_g_use_CH__W  
        
        df_result['g_use_CH__W'] = df_result['g_use_CH__W'] * scaling_factor

        df_result['g_use_noCH__W'] = avg_g_use_noCH__W

        # check whether split was done connectly in the sense that no gas use is lost
        if not math.isclose(df_data_id_g_use__W.mean(), (df_result['g_use_noCH__W'].mean() + df_result['g_use_CH__W'].mean())):
            logging.error(f'ERROR splitting gas: before {avg_g_use__W:.2f} W, split: {df_result.g_use_noCH__W.mean():.2f} + {df_result.g_use_CH__W.mean():2f} = {df_result.g_use_noCH__W.mean() + df_result.g_use_CH__W.mean():.2f}')
        else:
            logging.info(f'correct gas split: before {avg_g_use__W:.2f} W, split: {df_result.g_use_noCH__W.mean():.2f} + {df_result.g_use_CH__W.mean():2f} = {df_result.g_use_noCH__W.mean() + df_result.g_use_CH__W.mean():.2f}')

        return df_result


                    
    @staticmethod
    def learn_home_parameters(df_data:pd.DataFrame,
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
              - property_sources['temp_out__degC']: 
              - property_sources['wind__m_s_1']:
              - property_sources['ghi__W_m_2']:
              - property_sources['g_use__W']:
              - property_sources['e_use__W']:
              - property_sources['e_ret__W']:
        - 'property_sources', a dictionary that maps key listed above to actual column names in df_data
        - 'req_col' list: a list of column names: 
            - If any of the values in this column are NaN, the interval is not considered 'sane'.
            - If you do not specify a value for req_col or specify req_col = None, then all properties from the property_sources dictionary are considered required
            - to speficy NO volumns are required, specify property_sources = []
        - a df_metadata with index 'id' and columns:
        - hints: a dictionary that maps keys to fixed values to be used for analysis (set value for None to learn it):
            - 'A_sol__m2': apparent solar aperture [m^2]
            - 'eta_sup_CH__0': superior efficiency [-] of the heating system (in NL 0.97 is a reasonable hint)
            - 'g_noCH__m3_a_1': average yearly gas use for other purposes than heating (in NL 334 [m^3/a] is a reasonable hint)
            - 'eta_sup_noCH__0': superior efficiency [-] of heating the home indirectly using gas (in NL 0.34 uis a reasonable hint)
            - 'wind_chill__degC_s_m_1': (in NL typically 0.67)
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
                           'eta_sup_CH__0',
                           'eta_sup_noCH__0',
                           'g_noCH__m3_a_1',
                           'occupancy__p',
                           'Q_gain_int__W_p_1',
                           'wind_chill__degC_s_m_1'
                          ]
        for hint in mandatory_hints:
            if not (hint in hints or isinstance(hints[hint], numbers.Number)):
                raise TypeError(f'hints[{hint}] parameter must be a number')

        # check for unlearnable parameters
        not_learnable =   ['eta_sup_noCH__0',
                           'g_noCH__m3_a_1',
                           'occupancy__p',
                           'Q_gain_int__W_p_1'
                          ]
        
        for param in learn:
            if param in not_learnable:
                raise LearnError(f'No support for learning {param} (yet).')

        # Conversion factors
        s_min_1 = 60                # [s] per [min]
        min_h_1 = 60                # [min] per [h]
        s_h_1 = s_min_1 * min_h_1   # [s] per [h]
        s_d_1 = (24 * s_h_1)        # [s] per [d]
        s_a_1 = (365.25 * s_d_1)    # [s] per [a] 

        J_kWh_1 = 1000 * s_h_1      # [J] per [kWh]

        # National averages
        h_sup_J_m_3 = 35.17e6                                                       # superior calorific value of natural gas from the Groningen field
        avg_g_use_noCH__W = hints['g_noCH__m3_a_1'] * h_sup_J_m_3 / s_a_1           # average gas usage per year for cooking and DHW, i.e. not for CH  
        Q_gain_int_occup__W = hints['Q_gain_int__W_p_1'] * hints['occupancy__p']    # average heat gain per occupant
      
        # create empty dataframe for results of all homes
        df_results_per_period = pd.DataFrame()

        # ensure that dataframee is sorted
        if not df_data.index.is_monotonic_increasing:
            df_data = df_data.sort_index()  
        
        # add empty columns to store fitting and learning results for time-varying 
        df_data['sim_temp_in__degC'] = np.nan

        ids = df_data.index.unique('id').dropna()
        logging.info(f'ids to analyze: {ids}')

        start_analysis_period = df_data.index.unique('timestamp').min().to_pydatetime()
        end_analysis_period = df_data.index.unique('timestamp').max().to_pydatetime()
        logging.info(f'Start of analyses: {start_analysis_period}')
        logging.info(f'End of analyses: {end_analysis_period}')

        daterange_frequency = str(learn_period__d) + 'D'
        logging.info(f'learn period: {daterange_frequency}')
       
        # perform sanity check; not any required column may be missing a value
        if req_col is None: # then we assume all properties from property_sources are required
            req_col = list(property_sources.values())
        if not req_col: # then the caller explicitly set the list to be empty
            df_data.loc[:,'sanity'] = True
        else:
            df_data.loc[:,'sanity'] = ~np.isnan(df_data[req_col]).any(axis="columns")

        # iterate over ids
        for id in tqdm(ids):
            
            if any(df_data.columns.str.startswith('model_')): 
                # calculate values from virtual home based on id 
                actual_H__W_K_1 = id // 1e5
                actual_tau__h = (id % 1e5) // 1e2
                actual_A_sol__m2 = id % 1e2
                actual_C__Wh_K_1 = actual_H__W_K_1 * actual_tau__h
                actual_eta_sup_CH__0 = 0.97 # efficiency used for calculating virtual room values)
                actual_wind_chill__degC_s_m_1 = 0.67 # efficiency used for calculating virtual room values)
            else:
                actual_H__W_K_1 = np.nan
                actual_tau__h = np.nan
                actual_A_sol__m2 = np.nan
                actual_C__Wh_K_1 = np.nan
                actual_eta_sup_CH__0 = np.nan
                actual_wind_chill__degC_s_m_1 = np.nan

            df_data.loc[id, ['g_use_CH__W', 'g_use_noCH__W']] = Learner.gas_split_simple(df_data.loc[id][property_sources['g_use__W']], avg_g_use_noCH__W).values

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
                logging.info(f'Start datetime longest sane streak: {learn_streak_period_start}')
                logging.info(f'End datetime longest sane streak: {learn_streak_period_end}')
                logging.info(f'#rows in longest sane streak: {learn_streak_period_len}')
                
                step__s = ((learn_streak_period_end - learn_streak_period_start).total_seconds()
                          /
                          (learn_streak_period_len-1)
                         )
                logging.info(f'step__s:  {step__s}')

                duration__s = step__s * learn_streak_period_len
                logging.info(f'duration__s: {duration__s}')

                # setup learned_ and mae_ variables
                mae_temp_in__degC = np.nan
                rmse_temp_in__degC = np.nan

                learned_H__W_K_1 = np.nan
                mae_H__W_K_1 = np.nan

                learned_tau__h = np.nan
                mae_tau__h = np.nan

                learned_C__Wh_K_1 = np.nan
                mae_C__Wh_K_1 = np.nan

                learned_A_sol__m2 = np.nan
                mae_A_sol__m2 = np.nan

                learned_eta_sup_CH__0 = np.nan
                mae_eta_sup_CH__0 = np.nan

                learned_wind_chill__degC_s_m_1 = np.nan
                mae_wind_chill__degC_s_m_1 = np.nan
                
                try:
            
                    ##################################################################################################################
                    # Gekko Model - Initialize
                    m = GEKKO(remote=False)
                    m.time = np.arange(0, duration__s, step__s)

                    # Model parameter: H [W/K]: specific heat loss
                    H__W_K_1 = m.FV(value=300.0, lb=0, ub=1000)
                    H__W_K_1.STATUS = 1; H__W_K_1.FSTATUS = 0
                    
                    # Model parameter: tau [s]: effective thermal inertia
                    tau__s = m.FV(value=(100 * s_h_1), lb=(10 * s_h_1), ub=(1000 * s_h_1))
                    tau__s.STATUS = 1; tau__s.FSTATUS = 0

                    # eta_sup_CH__0 [-]: upper heating efficiency of the central heating system
                    if 'eta_sup_CH__0' in learn:
                        eta_sup_CH__0 = m.FV(value = hints['eta_sup_CH__0'], lb = 0, ub = 1.0)
                        eta_sup_CH__0.STATUS = 1; eta_sup_CH__0.FSTATUS = 0
                        # eta_sup_CH__0.DMAX = 0.25
                    else:
                        # Set eta_sup_CH__0 to a fixed value when it should not be learned 
                        eta_sup_CH__0 = m.Param(value = hints['eta_sup_CH__0'])
                        learned_eta_sup_CH__0 = np.nan

                    g_use_CH__W = m.MV(value = df_learn['g_use_CH__W'].astype('float32').values)
                    g_use_CH__W.STATUS = 0; g_use_CH__W.FSTATUS = 1

                    # Q_gain_CH [W]: heat gain from natural gas used for central heating
                    Q_gain_g_CH__W = m.Intermediate(g_use_CH__W * eta_sup_CH__0)
                
                    g_use_noCH__W = m.MV(value = df_learn['g_use_noCH__W'].astype('float32').values)
                    g_use_noCH__W.STATUS = 0; g_use_noCH__W.FSTATUS = 1

                    # Q_gain_noCH  [W]: heat gain from natural gas used for central heating
                    Q_gain_g_noCH__W = m.Intermediate(g_use_noCH__W * hints['eta_sup_noCH__0'])

                    # e_use [W] - e_ret [W] : internal heat gain from internally used electricity
                    e_use__W = m.MV(value = df_learn[property_sources['e_use__W']].astype('float32').values)
                    e_use__W.STATUS = 0; e_use__W.FSTATUS = 1

                    e_ret__W = m.MV(value = df_learn[property_sources['e_ret__W']].astype('float32').values)
                    e_ret__W.STATUS = 0; e_ret__W.FSTATUS = 1

                    # Q_gain_int [W]: calculated heat gain from internal sources
                    Q_gain_int__W = m.Intermediate(e_use__W - e_ret__W + Q_gain_int_occup__W + Q_gain_g_noCH__W)

                    # A_sol__m2 [m^2]: calculated heat gain from internal sources
                    if 'A_sol__m2' in learn:
                        A_sol__m2 = m.FV(value = hints['A_sol__m2'], lb=1, ub=100); A_sol__m2.STATUS = 1; A_sol__m2.FSTATUS = 0
                    else:
                        A_sol__m2 = m.Param(value = hints['A_sol__m2'])
                        learned_A_sol__m2 = np.nan

                    # ghi [W/m^2]: measured global horizontal irradiation
                    ghi__W_m_2 = m.MV(value = df_learn[property_sources['ghi__W_m_2']].astype('float32').values)
                    ghi__W_m_2.STATUS = 0; ghi__W_m_2.FSTATUS = 1

                    # Q_gain_sol [W]: calculated heat gain from solar irradiation
                    Q_gain_sol__W = m.Intermediate(ghi__W_m_2 * A_sol__m2)
                    
                    # temp_out [°C]: measured outdoor temperature
                    temp_out__degC = m.MV(value = df_learn[property_sources['temp_out__degC']].astype('float32').values)
                    temp_out__degC.STATUS = 0; temp_out__degC.FSTATUS = 1

                    # wind [m/s]: measured wind speed
                    wind__m_s_1 = m.MV(value = df_learn[property_sources['wind__m_s_1']].astype('float32').values)
                    wind__m_s_1.STATUS = 0; wind__m_s_1.FSTATUS = 1
                    
                    # wind chill factor [°C/(m/s)]: cooling effect of wind on homes
                    if 'wind_chill__degC_s_m_1' in learn:
                        # learn wind_chill__degC_s_m_1 
                        wind_chill__degC_s_m_1 = m.FV(value = hints['wind_chill__degC_s_m_1'], lb=0, ub=5.0)
                        wind_chill__degC_s_m_1.STATUS = 1; wind_chill__degC_s_m_1.FSTATUS = 0
                    else:
                        # set fixed wind chill factor based on hint in home_metadata
                        wind_chill__degC_s_m_1 = m.Param(value = hints['wind_chill__degC_s_m_1'])
                        learned_wind_chill__degC_s_m_1 = np.nan

                    # calculate effective outdoor temperature by compensating for wind chill factor
                    temp_out_e__degC = m.Intermediate(temp_out__degC - wind_chill__degC_s_m_1 * wind__m_s_1)

                    # temp_in [°C]: Indoor temperature; objective (Control Variable)
                    temp_in__degC = m.CV(value = df_learn[property_sources['temp_in__degC']].astype('float32').values)
                    temp_in__degC.STATUS = 1; temp_in__degC.FSTATUS = 1
                    # temp_in__degC.MEAS_GAP= 0.25

                    # Main Equations 
                    Q_gain__W = m.Intermediate(Q_gain_g_CH__W + Q_gain_sol__W + Q_gain_int__W)
                    Q_loss__W = m.Intermediate(H__W_K_1 * (temp_in__degC - temp_out_e__degC)) 
                    C__J_K_1  = m.Intermediate(H__W_K_1 * tau__s) 
                    m.Equation(temp_in__degC.dt() == ((Q_gain__W - Q_loss__W) / C__J_K_1))
                    
                    # GEKKO - Solver setting
                    m.options.IMODE = 5
                    m.options.EV_TYPE = ev_type # specific objective function (1 = MAE; 2 = RMSE)
                    m.solve(disp = False)      

                    # Write best fitting temperatures into df_data
                    df_data.loc[(id,learn_streak_period_start):(id,learn_streak_period_end), 'sim_temp_in__degC'] = np.asarray(temp_in__degC)

                    # set learned variables and calculate error metrics: mean absolute error (mae) for all learned parameters; root mean squared error (rmse) only for predicted time series
                    mae_temp_in__degC = Learner.mae(temp_in__degC, df_learn[property_sources['temp_in__degC']])
                    logging.info(f'mae_temp_in__degC: {mae_temp_in__degC}')
                    rmse_temp_in__degC = Learner.rmse(temp_in__degC, df_learn[property_sources['temp_in__degC']])
                    logging.info(f'rmse_temp_in__degC: {rmse_temp_in__degC}')

                    learned_H__W_K_1 = H__W_K_1.value[0]
                    mae_H__W_K_1 = abs(learned_H__W_K_1  - actual_H__W_K_1)

                    learned_tau__h = tau__s.value[0] / s_h_1
                    mae_tau__h = abs(learned_tau__h - actual_tau__h)

                    learned_C__Wh_K_1 = learned_H__W_K_1 * learned_tau__h
                    mae_C__Wh_K_1 = abs(learned_C__Wh_K_1 - actual_C__Wh_K_1)

                    if 'A_sol__m2' in learn:
                        learned_A_sol__m2 = A_sol__m2.value[0]
                        mae_A_sol__m2 = abs(learned_A_sol__m2 - actual_A_sol__m2) # evaluates to np.nan if no actual value

                    if 'eta_sup_CH__0' in learn:
                        learned_eta_sup_CH__0 = eta_sup_CH__0.value[0]
                        mae_eta_sup_CH__0 = abs(learned_eta_sup_CH__0 - actual_eta_sup_CH__0) # evaluates to np.nan if no actual value

                    if 'wind_chill__degC_s_m_1' in learn:
                        learned_wind_chill__degC_s_m_1 = wind_chill__degC_s_m_1.value[0]
                        mae_wind_chill__degC_s_m_1 = abs(learned_wind_chill__degC_s_m_1 - actual_wind_chill__degC_s_m_1) # evaluates to np.nan if no actual value

                    # create a results row and add to results dataframe
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
                                    'learned_H__W_K_1': [learned_H__W_K_1],
                                    'actual_H__W_K_1': [actual_H__W_K_1],
                                    'mae_H__W_K_1': [mae_H__W_K_1],
                                    'learned_tau__h': [learned_tau__h],
                                    'actual_tau__h': [actual_tau__h], 
                                    'mae_tau__h': [mae_tau__h], 
                                    'learned_C__Wh_K_1':[learned_C__Wh_K_1],
                                    'actual_C__Wh_K_1':[actual_C__Wh_K_1],
                                    'mae_C__Wh_K_1': [mae_C__Wh_K_1],
                                    'learned_A_sol__m2': [learned_A_sol__m2],
                                    'actual_A_sol__m2': [actual_A_sol__m2],
                                    'mae_A_sol__m2': [mae_A_sol__m2],
                                    'learned_eta_sup_CH__0': [learned_eta_sup_CH__0],
                                    'actual_eta_sup_CH__0': [actual_eta_sup_CH__0],
                                    'mae_eta_sup_CH__0': [mae_eta_sup_CH__0],
                                    'learned_wind_chill__degC_s_m_1': [learned_wind_chill__degC_s_m_1],
                                    'mae_wind_chill__degC_s_m_1': [mae_wind_chill__degC_s_m_1]
                                }
                            )
                        ]
                    )

                    m.cleanup()
                    ##################################################################################################################

                except KeyboardInterrupt:    
                    logging.error(f'KeyboardInterrupt; home analysis {id} not complete; saving results so far then will exit...')
                    # only then exit the function and return to caller
                    return df_results_per_period.set_index('id'), df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])
                    # return df_results_per_period.set_index('id'), df_data.drop(columns=['interval__s', 'sanity'])

                except Exception as e:
                    logging.error(f'Exception {e} for home {id} in period from {learn_streak_period_start} to {learn_streak_period_end}; skipping...')
                    # only then exit the function and return to caller
                    return df_results_per_period.set_index('id'), df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])
                    # return df_results_per_period.set_index('id'), df_data.drop(columns=['interval__s', 'sanity'])

            # after all learn periods of a single id
            
        # after all ids

        return df_results_per_period.set_index('id'), df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])
        # return df_results_per_period.set_index('id'), df_data.drop(columns=['interval__s', 'sanity'])
 
    
    @staticmethod
    def learn_room_parameters(df_data:pd.DataFrame,
                              property_sources = None,
                              df_metadata:pd.DataFrame=None,
                              hints:dict = None,
                              learn:List[str] = None,
                              learn_period__d=7, 
                              req_col:list = None,
                              sanity_threshold_timedelta:timedelta=timedelta(hours=24),
                              learn_A_inf__m2 = True,
                              learn_valve_frac__0 = False,
                              learn_occupancy__p = False,
                              learn_change_interval__min = None,
                              ev_type=2) -> pd.DataFrame:
        """
        Input:  
        - a preprocessed dataframe with
            - a MultiIndex ['id', 'timestamp'], where
                - the column 'timestamp' is timezone-aware
                - time intervals between consecutive measurements are constant
                - but there may be gaps of multiple intervals with no measurements
                - multiple sources for the same property are already dealth with in preprocessing
        - a preprocessed dataframe with
            - a MultiIndex ['id', 'timestamp'], where the column 'timestamp' is timezone-aware
            - columns:
              - property_sources['co2__ppm']: name of the column to use for measurements of average CO₂-concentration in the room,
              - property_sources['occupancy__p']: name of the column to use for measurements of average number of people present in the room,
              - property_sources['valve_frac__0']: name of the column to use for measurements of opening fraction of the ventilation valve 
        - 'property_sources', a 'req_col' list: a list of column names: 
            - If any of the values in this column are NaN, the interval is not considered 'sane'.
            - If you do not specify a value for req_col or specify req_col = None, then all properties from the property_sources dictionary are considered required
            - to speficy NO volumns are required, specify property_sources = []
        - a df_metadata with index 'id' and columns:
            - 'room__m3', the volume of the room [m^3]
            - 'vent_max__m3_h_1', the maximum ventilation rate of the room [m^3/h]
        and optionally,
        - boolean values to indicatete whether certain variables are to be learned (NB you cannot learn valve_frac__0 and occupancy__p at the same time)
        - the number of days to use as learn period in the analysis
        - learn_change_interval__min: the minimum interval (in minutes) that any time-varying-parameter may change
        - 'ev_type': type 2 is usually recommended, since this is typically more than 50 times faster
        
        Output:
        - a dataframe with per id the learned parameters and error metrics
        - a dataframe with additional column(s):
            - 'sim_co2__ppm': best fiting CO₂ concentrations in the room
            - 'learned_valve_frac__0': learned time-varying valve fraction
            - 'learned_occupancy__p': learned time-varying valve fraction

        """
        # check presence of hints
        mandatory_hints = ['A_inf__m2'
                          ]
        for hint in mandatory_hints:
            if not (hint in hints or isinstance(hints[hint], numbers.Number)):
                raise LearnError(f'hints[{hint}] parameter must be a number')

        # check for unlearnable parameters
        not_learnable =   [
                          ]
        
        for param in learn:
            if param in not_learnable:
                raise TypeError(f'No support for learning {param} (yet).')        

        # Conversion factors
        s_min_1 = 60                                                  # [s] per [min]
        min_h_1 = 60                                                  # [min] per [h]
        s_h_1 = s_min_1 * min_h_1                                     # [s] per [h]
        ml_m_3 = 1e3 * 1e3                                            # [ml] per [m^3]
        umol_mol_1 = 1e6                                              # [µmol] per [mol]
        cm2_m_2 = 1e2 * 1e2                                           # [cm^2] per [m^2]
        O2ml_min_1_kg_1_p_1_MET_1 = 3.5                               # [mlO₂‧kg^-1‧min^-1] per [MET] 

        # Constants
        desk_work__MET = 1.5                                          # Metabolic Equivalent of Task for desk work [MET]
        P_std__Pa = 101325                                            # standard gas pressure [Pa]
        R__m3_Pa_K_1_mol_1 = 8.3145                                   # gas constant [m^3⋅Pa⋅K^-1⋅mol^-1)]
        temp_room__degC = 20.0                                        # standard room temperature [°C]
        temp_std__degC = 0.0                                          # standard gas temperature [°C]
        temp_zero__K = 273.15                                         # 0 [°C] = 273.15 [K]
        temp_std__K = temp_std__degC + temp_zero__K                   # standard gas temperature [K]
        temp_room__K = temp_room__degC + temp_zero__K                 # standard room temperature [K]

        # Approximations
        air_density__mol_m_3 = (P_std__Pa 
                                / (R__m3_Pa_K_1_mol_1 * temp_room__K)
                               )                                      # molar quantity of an ideal gas under room conditions [mol⋅m^-3]
        std__mol_m_3 = (P_std__Pa 
                        / (R__m3_Pa_K_1_mol_1 * temp_std__K)
                       )                                              # molar quantity of an ideal gas under standard conditions [mol⋅m^-3] 

        metabolism__molCO2_molO2_1 = 0.894                            # ratio: moles of CO₂ produced by (aerobic) human metabolism per mole of O₂ consumed 

        # National averages
        co2_ext_2022__ppm = 415                                       # Yearly average CO₂ concentration in Europe in 2022
        weight__kg = 77.5                                             # average weight of Dutch adult [kg]
        umol_s_1_p_1_MET_1 = (O2ml_min_1_kg_1_p_1_MET_1
                           * weight__kg
                           / s_min_1 
                           * (umol_mol_1 * std__mol_m_3 / ml_m_3)
                           )                                          # molar quantity of O₂inhaled by an average Dutch adult at 1 MET [µmol/(p⋅s)]
        co2_exhale__umol_p_1_s_1 = (metabolism__molCO2_molO2_1
                                    * desk_work__MET
                                    * umol_s_1_p_1_MET_1
                                   )                                  # molar quantity of CO₂ exhaled by Dutch desk worker doing desk work [µmol/(p⋅s)]
      
        # create empty dataframe for results of all homes
        df_results_per_period = pd.DataFrame()

        # ensure that dataframee is sorted
        if not df_data.index.is_monotonic_increasing:
            df_data = df_data.sort_index()  
        
        # # add empty columns to store fitting and learning results for time-varying 
        # df_data['sim_co2__ppm'] = np.nan
        # df_data['learned_valve_frac__0'] = np.nan
        # df_data['learned_occupancy__p'] = np.nan
       
        ids = df_data.index.unique('id').dropna()

        start_analysis_period = df_data.index.unique('timestamp').min().to_pydatetime()
        end_analysis_period = df_data.index.unique('timestamp').max().to_pydatetime()
        logging.info(f'Start of analyses: {start_analysis_period}')
        logging.info(f'End of analyses: {end_analysis_period}')

        daterange_frequency = str(learn_period__d) + 'D'
        logging.info(f'learn period: {daterange_frequency}')
       
        # perform sanity check; not any required column may be missing a value
        if req_col is None: # then we assume all properties from property_sources are required
            req_col = list(property_sources.values())
        if not req_col: # then the caller explicitly set the list to be empty
            df_data.loc[:,'sanity'] = True
        else:
            df_data.loc[:,'sanity'] = ~np.isnan(df_data[req_col]).any(axis="columns")

        # iterate over ids
        for id in tqdm(ids):
            
            if any(df_data.columns.str.startswith('model_')): 
                # calculate values from virtual rooms based on id 
                co2_ext__ppm = co2_ext_2022__ppm                      # Average CO₂ concentration in Europe in 2022 
                wind__m_s_1 = 3.0                                     # assumed wind speed for virtual rooms that causes infiltration
                room__m3 = id % 1e3
                vent_min__m3_h_1 = (id % 1e6) // 1e3
                vent_max__m3_h_1 = id // 1e6
                actual_A_inf__m2 = vent_min__m3_h_1 / (s_h_1 * wind__m_s_1)
            else:
                # get for real measured room, determine room-specific constants
                co2_ext__ppm = df_data.loc[id][property_sources['co2__ppm']].min()-1  # to compensate for sensor drift use  lowest co2__ppm measured in the room as approximation 
                wind__m_s_1 = 3.0                                                     # TODO assume this wind speed for real rooms as well, or use geospatially interpolated weather?
                room__m3 = df_metadata.loc[id]['room__m3']                            # get this parameter from the table passed as dataFrame
                vent_max__m3_h_1 = df_metadata.loc[id]['vent_max__m3_h_1']            # get this parameter from the table passed as dataFrame
                actual_A_inf__m2 = np.nan                                            # we don't knwo the actual infiltration area for real rooms

            vent_max__m3_s_1 = vent_max__m3_h_1 / s_h_1

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
                logging.info(f'Start datetime longest sane streak: {learn_streak_period_start}')
                logging.info(f'End datetime longest sane streak: {learn_streak_period_end}')
                logging.info(f'#rows in longest sane streak: {learn_streak_period_len}')
                 
                step__s = ((learn_streak_period_end - learn_streak_period_start).total_seconds()
                          /
                          (learn_streak_period_len-1)
                         )

                if learn_change_interval__min is None:
                    learn_change_interval__min = np.nan
                    MV_STEP_HOR =  1
                else:
                    # implement ceiling integer division by 'upside down' floor integer division
                    MV_STEP_HOR =  -((learn_change_interval__min * 60) // -step__s)

                logging.info(f'step__s:  {step__s}')
                logging.info(f'MV_STEP_HOR: {MV_STEP_HOR}')

                duration__s = step__s * learn_streak_period_len
                logging.info(f'duration__s:  {duration__s}')
                
                # setup learned_ and mae_ variables
                learned_A_inf__m2 = np.nan
                mae_A_inf__m2 = np.nan

                mae_valve_frac__0 = np.nan
                rmse_valve_frac__0 = np.nan

                mae_occupancy__p = np.nan
                rmse_occupancy__p = np.nan

                try:
            
                    ##################################################################################################################
                    # Gekko Model - Initialize
                    m = GEKKO(remote = False)
                    m.time = np.arange(0, duration__s, step__s)


                    # GEKKO time-varying variables: measured values or learned
                    if learn_occupancy__p:
                        occupancy__p = m.MV(value = df_learn[property_sources['occupancy__p']].astype('float32').values, lb=0, ub=12, integer=True)
                        occupancy__p.STATUS = 1; occupancy__p.FSTATUS = 1
                        if learn_change_interval__min is not None:
                            occupancy__p.MV_STEP_HOR = MV_STEP_HOR
                    else:
                        occupancy__p = m.MV(value = df_learn[property_sources['occupancy__p']].astype('float32').values)
                        occupancy__p.STATUS = 0; occupancy__p.FSTATUS = 1

                    if learn_valve_frac__0:
                        valve_frac__0 = m.MV(value = df_learn[property_sources['valve_frac__0']].values, lb=0, ub=1)
                        valve_frac__0.STATUS = 1; valve_frac__0.FSTATUS = 1
                        if learn_change_interval__min is not None:
                            valve_frac__0.MV_STEP_HOR = MV_STEP_HOR
                    else:
                        valve_frac__0 = m.MV(value = df_learn[property_sources['valve_frac__0']].values)
                        valve_frac__0.STATUS = 0; valve_frac__0.FSTATUS = 1


                    # GEKKO time-independent variables: approximated or learned
                    if 'A_inf__m2' in learn:
                        A_inf__m2 = m.FV(value = hints['A_inf__m2'], lb = 0)
                        A_inf__m2.STATUS = 1; A_inf__m2.FSTATUS = 0
                    else:
                        A_inf__m2 = hints['A_inf__m2']  

                    # GEKKO Control Varibale (predicted variable for which fit is optimized)
                    co2__ppm = m.CV(value = df_learn[property_sources['co2__ppm']].values) #[ppm]
                    co2__ppm.STATUS = 1; co2__ppm.FSTATUS = 1

                    # GEKKO - Equations
                    co2_elevation__ppm = m.Intermediate(co2__ppm - co2_ext__ppm)
                    co2_loss_vent__ppm_s_1 = m.Intermediate(co2_elevation__ppm * vent_max__m3_s_1 * valve_frac__0 / room__m3)
                    co2_loss_wind__ppm_s_1 = m.Intermediate(co2_elevation__ppm * wind__m_s_1 * A_inf__m2 / room__m3)
                    co2_loss__ppm_s_1 = m.Intermediate(co2_loss_vent__ppm_s_1 + co2_loss_wind__ppm_s_1)
                    co2_gain__ppm_s_1 = m.Intermediate(occupancy__p * co2_exhale__umol_p_1_s_1 / (room__m3 * air_density__mol_m_3))
                    m.Equation(co2__ppm.dt() == co2_gain__ppm_s_1 - co2_loss__ppm_s_1)


                    # GEKKO - Solver setting
                    m.options.IMODE = 5
                    if (learn_occupancy__p or learn_valve_frac__0):
                        m.options.SOLVER = 3
                    m.options.EV_TYPE = ev_type
                    m.options.NODES = 2
                    m.solve(disp = False)

                    # setting learned values and calculating error metrics
                    df_data.loc[(id,learn_streak_period_start):(id,learn_streak_period_end), 'sim_co2__ppm'] = np.asarray(co2__ppm)
                    mae_co2__ppm = Learner.mae(co2__ppm, df_learn[property_sources['co2__ppm']])
                    rmse_co2__ppm = Learner.rmse(co2__ppm, df_learn[property_sources['co2__ppm']])

                    if 'A_inf__m2' in learn:
                        learned_A_inf__m2 = A_inf__m2.value[0]
                        mae_A_inf__m2 = abs(learned_A_inf__m2 - actual_A_inf__m2)

                    if 'valve_frac__0' in learn:
                        df_data.loc[(id,learn_streak_period_start):(id,learn_streak_period_end), 'learned_valve_frac__0'] = np.asarray(valve_frac__0)
                        mae_valve_frac__0 = Learner.mae(valve_frac__0, df_learn[property_sources['valve_frac__0']])
                        rmse_valve_frac__0 = Learner.rmse(valve_frac__0, df_learn[property_sources['valve_frac__0']])

                    if 'occupancy__p'in learn:
                        df_data.loc[(id,learn_streak_period_start):(id,learn_streak_period_end), 'learned_occupancy__p'] = np.asarray(occupancy__p)
                        mae_occupancy__p = Learner.mae(occupancy__p, df_learn[property_sources['occupancy__p']])
                        rmse_occupancy__p = Learner.rmse(occupancy__p, df_learn[property_sources['occupancy__p']])

                    # Create a results row and add to results dataframe
                    df_results_per_period = pd.concat(
                        [
                            df_results_per_period,
                            pd.DataFrame(
                                {
                                    'id': [id],
                                    'learn_streak_period_start': [learn_streak_period_start],
                                    'learn_streak_period_end': [learn_streak_period_end],
                                    'step__s': [step__s],
                                    'learn_change_interval__min': [learn_change_interval__min],
                                    'duration__s': [duration__s],
                                    'EV_TYPE': [m.options.EV_TYPE],
                                    'vent_max__m3_h_1': [vent_max__m3_h_1],
                                    'actual_room__m3': [room__m3],
                                    'learned_A_inf__cm2': [learned_A_inf__m2 * 1e4],
                                    'actual_A_inf__cm2': [actual_A_inf__m2 * 1e4],
                                    'mae_A_inf__cm2': [mae_A_inf__m2 * 1e4],
                                    'mae_co2__ppm': [mae_co2__ppm],
                                    'rmse_co2__ppm': [rmse_co2__ppm],
                                    'mae_valve_frac__0': [mae_valve_frac__0],
                                    'rmse_valve_frac__0': [rmse_valve_frac__0],
                                    'mae_occupancy__p': [mae_occupancy__p],
                                    'rmse_occupancy__p': [rmse_occupancy__p]
                                }
                            )
                        ]
                    )

                    m.cleanup()

                    ##################################################################################################################

                except KeyboardInterrupt:    
                    logging.error(f'KeyboardInterrupt; home analysis {id} not complete; saving results so far then will exit...')
                    return df_results_per_period.set_index('id'), df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])

                except Exception as e:
                    logging.error(f'Exception {e} for home {id} in period from {learn_streak_period_start} to {learn_streak_period_end}; skipping...')
                    return df_results_per_period.set_index('id'), df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])
                    
            # after all learn periods of a single id

            
        # after all ids

        return df_results_per_period.set_index('id'), df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])

