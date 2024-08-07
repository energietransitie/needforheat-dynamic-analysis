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
              - property_sources['g_use_hhv__W']:
              - property_sources['e_use__W']:
              - property_sources['e_ret__W']:
        - 'property_sources', a dictionary that maps key listed above to actual column names in df_data
        - 'req_col' list: a list of column names: 
            - If any of the values in this column are NaN, the interval is not considered 'sane'.
            - If you do not specify a value for req_col or specify req_col = None, then all properties from the property_sources dictionary are considered required
            - to speficy NO volumns are required, specify property_sources = []
        - a df_metadata with index 'id' and columns:
            - none (this feature is not used in the current implementation yet, but added here for consistentcy with the learn_room_parameters() function)
        - hints: a dictionary that maps keys to fixed values to be used for analysis (set value for None to learn it):
            - 'A_sol__m2': apparent solar aperture [m^2]
            - 'eta_ch_hhv__W0': superior efficiency [-] of the heating system (in NL 0.963 is a reasonable hint)
            - 'g_not_ch_hhv__W': average yearly gas power (higher heating value)  for other purposes than heating 
            - 'eta_not_ch_hhv__W0': superior efficiency [-] of heating the home indirectly using gas (in NL 0.34 is a reasonable hint)
            - 'wind_chill__degC_s_m_1': (in NL typically 0.67, according to KNMI: https://cdn.knmi.nl/knmi/pdf/bibliotheek/knmipubmetnummer/knmipub219.pdf)
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
                           'eta_ch_hhv__W0',
                           'eta_not_ch_hhv__W0',
                           'g_not_ch_hhv__W',
                           'occupancy__p',
                           'Q_gain_int__W_p_1',
                           'wind_chill__degC_s_m_1'
                          ]
        for hint in mandatory_hints:
            if not (hint in hints or isinstance(hints[hint], numbers.Number)):
                raise TypeError(f'hints[{hint}] parameter must be a number')

        # check for unlearnable parameters
        not_learnable =   ['eta_not_ch_hhv__W0',
                           'g_not_ch_hhv__W',
                           'occupancy__p',
                           'Q_gain_int__W_p_1'
                          ]
        
        for param in learn:
            if param in not_learnable:
                raise LearnError(f'No support for learning {param} (yet).')


        # National averages
        g_not_ch_hhv__W = hints['g_not_ch_hhv__W'] # average gas usage per year for cooking and DHW, i.e. not for CH  
        Q_gain_int_occup__W = hints['Q_gain_int__W_p_1'] * hints['occupancy__p']    # average heat gain per occupant
      
        # create empty dataframe for results of all homes
        df_results_per_period = pd.DataFrame()

        # ensure that dataframe is sorted
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
                # calculate values from synthetic home based on id 
                actual_H__W_K_1 = id // 1e5
                actual_tau__h = (id % 1e5) // 1e2
                actual_A_sol__m2 = id % 1e2
                actual_C__Wh_K_1 = actual_H__W_K_1 * actual_tau__h
                actual_eta_ch_hhv__W0 = eta_ch_nl_avg_hhv__J0 # efficiency used for calculating synthetic home values)
                actual_wind_chill__degC_s_m_1 = 0.67 # efficiency used for calculating synthetic home values)
            else:
                actual_H__W_K_1 = np.nan
                actual_tau__h = np.nan
                actual_A_sol__m2 = np.nan
                actual_C__Wh_K_1 = np.nan
                actual_eta_ch_hhv__W0 = np.nan
                actual_wind_chill__degC_s_m_1 = np.nan

            df_data.loc[id, ['g_use_ch_hhv_W', 'g_use_not_ch__W']] = Learner.gas_split_simple(df_data.loc[id][property_sources['g_use_hhv__W']], g_not_ch_hhv__W).values

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

                learned_eta_ch_hhv__W0 = np.nan
                mae_eta_ch_hhv__W0 = np.nan

                learned_wind_chill__degC_s_m_1 = np.nan
                mae_wind_chill__degC_s_m_1 = np.nan
                
                ##################################################################################################################
                # GEKKO code

                try:
            
                    # GEKKO Model - Initialize
                    m = GEKKO(remote=False)
                    m.time = np.arange(0, duration__s, step__s)

                    # Model parameter: H [W/K]: specific heat loss
                    H__W_K_1 = m.FV(value=300.0, lb=0, ub=1000)
                    H__W_K_1.STATUS = 1; H__W_K_1.FSTATUS = 0
                    
                    # Model parameter: tau [s]: effective thermal inertia
                    tau__s = m.FV(value=(100 * s_h_1), lb=(10 * s_h_1), ub=(1000 * s_h_1))
                    tau__s.STATUS = 1; tau__s.FSTATUS = 0

                    # eta_ch_hhv__W0 [-]: upper heating efficiency of the central heating system
                    if 'eta_ch_hhv__W0' in learn:
                        eta_ch_hhv__W0 = m.FV(value = hints['eta_ch_hhv__W0'], lb = 0, ub = 1.0)
                        eta_ch_hhv__W0.STATUS = 1; eta_ch_hhv__W0.FSTATUS = 0
                        # eta_ch_hhv__W0.DMAX = 0.25
                    else:
                        # Set eta_ch_hhv__W0 to a fixed value when it should not be learned 
                        eta_ch_hhv__W0 = m.Param(value = hints['eta_ch_hhv__W0'])
                        learned_eta_ch_hhv__W0 = np.nan

                    g_use_ch_hhv_W = m.MV(value = df_learn['g_use_ch_hhv_W'].astype('float32').values)
                    g_use_ch_hhv_W.STATUS = 0; g_use_ch_hhv_W.FSTATUS = 1

                    # Q_gain_CH [W]: heat gain from natural gas used for central heating
                    Q_gain_g_CH__W = m.Intermediate(g_use_ch_hhv_W * eta_ch_hhv__W0)
                
                    g_use_not_ch__W = m.MV(value = df_learn['g_use_not_ch__W'].astype('float32').values)
                    g_use_not_ch__W.STATUS = 0; g_use_not_ch__W.FSTATUS = 1

                    # Q_gain_not_ch  [W]: heat gain from natural gas used for central heating
                    Q_gain_g_not_ch__W = m.Intermediate(g_use_not_ch__W * hints['eta_not_ch_hhv__W0'])

                    # e_use [W] - e_ret [W] : internal heat gain from internally used electricity
                    e_use__W = m.MV(value = df_learn[property_sources['e_use__W']].astype('float32').values)
                    e_use__W.STATUS = 0; e_use__W.FSTATUS = 1

                    e_ret__W = m.MV(value = df_learn[property_sources['e_ret__W']].astype('float32').values)
                    e_ret__W.STATUS = 0; e_ret__W.FSTATUS = 1

                    # Q_gain_int [W]: calculated heat gain from internal sources
                    Q_gain_int__W = m.Intermediate(e_use__W - e_ret__W + Q_gain_int_occup__W + Q_gain_g_not_ch__W)

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
                        # set fixed wind chill factor based on hint
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

                    if 'eta_ch_hhv__W0' in learn:
                        learned_eta_ch_hhv__W0 = eta_ch_hhv__W0.value[0]
                        mae_eta_ch_hhv__W0 = abs(learned_eta_ch_hhv__W0 - actual_eta_ch_hhv__W0) # evaluates to np.nan if no actual value

                    if 'wind_chill__degC_s_m_1' in learn:
                        learned_wind_chill__degC_s_m_1 = wind_chill__degC_s_m_1.value[0]
                        mae_wind_chill__degC_s_m_1 = abs(learned_wind_chill__degC_s_m_1 - actual_wind_chill__degC_s_m_1) # evaluates to np.nan if no actual value


                except KeyboardInterrupt:    
                    logging.error(f'KeyboardInterrupt; home analysis {id} not complete; saving results so far then will exit...')
                    # only then exit the function and return to caller
                    return df_results_per_period.set_index('id'), df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])

                except Exception as e:
                    logging.error(f'Exception {e} for home {id} in period from {learn_streak_period_start} to {learn_streak_period_end}; skipping...')
                
                finally:
                    # create a results row and add to results per period dataframe
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
                                    'learned_eta_ch_hhv__W0': [learned_eta_ch_hhv__W0],
                                    'actual_eta_ch_hhv__W0': [actual_eta_ch_hhv__W0],
                                    'mae_eta_ch_hhv__W0': [mae_eta_ch_hhv__W0],
                                    'learned_wind_chill__degC_s_m_1': [learned_wind_chill__degC_s_m_1],
                                    'mae_wind_chill__degC_s_m_1': [mae_wind_chill__degC_s_m_1]
                                }
                            )
                        ]
                    )

                    m.cleanup()
                ##################################################################################################################

            # after all learn periods of a single id
            
        # after all ids

        return df_results_per_period.set_index('id'), df_data.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s', 'sanity'])
        # return df_results_per_period.set_index('id'), df_data.drop(columns=['interval__s', 'sanity'])
 
    

