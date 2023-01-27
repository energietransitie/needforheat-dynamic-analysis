from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from gekko import GEKKO
from tqdm.notebook import tqdm

import numbers
import logging

class Learner():
    
    @staticmethod
    def learn_home_parameters(df_data_ids:pd.DataFrame,
                             learn_period_d=7, 
                             req_col:list = [], sanity_threshold_timedelta:timedelta=timedelta(hours=24),
                             hint_A__m2=None, hint_eta_sup_CH__0=0.97, ev_type=2) -> pd.DataFrame:
        """
        Input:  
        - a preprocessed dataframe with
            - a MultiIndex ['id', 'timestamp'], where
                - the column 'timestamp' is timezone-aware
                - time intervals between consecutive measurements are constant
                - but there may be gaps of multiple intervals with no measurements
                - multiple sources for the same property are already dealth with in preprocessing
            - columns:
                'temp_in__degC',
                'temp_out__degC',
                'wind__m_s_1',
                'ghi__W_m_2',
                'g_use__W',
                'e_use__W',
                'e_ret__W', 
            ]
        and optionally,
        - the number of days to use as learn period in the analysis
        - 'ev_type'
        - a 'req_col' list: a list of column names: if any of the values in this column are NaN, the interval is not considered 'sane'
        - a sanity_theshold_timedelta: only the longest streaks with sane data longer than this is considered for analysis during each learn_period
        
        Output:
        - a dataframe with per id and per learn_period the learned parameters
        - a dataframe with additional column:
          - 'temp_in_sim__degC' best fitting temperature series for each learn period
        """
        
        if not ((hint_A__m2 is None) or isinstance(hint_A__m2, numbers.Number)):
            raise TypeError('hint_A__m2 parameter must be a number or None')

        # set default values for parameters not set
        
        if (learn_period_d is None):
            learn_period_d = 7

        ids= df_data_ids.index.unique('id').dropna()
        start_analysis_period = df_data_ids.index.unique('timestamp').min().to_pydatetime()
        end_analysis_period = df_data_ids.index.unique('timestamp').max().to_pydatetime()
        
        daterange_frequency = str(learn_period_d) + 'D'

        logging.info('Homes to analyse: ', ids)
        logging.info('Start of analyses: ', start_analysis_period)
        logging.info('End of analyses: ', end_analysis_period)
        logging.info('learn period: ', daterange_frequency)
        logging.info('Hint for effective window are A [m2]: ', hint_A__m2)
        logging.info('Hint for superior heating efficiency eta [-]: ', hint_eta_sup_CH__0)
        logging.info('EV_TYPE: ', ev_type)

        # perform sanity check; not any required column may be missing a value
        if (req_col == []):
            df_data_ids.loc[:,'sanity'] = True
        else:
            df_data_ids.loc[:,'sanity'] = ~np.isnan(df_data_ids[req_col]).any(axis="columns")
            
        # Conversion factor s_h_1 [s/h]  = 60 [min/h] * 60 [s/min] 
        s_h_1 = (60 * 60) 

        # Conversion factor J_kWh_1 [J/kWh]  = 1000 [Wh/kWh] * s_h_1 [s/h] * 1 [J/Ws]
        J_kWh_1 = 1000 * s_h_1

        # Conversion factor s_d_1 [s/d]  = 24 [h/d] * s_h_1 [s/h] 
        s_d_1 = (24 * s_h_1) 
        # Conversion factor s_a_1 [s/a]  = 365.25 [d/a] * s_d_1 [s/d] 
        s_a_1 = (365.25 * s_d_1) 

        # Conversion factor h_sup_J_m_3 superior calorific value of natural gas from the Groningen field = 35.17 [MJ/m3]
        h_sup_J_m_3 = value=35.17e6


        # National averages

        p = 2.2  # average number of people in Dutch household
        Q_gain_int__W_p_1 = 61  # average heat gain per average person with average behaviour and occupancy
        Q_gain_int_occup__W = Q_gain_int__W_p_1 * p

        # currently, we use 339 [m3/a] national average gas usage per year for cooking and DHW, i.e. not for CH
        g_noCH__m3_s_1 = (339.0 / s_a_1)  

        # create empty dataframe for results of all homes
        df_results = pd.DataFrame()

        # create empty dataframe for temperature simultion results of all homes
        df_results_allhomes_allweeks_tempsim = pd.DataFrame()
           
        # iterate over homes
        for id in tqdm(ids):
            
            if id > 1e6: # This ismeans we're analysing a virtual home constants 
                actual_H__W_K_1 = id // 1e5
                actual_tau__h = (id % 1e5) // 1e2
                actual_tau__s = actual_tau__h * s_h_1
                actual_A__m2 = id % 1e2
                actual_C__Wh_K_1 = actual_H__W_K_1 * actual_tau__h
                actual_C__J_K_1 = actual_H__W_K_1 * actual_tau__s
            
            # create empty dataframe for results of a home
            df_results_home = pd.DataFrame()

            # create empty dataframe for temperature simulation results of a single home
            df_results_home_allweeks_tempsim = pd.DataFrame()
            
            # create empty dataframe for temperature simulation results of a single week of a single home
            df_results_homeweek_tempsim  = pd.DataFrame()

            logging.info('Home id: ', id)

            df_learn = df_data_ids.loc[id].copy() #TODO: don't work with a copy, nut add learnt data (_sim) to original DataFrame, like in learn_room_parameters
            
            # calculate timedelta for each interval (code is suitable for unevenly spaced measurementes)
            df_learn['interval__s'] = (df_learn
                                       .index.to_series()
                                       .diff()
                                       .shift(-1)
                                       .apply(lambda x: x.total_seconds())
                                       .fillna(0)
                                       .astype(int)
                                       )

            df_learn['streak_id'] = np.nan
            df_learn['streak_cumulative_duration__s'] = np.nan
                        
            # split gas over CH and noCH per home based on the entire period 
            
            # in a future version we intend to use a value specific per home based on average usage of natural gas in the summer months (June - August) 
           
            g_use_noCH__W = g_noCH__m3_s_1 * h_sup_J_m_3 

            logging.info('g_noCH__m3_s_1: {:.5E}'.format(g_noCH__m3_s_1))
            logging.info('g_use_noCH__W: ', g_use_noCH__W)

            # using this average, distribute gas usage over central heating (CH) versus no Central Heating (noCH)
            df_learn['g_use_noCH__W'] = g_use_noCH__W
            df_learn['g_use_CH__W'] = df_learn['g_use__W'] - g_use_noCH__W

            # Avoid negative values for heating; simple fix: negative value with zero in g_use_CH__W
            df_learn.loc[df_learn.g_use_CH__W < 0, 'g_use_CH__W'] = 0

            # Compensate by scaling down g_use_CH__W 
            g_use_home__W = df_learn['g_use__W'].mean()
            uncorrected_g_use_CH__W = df_learn['g_use_CH__W'].mean()
            scaling_factor =   (g_use_home__W - g_use_noCH__W) / uncorrected_g_use_CH__W  
            df_learn['g_use_CH__W'] = df_learn['g_use_CH__W'] * scaling_factor
            corrected_g_use_CH__W = df_learn['g_use_CH__W'].mean()

            logging.info('id: ', id)
            logging.info('g_use_home__W: ', g_use_home__W)
            logging.info('uncorrected_g_use_CH__W: ', uncorrected_g_use_CH__W)
            logging.info('scaling_factor: ', scaling_factor)
            logging.info('corrected_g_use_CH__W: ', corrected_g_use_CH__W)
            logging.info('g_use_noCH__W + corrected_g_use_CH__W: ', g_use_noCH__W + corrected_g_use_CH__W)
            
            learn_period_starts = pd.date_range(start=start_analysis_period, end=end_analysis_period, inclusive='both', freq=daterange_frequency)

            learn_period_iterator = tqdm(learn_period_starts)

            # iterate over learn periods
            for learn_period_start in learn_period_iterator:

                learn_period_end = min(end_analysis_period, learn_period_start + timedelta(days=learn_period_d))

                if (learn_period_end < end_analysis_period):
                    df_learn = df_learn[learn_period_start:learn_period_end].iloc[:-1]
                else:
                    df_learn = df_learn[learn_period_start:end_analysis_period]
                    
                learn_period_end = df_learn.index.max()

                logging.info('Start datetime: ', learn_period_start)
                logging.info('End datetime: ', learn_period_end)
                
                #first check whether there is even a single sane value
                if len(df_learn.query('sanity == True')) == 0:
                    logging.info(f'For home {id} there is no sane data in the period from {learn_period_start} to {learn_period_start}; skipping...')
                    continue                       
                
                # restrict the df_learn to the longest streak of sane data
                ## give each streak a separate id
                df_learn.streak_id = df_learn.sanity.ne(df_learn.sanity.shift()).cumsum()
                df_learn.streak_cumulative_duration__s = df_learn.groupby('streak_id').interval__s.cumsum()
                ## make sure streaks with insane values are not considered
                df_learn.loc[df_learn.sanity == False, 'streak_cumulative_duration__s'] = np.nan 
                ## get the longest streak: the part of the dataframe where the streak_id matches the (first) streak_id that has the longest cumulative duration
                df_learn = df_learn.query('streak_id == ' + str(df_learn.loc[df_learn.streak_cumulative_duration__s.idxmax()].streak_id))

                logging.info('Start datetime longest sane streak: ', learn_period_start)
                logging.info('End datetime longest sane streak: ', learn_period_end)
                
                df_learn = df_learn.drop(columns=['streak_id', 'streak_cumulative_duration__s', 'interval__s'])

                # then check whether enough data, if not then skip this homeweek, move on to next
                if ((learn_period_end - learn_period_start) < sanity_threshold_timedelta):
                    logging.info(f'For home {id} the longest streak of sane data is less than {sanity_threshold_timedelta} in the period from {learn_period_start} to {learn_period_start}; skipping...')
                    continue

                step__s = ((df_learn.index.max() - df_learn.index.min()).total_seconds()
                          /
                          (len(df_learn)-1)
                         )
                duration__s = step__s * len(df_learn)
                
                if (hint_A__m2 is None):
                    logging.info('home {0} from {1} to {2} only run learning loop once with A as learnable model parameter'
                          .format(id,learn_period_start,learn_period_end))
                    innerloop=[np.NaN]
                else:
                    logging.info('home {0} from {1} to {2} run learning loop twice: once with A={3} and once as as learnable model parameter'
                          .format(id,learn_period_start,learn_period_end, hint_A__m2))
                    innerloop=[hint_A__m2, np.NaN]

                for iterator_A__m2 in innerloop:

                    try:


                        ########################################################################################################################
                        # Gekko Model - Initialize
                        ########################################################################################################################
 
                        # initialize gekko
                        m = GEKKO(remote=False)
                        m.time = np.arange(0, duration__s, step__s)


                        ########################################################################################################################
                        # Gekko Model - Model Parameters
                        ########################################################################################################################
                        
                        # Model parameter: H [W/K]: specific heat loss
                        H__W_K_1 = m.FV(value=300.0, lb=0, ub=1000)
                        H__W_K_1.STATUS = 1; H__W_K_1.FSTATUS = 0
                        
                        # Model parameter: tau [s]: effective thermal inertia
                        tau__s = m.FV(value=(100 * s_h_1), lb=(10 * s_h_1), ub=(1000 * s_h_1))
                        tau__s.STATUS = 1; tau__s.FSTATUS = 0

                        
                        ########################################################################################################################
                        #                                               Gekko - Equations
                        ########################################################################################################################

                        
                        ########################################################################################################################
                        # Equation - Q_gain_CH__W: heat gain from natural gas used for central heating
                        ########################################################################################################################

                        # eta_sup_CH__0 [-]: upper heating efficiency of the central heating system

                        # This section of code (later) when  eta_sup_CH__0 is estimated using the model
                        # eta_sup_CH__0 = m.FV(value=0.8, lb=0, ub=1.0)
                        # eta_sup_CH__0.STATUS = 1
                        # eta_sup_CH__0.FSTATUS = 0
                        ## eta_sup_CH__0.DMAX = 0.25

                        # Fix eta_sup_CH__0 when a hint is given 
                        eta_sup_CH__0 = m.Param(value=hint_eta_sup_CH__0)
                        
                        g_use_CH__W = m.MV(value = df_learn.g_use_CH__W.values)
                        g_use_CH__W.STATUS = 0; g_use_CH__W.FSTATUS = 1

                        Q_gain_g_CH__W = m.Intermediate(g_use_CH__W * eta_sup_CH__0)
                    
                    
                        ########################################################################################################################
                        # Equation - Q_gain_noCH__W: heat gain from natural gas used for central heating
                        ########################################################################################################################

                        # eta_sup_noCH__0 [-]: superior efficiency of heating the home indirectly using gas, 
                        # for other primary purposes than heating the home, # eq48, PPT slide 24
                        eta_sup_noCH__0 = m.Param(value=0.34)
                        
                        g_use_noCH__W = m.MV(value = df_learn.g_use_noCH__W.values)
                        g_use_noCH__W.STATUS = 0; g_use_noCH__W.FSTATUS = 1
                        
                        Q_gain_g_noCH__W = m.Intermediate(g_use_noCH__W * eta_sup_noCH__0)

                        ########################################################################################################################
                        # Equation - Q_gain_int__W: Heat gain from internal sources
                        ########################################################################################################################
                        # Manipulated Variables: e_use__W - e_ret__W : internal heat gain from internally used electricity

                        e_use__W = m.MV(value = df_learn.e_use__W.values)
                        e_use__W.STATUS = 0; e_use__W.FSTATUS = 1

                        e_ret__W = m.MV(value = df_learn.e_ret__W.values)
                        e_ret__W.STATUS = 0; e_ret__W.FSTATUS = 1

                        Q_gain_int__W = m.Intermediate(e_use__W - e_ret__W + Q_gain_int_occup__W + Q_gain_g_noCH__W)

                        ########################################################################################################################
                        # Equation - Q_gain_sol__W: : heat gain from solar irradiation
                        ########################################################################################################################

                        # Model parameter/fixed value: A [m2]: Effective area of the solar aperture
                        if np.isnan(iterator_A__m2):
                            A__m2 = m.FV(value=5, lb=1, ub=100); A__m2.STATUS = 1; A__m2.FSTATUS = 0
                        else:
                            A__m2 = m.Param(value=iterator_A__m2)

                        ghi__W_m_2 = m.MV(value = df_learn.ghi__W_m_2.values)
                        ghi__W_m_2.STATUS = 0; ghi__W_m_2.FSTATUS = 1
                        
                        Q_gain_sol__W = m.Intermediate(ghi__W_m_2 * A__m2)
                        
                        ########################################################################################################################
                        # Manipulated Variable (MV): temp_out_e__degC [°C]: effective outdoor temperature
                        ########################################################################################################################

                        temp_out__degC = m.MV(value = df_learn.temp_out__degC.values)
                        temp_out__degC.STATUS = 0; temp_out__degC.FSTATUS = 1

                        wind__m_s_1 = m.MV(value = df_learn.wind__m_s_1.values)
                        wind__m_s_1.STATUS = 0; wind__m_s_1.FSTATUS = 1
                        
                        # This section of code (later) when wind_chill__degC_s_m_1 is estimated using the model
                        # wind_chill__degC_s_m_1 = m.FV(value=0.67, lb=0, ub=1.0)
                        # wind_chill__degC_s_m_1.STATUS = 1; wind_chill__degC_s_m_1.FSTATUS = 0

                        # Fix average wind sensitivity based on KNMI estimate
                        # Source: https://cdn.knmi.nl/knmi/pdf/bibliotheek/knmipubmetnummer/knmipub219.pdf 
                        wind_chill__degC_s_m_1 = m.Param(value=0.67)

                        #calculate effective outdoor temperature by compensating for wind chill factor
                        temp_out_e__degC = m.Intermediate(temp_out__degC - wind_chill__degC_s_m_1 * wind__m_s_1)

                        ########################################################################################################################
                        # Control Variable temp_in__degC: Indoor temperature
                        ########################################################################################################################
                        temp_in__degC = m.CV(value = df_learn.temp_in__degC.values)
                        temp_in__degC.STATUS = 1; temp_in__degC.FSTATUS = 1
                        # temp_in__degC.MEAS_GAP= 0.25
                        

                        ########################################################################################################################
                        # Main Equations & Solver
                        ########################################################################################################################
                        Q_gain__W = m.Intermediate(Q_gain_g_CH__W + Q_gain_sol__W + Q_gain_int__W)
                        Q_loss__W = m.Intermediate(H__W_K_1 * (temp_in__degC - temp_out_e__degC)) 
                        C__J_K_1  = m.Intermediate(H__W_K_1 * tau__s) 
                        m.Equation(temp_in__degC.dt() == ((Q_gain__W - Q_loss__W) / C__J_K_1))
                        
                        m.options.IMODE = 5
                        m.options.EV_TYPE = ev_type # specific objective function (L1-norm vs L2-norm)
                        m.solve(False)      

                        ########################################################################################################################
                        #                                                       Result
                        ########################################################################################################################

                        # create a deep copy
                        df_results_homeweek_tempsim = df_learn.copy(deep=True)
                        df_results_homeweek_tempsim['temp_in_sim__degC'] = list(temp_in__degC.value)
                        df_results_homeweek_tempsim['A_is_fixed__bool'] = not np.isnan(iterator_A__m2)

                        # TODO: don't use copy but paste directly into df_learn or df_data_ids
                        # TODO: determine start and end
                        # df_learn.loc[start:end, 'temp_in_sim__degC'] = temp_in__degC
                        # df_learn.loc[start:end, 'A_is_fixed__bool'] = not np.isnan(iterator_A__m2)
                        # df_data_ids.loc[id, start:end, 'temp_in_sim__degC'] = temp_in__degC
                        # df_data_ids.loc[id, start:end, 'A_is_fixed__bool'] = not np.isnan(iterator_A__m2)

                        
                        mae_K = (abs(df_results_homeweek_tempsim['temp_in_sim__degC'] - df_results_homeweek_tempsim['temp_in__degC'])).mean()
                        rmse_K = ((df_results_homeweek_tempsim['temp_in_sim__degC'] - df_results_homeweek_tempsim['temp_in__degC'])**2).mean()**0.5

                        logging.info('duration [s]: ', duration__s)
                        logging.info('EV_TYPE: ', m.options.EV_TYPE)
                        logging.info('H [W/K]: ', round(H__W_K_1.value[0], 4))
                        logging.info('tau [h]: ', round(tau__s.value[0] / s_h_1, 2))
                        logging.info('A [m2]: ', round(A__m2.value[0], 2))
                        logging.info('A value fixed: ', not np.isnan(iterator_A__m2))
                        logging.info('eta_sup [-]: ', round(eta_sup_CH__0.value[0], 2))
                        logging.info('eta_sup value fixed: ', True)
                        logging.info('MAE: ', mae_K)
                        logging.info('RMSE: ', rmse_K)
                        

                        # Create a results row
                        if id > 1e6: # This means we're analysing a virtual home constants 
                            df_result_row = pd.DataFrame({
                                'start_learn_period': [learn_period_start],
                                'end_learn_period': [learn_period_end],
                                'pseudonym': [id],
                                'duration__s': [duration__s],
                                'EV_TYPE': [m.options.EV_TYPE],
                                'actual_H__W_K_1': [actual_H__W_K_1], 
                                'actual_tau__h': [actual_tau__h],
                                'actual_A__m2': [actual_A__m2],
                                'actual_C__Wh_K_1': [actual_C__Wh_K_1],
                                'H__W_K_1': [H__W_K_1.value[0]],
                                'tau__h': [tau__s.value[0] / s_h_1],
                                'C__Wh_K_1':[H__W_K_1.value[0] * tau__s.value[0] / s_h_1],
                                'A__m2': [A__m2.value[0]],
                                'A__m2_fixed': [not np.isnan(iterator_A__m2)],
                                'eta_sup__0': [eta_sup_CH__0.value[0]],
                                'eta_sup_fixed': [True],
                                'MAE_K': [mae_K],
                                'RMSE_K': [rmse_K]})
                        else:
                            df_result_row = pd.DataFrame({
                                'start_learn_period': [learn_period_start],
                                'end_learn_period': [learn_period_end],
                                'pseudonym': [id],
                                'duration__s': [duration__s],
                                'EV_TYPE': [m.options.EV_TYPE],
                                'H__W_K_1': [H__W_K_1.value[0]],
                                'tau__h': [tau__s.value[0] / s_h_1],
                                'C__Wh_K_1':[H__W_K_1.value[0] * tau__s.value[0] / s_h_1],
                                'A__m2': [A__m2.value[0]],
                                'A__m2_fixed': [not np.isnan(iterator_A__m2)],
                                'eta_sup__0': [eta_sup_CH__0.value[0]],
                                'eta_sup_fixed': [True],
                                'MAE_K': [mae_K],
                                'RMSE_K': [rmse_K]})
                            
                        df_result_row.set_index(['start_learn_period'], inplace=True)
                        
                        # add week to home results dataframe

                    except KeyboardInterrupt:    
                        logging.error(str('KeyboardInterrupt; home analysis {0} not complete; saving results so far then will exit...'.format(id)))

                        # do NOT write an empty line for this iteration, to indicate it is not fully processed and we don't know 
                        # but DO include the incomplete home results in the final export
                        df_results = pd.concat([df_results, df_results_home])

                        # only then exit the function and return to caller
                        return  df_results, df_results_allhomes_allweeks_tempsim

                    except Exception as e:
                        # do write an empty line for this iteration, to indicate it is fully processed 
                        # and to indicate that we do know know  GEKKO could not learn parameters for this learn period for this home 
                        
                        logging.error(str('Exception {0} for home {1} in period from {2} to {3}; skipping...'
                                  .format(e, id,learn_period_start,learn_period_end)))
                        if id > 1e6: # This means we're analysing a virtual home constants 
                            df_result_row = pd.DataFrame({
                                'start_learn_period': [learn_period_start],
                                'end_learn_period': [learn_period_end],
                                'pseudonym': [id],
                                'duration__s': [np.nan],
                                'EV_TYPE': [np.nan],
                                'actual_H__W_K_1': [np.nan], 
                                'actual_tau__h': [np.nan],
                                'actual_A__m2': [np.nan],
                                'actual_C__J_K_1': [np.nan],
                                'H__W_K_1': [np.nan],
                                'tau__h': [np.nan],
                                'C__Wh_K_1':[np.nan],
                                'A__m2': [np.nan],
                                'A__m2_fixed': [not (np.isnan(iterator_A__m2))],
                                'eta_sup__0': [np.nan],
                                'eta_sup_fixed': [True],
                                'MAE_K': [np.nan],
                                'RMSE_K': [np.nan]})
                        else:
                            df_result_row = pd.DataFrame({
                                'start_learn_period': [learn_period_start],
                                'end_learn_period': [learn_period_end],
                                'pseudonym': [id],
                                'duration__s': [np.nan],
                                'EV_TYPE': [np.nan],
                                'H__W_K_1': [np.nan],
                                'tau__h': [np.nan],
                                'C__Wh_K_1':[np.nan],
                                'A__m2': [np.nan],
                                'A__m2_fixed': [not (np.isnan(iterator_A__m2))],
                                'eta_sup__0': [np.nan],
                                'eta_sup_fixed': [True],
                                'MAE_K': [np.nan],
                                'RMSE_K': [np.nan]})
                        df_result_row.set_index(['start_learn_period'], inplace=True)
                        pass

                    #after a single innerloop for A fixed or learnable
                    try:
                        df_results_home = pd.concat([df_results_home, df_result_row])
                        
                        if df_results_homeweek_tempsim is not None and len(df_results_homeweek_tempsim.columns)>0:
                            df_results_home_allweeks_tempsim = pd.concat([df_results_home_allweeks_tempsim, df_results_homeweek_tempsim])
                            df_results_home_allweeks_tempsim.describe(include='all')
                        
                    except KeyboardInterrupt:    
                        logging.error(str('KeyboardInterrupt; home analysis {0} not complete; saving results so far then will exit...'.format(id)))

                        # do write full line for this iteration, to indicate it is fully processed and we do know 
                        df_results_home = pd.concat([df_results_home, df_result_row])

                        # and include the incomplete home results in the final export
                        df_results = pd.concat([df_results, df_results_home])

                        # only then exit the function and return to caller
                        return  df_results, df_results_allhomes_allweeks_tempsim

                #after a single innerloop for A fixed or learnable
                logging.info(str('Analysis of all learn periods for a single inner loop for home {0} complete.'.format(id)))

            #after all learn periods of a single home; after a single innerloop for A fixed or learnable
            logging.info(str('Analysis of all learn periods for home {0} complete.'.format(id)))
            try:
                df_results = pd.concat([df_results, df_results_home])
                
                # label each line in the temperature simulation result dataframa with homepseudonym
                df_results_home_allweeks_tempsim.insert(loc=0, column='id', value=id)
                #and add to result dataframe of all homes
                df_results_allhomes_allweeks_tempsim = pd.concat([df_results_allhomes_allweeks_tempsim, df_results_home_allweeks_tempsim])
                
                
            except KeyboardInterrupt:    
                logging.error(str('KeyboardInterrupt; home analysis {0} complete; saving results so far then will exit...'.format(id)))
                return  df_results, df_results_allhomes_allweeks_tempsim


        # reset after all homes
        df_results = df_results.reset_index().rename(columns = {'pseudonym':'id'}).set_index(['id', 'start_learn_period'])

        # cols = list(df_results_allhomes_allweeks_tempsim.columns)
        
        df_results_allhomes_allweeks_tempsim = df_results_allhomes_allweeks_tempsim.drop(columns=['sanity'])
        df_results_allhomes_allweeks_tempsim = df_results_allhomes_allweeks_tempsim.reset_index().rename(columns = {'index':'timestamp'}).set_index(['id', 'timestamp'])
        
        #return simulation results via df_data_ids parameter
        df_data_ids = df_results_allhomes_allweeks_tempsim
        
        return  df_results, df_results_allhomes_allweeks_tempsim
    
    
    @staticmethod
    def learn_room_parameters(df_data_ids:pd.DataFrame, ev_type=2) -> pd.DataFrame:
        """
        Input:  
        - a preprocessed dataframe with
            - a MultiIndex ['id', 'timestamp'], where
                - the column 'timestamp' is timezone-aware
                - time intervals between consecutive measurements are constant
                - but there may be gaps of multiple intervals with no measurements
                - multiple sources for the same property are already dealth with in preprocessing
        - a dataframe with
            - a MultiIndex ['id', 'source', 'timestamp'], where the column 'timestamp' is timezone-aware
            - columns:
              - 'occupancy__p': average number of people present in the room,
              - 'co2__ppm': average CO₂-concentration in the room,
              - 'valve_frac__0' opening fraction of the ventilation valve 
        and optionally,
        - 'ev_type': type 2 is usually recommended, since this is typically more than 50 times faster
        
        Output:
        - a dataframe with per id the learned parameters
        - a dataframe with additional column(s):
          - 'co2_sim__ppm' best fiting temperature series for id
        """
        

        # Conversion factors
        s_min_1 = 60
        min_h_1 = 60
        s_h_1 = s_min_1 * min_h_1
        mL_m_3 = 1e3 * 1e3
        million = 1e6
        mL_min_1_kg_1_p_1_MET_1 = 3.5                                 # conversion factor for Metabolic Equivalent of Task [mlO₂‧kg^-1‧min^-1‧MET^-1] 

        # Constants
        desk_work__MET = 1.5                                          # Metabolic Equivalent of Task for desk work [MET]
        P_std__Pa = 101325                                            # standard gas pressure [Pa]
        R__m3_Pa_K_1_mol_1 = 8.3145                                   # gas constant [m^3⋅Pa⋅K^-1⋅mol^-1)]
        T_room__degC = 20.0                                           # standard room temperature [°C]
        T_std__degC = 0.0                                             # standard gas temperature [°C]
        T_zero__K = 273.15                                            # 0 [°C] = 273.15 [K]
        T_std__K = T_zero__K + T_std__degC                            # standard gas temperature [K]
        T_room__K = T_zero__K + T_room__degC                          # standard room temperature [K]

        # Approximations
        room__mol_m_3 = P_std__Pa / (R__m3_Pa_K_1_mol_1 * T_room__K)  # molar quantity of an ideal gas under room conditions [mol⋅m^-3]
        std__mol_m_3 = P_std__Pa / (R__m3_Pa_K_1_mol_1 * T_std__K)    # molar quantity of an ideal gas under standard conditions [mol⋅m^-3] 
        co2_ext__ppm = 415                                            # Yearly average CO₂ concentration in Europe 
        co2_o2__mol0 = 0.894                                          # ratio: moles of CO₂ produced by (aerobic) human metabolism per mole of O₂ consumed 

        # National averages
        weight__kg = 77.5                                             # average weight of Dutch adult [kg]
        umol_s_1_p_1_MET_1 = (mL_min_1_kg_1_p_1_MET_1
                           * weight__kg
                           / s_min_1 
                           * (million * std__mol_m_3 / mL_m_3)
                           )                                          # molar quantity of O₂inhaled by an average Dutch adult at 1 MET [µmol/(p⋅s)]
        co2__umol_p_1_s_1 = (co2_o2__mol0
                             * desk_work__MET
                             * umol_s_1_p_1_MET_1
                            )                                         # molar quantity of CO₂ exhaled by Dutch desk worker doing desk work [µmol/(p⋅s)]
        # Room averages
        wind__m_s_1 = 3.0                                             # assumed wind speed for virtual rooms that causes infiltration
        
        # create empty dataframe for results of all homes
        df_results = pd.DataFrame()
        
        ids = df_data_ids.index.unique('id').dropna()
        logging.info('ids to analyze: ', list(ids.values))

        for id in tqdm(ids):
            df_learn = df_data_ids.loc[id]
            step__s = ((df_learn.index.max() - df_learn.index.min()).total_seconds()
                      /
                      (len(df_learn)-1)
                     )
            duration__s = step__s * len(df_learn)
            
            # Virtual room constants 
            room__m3 = id % 1e3
            vent_min__m3_h_1 = (id % 1e6) // 1e3
            vent_max__m3_h_1 = id // 1e6
            vent_max__m3_s_1 = vent_max__m3_h_1 / s_h_1
            
            ##################################################################################################################
            # Gekko Model - Initialize
            m = GEKKO(remote = False)
            m.time = np.arange(0, duration__s, step__s)


            # GEKKO Manipulated Variables: measured values
            occupancy__p = m.MV(value = df_learn.occupancy__p.values)
            occupancy__p.STATUS = 0; occupancy__p.FSTATUS = 1

            valve_frac__0 = m.MV(value = df_learn.valve_frac__0.values)
            valve_frac__0.STATUS = 0; valve_frac__0.FSTATUS = 1


            # GEKKO Fixed Variable  model parameters
            infilt__m2 = m.FV(value = 0.001, lb = 0)
            infilt__m2.STATUS = 1; infilt__m2.FSTATUS = 0

            # GEKKO Control Varibale (predicted variable)
            co2__ppm = m.CV(value = df_learn.co2__ppm.values) #[ppm]
            co2__ppm.STATUS = 1; co2__ppm.FSTATUS = 1

            # GEKKO - Equations
            co2_loss__ppm_s_1 = m.Intermediate((co2__ppm - co2_ext__ppm) * (vent_max__m3_s_1 * valve_frac__0 + wind__m_s_1 * infilt__m2) / room__m3)
            co2_gain_mol0_s_1 = m.Intermediate(occupancy__p * co2__mol0_p_1_s_1 / (room__m3 * room__mol_m_3))
            co2_gain__ppm_s_1 = m.Intermediate(co2_gain_mol0_s_1 * million)
            m.Equation(co2__ppm.dt() == co2_gain__ppm_s_1 - co2_loss__ppm_s_1)


            # GEKKO - Solver setting
            m.options.IMODE = 5
            m.options.EV_TYPE = ev_type
            m.options.NODES = 2
            m.solve(disp = False)

            df_data_ids.loc[id, 'co2_sim__ppm'] = co2__ppm

            logging.info(f'room {id}: effective infiltration area = {infilt__m2.value[0] * 1e4: .2f} [cm2]')

            mae__ppm = (abs(df_data_ids.loc[id].co2_sim__ppm - df_data_ids.loc[id].co2__ppm)).mean()
            rmse__ppm = ((df_data_ids.loc[id].co2_sim__ppm - df_data_ids.loc[id].co2__ppm)**2).mean()**0.5

            # Create a results row and add to results dataframe
            df_results = pd.concat(
                [
                    df_results,
                    pd.DataFrame(
                        {
                            'id': [id],
                            'duration__s': [duration__s],
                            'EV_TYPE': [m.options.EV_TYPE],
                            'vent_min__m3_h_1': [vent_min__m3_h_1],
                            'vent_max__m3_h_1': [vent_max__m3_h_1],
                            'actual_room__m3': [room__m3],
                            'actual_infilt__cm2': [vent_min__m3_h_1 / (s_h_1 * wind__m_s_1) * 1e4],
                            'infilt__cm2': [infilt__m2.value[0] * 1e4],
                            'mae__ppm': [mae__ppm],
                            'rmse__ppm': [rmse__ppm]
                        }
                    )
                ]
            )
            
            m.cleanup()
            
            ##################################################################################################################




        return df_results.set_index('id'), df_data_ids