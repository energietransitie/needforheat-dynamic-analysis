from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from gekko import GEKKO
from tqdm.notebook import tqdm

from filewriter import ExcelWriter as ex
import numbers
import logging

class Learner():
    
    @staticmethod
    def learn_home_parameter_moving_horizon(df_data_homes:pd.DataFrame, 
                                            n_std:int, up_intv:str, gap_n_intv:int, int_intv:str, 
                                            moving_horizon_duration_d=7, 
                                            req_col:list = [], sanity_threshold_timedelta:timedelta=timedelta(hours=24),
                                            hint_A_m2=None, hint_eta_sup_CH_frac=0.97, ev_type=2) -> pd.DataFrame:
        """
        Input:  
        - a dataframe with a timezone-aware datetime index and measurement values: with at least the following columns
            [
                'home_id', 
                'T_out_e_avg_C', 'irradiation_hor_avg_W_p_m2',
                'T_in_avg_C', 'gas_sup_avg_W', 'e_remaining_heat_avg_W', 
                'ev_type'
            ]
        and optionally,
        - the number of days to use as moving horizon duration in the analysis
        - a 'req_col' list: a list of coumn names: if any of the values in this column are NaN, the interval is not considered 'sane'
        - a sanity_theshold_timedelta: only the longest streaks with sane data longer than this is considered for analysis during each moving_horizon_duration
        
        Output:
        - a dataframe with per home_id and per moving horizon_duration the learned parameters
        - a dataframe with additional columns:
          - 'sanity': none of the required columns in the req_col list are NaN
          - 'interval_s': interval time  in the 
          - 'T_in_sim_avg_C' best fiting temperature seris for each moving_horizon
        """
        
        if not ((hint_A_m2 is None) or isinstance(hint_A_m2, numbers.Number)):
            raise TypeError('hint_A_m2 parameter must be a number or None')
        # get starting time of this analysis; to be used as prefix for filenames
        filename_prefix = datetime.now().astimezone(pytz.timezone('Europe/Amsterdam')).replace(microsecond=0).isoformat().replace(":","")

        # set default values for parameters not set
        
        if (moving_horizon_duration_d is None):
            moving_horizon_duration_d = 7

        homes_to_analyze= df_data_homes.index.unique('home_id').dropna()
        start_analysis_period = df_data_homes.index.unique('timestamp').min().to_pydatetime()
        end_analysis_period = df_data_homes.index.unique('timestamp').max().to_pydatetime()
        
        daterange_frequency = str(moving_horizon_duration_d) + 'D'

        print('Homes to analyse: ', homes_to_analyze)
        print('Start of analyses: ', start_analysis_period)
        print('End of analyses: ', end_analysis_period)
        print('Moving horizon: ', daterange_frequency)
        print('#standard deviations for outlier removal: ', n_std)
        print('Upsampling_interval: ', up_intv)
        print('#upsampling intervals bridged during interpolation (max): ', gap_n_intv)
        print('Interpolation interval: ', int_intv)
        print('Hint for effective window are A [m^2]: ', hint_A_m2)
        print('Hint for superior heating efficiency eta [-]: ', hint_eta_sup_CH_frac)
        print('EV_TYPE: ', ev_type)

        # perform sanity check; not any required column may be missing a value
        if (req_col == []):
            df_data_homes.loc[:,'sanity'] = True
        else:
            df_data_homes.loc[:,'sanity'] = ~np.isnan(df_data_homes[req_col]).any(axis="columns")
            
        total_measurement_time = timedelta(seconds = int(df_data_homes.interval_s.sum()))
        print('Total measurement time: ', total_measurement_time)
        sane_fraction = df_data_homes.sanity.astype(float).mean()
        print('Sane fraction measurement time: {:.2f}'.format(sane_fraction))
        print('Sane  measurement time: ', total_measurement_time * sane_fraction)

        #see more statisctics
        df_data_homes.describe(include='all')
                
        # Conversion factor s_p_h [s/h]  = 60 [min/h] * 60 [s/min] 
        s_p_h = (60 * 60) 

        # Conversion factor J_p_kWh [J/kWh]  = 1000 [Wh/kWh] * s_p_h [s/h] * 1 [J/Ws]
        J_p_kWh = 1000 * s_p_h

        # Conversion factor s_p_d [s/d]  = 24 [h/d] * s_p_h [s/h] 
        s_p_d = (24 * s_p_h) 
        # Conversion factor s_p_a [s/a]  = 365.25 [d/a] * s_p_d [s/d] 
        s_p_a = (365.25 * s_p_d) 

        # Conversion factor h_sup_J_p_m3 superior calorific value of natural gas from the Groningen field = 35,170,000.00 [J/m^3]
        h_sup_J_p_m3 = value=35170000.0


        # National averages

        avg_persons = 2.2  # average number of people in Dutch household
        Q_gain_int_W_p_person = 61  # average heat gain per average person with average behaviour asnd occupancy
        Q_gain_int_occup_avg_W = Q_gain_int_W_p_person * avg_persons

        # currently, we use 339 [m^3/a] national average gas usage per year for cooking and DHW, i.e. not for CH
        gas_no_CH_avg_m3_p_s = (339.0 / s_p_a)  

        # create empty dataframe for results of all homes
        df_results = pd.DataFrame()

        # create empty dataframe for temperature simultion results of all homes
        df_results_allhomes_allweeks_tempsim = pd.DataFrame()
           
        # iterate over homes
        for home_id in tqdm(homes_to_analyze):
            
            # create empty dataframe for results of a home
            df_results_home = pd.DataFrame()

            # create empty dataframe for temperature simulation results of a single home
            df_results_home_allweeks_tempsim = pd.DataFrame()
            
            # create empty dataframe for temperature simulation results of a single week of a single home
            df_results_homeweek_tempsim  = pd.DataFrame()

            logging.info('Home pseudonym: ', home_id)

            df_data_one_home = df_data_homes.loc[home_id].copy()
            df_data_one_home['streak_id'] = np.nan
            df_data_one_home['streak_cumulative_duration_s'] = np.nan
                        
            # split gas over CH and no_CH per home based on the entire period 
            
            # in a future version we intend to use a value specific per home based on average usage of natural gas in the summer months (June - August) 
           
            gas_sup_no_CH_avg_W = gas_no_CH_avg_m3_p_s * h_sup_J_p_m3 

            logging.info('gas_no_CH_avg_m3_p_s: {:.5E}'.format(gas_no_CH_avg_m3_p_s))
            logging.info('gas_sup_no_CH_avg_W: ', gas_sup_no_CH_avg_W)

            # using this average, distribute gas usage over central heating (CH) versus no Central Heating (no_CH)
            df_data_one_home['gas_sup_no_CH_avg_W'] = gas_sup_no_CH_avg_W
            df_data_one_home['gas_sup_CH_avg_W'] = df_data_one_home['gas_sup_avg_W'] - gas_sup_no_CH_avg_W

            # Avoid negative values for heating; simple fix: negative value with zero in gas_sup_CH_avg_W array
            df_data_one_home.loc[df_data_one_home.gas_sup_CH_avg_W < 0, 'gas_sup_CH_avg_W'] = 0

            # Compensate by scaling down gas_sup_CH_avg_W 
            gas_sup_home_avg_W = df_data_one_home['gas_sup_avg_W'].mean()
            uncorrected_gas_CH_sup_home_avg_W = df_data_one_home['gas_sup_CH_avg_W'].mean()
            scaling_factor =   (gas_sup_home_avg_W - gas_sup_no_CH_avg_W) / uncorrected_gas_CH_sup_home_avg_W  
            df_data_one_home['gas_sup_CH_avg_W'] = df_data_one_home['gas_sup_CH_avg_W'] * scaling_factor
            corrected_gas_CH_sup_home_avg_W = df_data_one_home['gas_sup_CH_avg_W'].mean()

            logging.info('home_id: ', home_id)
            logging.info('gas_sup_home_avg_W: ', gas_sup_home_avg_W)
            logging.info('uncorrected_gas_CH_sup_home_avg_W: ', uncorrected_gas_CH_sup_home_avg_W)
            logging.info('scaling_factor: ', scaling_factor)
            logging.info('corrected_gas_CH_sup_home_avg_W: ', corrected_gas_CH_sup_home_avg_W)
            logging.info('gas_sup_no_CH_avg_W + corrected_gas_CH_sup_home_avg_W: ', gas_sup_no_CH_avg_W + corrected_gas_CH_sup_home_avg_W)
            
            moving_horizon_starts = pd.date_range(start=start_analysis_period, end=end_analysis_period, inclusive='both', freq=daterange_frequency)

            moving_horizon_iterator = tqdm(moving_horizon_starts)

            # iterate over horizons
            for moving_horizon_start in moving_horizon_iterator:

                moving_horizon_end = min(end_analysis_period, moving_horizon_start + timedelta(days=moving_horizon_duration_d))

                if (moving_horizon_end < end_analysis_period):
                    df_moving_horizon = df_data_one_home[moving_horizon_start:moving_horizon_end].iloc[:-1]
                else:
                    df_moving_horizon = df_data_one_home[moving_horizon_start:end_analysis_period]
                    
                moving_horizon_end = df_moving_horizon.index.max()

                logging.info('Start datetime: ', moving_horizon_start)
                logging.info('End datetime: ', moving_horizon_end)
                
                #first check whether there is even a single sane value
                if len(df_moving_horizon.query('sanity == True')) == 0:
                    logging.info(f'For home {home_id} there is no sane data in the period from {moving_horizon_start} to {moving_horizon_start}; skipping...')
                    continue                       
                
                # restrict the df_moving_horizon to the longest streak of sane data
                ## give each streak a separate id
                df_moving_horizon.streak_id = df_moving_horizon.sanity.ne(df_moving_horizon.sanity.shift()).cumsum()
                df_moving_horizon.streak_cumulative_duration_s = df_moving_horizon.groupby('streak_id').interval_s.cumsum()
                ## make sure streaks with insane values are not considered
                df_moving_horizon.loc[df_moving_horizon.sanity == False, 'streak_cumulative_duration_s'] = np.nan 
                ## get the longest streak: the part of the dataframe where the streak_id matches the (first) streak_id that has the longest cumulative duration
                df_moving_horizon = df_moving_horizon.query('streak_id == ' + str(df_moving_horizon.loc[df_moving_horizon.streak_cumulative_duration_s.idxmax()].streak_id))

                logging.info('Start datetime longest sane streak: ', moving_horizon_start)
                logging.info('End datetime longest sane streak: ', moving_horizon_end)

                # then check whether enough data, if not then skip this homeweek, move on to next
                if ((moving_horizon_end - moving_horizon_start) < sanity_threshold_timedelta):
                    logging.info(f'For home {home_id} the longest streak of sane data is less than {sanity_threshold_timedelta} in the period from {moving_horizon_start} to {moving_horizon_start}; skipping...')
                    continue
                


                
                # T_set_first_C_array = df_moving_horizon['T_set_first_C'].to_numpy()
                T_in_avg_C_array = df_moving_horizon['T_in_avg_C'].to_numpy()

                step_s = df_moving_horizon['interval_s'].mean()
                number_of_timesteps = len(T_in_avg_C_array)
                duration_s = number_of_timesteps * step_s

                # load data from dataframe into np.arrays
                # logging.info(df_moving_horizon)


                T_out_e_avg_C_array = df_moving_horizon['T_out_e_avg_C'].to_numpy()
                irradiation_hor_avg_W_p_m2_array = df_moving_horizon['irradiation_hor_avg_W_p_m2'].to_numpy()
                e_remaining_heat_avg_W_array = df_moving_horizon['e_remaining_heat_avg_W'].to_numpy()
                gas_sup_no_CH_avg_W_array = df_moving_horizon['gas_sup_no_CH_avg_W'].to_numpy()
                gas_sup_CH_avg_W_array = df_moving_horizon['gas_sup_CH_avg_W'].to_numpy()
               
                # print length of arrays and check uquality

                # logging.info('#T_in_avg_C_array', len(T_in_avg_C_array))
                # logging.info('#T_out_e_avg_C_array', len(T_out_e_avg_C_array))
                # logging.info('#irradiation_hor_avg_W_p_m2_array', len(irradiation_hor_avg_W_p_m2_array))
                # logging.info('#gas_sup_no_CH_avg_W_array', len(gas_sup_no_CH_avg_W_array))
                # logging.info('#gas_sup_CH_avg_W_array', len(gas_sup_CH_avg_W_array))

                # check for equal length

                # logging.info(len(T_in_avg_C_array) == len(irradiation_hor_avg_W_p_m2_array) 
                #       == len(T_out_e_avg_C_array) == len(gas_sup_no_CH_avg_W_array) 
                #       == len(gas_sup_CH_avg_W_array) == len(e_remaining_heat_avg_W_array))
            
            
                if (hint_A_m2 is None):
                    logging.info('home {0} from {1} to {2} only run learning loop once with A as learnable model parameter'
                          .format(home_id,moving_horizon_start,moving_horizon_end))
                    innerloop=[np.NaN]
                else:
                    logging.info('home {0} from {1} to {2} run learning loop twice: once with A={3} and once as as learnable model parameter'
                          .format(home_id,moving_horizon_start,moving_horizon_end, hint_A_m2))
                    innerloop=[hint_A_m2, np.NaN]

                for iterator_A_m2 in innerloop:

                    try:


                        ########################################################################################################################
                        # Gekko Model - Initialize
                        ########################################################################################################################
 
                        # initialize gekko
                        m = GEKKO(remote=False)
                        m.time = np.arange(0, duration_s, step_s)








                        ########################################################################################################################
                        # Gekko Model - Model Parameters
                        ########################################################################################################################
                        
                        # Model parameter: H [W/K]: specific heat loss
                        H_W_p_K = m.FV(value=300.0, lb=0, ub=1000)
                        H_W_p_K.STATUS = 1; H_W_p_K.FSTATUS = 0
                        
                        # Model parameter: tau [s]: effective thermal inertia
                        tau_s = m.FV(value=(100 * s_p_h), lb=(10 * s_p_h), ub=(1000 * s_p_h))
                        tau_s.STATUS = 1; tau_s.FSTATUS = 0

                        ########################################################################################################################
                        #                                               Gekko - Equations
                        ########################################################################################################################

                        
                        ########################################################################################################################
                        # Equation - Q_gain_CH_avg_W: heat gain from natural gas used for central heating
                        ########################################################################################################################

                        # eta_sup_CH_frac [-]: upper heating efficiency of the central heating system

                        # This section of code (later) when  eta_sup_CH_frac is estimated using the model
                        # eta_sup_CH_frac = m.FV(value=0.8, lb=0, ub=1.0)
                        # eta_sup_CH_frac.STATUS = 1
                        # eta_sup_CH_frac.FSTATUS = 0
                        ## eta_sup_CH_frac.DMAX = 0.25

                        # Fix eta_sup_CH_frac when a hint is given 
                        eta_sup_CH_frac = m.Param(value=hint_eta_sup_CH_frac)
                        
                        gas_sup_CH_avg_W = m.MV(value=gas_sup_CH_avg_W_array)
                        gas_sup_CH_avg_W.STATUS = 0; gas_sup_CH_avg_W.FSTATUS = 1

                        Q_gain_gas_CH_avg_W = m.Intermediate(gas_sup_CH_avg_W * eta_sup_CH_frac)
                    
                    
                        ########################################################################################################################
                        # Equation - Q_gain_no_CH_avg_W: heat gain from natural gas used for central heating
                        ########################################################################################################################

                        # eta_sup_no_CH_frac [-]: superior efficiency of heating the home indirectly using gas, 
                        # for other primary purposes than heating the home, # eq48, PPT slide 24
                        eta_sup_no_CH_frac = m.Param(value=0.34)
                        
                        gas_sup_no_CH_avg_W = m.MV(value=gas_sup_no_CH_avg_W_array)
                        gas_sup_no_CH_avg_W.STATUS = 0; gas_sup_no_CH_avg_W.FSTATUS = 1
                        
                        Q_gain_gas_no_CH_avg_W = m.Intermediate(gas_sup_no_CH_avg_W * eta_sup_no_CH_frac)

                        ########################################################################################################################
                        # Equation - Q_gain_int_avg_W: Heat gain from internal sources
                        ########################################################################################################################
                        # Manipulated Variable: e_remaining_heat_avg_W: internal heat gain from internally used electricity
                        e_remaining_heat_avg_W = m.MV(value=e_remaining_heat_avg_W_array)
                        e_remaining_heat_avg_W.STATUS = 0; e_remaining_heat_avg_W.FSTATUS = 1

                        Q_gain_int_avg_W = m.Intermediate(e_remaining_heat_avg_W + Q_gain_int_occup_avg_W + Q_gain_gas_no_CH_avg_W)

                        ########################################################################################################################
                        # Equation - Q_gain_sol_avg_W: : heat gain from solar irradiation
                        ########################################################################################################################

                        # Model parameter/fixed value: A [m^2]: Effective area of the solar aperture
                        if np.isnan(iterator_A_m2):
                            A_m2 = m.FV(value=5, lb=1, ub=100); A_m2.STATUS = 1; A_m2.FSTATUS = 0
                        else:
                            A_m2 = m.Param(value=iterator_A_m2)

                        irradiation_hor_avg_W_p_m2 = m.MV(value=irradiation_hor_avg_W_p_m2_array)
                        irradiation_hor_avg_W_p_m2.STATUS = 0; irradiation_hor_avg_W_p_m2.FSTATUS = 1
                        
                        Q_gain_sol_avg_W = m.Intermediate(irradiation_hor_avg_W_p_m2 * A_m2)
                        
                        ########################################################################################################################
                        # Manipulated Variable (MV): T_out_e_avg_C [°C]: effective outdoor temperature
                        ########################################################################################################################
                        T_out_e_avg_C = m.MV(value=T_out_e_avg_C_array)
                        T_out_e_avg_C.STATUS = 0; T_out_e_avg_C.FSTATUS = 1

                        ########################################################################################################################
                        # Control Variable T_in_avg_C: Indoor temperature
                        ########################################################################################################################
                        T_in_avg_C = m.CV(value=T_in_avg_C_array)
                        T_in_avg_C.STATUS = 1; T_in_avg_C.FSTATUS = 1
                        # T_in_avg_C.MEAS_GAP= 0.25
                        

                        ########################################################################################################################
                        # Main Equations & Solver
                        ########################################################################################################################
                        Q_gain_W = m.Intermediate(Q_gain_gas_CH_avg_W + Q_gain_sol_avg_W + Q_gain_int_avg_W)
                        Q_loss_W = m.Intermediate(H_W_p_K * (T_in_avg_C - T_out_e_avg_C)) 
                        C_J_p_K  = m.Intermediate(H_W_p_K * tau_s) 
                        m.Equation(T_in_avg_C.dt() == ((Q_gain_W - Q_loss_W) / C_J_p_K))
                        
                        m.options.IMODE = 5
                        m.options.EV_TYPE = ev_type # specific objective function (L1-norm vs L2-norm)
                        m.solve(False)      

                        ########################################################################################################################
                        #                                                       Result
                        ########################################################################################################################

                        #add simulated indoor temperature for optimized solution to sim_T_in_avg_C column
                        # logging.info(T_in_avg_C)
                        # logging.info(len(list(T_in_avg_C.value)))
                        # logging.info(list(T_in_avg_C.value))
                        
                        # create a deep copy
                        df_results_homeweek_tempsim = df_moving_horizon.copy(deep=True)

                        df_results_homeweek_tempsim['T_in_sim_avg_C'] = list(T_in_avg_C.value)
                        df_results_homeweek_tempsim['A_value_is_fixed'] = not np.isnan(iterator_A_m2)
                        df_results_homeweek_tempsim['home_solar_irradiation_avg_W'] =  df_results_homeweek_tempsim['irradiation_hor_avg_W_p_m2'] * A_m2.value[0]
                        
                        # logging.info(df_results_homeweek_tempsim) 

                        # filename_prefix = datetime.now().astimezone(pytz.timezone('Europe/Amsterdam')).replace(microsecond=0).isoformat().replace(":","")
                        # ex.write(df_results_homeweek_tempsim, str('{0}-simdata_home-{1}-{2}-{3}.xlsx'.format(home_id,
                        #                                                                                      filename_prefix, 
                        #                                                                                      moving_horizon_start.isoformat(),
                        #                                                                                      moving_horizon_end.isoformat())))
                        
                        # error_K = (m.options.OBJFCNVAL ** (1/m.options.EV_TYPE))/duration_s
                        mae_K = (abs(df_results_homeweek_tempsim['T_in_sim_avg_C'] - df_results_homeweek_tempsim['T_in_avg_C'])).mean()
                        rmse_K = ((df_results_homeweek_tempsim['T_in_sim_avg_C'] - df_results_homeweek_tempsim['T_in_avg_C'])**2).mean()**0.5

                        logging.info('duration [s]: ', duration_s)
                        logging.info('OBJFCNVAL: ', m.options.OBJFCNVAL)
                        logging.info('EV_TYPE: ', m.options.EV_TYPE)
                        logging.info('H [W/K]: ', round(H_W_p_K.value[0], 4))
                        logging.info('tau [h]: ', round(tau_s.value[0] / s_p_h, 2))
                        logging.info('A [m^2]: ', round(A_m2.value[0], 2))
                        logging.info('A value fixed: ', not np.isnan(iterator_A_m2))
                        logging.info('eta_sup [-]: ', round(eta_sup_CH_frac.value[0], 2))
                        logging.info('eta_sup value fixed: ', True)
                        logging.info('MAE: ', mae_K)
                        logging.info('RMSE: ', rmse_K)
                        

                        # Create a results row
                        df_result_row = pd.DataFrame({
                            'start_horizon': [moving_horizon_start],
                            'end_horizon': [moving_horizon_end],
                            'pseudonym': [home_id],
                            'n_std_outlier_removal': [n_std], 
                            'upsampling_interval': [up_intv], 
                            'n_intv_gap_bridge_upper_bound': [gap_n_intv], 
                            'interpolation_interval': [int_intv],
                            'duration_s': [duration_s],
                            'OBJFCNVAL': [m.options.OBJFCNVAL],
                            'EV_TYPE': [m.options.EV_TYPE],
                            'H_W_p_K': [H_W_p_K.value[0]],
                            'tau_h': [tau_s.value[0] / s_p_h],
                            'C_Wh_p_K':[H_W_p_K.value[0] * tau_s.value[0] / s_p_h],
                            'A_m^2': [A_m2.value[0]],
                            'A_m^2_fixed': [not np.isnan(iterator_A_m2)],
                            'eta_sup': [eta_sup_CH_frac.value[0]],
                            'eta_sup_fixed': [True],
                            'MAE_K': [mae_K],
                            'RMSE_K': [rmse_K]})
                        df_result_row.set_index(['start_horizon'], inplace=True)
                        
                        # add week to home results dataframe

                    except KeyboardInterrupt:    
                        logging.error(str('KeyboardInterrupt; home analysis {0} not complete; saving results so far then will exit...'.format(home_id)))

                        # do NOT write an empty line for this iteration, to indicate it is not fully processed and we don't know 
                        # ex.write(df_results_home, str(filename_prefix+'-results-aborted-{0}.xlsx'.format(home_id)))

                        # but DO include the incomplete home results in the final export
                        df_results = pd.concat([df_results, df_results_home])
                        # ex.write(df_results_home, str(filename_prefix+'-results-aborted.xlsx'.format(home_id)))

                        # only then exit the function and return to caller
                        return  df_results, df_results_allhomes_allweeks_tempsim

                    except Exception as e:
                        # do write an empty line for this iteration, to indicate it is fully processed 
                        # and to indicate that we do know know  GEKKO could not learn parameters for this moving horizon for this home 
                        
                        logging.error(str('Exception {0} for home {1} in period from {2} to {3}; skipping...'
                                  .format(e, home_id,moving_horizon_start,moving_horizon_end)))
                        df_result_row = pd.DataFrame({
                            'start_horizon': [moving_horizon_start],
                            'end_horizon': [moving_horizon_end],
                            'pseudonym': [home_id],
                            'n_std_outlier_removal': [n_std], 
                            'upsampling_interval': [up_intv], 
                            'n_intv_gap_bridge_upper_bound': [gap_n_intv], 
                            'interpolation_interval': [int_intv],
                            'duration_s': [np.nan],
                            'OBJFCNVAL': [np.nan],
                            'EV_TYPE': [np.nan],
                            'H_W_p_K': [np.nan],
                            'tau_h': [np.nan],
                            'C_Wh_p_K':[np.nan],
                            'A_m^2': [np.nan],
                            'A_m^2_fixed': [not (np.isnan(iterator_A_m2))],
                            'eta_sup': [np.nan],
                            'eta_sup_fixed': [True],
                            'MAE_K': [np.nan],
                            'RMSE_K': [np.nan]})
                        df_result_row.set_index(['start_horizon'], inplace=True)
                        pass

                    #after a single innerloop for A fixed or learnable
                    try:
                        df_results_home = pd.concat([df_results_home, df_result_row])
                        
                        if df_results_homeweek_tempsim is not None and len(df_results_homeweek_tempsim.columns)>0:
                            df_results_home_allweeks_tempsim = pd.concat([df_results_home_allweeks_tempsim, df_results_homeweek_tempsim])
                            df_results_home_allweeks_tempsim.describe(include='all')
                        
                    except KeyboardInterrupt:    
                        logging.error(str('KeyboardInterrupt; home analysis {0} not complete; saving results so far then will exit...'.format(home_id)))

                        # do write full line for this iteration, to indicate it is fully processed and we do know 
                        df_results_home = pd.concat([df_results_home, df_result_row])
                        # ex.write(df_results_home, str(filename_prefix+'-results-aborted-{0}.xlsx'.format(home_id)))

                        # and include the incomplete home results in the final export
                        df_results = pd.concat([df_results, df_results_home])
                        # ex.write(df_results_home, str(filename_prefix+'-results-aborted.xlsx'.format(home_id)))

                        # only then exit the function and return to caller
                        return  df_results, df_results_allhomes_allweeks_tempsim

                #after a single innerloop for A fixed or learnable
                logging.info(str('Analysis of all moving horizons for a single inner loop for home {0} complete.'.format(home_id)))

            #after all moving horizons of a single home; after a single innerloop for A fixed or learnable
            logging.info(str('Analysis of all moving horizons for home {0} complete.'.format(home_id)))
            try:
                df_results = pd.concat([df_results, df_results_home])
                # ex.write(df_results_home, str(filename_prefix+'-results-{0}.xlsx'.format(home_id)))
                
                # label each line in the temperature simulation result dataframa with homepseudonym
                df_results_home_allweeks_tempsim.insert(loc=0, column='home_id', value=home_id)
                #and add to result dataframe of all homes
                df_results_allhomes_allweeks_tempsim = pd.concat([df_results_allhomes_allweeks_tempsim, df_results_home_allweeks_tempsim])
                
                
            except KeyboardInterrupt:    
                logging.error(str('KeyboardInterrupt; home analysis {0} complete; saving results so far then will exit...'.format(home_id)))
                # ex.write(df_results, (filename_prefix+'-results-aborted.xlsx'))
                return  df_results, df_results_allhomes_allweeks_tempsim


        # reset after all homes
        df_results = df_results.reset_index().rename(columns = {'pseudonym':'home_id'}).set_index(['home_id', 'start_horizon'])

        # cols = list(df_results_allhomes_allweeks_tempsim.columns)
        
        df_results_allhomes_allweeks_tempsim = df_results_allhomes_allweeks_tempsim.reset_index().rename(columns = {'index':'timestamp'}).set_index(['home_id', 'timestamp'])
        
        # print('DONE: Analysis of all homes complete; writing files.')
        
        # print(df_results_allhomes_allweeks_tempsim.describe(include='all'))
        
        # try:
        #     ex.write(df_results, (filename_prefix+'-results.xlsx'))
        # except KeyboardInterrupt:    
        #     logging.error(str('KeyboardInterrupt; all home analyses complete; will continue saving results and then exit...'.format(home_id)))
        #     # ex.write(df_results, (filename_prefix+'-results.xlsx'))
        #     return df_results, df_results_allhomes_allweeks_tempsim

        #return simulation results via df_data_homes parameter
        df_data_homes = df_results_allhomes_allweeks_tempsim
        
        filename_prefix = datetime.now().astimezone(pytz.timezone('Europe/Amsterdam')).replace(microsecond=0).isoformat().replace(":","")
        # ex.write(df_results_allhomes_allweeks_tempsim, str('{0}-data_homes_tempsim.xlsx'.format(filename_prefix)))
        # print('DONE: all result files written.')
    
        return  df_results, df_results_allhomes_allweeks_tempsim
    
    
    @staticmethod
    def learn_room_parameter(df_data_rooms:pd.DataFrame, ev_type=2) -> pd.DataFrame:
        """
        Input:  
        - a dataframe with a MultiIndex ['home_id', 'timestamp]; timestamp is timezone-aware
        - columns:
          - 'occupancy_p': average number of people present in the room,
          - 'co2_ppm': average CO₂-concentration in the room,
          - 'valve_frac_0' opening fraction of the ventilation valve 
        and optionally,
        - 'ev_type': type 2 is usually recommended, since this is typically more thatn 50 times faster
        
        Output:
        - a dataframe with per room_id the learned parameters
        - a dataframe with additional column(s):
          - 'co2_sim_ppm' best fiting temperature series for room_id
        """
        

        # Conversion factors
        s_min_1 = 60
        min_h_1 = 60
        s_h_1 = s_min_1 * min_h_1
        mL_m_3 = 1e3 * 1e3
        million = 1e6

        # Constants
        MET_mL_min_1_kg_1_p_1 = 3.5                               # Metabolic Equivalent of Task, per kg body weight
        desk_work = 1.5                                           # MET factor for desk work
        P_std_Pa = 101325                                         # standard gas pressure
        R_m3_Pa_K_1_mol_1 = 8.3145                                # gas constant
        T_room_C = 20.0                                           # standard room temperature
        T_std_C = 0.0                                             # standard gas temperature
        T_zero_K = 273.15                                         # 0 ˚C
        T_std_K = T_zero_K + T_std_C                              # standard gas temperature
        T_room_K = T_zero_K + T_room_C                            # standard room temperature

        # Approximations
        room_mol_m_3 = P_std_Pa / (R_m3_Pa_K_1_mol_1 * T_room_K)  # gas molecules in 1 m3 under room conditions 
        std_mol_m_3 = P_std_Pa / (R_m3_Pa_K_1_mol_1 * T_std_K)    # gas molecules in 1 m3 under standard conditions 
        co2_ext_ppm = 415                                         # Yearly average CO₂ concentration in Europe 
        
        # National averages
        W_avg_kg = 77.5                                           # average weight of Dutch adult
        MET_m3_s_1_p_1 = MET_mL_min_1_kg_1_p_1 * W_avg_kg / (s_min_1 * mL_m_3)
        
        MET_mol_s_1_p_1 = MET_m3_s_1_p_1 * std_mol_m_3            # Metabolic Equivalent of Task, per person
        co2_p_o2 = 0.894                                          # fraction molecules CO₂ exhaled versus molecule O₂ inhaled
        co2_mol0_p_1_s_1 = co2_p_o2 * desk_work * MET_mol_s_1_p_1 # CO₂ raise by Dutch desk worker [mol/mol]

        # Room averages
        wind_m_s_1 = 3.0                                          # assumed wind speed for virtual rooms that causes infiltration
        
        # create empty dataframe for results of all homes
        df_results = pd.DataFrame()
        
        rooms = df_data_rooms.index.unique('room_id').dropna()
        logging.info('Rooms to analyze: ', list(rooms.values))

        for room_id in tqdm(rooms):
            df = df_data_rooms.loc[room_id]
            step_s = ((df.index.max() - df.index.min()).total_seconds()
                      /
                      (len(df)-1)
                     )
            duration_s = step_s * len(df)
            
            # Virtual room constants 
            room_m3 = room_id % 1e3
            vent_min_m3_h_1 = (room_id % 1e6) // 1e3
            vent_max_m3_h_1 = room_id // 1e6
            vent_max_m3_s_1 = vent_max_m3_h_1 / s_h_1
            
            ##################################################################################################################
            # Gekko Model - Initialize
            m = GEKKO(remote = False)
            m.time = np.arange(0, duration_s, step_s)


            # GEKKO Manipulated Variables: measured values
            occupancy_p = m.MV(value = df.occupancy_p.values)
            occupancy_p.STATUS = 0; occupancy_p.FSTATUS = 1

            valve_frac_0 = m.MV(value = df.valve_frac_0.values)
            valve_frac_0.STATUS = 0; valve_frac_0.FSTATUS = 1


            # GEKKO Fixed Variable  model parameters
            infilt_m2 = m.FV(value = 0.001, lb = 0)
            infilt_m2.STATUS = 1; infilt_m2.FSTATUS = 0

            # GEKKO Control Varibale (predicted variable)
            co2_ppm = m.CV(value = df.co2_ppm.values) #[ppm]
            co2_ppm.STATUS = 1; co2_ppm.FSTATUS = 1

            # GEKKO - Equations
            co2_loss_ppm_s_1 = m.Intermediate((co2_ppm - co2_ext_ppm) * (vent_max_m3_s_1 * valve_frac_0 + wind_m_s_1 * infilt_m2) / room_m3)
            co2_gain_mol0_s_1 = m.Intermediate(occupancy_p * co2_mol0_p_1_s_1 / (room_m3 * room_mol_m_3))
            co2_gain_ppm_s_1 = m.Intermediate(co2_gain_mol0_s_1 * million)
            m.Equation(co2_ppm.dt() == co2_gain_ppm_s_1 - co2_loss_ppm_s_1)


            # GEKKO - Solver setting
            m.options.IMODE = 5
            m.options.EV_TYPE = ev_type
            m.options.NODES = 2
            m.solve(disp = False)

            df_data_rooms.loc[room_id, 'co2_sim_ppm'] = co2_ppm

            logging.info(f'room {room_id}: effective infiltration area = {infilt_m2.value[0] * 1e4: .2f} [cm^2]')

            mae_ppm = (abs(df_data_rooms.loc[room_id].co2_sim_ppm - df_data_rooms.loc[room_id].co2_ppm)).mean()
            rmse_ppm = ((df_data_rooms.loc[room_id].co2_sim_ppm - df_data_rooms.loc[room_id].co2_ppm)**2).mean()**0.5

            # Create a results row and add to results dataframe
            df_results = pd.concat(
                [
                    df_results,
                    pd.DataFrame(
                        {
                            'room_id': [room_id],
                            'duration_s': [duration_s],
                            'OBJFCNVAL': [m.options.OBJFCNVAL],
                            'EV_TYPE': [m.options.EV_TYPE],
                            'vent_min_m3_h_1': [vent_min_m3_h_1],
                            'vent_max_m3_h_1': [vent_max_m3_h_1],
                            'room_true_m3': [room_m3],
                            'infilt_true_cm2': [vent_min_m3_h_1 / (s_h_1 * wind_m_s_1) * 1e4],
                            'infilt_cm2': [infilt_m2.value[0] * 1e4],
                            'mae_ppm': [mae_ppm],
                            'rmse_ppm': [rmse_ppm]
                        }
                    )
                ]
            )
            
            m.cleanup()
            
            ##################################################################################################################




        return df_results.set_index('room_id'), df_data_rooms