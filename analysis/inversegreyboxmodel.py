from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
from gekko import GEKKO
from tqdm import tqdm_notebook
from filewriter import ExcelWriter as ex
import numbers
import logging

class Learner():
    
    @staticmethod
    def learn_home_parameter_moving_horizon(df_data_homes:pd.DataFrame, 
                                            n_std:int, up_intv:str, gap_n_intv:int, int_intv:str, 
                                            moving_horizon_duration_d=7, sanity_lb:float=0.5,
                                            homes_to_analyze=None, 
                                            start_analysis_period:datetime=None, 
                                            end_analysis_period:datetime=None, 
                                            hint_A_m2=None, hint_eta_sup_CH_frac=0.9) -> pd.DataFrame:
        """
        Input:  
        - a dataframe with a timezone-aware datetime index and measurement values: with at least the following columns
            [
                'home_id', 
                'T_out_e_avg_C', 'irradiation_hor_avg_W_p_m2',
                'T_in_avg_C', 'gas_sup_avg_W', 'e_remaining_heat_avg_W', 
                'interval_s', 'sanity_frac'
            ]
        and optionally,
        - the number of days to use as moving horizon duration in the analysis
        - start datetime for the analysis (defaults to earliest datatime in the index column)
        - end datatime for the analysis (defaults to latest datatime in the index column)
        
        Output:
        - a dataframe with results
        - excel files with intermediate results per home and all homes
        """
        
        if not ((hint_A_m2 is None) or isinstance(hint_A_m2, numbers.Number)):
            raise TypeError('hint_A_m2 parameter must be a number or None')
        # get starting time of this analysis; to be used as prefix for filenames
        filename_prefix = datetime.now().astimezone(pytz.timezone('Europe/Amsterdam')).replace(microsecond=0).isoformat().replace(":","")

        # set default values for parameters not set
        
        if (moving_horizon_duration_d is None):
            moving_horizon_duration_d = 7

        if (homes_to_analyze is None):
            homes_to_analyze= df_data_homes['home_id'].unique()
        if (start_analysis_period is None):
                start_analysis_period = df_data_homes.index.min()
        if (end_analysis_period is None): 
                end_analysis_period = df_data_homes.index.max()

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

        # create empty dataframe for results of all homes
        df_results = pd.DataFrame()

        home_iterator = tqdm_notebook(homes_to_analyze)
           
        # iterate over homes
        for home_id in home_iterator:
            
            # create empty dataframe for results of a home
            df_results_home = pd.DataFrame()

            # create empty dataframe for temperature simultion results of a homes
            df_results_home_tempsim = pd.DataFrame()

            print('Home pseudonym: ', home_id)

            df_data_one_home = df_data_homes.loc[home_id]
            
            
            # split gas over CH and no_CH per home based on the entire period 
            
            # currently, we use 339 [m^3/a] national average gas usage per year for cooking and DHW, i.e. not for CH
            gas_no_CH_avg_m3_p_s = (339.0 / s_p_a)  
            # in a future version we intend to use a value specific per home based on average usage of natural gas in the summer months (June - August) 
           
            gas_sup_no_CH_avg_W = gas_no_CH_avg_m3_p_s * h_sup_J_p_m3 

            logging.info('gas_no_CH_avg_m3_p_s: {:.5E}'.format(gas_no_CH_avg_m3_p_s))
            logging.info('gas_sup_no_CH_avg_W: ', gas_sup_no_CH_avg_W)

            # using this average, distribute gas usage over central heating (CH) versus no Central Heating (no_CH)
            df_data_one_home['gas_sup_no_CH_avg_W'] = gas_sup_no_CH_avg_W
            df_data_one_home['gas_sup_CH_avg_W'] = df_data_one_home['gas_sup_avg_W'] - df_data_one_home['gas_sup_no_CH_avg_W']

            # Avoid negative values for heating; simple fix: negative value with zero in gas_sup_CH_avg_W array
            df_data_one_home.loc[df_data_one_home.gas_sup_CH_avg_W < 0, 'gas_sup_CH_avg_W'] = 0

            # Compensate by scaling down gas_sup_CH_avg_W 
            gas_sup_home_avg_W = df_data_one_home['gas_sup_avg_W'].mean()
            uncorrected_gas_CH_sup_home_avg_W = df_data_one_home['gas_sup_CH_avg_W'].mean()
            scaling_factor =   (gas_sup_home_avg_W - gas_sup_no_CH_avg_W) / uncorrected_gas_CH_sup_home_avg_W  
            df_data_one_home['gas_sup_CH_avg_W'] = df_data_one_home['gas_sup_CH_avg_W'] * scaling_factor
            corrected_gas_CH_sup_home_avg_W = df_data_one_home['gas_sup_CH_avg_W'].mean()

            print('home_id: ', home_id)
            print('gas_sup_home_avg_W: ', gas_sup_home_avg_W)
            print('uncorrected_gas_CH_sup_home_avg_W: ', uncorrected_gas_CH_sup_home_avg_W)
            print('scaling_factor: ', scaling_factor)
            print('corrected_gas_CH_sup_home_avg_W: ', corrected_gas_CH_sup_home_avg_W)
            print('gas_sup_no_CH_avg_W + corrected_gas_CH_sup_home_avg_W: ', gas_sup_no_CH_avg_W + corrected_gas_CH_sup_home_avg_W)
            
            moving_horizon_starts = pd.date_range(start=start_analysis_period, end=end_analysis_period, inclusive='left', freq=daterange_frequency)

            moving_horizon_iterator = tqdm_notebook(moving_horizon_starts)

            # iterate over horizons
            for moving_horizon_start in moving_horizon_iterator:

                moving_horizon_end = min(end_analysis_period, moving_horizon_start + timedelta(days=moving_horizon_duration_d) - timedelta(minutes=1))

                df_moving_horizon = df_data_one_home[moving_horizon_start:moving_horizon_end]
                                  
                moving_horizon_end = df_moving_horizon.index.max()

                print('Start datetime: ', moving_horizon_start)
                print('End datetime: ', moving_horizon_end)

                # first check whether sanity of the data is sufficient, if not then skip this homeweek, move on to next
                sanity_moving_horizon = df_moving_horizon['sanity_frac'].mean()
                if (sanity_moving_horizon < sanity_lb):
                    print(str('Sanity {0:.2f} for home {1} in period from {2} to {3} lower than {4:.2f}; skipping...'
                              .format(sanity_moving_horizon, home_id, moving_horizon_start, moving_horizon_end, sanity_lb)))
                    continue
                else:
                    print(str('Sanity {0:.2f} for home {1} in period from {2} to {3} higher than {4:.2f}; sufficient for analysis...'
                              .format(sanity_moving_horizon, home_id, moving_horizon_start, moving_horizon_end, sanity_lb)))
                
                # T_set_first_C_array = df_moving_horizon['T_set_first_C'].to_numpy()
                T_in_avg_C_array = df_moving_horizon['T_in_avg_C'].to_numpy()
                # print(df_moving_horizon['T_in_avg_C'])
                # print(list(T_in_avg_C_array))

                step_s = df_moving_horizon['interval_s'].mean()
                number_of_timesteps = len(T_in_avg_C_array)

                # load data from dataframe into np.arrays
                # print(df_moving_horizon)


                T_out_e_avg_C_array = df_moving_horizon['T_out_e_avg_C'].to_numpy()
                irradiation_hor_avg_W_p_m2_array = df_moving_horizon['irradiation_hor_avg_W_p_m2'].to_numpy()
                e_remaining_heat_avg_W_array = df_moving_horizon['e_remaining_heat_avg_W'].to_numpy()
                gas_sup_no_CH_avg_W_array = df_moving_horizon['gas_sup_no_CH_avg_W'].to_numpy()
                gas_sup_CH_avg_W_array = df_moving_horizon['gas_sup_CH_avg_W'].to_numpy()
               
                # print length of arrays and check uquality

                # print('#T_in_avg_C_array', len(T_in_avg_C_array))
                # print('#T_out_e_avg_C_array', len(T_out_e_avg_C_array))
                # print('#irradiation_hor_avg_W_p_m2_array', len(irradiation_hor_avg_W_p_m2_array))
                # print('#gas_sup_no_CH_avg_W_array', len(gas_sup_no_CH_avg_W_array))
                # print('#gas_sup_CH_avg_W_array', len(gas_sup_CH_avg_W_array))

                # check for equal length

                # print(len(T_in_avg_C_array) == len(irradiation_hor_avg_W_p_m2_array) 
                #       == len(T_out_e_avg_C_array) == len(gas_sup_no_CH_avg_W_array) 
                #       == len(gas_sup_CH_avg_W_array) == len(e_remaining_heat_avg_W_array))
            
            
                if (hint_A_m2 is None):
                    print('home {0} from {1} to {2} only run learning loop once with A as learnable model parameter'
                          .format(home_id,moving_horizon_start,moving_horizon_end))
                    innerloop=[np.NaN]
                else:
                    print('home {0} from {1} to {2} run learning loop twice: once with A={3} and once as as learnable model parameter'
                          .format(home_id,moving_horizon_start,moving_horizon_end, hint_A_m2))
                    innerloop=[hint_A_m2, np.NaN]

                for iterator_A_m2 in innerloop:

                    try:


                        ########################################################################################################################
                        # Gekko Model - Initialize
                        ########################################################################################################################
 
                        # initialize gekko
                        m = GEKKO(remote=False)
                        # m.time = np.linspace(0, (number_of_timesteps-1) * step_s, number_of_timesteps)
                        m.time = range(number_of_timesteps)







                        ########################################################################################################################
                        # Gekko Model - Model Parameters
                        ########################################################################################################################
                        
                        # Model parameter: H [W/K]: specific heat loss
                        m.H_W_p_K = m.FV(value=300.0, lb=0, ub=1000)
                        m.H_W_p_K.STATUS = 1
                        m.H_W_p_K.FSTATUS = 0
                        
                        # Model parameter: tau [s]: effective thermal inertia
                        m.tau_s = m.FV(value=(100 * s_p_h), lb=(10 * s_p_h), ub=(1000 * s_p_h))
                        m.tau_s.STATUS = 1
                        m.tau_s.FSTATUS = 0

                        ########################################################################################################################
                        #                                               Gekko - Equations
                        ########################################################################################################################

                        
                        ########################################################################################################################
                        # Equation - Q_gain_CH_avg_W: heat gain from natural gas used for central heating
                        ########################################################################################################################

                        # eta_sup_CH_frac [-]: upper heating efficiency of the central heating system

                        # This section of code (later) when  eta_sup_CH_frac is estimated using the model
                        # m.eta_sup_CH_frac = m.FV(value=0.8, lb=0, ub=1.0)
                        # m.eta_sup_CH_frac.STATUS = 1
                        # m.eta_sup_CH_frac.FSTATUS = 0
                        ## eta_sup_CH_frac.DMAX = 0.25

                        # Fix eta_sup_CH_frac when a hint is given 
                        m.eta_sup_CH_frac = m.Param(value=hint_eta_sup_CH_frac)
                        
                        m.gas_sup_CH_avg_W = m.MV(value=gas_sup_CH_avg_W_array)
                        m.gas_sup_CH_avg_W.STATUS = 0
                        m.gas_sup_CH_avg_W.FSTATUS = 1

                        m.Q_gain_gas_CH_avg_W = m.Intermediate(m.gas_sup_CH_avg_W * m.eta_sup_CH_frac)
                    
                    
                        ########################################################################################################################
                        # Equation - Q_gain_no_CH_avg_W: heat gain from natural gas used for central heating
                        ########################################################################################################################

                        # eta_sup_no_CH_frac [-]: superior efficiency of heating the home indirectly using gas, 
                        # for other primary purposes than heating the home, # eq48, PPT slide 24
                        m.eta_sup_no_CH_frac = m.Param(value=0.34)
                        
                        m.gas_sup_no_CH_avg_W = m.MV(value=gas_sup_no_CH_avg_W_array)
                        m.gas_sup_no_CH_avg_W.STATUS = 0
                        m.gas_sup_no_CH_avg_W.FSTATUS = 1
                        
                        m.Q_gain_gas_no_CH_avg_W = m.Intermediate(m.gas_sup_no_CH_avg_W * m.eta_sup_no_CH_frac)

                        ########################################################################################################################
                        # Equation - Q_gain_int_avg_W: Heat gain from internal sources
                        ########################################################################################################################
                        # Manipulated Variable: m.e_remaining_heat_avg_W: internal heat gain from internally used electricity
                        m.e_remaining_heat_avg_W = m.MV(value=e_remaining_heat_avg_W_array)
                        m.e_remaining_heat_avg_W.STATUS = 0
                        m.e_remaining_heat_avg_W.FSTATUS = 1

                        m.Q_gain_int_avg_W = m.Intermediate(m.e_remaining_heat_avg_W + Q_gain_int_occup_avg_W + m.Q_gain_gas_no_CH_avg_W)

                        ########################################################################################################################
                        # Equation - m.Q_gain_sol_avg_W: : heat gain from solar irradiation
                        ########################################################################################################################

                        if np.isnan(iterator_A_m2):
                            # Model parameter: A [m^2]: Effective area of the imaginary solar aperture in the horizontal plane
                            m.A_m2 = m.FV(value=5, lb=1, ub=100)
                            m.A_m2.STATUS = 1
                            m.A_m2.FSTATUS = 0
                        else:
                            # Fixed value: A [m^2]: Effective area of the imaginary solar aperture in the horizontal plane 
                            m.A_m2 = m.Param(value=iterator_A_m2)

                        m.irradiation_hor_avg_W_p_m2 = m.MV(value=irradiation_hor_avg_W_p_m2_array)
                        m.irradiation_hor_avg_W_p_m2.STATUS = 0
                        m.irradiation_hor_avg_W_p_m2.FSTATUS = 1
                        
                        m.Q_gain_sol_avg_W = m.Intermediate(m.irradiation_hor_avg_W_p_m2 * m.A_m2)
                        

                        ########################################################################################################################
                        # Equation - Q_gain_W: all heat gains combined
                        ########################################################################################################################
                        m.Q_gain_W = m.Intermediate(m.Q_gain_gas_CH_avg_W + m.Q_gain_sol_avg_W + m.Q_gain_int_avg_W)
                      

                        ########################################################################################################################
                        # Manipulated Variable (MV): T_out_e_avg_C [°C]: effective outdoor temperature
                        ########################################################################################################################
                        m.T_out_e_avg_C = m.MV(value=T_out_e_avg_C_array)
                        m.T_out_e_avg_C.STATUS = 0
                        m.T_out_e_avg_C.FSTATUS = 1

                        ########################################################################################################################
                        # Control Variable T_in_avg_C: Indoor temperature
                        ########################################################################################################################
                        m.T_in_avg_C = m.CV(value=T_in_avg_C_array)
                        m.T_in_avg_C.STATUS = 1
                        m.T_in_avg_C.FSTATUS = 1  # m.T_in_avg_C.MEAS_GAP= 0.25
                        

                        ########################################################################################################################
                        # Final Equations
                        ########################################################################################################################
                        m.C_J_p_K  = m.Intermediate(m.H_W_p_K * m.tau_s) 
                        m.Equation(m.T_in_avg_C.dt() == ((step_s * (m.Q_gain_W - m.H_W_p_K * (m.T_in_avg_C - m.T_out_e_avg_C))) / m.C_J_p_K))
                        
                        ########################################################################################################################
                        # Solve Equations
                        ########################################################################################################################
                        m.options.IMODE = 5
                        m.options.EV_TYPE = 2  # specific objective function (L1-norm vs L2-norm)
                        # m.options.NODES = 2  # when commented, the default is active: m.options.NODES = 3
                        m.solve(True)      

                        ########################################################################################################################
                        #                                                       Result
                        ########################################################################################################################

                        #add simulated indoor temperature for optimized solution to sim_T_in_avg_C column
                        # print(m.T_in_avg_C)
                        # print(len(list(m.T_in_avg_C.value)))
                        # print(list(m.T_in_avg_C.value))
                        
                        df_results_home_tempsim = df_moving_horizon.copy(deep=True)

                        df_results_home_tempsim['T_in_sim_avg_C'] = list(m.T_in_avg_C.value)
                        # print(df_results_home_tempsim) 

                        filename_prefix = datetime.now().astimezone(pytz.timezone('Europe/Amsterdam')).replace(microsecond=0).isoformat().replace(":","")
                        ex.write(df_results_home_tempsim, str('{0}-Alearnt-{1}-simdata_home-{2}-{3}-{4}.xlsx'.format(home_id,
                                                                                                              np.isnan(iterator_A_m2),
                                                                                                              filename_prefix, 
                                                                                                              moving_horizon_start.isoformat(),
                                                                                                              moving_horizon_end.isoformat())))
                        
                        duration_s = number_of_timesteps * step_s
                        error_K = (m.options.OBJFCNVAL ** (1/m.options.EV_TYPE))/duration_s

                        print('duration [s]: ', duration_s)
                        print('sanity: {0:.2f}'.format(sanity_moving_horizon))
                        print('error [K]: ', round(error_K, 4))
                        print('H [W/K]: ', round(m.H_W_p_K.value[0], 4))
                        print('tau [h]: ', round(m.tau_s.value[0] / s_p_h, 2))
                        print('A [m^2]: ', round(m.A_m2.value[0], 2))
                        print('A value fixed: ', not np.isnan(iterator_A_m2))
                        print('eta_sup [-]: ', round(m.eta_sup_CH_frac.value[0], 2))
                        print('eta_sup value fixed: ', True)

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
                            'sanity_frac': [sanity_moving_horizon],
                            'error_K': [error_K],
                            'H_W_p_K': [m.H_W_p_K.value[0]],
                            'tau_h': [m.tau_s.value[0] / s_p_h],
                            'A_m^2': [m.A_m2.value[0]],
                            'A_m^2_fixed': [not np.isnan(iterator_A_m2)],
                            'eta_sup': [m.eta_sup_CH_frac.value[0]],
                            'eta_sup_fixed': [True]})
                        df_result_row.set_index(['start_horizon'], inplace=True)

                    except KeyboardInterrupt:    
                        logging.error(str('KeyboardInterrupt; home analysis {0} not complete; saving results so far then will exit...'.format(home_id)))

                        # do NOT write an empty line for this iteration, to indicate it is not fully processed and we don't know 
                        ex.write(df_results_home, str(filename_prefix+'-results-aborted-{0}.xlsx'.format(home_id)))

                        # but DO include the incomplete home results in the final export
                        df_results = pd.concat([df_results, df_results_home])
                        ex.write(df_results_home, str(filename_prefix+'-results-aborted.xlsx'.format(home_id)))

                        # only then exit the function and return to caller
                        return

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
                            'sanity_frac': [sanity_moving_horizon],
                            'error_K': [np.nan],
                            'H_W_p_K': [np.nan],
                            'tau_h': [np.nan],
                            'A_m^2': [np.nan],
                            'A_m^2_fixed': [not (np.isnan(iterator_A_m2))],
                            'eta_sup': [np.nan],
                            'eta_sup_fixed': [True]})
                        df_result_row.set_index(['start_horizon'], inplace=True)
                        pass

                    #after allafter a single innerloop for A fixed or learnable
                    try:
                        df_results_home = pd.concat([df_results_home, df_result_row])
                    except KeyboardInterrupt:    
                        logging.error(str('KeyboardInterrupt; home analysis {0} not complete; saving results so far then will exit...'.format(home_id)))

                        # do write full line for this iteration, to indicate it is fully processed and we do know 
                        df_results_home = pd.concat([df_results_home, df_result_row])
                        ex.write(df_results_home, str(filename_prefix+'-results-aborted-{0}.xlsx'.format(home_id)))

                        # and include the incomplete home results in the final export
                        df_results = pd.concat([df_results, df_results_home])
                        ex.write(df_results_home, str(filename_prefix+'-results-aborted.xlsx'.format(home_id)))

                        # only then exit the function and return to caller
                        return

                #after a single innerloop for A fixed or learnable
                print(str('Analysis of all moving horizons for a single inner loop for home {0} complete.'.format(home_id)))

            #after all moving horizons of a single home; after a single innerloop for A fixed or learnable
            print(str('Analysis of all moving horizons for home {0} complete.'.format(home_id)))
            try:
                df_results = pd.concat([df_results, df_results_home])
                ex.write(df_results_home, str(filename_prefix+'-results-{0}.xlsx'.format(home_id)))
            except KeyboardInterrupt:    
                logging.error(str('KeyboardInterrupt; home analysis {0} complete; saving results so far then will exit...'.format(home_id)))
                ex.write(df_results, (filename_prefix+'-results-aborted.xlsx'))


        # and after all homes
        print('DONE: Analysis of all homes complete.')
        try:
            ex.write(df_results, (filename_prefix+'-results.xlsx'))
        except KeyboardInterrupt:    
            logging.error(str('KeyboardInterrupt; all home analyses complete; will continue saving results and then exit...'.format(home_id)))
            ex.write(df_results, (filename_prefix+'-results.xlsx'))
        return df_results