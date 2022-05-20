from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from gekko import GEKKO
import tqdm


class Learner():
    
    @staticmethod
    def learn_home_parameter_moving_horizon(df_data_homes:pd.DataFrame, 
                                            moving_horizon_duration_d=7, 
                                            homes_to_analyze=None, 
                                            start_analysis_period:datetime=None, 
                                            end_analysis_period:datetime=None, showdetails=False) -> pd.DataFrame:
        """
        Input:  
        - a dataframe with a timezone-aware datetime index and measurement values: with the following columns
            [
                'homepseudonym', 'heartbeat',
                'outdoor_temp_degC','windspeed_m_per_s', 'effective_outdoor_temp_degC', 'hor_irradiation_J_per_h_per_cm^2', 'hor_irradiation_W_per_m^2',  
                'indoor_temp_degC', 'indoor_temp_degC_CO2', 'indoor_setpoint_temp_degC',
                'gas_m^3', 'e_used_normal_kWh', 'e_used_low_kWh', 'e_returned_normal_kWh', 'e_returned_low_kWh', 'e_used_net_kWh', 'e_remaining_heat_kWh', 
                'timedelta', 'timedelta_s', 'daycompleteness'
            ]
        and optionally,
        - the number of days to use as moving horizon duration in the analysis
        - start datetime for the analysis (defaults to earliest datatime in the index column)
        - end datatime for the analysis (defaults to latest datatime in the index column)
        """

        # set default values for parameters not set
        
        if (moving_horizon_duration_d is None):
            moving_horizon_duration_d = 7

        if (homes_to_analyze is None):
            homes_to_analyze= df_data_homes['homepseudonym'].unique()
        if (start_analysis_period is None):
                start_analysis_period = df_data_homes.index.min()
        if (end_analysis_period is None): 
                end_analysis_period = df_data_homes.index.max()

        daterange_frequency = str(moving_horizon_duration_d) + 'D'

        print('Homes to analyse: ', homes_to_analyze)
        print('Start of analyses: ', start_analysis_period)
        print('End of analyses: ', end_analysis_period)
        print('Moving horizon: ', daterange_frequency)

        # create empty dataframe for results
        df_results = pd.DataFrame()

        # # make home iterator
        # if showdetails:
        #     home_iterator = homes_to_analyze
        # else:
            # home_iterator = tqdm.tqdm(homes_to_analyze)
            
        home_iterator = tqdm.tqdm(homes_to_analyze)
           
        # iterate over homes
        for home_id in home_iterator:

            if showdetails:
                print('Home pseudonym: ', home_id)

            df_data_one_home = df_data_homes[df_data_homes['homepseudonym'] == home_id]

            moving_horizon_starts = pd.date_range(start=start_analysis_period, end=end_analysis_period, inclusive='left', freq=daterange_frequency)

            # make moving horizon iterator
            # if showdetails:
            #     moving_horizon_iterator = moving_horizon_starts
            # else:
            #     moving_horizon_iterator = tqdm.tqdm(moving_horizon_starts)
            moving_horizon_iterator = tqdm.tqdm(moving_horizon_starts)
                
            # iterate over horizons
            for moving_horizon_start in moving_horizon_iterator:

                moving_horizon_end = min(end_analysis_period, moving_horizon_start + timedelta(days=moving_horizon_duration_d))

                df_moving_horizon = df_data_one_home[moving_horizon_start:moving_horizon_end]

                if showdetails:
                    print('Start datetime: ', moving_horizon_start)
                    print('End datetime: ', moving_horizon_end)

                delta_t = df_moving_horizon['timedelta_s'].mean()

                # load data from dataframe into np.arrays

                setpoint = np.asarray(df_moving_horizon['indoor_setpoint_temp_degC'])
                T_in_meas = np.asarray(df_moving_horizon['indoor_temp_degC'])
                T_out_eff_arr = np.asarray(df_moving_horizon['effective_outdoor_temp_degC'])
                T_out = np.asarray(df_moving_horizon['outdoor_temp_degC'])

                gas_total = np.asarray(df_moving_horizon['gas_m^3'])

                e_used_normal_val = np.asarray(df_moving_horizon['e_used_normal_kWh'])
                e_used_low_val = np.asarray(df_moving_horizon['e_used_low_kWh'])
                e_returned_normal_val = np.asarray(df_moving_horizon['e_returned_normal_kWh'])
                e_returned_low_val = np.asarray(df_moving_horizon['e_returned_low_kWh'])

                delta_E_supply_val = np.asarray(e_used_normal_val + e_used_low_val)

                delta_E_PV_val = 0

                delta_E_ret_val = np.asarray(e_returned_normal_val + e_returned_low_val)
                delta_EV_charge_val = 0

                delta_E_CH_val = 0


                delta_E_int_val = np.asarray(
                    (delta_E_supply_val + delta_E_PV_val - delta_E_ret_val - delta_EV_charge_val - delta_E_CH_val) / delta_t)   # [kWh/s]
                delta_Q_int_e_val = np.asarray(delta_E_int_val * 1000 * 60 * 60)    # [W]
                I_geo_eff_val = np.asarray(df_moving_horizon['hor_irradiation_W_per_m^2'])
                
                
                # print length of arrays and check uquality

                # print('#setpoint', len(setpoint))
                # print('#T_in_meas', len(T_in_meas))
                # print('#T_out_eff_arr', len(T_out_eff_arr))
                # print('#T_out', len(T_out))
                # print('#gas_total', len(gas_total))
                # print('#e_used_normal_val', len(e_used_normal_val))
                # print('#e_used_low_val', len(e_used_low_val))
                # print('#e_returned_normal_val', len(e_returned_normal_val))
                # print('#e_returned_low_val', len(e_returned_low_val))
                # print('#delta_E_supply_val', len(delta_E_supply_val))
                # print('#delta_E_ret_val', len(delta_E_ret_val))
                # print('#delta_E_int_val', len(delta_E_int_val))
                # print('#delta_Q_int_e_val', len(delta_Q_int_e_val))
                # print('#I_geo_eff_val', len(I_geo_eff_val))

                # check for equal length
                
                # print(len(setpoint) == len(T_in_meas) == len(T_out_eff_arr) 
                #       == len(T_out) == len(gas_total) == len(e_used_normal_val) 
                #       == len(e_used_low_val) == len(e_returned_normal_val) 
                #       == len(e_returned_low_val) == len(delta_E_supply_val) 
                #       == len(delta_E_ret_val) == len(delta_E_int_val) 
                #       == len(delta_Q_int_e_val) == len(I_geo_eff_val))

                ########################################################################################################################
                #                                                   tau initial values input
                ########################################################################################################################
                # tau Input: the following value should be based on hour for tau [hr]
                tau_init_val_hr = 100
                tau_lb_hr = 10
                tau_ub_hr = 1000

                # Internal conversion (do not change this part)
                tau_init_val = tau_init_val_hr * 3600
                tau_lb = tau_lb_hr * 3600
                tau_ub = tau_ub_hr * 3600

                ########################################################################################################################
                #                                                   Gekko Model - Initialize
                ########################################################################################################################
                # initialize gekko
                m = GEKKO(remote=False)
                m.time = np.linspace(delta_t, len(T_in_meas) * delta_t, len(T_in_meas))  # [s]
                
                # line below added to avoid "Warning: shifting time horizon to start at zero; Current starting time value: 900.000000000000"
                m.time = m.time - delta_t
                # print('m.time: ', m.time)
                # print ('len(T_in_meas): ', len(T_in_meas)) 
                # print('m.time[-1]: ', m.time[-1])
                
                ########################################################################################################################
                #                                                   Gekko Model - Variables
                ########################################################################################################################
                """"
                Model parameter:
                tau [hr]: effective thermal inertia
                eta_hs_CH [-]: upper heating efficiency of the central heating system
                COP_CH [-]: Coef. of Performance for heat pump
                H [W/K]: specific heat loss
                A_eff [m^2]: Effective area of the imaginary solar aperture in the horizontal plane
                """
                tau = m.FV(value=tau_init_val, lb=tau_lb, ub=tau_ub);
                tau.STATUS = 1;
                tau.FSTATUS = 0;  # tau.DMAX = 10
                H = m.FV(value=300.0, lb=0, ub=1000);
                H.STATUS = 1;
                H.FSTATUS = 0;  # H.DMAX=50                #[W/K]
                # eta_hs_CH = m.FV(value=0.8, lb=0, ub=1.0); eta_hs_CH.STATUS = 1; eta_hs_CH.FSTATUS = 0;  # eta_hs_CH.DMAX = 0.25
                # COP_CH = m.FV(value=1, lb=0.1, ub=7) ; COP_CH.STATUS = 1 ; COP_CH.FSTATUS = 0 ; #COP_CH.DMAX=1
                A_eff = m.FV(value=5, lb=1, ub=100) ; A_eff.STATUS = 1 ; A_eff.FSTATUS = 0            #[m^2]

                """"
                Constant parameter:
                h_E [J/kWh]: Convertion factor ( [kWh] to [J] ) = 1000 * 60 * 60

                h_sup [J/Nm^3]: superior calorific value of natural gas from the Groningen field = 35,170,000.00
                eta_hs_noCH [-]: upper efficiency of heating the home indirectly using gas, for other primary purposes than heating the home

                delta_Q_sol [J/s]: heat gain from solar irradiation
                delta_G_noCH [Nm^3/s]: the natural gas used for other purposes than central heating
                delta_Q_int_gas_noCH [J/s]: natural gas used for central heating

                delta_Q_int_occup [W]: internal heat gain from occupants
                delta_Q_int_occup [W] = Np * Q_int_person_avg [W]
                Np [-]: number of persons in the household living in the home
                Q_int_person_avg [W]: internal heat gain from persons
                """
                h_E = m.Param(value=60 * 60 * 1000)  # [J/kWh"], the conversion factor [kWh] to [J]
                h_sup = m.Param(value=35170000.0)  # [J/Nm^3] "superior calorific value of natural gas from the Groningen field"
                eta_hs_noCH = m.Param(value=0.34)  # eq48. and PowerPoint Slide 24 (Effective upper home for indirect heating eff.)

                eta_hs_CH = m.Param(value=0.9)
                COP_CH = m.Param(value=4)
                # A_eff = m.Param(value=6)

                delta_G_noCH = m.Param(value=339.0 / (365.25 * 24 * 60 * 60))  # [Nm^3/s]
                delta_Q_int_gas_noCH = m.Param(value=delta_G_noCH * eta_hs_noCH * h_sup)  # [W]=[J/s]

                Np = m.Param(value=2.2)  # average number of people in Dutch household
                Q_int_person_avg = m.Param(value=61)  # [J/s] average heat gain for each average person with average behaviour
                delta_Q_int_occup = m.Param(value=Np * Q_int_person_avg)  # [J/s]

                """"
                Manipulated parameter:
                delta_Q_int_e [J/s]: internal heat gain from internally used electricity
                delta_Q_int_e [J/s] = delta_E_int [kWh/s] * hE [J/kWh]
                T_out_eff [K]: effective outdoor temperature
                delta_E_CH [kWh]: Electricity used for heat pump
                delta_G [Nm3/s] = Natural gas supplied to the home via the natural gas net
                I_geo_eff [W/m^2] = geospatially interpolated global horizontal irradiation
                """
                delta_Q_int_e = m.MV(value=delta_Q_int_e_val);
                delta_Q_int_e.STATUS = 0;
                delta_Q_int_e.FSTATUS = 1  # [J/s]
                T_out_eff = m.MV(value=T_out_eff_arr);
                T_out_eff.STATUS = 0;
                T_out_eff.FSTATUS = 1  # [K]
                delta_E_CH = m.MV(value=delta_E_CH_val / delta_t);
                delta_E_CH.STATUS = 0;
                delta_E_CH.FSTATUS = 1  # [kWh/s]
                delta_G = m.MV(value=gas_total / delta_t);
                delta_G.STATUS = 0;
                delta_G.FSTATUS = 1  # [Nm^3/s]
                I_geo_eff = m.MV(value=I_geo_eff_val);
                I_geo_eff.STATUS = 0;
                I_geo_eff.FSTATUS = 1

                """"
                Control variable:
                T_in_sim [K]: Indoor temperature
                """
                T_in_sim = m.CV(value=T_in_meas);
                T_in_sim.STATUS = 1;
                T_in_sim.FSTATUS = 1;  # T_in_sim.MEAS_GAP= 0.25

                ########################################################################################################################
                #                                               Gekko - Equations
                ########################################################################################################################
                """
                delta_Q_gain [J/s]= delta_Q_CH [J/s] + delta_Q_int [J/s] + delta_Q_sol [J/s]
                delta_Q_gain [J/s]= Heat gain
                delta_Q_CH [J/s]= Heat gain from central hearting
                delta_Q_int [J/s]= Heat gain from internal devices
                delta_Q_sol [J/s]= delta_Q_int from solar irradiation
                """

                ########################################################################################################################
                #                                               Equation - delta_Q_CH
                ########################################################################################################################
                """"
                delta_Q_CH [J/s] = (delta_G_CH [Nm3/s] * eta_hs_ch [-] * h_sup [J/Nm3]) + (delta_E_CH [kWh/s] * COP_CH [-] * hE [J/kWh])
                delta_G_CH [Nm3/s] = Natural gas used for central heating
                delta_G_CH [Nm3/s] = delta_G [Nm3/s]- delta_G_noCH [Nm3/s]
                """

                delta_G_CH = m.Intermediate(delta_G - delta_G_noCH)  # [Nm3/s]
                delta_Q_CH = m.Intermediate((delta_G_CH * eta_hs_CH * h_sup) + (delta_E_CH * COP_CH * h_E))  # [J/s]
                # delta_Q_CH = m.Intermediate((delta_Q_CH * eta_hs_CH * h_sup) + (delta_E_CH * COP_CH * h_E))  # [J/s]
                ########################################################################################################################
                #                                                   Equation - delta_Q_int
                ########################################################################################################################
                """"
                delta_Q_int [J/s]: total internal heat
                delta_Q_int [J/s]= delta_Q_int_e + delta_Q_int_occup + delta_Q_int_gas_noCH
                delta_E_int [kWh/s] = delta_E_supply [kWh/s] + delta_E_PV [kWh/s] - delta_E_ret [kWh/s] - delta_E_EVcharge [kWh/s]
                """
                delta_Q_int = m.Intermediate(delta_Q_int_e + delta_Q_int_occup + delta_Q_int_gas_noCH)  # [J/s]

                ########################################################################################################################
                #                                                   Equation - delta_Q_sol
                ########################################################################################################################
                delta_Q_sol = m.Intermediate(A_eff * I_geo_eff)  # [J/s]

                ########################################################################################################################
                #                                                    Equation - delta_Q_gain
                ########################################################################################################################
                delta_Q_gain = m.Intermediate(delta_Q_CH + delta_Q_sol + delta_Q_int)  # [J/s]

                ########################################################################################################################
                #                                                   Final Equations
                ########################################################################################################################
                C_eff = m.Intermediate(H * tau)
                m.Equation(T_in_sim.dt() == (delta_Q_gain - (H * (T_in_sim - T_out_eff))) / C_eff)

                ########################################################################################################################
                #                                                    Solve Equations
                ########################################################################################################################
                m.options.IMODE = 5
                m.options.EV_TYPE = 1  # specific objective function (L1-norm vs L2-norm)
                m.options.NODES = 2
                # m.options.CV_TYPE = 2
                # add dead-band for measurement to avoid overfitting
                # T_in_sim.MEAS_GAP = 0.25
                m.solve(disp=showdetails)

                ########################################################################################################################
                #                                                       Result
                ########################################################################################################################

                duration_s = m.time[-1]
                error_K = (m.options.OBJFCNVAL ** (1/m.options.EV_TYPE))/duration_s
                
                if showdetails:
                    print('duration [s]: ', duration_s)
                    print('error [K]: ', round(error_K, 4))
                    print('H [W/K]: ', round(H.value[0], 4))
                    print('tau [h]: ', round(tau.value[0] / 3600, 2))
                    print('A [m^2]: ', round(A_eff.value[0], 2))
                    print('eta_hs [-]: ', round(eta_hs_CH.value[0], 2))
                    # print('COP_CH [-]: ', round(COP_CH.value[0], 2))

                # Create a results row
                df_result_row = pd.DataFrame(
                    {'pseudonym': [home_id],
                     'start_horizon': [moving_horizon_start],
                     'end_horizon': [moving_horizon_end],
                     'duration_s': [duration_s],
                     'error_K': [error_K],
                     'H_W_per_K': [H.value[0]],
                     'tau_h': [tau.value[0] / 3600],
                     'eta_hs': [eta_hs_CH.value[0]],
                     'A_m^2': [A_eff.value[0]]
                    }
                )
                df_results = pd.concat([df_results, df_result_row])
                
            #after all moving horizons
        # and after all homes
        return df_results