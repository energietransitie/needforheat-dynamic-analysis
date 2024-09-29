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
                             df_home_bag_data:pd.DataFrame=None,
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
              - property_sources['temp_indoor__degC']: indoor temperature
              - property_sources['temp_outdoor__degC']: outdoor temperature 
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
            - 'aperture_sol__m2': apparent solar aperture [m^2]
            - 'eta_ch_hhv__W0': higher heating value efficiency [-] of the heating system 
              In the Netherlands, eta_ch_nl_avg_hhv__W0 = 0.963 from nfh_utils is a reasonable hint
            - 'g_not_ch_hhv__W': average yearly gas power (higher heating value)  for other purposes than heating 
              In the Netherlands, g_not_ch_nl_avg_hhv__W = 377 from nfh_utils is a reasonable hint
            - 'eta_not_ch_hhv__W0': superior efficiency [-] of heating the home indirectly using gas
              I the Netherlands, 0.34 is a reasonable hint
            - 'wind_chill__K_s_m_1': wind chill factor (in NL: 0.67 is a reasonable hint)
            - 'aperture_inf__cm2': effective infiltration area (in NL, 108 is a reasonable hint)
            - 'heat_tr_building_cond__W_K_1': specific heat loss (in NL, 250 is a reasonable hint)
            - 'eta_dhw_hhv__W0': domestic hot water efficiency (in NL, 0.716 is a reasonable hint)
            - 'frac_remain_dhw__0': fraction of domestic hot water heat contributing to heating the home (in NL, 0.500 is a reasonable hint)
            - 'g_use_cooking_hhv__W': average gas power (higher heating value) for cooking (in NL, 72 is a reasonable hint)
            - 'eta_cooking_hhv__W0': cooking efficiency (in NL, 0.444 is a reasonable hint)
            - 'frac_remain_cooking__0': fraction of cooking heat contributing to heating the home (in NL, 0.460 is a reasonable hint)
        - df_home_bag_data: a DataFrame with index id and columns
            - 'building_floor_area__m2': usable floor area of a dwelling in whole square meters according to NEN 2580:2007.
            - 'building_volume__m3': (an estimate of) the building volume, e.g. 3D-BAG attribute b3_volume_lod22 (https://docs.3dbag.nl/en/schema/attributes/#b3_volume_lod22) 
            - (optionally) 'building_floors__0': the number of floors, e.g. 3D-BAG attribute b3_bouwlagen (https://docs.3dbag.nl/en/schema/attributes/#b3_bouwlagen)
        and optionally,
        - the number of days to use as learn period in the analysis
        - 'ev_type': type 2 is usually recommended, since this is typically more than 50 times faster
        
        Output:
        - a dataframe with per id the learned parameters and error metrics
        - a dataframe with additional column(s):
            - 'sim_temp_indoor__degC' best fiting indoor temperatures

        """
        
        # check presence of hints
        mandatory_hints = ['aperture_sol__m2',
                           'occupancy__p',
                           'heat_int__W_p_1',
                           'wind_chill__K_s_m_1',
                           'aperture_inf__cm2',
                           'heat_tr_building_cond__W_K_1', 
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


        # Use National averages, depending on hints provided
        
        heat_int_occup__W = hints['heat_int__W_p_1'] * hints['occupancy__p']    # average heat gain per occupant
      
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
                actual_heat_tr_building_cond__W_K_1 = id // 1e5
                actual_th_inertia_building__h = (id % 1e5) // 1e2
                actual_aperture_sol__m2 = id % 1e2
                actual_th_mass_building__Wh_K_1 = actual_heat_tr_building_cond__W_K_1 * actual_th_inertia_building__h
                actual_eta_ch_hhv__W0 = eta_ch_nl_avg_hhv__W0 
                actual_aperture_inf__cm2 = aperture_inf_nl_avg__cm2
                actual_heat_tr_dist__W_K_1 = heat_tr_dist_nl_avg__W_K_1
                actual_th_mass_dist__Wh_K_1 = th_mass_dist_nl_avg__W_K_1
            else:
                actual_heat_tr_building_cond__W_K_1 = np.nan
                actual_th_inertia_building__h = np.nan
                actual_aperture_sol__m2 = np.nan
                actual_th_mass_building__Wh_K_1 = np.nan
                actual_eta_ch_hhv__W0 = np.nan
                actual_aperture_inf__cm2 = np.nan
                actual_heat_tr_dist__W_K_1 = np.nan
                actual_th_mass_dist__Wh_K_1 = np.nan
                
            # Get building_volume__m3 and building_floor_area__m2 from building-specific table
            building_volume__m3 = df_home_bag_data.loc[id]['building_volume__m3']
            building_floor_area__m2 = df_home_bag_data.loc[id]['building_floor_area__m2']

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
                mae_temp_indoor__degC = np.nan
                rmse_temp_indoor__degC = np.nan

                # TODO loop over learn list

                learned_heat_tr_building_cond__W_K_1 = np.nan
                mae_heat_tr_building_cond__W_K_1 = np.nan

                learned_th_inertia_building__h = np.nan
                mae_th_inertia_building__h = np.nan

                learned_th_mass_building__Wh_K_1 = np.nan
                mae_th_mass_building__Wh_K_1 = np.nan

                learned_aperture_sol__m2 = np.nan
                mae_aperture_sol__m2 = np.nan

                learned_aperture_inf__cm2 = np.nan
                mae_aperture_inf__cm2 = np.nan
                
                learned_heat_tr_dist__W_K_1 = np.nan
                mae_heat_tr_dist__W_K_1 = np.nan

                learned_th_mass_dist__Wh_K_1 = np.nan
                mae_th_mass_dist__Wh_K_1 = np.nan

                
                ##################################################################################################################
                # GEKKO code

                try:
            
                    # GEKKO Model - Initialize
                    m = GEKKO(remote=False)
                    m.time = np.arange(0, duration__s, step__s)


                    ### Heat gains ###
                    
                    ## Heat gains from central heating ##

                    # g_use_ch_hhv_W [-]: higher heating value of gas input to the boiler for central heating purposes
                    g_use_ch_hhv_W = m.MV(value = df_learn[property_sources['g_use_ch_hhv__W']].astype('float32').values)
                    g_use_ch_hhv_W.STATUS = 0; g_use_ch_hhv_W.FSTATUS = 1

                    # eta_ch_hhv__W0 [-]: efficiency (relative to higher heating value) of the boiler for central heating
                    eta_ch_hhv__W0 = m.MV(value = df_learn[property_sources['eta_ch_hhv__W0']].astype('float32').values)
                    eta_ch_hhv__W0.STATUS = 0; eta_ch_hhv__W0.FSTATUS = 1

                    # heat_g_ch [W]: heat gain from natural gas used for central heating
                    heat_g_ch__W = m.Intermediate(g_use_ch_hhv_W * eta_ch_hhv__W0)

                    # heat_e_ch [W]: heat gain from natural electricity used for central heating (e.g. a heat pump)
                    heat_e_ch__W = 0 # in this version model, we do not (yet) include potential 
                    
                    # heat_ch [W]: heat gain for heat distribution system coming from the central heating system
                    heat_ch__W = m.Intermediate(heat_g_ch__W + heat_e_ch__W)

                    # heat_ch [W]: heat gain for the home from the heat distribution system
                    if 'heat_tr_dist__W_K_1' in learn : 
                        # set this parameter up so it can be learnt
                        heat_tr_dist__W_K_1 = m.FV(value = hints['heat_tr_dist__W_K_1'], lb=0, ub=1000); heat_tr_dist__W_K_1.STATUS = 1; heat_tr_dist__W_K_1.FSTATUS = 0
                    else:
                        heat_tr_dist__W_K_1 = hints['heat_tr_dist__W_K_1']

                    if 'th_mass_dist__Wh_K_1' in learn : 
                        # set this parameter up so it can be learnt
                        th_mass_dist__Wh_K_1 = m.FV(value = hints['th_mass_dist__Wh_K_1'], lb=0, ub=10000); th_mass_dist__Wh_K_1.STATUS = 1; th_mass_dist__Wh_K_1.FSTATUS = 0
                    else:
                        th_mass_dist__Wh_K_1 = hints['th_mass_dist__Wh_K_1']
                        
                    if ('heat_tr_dist__W_K_1' in learn) or ('th_mass_dist__J_K_1' in learn): 
                        # Temperture of hot water supplied by the heat generation system to the heat distributon system [°C]
                        temp_sup__degC = m.MV(value = df_learn[property_sources['temp_sup__degC']].astype('float32').values)
                        temp_sup__degC.STATUS = 0; temp_sup__degC.FSTATUS = 1
    
                        # Temperture of water returned to the heat generation system from the heat distributon system [°C]
                        temp_ret__degC = m.MV(value = df_learn[property_sources['temp_ret__degC']].astype('float32').values)
                        temp_ret__degC.STATUS = 0; temp_ret__degC.FSTATUS = 1

                        temp_dist__degC = m.Intermediate((temp_sup__degC + temp_ret__degC)/2) # TODO: check whether this should be an MV
                        heat_dist__W = m.Intermediate(heat_tr_dist__W_K_1 * (temp_dist__degC - temp_indoor__degC))
                        m.Equation(temp_dist__degC.dt() == (heat_ch__W - heat_dist__W ) / (th_mass_dist__Wh_K_1 * s_h_1 ))
                    else:
                        # temp_dist__degC = m.Intermediate((temp_sup__degC + temp_ret__degC)/2)
                        heat_dist__W = heat_ch__W
                    
                    ## Heat gains from domestic hot water ##

                    g_use_dhw_hhv__W = m.MV(value = df_learn[property_sources['g_use_dhw_hhv__W']].astype('float32').values)
                    g_use_dhw_hhv__W.STATUS = 0; g_use_dhw_hhv__W.FSTATUS = 1

                    # heat_not_ch [W]: heat gain from natural gas NOT used for central heating c.q. dhw + cooking
                    heat_g_not_ch__W = m.Intermediate(
                        g_use_dhw_hhv__W * hints['eta_dhw_hhv__W0'] * hints['frac_remain_dhw__0']
                        + hints['g_use_cooking_hhv__W'] * hints['eta_cooking_hhv__W0'] + hints['frac_remain_cooking__0']
                    )

                    ## Heat gains from electricity ##

                    # e [W] : internal heat gain from internally used electricity
                    e__W = m.MV(value = df_learn[property_sources['e__W']].astype('float32').values)
                    e__W.STATUS = 0; e__W.FSTATUS = 1

                    # heat_int [W]: calculated heat gain from internal sources
                    heat_int__W = m.Intermediate(e__W + heat_int_occup__W + heat_g_not_ch__W)

                    ## Heat gains from the sun ##

                    # aperture_sol__m2 [m^2]: calculated heat gain from internal sources
                    if 'aperture_sol__m2' in learn:
                        # set this parameter up so it can be learnt
                        aperture_sol__m2 = m.FV(value = hints['aperture_sol__m2'], lb=1, ub=100); aperture_sol__m2.STATUS = 1; aperture_sol__m2.FSTATUS = 0
                    else:
                        # do not learn this parameter, but use a fixed value based on hint
                        aperture_sol__m2 = m.Param(value = hints['aperture_sol__m2'])
                        learned_aperture_sol__m2 = np.nan

                    # Global horizontal irradiation [W/m^2] 
                    ghi__W_m_2 = m.MV(value = df_learn[property_sources['ghi__W_m_2']].astype('float32').values)
                    ghi__W_m_2.STATUS = 0; ghi__W_m_2.FSTATUS = 1

                    # Heat gain from solar irradiation [W] 
                    heat_sol__W = m.Intermediate(ghi__W_m_2 * aperture_sol__m2)
                    

                    ### Heat losses ###

                    ## Conductive heat loss ##
                    
                    # Conductive heat transmissivity of the building [W/K]
                    if 'heat_tr_building_cond__W_K_1' in learn:
                        # set this parameter up so it can be learnt
                        heat_tr_building_cond__W_K_1 = m.FV(value=hints['heat_tr_building_cond__W_K_1'], lb=0, ub=1000)
                        heat_tr_building_cond__W_K_1.STATUS = 1; heat_tr_building_cond__W_K_1.FSTATUS = 0
                    else:
                        # do not learn this parameter, but use a fixed value based on hint
                        heat_tr_building_cond__W_K_1 = m.Param(value = hints['heat_tr_building_cond__W_K_1'])
                        learned_heat_tr_building_cond__W_K_1 = np.nan
                    
                    # Indoor temperature [°C]: objective (Control Variable)
                    temp_indoor__degC = m.CV(value = df_learn[property_sources['temp_indoor__degC']].astype('float32').values)
                    temp_indoor__degC.STATUS = 1; temp_indoor__degC.FSTATUS = 1
                    # temp_indoor__degC.MEAS_GAP= 0.25

                    # Outdoor temperature [°C]
                    temp_outdoor__degC = m.MV(value = df_learn[property_sources['temp_outdoor__degC']].astype('float32').values)
                    temp_outdoor__degC.STATUS = 0; temp_outdoor__degC.FSTATUS = 1

                    # Indoor-outdoor temperature difference [K]
                    indoor_outdoor_delta__K = m.Intermediate(temp_indoor__degC - temp_outdoor__degC)
                    
                    # Heat loss due to conduction [W]
                    heat_loss_building_cond__W = m.Intermediate(heat_tr_building_cond__W_K_1 * indoor_outdoor_delta__K) 

                    ## Infiltration heat loss ##
                    
                    # Wind speed [m/s]
                    wind__m_s_1 = m.MV(value = df_learn[property_sources['wind__m_s_1']].astype('float32').values)
                    wind__m_s_1.STATUS = 0; wind__m_s_1.FSTATUS = 1
                    
                    # Infiltration aperture [cm^2]
                    if 'aperture_inf__cm2' in learn:
                        aperture_inf__cm2 = m.FV(value=hints['aperture_inf__cm2'], lb=0, ub=100000.0)
                        aperture_inf__cm2.STATUS = 1; aperture_inf__cm2.FSTATUS = 0
                    else:
                        aperture_inf__cm2 = m.Param(value=hints['aperture_inf__cm2'])
                        learned_aperture_inf__cm2 = np.nan  
                    
                    # Heat loss due to infiltration [W]
                    air__J_m_3_K_1 = air_room__J_m_3_K_1 # if needed, the volumetric heat capacity can be made specific for pressure and temperature
                    air_indoor__mol_m_3 = gas_room__mol_m_3 # if needed, the molar quantity can be made specific for pressure and temperature

                    air_inf__m3_s_1 = m.Intermediate(wind__m_s_1 * aperture_inf__cm2 / cm2_m_2)
                    heat_tr_building_inf__W_K_1 = m.Intermediate(air_inf__m3_s_1 * air__J_m_3_K_1) 
                    heat_loss_building_inf__W = m.Intermediate(heat_tr_building_inf__W_K_1 * indoor_outdoor_delta__K)
                  
                    ## Ventilation heat loss ##

                    if 'ventilation__dm3_s_1' in learn:
    
                        
                        # Ventilation rate based on CO₂ concentration model
                        
                        # CO₂ concentration [ppm]
                        co2_indoor__ppm = m.CV(value = df_learn[property_sources['co2_indoor__ppm']].values)
                        co2_indoor__ppm.STATUS = 1; co2_indoor__ppm.FSTATUS = 1
                        
                        # CO₂ concentration gain [ppm/s]
                        occupancy__p = m.MV(value = df_learn[property_sources['occupancy__p']].astype('float32').values)
                        occupancy__p.STATUS = 0; occupancy__p.FSTATUS = 1
                        
                        co2_indoor_gain__ppm_s_1 = m.Intermediate(occupancy__p 
                                                           * co2_exhale_desk_work__umol_p_1_s_1 
                                                           / (building_volume__m3 * air_indoor__mol_m_3)
                                                          )
    
                        # CO₂ concentration loss [ppm/s]
                        ventilation__dm3_s_1 = m.MV(value=hints['ventilation_default__dm3_s_1'],
                                                    lb=0.0,
                                                    ub=(hints['ventilation_max__dm3_s_1_m_2'] * building_floor_area__m2)
                                                   ) 
                        ventilation__dm3_s_1.STATUS = 1; ventilation__dm3_s_1.FSTATUS = 1
                        
                        air_changes_vent__s_1 = m.Intermediate(ventilation__dm3_s_1 / (building_volume__m3 * dm3_m_3))
                        air_changes_inf__s_1 = m.Intermediate(air_inf__m3_s_1 / building_volume__m3)
                        air_changes_total__s_1 = m.Intermediate(air_changes_vent__s_1 + air_changes_inf__s_1)
    
                        co2_elevation__ppm = m.Intermediate(co2_indoor__ppm - hints['co2_outdoor__ppm'])
                        co2_indoor_loss__ppm_s_1 = m.Intermediate(air_changes_total__s_1 * co2_elevation__ppm)
    
                        # CO₂ concentration balance [ppm/s]
                        m.Equation(co2_indoor__ppm.dt() == co2_indoor_gain__ppm_s_1 - co2_indoor_loss__ppm_s_1)
                        
                        # Ventilation heat transmissivity of the building [W/K]
   
                        heat_tr_building_vent__W_K_1 = m.Intermediate(air_changes_vent__s_1 * building_volume__m3 * air__J_m_3_K_1)
                        heat_loss_building_vent__W = m.Intermediate(heat_tr_building_vent__W_K_1 * indoor_outdoor_delta__K)

                    else: # do NOT learn ventilation heat losses separately (but incorporate these losses in conduciton and infiltration losses)
                        heat_tr_building_vent__W_K_1 = 0
                        heat_loss_building_vent__W = 0

                    ## Thermal inertia ##
                    
                    # Thermal inertia of the building [h]
                    if 'th_inertia_building__h' in learn:
                        # set this parameter up so it can be learnt
                        th_inertia_building__h = m.FV(value = hints['th_inertia_building__h'], lb=(10), ub=(1000))
                        th_inertia_building__h.STATUS = 1; th_inertia_building__h.FSTATUS = 0
                    else:
                        # do not learn this parameter, but use a fixed value based on hint
                        th_inertia_building__h = m.Param(value = hints['th_inertia_building__h'])
                        learned_th_inertia_building__h = np.nan
                    
                    ### Heat balance ###

                    heat_gain_building__W = m.Intermediate(heat_dist__W + heat_sol__W + heat_int__W)
                    heat_loss_building__W = m.Intermediate(heat_loss_building_cond__W + heat_loss_building_inf__W + heat_loss_building_vent__W)
                    heat_tr_building_building__W_K_1 = m.Intermediate(heat_tr_building_cond__W_K_1 + heat_tr_building_inf__W_K_1 + heat_tr_building_vent__W_K_1)
                    th_mass_building__J_K_1  = m.Intermediate(heat_tr_building_building__W_K_1 * th_inertia_building__h * s_h_1) 
                    m.Equation(temp_indoor__degC.dt() == ((heat_gain_building__W - heat_loss_building__W) / th_mass_building__J_K_1))
                    
                    # GEKKO - Solver setting
                    m.options.IMODE = 5
                    m.options.EV_TYPE = ev_type # specific objective function (1 = MAE; 2 = RMSE)
                    m.solve(disp = False)      

                    # Write best fitting temperatures into df_data
                    df_data.loc[(id,learn_streak_period_start):(id,learn_streak_period_end), 'sim_temp_indoor__degC'] = np.asarray(temp_indoor__degC)

                    if ('heat_tr_dist__W_K_1' in learn) or ('th_mass_dist__J_K_1' in learn): 
                        df_data.loc[(id,learn_streak_period_start):(id,learn_streak_period_end), 'sim_temp_dist__degC'] = np.asarray(temp_dist__degC)

                    # set learned variables and calculate error metrics: 
                    # mean absolute error (mae) for all learned parameters; 
                    # root mean squared error (rmse) only for predicted time series
                    
                    mae_temp_indoor__degC = mae(temp_indoor__degC, df_learn[property_sources['temp_indoor__degC']])
                    logging.info(f'mae_temp_indoor__degC: {mae_temp_indoor__degC}')
                    rmse_temp_indoor__degC = rmse(temp_indoor__degC, df_learn[property_sources['temp_indoor__degC']])
                    logging.info(f'rmse_temp_indoor__degC: {rmse_temp_indoor__degC}')

                    # TODO loop over learn list
                    if 'heat_tr_building_cond__W_K_1' in learn:
                        learned_heat_tr_building_cond__W_K_1 = heat_tr_building_cond__W_K_1.value[0]
                        mae_heat_tr_building_cond__W_K_1 = abs(learned_heat_tr_building_cond__W_K_1  - actual_heat_tr_building_cond__W_K_1)                  # evaluates to np.nan if no actual value
                    if 'th_inertia_building__h' in learn:
                        learned_th_inertia_building__h = th_inertia_building__h.value[0]
                        mae_th_inertia_building__h = abs(learned_th_inertia_building__h - actual_th_inertia_building__h)                                        # evaluates to np.nan if no actual value
                    if 'heat_tr_building_cond__W_K_1' in learn or 'th_inertia_building__h' in learn :
                        learned_th_mass_building__Wh_K_1 = learned_heat_tr_building_cond__W_K_1 * learned_th_inertia_building__h
                        mae_th_mass_building__Wh_K_1 = abs(learned_th_mass_building__Wh_K_1 - actual_th_mass_building__Wh_K_1)                            # evaluates to np.nan if no actual value
                    if 'aperture_sol__m2' in learn:
                        learned_aperture_sol__m2 = aperture_sol__m2.value[0]
                        mae_aperture_sol__m2 = abs(learned_aperture_sol__m2 - actual_aperture_sol__m2)                               # evaluates to np.nan if no actual value
                    if 'aperture_inf__cm2' in learn:
                        learned_aperture_inf__cm2 = aperture_inf__cm2.value[0]
                        mae_aperture_inf__cm2 = abs(learned_aperture_inf__cm2 - actual_aperture_inf__cm2)                            # evaluates to np.nan if no actual value
                    if 'heat_tr_dist__W_K_1' in learn:
                        learned_heat_tr_dist__W_K_1 = heat_tr_dist__W_K_1.value[0]
                        mae_heat_tr_dist__W_K_1 = abs(learned_heat_tr_dist__W_K_1 - actual_heat_tr_dist__W_K_1)                   # evaluates to np.nan if no actual value
                    if 'th_mass_dist__Wh_K_1' in learn:
                        learned_th_mass_dist__Wh_K_1 = th_mass_dist__Wh_K_1.value[0]
                        mae_th_mass_dist__Wh_K_1 = abs(learned_th_mass_dist__Wh_K_1 - actual_th_mass_dist__Wh_K_1)             # evaluates to np.nan if no actual value


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
                                    'mae_temp_indoor__degC': [mae_temp_indoor__degC],
                                    'rmse_temp_indoor__degC': [rmse_temp_indoor__degC],
                                    'learned_heat_tr_building_cond__W_K_1': [learned_heat_tr_building_cond__W_K_1],
                                    'actual_heat_tr_building_cond__W_K_1': [actual_heat_tr_building_cond__W_K_1],
                                    'mae_heat_tr_building_cond__W_K_1': [mae_heat_tr_building_cond__W_K_1],
                                    'learned_th_inertia_building__h': [learned_th_inertia_building__h],
                                    'actual_th_inertia_building__h': [actual_th_inertia_building__h], 
                                    'mae_th_inertia_building__h': [mae_th_inertia_building__h], 
                                    'learned_th_mass_building__Wh_K_1':[learned_th_mass_building__Wh_K_1],
                                    'actual_th_mass_building__Wh_K_1':[actual_th_mass_building__Wh_K_1],
                                    'mae_th_mass_building__Wh_K_1': [mae_th_mass_building__Wh_K_1],
                                    'learned_aperture_sol__m2': [learned_aperture_sol__m2],
                                    'actual_aperture_sol__m2': [actual_aperture_sol__m2],
                                    'mae_aperture_sol__m2': [mae_aperture_sol__m2],
                                    'learned_aperture_inf__cm2': [learned_aperture_inf__cm2],
                                    'mae_aperture_inf__cm2': [mae_aperture_inf__cm2],
                                    'learned_heat_tr_dist__W_K_1': [learned_heat_tr_dist__W_K_1],
                                    'mae_heat_tr_dist__W_K_1': [mae_heat_tr_dist__W_K_1],
                                    'learned_th_mass_dist__Wh_K_1': [learned_th_mass_dist__Wh_K_1],
                                    'mae_th_mass_dist__Wh_K_1': [mae_th_mass_dist__Wh_K_1],
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
 
    

