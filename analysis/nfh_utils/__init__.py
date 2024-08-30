# needforheat-analysis/analysis/utils/__init__.py

from .nfh_metrics import mae, rmse, rmae

from .nfh_constants import (
    s_min_1, min_h_1, h_d_1, d_a_1, s_h_1, s_d_1, s_a_1,
    J_kWh_1, J_MJ_1, ml_m_3, umol_mol_1,
    cm2_m_2, temp_0__degC__K, P_std__Pa,
    R__m3_Pa_K_1_mol_1, temp_room_std__degC,
    temp_gas_std__degC, temp_gas_std__K,
    temp_room_std__K, room_std__mol_m_3,
    temp_gas_ref__degC, temp_gas_ref__K, 
    gas_std__mol_m_3, co2_ext_2022__ppm,
    O2ml_min_1_kg_1_p_1_MET_1, desk_work__MET,
    metabolism__molCO2_molO2_1, adult_weight_nl_avg__kg,
    O2umol_s_1_p_1_MET_1, co2_exhale_desk_work__umol_p_1_s_1,
    gas_groningen_nl_avg_hhv__J_m_3, gas_groningen_nl_avg_lhv__J_m_3,
    gas_g20_lhv__J_m_3, gas_g20_hhv__J_m_3,
    gas_g25_lhv__J__m3, gas_g25_hhv__J__m3, 
    gas_g25_3_lhv__J__m3, gas_g25_3_hhv__J__m3,
    household_nl_avg__p, 
    asleep_at_home_nl_avg__h_d_1, awake_at_home_nl_avg__h_d_1,
    at_home_nl_avg__h_d_1, away_from_home_nl_avg, 
    occupancy_nl_avg__p,
    Q_gain_awake_int_nl_avg__W_p_1, Q_gain_asleep_int_nl_avg__W_p_1, 
    Q_gain_int_present_nl_avg__W_p_1, Q_gain_int_nl_avg__W_p_1,
    eta_ch_nl_avg_hhv__W0, g_not_ch_nl_avg__m3_a_1, g_not_ch_nl_avg_hhv__W,
    H_nl_avg__W_K_1, A_sol_nl_avg__m2,
    temp_in_heating_season_nl_avg__degC, temp_out_heating_season_nl_avg__degC, delta_temp_heating_season_nl_avg__K,
    wind_chill_nl_avg__K_s_m_1, A_inf_nl_avg__m2,
    g_use_cooking_nl_avg__m3_a_1, eta_sup_cooking_nl_avg__0, frac_remain_cooking_nl_avg__0, 
    g_use_dhw_nl_avg__m3_a_1, eta_sup_dhw_nl_avg__0, frac_remain_dhw_nl_avg__0, 
    eta_not_ch_nl_avg_hhv__W0
)