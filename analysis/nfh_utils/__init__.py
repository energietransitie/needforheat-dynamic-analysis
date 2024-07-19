# needforheat-analysis/analysis/utils/__init__.py

from .nfh_metrics import mae, rmse, rmae

from .nfh_constants import (
    s_min_1, min_h_1, s_h_1, s_d_1, s_a_1,
    J_kWh_1, J_MJ_1, ml_m_3, umol_mol_1,
    cm2_m_2, temp_0__degC__K, P_std__Pa,
    R__m3_Pa_K_1_mol_1, temp_room_std__degC,
    temp_gas_std__degC, temp_gas_std__K,
    temp_room_std__K, room_std__mol_m_3,
    gas_std__mol_m_3, co2_ext_2022__ppm,
    O2ml_min_1_kg_1_p_1_MET_1, desk_work__MET,
    metabolism__molCO2_molO2_1, average_Dutch_adult_weight__kg,
    O2umol_s_1_p_1_MET_1, co2_exhale_desk_work__umol_p_1_s_1,
    g_groningen_hhv___MJ_m_3, g_groningen_lhv___MJ_m_3,
    eta_ch_nl_avg_hhv__J0
)
