import numpy as np

# Time conversion factors
s_min_1 = 60                                                  # [s] per [min]
min_h_1 = 60                                                  # [min] per [h]
h_d_1 = 24                                                    # [h] per [d]
d_a_1 = 365.25                                                # [d] per [a]]
s_h_1 = s_min_1 * min_h_1                                     # [s] per [h]
s_d_1 = (h_d_1 * s_h_1)                                       # [s] per [d]
s_a_1 = (d_a_1 * s_d_1)                                      # [s] per [a] 

# Energy conversion factors
J_kWh_1 = 1000 * s_h_1                                        # [J] per [kWh]
J_MJ_1 = 1e6                                                  # [J] per [MJ]

# Volumetric conversion factors
ml_m_3 = 1e3 * 1e3                                            # [ml] per [m^3]

# Pressure conversion factors
Pa_mbar_1 = 1e2                                               # [Pa] per [mbar]

# Molar conversion factors
umol_mol_1 = 1e6                                              # [µmol] per [mol]

# Area conversion factors
cm2_m_2 = 1e2 * 1e2                                           # [cm^2] per [m^2]

# Temperature conversion
temp_0__degC__K = 273.15                                      # 0 [°C] = 273.15 [K]
  
# Gas conversion factors
P_std__Pa = 101325                                            # standard gas pressure [Pa]
R__m3_Pa_K_1_mol_1 = 8.3145                                   # gas constant [m^3⋅Pa⋅K^-1⋅mol^-1)]
temp_room_std__degC = 20.0                                    # standard room temperature [°C]
temp_gas_std__degC = 0.0                                      # standard gas temperature [°C]
temp_gas_ref__degC = 15.0                                     # gas temperature for reference conditions, according to EN 437:2021 (E) [°C]
temp_gas_std__K = temp_gas_std__degC + temp_0__degC__K        # standard gas temperature [K]
temp_room_std__K = temp_room_std__degC + temp_0__degC__K      # standard room temperature [K]
temp_gas_ref__K = temp_gas_ref__degC + temp_0__degC__K        # gas temperature for reference conditions, according to EN 437:2021 (E) [K]
room_std__mol_m_3 = (P_std__Pa
                / (R__m3_Pa_K_1_mol_1 * temp_room_std__K)
               )                                              # molar quantity of an ideal gas under room conditions [mol⋅m^-3]
gas_std__mol_m_3 = (P_std__Pa 
                / (R__m3_Pa_K_1_mol_1 * temp_gas_std__K)
               )                                              # molar quantity of an ideal gas under standard conditions [mol⋅m^-3] 
air_std__J_mol_K = 29.07                                      # Molar specific heat of air at constant pressure [J/mol⋅K] at typical room conditions
air_room__J_mol_K = 29.19                                     # Molar specific heat of air at constant pressure [J/mol⋅K] at typical room conditions

air_std__J_m_3_K_1 = (air_std__J_mol_K 
                      * (P_std__Pa 
                         / 
                         (R__m3_Pa_K_1_mol_1 
                          * 
                          temp_gas_std__K)
                        )
                     )                                         # volumetric specific heat of air at standard conditions

air_room__J_m_3_K_1 = (air_room__J_mol_K 
                      * (P_std__Pa 
                         / 
                         (R__m3_Pa_K_1_mol_1 
                          * 
                          temp_room_std__K)
                        )
                     )                                         # volumetric specific heat of air at standard conditions

# CO₂ concentration averages
co2_ext_2022__ppm = 415                                       # Yearly average CO₂ concentration in Europe in 2022

# Metabolic conversion factors
O2ml_min_1_kg_1_p_1_MET_1 = 3.5                               # [mlO₂‧kg^-1‧min^-1] per [MET] 
desk_work__MET = 1.5                                          # Metabolic Equivalent of Task for desk work [MET]
metabolism__molCO2_molO2_1 = 0.894                            # ratio: moles of CO₂ produced by (aerobic) human metabolism per mole of O₂ consumed 
adult_weight_nl_avg__kg = 77.5                                # average weight of Dutch adult [kg]
O2umol_s_1_p_1_MET_1 = (O2ml_min_1_kg_1_p_1_MET_1
                   * adult_weight_nl_avg__kg
                   / s_min_1 
                   * (umol_mol_1 * room_std__mol_m_3 / ml_m_3)
                   )                                          # molar quantity of O₂ inhaled by an average Dutch adult at 1 MET [µmol/(p⋅s)]
co2_exhale_desk_work__umol_p_1_s_1 = (metabolism__molCO2_molO2_1
                            * desk_work__MET
                            * O2umol_s_1_p_1_MET_1
                           )                                  # molar quantity of CO₂ exhaled by Dutch desk worker doing desk work [µmol/(p⋅s)]

# Average Dutch occupancy and internal heat gain
household_nl_avg__p = 2.2                                     # average number of persons per Dutch household
asleep_at_home_nl_avg__h_d_1 = 8.6                            # average hours per day asleep for an average Dutch person
awake_at_home_nl_avg__h_d_1 = 7.7                             # average hours per day awake and at home for an average Dutch person
at_home_nl_avg__h_d_1 = (asleep_at_home_nl_avg__h_d_1
                         + 
                         awake_at_home_nl_avg__h_d_1
                        )
away_from_home_nl_avg = h_d_1 - at_home_nl_avg__h_d_1
occupancy_nl_avg__p = (household_nl_avg__p
                      * 
                      at_home_nl_avg__h_d_1
                      / h_d_1
                     )
Q_gain_awake_int_nl_avg__W_p_1 = 105
Q_gain_asleep_int_nl_avg__W_p_1 = 77

Q_gain_int_present_nl_avg__W_p_1 = np.average(
    np.array([Q_gain_asleep_int_nl_avg__W_p_1, Q_gain_awake_int_nl_avg__W_p_1]),
    weights=np.array([asleep_at_home_nl_avg__h_d_1, awake_at_home_nl_avg__h_d_1])
)

Q_gain_int_nl_avg__W_p_1 = (
    Q_gain_int_present_nl_avg__W_p_1
    * at_home_nl_avg__h_d_1
    / h_d_1
)                                                             # daily average internal heat gain from an average Dutch person with average presence
                     
# Groningen natural gas averages (81,30%vol CH4, 14,35%vol N2), presumably at P_std__Pa and temp_gas_std__degC
gas_groningen_nl_avg_std_hhv__J_m_3 = 35.17e6                # average higher heating value of Gronings gas: https://nl.wikipedia.org/wiki/Gronings_gas
gas_groningen_nl_avg_std_lhv__J_m_3 = 31.65e6                # average lower heating value of Gronings gas: https://nl.wikipedia.org/wiki/Gronings_gas

# Characteristics of reference gas G25.3 (88%vol CH4, 12%vol N2) according to EN 437:2021 (E), Table B.5 (reference gas prescribed by Kiwa BRL 2021 for tests)
gas_g25_3_ref_lhv__J_m_3 = 29.92e6                            # Lower heating value of G25.3 reference gas at P_std__Pa and temp_gas_ref__degC 
gas_g25_3_ref_hhv__J_m_3 = 33.20e6                            # Higher heating value of G25.3 reference gas at P_std__Pa and temp_gas_ref__degC
gas_g25_3_test_pressure__Pa = 25  * Pa_mbar_1                 # Test pressure (gauge pressure) to be used in tests

# average Dutch boiler efficiency
eta_ch_nl_avg_hhv__W0 = 0.963                                 # average superior efficiency of boilers in the Netherlands (source: WoON2008; ISSO 82.3)

# Dutch meter code related averages
# Source: https://www.acm.nl/sites/default/files/old_publication/publicaties/12068_wijziging-informatiecode-elektriciteit-en-gas-en-begrippenlijst-gas-over-administratieve-volumeherleiding-voor-gasmeters.pdf
P_nl_avg__Pa = 101550                                         # average air pressure to be used for conversion of measured gas volumes 
overpressure_gas_nl_avg__Pa = 28 * Pa_mbar_1                  # Presumed overpressure of the gas arriving at a home, relative to air pressure
temp_gas_avg_nl__degC = 15                                    # Presumed average temperature of the gas arriving at a home
temp_gas_avg_nl__K = temp_gas_avg_nl__degC + temp_0__degC__K  

# Dutch weather related averages
temp_in_heating_season_nl_avg__degC = 18.33                   # derived from reference climate used in NTA8800
temp_out_heating_season_nl_avg__degC = 6.44                   # derived from reference climate used in NTA8800
delta_temp_heating_season_nl_avg__K = temp_in_heating_season_nl_avg__degC - temp_out_heating_season_nl_avg__degC

# Dutch home related averages
wind_chill_nl_avg__K_s_m_1 =  0.67                            # derived from KNMI report https://cdn.knmi.nl/knmi/pdf/bibliotheek/knmipubmetnummer/knmipub219.pdf 
H_nl_avg__W_K_1 = 250                                         # derived from NTA8800; TODO: move calculation from Excel EnergyFingerPrintCalculation.xlsx to here
A_sol_nl_avg__m2 = 3.7                                        # derived from NTA8800; TODO: move calculation from Excel EnergyFingerPrintCalculation.xlsx to here  
A_inf_nl_avg__m2 = (
    (H_nl_avg__W_K_1 * wind_chill_nl_avg__K_s_m_1)
    /
    (delta_temp_heating_season_nl_avg__K * air_std__J_m_3_K_1)
)                                      

# Dutch household related averages
g_use_cooking_nl_avg__m3_a_1 = 65                             # derived from ISSO 82.3, kookgas aannames, for household_nl_avg__p = 2.2  
g_use_cooking_nl_avg_hhv__W = (
    g_use_cooking_nl_avg__m3_a_1 
    * gas_groningen_nl_avg_std_hhv__J_m_3
    / s_a_1
)                                                             # average gas power (higher heating value) for cooking [W]
eta_cooking_nl_avg_hhv__W0 = 0.444                            # derived from https://publications.tno.nl/publication/34635174/QGAWjF/TNO-2019-P10600.pdf 
frac_remain_cooking_nl_avg__0 = 0.460                         # derived from https://publications.tno.nl/publication/34635174/QGAWjF/TNO-2019-P10600.pdf 


g_use_dhw_hhv__kWh_a_1_p_1 = 856                              # derived from NTA8800
g_use_dhw_nl_avg__m3_a_1 = (
    g_use_dhw_hhv__kWh_a_1_p_1 *  J_kWh_1
    * household_nl_avg__p
    / gas_groningen_nl_avg_std_hhv__J_m_3
)
g_use_dhw_nl_avg_hhv__W = (
    g_use_dhw_nl_avg__m3_a_1 
    * gas_groningen_nl_avg_std_hhv__J_m_3
    / s_a_1
)                                                             # average gas power (higher heating value) for domestic hot water [W]

eta_dhw_nl_avg_hhv__W0 = 0.716                                # derived from https://publications.tno.nl/publication/34635174/QGAWjF/TNO-2019-P10600.pdf
frac_remain_dhw_nl_avg__0 = 0.500                             # derived from https://publications.tno.nl/publication/34635174/QGAWjF/TNO-2019-P10600.pdf

g_not_ch_nl_avg__m3_a_1 = (
    g_use_cooking_nl_avg__m3_a_1 
    +
    g_use_dhw_nl_avg__m3_a_1
)                                                             # average gas use in m^3 per year for other purposes than home heating 
g_not_ch_nl_avg_hhv__W = (
    g_not_ch_nl_avg__m3_a_1 
    * gas_groningen_nl_avg_std_hhv__J_m_3
    / s_a_1
)                                                             # average gas power (heating value) for other purposes than home heating [W]


eta_not_ch_nl_avg_hhv__W0 = np.average(
    np.array([eta_cooking_nl_avg_hhv__W0 * frac_remain_cooking_nl_avg__0, eta_dhw_nl_avg_hhv__W0 * frac_remain_dhw_nl_avg__0]),
    weights=np.array([g_use_cooking_nl_avg__m3_a_1, g_use_dhw_nl_avg__m3_a_1])
)
