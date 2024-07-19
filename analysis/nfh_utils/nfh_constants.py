# Time conversion factors
s_min_1 = 60                                                  # [s] per [min]
min_h_1 = 60                                                  # [min] per [h]
s_h_1 = s_min_1 * min_h_1                                     # [s] per [h]
s_d_1 = (24 * s_h_1)                                          # [s] per [d]
s_a_1 = (365.25 * s_d_1)                                      # [s] per [a] 

# Energy conversion factors
J_kWh_1 = 1000 * s_h_1                                        # [J] per [kWh]
J_MJ_1 = 1e6                                                  # [J] per [MJ]

# Volumetric conversion factors
ml_m_3 = 1e3 * 1e3                                            # [ml] per [m^3]

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
temp_gas_std__K = temp_gas_std__degC + temp_0__degC__K        # standard gas temperature [K]
temp_room_std__K = temp_room_std__degC + temp_0__degC__K      # standard room temperature [K]
room_std__mol_m_3 = (P_std__Pa
                / (R__m3_Pa_K_1_mol_1 * temp_room_std__K)
               )                                              # molar quantity of an ideal gas under room conditions [mol⋅m^-3]
gas_std__mol_m_3 = (P_std__Pa 
                / (R__m3_Pa_K_1_mol_1 * temp_gas_std__K)
               )                                              # molar quantity of an ideal gas under standard conditions [mol⋅m^-3] 

# CO₂ concentration averages
co2_ext_2022__ppm = 415                                       # Yearly average CO₂ concentration in Europe in 2022

# Metabolic conversion factors
O2ml_min_1_kg_1_p_1_MET_1 = 3.5                               # [mlO₂‧kg^-1‧min^-1] per [MET] 
desk_work__MET = 1.5                                          # Metabolic Equivalent of Task for desk work [MET]
metabolism__molCO2_molO2_1 = 0.894                            # ratio: moles of CO₂ produced by (aerobic) human metabolism per mole of O₂ consumed 
average_Dutch_adult_weight__kg = 77.5                         # average weight of Dutch adult [kg]
O2umol_s_1_p_1_MET_1 = (O2ml_min_1_kg_1_p_1_MET_1
                   * average_Dutch_adult_weight__kg
                   / s_min_1 
                   * (umol_mol_1 * room_std__mol_m_3 / ml_m_3)
                   )                                          # molar quantity of O₂ inhaled by an average Dutch adult at 1 MET [µmol/(p⋅s)]
co2_exhale_desk_work__umol_p_1_s_1 = (metabolism__molCO2_molO2_1
                            * desk_work__MET
                            * O2umol_s_1_p_1_MET_1
                           )                                  # molar quantity of CO₂ exhaled by Dutch desk worker doing desk work [µmol/(p⋅s)]

# Groningen natural gas averages 
g_groningen_hhv___MJ_m_3=35.17                                # average higher heating value of natural gas from the Groningen gas field
g_groningen_lhv___MJ_m_3=31.65                                # average lower heating value of natural gas from the Groningen gas field

# average Dutch boiler efficiency
eta_ch_nl_avg_hhv__J0 = 0.963                                 # average superior efficiency of boilers in the Netherlands (source: WoON2008; ISSO 82.3)

