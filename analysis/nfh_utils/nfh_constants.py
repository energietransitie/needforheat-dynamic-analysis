import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from scipy import constants

# Time conversion factors
s_min_1 = constants.minute                                    # [s] per [min]
min_h_1 = constants.hour/constants.minute                     # [min] per [h]
h_d_1 = constants.day/constants.hour                          # [h] per [d]
d_a_1 = constants.year/constants.day                          # [d] per [a]]
s_h_1 = constants.hour                                        # [s] per [h]
s_d_1 = constants.day                                         # [s] per [d]
s_a_1 = constants.year                                        # [s] per [a] 

# Energy conversion factors
J_kWh_1 = constants.kilo * s_h_1                              # [J] per [kWh]
J_MJ_1 = constants.mega                                       # [J] per [MJ]

# Volumetric conversion factors
dm3_m_3 = constants.deka**3                                   # [dm^3] per [m^3]
ml_m_3 = (1/constants.milli) * dm3_m_3                        # [ml] per [m^3]

# Pressure conversion factors
Pa_mbar_1 = constants.milli * constants.bar                   # [Pa] per [mbar]

# Molar conversion factors
umol_mol_1 = 1/constants.micro                                # [µmol] per [mol]

# Area conversion factors
cm2_m_2 = 1/(constants.centi * constants.centi)               # [cm^2] per [m^2]

# Temperature conversion
temp_0__degC__K = constants.zero_Celsius                      # 0 [°C] = 273.15 [K]
  
# Gas conversion factors and fuctions for air
P_std__Pa = constants.atm                                     # standard gas pressure [Pa]
R__m3_Pa_K_1_mol_1 = PropsSI('GAS_CONSTANT', 'Air')           # gas constant [m^3⋅Pa⋅K^-1⋅mol^-1)]

# Air density
def air_density__kg_m_3(air_temp__degC, air_abs__Pa):
    """Calculate air density based on temperature and pressure."""
    if pd.isna(air_temp__degC) or pd.isna(air_abs__Pa):
        return np.nan
    else:
        return PropsSI('D', 'T', air_temp__degC + temp_0__degC__K, 'P', air_abs__Pa, 'Air')  # kg/m³

# Air specific heat capacity (mass-based)
def air_specific_heat_capacity__J_kg__1_K_1(air_temp__degC, air_abs__Pa):
    """Calculate air specific heat capacity at constant pressure based on temperature and pressure."""
    if pd.isna(air_temp__degC) or pd.isna(air_abs__Pa):
        return np.nan
    else:
        return PropsSI('C', 'T', air_temp__degC + temp_0__degC__K, 'P', air_abs__Pa, 'Air')  # J/(kg·K)

# Air molar specific heat capacity
def air_specific_heat_capacity__J_mol__1_K_1(air_temp__degC, air_abs__Pa):
    """Calculate air molar specific heat capacity based on temperature and pressure."""
    if pd.isna(air_temp__degC) or pd.isna(air_abs__Pa):
        return np.nan
    else:
        cp_mass = air_specific_heat_capacity__J_kg__1_K_1(air_temp__degC, air_abs__Pa)  # J/(kg·K)
        molar_mass = PropsSI('M', 'T', air_temp__degC + temp_0__degC__K, 'P', air_abs__Pa, 'Air')  # kg/mol
        return cp_mass * molar_mass  # J/(mol·K)

# Air volumetric specific heat capacity
def air_volumetric_specific_heat_capacity__J_m__3_K_1(air_temp__degC, air_abs__Pa):
    """Calculate air volumetric specific heat capacity based on temperature and pressure."""
    if pd.isna(air_temp__degC) or pd.isna(air_abs__Pa):
        return np.nan
    else:
        cp_mass = air_specific_heat_capacity__J_kg__1_K_1(air_temp__degC, air_abs__Pa)  # J/(kg·K)
        density = air_density__kg_m_3(air_temp__degC, air_abs__Pa)  # kg/m³
        return cp_mass * density  # J/(m³·K)
    
temp_room_std__degC = 20.0                                    # standard room temperature [°C]
temp_gas_std__degC = 0.0                                      # standard gas temperature [°C]
temp_gas_ref__degC = 15.0                                     # gas temperature for reference conditions, according to EN 437:2021 (E) [°C]
temp_gas_std__K = temp_gas_std__degC + temp_0__degC__K        # standard gas temperature [K]
temp_room_std__K = temp_room_std__degC + temp_0__degC__K      # standard room temperature [K]
temp_gas_ref__K = temp_gas_ref__degC + temp_0__degC__K        # gas temperature for reference conditions, according to EN 437:2021 (E) [K]
gas_room__mol_m_3 = (P_std__Pa
                / (R__m3_Pa_K_1_mol_1 * temp_room_std__K)
               )                                              # molar quantity of an ideal gas under room conditions [mol⋅m^-3]
gas_std__mol_m_3 = (P_std__Pa 
                / (R__m3_Pa_K_1_mol_1 * temp_gas_std__K)
               )    
air_std__J_mol_K = air_specific_heat_capacity__J_mol__1_K_1(
    temp_gas_std__degC, P_std__Pa
)                                                             # molar specific heat of air [J/mol⋅K] at constant pressure at standard conditions
air_room__J_mol_K = air_specific_heat_capacity__J_mol__1_K_1(
    temp_room_std__degC, P_std__Pa
)                                                             # molar specific heat of air [J/mol⋅K] at constant pressure at room conditions

air_std__J_m_3_K_1 = air_volumetric_specific_heat_capacity__J_m__3_K_1(
    temp_gas_std__degC, P_std__Pa
)                                                            # volumetric specific heat of air [J/(m³⋅K)] at standard conditions

air_room__J_m_3_K_1 = air_volumetric_specific_heat_capacity__J_m__3_K_1(
    temp_room_std__degC, P_std__Pa
)                                                             # volumetric specific heat of air [J/(m³⋅K)] at standard conditions

# Gravitational acceleration on earth
g__m_s_2 = constants.g                                        # Gravitational acceleration [m/s²] (https://en.wikipedia.org/wiki/Gravity_of_Earth#Conventional_value)

def water_density__kg_dm_3(water_temp__degC, water_abs__Pa):
    """Calculate water density based on temperature and pressure."""
    if pd.isna(water_temp__degC) or pd.isna(water_abs__Pa):
        return np.nan
    else:
        return PropsSI('D', 'T', water_temp__degC + temp_0__degC__K, 'P', water_abs__Pa, 'Water') / dm3_m_3  # kg/dm³

# Heat capacities
def water_specific_heat_capacity__J_kg_1_K_1(water_temp__degC, water_abs__Pa):
    """Calculate water specific heat capacity based on temperature and pressure."""
    if pd.isna(water_temp__degC) or pd.isna(water_abs__Pa):
        return np.nan
    else:
        return PropsSI('C', 'T', water_temp__degC + temp_0__degC__K, 'P', water_abs__Pa, 'Water')  # J/(kg·K)

def water_volumetric_heat_capacity__J_dm_3_K_1(water_temp__degC, water_abs__Pa):
    """Calculate water volumetric heat capacity (J/(dm³·K)) based on temperature and pressure."""
    if pd.isna(water_temp__degC) or pd.isna(water_abs__Pa):
        return np.nan
    else:
        specific_heat_capacity = water_specific_heat_capacity__J_kg_1_K_1(water_temp__degC, water_abs__Pa)
        density = water_density__kg_dm_3(water_temp__degC, water_abs__Pa)
        return specific_heat_capacity * density

steel__J_kg_1_K_1 = 466                                       # source: https://en.wikipedia.org/wiki/Table_of_specific_heat_capacities

# CO₂ concentration averages
co2_outdoor_eu_avg_2022__ppm = 415                            # Yearly average CO₂ concentration in Europe in 2022

# Metabolic conversion factors
O2ml_min_1_kg_1_p_1_MET_1 = 3.5                               # [mlO₂‧kg^-1‧min^-1] per [MET] 
desk_work__MET = 1.5                                          # Metabolic Equivalent of Task [MET] for desk work
sedentary__MET = 1.2                                          # Metabolic Equivalent of Task [MET] for sedentary activities according to NEN-EN 1521:2007
metabolism__molCO2_molO2_1 = 0.894                            # ratio: moles of CO₂ produced by (aerobic) human metabolism per mole of O₂ consumed 
adult_weight_nl_avg__kg = 77.5                                # average weight of Dutch adult [kg]
O2umol_s_1_p_1_MET_1 = (O2ml_min_1_kg_1_p_1_MET_1
                   * adult_weight_nl_avg__kg
                   / s_min_1 
                   * (umol_mol_1 * gas_room__mol_m_3 / ml_m_3)
                   )                                          # molar quantity of O₂ inhaled by an average Dutch adult at 1 MET [µmol/(p⋅s)]
co2_exhale_desk_work__umol_p_1_s_1 = (metabolism__molCO2_molO2_1
                            * desk_work__MET
                            * O2umol_s_1_p_1_MET_1
                           )                                  # molar quantity of CO₂ exhaled by Dutch desk worker doing desk work [µmol/(p⋅s)]
co2_exhale_sedentary__umol_p_1_s_1 = (metabolism__molCO2_molO2_1
                            * sedentary__MET
                            * O2umol_s_1_p_1_MET_1
                           )                                  # molar quantity of CO₂ exhaled by a sedentary person [µmol/(p⋅s)]

# Average Dutch occupancy and internal heat gain
household_nl_avg__p = 2.2                                     # average number of persons per Dutch household
asleep_at_home_nl_avg__h_d_1 = 8.6                            # average hours per day asleep for an average Dutch person (Source: TBO research of SCP)
awake_at_home_nl_avg__h_d_1 = 7.7                             # average hours per day awake and at home for an average Dutch person (source: TBO research of SCP)
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
heat_awake_int_nl_avg__W_p_1 = 105
heat_asleep_int_nl_avg__W_p_1 = 77

heat_int_present_nl_avg__W_p_1 = np.average(
    np.array([heat_asleep_int_nl_avg__W_p_1, heat_awake_int_nl_avg__W_p_1]),
    weights=np.array([asleep_at_home_nl_avg__h_d_1, awake_at_home_nl_avg__h_d_1])
)

heat_int_nl_avg__W_p_1 = (
    heat_int_present_nl_avg__W_p_1
    * at_home_nl_avg__h_d_1
    / h_d_1
)                                                             # daily average internal heat gain from an average Dutch person with average presence
                     
# Groningen natural gas averages (81,30%vol CH4, 14,35%vol N2), presumably at P_std__Pa and temp_gas_std__degC
gas_groningen_nl_avg_std_hhv__J_m_3 = 35.17e6                 # average higher heating value of Gronings gas: https://nl.wikipedia.org/wiki/Gronings_gas
gas_groningen_nl_avg_std_lhv__J_m_3 = 31.65e6                 # average lower heating value of Gronings gas: https://nl.wikipedia.org/wiki/Gronings_gas

# CO₂ emission factors (NL, 2024)
co2_wtw_groningen_gas_std_nl_avg_2024__g__m_3 = 2134          # https://www.co2emissiefactoren.nl/lijst-emissiefactoren/, January 2024
co2_wtw_e_onbekend_nl_avg_2024__g__kWh_1 = 328                # https://www.co2emissiefactoren.nl/lijst-emissiefactoren/, January 2024

# Characteristics of reference gas G25.3 (88%vol CH4, 12%vol N2) according to EN 437:2021 (E), Table B.5 (reference gas prescribed by Kiwa BRL 2021 for tests)
gas_g25_3_ref_lhv__J_m_3 = 29.92e6                            # lower heating value of G25.3 reference gas at P_std__Pa and temp_gas_ref__degC 
gas_g25_3_ref_hhv__J_m_3 = 33.20e6                            # higher heating value of G25.3 reference gas at P_std__Pa and temp_gas_ref__degC
gas_g25_3_test_pressure__Pa = 25  * Pa_mbar_1                 # test pressure (gauge pressure) to be used in tests

# Dutch meter code related averages
# Source: https://www.acm.nl/sites/default/files/old_publication/publicaties/12068_wijziging-informatiecode-elektriciteit-en-gas-en-begrippenlijst-gas-over-administratieve-volumeherleiding-voor-gasmeters.pdf
P_nl_avg__Pa = 101550                                         # average air pressure to be used for conversion of measured gas volumes 
overpressure_gas_nl_avg__Pa = 28 * Pa_mbar_1                  # presumed overpressure of the gas arriving at a home, relative to air pressure
temp_gas_nl_avg__degC = 15                                    # presumed average temperature of the gas arriving at a home
temp_gas_nl_avg__K = temp_gas_nl_avg__degC + temp_0__degC__K  

# Dutch weather related averages
temp_indoor_heating_season_nl_avg__degC = 18.33               # derived from reference climate used in NTA8800
temp_outdoor_heating_season_nl_avg__degC = 6.44               # derived from reference climate used in NTA8800
delta_temp_heating_season_nl_avg__K = temp_indoor_heating_season_nl_avg__degC - temp_outdoor_heating_season_nl_avg__degC

# Dutch home related averages
heat_tr_bldng_nl_avg__W_K_1 = 250                             # derived from NTA8800; TODO: move calculation from Excel EnergyFingerPrintCalculation.xlsx to here
th_inert_bldng_nl_avg__h = 50                                 # derived from https://publications.tno.nl/publication/34635174/QGAWjF/TNO-2019-P10600.pdf
th_mass_bldng_nl_avg__Wh_K_1 = (
    heat_tr_bldng_nl_avg__W_K_1 
    * th_inert_bldng_nl_avg__h
    )                                                         # (effective) thermal mass
floor_single_home_dwelling_nl_avg__m2 = 143                   # derived from https://opendata.cbs.nl/statline/#/CBS/nl/dataset/82550NED/table?dl=ACE80
aperture_sol_nl_avg__m2 = 3.7                                 # derived from NTA8800; TODO: move calculation from Excel EnergyFingerPrintCalculation.xlsx to here  
wind_chill_nl_avg__K_s_m_1 =  0.67                            # derived from KNMI report https://cdn.knmi.nl/knmi/pdf/bibliotheek/knmipubmetnummer/knmipub219.pdf 
aperture_inf_nl_avg__cm2 = (
    (heat_tr_bldng_nl_avg__W_K_1 * wind_chill_nl_avg__K_s_m_1)
    /
    (delta_temp_heating_season_nl_avg__K * air_std__J_m_3_K_1)
    *
    cm2_m_2
)                                      

# average Dutch boiler efficiency
eta_ch_nl_avg_hhv__W0 = 0.963                                 # average superior efficiency of boilers in the Netherlands (source: WoON2008; ISSO 82.3)

# average Dutch heat distribution system characteristics
radiator_capacity_design__W_m_2 = 100                         # recommended radiator heatig capacity per square meter floor area 
design_temp_outdoor__degC = -10.0                             # outdoor design temperature
design_temp_dstr_hi__degC = 70.0                              # design temperature for high temperature radiators
heated_fraction_nl_avg__0 = 0.4                               # estimate of typical fraction of an average Dutch home that is heated
heat_dstr_nl_avg_abs__Pa = (1 + 1.5) * P_std__Pa              # Typical absolute heat distribution system pressure [Pa]

heat_tr_dstr_nl_avg__W_K_1 = (
    radiator_capacity_design__W_m_2
    * floor_single_home_dwelling_nl_avg__m2
    * heated_fraction_nl_avg__0
    /
    (design_temp_dstr_hi__degC - temp_room_std__degC)
)                                                             # heat dissipation capacity of the heat distribution system

heat_dstr_water_nl_avg__kg = 100
heat_dstr_steel_nl_avg__kg = 100
th_mass_dstr_nl_avg__Wh_K_1 = (
    (
        (heat_dstr_water_nl_avg__kg * water_specific_heat_capacity__J_kg_1_K_1(25,heat_dstr_nl_avg_abs__Pa))
        +
        (heat_dstr_steel_nl_avg__kg * steel__J_kg_1_K_1)
    ) * heated_fraction_nl_avg__0
    / s_h_1
)                                                             # thermal mass of the heat distribution system


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
