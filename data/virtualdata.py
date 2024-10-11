import pandas as pd

class Virtualdata:
    """
    Use this class to get data from the files that contain virtual data.
    """
        

    @staticmethod
    def get_virtual_room_data_csv(filename: str, tz:str) -> pd.DataFrame:
        """
        Obtain data from an csv file with virtual room data 
        Output:  
        - a dataframe with a MultiIndex ['id', 'timestamp]; timestamp is timezone-aware in the tz timezone
        - columns:
          - 'occupancy__p': average number of people present in the room,
          - 'co2_outdoor__ppm': average CO2-concentration in the room,
          - 'valve_frac__0' opening fraction of the ventilation valve 
        """
        df = (pd.read_csv(filename,
                          usecols=['id', 'source', 'timestamp', 'occupancy__p', 'co2_outdoor__ppm', 'valve_frac__0'],
                          parse_dates=['timestamp']
                        )
              .set_index(['id', 'source', 'timestamp'])
              .dropna(axis='index')
              .tz_convert(tz, level='timestamp')
              )
        return df
    
    
    @staticmethod
    def get_virtual_home_data_csv(filename: str, tz:str) -> pd.DataFrame:
        """
        Obtain data from an csv file with virtual home data 
        Output:  
        - a dataframe with a MultiIndex ['id', 'timestamp]; timestamp is timezone-aware in the tz timezone
        - columns:
            - 'temp_outdoor__degC',
            - 'wind__m_s_1',
            - 'ghi__W_m_2',
            - 'temp_indoor__degC',
            - 'temp_set__degC',
            - 'g_use__W',
            - 'e_use__W',
            - 'e_ret__W'
        """
        df = (pd.read_csv(filename,
                          usecols=['id', 
                                   'source', 
                                   'timestamp', 
                                   'temp_outdoor__degC', 
                                   'wind__m_s_1', 
                                   'ghi__W_m_2', 
                                   'temp_indoor__degC', 
                                   'temp_set__degC',
                                   'g_use__W',
                                   'e_use__W',
                                   'e_ret__W'
                                  ],
                          parse_dates=['timestamp']
                        )
              .set_index(['id', 'source', 'timestamp'])
              .dropna(axis='index')
              .tz_convert(tz, level='timestamp')
              )
        return df
