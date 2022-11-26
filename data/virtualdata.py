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
        - a dataframe with a MultiIndex ['home_id', 'timestamp]; timestamp is timezone-aware in the tz timezone
        - columns:
          - 'occupancy_p': average number of people present in the room,
          - 'co2_ppm': average CO2-concentration in the room,
          - 'valve_frac_0' opening fraction of the ventilation valve 
        """
        df = (pd.read_csv(filename,
                          delimiter=';',
                          decimal=',',
                          usecols=['room_id', 'timestamp', 'occupancy_p', 'co2_ppm', 'valve_frac_0'],
                          parse_dates=['timestamp']
                        )
              # .rename(columns={'timestamp_ISO8601': 'timestamp',
              #                  'cCO2_ppm': 'co2_ppm',
              #                  'valve_frac': 'valve_frac_0',
              #                  'occupancy': 'occupancy_p'
              #                 }
              #        )
              .set_index(['room_id', 'timestamp'])
              .dropna(axis='index')
              .tz_convert(tz, level='timestamp')
              )
        return df