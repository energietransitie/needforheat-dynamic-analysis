import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
from tqdm.notebook import tqdm
from sqlalchemy import create_engine, engine
import logging

class Measurements:
    """
    Use this class to get data from the database that contains measurements.
    """
        
    @staticmethod    
    def get_property_ids(property_mapping_dict = None) -> str:
        """
        gets a list of property_ids to speed up the query
        """

        db_url_env = os.getenv("TWOMES_DB_URL")
        assert db_url_env, 'Environment variable TWOMES_DB_URL not set. Format: user:pass@host:port/db '

        db = create_engine("mysql+mysqlconnector://"+db_url_env).connect().execution_options(stream_results=True)
        
        sql_query = "SELECT id FROM property WHERE name IN "+ str(tuple(property_mapping_dict.keys()))
        print(sql_query)

        df = pd.DataFrame()
        df = pd.read_sql(sql=sql_query, con=db)
            
        db.close()   
        
        return str(tuple(df['id']))

        
    @staticmethod    
    def get_raw_measurements(homes,
                             first_day:datetime=None, last_day:datetime=None,
                             property_mapping_dict = None,
                             tz_source:str = 'UTC', tz_home:str = 'Europe/Amsterdam') -> pd.DataFrame:
        
        """
        in: 
        - homes: list of home_ids (aka account.pseudonyms in the twomes database)
        - first_day: timezone-aware date
        - last_day: , timezone-aware date; data is extracted until end of day
        out: dataframe with measurements
        """
        db_url_env = os.getenv("TWOMES_DB_URL")
        assert db_url_env, 'Environment variable TWOMES_DB_URL not set. Format: user:pass@host:port/db '

        db = create_engine("mysql+mysqlconnector://"+db_url_env).connect().execution_options(stream_results=True)
        
        largest_measurement_interval = timedelta(hours=1)
        # convert starttime & endtime to database timezone; extend on both sides with interval of one 
        if first_day is not None: 
            logging.info(f'first_day: {first_day}')
            extractor_starttime = (first_day - largest_measurement_interval).astimezone(pytz.timezone(tz_source))
            logging.info(f'extractor_starttime: {extractor_starttime}')
            first_str = "'" + extractor_starttime.strftime('%Y-%m-%d') + "'"
            logging.info(f'first_str: {first_str}')
        if last_day is not None: 
            logging.info(f'last_day: {last_day}')
            extractor_endtime = (last_day + timedelta(days=1) + largest_measurement_interval).astimezone(pytz.timezone(tz_source))
            logging.info(f'extractor_endtime: {extractor_endtime}')
            last_str = "'" + extractor_endtime.strftime('%Y-%m-%d') + "'"
            logging.info(f'last_str: {last_str}')

        sql_data_Twomes = """
        SELECT
            m.timestamp AS timestamp,
            a.pseudonym AS home_id,
            dt.name AS device_type,
            p.name AS property,
            m.value AS value,
            p.unit AS unit
        FROM
            measurement m
        JOIN device d ON
            m.device_id = d.id
        JOIN building b ON
            d.building_id = b.id
        JOIN account a ON
            b.account_id = a.id
        JOIN device_type dt ON
            d.device_type_id = dt.id
        JOIN property p ON
            m.property_id = p.id"""

        match len(homes):
            case 0:
                logging.warning('empty list of homes')
            case 1:
                sql_data_Twomes = sql_data_Twomes + " WHERE a.pseudonym = "+ str(homes[0])
            case _:
                sql_data_Twomes = sql_data_Twomes + " WHERE a.pseudonym IN "+ f'{tuple(map(str, homes))}'        
        
        match len(property_mapping_dict):
            case 0: 
                logging.warning('empty list of property names')
            case 1:
                sql_query_properties = "SELECT id FROM property WHERE name = '"+ str(list(property_mapping_dict.keys())[0]) + "'"
                df_properties = pd.read_sql(sql=sql_query_properties, con=db)
                logging.info(f'first_day: {sql_query_properties}')
            case _:
                sql_query_properties = "SELECT id FROM property WHERE name IN "+ str(tuple(property_mapping_dict.keys()))
                df_properties = pd.read_sql(sql=sql_query_properties, con=db)
                logging.info(f'first_day: {sql_query_properties}')
            
        match len(df_properties.index):
            case 0:
                logging.warning('empty list of properties found')
            case 1:
                sql_data_Twomes = sql_data_Twomes + " AND p.id = "+ str(df_properties['id'].iloc[0])
            case _:
                sql_data_Twomes = sql_data_Twomes + " AND p.id IN "+ str(tuple(df_properties['id']))

        if first_day is not None: 
            sql_data_Twomes = sql_data_Twomes + " AND m.timestamp >= "+ first_str

        if last_day is not None: 
            sql_data_Twomes = sql_data_Twomes + " AND m.timestamp <= "+ last_str 

        logging.info(sql_data_Twomes.replace('\n',' '))

        df = pd.DataFrame()

        #TODO: react on tz_source, depending on whether tz_source == 'UTC'. 
        for chunk in tqdm(pd.read_sql(sql=sql_data_Twomes.replace('\n',' '),
                                               con=db,
                                               chunksize = 2000000,
                                               parse_dates={"timestamp": {"utc": "True"}}
                                              )
                                       ):
            df = pd.concat([df, chunk.astype({'home_id' :'category','device_type':'category','property':'category'})])
            
        db.close()

        logging.info("Dropping duplicates...")
        df.drop_duplicates(subset=['home_id', 'timestamp','device_type', 'property'], keep='first', inplace=True)
        
        logging.info("Setting index...")
        df = (df.set_index(['home_id',
                            'timestamp',
                            'device_type',
                            'property']
                          )
              .sort_index()
              .tz_convert(tz_home, level='timestamp')
             )
        
        return df
        
        
    @staticmethod    
    def get_raw_homes_data(homes,
                           first_day:datetime=None, last_day:datetime=None,
                           property_mapping_dict = None,
                           tz_source:str = 'UTC', tz_home:str = 'Europe/Amsterdam') -> pd.DataFrame:
        """
        in: 
        - homes: list of home_ids (aka account.pseudonyms in the twomes database)
        - first_day: timezone-aware date
        - last_day: , timezone-aware date; data is extracted until end of day
        out:
        - DataFrame with measurement values in columns
        """

        df = Measurements.get_raw_measurements(homes,
                                        first_day, last_day,
                                        property_mapping_dict,
                                        tz_source, tz_home)
        
        del df['unit']
        if property_mapping_dict is not None:
            logging.info("Unstacking properties...")
            df = df.unstack()
            df.columns = df.columns.droplevel()

        if ('relativeHumidity' in df.columns) and ('roomTempCO2' in df.columns):
            logging.info("Swapping relativeHumidity and roomTemp2...")
            df.rename(columns = {'relativeHumidity':'roomTemp2', 'roomTempCO2':'relativeHumidity'}, inplace = True)

        logging.info("Changing column types...")
        df = df.astype({k:property_mapping_dict[k] for k in property_mapping_dict.keys() if k in df.columns})
        if 'listRSSI' in df.columns:
            df.loc[df['listRSSI'] == 'nan', 'listRSSI'] = ''

        # line below is needed to enable writing these DataFrames to parquet
        df.columns = df.columns.astype('string')

        return df

    @staticmethod
    def get_interpolated_weather_nl(first_day:datetime, last_day:datetime, 
                                    lat:float, lon:float, 
                                    tz_source:str, tz_home:str, int_intv:str) -> pd.DataFrame:
        """
        get weather data using a linear geospatial interpolation 
        based on three nearby KNMI weather stations 
        using the GitHub repo https://github.com/stephanpcpeters/HourlyHistoricWeather 
        for KNMI metrics=['T', 'FH', 'Q'] 
        with NO outlier removal (assuming that KNMI already did this)
        interpolated with the int_intv
        rendered as a dataframe with a timezone-aware datetime index
        columns ['T_out_avg_C', 'wind_avg_m_p_s', 'irradiation_hor_avg_W_p_m2', 'T_out_e_avg_C']
        """
        
        up = '15min'
        
        largest_measurement_interval = timedelta(hours=1)
        extractor_starttime = first_day
        logging.info('weather_extractor_starttime: ', extractor_starttime)
        extractor_endtime = last_day
        logging.info('weather_extractor_endtime: ', extractor_endtime)

        
        df = historicdutchweather.get_local_weather(extractor_starttime, extractor_endtime, lat, lon, metrics=['T', 'FH', 'Q'])
       

        logging.info('Resampling weather data...' )

        outdoor_T_interpolated = WeatherExtractor.get_weather_parameter_timeseries_mean(df, 'T', 'T_out_avg_C', 
                                                                                           up, int_intv, 
                                                                                           tz_source, tz_home)
        windspeed_interpolated = WeatherExtractor.get_weather_parameter_timeseries_mean(df, 'FH', 'wind_avg_m_p_s', 
                                                                                        up, int_intv, 
                                                                                        tz_source, tz_home)
        irradiation_interpolated = WeatherExtractor.get_weather_parameter_timeseries_mean(df, 'Q', 'irradiation_hor_J_p_h_p_cm2_avg', 
                                                                                          up, int_intv, 
                                                                                          tz_source, tz_home)


        # merge weather data in a single dataframe
        df = pd.concat([outdoor_T_interpolated, windspeed_interpolated, irradiation_interpolated], axis=1, join='outer') 
        
        df['irradiation_hor_avg_W_p_m2'] = df['irradiation_hor_J_p_h_p_cm2_avg']  * (100 * 100) / (60 * 60)
        df = df.drop('irradiation_hor_J_p_h_p_cm2_avg', axis=1)

        #oddly enough the column contains values that are minutely negative, which look weird and are impossible; hence: replace
        df.loc[(df.irradiation_hor_avg_W_p_m2 < 0), 'irradiation_hor_avg_W_p_m2'] = 0

        #calculate effective outdoor temperature based on KNMI formula
        df['T_out_e_avg_C'] = df['T_out_avg_C'] - 2/3 * df['wind_avg_m_p_s'] 
        
       
        return df

    @staticmethod
    def get_weather_parameter_timeseries_mean(df: pd.DataFrame, parameter: str, seriesname: str, 
                                 up_to: str, int_intv: str,
                                 tz_source:str, tz_home: str) -> pd.DataFrame:

        df = pd.DataFrame(df[parameter])
        logging.info(df)
        # df.set_index('datetime', inplace=True)

        if not(tz_source == tz_home):
            df = df.tz_convert(tz_home)

        #first sort on datetime index
        df.sort_index(inplace=True)
        # tempting, but do NOT drop duplicates since this ignores index column
        # df.drop_duplicates(inplace=True)

        # Converting str to float needed
        df[parameter] = df[parameter].astype(float)

        df = df.resample(up_to).first()
        df.interpolate(method='time', inplace=True)

        df.rename(columns={parameter:seriesname}, inplace=True)
        df = df.resample(int_intv).mean()
        
        return df
    
    @staticmethod
    def remove_weather_outliers(df: pd.DataFrame, columns, n_std) -> pd.DataFrame:
        """
        Simple procedure to replace outliers in the [col] columns with NaN
        Where outliers are those values more than n_std standard deviations away from the average of the [col] columns in a dataframe
        """
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            df[(df[col]-mean).abs() > (n_std*std)] = np.nan
            
        return df