import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
from tqdm.notebook import tqdm
from sqlalchemy import create_engine, text
import logging

class Measurements:
    """
    Use this class to get data from the database that contains measurements.
    """
        
    @staticmethod    
    def get_property_ids(property_types = None) -> str:
        """
        gets a list of property_ids to speed up the query
        """

        db_url_env = os.getenv("TWOMES_DB_URL")
        assert db_url_env, 'Environment variable TWOMES_DB_URL not set. Format: user:pass@host:port/db '

        db = create_engine("mysql+mysqlconnector://"+db_url_env)
        
        sql_query = "SELECT id FROM property WHERE name IN "+ str(tuple(property_types.keys()))
        print(sql_query)

        df = pd.DataFrame()
        df = pd.read_sql(sql=sql_query, con=db.connect().execution_options(stream_results=True))
        
        return str(tuple(df['id']))

        
    @staticmethod    
    def get_raw_measurements(ids,
                             first_day:datetime=None, last_day:datetime=None,
                             db_properties = None, property_rename = None,
                             tz_source:str = 'UTC', tz_building:str = 'Europe/Amsterdam') -> pd.DataFrame:
        
        """
        in: 
        - ids: list of ids (aka account.pseudonyms in the twomes database)
        - first_day: timezone-aware date
        - last_day: , timezone-aware date; data is extracted until end of day
        - db_properties: list of properties to retrieve from database
        out: dataframe with measurements
        - result.index = ['id', 'device_name', 'source', 'timestamp', 'property']
        -- id: id of e.g. home / utility building / room 
        -- source: device_type from the database
        -- timestamp: timezone-aware timestamp
        - columns = ['value', 'unit']:
        """
        db_url_env = os.getenv("TWOMES_DB_URL")
        assert db_url_env, 'Environment variable TWOMES_DB_URL not set. Format: user:pass@host:port/db '

        db = create_engine("mysql+mysqlconnector://"+db_url_env)

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

        sql_query = """
        SELECT
            m.timestamp AS timestamp,
            a.pseudonym AS id,
            d.name AS device_name,
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

        match len(ids):
            case 0:
                logging.warning('empty list of ids')
            case 1:
                sql_query = sql_query + " WHERE a.pseudonym = "+ str(ids[0])
            case _:
                sql_query = sql_query + " WHERE a.pseudonym IN "+ f'{tuple(map(str, ids))}'        
        
        match len(db_properties):
            case 0: 
                logging.warning('empty list of property names')
            case 1:
                sql_query_properties = "SELECT id FROM property WHERE name = '"+ db_properties[0] + "'"
                df_properties = pd.read_sql(sql=text(sql_query_properties), con=db.connect())
                logging.info(f'first_day: {sql_query_properties}')
            case _:
                sql_query_properties = "SELECT id FROM property WHERE name IN "+ str(tuple(db_properties))
                df_properties = pd.read_sql(sql=text(sql_query_properties), con=db.connect())
                logging.info(f'first_day: {sql_query_properties}')
            
        match len(df_properties.index):
            case 0:
                logging.warning('empty list of properties found')
            case 1:
                sql_query = sql_query + " AND p.id = "+ str(df_properties['id'].iloc[0])
            case _:
                sql_query = sql_query + " AND p.id IN "+ str(tuple(df_properties['id']))

        if first_day is not None: 
            sql_query = sql_query + " AND m.timestamp >= "+ first_str

        if last_day is not None: 
            sql_query = sql_query + " AND m.timestamp <= "+ last_str 

        logging.info(sql_query.replace('\n',' '))

        df = pd.DataFrame()

        #TODO: react on tz_source, depending on whether tz_source == 'UTC'. 
        for chunk in tqdm(pd.read_sql(sql=text(sql_query.replace('\n',' ')),
                                               con=db.connect().execution_options(stream_results=True),
                                               chunksize = 2000000,
                                               parse_dates={"timestamp": {"utc": "True"}}
                                              )
                                       ):
            df = pd.concat([df,chunk.astype({'id':'category',
                                             'device_name':'category',
                                             'device_type':'category',
                                             'property':'category',
                                             'unit':'category'
                                            }
                                           )
                           ]
                          )
        
        df = (df
                .drop_duplicates(subset=['id', 'timestamp','device_type', 'device_name', 'property', 'value'], keep='first')
                .rename(columns = {'device_type':'source'}) 
                .set_index(['id', 'device_name', 'source', 'timestamp', 'property'])
                .tz_convert(tz_building, level='timestamp')
               )
        if property_rename is not None:
            return df.rename(index=property_rename).sort_index()

        else:
            return df.sort_index()

    @staticmethod    
    def get_needforheat_measurements(ids,
                                     first_day:datetime=None, last_day:datetime=None,
                                     db_properties = None,
                                     tz_source:str = 'UTC', tz_building:str = 'Europe/Amsterdam') -> pd.DataFrame:
        
        """
        in: 
        - ids: list of account ids
        - first_day: timezone-aware date
        - last_day: , timezone-aware date; data is extracted until end of day
        - db_properties: list of properties to retrieve from database
        out: dataframe with measurements
        - result.index = ['id', 'device_name', 'source', 'timestamp', 'property']
        -- id: id of e.g. home / utility building / room 
        -- source: device_type from the database
        -- timestamp: timezone-aware timestamp
        - columns = ['value']:
        """
        db_url_env = os.getenv("TWOMES_DB_URL")
        assert db_url_env, 'Environment variable TWOMES_DB_URL not set. Format: user:pass@host:port/db '

        db = create_engine("mysql+mysqlconnector://"+db_url_env)

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

        sql_query = """
        SELECT
            m.time AS timestamp,
            a.id AS id,
            d.name AS device_name,
            dt.name AS device_type,
            p.name AS property,
            m.value AS value
        FROM
            measurement m
        JOIN upload u ON
            u.id = m.upload_id
        JOIN property p ON
            p.id = m.property_id
        JOIN device d ON
            d.id = u.device_id
        JOIN device_type dt ON
            dt.id = d.device_type_id
        JOIN building b ON
            b.id = d.building_id
        JOIN account a on
            a.id = b.account_id
        JOIN campaign c ON
            a.campaign_id = c.id"""

        match len(ids):
            case 0:
                logging.warning('empty list of ids')
            case 1:
                sql_query = sql_query + " WHERE a.id = "+ str(ids[0])
            case _:
                sql_query = sql_query + " WHERE a.id IN "+ f'{tuple(map(str, ids))}'        
        
        match len(db_properties):
            case 0:
                logging.warning('empty list of properties found')
            case 1:
                sql_query = sql_query + " AND p.name = '"+ str(db_properties[0]) + "'"
            case _:
                sql_query = sql_query + " AND p.name IN "+ str(tuple(db_properties)) 

        if first_day is not None: 
            sql_query = sql_query + " AND m.time >= "+ first_str

        if last_day is not None: 
            sql_query = sql_query + " AND m.time <= "+ last_str 

        logging.info(sql_query.replace('\n',' '))

        df = pd.DataFrame()

        #TODO: react on tz_source, depending on whether tz_source == 'UTC'. 
        for chunk in tqdm(pd.read_sql(sql=text(sql_query.replace('\n',' ')),
                                               con=db.connect().execution_options(stream_results=True),
                                               chunksize = 2000000,
                                               parse_dates={"timestamp": {"utc": "True"}}
                                              )
                                       ):
            df = pd.concat([df,chunk.astype({'id':'category',
                                             'device_name':'category',
                                             'device_type':'category',
                                             'property':'category'
                                            }
                                           )
                           ]
                          )
        
        #TODO: handle errors when dataframa is empty
        #TODO: handle campaigns where timezone may be different per building
        df = (df
                .drop_duplicates(subset=['id', 'timestamp','device_type', 'device_name', 'property', 'value'], keep='first')
                .rename(columns = {'device_type':'source'}) 
                .set_index(['id', 'device_name', 'source', 'timestamp', 'property'])
                .tz_convert(tz_building, level='timestamp')
               )
        return df.sort_index()

        
    @staticmethod    
    def to_properties(df_meas, properties_types = None) -> pd.DataFrame:
        
        """
        in: dataframe with measurements
        - index = ['id', 'device_name', 'source', 'timestamp', 'property']
        -- id: id of e.g. home / utility building / room 
        -- source: device_type from the database
        -- timestamp: timezone-aware timestamp
        - columns = ['value', 'unit']:
        - properties_types: disctionary that specifies per property (key) which type to apply (value) 
        
        for duplicate measurements (index the same) only the first value is saved
        properties are unstacked into columns
        the unit column is dropped
        the device name is dropped
        types are applied per column as specified in 
        
        out: dataframe with
        - result.index = ['id', 'source', 'timestamp', 'property']
        -- id: id of e.g. home / utility building / room 
        -- source: device_type from the database
        -- timestamp: timezone-aware timestamp
        - columns = all properties in the input column
        
        """
        measurement_count = df_meas.shape[0]
        df_prop = (df_meas
                   .reset_index()
                   [['id', 'timestamp','source', 'property', 'value']]
                   # duplicate values for exacly the same time cannot be unstacked we drop the later values
                   .drop_duplicates(subset=['id', 'timestamp','source', 'property'], keep='first')
                   .set_index(['id', 'source', 'timestamp', 'property'])
                  )
        logging.info(f'Dropped {measurement_count - df_prop.shape[0]} measurements for unstacking')
        df_prop = df_prop.unstack()
        df_prop.columns = df_prop.columns.droplevel()
        
        return df_prop.astype({k:properties_types[k] for k in properties_types.keys() if k in df_prop.columns})
    
    @staticmethod    
    def get_accounts_devices(first_day:datetime=None, last_day:datetime=None,
                             tz_source:str = 'UTC', tz_building:str = 'Europe/Amsterdam') -> pd.DataFrame:
        
        """
        in: 
        - first_day: timezone-aware date
        - last_day: , timezone-aware date; data is extracted until end of day
        out: dataframe with accounts and devices that provided data
        """
        db_url_env = os.getenv("TWOMES_DB_URL")
        assert db_url_env, 'Environment variable TWOMES_DB_URL not set. Format: user:pass@host:port/db '

        db = create_engine("mysql+mysqlconnector://"+db_url_env)
        
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

        sql_query = """
            SELECT
                a.pseudonym AS account_id,
                d.id AS device_id,
                d.name AS device_name,
                m.timestamp AS latest_timestamp_UTC,
                p.name AS property,
                m.value,
                p.unit
            FROM
                measurement m
            JOIN device d ON
                m.device_id = d.id
            JOIN property p ON
                m.property_id = p.id
            JOIN building b ON
                d.building_id = b.id
            JOIN account a ON
                b.account_id = a.id
            WHERE
                device_id BETWEEN 114 AND 121
                AND (device_id,
                property_id,
                timestamp) IN (
                SELECT
                device_id,
                property_id,
                MAX(timestamp)
                FROM
                measurement
                WHERE
                device_id BETWEEN 114 AND 121
                AND timestamp > '2022-01-01 0:00'
                GROUP BY
                device_id,
                property_id
                )
            ORDER BY
            property,
            device_name,
            latest_timestamp_UTC DESC"""

        logging.info(sql_query.replace('\n',' '))

        df = pd.DataFrame()

        #TODO: react on tz_source, depending on whether tz_source == 'UTC'. 
        for chunk in tqdm(pd.read_sql(sql=text(sql_query.replace('\n',' ')),
                                               con=db.connect().execution_options(stream_results=True),
                                               chunksize = 2000000,
                                               parse_dates={"timestamp": {"utc": "True"}}
                                              )
                                       ):
            df = pd.concat([df, chunk])
        
        return df
        
        
    @staticmethod
    def get_interpolated_weather_nl(first_day:datetime, last_day:datetime, 
                                    lat:float, lon:float, 
                                    tz_source:str, tz_building:str, int_intv:str) -> pd.DataFrame:
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
                                                                                           tz_source, tz_building)
        windspeed_interpolated = WeatherExtractor.get_weather_parameter_timeseries_mean(df, 'FH', 'wind_avg_m_p_s', 
                                                                                        up, int_intv, 
                                                                                        tz_source, tz_building)
        irradiation_interpolated = WeatherExtractor.get_weather_parameter_timeseries_mean(df, 'Q', 'irradiation_hor_J_p_h_p_cm2_avg', 
                                                                                          up, int_intv, 
                                                                                          tz_source, tz_building)


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
                                 tz_source:str, tz_building: str) -> pd.DataFrame:

        df = pd.DataFrame(df[parameter])
        logging.info(df)
        # df.set_index('datetime', inplace=True)

        if not(tz_source == tz_building):
            df = df.tz_convert(tz_building)

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