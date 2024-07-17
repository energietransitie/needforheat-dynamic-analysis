import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from tqdm.notebook import tqdm
from sqlalchemy import create_engine, text
import logging
import historicdutchweather
from urllib.error import HTTPError
from pandas.errors import ParserError
import requests
import io
import re
from pytz import NonExistentTimeError
from scipy.interpolate import RBFInterpolator


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
        
        if not df.empty:
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
        
        #TODO: handle campaigns where timezone may be different per building
        if not df.empty:
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

        # converted_columns = {}

        # for k in properties_types.keys():
        #     if k in df_prop.columns and k.endswith('__bool') and properties_types[k] in ['bool', 'boolean']:
        #         converted_columns[k] = df_prop[k].replace({'0': False, '1': True})
        
        # # Apply the conversions
        # for k, v in converted_columns.items():
        #     df_prop[k] = v
        
        return df_prop.astype({k:properties_types[k] for k in properties_types.keys() if k in df_prop.columns})
    
    @staticmethod    
    def to_properties_with_source_category_and_type(df_meas, properties_types = None) -> pd.DataFrame:
        
        """
        in: dataframe with measurements
        - index = ['id', 'device_name', 'source', 'timestamp', 'property']
        -- id: id of e.g. home / utility building / room 
        -- source_category:e.g. batch_import, device, clould_feed, energy_query
        -- source_type: e.g. device_type from the database
        -- timestamp: timezone-aware timestamp
        - columns = ['value']:
        - properties_types: disctionary that specifies per property (key) which type to apply (value) 
        
        for duplicate measurements (index the same) only the first value is saved
        properties are unstacked into columns
        types are applied per column as specified in 
        
        out: dataframe with
        - result.index = ['id', 'source', 'timestamp', 'property']
        -- id: id of e.g. home / utility building / room 
        -- source_category:e.g. batch_import, device, clould_feed, energy_query
        -- source_type: e.g. device_type from the database
        -- timestamp: timezone-aware timestamp
        - columns = all properties in the input column
        
        """
        measurement_count = df_meas.shape[0]
        df_prop = (df_meas
                   .reset_index()
                   [['id', 'timestamp', 'source_category', 'source_type', 'property', 'value']]
                   # duplicate values for exacly the same time cannot be unstacked we drop the later values
                   .drop_duplicates(subset=['id', 'timestamp', 'source_category', 'source_type', 'property'], keep='first')
                   .set_index(['id', 'source_category', 'source_type', 'timestamp', 'property'])
                  )
        logging.info(f'Dropped {measurement_count - df_prop.shape[0]} measurements for unstacking')
        df_prop = df_prop.unstack()
        df_prop.columns = df_prop.columns.droplevel()

        # Convert string representations to boolean values for boolean columns
        for col in df_prop.columns:
            if col.endswith('__bool'):
                df_prop[col] = df_prop[col].map({'True': True, 'False': False})
    
        # # Replace <NA> values with pd.NA for nullable boolean columns
        # for col in df_prop.columns:
        #     if col.endswith('__bool'):
        #         df_prop[col] = df_prop[col].replace(pd.NA, pd.NA)
    
        # Convert columns to specified types
        if properties_types:
            for col, dtype in properties_types.items():
                if col in df_prop.columns:
                    df_prop[col] = df_prop[col].astype(dtype)
        
        return df_prop

    
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

class WeatherMeasurements:


    @staticmethod
    def download_knmi_uurgegevens(start__YYYYMMDD, end_YYYYMMDD, metrics=['T', 'FH', 'Q']):
        # KNMI API endpoint
        KNMI_API_URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
        
        # NB For a future version of these functions, you may also need an  API key for KNMI and put it in a file with the name below and one line KNMI_API_KEY=your_KNMI_API_key 
        knmi_api_keys_file='knmi_api_key.txt'
        # If your organistion does not have one yet, request one here: https://developer.dataplatform.knmi.nl/open-data-api#token
        base_url = KNMI_API_URL
        params = {
            'start': start__YYYYMMDD+'01',
            'end': end_YYYYMMDD+'24',
            'vars': ':'.join(metrics)
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching data from KNMI: {response.text}")
        return response.text
    
    @staticmethod
    def process_knmi_weather_data(raw_data):
       # Split raw data by lines
        lines = raw_data.splitlines()
        
    
        # Ignore the first 5 lines
        lines = lines[5:]
    
        # Extract station info
        station_lines = [lines[0].lstrip('# ')]
        header_found = False
        data_start_line = 0
    
    
        for i, line in enumerate(lines):
            if re.match(r'^# \d{3}', line):
                station_lines.append(line.lstrip('# '))
            elif line.startswith('# YYYYMMDD'):
                continue
            elif line.startswith('# STN,YYYYMMDD'):
                header_found = True
            elif header_found:
                data_start_line = i
                break
    
    
        # Create station DataFrame
        station_data = "\n".join(station_lines)
    
        df_stations = pd.read_fwf(io.StringIO(station_data))
        df_stations.columns = df_stations.columns.str.replace(r'\(.*\)', '', regex=True).str.strip()
        df_stations = df_stations.set_index(['STN'])
        
        df_weather_chunk = pd.read_csv(io.StringIO(raw_data), skiprows=data_start_line+4, delimiter=',')    
    
        # Rename columns
        df_weather_chunk.columns = [col.replace('#', '').strip() for col in df_weather_chunk.columns]
        
        # Parse timestamp
        df_weather_chunk['timestamp'] = pd.to_datetime(df_weather_chunk['YYYYMMDD'].astype(str) + df_weather_chunk['HH'].astype(int).sub(1).astype(str).str.zfill(2), format='%Y%m%d%H')
        
        # Localize to UTC
        df_weather_chunk['timestamp'] = df_weather_chunk['timestamp'].dt.tz_localize('UTC')
    
        df_weather_chunk.drop(columns=['YYYYMMDD', 'HH'], inplace=True)
    
        # drop rows where timestamps are NaT (Not a Time)
        df_weather_chunk = df_weather_chunk.dropna(subset=['timestamp'])
    
        df_weather_chunk = df_weather_chunk.set_index(['STN', 'timestamp'])

        df_weather_chunk = df_weather_chunk.merge(df_stations[['LON', 'LAT']], left_on='STN', right_index=True, how='left')

        # Set the multi-index with lat, lon, and timestamp
        df_weather_chunk = df_weather_chunk.reset_index()

        # Drop the station identifier from the data
        df_weather_chunk = df_weather_chunk.drop(columns=['STN'])
        
        # Rename columns
        df_weather_chunk = df_weather_chunk.rename(columns={'LAT': 'lat__degN', 'LON': 'lon__degE'})
        df_weather_chunk = df_weather_chunk.set_index(['lat__degN', 'lon__degE', 'timestamp']).sort_index()

        # Drop rows with missing values 
        df_weather_chunk = df_weather_chunk.dropna()
    
        return df_weather_chunk

    
    @staticmethod
    def fetch_weather_data(time_interval, 
                           chunk_freq="4W", 
                           metrics={'T': ('temp_in__degC', 0.1), # H Temperature (in 0.1 degrees Celsius) at 1.50 m at the time of observation
                                    'FH': ('wind__m_s_1', 0.1), # FH: Hourly mean wind speed (in 0.1 m/s)
                                    'Q': ('ghi__W_m_2', (100 * 100) / (60 * 60)) # Q: Global radiation (in J/cm^2) during the hourly division, 1 m^2 = 100 cm/m^2 * 100 cm/m^2, 1 h = 60 min/h * 60 s/min 
                                   }
                          ):
        """
        Fetch and process weather data in chunks over a specified period.
        
        Parameters:
        time_interval (pd.Interval): Closed interval for the data collection period.
        chunk_freq (str): Frequency for dividing the data collection period into chunks.
        metrics (dict): Dictionary of metrics with a tuple specifying a property name following the physiquant__unit naming convention and conversion factor.
        
        Returns:
        pd.DataFrame: Processed weather data with multi-index of latitude, longitude, and timestamp. The timezone of time_interval.left.tzinfo (if any) is used as the target timezone for the timestamps.
        """
        df_weather = pd.DataFrame()

        # Ensure the start date is included in the first chunk
        target__tz = time_interval.left.tzinfo
        start_date = time_interval.left.tz_convert('UTC').normalize()
        end_date = time_interval.right.tz_convert('UTC').normalize() + pd.Timedelta(days=1)
        first_chunk_start = pd.date_range(start=start_date, end=end_date, freq=chunk_freq)[0]
        first_chunk_start = first_chunk_start if start_date >= first_chunk_start else first_chunk_start - pd.Timedelta(chunk_freq)

        # Iterate over date ranges using the specified frequency
        for current_start in tqdm(pd.date_range(start=first_chunk_start, end=end_date, freq=chunk_freq)):
            current_end = min(end_date, current_start + pd.Timedelta(chunk_freq) - timedelta(seconds=1))
            current_start = max(start_date, current_start)

            raw_data = WeatherMeasurements.download_knmi_uurgegevens(current_start.strftime('%Y%m%d'),
                                                                     current_end.strftime('%Y%m%d'),
                                                                     metrics.keys()
                                                                    )
            try:
                df_weather_chunk = WeatherMeasurements.process_knmi_weather_data(raw_data)
                
            except pd.errors.ParserError:
                print(f"Parsing raw data with start {current_start} and end {current_end} gives ParserError; date: {raw_data}")
                continue

            # Convert all columns to numeric, coercing errors to NaN
            df_weather_chunk = df_weather_chunk.apply(pd.to_numeric, errors='coerce')

            # Apply property renaming and conversion factors
            columns_to_drop = []
            for metric, (new_name, conversion_factor) in metrics.items():
                if metric in df_weather_chunk.columns:
                    if conversion_factor is not None:
                        df_weather_chunk[new_name] = df_weather_chunk[metric] * conversion_factor
                    else:
                        df_weather_chunk[new_name] = df_weather_chunk[metric
                        ]
                # Mark original column for dropping if the new name is different
                if new_name != metric:
                    columns_to_drop.append(metric)
            
            # Remove original metric columns if they were renamed
            df_weather_chunk = df_weather_chunk.drop(columns=columns_to_drop)
        
            # Append chunk data to df_weather
            df_weather = pd.concat([df_weather, df_weather_chunk])

        # Cleanup and final formatting

        df_weather = df_weather.reset_index()
        df_weather = df_weather.dropna()
        df_weather = df_weather.drop_duplicates()
        
        # Convert the 'timestamp' column to the target timezone, if any
        if target__tz is not None:
            df_weather['timestamp'] = df_weather['timestamp'].dt.tz_convert(target__tz)

        df_weather = df_weather.set_index(['timestamp', 'lat__degN', 'lon__degE']).sort_index()

        return df_weather
        
    
    @staticmethod
    def interpolate_weather_data(df_weather, df_homes):
        """
        Interpolate weather data to home locations using scipy.interpolate.RBFInterpolator.
        
        Parameters:
        - df_weather (pd.DataFrame): DataFrame containing weather data with multi-index ['lat__degN', 'lon__degE', 'timestamp'].
        - df_homes (pd.DataFrame): DataFrame containing index ['id'] and the weather location of the home in columns 'weather_lat__degN', 'weather_lon__degE'.
        
        Returns:
        - pd.DataFrame: Interpolated weather data with multi-index ['id', 'source_category', 'source_type', 'timestamp', 'property'].
        """
        
        # Prepare the output DataFrame
        interpolated_data = []

        df_weather = df_weather.reorder_levels(['timestamp', 'lat__degN', 'lon__degE']).sort_index()
        
       
        # Iterate over each timestamp
        for timestamp in tqdm(df_weather.index.get_level_values('timestamp').unique()):
            df_timestamp = df_weather.xs(timestamp, level='timestamp')
            lat_lon = df_timestamp.index.values
            lat_lon_array = np.array([[lat__degN, lon__degE] for lat__degN, lon__degE in lat_lon])
            
            for metric in df_timestamp.columns:
                values = df_timestamp[metric].values
        
                # Set up the interpolator
                interpolator = RBFInterpolator(lat_lon_array, values)
        
                # Perform interpolation for each home location
                for home_id, home_data in df_homes.iterrows():

                    # Create an array of the weather location coordinates
                    weather_coords = np.array([[home_data['weather_lat__degN'], home_data['weather_lon__degE']]])
 
                    interpolated_value = interpolator(weather_coords)

                    # Convert the interpolated value to Float32
                    interpolated_value = float(interpolated_value.astype('float32'))
                    
                    # Append the interpolated value to the results list
                    interpolated_data.append({
                        'id': home_id,
                        'source_category': 'batch_import',
                        'source_type': 'KNMI',
                        'timestamp': timestamp,
                        'property': metric,
                        'value': interpolated_value
                    })
        
        # Convert the results list to a DataFrame
        df_meas_weather = pd.DataFrame(interpolated_data)
        
        # Set the appropriate multi-index
        df_meas_weather.set_index(['id', 'source_category', 'source_type', 'timestamp', 'property'], inplace=True)
        
        return df_meas_weather 


    @staticmethod
    def get_weather_measurements(df_weather_locations, weather_min_timestamp, weather_max_timestamp):
        """
        Retrieve and process KNMI weather data for given locations and timestamps, and compile it into a cumulative dataframe.
        
        Parameters:
        df_weather_locations (pd.DataFrame): DataFrame containing weather location data with a home id for index and 'weather_lat__degN' and 'weather_lon__degE' columns.
        weather_min_timestamp (pd.Timestamp): Minimum timestamp for weather data retrieval.
        weather_max_timestamp (pd.Timestamp): Maximum timestamp for weather data retrieval.
        
        Returns:
        pd.DataFrame: Cumulative DataFrame with weather measurements.
        """
        new_column_names = {'T': 'temp_in__degC', 'FH': 'wind__m_s_1', 'Q': 'ghi__J_h_1_cm_2'}
        
        # Initialize an empty DataFrame to store cumulative weather measurements
        df_meas_weather = pd.DataFrame()
        
        # Get unique weather locations
        weather_locations = df_weather_locations[['weather_lat__degN', 'weather_lon__degE']].drop_duplicates()
        
        for lat, lon in tqdm(weather_locations.values):
            try:
                # Extractor start and end time
                extractor_starttime = weather_min_timestamp - pd.Timedelta(hours=1)
                extractor_endtime = weather_max_timestamp - pd.Timedelta(hours=1)
                
                # Retrieve weather data
                df_weather = historicdutchweather.get_local_weather(
                    extractor_starttime, extractor_endtime, lat, lon, metrics=['T', 'FH', 'Q']
                )
                
                # Transform the DataFrame to fit the desired format
                df_weather.rename(columns=new_column_names, inplace=True)
                df_weather = df_weather.stack().reset_index()
                df_weather.columns = ['timestamp', 'property', 'value']
                
                # Find home_ids with this weather location
                home_ids = df_weather_locations[
                    (df_weather_locations.weather_lat__degN == lat) & 
                    (df_weather_locations.weather_lon__degE == lon)
                ].index
                
                for home_id in home_ids:
                    # Set id and other columns for df_weather
                    df_weather_home = df_weather.copy()
                    df_weather_home['id'] = home_id
                    df_weather_home['source_category'] = 'batch_import'
                    df_weather_home['source_type'] = 'KNMI'
                    df_weather_home.set_index(['id', 'source_category', 'source_type', 'timestamp', 'property'], inplace=True)
                    
                    # Concatenate to cumulative df_meas_weather
                    df_meas_weather = pd.concat([df_meas_weather, df_weather_home])
                    
            except HTTPError as e:
                print(f"HTTP error {e.code} for lat {lat}, lon {lon}. Skipping...")
                continue
        
        return df_meas_weather
       
        
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