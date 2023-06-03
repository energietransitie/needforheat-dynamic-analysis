import pandas as pd
import numpy as np
from scipy import stats
from tqdm.notebook import tqdm
import pytz
from datetime import datetime, timedelta
from extractor import WeatherExtractor

class Preprocessor:
    """
    Use this class to get data from the database that contains measurements.
    """

    
#TODO:check if dynamic outliers removal done by a DataFrame mask based on a hampel filter
# (either before interpolation or after interpolation:
# hampel(df[prop], window_size=5, n=3, imputation=False)


    @staticmethod
    def filter_min_max(df: pd.DataFrame,
                       col:str,
                       min:float=None, max:float=None,
                       inplace=True
                      ) -> pd.DataFrame:
        
        """
        Replace outliers in the col column with NaN,
        where outliers are those values more below a minimum value or above a maximum value
        
        in: df: pd.DataFrame with
        - index = ['id', 'source', 'timestamp']
        -- id: id of the unit studied (e.g. home / utility building / room) 
        -- source: device_type from the database
        -- timestamp: timezone-aware timestamp
        - columns = properties with measurement values
        
       
        out: pd.DataFrame with same structure as df, 
        - 
       
        """
        
        if not len(df) or min is None and max is None:
            return df
        df_result = df
        if not inplace:
            df_result = df.copy(deep=True)
        if min is not None:
            df_result[col] = df_result[col].mask(df_result[col] < min)
        if max is not None:
            df_result[col] = df_result[col].mask(df_result[col] > max)            
        return df_result


    @staticmethod
    def filter_static_outliers(df: pd.DataFrame,
                               col:str,
                               n_sigma:float=3.0,
                               per_id=True,
                               inplace=True
                              ) -> pd.DataFrame:

        """
        Simple procedure to replace outliers in the 'value' column with NaN
        Where outliers are those values more than n_sigma standard deviations away from the average of the property 
        column in a dataframe
        """
        
        if (not len(df)
            or
            (col not in df.columns)):
            return df
        df_result = df
        if not inplace:
            df_result = df.copy(deep=True)
        if per_id:
            for id in df.index.unique(level='id'):
                df_result.loc[id, col] = (df_result
                                          .loc[id][col]
                                          .mask(stats.zscore(df_result.loc[id][col], nan_policy='omit').abs() > n_sigma)
                                         .values)            
        else:
            df_result[col] = (df_result[col]
                              .mask(stats.zscore(df_result[col], nan_policy='omit').abs() > n_sigma)
                              .values)            
        return df_result

    @staticmethod
    def unstack_prop(df_prop: pd.DataFrame) -> pd.DataFrame:
        
        """
        Unstack a DataFrame resamplingto interpolate__min minutes and linear interpolation,
        while making sure not to bridge gaps larger than limin__min minutes.
        The final dataframe has column datatypes as indicated in the property_types dictionary
        
        in: df_prop: pd.DataFrame with
        - index = ['id', 'source', 'timestamp']
        -- id: id of the unit studied (e.g. home / utility building / room) 
        -- source: device_type from the database
        -- timestamp: timezone-aware timestamp
        - columns = properties with measurement values
       
        out: pd.DataFrame with the source names prefixed to column names 
       
        """
        df_prep = df_prop.unstack([1])
        df_prep.columns = df_prep.columns.swaplevel(0,1)
        df_prep.columns = ['_'.join(col) for col in df_prep.columns.values]
        
        return df_prep

    @staticmethod
    def interpolate_time(df_prop: pd.DataFrame,
                         property_dict = None,
                         upsample__min = 5,
                         interpolate__min = 15,
                         limit__min = 60,
                         inplace=False
                        ) -> pd.DataFrame:
        
        """
        Interpolate a DataFrame by resampling first to upsample_min intervals,
        then interpolating to interpolate__min minutes using linear interpolation,
        while making sure not to bridge gaps larger than limin__min minutes,
        then resampling using interpolate__min
        The final dataframe has column datatypes as indicated in the property_types dictionary
        
        in: df_prop: pd.DataFrame with
        - index = ['id', 'source', 'timestamp']
        -- id: id of the unit studied (e.g. home / utility building / room) 
        -- source: device_type from the database
        -- timestamp: timezone-aware timestamp
        - columns = properties with measurement values
       
        out: pd.DataFrame with same structure as df_prop 
       
        """
        lim = (limit__min - 1) // upsample__min
        df_result = pd.DataFrame()
        for id in tqdm(df_prop.index.unique('id').dropna()):
            for source in df_prop.loc[id].index.unique('source').dropna():
                df_interpolated = df_prop.loc[id, source,:]
                if not len(df_interpolated):
                    continue
                if not inplace:
                    df_interpolated = df_interpolated.copy(deep=True)
                df_interpolated = (df_interpolated
                             .resample(str(upsample__min) + 'T')
                             .first()
                             .astype('float32')
                             .interpolate(method='time', limit=lim)
                             .resample(str(interpolate__min) + 'T').mean()
                            )
                df_interpolated['id'] = id
                df_interpolated['source'] = source
                df_result = pd.concat([df_result, df_interpolated.reset_index().set_index(['id','source','timestamp'])])
        
        df_result = df_result.sort_index()                  
        for col in df_result.columns:
            # match property_dict[col]:
            #     case 'int'| 'Int8' | 'Int16' | 'Int32'| 'Int64' | 'UInt8' | 'UInt16' | 'UInt32' | 'UInt64':
            #         df_result[col] = df_result[col].round(0).astype(property_dict[col])
            #     case 'float' | 'float32' | 'float64':
            #         df_result[col] = df_result[col].astype(property_dict[col])
            if property_dict[col] in ['int', 'Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64']:
                    df_result[col] = df_result[col].round(0).astype(property_dict[col])
            elif property_dict[col] in ['float', 'float32', 'float64']:
                    df_result[col] = df_result[col].astype(property_dict[col])
        return df_result
    
    @staticmethod
    def preprocess_room_data(df_prop: pd.DataFrame,
                             lat, lon : float,
                             timezone_ids = 'Europe/Amsterdam'
                            ) -> pd.DataFrame:
        
        """
        Preprocess, iunstack and interpolate room data for the B4B project and add weather data.
        Filter co2_ppm: remove outliers below 5 ppm and co2 sensor data that does not vary in a room.
        
        in: df: pd.DataFrame with
        - index = ['id', 'source', 'timestamp']
        -- id: id of the room studied 
        -- source: device_type from the database
        -- timestamp: timezone-aware timestamp
        - columns = properties with measurement values, including at least co2__ppm
       
        out: pd.DataFrame  
        - unstacked: i.e. with the source names prefixed to column names
        - interpolated_min: interpolation interval with 15 minute as default
        - KNMI weather data for lat, lon merged 'temp_out__degC', 'wind__m_s_1', 'ghi__W_m_2'
               
        """
        
        # first, preprocess co2__ppm data
        prop = 'co2__ppm'
        df_prop = Preprocessor.filter_min_max(df_prop, prop, min=5)
        std = df_prop[prop].groupby(['id', 'source']).transform('std')
        # set values to np.nan where std is zero
        mask = std == 0
        df_prop[mask] = np.nan

        property_types = {
            'temp_in__degC' : 'float32',
            'co2__ppm' : 'float32',
            'rel_humidity__0' : 'float32',
            'valve_frac__0' : 'float32',
            'door_open__bool': 'Int8',
            'window_open__bool': 'Int8',
            'occupancy__bool': 'Int8',
            'occupancy__p' : 'Int8'
        }

        interpolate__min = 15
        
        df_interpolated = Preprocessor.interpolate_time(df_prop,
                                                        property_dict = property_types,
                                                        upsample__min = 5,
                                                        interpolate__min = interpolate__min,
                                                        limit__min = 90,
                                                        inplace=False
                                                       )
        
        df_prep = Preprocessor.unstack_prop(df_interpolated)

        return Preprocessor.merge_weather_data_nl(df_prep, lat, lon, interpolate__min, timezone_ids)


    
    @staticmethod
    def merge_weather_data_nl(df_prep: pd.DataFrame,
                           lat, lon : float,
                           interpolate__min = 15,
                           timezone_ids = 'Europe/Amsterdam'
                           ) -> pd.DataFrame:
        
        """
        Add weather data to a preprocessed properties DataFrame.
        
        in: df: pd.DataFrame with
        - index = ['id', 'timestamp']
        -- id: id of the room studied 
        -- timestamp: timezone-aware timestamp
        - columns = properties with measurement values
        interpolate__min = 15,
        lat, lon: latitude and logitude of the weather location
        timezone_ids = timezone of the objects, defaults to 'Europe/Amsterdam'
       
        out: pd.DataFrame with weather data attached
               
        """
        
        # get geospatially interpolated weather from KNMI
        # get the dataframe only once for all homes to save time

        tz_knmi='Europe/Amsterdam'

        # Extract earliest and latest timestamps
        earliest_timestamp = (df_prep.index.get_level_values('timestamp').min() + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
        latest_timestamp = (df_prep.index.get_level_values('timestamp').max() +  + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
        
        df_weather = WeatherExtractor.get_interpolated_weather_nl(
            earliest_timestamp, 
            latest_timestamp, 
            lat, lon, 
            tz_knmi, 
            timezone_ids, 
            str(interpolate__min) + 'T'
        ).rename_axis('timestamp')

        return df_prep.reset_index().merge(df_weather, on='timestamp').set_index(['id', 'timestamp']).sort_index()  

    
    def property_filter(df, parameter:str, prop:str, metertimestamp:str,
                        tz_source:str, tz_home:str,
                        process_meter_reading:bool, 
                        min_value:float, max_value:float, n_sigma:int, 
                        up_intv:str, gap_n_intv:int, 
                        summ_intv:str, summ) -> pd.DataFrame:

        # Type checking
        if not isinstance(summ, Summarizer):
            raise TypeError('summ parameter must be an instance of Summarizer Enum')

        if (df is not None and len(df.index)>0):
            logging.info('df.index[0]: ', df.index[0])
            logging.info('df.index[0].tzinfo: ', df.index[0].tzinfo)

        if metertimestamp is not None:
            #TODO: create new row for all e-meter values, replace index value by meter timestamp, remove meter values from from
            #TODO: perhaps we need to replace processing timestamps of meter reading values to before the unstacking?
            
            df_metertimestamp = self.get(metertimestamp)
            df_metertimestamp.set_index('datetime', inplace=True)
            df_metertimestamp.drop(['index', 'timestamp'], axis=1, inplace=True)
            if tz_source == tz_home:
                df_metertimestamp = df_metertimestamp.tz_localize(tz_source)
            else:
                df_metertimestamp = df_metertimestamp.tz_localize(tz_source).tz_convert(tz_home)

            logging.info(prop, 'df_metertimestamp before parsing YYMMDDhhmmssX values:', df_metertimestamp.head(25))
            logging.info(prop, 'df_metertimestamp.index[0]: ', df_metertimestamp.index[0])
            logging.info(prop, 'df_metertimestamp.index[0].tzinfo: ', df_metertimestamp.index[0].tzinfo)

            # parse DSMR TST value format: YYMMDDhhmmssX
            # meaning according to DSMR 5.0.2 standard: 
            # "ASCII presentation of Time stamp with Year, Month, Day, Hour, Minute, Second, 
            # and an indication whether DST is active (X=S) or DST is not active (X=W)."
            if df_metertimestamp['value'].str.contains('W|S', regex=True).any():
                logging.info(prop, 'parsing DSMR>v2 $S $W timestamps')
                df_metertimestamp['value'].replace(to_replace='W$', value='+0100', regex=True, inplace=True)
                logging.info(prop, 'df_metertimestamp after replace W:', df_metertimestamp.head(25))
                df_metertimestamp['value'].replace(to_replace='S$', value='+0200', regex=True, inplace=True)
                logging.info(prop, 'df_metertimestamp after replace S, before parsing:', df_metertimestamp.head(25))
                df_metertimestamp['meterdatetime'] = df_metertimestamp['value'].str.strip()
                logging.info(prop, 'df_metertimestamp after stripping, before parsing:', df_metertimestamp.head(25))
                df_metertimestamp['meterdatetime'] = pd.to_datetime(df_metertimestamp['meterdatetime'], format='%y%m%d%H%M%S%z', errors='coerce')
                logging.info(prop, df_metertimestamp[df_metertimestamp.meterdatetime.isnull()])
                df_metertimestamp['meterdatetime'] = df_metertimestamp['meterdatetime'].tz_convert(tz_home)
            else:
                logging.info(prop, 'parsing DSMR=v2 timestamps without $W $S indication')
                if df_metertimestamp['value'].str.contains('[0-9]', regex=True).any():
                    # for smart meters of type Kamstrup 162JxC - KA6U (DSMR2), there is no W or S at the end; timeoffset needs to be inferred
                    logging.info(prop, 'df_metertimestamp before parsing:', df_metertimestamp.head(25))
                    df_metertimestamp['meterdatetime'] = df_metertimestamp['value'].str.strip() 
                    logging.info(prop, 'df_metertimestamp after stripping, before parsing:', df_metertimestamp.head(25))
                    df_metertimestamp['meterdatetime'] = pd.to_datetime(df_metertimestamp['meterdatetime'], format='%y%m%d%H%M%S', errors='coerce')
                    logging.info(prop, df_metertimestamp[df_metertimestamp.meterdatetime.isnull()])
                    df_metertimestamp['meterdatetime'] = df_metertimestamp['meterdatetime'].tz_localize(None).tz_localize(tz_home, ambiguous='infer')
                else: # DSMRv2 did not speficy eMeterReadingTimestamps
                    df_metertimestamp['meterdatetime'] = df_metertimestamp.index 
                    
                logging.info(prop, 'df_metertimestamp after all parsing:', df_metertimestamp.head(25))



            # dataframe contains NaT values  
            logging.info('before NaT replacement:', df_metertimestamp[df_metertimestamp.meterdatetime.isnull()])

            # unfortunately, support for linear interpolation of datetimes is not properly implemented, so we do it via Unix time
            df_metertimestamp['meterdatetime']= df_metertimestamp['meterdatetime']\
                                                .apply(lambda x: np.nan if pd.isnull(x) else x.timestamp())\
                                                .interpolate(method='linear', limit=2)
            df_metertimestamp['meterdatetime'] = pd.to_datetime(df_metertimestamp['meterdatetime'], unit='s', utc=True,  origin='unix')\
                                                 .dt.tz_convert(tz_home)


            df_metertimestamp.reset_index(inplace=True)
            df_metertimestamp.set_index('datetime', inplace=True)
            df_metertimestamp.drop(['value'], axis=1, inplace=True)
            logging.info('df_metertimestamp after NaT replacement:', df_metertimestamp.head(25))
            logging.info('df_metertimestamp.index[0]: ', df_metertimestamp.index[0])
            logging.info('df_metertimestamp.index[0].tzinfo: ', df_metertimestamp.index[0].tzinfo)

            df = pd.concat([df, df_metertimestamp], axis=1, join='outer')
            logging.info('df:', df.head(25))
            logging.info('df.index[0]: ', df.index[0])
            logging.info('df.index[0].tzinfo: ', df.index[0].tzinfo)

            df.reset_index(inplace=True)
            df.drop(['datetime'], axis=1, inplace=True)
            df.rename(columns = {'meterdatetime':'datetime'}, inplace = True)
            logging.info('df after rename:', df.head(25))
            df.drop_duplicates(inplace=True)
            logging.info('df after dropping duplicates:', df.head(25))
            df.set_index('datetime', inplace=True)
            logging.info('df:', df.head(25))
            
        #first sort on datetime index
        df.sort_index(inplace=True)
        # tempting, but do NOT drop duplicates since this ignores index column, 
        # for meter readings, we already dropped duplicates earlier when both smart meter timestamp and smart meter reading were identical 

        # remove index, timestamp and value columns and rename column to prop
        logging.info('before rename:', df.head(25))
        df.rename(columns = {'value':prop}, inplace = True)

        # Converting str to float
        df[prop] = df[prop].astype(float)
        logging.info('after rename:', df.head(25))


        if process_meter_reading:
            
            #first, correct for occasional zero meter readings; this involves taking a diff and removing the meter reading that causes the negative jump
            logging.info('before zero meter reading filter:', df.head(25))
            df['diff'] = df[prop].diff()
            logging.info('after diff:', df.head(25))
            df = df[df['diff'] >= 0]
            df.drop(['diff'], axis=1, inplace=True)
            logging.info('after zero meter reading filter:', df.head(25))

            #then, correct for meter changes; this involves taking a diff and removing the negative jum and cumulating again
            #first, correct for potential meter changes
            df[prop] = df[prop].diff().shift(-1)
            logging.info('after diff:', df.head(25))
            df.loc[df[prop] < 0, prop] = 0
            logging.info('after filter negatives:', df.head(25))

            # now, cumulate again
            df[prop] = df[prop].shift(1).fillna(0).cumsum()
            logging.info(prop, 'after making series cumulative again before resampling and interpolation:', df.head(25))
            
            # then interpolate the cumulative series
            logging.info(df[df.index.isnull()])
            df = df.resample(up_intv).first()
            logging.info('after resample:', df.head(25))
            df.interpolate(method='time', inplace=True, limit=gap_n_intv)
            logging.info('after interpolation:', df.head(25))
          
            # finally, differentiate a last time to get rate of use
            df[prop] = df[prop].diff().shift(-1)
            logging.info('after taking differences:', df.head(25))
            

        df = Preprocessor.filter_min_max(df, prop, min=min_value, max=max_value)
        df = Preprocessor.filter_static_outliers(df_plot, prop, n_sigma=n_sigma)

        # then first upsample to regular intervals; this creates various colums with np.NaN as value
        df = df.resample(up_intv).first()

        # procedures above may have removed values; fill these, but don't bridge gaps larger than gap_n_intv times the interval
        if (summ == Summarizer.first):
            df.interpolate(method='pad', inplace=True)
        else: 
            df.interpolate(method='time', inplace=True, limit=gap_n_intv)
            

        # interplolate and summarize data using  resampling 
        if (summ == Summarizer.add):
            df = df[prop].resample(summ_intv).sum()
        elif (summ == Summarizer.mean):
            df = df[prop].resample(summ_intv).mean()
        elif (summ == Summarizer.count):
            df = df[prop].resample(summ_intv).count()
        elif (summ == Summarizer.first):
            # not totally sure, mean seems to be a proper summary of e.g. a thermostat setpoint
            df = df[prop].resample(summ_intv).mean()
                
        return df