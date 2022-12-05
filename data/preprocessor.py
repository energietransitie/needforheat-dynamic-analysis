import pandas as pd
from scipy import stats

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
        
        df_result = df
        if not len(df):
            return df_result
        if (min is None) | (max is None) | (min <= max):
            if not inplace:
                df_result = df.copy(deep=True)
            df_result[col] = df_result[col].mask(
                ((min is not None) & (df_result[col] < min))
                |
                ((max is not None) & (df_result[col] > max))
                )

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
        
        df_result = df
        if not len(df):
            return df_result
        if not inplace:
            df_result = df.copy(deep=True)
        if per_id:
            for id in df.index.unique(level='id'):
                df_result.loc[id, col] = (df_result
                                          .loc[id][col]
                                          .mask(stats.zscore(df_result.loc[id][col], nan_policy='omit').abs() > n_sigma)
                                         .values)            
        else:
            df_result.loc[col] = (df_result
                                      .loc[col]
                                      .mask(stats.zscore(df_result.loc[col], nan_policy='omit').abs() > n_sigma)
                                     .values)            
        return df_result
            
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