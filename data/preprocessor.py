import pandas as pd
import numpy as np
from scipy import stats
from tqdm.notebook import tqdm
import pytz
from datetime import datetime, timedelta
from extractor import WeatherExtractor

def update_metadata(meta_df, func_name, params, df_before, df_after, col):
    non_null_before = df_before.notnull().sum().sum()
    non_null_after = df_after.notnull().sum().sum()
    measurements_deleted = non_null_before - non_null_after
    ids_before = df_before.dropna(how='all').index.unique(level='id').nunique()
    ids_after = df_after.dropna(how='all').index.unique(level='id').nunique()
    ids_deleted = ids_before - ids_after
    properties_before = df_before.notnull().any().sum()
    properties_after = df_after.notnull().any().sum()

    # Ensure indices are aligned for comparison
    df_before_aligned, df_after_aligned = df_before.align(df_after, join='inner', axis=0)
    
    # Identify filtered properties correctly
    filtered_properties_mask = df_before_aligned[col].notnull() & df_after_aligned[col].isnull()
    filtered_properties = [col] if filtered_properties_mask.any() else []
    
    ids_filtered = df_before_aligned.loc[filtered_properties_mask].index.unique(level='id').nunique()

    new_row = pd.DataFrame([{
        'step': func_name,
        'property_to_filter': col,
        'params': params,
        'non_null_before': non_null_before,
        'non_null_after': non_null_after,
        'measurements_deleted': measurements_deleted,
        'ids_before': ids_before,
        'ids_after': ids_after,
        'ids_deleted': ids_deleted,
        'properties_before': properties_before,
        'properties_after': properties_after,
        'filtered_properties': filtered_properties,
    }])
    
    return pd.concat([meta_df, new_row], ignore_index=True)


def track_metadata(func):
    def wrapper(*args, **kwargs):
        meta_df = kwargs.pop('meta_df', None)
        df = args[0]
        df_before = df.copy(deep=True)
        result = func(*args, **kwargs)

        # Handle case where meta_df is None (initialize)
        if meta_df is None:
            meta_df = pd.DataFrame(columns=[
                'step', 'property_to_filter', 'params', 'non_null_before', 'non_null_after',
                'measurements_deleted', 'ids_before', 'ids_after', 'ids_deleted',
                'properties_before', 'properties_after', 'filtered_properties'
            ])
        col = kwargs.get('col', args[1])
        params = {k: v for k, v in kwargs.items() if k != 'meta_df'}
        meta_df = update_metadata(meta_df, func.__name__, params, df_before, result, col)
        return result, meta_df
        
    return wrapper


class Preprocessor:
    """
    Use this class to get data from the database that contains measurements.
    """

    
#TODO:check if dynamic outliers removal done by a DataFrame mask based on a hampel filter
# (either before interpolation or after interpolation:
# hampel(df[prop], window_size=5, n=3, imputation=False)


    @staticmethod
    @track_metadata
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
        # Check if prop exists in df_prop.columns
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in df_prop.")
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
    @track_metadata
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
        
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in df_prop.")
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
    @track_metadata
    def filter_id_prop_with_std_zero(df: pd.DataFrame, col: str, inplace=True) -> pd.DataFrame:
        """
        Replace measurement values with NaN for an `id` in the `col` column
        where the standard deviation (`std`) of the measurement values for that `id` is 0.
    
        in: df: pd.DataFrame with
        - index = ['id', 'source', 'timestamp']
          -- id: id of the unit studied (e.g. home / utility building / room)
          -- source: device_type from the database
          -- timestamp: timezone-aware timestamp
        - columns = properties with measurement values
        
        out: pd.DataFrame with same structure as df
        """
    
        if not len(df) or col not in df.columns:
            return df
    
        df_result = df
        if not inplace:
            df_result = df.copy(deep=True)
    
        # Calculate the standard deviation per `id` for the specified column
        std_per_id = df_result[col].groupby(level='id').std()
    
        # Find `id`s where the standard deviation is 0
        ids_with_zero_std = std_per_id[std_per_id == 0].index
    
        # Set values to NaN for the identified `id`s
        df_result.loc[df_result.index.get_level_values('id').isin(ids_with_zero_std), col] = np.nan
    
        return df_result

    @staticmethod
    @track_metadata
    def co2_baseline_adjustment(df: pd.DataFrame,
                                col:str,
                                co2_ext__ppm: int = 415,
                                co2_min_margin__ppm = 1,
                                inplace=True
                                ) -> pd.DataFrame:

        """
        This function adjusts the baseline of all CO₂ measurements, on a per id and per source id basis.
        As a result, the minimum of CO₂ measurements in the col column will be co2_ext__ppm + co2_min_margin__ppm.
        The co2_min_margin__ppm helps to to avoid co2_elevations (typically calculated as co2__ppm - co2_ext__ppm) to be zero during analysis.
        Use case:
        CO₂ sensors are subject to long term drift. Some CO₂ sensors provide automatic occasional recalibration to a pre-determined CO₂ level.
        Not all CO₂ sensors used in a study may have this feature, some may have this turned off (sometimes deliberately, to avoid sudden jumps).
        Some CO₂ sensor may have been calibrated once, but not all in the same circumstances.
        """
        
        if (not len(df)
            or
            (col not in df.columns)):
            return df
        df_result = df
        if not inplace:
            df_result = df.copy(deep=True)
        for id_val in df.index.unique(level='id'):
            for source in df.loc[id_val].index.unique(level='source'):
                min__ppm = df_result.loc[(id_val, source), col].min()
                baseline_adjustment__ppm = (co2_ext__ppm + co2_min_margin__ppm) - min__ppm 
                # Only perform if baseline_adjustment__ppm is not NaN
                if not np.isnan(baseline_adjustment__ppm):
                    df_result.loc[(id_val, source), col] = (df_result.loc[(id_val, source), col] + baseline_adjustment__ppm).values
        return df_result

    
    @staticmethod
    @track_metadata
    def encode_categorical_property_as_boolean_properties(df: pd.DataFrame, 
                                                          property_to_encode: str,
                                                          property_categories: str) -> pd.DataFrame:
        """
        Convert a categorical measurement to one or more dummy properties, each boolean.
    
        Parameters:
        - df (DataFrame): DataFrame with measurements
            - a multi-index consisting of
            -- id: id of the unit studied (e.g. home / utility building / room) 
            -- source_category: e.g. batch_import / cloud_feed / device
            -- source_type: e.g. device_type from the database
            -- timestamp: timezone-aware timestamp
            -- property: property measured 
            - column = 'value', with string representation of measurement value
        - property_to_encode (str): Name of the property to convert.
        - property_categories (dict): Translation table mapping categories to dummy property names.
    
        Returns:
        - DataFrame: Modified DataFrame with only property_to_encode converted to dummy properties.
        """
    
        # Extract values for the property to dummify
        property_values = df.loc[df.index.get_level_values('property') == property_to_encode, 'value']
    
        # Convert to categorical data
        property_values = property_values.astype('category')
    
        # Create binary measurement columns for each category
        binary_columns = pd.get_dummies(property_values).astype(bool)

        # Iterate over the columns and print unique values
        for col in binary_columns.columns:
            unique_values = binary_columns[col].unique()
            print(f"Unique values for column '{col}': {unique_values}")
    
        # Rename columns based on translation table
        binary_columns.rename(columns=property_categories, inplace=True)
    
        # Add '__bool' suffix to column names
        binary_columns.columns = [col.lower().replace(' ', '_') + '__bool' for col in binary_columns.columns]
    
        # Stack binary_columns DataFrame to create long format
        stacked_df = binary_columns.stack()
    
        # Reset index to convert the MultiIndex to columns
        stacked_df = stacked_df.reset_index()
    
        # Rename the measurement value column to 'value'
        stacked_df.rename(columns={stacked_df.columns[-1]: 'value'}, inplace=True)

        # Drop the 'property' column
        stacked_df.drop(columns=['property'], inplace=True)

        # Rename the second to last index level to 'property'
        stacked_df.rename(columns={stacked_df.columns[-2]: 'property'}, inplace=True)
        
        # Rename the measurement value column to 'value'
        stacked_df.rename(columns={0: 'value'}, inplace=True)
    
        # Set the index levels
        index_levels = ['id', 'source_category', 'source_type', 'timestamp', 'property']
        stacked_df.set_index(index_levels, inplace=True)
    
        # Convert values in the 'value' column from int to string
        stacked_df['value'] = stacked_df['value'].astype(str)
    
        # Add the converted measurements to the original DataFrame
        df = pd.concat([df, stacked_df])
    
        # Remove the measurements with 'property' equal to 'boiler_status__str' from df
        df = df[df.index.get_level_values('property') != property_to_encode]
    
        return df

    @staticmethod
    def compute_calibration_factors(df_prop: pd.DataFrame,
                                    prop: str,
                                    source_type_to_calibrate: str,
                                    reference_source_type: str,
                                    min_measurements_per_day=20) -> pd.DataFrame:
        """
        Compute calibration corrections for a specific property based on two source types.
    
        Parameters:
        -----------
        df_prop_filtered : pandas.DataFrame
            Filtered DataFrame containing only relevant property and specified source types.
        prop : str
            Name of the property (column) to calibrate.
        source_type_to_calibrate : str
            Source type whose measurements are to be calibrated.
        reference_source_type : str
            Source type used as the reference for calibration.
        min_measurements_per_day : int, optional
            Minimum number of measurements per day required for calibration, default is 20.
    
        Returns:
        --------
        df_corrections : pandas.DataFrame
            DataFrame with corrections for mean and standard deviation scaling based on the specified source types.
        """ 
        df_prop_filtered = df_prop[[prop]].reset_index()
        df_prop_filtered['date'] = df_prop_filtered['timestamp'].dt.date

        df_prop_filtered = df_prop_filtered[df_prop_filtered['source_type'].isin([source_type_to_calibrate, reference_source_type])]
        
        counts = df_prop_filtered.groupby(['id', 'date', 'source_type']).size().reset_index(name='count')
        counts = counts[counts['count'] >= min_measurements_per_day]

        filtered_df = pd.merge(df_prop_filtered, counts[['id', 'date', 'source_type']], on=['id', 'date', 'source_type'])

        pivoted_df = filtered_df.pivot_table(index=['id', 'date'], columns='source_type', values=prop, aggfunc=['mean', 'std'])
        pivoted_columns = [f'{agg_func}_{source_type}' for agg_func, source_type in pivoted_df.columns]
        pivoted_df.columns = pivoted_columns
        pivoted_df = pivoted_df.reset_index()
    
        pivoted_df.dropna(subset=pivoted_columns, inplace=True)
    
        df_corrections = pivoted_df.groupby('id').mean().reset_index()

        return df_corrections

    @staticmethod
    @track_metadata
    def create_calibrated_property(df_prop: pd.DataFrame,
                                   prop: str, 
                                   source_type_to_calibrate: str, 
                                   reference_source_type: str,
                                   min_measurements_per_day=20) -> pd.DataFrame:
        """
        Perform calibration of measurements for a specific property based on two source types.
    
        Parameters:
        -----------
        df_prop : pandas.DataFrame
            DataFrame with MultiIndex (id, source_category, source_type, timestamp) and properties as columns.
        prop : str
            Name of the property (column) to calibrate.
        source_type_to_calibrate : str
            Source type whose measurements are to be calibrated.
        reference_source_type : str
            Source type used as the reference for calibration.
        min_measurements_per_day : int, optional
            Minimum number of measurements per day required for calibration, default is 20.
    
        Returns:
        --------
        df_prop_final : pandas.DataFrame
            Original DataFrame with calibrated measurements added as a new source type ('{source_type_to_calibrate}_calibrated').
        """
        df_corrections = Preprocessor.compute_calibration_factors(df_prop, 
                                                                  prop, 
                                                                  source_type_to_calibrate, 
                                                                  reference_source_type, 
                                                                  min_measurements_per_day)
    
        df_filtered = df_prop[df_prop.index.get_level_values('source_type') == source_type_to_calibrate][[prop]]
        df_filtered = df_filtered.join(df_corrections.set_index('id')[
                                       [f'mean_{source_type_to_calibrate}', 
                                        f'std_{source_type_to_calibrate}', 
                                        f'mean_{reference_source_type}', 
                                        f'std_{reference_source_type}']], on='id')
    
        # Calculate Z-score using mean and std of source_type_to_calibrate
        z_score = (df_filtered[prop] - df_filtered[f'mean_{source_type_to_calibrate}']) / df_filtered[f'std_{source_type_to_calibrate}']
    
        # Calculate calibrated property using mean and std of reference_source_type
        df_filtered['prop_calibrated'] = (z_score
                                          * df_filtered[f'std_{reference_source_type}']
                                          + df_filtered[f'mean_{reference_source_type}'])
    
        df_filtered_calibrated = df_filtered.copy()
        df_filtered_calibrated['source_type'] = f'{source_type_to_calibrate}_calibrated'
        df_filtered_calibrated = (df_filtered_calibrated
                                  .reset_index(level='source_type', drop=True)
                                  .set_index('source_type', append=True))
        df_filtered_calibrated = df_filtered_calibrated.drop(columns=[prop,
                                                                      f'mean_{source_type_to_calibrate}',
                                                                      f'std_{source_type_to_calibrate}',
                                                                      f'mean_{reference_source_type}',
                                                                      f'std_{reference_source_type}'])
        df_filtered_calibrated = df_filtered_calibrated.rename(columns={'prop_calibrated': prop})
    
        df_filtered_calibrated = df_filtered_calibrated.reorder_levels(df_prop.index.names)
    
        df_prop_final = pd.concat([df_prop, df_filtered_calibrated])
        return df_prop_final

    
    @staticmethod
    def highlight_specific_value(val, specific_value=0):
        """
        Highlight cells in a DataFrame with a specific value.
        
        Parameters:
        -----------
        val : any
            The value to be checked.
        specific_value : any
            The value to be highlighted, default is 0.
        
        Returns:
        --------
        str
            The background color for highlighting.
        """
        color = 'red' if val == specific_value else ''
        return f'background-color: {color}'

    @staticmethod
    def count_non_null_measurements(df: pd.DataFrame) -> pd.DataFrame:
        """
        Count non-null measurements per column and per id.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with a MultiIndex with levels id, source_category, source_type, timestamp, and measured properties in columns.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with counts of non-null measurements and total non-null values per id.
        """
        non_null_counts_per_col = df.groupby(level='id').count()
        non_null_counts_per_col['total_non_null'] = non_null_counts_per_col.sum(axis=1)
        return non_null_counts_per_col.sort_values(by='total_non_null', ascending=False)

    @staticmethod
    def calculate_covered_time(df: pd.DataFrame, max_interval=90*60, unit='days') -> pd.DataFrame:
        """
        Calculate the total covered time excluding large intervals.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with a MultiIndex with levels id, source_category, source_type, timestamp, and measured properties in columns.
        max_interval : int, optional
            Maximum interval in seconds to be considered for covered time, default is 90 minutes (5400 seconds).
        unit : str, optional
            Unit of time to return the covered time in. Options are 'seconds', 'minutes', 'hours', 'days'. Default is 'days'.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with total covered time per id, in the specified unit.
        """
        df_analysis = df.copy()
        df_analysis = df_analysis[~df_analysis.index.duplicated(keep='first')]
        df_analysis.sort_index(level=['id', 'timestamp'], inplace=True)

        def calculate_time_covered(group):
            intervals = group.dropna().index.get_level_values('timestamp').to_series().diff().dt.total_seconds()
            valid_intervals = intervals[intervals <= max_interval]
            return valid_intervals.sum()

        covered_time = df_analysis.groupby(level='id').apply(lambda x: x.apply(lambda col: calculate_time_covered(col), axis=0))

        # Convert to the desired unit
        unit_conversion = {
            'seconds': 1,
            'minutes': 60,
            'hours': 3600,
            'days': 86400
        }

        if unit not in unit_conversion:
            raise ValueError(f"Invalid unit '{unit}'. Choose from 'seconds', 'minutes', 'hours', 'days'.")

        covered_time /= unit_conversion[unit]
        covered_time['total'] = covered_time.sum(axis=1)

        return covered_time.sort_values(by='total', ascending=False)

   
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
                         property_limits: dict = None,
                         upsample__min: int = 5,
                         interpolate__min: int = 15,
                         restore_original_types: bool = False,
                         inplace: bool = False) -> pd.DataFrame:
        """
        Interpolate a DataFrame by resampling first to upsample_min intervals,
        then interpolating to interpolate_min minutes using linear interpolation,
        while making sure not to bridge gaps larger than limit_min minutes.
        The final DataFrame has column datatypes as indicated in the property_types dictionary.
    
        Parameters:
        - df_prop: pd.DataFrame with MultiIndex ['id', 'source_category', 'source_type', 'timestamp'] and property columns.
        - property_limits: Dictionary specifying limit__min for specific properties.
        - upsample__min: Interval for initial resampling.
        - interpolate__min: Interval for final resampling after interpolation.
        - restore_original_types: Flag to restore original data types after interpolation.
        - inplace: Flag to modify df_prop in place.
    
        Returns:
        - pd.DataFrame with same structure as df_prop, interpolated and potentially restored data types.
        """
        
        if property_limits is None:
            property_limits = {}
    
        default_limit_min = 90
        
        df_result_list = []
        
        # Ensure MultiIndex levels exist and are in the correct order
        if not isinstance(df_prop.index, pd.MultiIndex):
            raise ValueError("Input DataFrame df_prop must have a MultiIndex.")
        expected_levels = ['id', 'source_category', 'source_type', 'timestamp']
        if not all(level in df_prop.index.names for level in expected_levels):
            raise ValueError(f"Input DataFrame df_prop must have MultiIndex levels: {expected_levels}.")
        
        for id_value in tqdm(df_prop.index.get_level_values('id').unique()):
            for cat_value in df_prop.loc[id_value].index.get_level_values('source_category').unique():
                for type_value in df_prop.loc[(id_value, cat_value)].index.get_level_values('source_type').unique():
                    df_interpolated = df_prop.loc[(id_value, cat_value, type_value), :]
                    if not len(df_interpolated):
                        continue
                    if not inplace:
                        df_interpolated = df_interpolated.copy()
                    
                    for col in df_interpolated.columns:
                        limit__min = property_limits.get(col, default_limit_min)
                        limit = max((limit__min - 1) // upsample__min, 1)  # Ensure limit is at least 1
    
                        if df_interpolated[col].dtype == 'bool':
                            df_interpolated[col] = df_interpolated[col].astype('int32')
                            df_interpolated[col] = (df_interpolated[col]
                                                    .resample(str(upsample__min) + 'T')
                                                    .first()
                                                    .astype('float32')
                                                    .interpolate(method='time', limit=limit)
                                                    .resample(str(interpolate__min) + 'T').mean()
                                                   )
                        elif df_interpolated[col].dtype in ['object', 'string']:
                            df_interpolated[col] = (df_interpolated[col]
                                                    .resample(str(upsample__min) + 'T')
                                                    .first()
                                                    .ffill(limit=limit)
                                                    .resample(str(interpolate__min) + 'T').first()
                                                   )
                        else:
                            df_interpolated[col] = (df_interpolated[col]
                                                    .resample(str(upsample__min) + 'T')
                                                    .first()
                                                    .astype('float32')
                                                    .interpolate(method='time', limit=limit)
                                                    .resample(str(interpolate__min) + 'T').mean()
                                                   )
                    
                    df_interpolated['id'] = id_value
                    df_interpolated['source_category'] = cat_value
                    df_interpolated['source_type'] = type_value
                    df_interpolated = df_interpolated.reset_index()
                    df_result_list.append(df_interpolated)
        
        df_result = pd.concat(df_result_list)
        df_result = df_result.set_index(['id', 'source_category', 'source_type', 'timestamp']).sort_index()
        
        if restore_original_types:
            for col in df_result.columns:
                original_dtype = df_prop[col].dtype
                if original_dtype == 'bool':
                    df_result[col] = df_result[col].round().astype('int32').astype('bool')
                elif original_dtype != 'float32' and original_dtype != 'float64' and original_dtype not in ['object', 'string']:
                    df_result[col] = df_result[col].round().astype(original_dtype)
                elif original_dtype in ['object', 'string']:
                    df_result[col] = df_result[col].astype(original_dtype)
        
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
        
        # filter out clearly wrong measurements (< 5 ppm)
        df_prop = Preprocessor.filter_min_max(df_prop, prop, min=5)

        # also filter out measurements by CO₂sensors that are always constant
        std = df_prop[prop].groupby(['id', 'source']).transform('std')
        # set values to np.nan where std is zero
        mask = std == 0
        df_prop[mask] = np.nan
        
        # adjust the CO₂ baseline, on a per room and per source basis
        df_prop = Preprocessor.co2_baseline_adjustment(df_prop,
                                                       prop,
                                                       co2_ext__ppm = 415,
                                                       co2_min_margin__ppm = 1
                                                      )


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