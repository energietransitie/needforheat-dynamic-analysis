import pylab as plt
import pandas as pd
from importlib import reload
from datetime import datetime, timedelta
import numpy as np
import pylab as plt
import seaborn as sns
from preprocessor import Preprocessor
import missingno as msno
import warnings
import folium
import h3
from geopy.distance import distance

import matplotlib.dates as mdates
# import mplcursors  # Import mplcursors for interactive hovering

class Plot:

    @staticmethod
    def nfh_measurements_plot(df_source: pd.DataFrame, ids=None, source_categories=None, source_types=None, units=None, properties=None, units_to_mathtext=None):
        """
        Plot data in df DataFrame, one plot per id, one subplot for all properties with the same unit
        
        in: dataframe with
        - MultiIndex = ['id', 'source_category', 'source_type', 'timestamp', 'property']
            - id: id of e.g. home / utility building / room 
            - source_category: category of the data source
            - source_type: type of the data source
            - timestamp: timezone-aware timestamp
            - property: type of the property measured
        - value: measurement value (string)
        - units_to_mathtext: table that translates property unit postfixes to mathtext.
        """      
        
        df = df_source.copy()
        
        if ids is not None:
            df = df.loc[df.index.get_level_values('id').isin(ids)]
        if source_types is not None:
            df = df.loc[df.index.get_level_values('source_type').isin(source_types)]
        if units is not None:
            property_names = df.index.get_level_values('property')
            filtered_properties = [p for p in property_names if any(p.endswith(f'__{unit}') for unit in units)]
            df = df.loc[df.index.get_level_values('property').isin(filtered_properties)]
        if source_categories is not None:
            df = df.loc[df.index.get_level_values('source_category').isin(source_categories)]
        if properties is not None:
            df = df.loc[df.index.get_level_values('property').isin(properties)]

        # Filter out rows with measurement values whose property name ends with '__str'
        df = df.loc[~df.index.get_level_values('property').str.endswith('__str')]

        
        # Convert measurement values to float, coercing non-numeric values to NaN
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        for id_ in df.index.get_level_values('id').unique():
            try:
                df_plot = df.xs(id_)

                # Merge source_category, source_type, and property into a single index level
                df_plot.index = df_plot.index.map(lambda x: (x[0], x[1], x[2], f"{x[0]}_{x[1]}_{x[3]}"))

                # Drop the first two levels and rename the last two
                df_plot.index = df_plot.index.droplevel([0, 1])

                df_plot.index.names = ['timestamp', 'merged_property']
                
                # Check for duplicates in the index after merging
                duplicate_entries = df_plot.index.duplicated().any()
                
                if duplicate_entries:
                    print("Duplicate entries found in the index after merging. Handled mby taking the average.")
                    df_plot = df_plot.groupby(['timestamp', 'merged_property'])['value'].mean()
                    df_plot = df_plot.unstack('merged_property')
                else: 
                    df_plot = df_plot.unstack('merged_property')  
                    df_plot.columns = list(df_plot.columns.droplevel(0))
                  
                props_with_data = [prop for prop in list(df_plot.columns) if df_plot[prop].count()>0] 
                units_with_data = np.unique(np.array([prop.split('__')[-1] for prop in props_with_data]))
                unit_tuples = [tuple([prop.split('__')[0] for prop in props_with_data if prop.split('__')[-1] == unit]) for unit in units_with_data]
                props_with_data = [prop.split('__')[0] for prop in props_with_data]
                labels = [col.split('__')[0] for col in df_plot.columns]
                df_plot.columns = labels
                axes = df_plot[props_with_data].plot(
                    subplots = unit_tuples,
                    style='.',
                    title=f'id: {id_}'
                )
                # Calculate the minimum and maximum timestamps for the current 'id'
                min_timestamp = df_plot.index.get_level_values('timestamp').min()
                max_timestamp = df_plot.index.get_level_values('timestamp').max()

                for unit in enumerate(units_with_data):
                    if units_to_mathtext is not None:
                        axes[unit[0]].set_ylabel(units_to_mathtext[unit[1]])
                    else:
                        axes[unit[0]].set_ylabel(unit[1])


                    # Add vertical grid lines at midnight each day for the current 'id'
                    dates = pd.date_range(start=min_timestamp.floor('D'), end=max_timestamp.floor('D'), freq='D')  # All midnights within the range for the current 'id'
                    for date in dates:
                        axes[unit[0]].axvline(x=date, color='gray', linestyle='--', linewidth=0.5)
                
                plt.tight_layout()
                plt.show()
            except TypeError:
                print(f'No data for id: {id_}')


    @staticmethod
    def dataframe_properties_plot(df: pd.DataFrame, units_to_mathtext = None): 
        """
        Plot data in df DataFrame, one plot per id, one subplot for all properties with the same unit
        
        in: dataframe with
        - index = ['id', 'source', 'timestamp']
            - id: id of e.g. home / utility building / room 
            - source: device_type from the database
            - timestamp: timezone-aware timestamp
        - columns = all properties in the input column
            - unit types are encoded as last part of property name, searated by '__'
        - units_to_mathtext: table tat translates property unit postfixes to mathtext.
        """      
        
        for id in list(df.index.get_level_values('id').unique()):
            try:
                df_plot = df.loc[id].unstack([0])
                df_plot.columns = df_plot.columns.swaplevel(0,1)
                df_plot.columns = ['_'.join(col) for col in df_plot.columns.values]
                props_with_data = [prop for prop in list(df_plot.columns) if df_plot[prop].count()>0] 
                units_with_data = np.unique(np.array([prop.split('__')[-1] for prop in props_with_data]))
                unit_tuples = [tuple([prop.split('__')[0] for prop in props_with_data if prop.split('__')[-1] == unit]) for unit in units_with_data]
                props_with_data = [prop.split('__')[0] for prop in props_with_data]
                labels = [col.split('__')[0] for col in df_plot.columns]
                df_plot.columns = labels
                axes = df_plot[props_with_data].plot(
                    subplots = unit_tuples,
                    style='.--',
                    alpha=0.5,  # Set alpha for transparency
                    title=f'id: {id}'
                )
                for unit in enumerate(units_with_data):
                    if units_to_mathtext is not None:
                        axes[unit[0]].set_ylabel(units_to_mathtext[unit[1]])
                    else:
                        axes[unit[0]].set_ylabel(unit[1])
                plt.show()
            except TypeError:
                print(f'No data for id: {id}')
    
    @staticmethod
    def dataframe_preprocessed_plot(df_prep: pd.DataFrame, units_to_mathtext = None): 
        """
        Plot data in df DataFrame, one plot per id, one subplot for all properties with the same unit
        
        - df_prep: DataFrame with preprocessed properties containing the data:
            - index = ['id', 'timestamp']
                - id: id of e.g. home / utility building / room 
                - timestamp: timezone-aware timestamp
            - columns = all source_properties in the input column
                - unit types are encoded as last part of property name, searated by '__'
        - units_to_mathtext: table tat translates property unit postfixes to mathtext.
        """      
        
        for id in list(df_prep.index.to_frame(index=False).id.unique()):
            try:
                df_plot = df_prep.loc[id]
                props_with_data = [prop for prop in list(df_plot.columns) if df_plot[prop].count()>0] 
                units_with_data = np.unique(np.array([prop.split('__')[-1] for prop in props_with_data]))
                unit_tuples = [tuple([prop.split('__')[0] for prop in props_with_data if prop.split('__')[-1] == unit]) for unit in units_with_data]
                props_with_data = [prop.split('__')[0] for prop in props_with_data]
                labels = [col.split('__')[0] for col in df_plot.columns]
                df_plot.columns = labels
                axes = df_plot[props_with_data].plot(
                    subplots = unit_tuples,
                    style='.--',
                    alpha=0.5,  # Set alpha for transparency
                    title=f'id: {id}'
                )
                for unit in enumerate(units_with_data):
                    if units_to_mathtext is not None:
                        axes[unit[0]].set_ylabel(units_to_mathtext[unit[1]])
                    else:
                        axes[unit[0]].set_ylabel(unit[1])
                plt.show()
            except TypeError:
                print(f'No data for id: {id}')

    @staticmethod
    def plot_data_availability(df_prep, properties_include=None, 
                               properties_exclude=None, 
                               alpha=0.5, 
                               figsize=(12, 8), 
                               title_fontsize=10):
        """
        Plots data availability over time for various IDs using subplots.
        
        Parameters:
        - df_prep: DataFrame with preprocessed properties containing the data:
            - index = ['id', 'timestamp']
                - id: id of e.g. home / utility building / room 
                - timestamp: timezone-aware timestamp
            - columns = all source_properties in the input column
        - properties_include: List of properties to include for validation. If None, all properties are included.
        - properties_exclude: List of properties to exclude for validation. If None, no properties are excluded.
        - alpha: Transparency level for the bars.
        - figsize: Tuple specifying the width and height of the figure.
        """
            
        # If properties_include is specified, use it; otherwise, use all columns except id_column and time_column
        if properties_include is not None:
            properties = properties_include
        else:
            properties = df_prep.columns
        
        # Exclude specified properties if properties_exclude is provided
        if properties_exclude is not None:
            properties = properties.difference(properties_exclude)

        # Determine the validity of mandatory properties for each timestamp and ID
        valid_mask = df_prep[properties].notnull().all(axis=1)
        
        # Get the timestamps and IDs where all mandatory properties are valid
        valid_timestamps = df_prep[valid_mask].index.get_level_values('timestamp')
        
        if not valid_timestamps.empty:
            # Determine the first and last valid timestamps
            first_valid_timestamp = valid_timestamps.min()
            last_valid_timestamp = valid_timestamps.max()

        # Filter df_prep to include only the rows within the determined timestamp range
        df_prep_filtered = df_prep.loc[(df_prep.index.get_level_values('timestamp') >= first_valid_timestamp) &
                              (df_prep.index.get_level_values('timestamp') <= last_valid_timestamp)].copy()
        
        df_pivot = df_prep_filtered[properties].notnull().all(axis=1).replace(False, np.nan).unstack('id')
        
        # Prepare the figure and subplots
        num_ids = df_pivot.shape[1]
        fig, axes = plt.subplots(num_ids, 1, figsize=figsize, sharex=True, sharey=True)
        
        if num_ids == 1:
            axes = [axes]  # Ensure axes is always a list even for single subplot
        
        for idx, (home_id, ax) in enumerate(zip(df_pivot.columns, axes)):
            data_avail = df_pivot[home_id]
            
            # Plot green for available data, red for missing data
            ax.fill_between(data_avail.index, 1, where=data_avail.isna(), facecolor='red', alpha=alpha, step='mid')
            ax.fill_between(data_avail.index, 1, where=data_avail.notna(), facecolor='green', alpha=alpha, step='mid')
            
            ax.set_ylabel(f'ID {home_id}', rotation=0, labelpad=40)
            ax.set_yticks([])  # Hide y-ticks for clarity
        
        plt.xlabel('Time')

        # Adjust the title based on included and excluded properties
        included_str = str(properties_include) if properties_include is not None else "All"
        excluded_str = str(properties_exclude) if properties_exclude is not None else "None"
        if properties_exclude is None:
            plt.suptitle(f"Overview of Valid Measurements Over Time\nincluded: {included_str}", fontsize=title_fontsize, wrap=True)
        else:
            plt.suptitle(f"Overview of Valid Measurements Over Time\nincluded: {included_str}\n Excluded: {excluded_str}", fontsize=title_fontsize, wrap=True)
            

        plt.tight_layout()
        plt.show()
        

    
    @staticmethod
    def plot_missing_data_overview(df_prep, 
                                   properties_include=None, 
                                   properties_exclude=None, 
                                   freq='1W',
                                   tick_label_fontsize=10,
                                   figsize=(10, 6),
                                   title_fontsize=10
                                  ):
        """
        Plots an overview of valid measurements over time for various IDs.
        
        Parameters:
        in: 
        - df_prep: DataFrame with preprocessed properties containing the data:
            - index = ['id', 'timestamp']
                - id: id of e.g. home / utility building / room 
                - timestamp: timezone-aware timestamp
            - columns = all source_properties in the input column
        - properties_include: List of properties to include for validation. If None, all properties are included.
        - properties_exclude: List of properties to exclude for validation. If None, no properties are excluded.
        """
            
        # If properties_include is specified, use it; otherwise, use all columns except id_column and time_column
        if properties_include is not None:
            properties = properties_include
        else:
            properties = df_prep.columns
        
        # Exclude specified properties if properties_exclude is provided
        if properties_exclude is not None:
            properties = properties.difference(properties_exclude)

        # Determine the validity of mandatory properties for each timestamp and ID
        valid_mask = df_prep[properties].notnull().all(axis=1)
        
        # Get the timestamps and IDs where all mandatory properties are valid
        valid_timestamps = df_prep[valid_mask].index.get_level_values('timestamp')
        
        if not valid_timestamps.empty:
            # Determine the first and last valid timestamps
            first_valid_timestamp = valid_timestamps.min()
            last_valid_timestamp = valid_timestamps.max()

        # Filter df_prep to include only the rows within the determined timestamp range
        df_prep_filtered = df_prep.loc[(df_prep.index.get_level_values('timestamp') >= first_valid_timestamp) &
                              (df_prep.index.get_level_values('timestamp') <= last_valid_timestamp)].copy()
        

        # Localize timestamps in df_prep_filtered 
        df_prep_filtered.index = df_prep_filtered.index.set_levels(df_prep_filtered.index.levels[1].tz_convert('UTC').tz_localize(None), level=1)
        
        df_pivot = df_prep_filtered[properties].notnull().all(axis=1).replace(False, np.nan).unstack('id')


        # Plot using missingno, suppressing warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        # Visualize the completeness of the data
        msno.matrix(df_pivot, sparkline=False, freq=freq, figsize=figsize)

        # Reset warnings filter to default state
        warnings.resetwarnings()
                                
        # Adjust the title based on included and excluded properties
        included_str = str(properties_include) if properties_include is not None else "All"
        excluded_str = str(properties_exclude) if properties_exclude is not None else "None"

        # Plot using missingno
        plt.suptitle("Overview of Valid Measurements Over Time")
        if properties_exclude is None:
            plt.title(f"included: {included_str}", fontsize=title_fontsize, wrap=True)
        else:
            plt.title(f"included: {included_str}\n Excluded: {excluded_str}", fontsize=title_fontsize, wrap=True)

        plt.xlabel("ID")
        plt.ylabel("Time (UTC)")
        # Adjust the font size of tick labels
        plt.xticks(fontsize=tick_label_fontsize)
        plt.yticks(fontsize=tick_label_fontsize)

        # Adjust layout to fit the figure size
        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def same_unit_property_histogram(df_prop: pd.DataFrame, regex_filter: str, units_to_mathtext = None, bins = 200, per_id= True):
        """
        Plot a histogram of all property columns, filtered by regex (all properties need to have the same unit. 
        
        in: dataframe with
        - index = ['id', 'source', 'timestamp']
            - id: id of e.g. home / utility building / room 
            - source: device_type from the database
            - timestamp: timezone-aware timestamp
        - columns = all properties in the input column
            - unit types are encoded as last part of property name, searated by '__'
            
        - regex_filer: regular expression used to filter property columns 
        - units_to_mathtext: table tat translates property unit postfixes to mathtext.
        - per_id: boolean that signals whether the plot should split the units per if

        action: histogram of CO₂ measurements per id
        """
        
        if per_id:
            df_hist = Preprocessor.unstack_prop(df_prop).filter(regex=regex_filter).dropna(axis=1, how='all').unstack([0])

            df_hist.columns = df_hist.columns.swaplevel(0,1)

            df_hist.columns = ['_'.join(map(str, col)) for col in df_hist.columns.values]
        else:
            df_hist = df_prop.filter(regex=regex_filter)

        labels = [col.split('__')[0] for col in df_hist.columns]
        units = [col.split('__')[-1] for col in df_hist.columns]
        df_hist.columns = labels
        df_hist.plot.hist(bins=bins, alpha=0.5)
        plt.xlabel(f'[{units_to_mathtext[units[0]]}]')
        plt.show()

    @staticmethod
    def temperature_and_power_one_home_plot(title:str, 
                                   df: pd.DataFrame,
                                   shared_x_label='Datetime',
                                   temp_plot_dict = {},
                                   temp_y_label = r'$Temperature [^oC]$',
                                   temp_plot_2nd_list = [],
                                   temp_2nd_y_label = r'$Wind speed [m/s]$',
                                   power_plot_dict = {},
                                   power_y_label = 'Power [W]',
                                   power_plot_2nd_list = [],
                                   power_2nd_y_label = r'$I\ [W/m^2]$'
                                  ):
        """
        Plot of temperature and power data 
        """
        
        #define subplot layout
        if len(power_plot_dict.keys()) == 0:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
            fig.suptitle(title)  
            #add DataFrames to subplots
            df[temp_plot_dict.keys()].plot(
                ax=ax,
                marker=".",
                ms=3,
                linestyle='None',
                secondary_y=temp_plot_2nd_list,
                grid=True, 
                legend=True, 
                color=temp_plot_dict,
                xlabel=shared_x_label,
                ylabel = temp_y_label
            )
            ax.set_facecolor('black')
            ax.legend(fontsize='small') 
        else:
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            fig.suptitle(title)  
        

            #add DataFrames to subplots
            df[temp_plot_dict.keys()].plot(
                ax=ax[0],
                marker=".",
                ms=3,
                linestyle='None',
                secondary_y=temp_plot_2nd_list,
                grid=True, 
                legend=True, 
                color=temp_plot_dict,
                xlabel=shared_x_label,
                ylabel = temp_y_label
            )
            ax[0].set_facecolor('black')
            ax[0].legend(fontsize='small') 

            if len(temp_plot_2nd_list) >0:
                ax[0].right_ax.set_ylabel(temp_2nd_y_label)
                ax[0].right_ax.legend(fontsize='small') 

            df[power_plot_dict.keys()].plot(
                ax=ax[1],
                secondary_y=power_plot_2nd_list, 
                marker=".",
                ms=3,
                linestyle='None',
                mark_right=True, 
                grid=True, 
                legend=True, 
                color=power_plot_dict,
                xlabel=shared_x_label,
                ylabel = power_y_label
            )
            ax[1].set_facecolor('black')
            ax[1].legend(fontsize='small') 
            if len(power_plot_2nd_list) >0:
                ax[1].right_ax.set_ylabel(power_2nd_y_label)
                ax[1].right_ax.legend(fontsize='small') 

        plt.show()
        
       
    @staticmethod
    def temperature_and_power_plot(df: pd.DataFrame, 
                                   shared_x_label='Datetime',
                                   temp_plot_dict = {},
                                   temp_y_label = r'$Temperature [^oC]$',
                                   temp_plot_2nd_list = [],
                                   temp_2nd_y_label = r'$Wind speed [m/s]$',
                                   power_plot_dict = {},
                                   power_y_label = 'Power [W]',
                                   power_plot_2nd_list = [],
                                   power_2nd_y_label = r'$I\ [W/m^2]$'):
        

        # for home_id in df.index.levels[0]:
        ## line above  did not work in specific cases, 
        ## e.g. when df = df_results_tempsim.query('home_id == 809743')

        ##  so we found a workaround:
        homes_to_plot = df.index.to_frame(index=False).home_id.unique()
        for home_id in homes_to_plot:
            Plot.temperature_and_power_one_home_plot(f'Learned model parameters for home: {home_id}',
                                                     df.loc[home_id],
                                                     shared_x_label = shared_x_label,
                                                     temp_plot_dict = temp_plot_dict,
                                                     temp_y_label = temp_y_label,
                                                     temp_plot_2nd_list = temp_plot_2nd_list,
                                                     temp_2nd_y_label = temp_2nd_y_label,
                                                     power_plot_dict = power_plot_dict,
                                                     power_y_label = power_y_label,
                                                     power_plot_2nd_list = power_plot_2nd_list,
                                                     power_2nd_y_label = power_2nd_y_label)

            
    @staticmethod
    def temperature_and_power_one_home_weekly_plot(home_id:int,
                                                   df: pd.DataFrame,
                                                   first_day: datetime,
                                                   last_day: datetime,
                                                   sanity_threshold_timedelta:timedelta=timedelta(hours=24),
                                                   shared_x_label='Datetime',
                                                   temp_plot_dict = {},
                                                   temp_y_label = r'$Temperature [^oC]$',
                                                   temp_plot_2nd_list = [],
                                                   temp_2nd_y_label = r'$Wind speed [m/s]$',
                                                   power_plot_dict = {},
                                                   power_y_label = 'Power [W]',
                                                   power_plot_2nd_list = [],
                                                   power_2nd_y_label = r'$I\ [W/m^2]$'):

       
        for moving_horizon_start in pd.date_range(start=first_day, end=last_day, inclusive='left', freq='7D'):
            moving_horizon_end = min(last_day, moving_horizon_start + timedelta(days=7))

            if (moving_horizon_end < last_day):
                df_moving_horizon = df[moving_horizon_start:moving_horizon_end].iloc[:-1]
            else:
                df_moving_horizon = df[moving_horizon_start:last_day]

            moving_horizon_end = df_moving_horizon.index.max()

            if ((moving_horizon_end - moving_horizon_start) >= sanity_threshold_timedelta):
                Plot.temperature_and_power_one_home_plot(f'Learned model parameters for home: {home_id} from {moving_horizon_start} to {moving_horizon_end}',
                                                         df_moving_horizon,
                                                         shared_x_label = shared_x_label,
                                                         temp_plot_dict = temp_plot_dict,
                                                         temp_y_label = temp_y_label,
                                                         temp_plot_2nd_list = temp_plot_2nd_list,
                                                         temp_2nd_y_label = temp_2nd_y_label,
                                                         power_plot_dict = power_plot_dict,
                                                         power_y_label = power_y_label,
                                                         power_plot_2nd_list = power_plot_2nd_list,
                                                         power_2nd_y_label = power_2nd_y_label)
            
            
            
        
    @staticmethod
    def plot_thermostat_programs(df):
        # Iterate through each unique 'id' (home)
    
        # Mapping each day to an offset in days to plot sequentially
        weekday_offsets = {
            'Monday': 0,
            'Tuesday': 1,
            'Wednesday': 2,
            'Thursday': 3,
            'Friday': 4,
            'Saturday': 5,
            'Sunday': 6
        }
    
        for home_id in df.index.get_level_values('id').unique():
            # Filter the dataframe for the current home_id
            home_df = df.loc[home_id]
            
            # Set up the plot for each home
            plt.figure(figsize=(12, 6))
            plt.title(f'Thermostat Programs for Home ID: {home_id}')
            plt.xlabel('Day of Week and Time of Day')
            plt.ylabel('Setpoint Temperature (°C)')
            
            # For distinct colors in plotting
            color_map = plt.cm.get_cmap('tab10', len(home_df))
            
            # Loop through each valid interval for this home and plot the program
            for idx, (interval, row) in enumerate(home_df.iterrows()):
                program = row['program']
    
                if program is not None:
                        
                    # Prepare data for plotting with step logic
                    times, temps = [], []
                    marker_times, marker_temps = [], []  # For plotting markers on actual setpoints
                    last_temp = None
                    
                    for entry in program:
                        # Parse weekday and start time
                        weekday = entry['weekday']
                        start_time = entry['start_time']
                        temp = entry['temp_set__degC']
                        
                        # Calculate the full datetime for plotting (start of week + offset days)
                        base_time = datetime.strptime("Monday 00:00", "%A %H:%M")
                        day_time = base_time + timedelta(days=weekday_offsets[weekday]) + timedelta(
                            hours=int(start_time.split(":")[0]),
                            minutes=int(start_time.split(":")[1])
                        )
                        
                        # Insert the previous temperature value 1 minute before the new one (for stair-step effect)
                        if last_temp is not None:
                            times.append(day_time - timedelta(minutes=1))
                            temps.append(last_temp)
                        
                        # Add the actual setpoint time and temperature
                        times.append(day_time)
                        temps.append(temp)
                        
                        # Track real setpoints separately for marker plotting
                        marker_times.append(day_time)
                        marker_temps.append(temp)
                        
                        # Update the last temperature
                        last_temp = temp
         
                    # Handle the weekly wrap-around
                    if last_temp is not None:
                        # Add the last Sunday setpoint to Sunday 23:59
                        sunday_end = base_time + timedelta(days=6, hours=23, minutes=59)
                        times.append(sunday_end)
                        temps.append(last_temp)
                    
                        # Prepare for Monday 0:00
                        next_monday_start = base_time  # This will represent Monday 0:00
                    
                        # Check if there is already a setpoint defined for Monday 0:00
                        monday_setpoint_defined = any(
                            entry['weekday'] == 'Monday' and entry['start_time'] == '00:00'
                            for entry in program
                        )
                    
                        if not monday_setpoint_defined:
                            # If no explicit setpoint for Monday 0:00, add the last temperature from Sunday
                            times.append(next_monday_start)
                            temps.append(last_temp)
                    
                            # Insert the same setpoint one minute before the first setpoint of the week
                            if times:  # Ensure there are existing times to reference
                                first_setpoint_time = times[1]  # The first actual setpoint is at index 1
                                times.append(first_setpoint_time - timedelta(minutes=1))
                                temps.append(last_temp)
        
                    # Sort times and corresponding temperatures
                    sorted_times_temps = sorted(zip(times, temps))  # Combine and sort by time
                    sorted_times, sorted_temps = zip(*sorted_times_temps)  # Unzip back into two lists
        
        
                    
                    # Plot the main step line with increased thickness
                    plt.step(sorted_times, sorted_temps, where='post', label=f"{interval}", color=color_map(idx), alpha=0.6, linewidth=3.5)
                    
                    # Overlay markers at real programmed setpoints only
                    # markers = plt.plot(marker_times, marker_temps, 'o', color=color_map(idx), markersize=6, label=f"{interval} markers")
                    markers = plt.plot(marker_times, marker_temps, 'o', color=color_map(idx), markersize=6)
        
                    # # Enable hover functionality using mplcursors
                    # mplcursors.cursor(markers, hover=True).connect(
                    #     "add", 
                    #     lambda sel: sel.annotation.set_text(
                    #         f"Date: {sel.target[0]:%A %H:%M}\nSetpoint: {sel.target[1]:.1f} °C"
                    #     )
                    # )
    
            # Set the x-axis to display days in order from Monday to Sunday
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
            plt.gca().xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))  # Including 0:00
            
            # Format the major ticks to show "Monday 0:00", "Tuesday 0:00", etc.
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%A %H:%M'))
            plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
            
            # Add vertical lines for major and minor ticks
            plt.grid(visible=True, which='major', color='black', linestyle='-', linewidth=1)
            plt.grid(visible=True, which='minor', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
            
            # Rotate all x-axis labels for clarity
            plt.gcf().autofmt_xdate(rotation=90)
            
            # Rotate the minor ticks separately
            for label in plt.gca().get_xticklabels(minor=True):
                label.set_rotation(90)
    
            # Set label alignment for better centering with vertical lines
            for label in plt.gca().get_xticklabels():
                label.set_horizontalalignment('center')
    
            # Add legend for valid intervals
            plt.legend(title="Valid Intervals", loc='upper right')
            plt.tight_layout()
            plt.show()
    
    
            
    @staticmethod
    def learned_parameters_one_home_plot(title:str, df: pd.DataFrame, propertycolors = []):
        """
        Plot learned temperatures for a single home
        """      
        fig, ax = plt.subplots()

        ax.grid(True)
        ax.set_title(title)

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel(r'$A\ [m^2]$')
        ax2.plot(df.index, df['A_m^2'], '.', label='A (effective solar aperture)', color='r', linestyle = 'dotted')  
        ax2.legend(loc=1)

        ax.plot(df.index, df['H_W_p_K'], '.', label='H (specific heat loss)', color='b', linestyle = 'solid')
        ax.plot(df.index, df['th_inertia_building_h'], '.', label='tau (thermal inertia)', color='g', linestyle = 'dashed') 


        # for property in propertycolors:
        #     ax.plot(df.index, df[property[0]], '.', label=property[0], color=property[1])

        ax.legend(loc=0);  # Add a legend.

        ax.set_xlabel('Datetime')  # Add an x-label to the axes.
        ax.set_ylabel(r'$H [W/K], tau [h]$')
        ax.tick_params(axis='x', labelrotation = 35)
        fig.subplots_adjust(bottom=0.2)
        
        plt.show()
 

    @staticmethod
    def learned_parameters_plot(df: pd.DataFrame, propertycolors = []):
        
        for home_id in df.index.levels[0]:
            Plot.learned_parameters_one_home_plot(f'Learned model parameters for home: {home_id}', 
                                                      df.loc[home_id], 
                                                      propertycolors = propertycolors)
            
    
    @staticmethod
    def learned_parameters_boxplot(title:str, df_results_model_parameters: pd.DataFrame, parameters = ['H_W_p_K', 'th_inertia_building_h', 'C_Wh_p_K']):
        """
        Visualize results of all learned model parameters of all homes in one box plot
        """

        # # TODO find a way to share the x-axis of all parameters,using parameters.index(parameter) 
        # fig, ax = plt.subplots(nrows=len(parameters), ncols=1, sharex=True)
        # fig.suptitle(title)  
        
        for parameter in parameters:
            (df_results_model_parameters
             [parameter]
             .reorder_levels(['start_horizon', 'home_id'])
             .unstack()
             .plot(kind='box', 
                   rot=35,
                   title=parameter)
            )

    
    @staticmethod
    def calculated_intervention_metrics_boxplot(df: pd.DataFrame, metrics: list):
        """
        Visualize box plots for calculated intervention metrics for different regimes of each ID.
        
        Parameters:
        - df: DataFrame containing the regime sequence numbers and metrics
        - metrics: List of metrics to plot
        """
        unique_ids = df.index.get_level_values('id').unique()
        
        for id_value in unique_ids:
            df_id = df.loc[id_value]  # Select data for the specific ID
            
            for metric in metrics:
                plt.figure(figsize=(8, 6))
                df_id.boxplot(column=metric, by='regime_sequence_number', grid=False)
                plt.title(f'ID: {id_value} - Boxplot of {metric}')  # Title with ID and metric
                plt.xlabel('Regime Sequence Number')
                plt.ylabel(metric)
                plt.suptitle("")  # Suppress the default suptitle            
                plt.tight_layout()
                plt.show()

    
    @staticmethod
    def learned_parameters_boxplot_b4b(df_results_model_parameters: pd.DataFrame, learned: str, actual: str, units_to_mathtext = None):
        """
        Visualize results of all learned model parameters of all homes in one box plot
        """
        
        # Group the DataFrame by 'id'
        grouped = df_results_model_parameters.groupby(level='id')

        # Create the figure and axes objects
        fig, ax = plt.subplots()

        # Create a box plot for the learned values
        boxplot_positions = np.arange(1, len(grouped) + 1) * 2 - 1
        learned_values = [group[learned].dropna() for _, group in grouped]  # Drop NaN values
        bp = ax.boxplot(learned_values,
                        positions=boxplot_positions,
                        boxprops={'facecolor': 'white', 'edgecolor': 'green'},
                        patch_artist=True,
                        showmeans=True,
                        meanline=True,
                        meanprops={'linestyle': '--', 'color': 'green'},
                        medianprops={'visible': False}
                       )        
        actual_line = dict(marker='<', markersize=8, linestyle='', color='red')

        if actual is not None:
            # Get the unique actual values
            actual_values = grouped[actual].unique()

            # Plot the actual values as red dots on top of the box plot
            ax.plot(np.arange(1, len(grouped) + 1) * 2, actual_values, **actual_line)

        # Set the x-axis labels
        ax.set_xticks(np.arange(1, len(grouped) + 1) * 2 - 0.5)
        ax.set_xticklabels(grouped.groups.keys())

        # Remove the tick lines
        ax.tick_params(axis='x', length=0)

        # Add vertical dashed lines between the box plots
        for i in range(len(grouped) - 1):
            ax.axvline((i + 1) * 2 + 0.5, linestyle='--', color='gray')
            
        # Create the legend
        legend_elements = []
        # Add the green box plot color and label for the learned values
        legend_elements.append(plt.Line2D([0], [0], color='green', lw=1, label=learned.split('__')[0]))
        # Add the red marker and label for the actual values if actual is not None
        if actual is not None:
            legend_elements.append(plt.Line2D([0], [0], **actual_line, label=actual.split('__')[0]))
            
        # Add the legend to the plot
        ax.legend(handles=legend_elements)
        plt.ylabel(units_to_mathtext[learned.split('__')[-1]] )
        
        # Show the plot
        plt.show()
        
    @staticmethod
    def learned_parameters_scatterplot(df: pd.DataFrame, parameters: list):
        """
        Visualize results of all learned model parameters of all homes in one box plot
        """
        df.plot.scatter(parameters[0], parameters[1])
        
        
        
    @staticmethod
    def features_scatter_plot(df: pd.DataFrame, features: list):
        """
        Visualize results of all learned model parameters of all homes in one box plot
        """
        colors = ['black', 'red', 'green']
        
        index_column_data = df.index.get_level_values(0)
        num = list(dict.fromkeys(index_column_data.to_list()))

        for i in range (len(num)):
            g = sns.PairGrid(df.loc[num[i]][features])
            g.map(sns.scatterplot)
            g.fig.suptitle(num[i])

    
    @staticmethod
    def plot_h3_cells_and_markers(h3_cell_ids, marker_df, output_file="map_with_h3_cells.html", num_closest_markers=0):
        """
        Plot H3 cells and markers on a Folium map, highlighting the specified number of nearest markers to each H3 cell.
    
        Parameters:
        - h3_cell_ids (list): List of H3 cell ids to plot.
        - marker_df (DataFrame): DataFrame with columns 'lat__degN', 'lon__degE', and 'popup_text' containing marker information.
        - output_file (str, optional): File path to save the HTML map. Default is 'map_with_h3_cells.html'.
        - num_closest_markers (int, optional): Number of closest markers to highlight in green and add to H3 cell popup text. Default is 3.
    
        Returns:
        - folium.Map: Folium map object with plotted markers and H3 cells.
        """
        
        def calculate_distance(marker_lat_lon, h3_center):
            return distance(marker_lat_lon, h3_center).kilometers
        
        def format_popup_text(cell_id, closest_markers_info):
            popup_text = f"{cell_id}<br>"
            for i, (popup, dist) in enumerate(closest_markers_info):
                popup_text += f"{popup} {dist:.2f} km<br>"
            return popup_text
        
        # Calculate H3 cell centers
        h3_centers = {cell_id: h3.h3_to_geo(cell_id) for cell_id in h3_cell_ids}
        
        # Collect all coordinates to determine the bounds, including H3 cell boundaries
        all_coords = [coord for coord in h3_centers.values()]
        for cell_id in h3_cell_ids:
            all_coords.extend(h3.h3_to_geo_boundary(cell_id))
        all_coords.extend(zip(marker_df['lat__degN'], marker_df['lon__degE']))
        
        if all_coords:
            # Determine the bounds of all points
            min_lat = min(coord[0] for coord in all_coords)
            max_lat = max(coord[0] for coord in all_coords)
            min_lon = min(coord[1] for coord in all_coords)
            max_lon = max(coord[1] for coord in all_coords)
    
            # Calculate the center of the bounds
            map_center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
            map_bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        else:
            # Default center if no coordinates are provided (Windesheim University of Applied Sciences in Zwolle)
            map_center = [52.5012, 6.0796]
            map_bounds = None
        
        # Create folium map
        mymap = folium.Map(location=map_center, zoom_start=7)
        
        if map_bounds:
            # Fit the map to the bounds
            mymap.fit_bounds(map_bounds)
        
        # Initialize empty dictionary to store closest markers for each cell
        closest_markers_dict = {cell_id: [] for cell_id in h3_cell_ids}
        
        # Add markers from marker_df
        if not marker_df.empty:
            for index, row in marker_df.iterrows():
                folium.Marker([row['lat__degN'], row['lon__degE']], popup=row['popup_text']).add_to(mymap)
                
                # Calculate distances to each H3 cell center and update closest markers list
                for cell_id in h3_cell_ids:
                    h3_center = h3_centers[cell_id]
                    marker_lat_lon = (row['lat__degN'], row['lon__degE'])
                    dist = calculate_distance(marker_lat_lon, h3_center)
                    closest_markers_dict[cell_id].append((row['popup_text'], dist))
        
        # Add H3 cells and center points
        for cell_id in h3_cell_ids:
            lat_lon = h3_centers[cell_id]
            h3_center = (lat_lon[0], lat_lon[1])
            
            # Sort closest markers by distance
            closest_markers = closest_markers_dict[cell_id]
            closest_markers.sort(key=lambda x: x[1])
            
            # Collect information for closest markers based on num_closest_markers parameter
            closest_markers_info = closest_markers[:num_closest_markers]
            
            # Format popup text for the H3 cell including closest marker information
            popup_text = format_popup_text(cell_id, closest_markers_info) if num_closest_markers > 0 else cell_id
            
            # Plot H3 cell and center point with updated popup text
            h3_marker = folium.Marker(lat_lon, icon=folium.Icon(color='red', icon='cross'), popup=popup_text)
            h3_marker.add_to(mymap)
            
            hexagon = h3.h3_to_geo_boundary(cell_id)
            folium.Polygon(locations=hexagon, color='blue', fill=True, fill_opacity=0.2, popup=popup_text).add_to(mymap)
            
            # Highlight the closest markers in green if num_closest_markers > 0
            if num_closest_markers > 0:
                for i, (popup, _) in enumerate(closest_markers_info):
                    marker_row = marker_df[marker_df['popup_text'] == popup].iloc[0]
                    folium.Marker([marker_row['lat__degN'], marker_row['lon__degE']], 
                                  popup=popup,
                                  icon=folium.Icon(color='green', icon=f'{i+1}')).add_to(mymap)
        
        # Save map to HTML file
        mymap.save(output_file)
        
        # Return the folium map object
        return mymap

    @staticmethod
    def nfh_property_per_id_boxplot(df, property_col, filter_value=True):
        """
        Create a boxplot for a specific property in the DataFrame, grouped by 'id' and filtered by a condition.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        property_col (str): The column name representing the property to be plotted.
        filter_value (bool): The value to filter the filter_col by. Defaults to True.

        Returns:
        None: Displays a boxplot.
        """

        # Calculate the mean of the property_col per id and sort in descending order
        mean_per_id = df.groupby(level='id')[property_col].mean().sort_values(ascending=False)

        # Extract 'id' and the selected property into a new DataFrame and drop missing values
        df_boxplot = df.reset_index()[['id', property_col]].dropna()

        # Convert 'id' to a categorical type based on the sorted 'id' values
        df_boxplot['id'] = pd.Categorical(df_boxplot['id'], categories=mean_per_id.index, ordered=True)
        df_boxplot = df_boxplot.sort_values('id')

        # Group by 'id' and collect the property_col values
        grouped = df_boxplot.groupby('id', observed=True)[property_col].apply(list)

        # Create a list of lists for the boxplot
        data = [grouped[id] for id in grouped.index]

        # Create the boxplot using matplotlib
        plt.figure(figsize=(12, 6))
        plt.boxplot(data, labels=grouped.index)
        plt.title(f'{property_col} per id (Sorted by High Average {property_col})')
        plt.xlabel('ID')
        plt.ylabel(property_col)
        plt.xticks(rotation=45)  # Rotate x labels if needed
        plt.show()


    @staticmethod
    def nfh_property_grouped_boxplot(df, property_col, groupby_level, filter_value=True):
        """
        Create a boxplot for a specific property in the DataFrame, grouped by a specified index level.
    
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        property_col (str): The column name representing the property to be plotted.
        groupby_level (str): The name of the index level to group by.
        filter_value (bool): A placeholder for future filtering. Defaults to True.
    
        Returns:
        None: Displays a boxplot.
        """
    
        # Ensure the specified index level exists
        if groupby_level not in df.index.names:
            raise ValueError(f"'{groupby_level}' is not an index level in the DataFrame.")
    
        # Calculate the mean of the property_col per group and sort in descending order
        mean_per_group = df.groupby(level=groupby_level)[property_col].mean().sort_values(ascending=False)
    
        # Reset the index and extract the relevant columns
        df_boxplot = df.reset_index()[[groupby_level, property_col]].dropna()
    
        # Convert the grouping level to a categorical type based on the sorted values
        df_boxplot[groupby_level] = pd.Categorical(df_boxplot[groupby_level], categories=mean_per_group.index, ordered=True)
        df_boxplot = df_boxplot.sort_values(groupby_level)
    
        # Group by the specified level and collect the property_col values
        grouped = df_boxplot.groupby(groupby_level, observed=True)[property_col].apply(list)
    
        # Create a list of lists for the boxplot
        data = [grouped[group] for group in grouped.index]
    
        # Create the boxplot using matplotlib
        plt.figure(figsize=(12, 6))
        plt.boxplot(data, labels=grouped.index)
        plt.title(f'{property_col} per {groupby_level} (Sorted by High Average {property_col})')
        plt.xlabel(groupby_level)
        plt.ylabel(property_col)
        plt.xticks(rotation=45)  # Rotate x labels if needed
        plt.show()
