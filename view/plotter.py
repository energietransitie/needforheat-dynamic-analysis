import pylab as plt
import pandas as pd
from importlib import reload
from datetime import datetime, timedelta
import numpy as np
import pylab as plt
import seaborn as sns
from preprocessor import Preprocessor


class Plot:

    
    @staticmethod
    def dataframe_properties_plot(df: pd.DataFrame, units_to_mathtext = None): 
        """
        Plot data in df DataFrame, one plot per id, one subplot for all propertyes with the same unit
        
        in: dataframe with
        - index = ['id', 'source', 'timestamp']
            - id: id of e.g. home / utility building / room 
            - source: device_type from the database
            - timestamp: timezone-aware timestamp
        - columns = all properties in the input column
            - unit types are encoded as last part of property name, searated by '__'
        - units_to_mathtext: table tat translates property unit postfixes to mathtext.
        """      
        
        for id in list(df.index.to_frame(index=False).id.unique()):
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
    def dataframe_preprocessed_plot(df: pd.DataFrame, units_to_mathtext = None): 
        """
        Plot data in df DataFrame, one plot per id, one subplot for all propertyes with the same unit
        
        in: dataframe with
        - index = ['id', 'timestamp']
            - id: id of e.g. home / utility building / room 
            - timestamp: timezone-aware timestamp
        - columns = all properties in the input column
            - unit types are encoded as last part of property name, searated by '__'
        - units_to_mathtext: table tat translates property unit postfixes to mathtext.
        """      
        
        for id in list(df.index.to_frame(index=False).id.unique()):
            try:
                df_plot = df.loc[id]
                props_with_data = [prop for prop in list(df_plot.columns) if df_plot[prop].count()>0] 
                units_with_data = np.unique(np.array([prop.split('__')[-1] for prop in props_with_data]))
                unit_tuples = [tuple([prop.split('__')[0] for prop in props_with_data if prop.split('__')[-1] == unit]) for unit in units_with_data]
                props_with_data = [prop.split('__')[0] for prop in props_with_data]
                labels = [col.split('__')[0] for col in df_plot.columns]
                df_plot.columns = labels
                axes = df_plot[props_with_data].plot(
                    subplots = unit_tuples,
                    style='.--',
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
        ax.plot(df.index, df['tau_h'], '.', label='tau (thermal inertia)', color='g', linestyle = 'dashed') 


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
    def learned_parameters_boxplot(title:str, df_results_model_parameters: pd.DataFrame, parameters = ['H_W_p_K', 'tau_h', 'C_Wh_p_K']):
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
    def working_days_scatter_plot(df_prep: pd.DataFrame, features: list, period: list):
        # Non-working days dataframe
        dfs=[]
        for start_date, end_date in period:
            for home_id, home_data in df_prep.groupby('id'):
                included_df = home_data.loc[(home_data.index.get_level_values(1) >= start_date) & (home_data.index.get_level_values(1) <= end_date)]
                dfs.append((home_id, included_df))
        df_non_working_days = pd.concat([df for _, df in dfs]) 
        
        # Working days dataframe
        df_working_days = df_prep.copy()
        for start_date, end_date in period:
            df_working_days = df_working_days.loc[~((df_working_days.index.get_level_values(1) >= start_date) & (df_working_days.index.get_level_values(1) <= end_date))]

        
        df_list = [df_working_days, df_non_working_days]
        df_names = ['Working days', 'Non working days']

        for x, df in enumerate(df_list): 
            
            unique_ids = df.index.get_level_values("id").unique()
            num_rows = len(unique_ids) * len(features)
            fig, axes = plt.subplots(num_rows, 1, figsize=(10, 5 * num_rows), sharex=True)
            counter=0
            
            for idx, id_ in enumerate(unique_ids):
                subset_df = df.loc[id_]       
                for j, feature in enumerate(features):
                    axes[counter].scatter(subset_df.index, subset_df[features[j]], label=features[j])
                    # axes[counter].set_title(f"Id {id_}")
                    axes[counter].set_title('{} | ID: {}'.format(df_names[x], id_))
                    axes[counter].set_xlabel("Date and time")
                    axes[counter].set_ylabel(features[j])
                    axes[counter].legend()
                    counter +=1