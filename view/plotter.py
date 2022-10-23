import pylab as plt
import pandas as pd
from importlib import reload
from datetime import datetime, timedelta


class Plot:

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
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        
        fig.suptitle(title)  

        #add DataFrames to subplots
        df[temp_plot_dict.keys()].plot(
            ax=ax[0],
            secondary_y=temp_plot_2nd_list,
            grid=True, 
            legend=True, 
            color=temp_plot_dict,
            xlabel=shared_x_label,
            ylabel = temp_y_label
        )
        ax[0].right_ax.set_ylabel(temp_2nd_y_label)

        df[power_plot_dict.keys()].plot(
            ax=ax[1],
            secondary_y=power_plot_2nd_list, 
            mark_right=True, 
            grid=True, 
            legend=True, 
            color=power_plot_dict,
            xlabel=shared_x_label,
            ylabel = power_y_label
        )
        ax[1].right_ax.set_ylabel(power_2nd_y_label)

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
        
        for home_id in df.index.levels[0]:
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
                                                   sanity_threshold = 0.9,
                                                   shared_x_label='Datetime',
                                                   temp_plot_dict = {},
                                                   temp_y_label = r'$Temperature [^oC]$',
                                                   temp_plot_2nd_list = [],
                                                   temp_2nd_y_label = r'$Wind speed [m/s]$',
                                                   power_plot_dict = {},
                                                   power_y_label = 'Power [W]',
                                                   power_plot_2nd_list = [],
                                                   power_2nd_y_label = r'$I\ [W/m^2]$'):

        first_day = df.index.min()
        last_day = df.index.max()
        for moving_horizon_start in pd.date_range(start=first_day, end=last_day, inclusive='left', freq='7D'):
            moving_horizon_end = min(last_day, moving_horizon_start + timedelta(days=7))

            if (moving_horizon_end < last_day):
                df_moving_horizon = df[moving_horizon_start:moving_horizon_end].iloc[:-1]
            else:
                df_moving_horizon = df[moving_horizon_start:last_day]

            moving_horizon_end = df_moving_horizon.index.max()


            print(f'Learned model parameters for home: {home_id} from {moving_horizon_start} to {moving_horizon_end} with sanity {df_moving_horizon.sanity.astype(float).mean():.2f}')
            if (df_moving_horizon.sanity.astype(float).mean() > sanity_threshold):
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
        plt.show()
 

    @staticmethod
    def learned_parameters_plot(df: pd.DataFrame, propertycolors = []):
        
        for home_id in df.index.levels[0]:
            Plot.learned_parameters_one_home_plot(f'Learned model parameters for home: {home_id}', 
                                                      df.loc[home_id], 
                                                      propertycolors = propertycolors)
            
    
    
