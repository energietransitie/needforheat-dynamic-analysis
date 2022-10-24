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
            marker=".",
            ms=2,
            linestyle='None',
            secondary_y=temp_plot_2nd_list,
            grid=True, 
            legend=True, 
            color=temp_plot_dict,
            xlabel=shared_x_label,
            ylabel = temp_y_label
        )
        
        if len(temp_plot_2nd_list) >0:
            ax[0].right_ax.set_ylabel(temp_2nd_y_label)

        df[power_plot_dict.keys()].plot(
            ax=ax[1],
            secondary_y=power_plot_2nd_list, 
            marker=".",
            ms=2,
            linestyle='None',
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


            print(f'Learned model parameters for home: {home_id} from {moving_horizon_start} to {moving_horizon_end}')
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
                   rot=90,
                   title=parameter)
            )
