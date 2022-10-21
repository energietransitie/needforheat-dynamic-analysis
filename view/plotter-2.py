import pylab as plt
import pandas as pd

class Plot:

    @staticmethod
    def temperature_and_power_plot(df: pd.DataFrame, 
                                   temp_plot_dict = {},
                                   temp_plot_2nd_list = [],
                                   power_plot_dict = {},
                                   power_plot_2nd_list = []):
        """
        Plot of temperature and power data 
        """
        
        #define subplot layout
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        
        #add DataFrames to subplots
        df[temp_plot_dict.keys()].plot(
            ax=axes[0],
            secondary_y=temp_plot_2nd_list,
            grid=True, 
            legend=True, 
            color=temp_plot_dict,
            xlabel='Datetime',
            ylabel = r'$Temperature [^oC]$'
        )

        df[power_plot_dict.keys()].plot(
            ax=axes[1],
            secondary_y=power_plot_2nd_list, 
            mark_right=True, 
            grid=True, 
            legend=True, 
            color=power_plot_dict,
            xlabel='Datetime',
            ylabel = 'Power [W]'
        )

        plt.show()