import pylab as plt
import pandas as pd

class Plot:

    @staticmethod
    def weather_and_other_temperatures(title:str, df: pd.DataFrame, propertycolors = []):
        """
        Temperature data plot of weather data and a list of other temperature properties
        """
        
        fig, ax = plt.subplots()

        ax.grid(True)

        ax.set_title(title)  

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel(r'$I\ [W/m^2]$')
        ax2.plot(df.index, df['irradiation_hor_avg_W_p_m2'], '.', label='global horizontal irradiation', alpha=0.5, color='y')  
        ax2.legend(loc=1)

        ax.plot(df.index, df['T_out_avg_C'], '.', label='outdoor temperature', color='orange')
        ax.plot(df.index, df['wind_avg_m_p_s'], '.', label='wind speed', color='c')  
        ax.plot(df.index, df['T_out_e_avg_C'], '.', label='effective outdoor temperature', color='b')

        for property in propertycolors:
            ax.plot(df.index, df[property[0]], '.', label=property[0], color=property[1])

        ax.legend(loc=0);  # Add a legend.

        ax.set_xlabel('Datetime')  # Add an x-label to the axes.
        ax.set_ylabel(r'$T_{out} [^oC], U [m/s]$')
        plt.show()