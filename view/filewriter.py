import pylab as plt
import pandas as pd

class ExcelWriter:

    @staticmethod
    def write(df_source:pd.DataFrame, filename:str):
        df = df_source
        for col in (df.select_dtypes(['datetimetz']).columns):
            df[col] = df[col].dt.tz_localize(None)
        df.tz_localize(None, level=0).to_excel(filename)        
