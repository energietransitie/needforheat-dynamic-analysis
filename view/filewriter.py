import pylab as plt
import pandas as pd

class ExcelWriter:

    @staticmethod
    def write(df_source:pd.DataFrame, filename:str):
        df = df_source.copy()
        df.reset_index(inplace=True)
        for col in (df.select_dtypes(['datetimetz']).columns):
            df[col] = df[col].dt.tz_localize(None)
        for col in (df.select_dtypes(['bool']).columns):
            df[col] = df[col].map({True: "TRUE", False: "FALSE"})
        df.to_excel(filename)
            
