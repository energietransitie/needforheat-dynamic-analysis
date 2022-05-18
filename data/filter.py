import pandas as pd


class Filter:
    """
    This class is able to use filters from other classes (delegates) and return the filtered result from these
    delegates by masking them or returning the results grouped by periods.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the Filter and adds a new column called valid and sets all the values within the column to True
        """

        data["valid"] = True
        self.data = data

    def filter(self, delegate: classmethod):
        """Uses the delegate to filter the Filter data"""
        self.data = delegate

    def get_result(self) -> pd.DataFrame:
        """Returns data from the Filter as a DataFrame"""
        return self.data

    def get_grouped_result(self) -> pd.DataFrame:
        """
        Returns the filtered data periods in seperate DataFrames in a list.
        The data is seperated where a valid == False.
        """

        data_grouped_result = []
        data_grouped = pd.DataFrame(columns=self.data.columns)

        i = 0
        while i < len(self.data.index):
            if self.data["valid"].iloc[i]:
                data = [
                    self.data["index"].iloc[i],
                    self.data["value"].iloc[i],
                    self.data["datetime"].iloc[i],
                    self.data["timestamp"].iloc[i],
                    self.data["valid"].iloc[i],
                ]
                data_grouped.loc[len(data_grouped.index)] = data
            else:
                if len(data_grouped.index) > 0:
                    data_grouped_result.append(data_grouped)
                    data_grouped = pd.DataFrame(columns=self.data.columns)
            i += 1

        if len(data_grouped.index) > 0:
            data_grouped_result.append(data_grouped)

        return data_grouped_result
