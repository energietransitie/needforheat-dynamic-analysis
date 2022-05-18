import datetime
from typing import Optional, List, Literal, Dict, Iterable, Union, Callable

import pandas as pd
import numpy as np

from period import Period
from database import Database


# before import use pip install git+https://github.com/stephanpcpeters/HourlyHistoricWeather.git#egg=historicdutchweather
# or include the following line (without comment) in requirements.txt and run `pip install -r requirements.txt` in a terminal
# -e git+https://github.com/stephanpcpeters/HourlyHistoricWeather.git#egg=historicdutchweather
import historicdutchweather



Operator = Literal[
    "=",  # Equal to
    ">",  # Greater than
    "<",  # Less than
    ">=",  # Greater than or equal to
    "<=",  # Less than or equal to
    "<>",  # Not equal to
]


class Extractor(Database):
    """
    Use the extractor to collect rows from the database or create filtered groups.
    """

    __house: int

    __period: Period = Period()

    _from: str = """
        FROM measurement m 
            JOIN property pr ON pr.id = m.property_id
            JOIN device d ON d.id = m.device_id
            JOIN building b ON b.id = d.building_id
            JOIN account a on a.id = b.id
    """

    def __init__(self, house: int, period: Optional[Period] = None):
        self.__house = house

        if period is not None:
            self.__period = period

    def set_period(self, period: Optional[Period] = None):
        """
        Set the period for the current query.
        """

        if period is None:
            self.__period.start = None
            self.__period.end = None
        else:
            self.__period = period

    def set_start(self, date: Optional[datetime.datetime] = None):
        """
        Set the start datetime for the current extraction.
        """

        self.__period.start = date

    def set_end(self, date: Optional[datetime.datetime] = None):
        """
        Set the end datetime for the current extraction.
        """

        self.__period.end = date

    def read(self, path: str = "data", sensor: str = None) -> pd.DataFrame:
        """
        Extract the raw dataset and add a datetime column.
        """

        location = (
            "%s.csv" % self.__house
            if path == ""
            else "%s/%s.csv" % (path.rstrip("/\\").strip(), self.__house)
        )

        raw_df = pd.read_csv(location)
        raw_df["datetime"] = pd.to_datetime(
            raw_df["timestamp"], format="%Y-%m-%d %H:%M:%S"
        ).dt.tz_localize(None)
        if self.__period.start is not None:
            raw_df = raw_df[raw_df["datetime"] >= self.__period.start]
        if self.__period.end is not None:
            raw_df = raw_df[raw_df["datetime"] <= self.__period.end]

        raw_df["timestamp"] = raw_df["datetime"].apply(lambda t: t.timestamp())
        raw_df = raw_df[["Unnamed: 0", "property", "value", "datetime", "timestamp"]]
        raw_df.columns = ["index", "property", "value", "datetime", "timestamp"]

        if sensor is not None:
            return raw_df[raw_df["property"] == sensor].drop(["property"], axis=1)

        return raw_df.drop(["property"], axis=1)

    def get(self, name: str, operator: Optional[Operator] = None, value: Optional[float] = None) -> pd.DataFrame:
        """
        Select all or filtered values for a house within the given date range.
        """

        var = {
            'name': name,
            'house': self.__house
        }

        where = ""
        if operator is not None and value is not None:
            where = ' AND m.value ' + operator + ' %(value)s'
            var['value'] = value

        return self.__get(where, var)

    def get_between(self, name: str, lower: float, upper: float) -> pd.DataFrame:
        """
        Select all values between a range for a house within the given date range.
        """

        var = {
            'name': name,
            'house': self.__house,
            'lower': lower,
            'upper': upper,
        }

        return self.__get(' AND m.value BETWEEN %(lower)s AND %(upper)s', var)

    def get_in(self, name: str, values: Iterable[float]) -> pd.DataFrame:
        """
        Select all values that are in the array for a house within the given date range.
        """

        var = {
            'name': name,
            'house': self.__house,
        }

        if len(values) < 1:
            return pd.DataFrame(columns=["index", "value", "datetime", "timestamp"])

        return self.__get(' AND m.value IN (' + ', '.join(list(map(float, values))) + ')', var)

    def __get(self, where: str, var: Dict[str, any]) -> pd.DataFrame:

        self._connect()
        cursor = self._db.cursor()

        start = ""
        if self.__period.start:
            start += " AND m.timestamp >= %(start)s"
            var['start'] = "%s" % self.__period.start

        end = ""
        if self.__period.end:
            end += " AND m.timestamp <= %(end)s"
            var['end'] = "%s" % self.__period.end

        sql = """
            SELECT 
                m.id AS 'index', 
                m.value AS 'value', 
                m.timestamp AS 'datetime', 
                UNIX_TIMESTAMP(m.timestamp) AS 'timestamp'
            """ + self._from + """
            WHERE 
                a.pseudonym = %(house)s AND 
                pr.name = %(name)s """ + start + end + where + """
            ORDER BY m.timestamp
        """

        cursor.execute(sql, var)
        rows = cursor.fetchall()
        self._close()

        return pd.DataFrame(rows, columns=["index", "value", "datetime", "timestamp"])

    def get_periods(self, data: Union[str, pd.DataFrame], operator: Operator, value: Optional[float] = None, singles: bool = False) -> List[Period]:
        """
        Get a list of periods from which the filters were true.
        """

        if isinstance(data, str):
            return self.__get_periods_from_database(data, operator, value, singles)

        return self.__get_periods_from_dataframe(data, operator, value, singles)

    def get_custom_periods(self, data: Union[str, pd.DataFrame], delegate: Union[Callable[[float, float], bool], Callable[[float, float, datetime.datetime, datetime.datetime], bool]]) -> List[Period]:
        """
        Filter data with a custom function and obtain the periods that match the conditions.
        Use a delegate with before and after values as float and optionally add the before and after methods.
        Example:
        def delegate(before: float, after: float, before_datetime: datetime.datetime, after_datetime: datetime.datetime) -> bool:
            return before > after
        """

        if isinstance(data, str):
            data = self.get(data)

        result = []
        length = len(data.index)

        if delegate.__code__.co_argcount == 2:
            def comparison(before, after, before_datetime, after_datetime):
                del after_datetime, before_datetime
                return delegate(before, after)
        else:
            comparison = delegate

        i = 0
        while i < length:
            date = data["datetime"].iloc[i]

            if self.__is_valid_date(date):

                value = data["value"].iloc[i]

                next_value = False
                if i + 1 < length:
                    next_date = data['datetime'].iloc[i + 1]
                    if self.__is_valid_date(next_date):
                        next_value = comparison(value, data['value'].iloc[i + 1], date, next_date)

                prev_value = False
                if i - 1 >= 0:
                    prev_date = data['datetime'].iloc[i - 1]
                    if self.__is_valid_date(prev_date):
                        prev_value = comparison(data['value'].iloc[i - 1], value, prev_date, date)

                if not next_value and prev_value:
                    result[-1].end = date

                    if result[-1].end <= result[-1].start:
                        del result[-1]

                elif next_value and not prev_value:
                    result.append(Period(date))

            i += 1

        return result

    def __get_periods_from_database(self, data: str, operator: Operator, value: Optional[float] = None, singles: bool = False) -> List[Period]:
        self._connect()
        cursor = self._db.cursor()

        var = {
            'name': data,
            'house': self.__house
        }

        start = ""
        if self.__period.start is not None:
            start += " AND m.timestamp >= %(start)s"
            var['start'] = self.__period.start

        end = ""
        if self.__period.end is not None:
            end += " AND m.timestamp <= %(end)s"
            var['end'] = self.__period.end

        if value is None:
            sql = """
                SELECT 
                    date_start, 
                    date_end 
                FROM (
                    SELECT 
                        ROW_NUMBER() OVER(ORDER BY timestamp) AS 'i', 
                        timestamp AS 'date_start', 
                        LEAD(timestamp) OVER(ORDER BY timestamp) AS 'date_end' 
                    FROM (
                        SELECT 
                            timestamp 
                        FROM (
                            SELECT 
                                m.timestamp AS 'timestamp', 
                                LAG(m.value) OVER(ORDER BY timestamp) AS 'prev_value', 
                                m.value AS 'curr_value', 
                                LEAD(m.value) OVER(ORDER BY timestamp) AS 'next_value' 
                            """ + self._from + """
                            WHERE 
                                a.pseudonym = %(house)s AND 
                                pr.name = %(name)s """ + start + end + """
                            ORDER BY m.timestamp
                        ) t
                        WHERE (
                            curr_value """ + operator + """ prev_value AND (
                                next_value IS NULL OR NOT (
                                    next_value """ + operator + """ curr_value
                                )
                            )
                        ) OR (
                            next_value """ + operator + """ curr_value AND (
                                prev_value IS NULL OR NOT (
                                    curr_value """ + operator + """ prev_value
                                )
                            )
                        )
                    ) t2
                ) t1 
                WHERE 
                    i % 2 = 1 AND 
                    date_start IS NOT NULL AND 
                    date_end IS NOT NULL AND 
                    date_end > date_start
            """

        else:
            var['value'] = value
            sql = """
                SELECT * 
                FROM (
                    SELECT 
                        timestamp AS 'date_start', 
                        LEAD(prev_timestamp) OVER(ORDER BY - i DESC) AS 'date_end' 
                    FROM (
                        SELECT 
                            i, 
                            CASE WHEN value """ + operator + """ %(value)s THEN NULL ELSE prev_timestamp END AS 'prev_timestamp',
                            CASE WHEN prev_value """ + operator + """ %(value)s THEN NULL ELSE timestamp END AS 'timestamp',
                            prev_value, 
                            value 
                        FROM (
                            SELECT 
                                ROW_NUMBER() OVER(ORDER BY timestamp) AS 'i', 
                                LAG(m.timestamp) OVER(ORDER BY timestamp) AS 'prev_timestamp', 
                                m.timestamp AS 'timestamp', 
                                LAG(m.value) OVER(ORDER BY timestamp) AS 'prev_value', 
                                m.value AS 'value' 
                            """ + self._from + """
                            WHERE 
                                a.pseudonym = %(house)s AND 
                                pr.name = %(name)s """ + start + end + """
                            ORDER BY m.timestamp
                        ) t 
                        WHERE (
                            value """ + operator + """ %(value)s AND (
                                prev_value IS NULL OR NOT (
                                    prev_value """ + operator + """ %(value)s
                                )
                            )
                        ) OR (
                            prev_value """ + operator + """ %(value)s AND NOT (
                                value """ + operator + """ %(value)s
                            )
                        )
                        UNION 
                        SELECT 
                            NULL AS 'i', 
                            (
                                SELECT 
                                    MAX(timestamp) 
                                """ + self._from + """
                                WHERE 
                                    a.pseudonym = %(house)s AND 
                                    pr.name = %(name)s """ + start + end + """
                            ) AS 'prev_timestamp', 
                            NULL AS 'timestamp',
                            0 AS 'prev_value',
                            0 AS 'value'
                    ) t2
                ) t1 
                WHERE 
                    date_start IS NOT NULL AND 
                    date_end IS NOT NULL AND 
                    date_end >""" + ("=" if singles else "") + """ date_start
            """

        cursor.execute(sql, var)
        rows = cursor.fetchall()

        self._close()

        result = []
        for row in rows:
            result.append(Period(row[0], row[1]))

        return self.close_period_gaps(result, datetime.timedelta())

    def __get_periods_from_dataframe(self, data: pd.DataFrame, operator: Operator, value: Optional[float] = None, singles: bool = False) -> List[Period]:

        comparators = {
            "=": lambda before, after: after == before,
            ">": lambda before, after: after > before,
            "<": lambda before, after: after < before,
            ">=": lambda before, after: after >= before,
            "<=": lambda before, after: after <= before,
            "<>": lambda before, after: after != before,
        }

        comparison = comparators[operator]

        if value is None:
            return self.get_custom_periods(data, comparison)

        result = []
        period = None
        length = len(data.index)
        prev = False

        i = 0
        while i < length:
            date = data["datetime"].iloc[i]

            if self.__is_valid_date(date):

                current = comparison(value, data["value"].iloc[i])

                if not prev and current:
                    period = Period(date)

                elif period is not None and prev and not current:
                    period.end = data["datetime"].iloc[i - 1]

                    if singles or period.end > period.start:
                        result.append(period)

                    period = None

                prev = current

            i += 1

        lastdate = data["datetime"].iloc[-1]
        if period is not None and prev and (singles or lastdate > period.start):
            period.end = lastdate
            result.append(period)

        return self.close_period_gaps(result, datetime.timedelta())

    def get_houses(self) -> pd.DataFrame:
        """
        Get a DataFrame with all houses.
        id, pseudonym
        """

        self._connect()
        cursor = self._db.cursor()

        cursor.execute("SELECT id, pseudonym FROM house")
        rows = cursor.fetchall()
        self._close()

        return pd.DataFrame(rows, columns=['id', 'pseudonym'])

    def get_accounts(self) -> pd.DataFrame:
        """
        Get a DataFrame with all houses.
        id, pseudonym
        """

        self._connect()
        cursor = self._db.cursor()

        cursor.execute("SELECT id, pseudonym FROM account")
        rows = cursor.fetchall()
        self._close()

        return pd.DataFrame(rows, columns=['id', 'pseudonym'])

    def get_locations(self) -> pd.DataFrame:
        """
        Get a DataFrame with all locations.
        id, name, longitude, latitude
        """

        self._connect()
        cursor = self._db.cursor()

        cursor.execute("SELECT * FROM location")
        rows = cursor.fetchall()
        self._close()

        return pd.DataFrame(rows, columns=['id', 'name', 'longitude', 'latitude'])

    def get_parameters(self) -> pd.DataFrame:
        """
        Get a DataFrame with all parameters that can be used to store and request data.
        id, name, unit
        """

        self._connect()
        cursor = self._db.cursor()

        cursor.execute("SELECT * FROM parameter")
        rows = cursor.fetchall()
        self._close()

        return pd.DataFrame(rows, columns=['id', 'name', 'unit'])

    def __is_valid_date(self, date: datetime.datetime) -> bool:
        return (self.__period.start is None or self.__period.start <= date) and (self.__period.end is None or date <= self.__period.end)

    @staticmethod
    def merge_periods(periods1: List[Period], periods2: List[Period], union: bool = False) -> List[Period]:
        """
        Merge two Lists of Periods into one List. The Lists are assumed to be sorted.

        The standard method is by taking their intersection, returning only Periods
        that are included in both (this includes partly overlapping Periods, it
        does not require the Periods to have the same start and end dates).

        If union is set to True, it returns a union of the Lists, returning Periods for all
        periods of time where at least one of the lists contains one. Periods that overlap are
        merged into one to avoid overlapping periods in the returned List.
        """
        merged_periods = []
        len1 = len(periods1)
        len2 = len(periods2)

        if union:
            # Merge lists, this is the merge part of a merge sort, since the input lists are assumed to be sorted
            i = 0  # index for periods1
            j = 0  # index for periods2
            combined = []

            while i < len1 or j < len2:
                if i < len1 and periods1[i] <= periods2[j]:
                    combined.append(periods1[i])
                    i += 1
                else:
                    combined.append(periods2[j])
                    j += 1

            if len(combined) < 2:
                return combined

            # Loop through combined, extending combined[i] if combined[j] extends it,
            # do nothing if combined[j] falls entirely within it and
            # append combined[i] to merged_periods if it doesn't overlap at all.
            i = 0
            j = 1
            lenc = len1 + len2
            while j < lenc:
                if combined[j].start <= combined[i].end:
                    if combined[j].end > combined[i].end:
                        combined[i].end = combined[j].end
                    j += 1
                else:
                    merged_periods.append(combined[i])
                    i = j

            # Add the final, still unhandled, element
            merged_periods.append(combined[i])

        else:  # Intersect
            index1 = index2 = 0

            while index1 < len1 and index2 < len2:
                current1 = periods1[index1]
                current2 = periods2[index2]

                # continue if the current Periods don't overlap at all
                if current1.start > current2.end:
                    index2 += 1
                    continue

                if current2.start > current1.end:
                    index1 += 1
                    continue

                # check how the current Periods overlap
                if current1.start >= current2.start:
                    if current1.end <= current2.end:
                        merged_periods.append(Period(current1.start, current1.end))
                        index1 += 1
                    else:
                        merged_periods.append(Period(current1.start, current2.end))
                        index2 += 1
                else:
                    if current1.end < current2.end:
                        merged_periods.append(Period(current2.start, current1.end))
                        index1 += 1
                    else:
                        merged_periods.append(Period(current2.start, current2.end))
                        index2 += 1

        return merged_periods

    @staticmethod
    def close_period_gaps(periods: List[Period], timedelta: datetime.timedelta) -> List[Period]:
        """
        Close the gaps between the periods in the list if the gap is less than or equal to the given timedelta.
        """

        length = len(periods)
        if length < 2:
            return periods

        i = 1
        result = []
        period = periods[0]
        while i < length and period.end is not None:
            if periods[i].start - period.end <= timedelta or periods[i].start < period.end:
                if periods[i].end is None or periods[i].end > period.end:
                    period.end = periods[i].end
            else:
                result.append(period)
                period = periods[i]

            i += 1

        result.append(period)

        return result

    def get_home_parameter_timeseries_sum(self, parameter: str, seriesname: str, 
                                 differentiate: bool, upsample_to: str, downsample_to: str,
                                 start: datetime, end: datetime, tz_name_home: str) -> pd.DataFrame:
        tz_system = 'UTC'
        timeseriesdata = self.get(parameter)
        timeseriesdata.set_index('datetime', inplace=True)

        if tz_system == tz_name_home:
            timeseriesdata = timeseriesdata.tz_localize(tz_system)
        else:
            timeseriesdata = timeseriesdata.tz_localize(tz_system).tz_convert(tz_name_home)

        #first sort on datetime index
        timeseriesdata.sort_index(inplace=True)
        # tempting, but do NOT drop duplicates since this ignores index column
        # timeseriesdata.drop_duplicates(inplace=True)


        # Converting str to float
        timeseriesdata['value'] = timeseriesdata['value'].astype(float)

        if differentiate:
            timeseriesdata['value'] = timeseriesdata['value'].diff().shift(-1)

        #remove static  outliers
        timeseriesdata = Extractor.remove_measurement_outliers(timeseriesdata, 3.0)

        timeseriesdata_minute = timeseriesdata.resample(upsample_to).first()

        timeseriesdata_minute.interpolate(method='time', inplace=True)
        timeseriesdata_minute['target_value'] = timeseriesdata_minute['value']
        timeseriesdata_minute.rename(columns={'target_value':seriesname}, inplace=True)
        timeseriesdata_minute.drop(['index', 'timestamp', 'value'], axis=1, inplace=True)
        timeseriesdata_by_day = timeseriesdata_minute[seriesname].resample(downsample_to).sum()

        # print(seriesname)
        # print(timeseriesdata_by_day.describe())

        return timeseriesdata_by_day

    def get_meter_parameter_timeseries_sum(self, parameter: str, seriesname: str, 
                                 differentiate: bool, upsample_to: str, downsample_to: str,
                                 start: datetime, end: datetime, tz_name_home: str) -> pd.DataFrame:
        tz_system = 'UTC'
        timeseriesdata = self.get(parameter)
        timeseriesdata.set_index('datetime', inplace=True)
        if tz_system == tz_name_home:
            timeseriesdata = timeseriesdata.tz_localize(tz_system)
        else:
            timeseriesdata = timeseriesdata.tz_localize(tz_system).tz_convert(tz_name_home)

        #first sort on datetime index
        timeseriesdata.sort_index(inplace=True)
        # tempting, but do NOT drop duplicates since this ignores index column
        # timeseriesdata.drop_duplicates(inplace=True)

        # Converting str to float
        timeseriesdata['value'] = timeseriesdata['value'].astype(float)


        # meter values should always be rising monotonic 
        # meter value resets, which occasionaly happen, should be removed
        # also, small occasional negative meter jumps should be ignored
        # so first calculate diff, filter out the negative jumps and recalculate the meter value, starting at zero
        timeseriesdata['value'] = timeseriesdata['value'].diff()

        #remove static  outliers
        timeseriesdata = Extractor.remove_measurement_outliers(timeseriesdata, 3.0)

        timeseriesdata['value'] = timeseriesdata['value'].diff().fillna(0).clip(0,None).cumsum()


        timeseriesdata_minute = timeseriesdata.resample(upsample_to).first()
        timeseriesdata_minute.interpolate(method='time', inplace=True)
        if differentiate:
            timeseriesdata_minute['value'] = timeseriesdata_minute['value'].diff().shift(-1)
        timeseriesdata_minute['target_value'] = timeseriesdata_minute['value']
        timeseriesdata_minute.rename(columns={'target_value':seriesname}, inplace=True)
        timeseriesdata_minute.drop(['index', 'timestamp', 'value'], axis=1, inplace=True)
        timeseriesdata_by_day = timeseriesdata_minute.resample(downsample_to).sum()

        # print(seriesname)
        # print(timeseriesdata_by_day.describe())

        return timeseriesdata_by_day

    def get_home_parameter_timeseries_count(self, parameter: str, seriesname: str, 
                                 differentiate: bool, upsample_to: str, downsample_to: str,
                                 start: datetime, end: datetime, tz_name_home: str) -> pd.DataFrame:
        tz_system = 'UTC'
        timeseriesdata = self.get(parameter)
        timeseriesdata.set_index('datetime', inplace=True)
        if tz_system == tz_name_home:
            timeseriesdata = timeseriesdata.tz_localize(tz_system)
        else:
            timeseriesdata = timeseriesdata.tz_localize(tz_system).tz_convert(tz_name_home)

        #first sort on datetime index
        timeseriesdata.sort_index(inplace=True)
        # tempting, but do NOT drop duplicates since this ignores index column
        # timeseriesdata.drop_duplicates(inplace=True)

        # Converting str to float
        timeseriesdata['value'] = timeseriesdata['value'].astype(float)

        timeseriesdata_minute = timeseriesdata.resample(upsample_to).first()
        timeseriesdata_minute.interpolate(method='time', inplace=True)
        if differentiate:
            timeseriesdata_minute['value'] = timeseriesdata_minute['value'].diff().shift(-1)
        timeseriesdata_minute['target_value'] = timeseriesdata_minute['value']
        timeseriesdata_minute.rename(columns={'target_value':seriesname}, inplace=True)
        timeseriesdata_minute.drop(['index', 'timestamp', 'value'], axis=1, inplace=True)
        timeseriesdata_by_day = timeseriesdata_minute.resample(downsample_to).count()

        # print(seriesname)
        # print(timeseriesdata_by_day.describe())

        return timeseriesdata_by_day

    def get_home_parameter_timeseries_mean(self, parameter: str, seriesname: str, 
                                 upsample_to: str, downsample_to: str,
                                 start: datetime, end: datetime, tz_name_home: str) -> pd.DataFrame:
        tz_system = 'UTC'
        timeseriesdata = self.get(parameter)
        timeseriesdata.set_index('datetime', inplace=True)
        if tz_system == tz_name_home:
            timeseriesdata = timeseriesdata.tz_localize(tz_system)
        else:
            timeseriesdata = timeseriesdata.tz_localize(tz_system).tz_convert(tz_name_home)

        #first sort on datetime index
        timeseriesdata.sort_index(inplace=True)
        # tempting, but do NOT drop duplicates since this ignores index column
        # timeseriesdata.drop_duplicates(inplace=True)


        # Converting str to float
        timeseriesdata['value'] = timeseriesdata['value'].astype(float)

        #remove static  outliers
        timeseriesdata = Extractor.remove_measurement_outliers(timeseriesdata, 3.0)

        timeseriesdata_minute = timeseriesdata.resample(upsample_to).first()
        timeseriesdata_minute.interpolate(method='time', inplace=True)
        timeseriesdata_minute['target_value'] = timeseriesdata_minute['value']
        timeseriesdata_minute.rename(columns={'target_value':seriesname}, inplace=True)
        timeseriesdata_minute.drop(['index', 'timestamp', 'value'], axis=1, inplace=True)
        timeseriesdata_by_day = timeseriesdata_minute.resample(downsample_to).mean()

        # print(seriesname)
        # print(timeseriesdata_by_day.describe())

        return timeseriesdata_by_day


    def get_indoor_setpoint_timeseries_mean(self, parameter: str, seriesname: str,  
                                 upsample_to: str, downsample_to: str,
                                 start: datetime, end: datetime, tz_name_home: str) -> pd.DataFrame:
        tz_system = 'UTC'
        timeseriesdata = self.get(parameter)
        timeseriesdata.set_index('datetime', inplace=True)
        if tz_system == tz_name_home:
            timeseriesdata = timeseriesdata.tz_localize(tz_system)
        else:
            timeseriesdata = timeseriesdata.tz_localize(tz_system).tz_convert(tz_name_home)

        #first sort on datetime index
        timeseriesdata.sort_index(inplace=True)
        # tempting, but do NOT drop duplicates since this ignores index column
        # timeseriesdata.drop_duplicates(inplace=True)


        # Converting str to float
        timeseriesdata['value'] = timeseriesdata['value'].astype(float)

        #remove static  outliers
        timeseriesdata = Extractor.remove_measurement_outliers(timeseriesdata, 3.0)

        timeseriesdata_minute = timeseriesdata.resample(upsample_to).first()
        timeseriesdata_minute.interpolate(method='time', inplace=True)
        timeseriesdata_minute['target_value'] = timeseriesdata_minute['value']
        timeseriesdata_minute.rename(columns={'target_value':seriesname}, inplace=True)
        timeseriesdata_minute.drop(['index', 'timestamp', 'value'], axis=1, inplace=True)

        timeseriesdata_by_day = timeseriesdata_minute.resample(downsample_to).mean()

        # print(seriesname)
        # print(timeseriesdata_by_day.describe())

        return timeseriesdata_by_day

    @staticmethod
    def remove_measurement_outliers(df: pd.DataFrame, n_std) -> pd.DataFrame:
        """
        Simple procedure to replace outliers in the 'value' column with NaN
        Where outliers are those values more than n_std standard deviations away from the average of the 'value' column in a dataframe
        """

        df[(df['value']-df['value'].mean()).abs() > (n_std*df['value'].std())] = np.nan

        return df

    
class WeatherExtractor:
    """
    Use the weather extractor to get Dutch weather.
    """

    @staticmethod
    def get_linear_geospatial_interpolation_weather_nl(starttime:datetime, endtime:datetime, lat:float, lon:float, timezone:str) -> pd.DataFrame:
        """
        get weather data using a linear geospatial interpolation 
        based on three nearby KNMI weather stations 
        using the GitHub repo https://github.com/stephanpcpeters/HourlyHistoricWeather 
        for metrics=['T', 'FH', 'Q'] 
        rendered as a dataframe with a timezone-aware datetime index
        """
        
        # the .tz_localize(None).tz_localize(tz_home) at the and is needed to work around a bug in the historicdutchweather library 
        # TODO: post an issue in the historicdutchweather library and change the code to the line directly below when repaired.
        # weather = historicdutchweather.get_local_weather(starttime, endtime, lat, lon, metrics=['T', 'FH', 'Q'])
        df = historicdutchweather.get_local_weather(starttime, endtime, lat, lon, metrics=['T', 'FH', 'Q']).tz_localize(None).tz_localize(timezone)
        return df

    @staticmethod
    def get_weather_parameter_timeseries_mean(weather: pd.DataFrame, parameter: str, seriesname: str, 
                                 upsample_to: str, downsample_to: str,
                                 start: datetime, end: datetime, tz_name_home: str) -> pd.DataFrame:

        tz_system = 'UTC'
        timeseriesdata = pd.DataFrame(weather[parameter])
        # print(timeseriesdata)
        # timeseriesdata.set_index('datetime', inplace=True)

        if not(tz_system == tz_name_home):
            timeseriesdata = timeseriesdata.tz_convert(tz_name_home)

        #first sort on datetime index
        timeseriesdata.sort_index(inplace=True)
        # tempting, but do NOT drop duplicates since this ignores index column
        # timeseriesdata.drop_duplicates(inplace=True)

        # Converting str to float not needed
        # timeseriesdata[parameter] = timeseriesdata[parameter].astype(float)

        timeseriesdata_minute = timeseriesdata.resample(upsample_to).first()
        timeseriesdata_minute.interpolate(method='time', inplace=True)

        timeseriesdata_minute.rename(columns={parameter:seriesname}, inplace=True)
        timeseriesdata_by_day = timeseriesdata_minute.resample(downsample_to).mean()

        # print(seriesname)
        # print(timeseriesdata_by_day.describe())

        return timeseriesdata_by_day
    
    @staticmethod
    def remove_weather_outliers(weather: pd.DataFrame, columns, n_std) -> pd.DataFrame:
        """
        Simple procedure to replace outliers in the [col] columns with NaN
        Where outliers are those values more than n_std standard deviations away from the average of the [col] columns in a dataframe
        """
        for col in columns:
            mean = weather[col].mean()
            std = weather[col].std()
            weather[(weather[col]-mean).abs() > (n_std*std)] = np.nan
            
        return weather