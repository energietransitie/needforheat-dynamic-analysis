from datetime import datetime, timedelta
import pytz
from typing import Optional, List, Literal, Dict, Iterable, Union, Callable

import pandas as pd
import numpy as np

from period import Period
from database import Database

from tqdm import tqdm_notebook

from enum import Enum, auto
import logging


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

class Summarizer(Enum):
    add = auto()
    mean = auto()
    count = auto()
    first = auto()
    last = auto()

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

    def set_start(self, date: Optional[datetime] = None):
        """
        Set the start datetime for the current extraction.
        """

        self.__period.start = date

    def set_end(self, date: Optional[datetime] = None):
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

    def get_custom_periods(self, data: Union[str, pd.DataFrame], delegate: Union[Callable[[float, float], bool], Callable[[float, float, datetime, datetime], bool]]) -> List[Period]:
        """
        Filter data with a custom function and obtain the periods that match the conditions.
        Use a delegate with before and after values as float and optionally add the before and after methods.
        Example:
        def delegate(before: float, after: float, before_datetime: datetime, after_datetime: datetime) -> bool:
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

        return self.close_period_gaps(result, timedelta())

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

        return self.close_period_gaps(result, timedelta())

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

    def __is_valid_date(self, date: datetime) -> bool:
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
    def close_period_gaps(periods: List[Period], timedelta: timedelta) -> List[Period]:
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

    def get_rawdata(self) -> pd.DataFrame:
        """
        Get a DataFrame with raw data from homes
        """

        self._connect()
        cursor = self._db.cursor()
        
        var = {
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


        
        sql = """
        SELECT DISTINCT
            a.pseudonym AS 'peudonym',
            DATE_FORMAT(m.timestamp, '%Y-%m-%dT%TZ') AS 'timestamp_UTC',
            UNIX_TIMESTAMP(m.timestamp) AS 'unix_timestamp',
            pr.name as 'property_name',
            m.value AS 'value',
            pr.unit AS 'unit',
            m.id AS 'measurement_id'
        FROM
            measurement m
        JOIN property pr ON
            pr.id = m.property_id
        JOIN device d ON
            d.id = m.device_id
        JOIN building b ON
            b.id = d.building_id
        JOIN account a on
            a.id = b.id
        WHERE
            a.pseudonym = '""" + str(self.__house) + """'
        AND m.timestamp BETWEEN '2021-10-24 23:00' AND '2022-05-08 01:00'
        ORDER BY
            a.pseudonym,
            m.timestamp

        """
        
        cursor.execute(sql)
        rows = cursor.fetchall()
        self._close()

        return pd.DataFrame(rows, columns=['pseudonym','timestamp_UTC','unix_timestamp','property_name','value','unit','measurement_id'])

    def get_property_preprocessed(self, parameter:str, seriesname:str, metertimestamp:str, 
                                  tz_source:str, tz_home:str,
                                  process_meter_reading:bool, 
                                  min_interval_value:float, max_interval_value:float, n_std:int, 
                                  up_intv:str, gap_n_intv:int, 
                                  summ_intv:str, summ) -> pd.DataFrame:

        # Type checking
        if not isinstance(summ, Summarizer):
            raise TypeError('summ parameter must be an instance of Summarizer Enum')

        df = self.get(parameter)
        
        logging.info(seriesname, 'df before localization:', df.head(25))

        df.set_index('datetime', inplace=True)
        df.drop(['index', 'timestamp'], axis=1, inplace=True)

        if tz_source == tz_home:
            df = df.tz_localize(tz_source)
        else:
            df = df.tz_localize(tz_source).tz_convert(tz_home)
            
        logging.info(seriesname, 'df after localization, before concatenation:', df.head(25))
        if (df is not None and len(df.index)>0):
            logging.info('df.index[0]: ', df.index[0])
            logging.info('df.index[0].tzinfo: ', df.index[0].tzinfo)

        if metertimestamp is not None:
            df_metertimestamp = self.get(metertimestamp)
            df_metertimestamp.set_index('datetime', inplace=True)
            df_metertimestamp.drop(['index', 'timestamp'], axis=1, inplace=True)
            if tz_source == tz_home:
                df_metertimestamp = df_metertimestamp.tz_localize(tz_source)
            else:
                df_metertimestamp = df_metertimestamp.tz_localize(tz_source).tz_convert(tz_home)

            logging.info(seriesname, 'df_metertimestamp before parsing YYMMDDhhmmssX values:', df_metertimestamp.head(25))
            logging.info(seriesname, 'df_metertimestamp.index[0]: ', df_metertimestamp.index[0])
            logging.info(seriesname, 'df_metertimestamp.index[0].tzinfo: ', df_metertimestamp.index[0].tzinfo)

            # parse DSMR TST value format: YYMMDDhhmmssX
            # meaning according to DSMR 5.0.2 standard: 
            # "ASCII presentation of Time stamp with Year, Month, Day, Hour, Minute, Second, 
            # and an indication whether DST is active (X=S) or DST is not active (X=W)."
            if df_metertimestamp['value'].str.contains('W|S', regex=True).any():
                logging.info(seriesname, 'parsing DSMR>v2 $S $W timestamps')
                df_metertimestamp['value'].replace(to_replace='W$', value='+0100', regex=True, inplace=True)
                logging.info(seriesname, 'df_metertimestamp after replace W:', df_metertimestamp.head(25))
                df_metertimestamp['value'].replace(to_replace='S$', value='+0200', regex=True, inplace=True)
                logging.info(seriesname, 'df_metertimestamp after replace S, before parsing:', df_metertimestamp.head(25))
                df_metertimestamp['meterdatetime'] = df_metertimestamp['value'].str.strip()
                logging.info(seriesname, 'df_metertimestamp after stripping, before parsing:', df_metertimestamp.head(25))
                df_metertimestamp['meterdatetime'] = pd.to_datetime(df_metertimestamp['meterdatetime'], format='%y%m%d%H%M%S%z', errors='coerce')
                logging.info(seriesname, df_metertimestamp[df_metertimestamp.meterdatetime.isnull()])
                df_metertimestamp['meterdatetime'] = df_metertimestamp['meterdatetime'].tz_convert(tz_home)
            else:
                logging.info(seriesname, 'parsing DSMR=v2 timestamps without $W $S indication')
                if df_metertimestamp['value'].str.contains('[0-9]', regex=True).any():
                    # for smart meters of type Kamstrup 162JxC - KA6U (DSMR2), there is no W or S at the end; timeoffset needs to be inferred
                    logging.info(seriesname, 'df_metertimestamp before parsing:', df_metertimestamp.head(25))
                    df_metertimestamp['meterdatetime'] = df_metertimestamp['value'].str.strip() 
                    logging.info(seriesname, 'df_metertimestamp after stripping, before parsing:', df_metertimestamp.head(25))
                    df_metertimestamp['meterdatetime'] = pd.to_datetime(df_metertimestamp['meterdatetime'], format='%y%m%d%H%M%S', errors='coerce')
                    logging.info(seriesname, df_metertimestamp[df_metertimestamp.meterdatetime.isnull()])
                    df_metertimestamp['meterdatetime'] = df_metertimestamp['meterdatetime'].tz_localize(None).tz_localize(tz_home, ambiguous='infer')
                else: # DSMRv2 did not speficy eMeterReadingTimestamps
                    df_metertimestamp['meterdatetime'] = df_metertimestamp.index 
                    
                logging.info(seriesname, 'df_metertimestamp after all parsing:', df_metertimestamp.head(25))



            # dataframe contains NaT values  
            logging.info('before NaT replacement:', df_metertimestamp[df_metertimestamp.meterdatetime.isnull()])

            # unfortunately, support for linear interpolation of datetimes is not properly implemented, so we do it via Unix time
            df_metertimestamp['meterdatetime']= df_metertimestamp['meterdatetime']\
                                                .apply(lambda x: np.nan if pd.isnull(x) else x.timestamp())\
                                                .interpolate(method='linear', limit=2)
            df_metertimestamp['meterdatetime'] = pd.to_datetime(df_metertimestamp['meterdatetime'], unit='s', utc=True,  origin='unix')\
                                                 .dt.tz_convert(tz_home)


            df_metertimestamp.reset_index(inplace=True)
            df_metertimestamp.set_index('datetime', inplace=True)
            df_metertimestamp.drop(['value'], axis=1, inplace=True)
            logging.info('df_metertimestamp after NaT replacement:', df_metertimestamp.head(25))
            logging.info('df_metertimestamp.index[0]: ', df_metertimestamp.index[0])
            logging.info('df_metertimestamp.index[0].tzinfo: ', df_metertimestamp.index[0].tzinfo)

            df = pd.concat([df, df_metertimestamp], axis=1, join='outer')
            logging.info('df:', df.head(25))
            logging.info('df.index[0]: ', df.index[0])
            logging.info('df.index[0].tzinfo: ', df.index[0].tzinfo)

            df.reset_index(inplace=True)
            df.drop(['datetime'], axis=1, inplace=True)
            df.rename(columns = {'meterdatetime':'datetime'}, inplace = True)
            logging.info('df after rename:', df.head(25))
            df.drop_duplicates(inplace=True)
            logging.info('df after dropping duplicates:', df.head(25))
            df.set_index('datetime', inplace=True)
            logging.info('df:', df.head(25))
            
        #first sort on datetime index
        df.sort_index(inplace=True)
        # tempting, but do NOT drop duplicates since this ignores index column, 
        # for meter readings, we already dropped duplicates earlier when both smart meter timestamp and smart meter reading were identical 

        # remove index, timestamp and value columns and rename column to seriesname
        logging.info('before rename:', df.head(25))
        df.rename(columns = {'value':seriesname}, inplace = True)

        # Converting str to float
        df[seriesname] = df[seriesname].astype(float)
        logging.info('after rename:', df.head(25))


        if process_meter_reading:
            
            #first, correct for occasional zero meter readings; this involves taking a diff and removing the meter reading that causes the negative jump
            logging.info('before zero meter reading filter:', df.head(25))
            df['diff'] = df[seriesname].diff()
            logging.info('after diff:', df.head(25))
            df = df[df['diff'] >= 0]
            df.drop(['diff'], axis=1, inplace=True)
            logging.info('after zero meter reading filter:', df.head(25))

            #then, correct for meter changes; this involves taking a diff and removing the negative jum and cumulating again
            #first, correct for potential meter changes
            df[seriesname] = df[seriesname].diff().shift(-1)
            logging.info('after diff:', df.head(25))
            df.loc[df[seriesname] < 0, seriesname] = 0
            logging.info('after filter negatives:', df.head(25))

            # now, cumulate again
            df[seriesname] = df[seriesname].shift(1).fillna(0).cumsum()
            logging.info(seriesname, 'after making series cumulative again before resampling and interpolation:', df.head(25))
            
            # then interpolate the cumulative series
            logging.info(df[df.index.isnull()])
            df = df.resample(up_intv).first()
            logging.info('after resample:', df.head(25))
            df.interpolate(method='time', inplace=True, limit=gap_n_intv)
            logging.info('after interpolation:', df.head(25))
          
            # finally, differentiate a last time to get rate of use
            df[seriesname] = df[seriesname].diff().shift(-1)
            logging.info('after taking differences:', df.head(25))
            

        #if min_interval_value given, then remove larger values
        if min_interval_value is not None:
            df.loc[df[seriesname] < min_interval_value, seriesname] = np.nan

        #if max_interval_value given, then remove larger values
        if max_interval_value is not None:
            df.loc[df[seriesname] > max_interval_value, seriesname] = np.nan

        #if n_std is given, them outliers more than n_std standard deviations away from mean
        if n_std is not None:
            df = Extractor.remove_measurement_outliers(df, seriesname, n_std)

        # then first upsample to regular intervals; this creates various colums with np.NaN as value
        df = df.resample(up_intv).first()

        # procedures above may have removed values; fill these, but don't bridge gaps larger than gap_n_intv times the interval
        if (summ == Summarizer.first):
            df.interpolate(method='pad', inplace=True)
        else: 
            df.interpolate(method='time', inplace=True, limit=gap_n_intv)
            

        # interplolate and summarize data using  resampling 
        if (summ == Summarizer.add):
            df = df[seriesname].resample(summ_intv).sum()
        elif (summ == Summarizer.mean):
            df = df[seriesname].resample(summ_intv).mean()
        elif (summ == Summarizer.count):
            df = df[seriesname].resample(summ_intv).count()
        elif (summ == Summarizer.first):
            # not totally sure, mean seems to be a proper summary of e.g. a thermostat setpoint
            df = df[seriesname].resample(summ_intv).mean()
                
        return df

    @staticmethod
    def remove_measurement_outliers(df: pd.DataFrame, col, n_std) -> pd.DataFrame:
        """
        Simple procedure to replace outliers in the 'value' column with NaN
        Where outliers are those values more than n_std standard deviations away from the average of the 'value' column in a dataframe
        """

        mean = df[col].mean()
        std = df[col].std()
        # df[(df[col]-mean).abs() > (n_std*std)] = np.nan
        df.loc[(df[col]-mean).abs() > (n_std*std), col] = np.nan

        return df
    

    @staticmethod
    def get_preprocessed_homes_data(homes, first_day:datetime, last_day:datetime, 
                                    tz_source:str, tz_home:str, 
                                    up_intv:str, gap_n_intv:int, summ_intv:str, 
                                    weather_interpolated:pd.DataFrame) -> pd.DataFrame:
        """
        Obtain data from twomes database 
        convert timestamps to the tz_home timezone 
        with outlier removal specific for each property
        interpolated with the int_intv
        rendered as a dataframe with a timezone-aware datetime index
        [
            'id', 'timestamp',
            'temp_out__degC','wind__m_s_1', 'ghi__W_m_2',   
            'temp_in__degC', 'temp_set__degC',
            'gas_use__W', 'gas_use_noCH__W', 'gas_use_CH__W', 
            'e_use__W',, 'e_ret__W'
        ]
        """

        # Conversion factor s_p_h [s/h]  = 60 [min/h] * 60 [s/min] 
        s_p_h = (60 * 60) 

        # Conversion factor J_p_kWh [J/kWh]  = 1000 [Wh/kWh] * s_p_h [s/h] * 1 [J/Ws]
        J_p_kWh = 1000 * s_p_h


        df_all_homes = pd.DataFrame()

        # if so, convert starttime & endtime to database timezone; extend on both sides with interval of one 
        largest_measurement_interval = timedelta(hours=1)
        extractor_starttime = (first_day - largest_measurement_interval).astimezone(pytz.timezone(tz_source))
        logging.info('extractor_starttime: ', extractor_starttime)
        extractor_endtime = (last_day + timedelta(days=1) + largest_measurement_interval).astimezone(pytz.timezone(tz_source))
        logging.info('extractor_endtime: ', extractor_endtime)

        for id in tqdm_notebook(homes):

            logging.info(f'Retrieving data for home {id} from {extractor_starttime.isoformat()} to {extractor_endtime.isoformat()} ...')
            
            
            # temporary fix: request more data
            extractor = Extractor(id, Period(extractor_starttime, extractor_endtime))

            df_indoortemp = extractor.get_property_preprocessed('roomTemp', 'temp_in__degC', 
                                                                metertimestamp=None, tz_source=tz_source, tz_home=tz_home,
                                                                process_meter_reading=False,
                                                                min_interval_value=0.0, max_interval_value=40.0, n_std=3,
                                                                up_intv=up_intv, gap_n_intv=gap_n_intv, 
                                                                summ_intv=summ_intv, summ=Summarizer.mean)
                
                
            if len(df_indoortemp.index)>=1:
                try:
                    # start with weather data
                    df = pd.DataFrame()
                    df = weather_interpolated.copy()
                    # label each line with homepseudonym
                    df.insert(loc=0, column='id', value=id)
                    # heartbeats_interpolated = extractor.get_home_parameter_timeseries_count('heartbeat', 'heartbeat', 
                    #                                                                         metertimestamp=None, tz_source=tz_source, tz_home=tz_home,
                    #                                                                         process_meter_reading=False,
                    #                                                                         min_interval_value=0.0, max_interval_value=None, n_std=None, 
                    #                                                                         up_intv=up_intv, gap_n_intv=gap_n_intv, 
                    #                                                                         summ_intv=summ_intv, summ=Summarizer.mean)
                    df = pd.concat([df, df_indoortemp], axis=1, join='outer')
                    # df = pd.concat([df, extractor.get_property_preprocessed('roomTempCO2', 'indoor_CO2T_C', 
                    #                                                         metertimestamp=None, tz_source=tz_source, tz_home=tz_home,
                    #                                                         process_meter_reading=False, 
                    #                                                         min_interval_value=0.0, max_interval_value=45.0, n_std,
                    #                                                         up_intv=up_intv, gap_n_intv=gap_n_intv,
                    #                                                         summ_intv=summ_intv, summ=Summarizer.mean)
                    #                ], axis=1, join='outer')
                    df = pd.concat([df, extractor.get_property_preprocessed('roomSetpointTemp', 'temp_set__degC', 
                                                                            metertimestamp=None, tz_source=tz_source, tz_home=tz_home,
                                                                            process_meter_reading=False, 
                                                                            min_interval_value=0.0, max_interval_value=45.0, n_std=None,
                                                                            up_intv=up_intv, gap_n_intv=gap_n_intv,
                                                                            summ_intv=summ_intv, summ=Summarizer.first)
                                   ],axis=1, join='outer')
                    df = pd.concat([df, extractor.get_property_preprocessed('gMeterReadingSupply', 'g_use_m3', 
                                                                            metertimestamp='gMeterReadingTimestamp', tz_source=tz_source, tz_home=tz_home,
                                                                            process_meter_reading=True, 
                                                                            min_interval_value=0.0, max_interval_value=None, n_std=None,
                                                                            up_intv=up_intv, gap_n_intv=gap_n_intv,
                                                                            summ_intv=summ_intv, summ=Summarizer.add)
                                   ], axis=1, join='outer')
                    df = pd.concat([df, extractor.get_property_preprocessed('eMeterReadingSupplyHigh', 'e_use_hi__kWh', 
                                                                            metertimestamp='eMeterReadingTimestamp', tz_source=tz_source, tz_home=tz_home,
                                                                            process_meter_reading=True, 
                                                                            min_interval_value=0.0, max_interval_value=None, n_std=None,
                                                                            up_intv=up_intv, gap_n_intv=gap_n_intv,
                                                                            summ_intv=summ_intv, summ=Summarizer.add)
                                   ], axis=1, join='outer')
                    df = pd.concat([df, extractor.get_property_preprocessed('eMeterReadingSupplyLow', 'e_use_lo__kWh', 
                                                                            metertimestamp='eMeterReadingTimestamp', tz_source=tz_source, tz_home=tz_home,
                                                                            process_meter_reading=True, 
                                                                            min_interval_value=0.0, max_interval_value=None, n_std=None,
                                                                            up_intv=up_intv, gap_n_intv=gap_n_intv,
                                                                            summ_intv=summ_intv, summ=Summarizer.add)

                                   ], axis=1, join='outer')
                    df = pd.concat([df, extractor.get_property_preprocessed('eMeterReadingReturnHigh', 'e_ret_hi__kWh', 
                                                                            metertimestamp='eMeterReadingTimestamp', tz_source=tz_source, tz_home=tz_home,
                                                                            process_meter_reading=True, 
                                                                            min_interval_value=0.0, max_interval_value=None, n_std=None,
                                                                            up_intv=up_intv, gap_n_intv=gap_n_intv,
                                                                            summ_intv=summ_intv, summ=Summarizer.add)

                                   ], axis=1, join='outer')
                    df = pd.concat([df, extractor.get_property_preprocessed('eMeterReadingReturnLow', 'e_ret_lo__kWh', 
                                                                            metertimestamp='eMeterReadingTimestamp', tz_source=tz_source, tz_home=tz_home,
                                                                            process_meter_reading=True, 
                                                                            min_interval_value=0.0, max_interval_value=None, n_std=None,
                                                                            up_intv=up_intv, gap_n_intv=gap_n_intv,
                                                                            summ_intv=summ_intv, summ=Summarizer.add)
                                   ], axis=1, join='outer')

                    # calculate timedelta for each interval (code is suitable for unevenly spaced measurementes)
                    df['interval__s'] = df.index.to_series().diff().shift(-1).apply(lambda x: x.total_seconds()).fillna(0).astype(int)

                    #remove intervals earlier than first_day or later than last_day
                    df = df[(df.index >= first_day) & (df.index < (last_day + timedelta(days=1)))]

                    # Superior calirific value superior calorific value of natural gas from the Groningen field = 35.17 [MJ/m^3]
                    h_sup__J_m_3 = 35.17e6

                    # converting gas values from m^3 per interval to averages 
                    df['gas_use__W'] = df['g_use_m3'] * h_sup__J_m_3 / df['interval__s']
                    df = df.drop('g_use_m3', axis=1)

                    # calculating derived columns
                    df['e_use__W'] = (df['e_use_hi__kWh'] + df['e_use_lo__kWh']) * J_p_kWh / df['interval__s']
                    df['e_ret__W'] = (df['e_ret_hi__kWh'] + df['e_ret_lo__kWh']) * J_p_kWh / df['interval__s']

                    df = df.drop('e_use_hi__kWh', axis=1)
                    df = df.drop('e_use_lo__kWh', axis=1)
                    df = df.drop('e_ret_hi__kWh', axis=1)
                    df = df.drop('e_ret_lo__kWh', axis=1)
                    
                        
                    # finally: add to results from other homes
                    df_all_homes = pd.concat([df_all_homes, df], axis=0)
                except Exception as e:
                    logging.exception(e)
                    return df


                # if no indoortemp then don't add data for this home

        # after all homes are done
        
        df_all_homes = df_all_homes.reset_index().rename(columns = {'index':'timestamp'}).set_index(['id', 'timestamp'])
                                          
        return df_all_homes.loc[df_all_homes.index.dropna()]

    @staticmethod
    def get_virtual_home_data_csv(filename: str, tz_home:str) -> pd.DataFrame:
        """
        Obtain data from an csv file with virtual home data 
        convert timestamps to the tz_home timezone 
        with outlier removal specific for each property
        interpolated with the int_intv
        rendered as a dataframe with a timezone-aware datetime index
        [
            'id', 'timestamp',
            'temp_out__degC','wind__m_s_1', 'ghi__W_m_2',  
            'temp_in__degC', 'temp_set__degC',
            'gas_use__W', 'gas_use_noCH__W', 'gas_use_CH__W', 
            'e_use__W', 'e_ret__W'
        ]
        """
        df_data_virtual_home = pd.read_csv(filename, delimiter=",", skipinitialspace=True, decimal=".", parse_dates=["timestamp_ISO8601"])
        df_data_virtual_home.rename(columns={"timestamp_ISO8601": "timestamp"}, inplace=True)
        df_data_virtual_home = df_data_virtual_home.set_index("timestamp")
        df_data_virtual_home = df_data_virtual_home.loc[df_data_virtual_home.index.dropna()]
        df_data_virtual_home = df_data_virtual_home.drop('Unnamed: 0', axis=1)
        df_data_virtual_home.index = pd.to_datetime(df_data_virtual_home.index)
        df_data_virtual_home.reset_index(inplace=True)
        cols = list(df_data_virtual_home.columns)
        df_data_virtual_home = df_data_virtual_home[[cols[1]] + [cols[0]] + cols [2::]]
        df_data_virtual_home = df_data_virtual_home.set_index(['id', 'timestamp'])
        df_data_virtual_home = df_data_virtual_home.loc[df_data_virtual_home.index.dropna()]
        return df_data_virtual_home
    
    
class WeatherExtractor:
    """
    Use the weather extractor to get Dutch weather.
    """

    @staticmethod
    def get_interpolated_weather_nl(first_day:datetime, last_day:datetime, lat:float, lon:float, tz_source:str, tz_home:str, int_intv:str) -> pd.DataFrame:
        """
        get weather data using a linear geospatial interpolation 
        based on three nearby KNMI weather stations 
        using the GitHub repo https://github.com/stephanpcpeters/HourlyHistoricWeather 
        for KNMI metrics=['T', 'FH', 'Q'] 
        with NO outlier removal (assuming that KNMI already did this)
        interpolated with the int_intv
        rendered as a dataframe with a timezone-aware datetime index
        columns ['temp_out__degC', 'wind__m_s_1', 'ghi__W_m_2', 'T_out_e_C']
        """
        
        up = '15min'
        
        largest_measurement_interval = timedelta(hours=1)
        extractor_starttime = first_day
        logging.info('weather_extractor_starttime: ', extractor_starttime)
        extractor_endtime = last_day
        logging.info('weather_extractor_endtime: ', extractor_endtime)

        
        df = historicdutchweather.get_local_weather(extractor_starttime, extractor_endtime, lat, lon, metrics=['T', 'FH', 'Q'])
       

        logging.info('Resampling weather data...' )

        outdoor_T_interpolated = WeatherExtractor.get_weather_parameter_timeseries_mean(df, 'T', 'temp_out__degC', 
                                                                                           up, int_intv, 
                                                                                           tz_source, tz_home)
        windspeed_interpolated = WeatherExtractor.get_weather_parameter_timeseries_mean(df, 'FH', 'wind__m_s_1', 
                                                                                        up, int_intv, 
                                                                                        tz_source, tz_home)
        irradiation_interpolated = WeatherExtractor.get_weather_parameter_timeseries_mean(df, 'Q', 'ghi__J_h_1_cm_2', 
                                                                                          up, int_intv, 
                                                                                          tz_source, tz_home)


        # merge weather data in a single dataframe
        df = pd.concat([outdoor_T_interpolated, windspeed_interpolated, irradiation_interpolated], axis=1, join='outer') 
        
        df['ghi__W_m_2'] = df['ghi__J_h_1_cm_2']  * (100 * 100) / (60 * 60)
        df = df.drop('ghi__J_h_1_cm_2', axis=1)

        #oddly enough the column contains values that are minutely negative, which look weird and are impossible; hence: replace
        df.loc[(df.ghi__W_m_2 < 0), 'ghi__W_m_2'] = 0
       
        return df

    @staticmethod
    def get_weather_parameter_timeseries_mean(df: pd.DataFrame, parameter: str, seriesname: str, 
                                 up_to: str, int_intv: str,
                                 tz_source:str, tz_home: str) -> pd.DataFrame:

        df = pd.DataFrame(df[parameter])
        logging.info(df)
        # df.set_index('datetime', inplace=True)

        if not(tz_source == tz_home):
            df = df.tz_convert(tz_home)

        #first sort on datetime index
        df.sort_index(inplace=True)
        # tempting, but do NOT drop duplicates since this ignores index column
        # df.drop_duplicates(inplace=True)

        # Converting str to float needed
        df[parameter] = df[parameter].astype(float)

        df = df.resample(up_to).first()
        df.interpolate(method='time', inplace=True)

        df.rename(columns={parameter:seriesname}, inplace=True)
        df = df.resample(int_intv).mean()
        
        return df
