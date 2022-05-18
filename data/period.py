from __future__ import annotations
import datetime
from typing import Optional


class Period:
    """
    A period of time to filter the extractor.
    """

    start: Optional[datetime.datetime] = None
    end: Optional[datetime.datetime] = None

    def __init__(self, start: Optional[datetime.datetime] = None, end: Optional[datetime.datetime] = None):
        self.start = start
        self.end = end

    def delta(self) -> datetime.timedelta:
        """
        Get the difference in time from start to end if set, otherwise delta is 0.
        """

        if self.start is None or self.end is None:
            return datetime.timedelta()

        return self.end - self.start

    def extend(self, timedelta: datetime.timedelta) -> Period:
        """
        Extend the period in both directions with the given timedelta.
        Note that this method creates a new instance of period and does not alter the current period.
        """

        start = None
        if self.start is not None:
            start = self.start - timedelta

        end = None
        if self.end is not None:
            end = self.end + timedelta

        return Period(start, end)

    def shrink(self, timedelta: datetime.timedelta) -> Period:
        """
        Shrink the period in both directions with the given timedelta.
        Note that this method creates a new instance of period and does not alter the current period.
        """

        start = None
        if self.start is not None:
            start = self.start + timedelta

        end = None
        if self.end is not None:
            end = self.end - timedelta

        return Period(start, end)

    def __repr__(self) -> str:
        return "%s - %s" % (self.start, self.end)

    def __lt__(self, other) -> bool:
        if self.start == other.start:
            if self.end is None:
                return False
            if other.end is None:
                return True

            return self.end < other.end

        if other.start is None:
            return False
        if self.start is None:
            return True

        return self.start < other.start

    def __le__(self, other) -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __eq__(self, other) -> bool:
        return self.start == other.start and self.end == other.end

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __ge__(self, other) -> bool:
        return self.__gt__(other) or self.__eq__(other)

    def __gt__(self, other) -> bool:
        if self.start == other.start:
            if other.end is None:
                return False
            if self.end is None:
                return True

            return self.end > other.end

        if self.start is None:
            return False
        if other.start is None:
            return True

        return self.start > other.start
