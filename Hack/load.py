import abc
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

DATA_DIR = Path("./Data")


class loader(abc.ABC):
    """Abstract class to load the data.

    Each dataset needs different date parsing

    """

    def load(self):
        df = self.load_csv(self.local_path)
        return self.set_date_index(df)

    def load_csv(self, fname, delimiter=",", header=0):
        return pd.read_csv(fname, delimiter=delimiter, header=header)

    @property
    def local_dir(self):
        return DATA_DIR

    @abc.abstractproperty
    def fname(self):
        # will handle fname formatting
        pass

    @property
    def local_path(self):
        return self.local_dir / self.fname

    def get_df_timespan(self, df):
        df_start = df.index[0].strftime("%Y-%m-%d %H:%M")
        df_end = df.index[-1].strftime("%Y-%m-%d %H:%M")
        # print(f"{self.inst} spans from {df_start} to {df_end}\n")
        return df_start, df_end


class systemprice(loader):
    @property
    def fname(self):
        return "systemprice.csv"

    def load(self):
        df = self.load_csv(self.local_path)
        df = self.set_date_index(df)
        df = add_local_time(df)
        return df

    def parse_dates(self, df):
        utc_dates = []
        start_date_str = df["Settlement Date"][0]
        start_date = datetime.strptime(start_date_str, "%d/%m/%Y")

        # just add 30 minutes intervals, and deal with local
        # time zone later
        for i in range(df.shape[0]):
            utc_date = start_date + timedelta(minutes=30 * i)
            utc_dates.append(utc_date)
        return utc_dates

    def set_date_index(self, df):
        parsed_dates = self.parse_dates(df)
        df["Date"] = parsed_dates
        df = df.set_index("Date")
        return df


class epex(loader):
    @property
    def fname(self):
        return "epex_day_ahead_price.csv"

    def parse_dates(self, date_strs):
        dates = []
        for date_str in date_strs:
            # this is UTC time
            date = datetime.strptime(date_str[:-6], "%Y-%m-%d %H:%M:%S:")
            dates.append(date)

        return dates

    def set_date_index(self, df):
        df = df.set_index("timestamp")
        return df

    def load_csv(self, fname, delimiter=",", header=0):
        return pd.read_csv(
            fname, delimiter=delimiter, header=header, parse_dates=["timestamp"]
        )


class spot(loader):
    @property
    def fname(self):
        return "spot_intraday_price.csv"

    def parse_dates(self, date_strs):
        dates = []
        for date_str in date_strs:
            # this is UTC time
            date = datetime.strptime(date_str[:-6], "%Y-%m-%d %H:%M:%S:")
            dates.append(date)

        return dates

    def set_date_index(self, df):
        df = df.set_index("timestamp")
        return df

    def load_csv(self, fname, delimiter=",", header=0):
        return pd.read_csv(
            fname, delimiter=delimiter, header=header, parse_dates=["timestamp"]
        )


def add_local_time(df):
    local_tz = pytz.timezone("Europe/London")
    local_timestamps = []
    local_dates = []
    for date_index in df.index:
        date = datetime(
            date_index.year,
            date_index.month,
            date_index.day,
            date_index.hour,
            date_index.minute,
        )
        date.astimezone(pytz.utc)
        date_as_tz = date.astimezone(local_tz)
        local_dates.append(date_as_tz)
        local_timestamps.append(
            datetime.strptime(
                (date_as_tz + date_as_tz.dst()).strftime("%Y-%m-%d %H:%M"),
                "%Y-%m-%d %H:%M",
            )
        )
    df["local_datetime"] = local_dates
    df["local_time"] = local_timestamps
    return df
