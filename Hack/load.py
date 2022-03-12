import abc
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path("./Data")


class loader(abc.ABC):
    """Abstract class to load the data.

    Each dataset needs different date parsing

    """

    def load(self):
        print(self.local_path)
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

    def parse_dates(self, date_strs, periods):
        dates = []
        for date_str, period in zip(date_strs, periods):
            date = datetime.strptime(date_str, "%d/%m/%Y")
            date += timedelta(minutes=30 * period)
            dates.append(date)

        return dates

    def set_date_index(self, df):
        parsed_dates = self.parse_dates(df["Settlement Date"], df["Settlement Period"])
        df["Date"] = parsed_dates
        df = df.set_index("Date")
        df.drop(columns=["Settlement Date"], inplace=True)
        return df


class epex(loader):
    @property
    def fname(self):
        return "epex_day_ahead_price.csv"

    def parse_dates(self, date_strs, periods):
        dates = []
        for date_str, period in zip(date_strs, periods):
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

    def parse_dates(self, date_strs, periods):
        dates = []
        for date_str, period in zip(date_strs, periods):
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
