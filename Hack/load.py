from datetime import datetime, timedelta

import pandas as pd


def load_csv(fname):
    """Loads the csv files

    Parameters
    ----------
    fname : Path
        filename to the data
    """

    df = pd.read_csv(fname, delimiter=",", header=0)
    return set_date_index(df)


def parse_dates(date_strs, periods):
    dates = []
    for date_str, period in zip(date_strs, periods):
        date = datetime.strptime(date_str, "%d/%m/%Y")
        date += timedelta(minutes=30 * period)
        dates.append(date)

    return dates


def set_date_index(df, date_str_key="Settlement Date", periods_key="Settlement Period"):
    parsed_dates = parse_dates(df[date_str_key], df[periods_key])
    df["Date"] = parsed_dates
    df = df.set_index("Date")
    df.drop(columns=[date_str_key], inplace=True)
    return df
