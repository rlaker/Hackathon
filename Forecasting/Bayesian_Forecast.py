from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
from pybats.analysis import analysis
from pybats.plot import *
from pybats.point_forecast import median


####################################
# Plotting Functions
####################################
def plot_forecast(
    fig,
    ax,
    y,
    f,
    samples,
    dates,
    linewidth=1,
    linecolor="b",
    credible_interval=95,
    **kwargs,
):
    """
    Plot observations along with sequential forecasts and credible intervals.
    """

    ax.scatter(dates, y, color="k")
    ax.plot(dates, f, color=linecolor, linewidth=linewidth)
    alpha = (100 - credible_interval) / 2
    upper = np.percentile(samples, [100 - alpha], axis=0).reshape(-1)
    lower = np.percentile(samples, [alpha], axis=0).reshape(-1)
    ax.fill_between(dates, upper, lower, alpha=0.3, color=linecolor)

    if kwargs.get("xlim") is None:
        kwargs.update({"xlim": [dates[0], dates[-1]]})

    if kwargs.get("legend") is None:
        legend = ["Observations", "Forecast", "Credible Interval"]

    ax = ax_style(ax, legend=legend, **kwargs)

    # If dates are actually dates, then format the dates on the x-axis
    if isinstance(dates[0], (datetime, date)):
        fig.autofmt_xdate()

    return ax


def forecast_ax_style(
    ax,
    ylim=None,
    xlim=None,
    xlabel=None,
    ylabel=None,
    title=None,
    legend=None,
    legend_inside_plot=True,
    topborder=False,
    rightborder=False,
    **kwargs,
):
    """
    A helper function to define many elements of axis style at once.
    """

    if legend is not None:
        if legend_inside_plot:
            ax.legend(legend)
        else:
            ax.legend(
                legend,
                bbox_to_anchor=(1.05, 1),
                loc=2,
                borderaxespad=0.5,
                frameon=False,
            )
            # Make room for the legend
            plt.subplots_adjust(right=0.85)

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # remove the top and right borders
    ax.spines["top"].set_visible(topborder)
    ax.spines["right"].set_visible(rightborder)

    plt.tight_layout()

    return ax


####################################
# Forecasting Functions
####################################


def evaluate(epex_data, horizon=6, forecast_start_index=0, forecast_end_index=-1):
    prices = epex_data.values[:, 0]
    datetimes = epex_data.index
    horizon

    forecast_start_date = datetimes[forecast_start_index]
    forecast_end_date = datetimes[forecast_end_index]

    mod, samples = analysis(
        prices,
        family="poisson",
        dates=datetimes,
        forecast_start=forecast_start_date,  # First time step to forecast on
        forecast_end=forecast_end_date,  # Final time step to forecast on
        ntrend=1,  # Intercept and slope in model
        nsamps=500,  # Number of samples taken in the Poisson process
        seasPeriods=[
            48
        ],  # Length of the seasonal variations in the data - i.e. every 24hr here
        seasHarmComponents=[
            [1, 2, 3, 4, 6]
        ],  # Variations to pick out from the seaonal period
        k=horizon,  # Forecast horizon. If k>1, default is to forecast 1:k steps ahead, marginally
        prior_length=48,  # How many data point to use in defining prior - 48=1 day
        rho=0.3,  # Random effect extension, which increases the forecast variance (see Berry and West, 2019)
        deltrend=0.98,  # Discount factor on the trend component (the intercept)
        delregn=0.98,  # Discount factor on the regression component
        delSeas=0.98,
    )

    forecast = median(samples)

    return datetimes, prices, samples, forecast
