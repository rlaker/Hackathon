from datetime import date, datetime
#from syslog import LOG_LOCAL1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pybats.analysis import analysis
from pybats.plot import *
from pybats.point_forecast import median


####################################
# Plotting Functions
####################################
def plot_forecast(
    fig,
    ax,
    epex,
    datetimes,
    plot_start,
    plot_length,
    horizon,
    f,
    samples,
    linewidth=1,
    linecolor="b",
    credible_interval=95,
    bool_plot_all_forecasts=False,
    **kwargs,
):
    """
    Plot observations along with sequential forecasts and credible intervals.
    """

    if(bool_plot_all_forecasts==True):
        future = np.arange(horizon)+1
    else:
        future = [horizon]

    for i in future:

        opacity = (i)/horizon # alpha of the forecast line to plot

        plot_start_date = datetimes[0] + pd.DateOffset(hours=(i + plot_start)/2.)
        plot_end_date   = plot_start_date + pd.DateOffset(hours=(plot_length - 1)/2.)

        dates=epex.loc[plot_start_date:plot_end_date].index,
        dates=dates[0]
        fore=f[plot_start:plot_start+plot_length,i-1]
        ax.plot(dates, fore, color=linecolor, linewidth=linewidth, alpha = opacity)

    y=epex.loc[plot_start_date:plot_end_date].values[:,0]
    samps=samples[:,plot_start:plot_start+plot_length,horizon-1]
    ax.scatter(dates, y, color="k")
    alpha = (100 - credible_interval) / 2
    upper = np.percentile(samps, [100 - alpha], axis=0).reshape(-1)
    lower = np.percentile(samps, [alpha], axis=0).reshape(-1)
    ax.fill_between(dates, upper, lower, alpha=0.3, color='g')
    
    if kwargs.get("xlim") is None:
        kwargs.update({"xlim": [dates[0], dates[-1]]})

    if kwargs.get("legend") is None:
        legend = ["Forecast", "Observations", "Credible Interval"]

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

def evaluate(epex_data, horizon=6, forecast_start_index=0, forecast_end_index=-1,prior_length=48,nsamps=500,rho=0.3):
    """Predicts future prices based upon the price array data.

    Parameters
    ----------
    price_array : dataframe
        The EPEX price data provided by load.epex().load()
    horizon : int, optional
        How many datapoints in the future will be foreacsted for every EPEX datapoint.
    forecast_start_index : int, optional
        The initial epex_data point to begin forecasting from.
    forecast_end_index : int, optional
        The final epex_data point to forecast up to.
    prior_length : int, optional
        The Number of previous EPEX data points from which to build a prior upon which the subsequent "horizon" data_points are sampled.
    nsamps : int, optional
        The number of samples drawn from the forecast distribution. More samples will increase accuracy and run time.
    rho : float, optional
        A random effect extension, which increases the forecast variance (see Berry and West, 2019). Increasing this number will allow increased variance in the forecast. Suggested between [0,1].
    

    Returns
    -------
    datetimes : array of datetimes
        An array of datetimes covering from the initial EPEX data.
    prices : nd.array
        An array of the EPEX prices from the intial EPEX data.
    samples : nd.array
        A array of size [nsamps, # epex data points, horizon]. This is array of all "nsamps" samples (zeroth index), at every time (first index), predicting the price in the future by a certain amount (second index). e.g. samples[:,10,2], will give an array of all "nsamps" points used to predict the time 1.5 hours in the future from the 10th EPEX data_point.
    forecast : nd.array
        An array of size [# epex data points, horizon]. This array gives the median of the samples array, which is the forecasted price. i.e. forecast[10,2] gives the price prediction based upon the prior up to time 10, 1.5 hours in the future. 
    """
    prices = epex_data.values[:, 0]
    datetimes = epex_data.index

    forecast_start_date = datetimes[forecast_start_index]
    forecast_end_date = datetimes[forecast_end_index]

    mod, samples = analysis(
        prices,
        family="poisson",
        dates=datetimes,
        forecast_start=forecast_start_date,  # First time step to forecast on
        forecast_end=forecast_end_date,  # Final time step to forecast on
        ntrend=1,  # Intercept and slope in model
        nsamps=nsamps,  # Number of samples taken in the Poisson process
        seasPeriods=[
            48
        ],  # Length of the seasonal variations in the data - i.e. every 24hr here
        seasHarmComponents=[
            [1, 2, 3, 4, 6]
        ],  # Variations to pick out from the seaonal period
        k=horizon,  # Forecast horizon. If k>1, default is to forecast 1:k steps ahead, marginally
        prior_length=prior_length,  # How many data point to use in defining prior - 48=1 day
        rho=rho,  # Random effect extension, which increases the forecast variance (see Berry and West, 2019)
        deltrend=0.98,  # Discount factor on the trend component (the intercept)
        delregn=0.98,  # Discount factor on the regression component
        delSeas=0.98,
    )

    forecast = median(samples)

    return datetimes, prices, samples, forecast
