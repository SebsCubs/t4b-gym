# -*- coding: utf-8 -*-
"""
This module implements the baseline control for testcases.

"""
import pandas as pd

def compute_control(y, forecasts=None):
    """Compute the control input from the measurement.

    Parameters
    ----------
    y : dict
        Contains the current values of the measurements.
        {<measurement_name>:<measurement_value>}
    forecasts : structure depends on controller, optional
        Forecasts used to calculate control.
        Default is None.

    Returns
    -------
    u : dict
        Defines the control input to be used for the next step.
        {<input_name> : <input_value>}

    """

    # Compute control
    u = {}

    return u


def initialize():
    """Initialize the control input u.

    Parameters
    ----------
    None

    Returns
    -------
    u : dict
        Defines the control input to be used for the next step.
        {<input_name> : <input_value>}

    """
    u = {}

    return u


def get_forecast_parameters():
    """Get forecast parameters within the controller.

    Returns
    -------
    forecast_parameters: dict
        {'point_names':[<string>],
         'horizon': <int>,
         'interval': <int>}

    """
    # Occupancy[cor], Occupancy[nor], Occupancy[sou], Occupancy[eas], Occupancy[wes]

    forecast_parameters = {'point_names':['UpperCO2[cor]',
                                            'UpperCO2[nor]',
                                            'UpperCO2[sou]',
                                            'UpperCO2[eas]',
                                            'UpperCO2[wes]'],
                           'horizon': 600,
                           'interval': 300}


    return forecast_parameters

def update_forecasts(forecast_data, forecasts):
    """Update forecast_store within the controller.

    This controller only uses the first timestep of the forecast for upper
    and lower temperature limits.


    Parameters
    ----------
    forecast_data: dict
        Dictionary of arrays with forecast data from BOPTEST
        {<point_name1: [<data>]}
    forecasts: DataFrame
        DataFrame of forecast values used over time.

    Returns
    -------
    forecasts: DataFrame
        Updated DataFrame of forcast values used over time.

    """
    forecast_config = get_forecast_parameters()['point_names']

    if forecasts is None:
        forecasts = pd.DataFrame(columns=forecast_config)
    for i in forecast_config:
        #Takes the first value of the forecast data only
        t = forecast_data['time'][0]
        forecasts.loc[t,i] = forecast_data[i][0]

    return forecasts
