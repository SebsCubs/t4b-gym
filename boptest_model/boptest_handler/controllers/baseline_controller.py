# -*- coding: utf-8 -*-
"""
This module implements a simple P controller of heater power control specifically
for testcase1.

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


    """
    Setpoints:
    - hvac_oveAhu_TSupSet_u: 12 degrees always
    - hvac_oveAhu_dpSet_u: 50 Pa unoccupied, 400 Pa occupied
    - hvac_oveAhu_yFan_u: PI controller with setpoint hvac_oveAhu_dpSet_u and feedback hvac_reaAhu_dp_sup_y (fan discharge static pressure)
    - hvac_oveAhu_yOA_u: 0 unoccupied, 1 occupied
    - hvac_oveAhu_yRet_u: 1 unoccupied, 0 occupied

    Control signals:
    - hvac_oveAhu_yCoo_u: PI controller with setpoint hvac_oveAhu_TSupSet_u and feedback hvac_reaAhu_TSup_y 
    - hvac_oveAhu_yHea_u: PI controller with setpoint hvac_oveAhu_TSupSet_u and feedback hvac_reaAhu_TSup_y 
    - hvac_oveAhu_yPumCoo_u: 1 if hvac_oveAhu_yCoo_u > 0.1, 0 otherwise
    - hvac_oveAhu_yPumHea_u: 1 if hvac_oveAhu_yHea_u > 0.1, 0 otherwise
    """
    # Controller parameters
    hvac_oveAhu_TSupSet_u = 273.15+12
    
    seconds_in_day = y['time'] % (24 * 3600)  # Get seconds within the current day
    hour_of_day = seconds_in_day / 3600  # Convert to hours (0-24)
    
    # Occupied between 7am and 5pm
    if hour_of_day >= 7 and hour_of_day < 17:
        occupied = 1
    else:
        occupied = 0

    # hvac_oveAhu_dpSet_u
    hvac_oveAhu_dpSet_u = 50 + (occupied * 350)
    
    # hvac_oveAhu_yFan_u
    
    k_p_hvac_oveAhu_yFan_u = 0.005
    k_i_hvac_oveAhu_yFan_u = 0.0005
    hvac_oveAhu_yFan_u = 0.5 + (k_p_hvac_oveAhu_yFan_u * (hvac_oveAhu_dpSet_u - y['hvac_reaAhu_dp_sup_y'])) + (k_i_hvac_oveAhu_yFan_u * (hvac_oveAhu_dpSet_u - y['hvac_reaAhu_dp_sup_y']))  
    hvac_oveAhu_yFan_u = min(max(hvac_oveAhu_yFan_u, 0), 1)
    

    #hvac_oveAhu_yFan_u = 0 + (1 * occupied) 
    # hvac_oveAhu_yOA_u
    hvac_oveAhu_yOA_u = 0 + (1 * occupied)  
    # hvac_oveAhu_yRet_u
    hvac_oveAhu_yRet_u = 1 - occupied

    # hvac_oveAhu_yCoo_u and hvac_oveAhu_yHea_u
    k_p = 0.01
    k_i = 0.001
    error = hvac_oveAhu_TSupSet_u - y['hvac_reaAhu_TSup_y']
    
    if y['hvac_reaAhu_TSup_y'] < hvac_oveAhu_TSupSet_u:
        # Need heating
        hvac_oveAhu_yHea_u = min(max(0.5 + (k_p * error) + (k_i * error), 0), 1)
        hvac_oveAhu_yCoo_u = 0
    else:
        # Need cooling 
        hvac_oveAhu_yCoo_u = min(max(0.5 - (k_p * error) - (k_i * error), 0), 1)
        hvac_oveAhu_yHea_u = 0
    
    # hvac_oveAhu_yPumCoo_u
    hvac_oveAhu_yPumCoo_u = 1 if hvac_oveAhu_yCoo_u > 0.1 else 0

    # hvac_oveAhu_yPumHea_u
    hvac_oveAhu_yPumHea_u = 1 if hvac_oveAhu_yHea_u > 0.1 else 0
    
    u = {

        'hvac_oveAhu_TSupSet_u': hvac_oveAhu_TSupSet_u,
        'hvac_oveAhu_TSupSet_activate': 1,
        'hvac_oveAhu_dpSet_u': hvac_oveAhu_dpSet_u,
        'hvac_oveAhu_dpSet_activate': 1,
        'hvac_oveAhu_yFan_u': hvac_oveAhu_yFan_u,
        'hvac_oveAhu_yFan_activate': 1,
        'hvac_oveAhu_yOA_u': hvac_oveAhu_yOA_u,
        'hvac_oveAhu_yOA_activate': 1,
        'hvac_oveAhu_yRet_u': hvac_oveAhu_yRet_u,
        'hvac_oveAhu_yRet_activate': 1,
        'hvac_oveAhu_yCoo_u': hvac_oveAhu_yCoo_u,
        'hvac_oveAhu_yCoo_activate': 1,
        'hvac_oveAhu_yHea_u': hvac_oveAhu_yHea_u,
        'hvac_oveAhu_yHea_activate': 1,
        'hvac_oveAhu_yPumCoo_u': hvac_oveAhu_yPumCoo_u,
        'hvac_oveAhu_yPumCoo_activate': 1,
        'hvac_oveAhu_yPumHea_u': hvac_oveAhu_yPumHea_u,
    }

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

    u = {
        'hvac_oveAhu_TSupSet_u': 0,
        'hvac_oveAhu_TSupSet_activate': 1,
        'hvac_oveAhu_dpSet_u': 0,
        'hvac_oveAhu_dpSet_activate': 1,
        'hvac_oveAhu_yFan_u': 0,
        'hvac_oveAhu_yFan_activate': 1,
        'hvac_oveAhu_yOA_u': 0,
        'hvac_oveAhu_yOA_activate': 1,
        'hvac_oveAhu_yRet_u': 0,
        'hvac_oveAhu_yRet_activate': 1,
        'hvac_oveAhu_yCoo_u': 0,
        'hvac_oveAhu_yCoo_activate': 1,
        'hvac_oveAhu_yHea_u': 0,
        'hvac_oveAhu_yHea_activate': 1,
        'hvac_oveAhu_yPumCoo_u': 0,
        'hvac_oveAhu_yPumCoo_activate': 1,
        'hvac_oveAhu_yPumHea_u': 0,
        'hvac_oveAhu_yPumHea_activate': 1,
    }

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

    forecast_parameters = {'point_names':['weaSta_reaWeaSolTim_y'],
                           'horizon': 600,
                           'interval': 600}


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
        t = forecast_data['time'][0]
        forecasts.loc[t,i] = forecast_data[i][0]

    return forecasts