"""
This script runs a simple simulation of the boptest test case: multizone_office_simple_air
It assumes the boptest service is running in the localhost:80 port.
"""

# GENERAL PACKAGE IMPORT
# ----------------------
import sys
import os
sys.path.insert(0, '/'.join((os.path.dirname(os.path.abspath(__file__))).split('/')[:-2]))
from interface import control_test_with_points
import pandas as pd

def run(scenario = 'typical_heat_day',points=None, plot=False, url='http://127.0.0.1:80', save_forecasts=False):
    """Run test case.
    Parameters
    ----------
    plot : bool, optional
        True to plot timeseries results.
        Default is False.

    Returns
    -------
    kpi : dict
        Dictionary of core KPI names and values.
        {kpi_name : value}
    res : dict
        Dictionary of trajectories of inputs and outputs.
    custom_kpi_result: dict
        Dictionary of tracked custom KPI calculations.
        Empty if no customized KPI calculations defined.

    """

    # RUN THE CONTROL TEST
    # --------------------
    control_module = 'controllers.baseline'
    forecast_points = ['Occupancy[cor]',
                        'Occupancy[nor]',
                        'Occupancy[sou]',
                        'Occupancy[eas]',
                        'Occupancy[wes]'] #from the controller definition

    scenario = {'time_period': scenario, 'electricity_price': 'dynamic'}
    step = 600
    # ---------------------------------------

    # RUN THE CONTROL TEST
    # --------------------
    """   
    points = ['hvac_reaAhu_TSup_y',         #vent_supply_air_temp_sensor
              'hvac_reaAhu_V_flow_sup_y',   #vent_airflow_sensor
              'weaSta_reaWeaTWetBul_y',     #vent_outdoor_air_temp_sensor
              'hvac_reaAhu_PFanSup_y',      #vent_power_sensor
              "hvac_reaAhu_TRet_y",         #vent_return_air_temp_sensor
              "hvac_reaAhu_TMix_y",        #vent_mixed_air_temp_sensor
              "hvac_reaAhu_V_flow_ret_y",   #vent_return_airflow_sensor
              "hvac_oveAhu_yOA_u",          #vent_supply_damper_setpoint, vent_return_damper_setpoint
              "hvac_oveAhu_yRet_u",         #vent_mixing_damper_setpoint
              'hvac_oveAhu_TSupSet_u',      #vent_supply_air_temp_setpoint
              ]
    
     
    # Control testing points
    points = ["hvac_oveAhu_yFan_u", 
              "hvac_oveAhu_dpSet_u" , 
              "hvac_oveAhu_TSupSet_u",
              "hvac_oveAhu_yOA_u",
              "hvac_oveAhu_yRet_u",
              "hvac_oveAhu_yHea_u",
              "hvac_oveAhu_yCoo_u",
              "hvac_oveAhu_yPumHea_u",
              "hvac_oveAhu_yPumCoo_u"
              ] #fan speed setpoint, duct pressure setpoint, supply air temperature setpoint
    """

    kpi, df_res, custom_kpi_result, forecasts = control_test_with_points('multizone_office_simple_air',
                                                             control_module,
                                                             scenario=scenario,
                                                             step=step,
                                                             points=points,
                                                             use_forecast=save_forecasts,
                                                             url=url)

    # POST-PROCESS RESULTS
    # --------------------
    time = df_res.index.values / 3600  # convert s --> hr
    #zone_temperature = df_res['hvac_reaAhu_TSup_y'].values - 273.15  # convert K --> C
    # Plot results
    if plot:
        try:
            from matplotlib import pyplot as plt
            import numpy as np
            
            # Plot supply air temperature
            plt.figure(1)
            plt.title('Fan speed setpoint')
            plt.plot(time, df_res['hvac_oveAhu_yFan_u'].values, label='Fan speed')
            plt.ylabel('Speed [0-1]')
            plt.xlabel('Time [hr]')
            plt.legend()
            
            # Plot outside air damper position
            plt.figure(2) 
            plt.title('Supply duct pressure setpoint')
            plt.plot(time, df_res['hvac_oveAhu_dpSet_u'].values, label='Supply duct pressure')
            plt.ylabel('Pressure [Pa]')
            plt.xlabel('Time [hr]')
            plt.legend()
            
            # Plot return air damper position
            plt.figure(3)
            plt.title('Supply air temperature setpoint')
            plt.plot(time, df_res['hvac_oveAhu_TSupSet_u'].values, label='Supply air temperature')
            plt.ylabel('Temperature [K]')
            plt.xlabel('Time [hr]')
            plt.legend()
            
            plt.show()
        except ImportError:
            print("Cannot import numpy or matplotlib for plot generation")

    # Save the results to a csv file for each point with two columns: time and value
    # Create directory for results if it doesn't exist
    results_dir = os.path.join('data', scenario['time_period'])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    for point in points:
        df_res[[point]].to_csv(os.path.join(results_dir, f'{point}.csv'), index=True)
    
    if save_forecasts:
        forecasts_dir = os.path.join('data', scenario['time_period'], 'forecasts')
        if not os.path.exists(forecasts_dir):
            os.makedirs(forecasts_dir)
        for point in forecast_points:
            # Save forecast data directly, similar to df_res points
            forecasts[[point]].to_csv(os.path.join(forecasts_dir, f'{point}.csv'), index=True)
    
    return kpi, df_res, custom_kpi_result


if __name__ == "__main__":
    scenarios = ['typical_heat_day', 'typical_cool_day', 'mix_day']
    """
    # Define zones and measurement types
    zones = ['Cor', 'Nor', 'Sou', 'Eas', 'Wes']
    measurements = [
        {
            'name': 'Supply air temperature',
            'prefix': 'hvac_reaZon',
            'suffix': 'TSup_y'
        },
        {
            'name': 'Supply air flow rate',
            'prefix': 'hvac_reaZon',
            'suffix': 'V_flow_y'
        },
        {
            'name': 'CO2 concentration',
            'prefix': 'hvac_reaZon',
            'suffix': 'CO2Zon_y'
        },
        {
            'name': 'Indoor air temperature',
            'prefix': 'hvac_reaZon',
            'suffix': 'TZon_y'
        },
        {
            'name': 'Supply damper position',
            'prefix': 'hvac_oveZonAct',
            'suffix': 'yDam_u'
        },
        {
            'name': 'Heating setpoint',
            'prefix': 'hvac_oveZonSup',
            'suffix': 'TZonHeaSet_u'
        },
        {
            'name': 'Cooling setpoint',
            'prefix': 'hvac_oveZonSup',
            'suffix': 'TZonCooSet_u'
        }
    ]

    # For each measurement type
    #for measurement in measurements:
    measurement =         {
        'name': 'Cooling setpoint',
        'prefix': 'hvac_oveZonSup',
        'suffix': 'TZonCooSet_u'
    }
    points = [f"{measurement['prefix']}{zone}_{measurement['suffix']}" for zone in zones]
    print(f"\nProcessing {measurement['name']} measurements...")
    print(f"Points to process: {points}")
    """
    points = ["hvac_oveZonActCor_yReaHea_u", "hvac_oveZonActNor_yReaHea_u", "hvac_oveZonActSou_yReaHea_u", "hvac_oveZonActEas_yReaHea_u", "hvac_oveZonActWes_yReaHea_u"] # Reheat control signal
    url='http://192.168.8.65:80'
    
    #url='http://127.0.0.1:80'
    # For each scenario
    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario}")
        kpi, df_res, custom_kpi_result = run(
            scenario=scenario, 
            points=points, 
            plot=False, 
            url=url,
            save_forecasts=False
        )
        print(f"Successfully completed {scenario}")



