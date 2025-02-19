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

def run(plot=False):
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
    scenario = {'time_period': 'mix_day', 'electricity_price': 'dynamic'}
    step = 600
    # ---------------------------------------

    # RUN THE CONTROL TEST
    # --------------------
    points = ['hvac_reaAhu_TSup_y', 'hvac_reaAhu_V_flow_sup_y', 'weaSta_reaWeaTWetBul_y', 'hvac_oveAhu_TSupSet_u', 'hvac_oveAhu_yOA_u', 'hvac_oveAhu_yRet_u', 'hvac_reaAhu_PFanSup_y']

    kpi, df_res, custom_kpi_result, forecasts = control_test_with_points('multizone_office_simple_air',
                                                             control_module,
                                                             scenario=scenario,
                                                             step=step,
                                                             points=points)

    # POST-PROCESS RESULTS
    # --------------------
    time = df_res.index.values / 3600  # convert s --> hr
    zone_temperature = df_res['hvac_reaAhu_TSup_y'].values - 273.15  # convert K --> C
    # Plot results
    if plot:
        try:
            from matplotlib import pyplot as plt
            import numpy as np
            
            # Plot supply air temperature
            plt.figure(1)
            plt.title('Supply Air Temperature')
            plt.plot(time, zone_temperature, label='Supply Air Temp')
            plt.ylabel('Temperature [C]')
            plt.xlabel('Time [hr]')
            plt.legend()
            
            # Plot outside air damper position
            plt.figure(2) 
            plt.title('Outside Air Damper Position')
            plt.plot(time, df_res['hvac_oveAhu_yOA_u'].values, label='OA Damper')
            plt.ylabel('Position [0-1]')
            plt.xlabel('Time [hr]')
            plt.legend()
            
            # Plot return air damper position
            plt.figure(3)
            plt.title('Return Air Damper Position')
            plt.plot(time, df_res['hvac_oveAhu_yRet_u'].values, label='Return Damper')
            plt.ylabel('Position [0-1]')
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
    
    return kpi, df_res, custom_kpi_result


if __name__ == "__main__":
    kpi, df_res, custom_kpi_result = run(plot=False)



