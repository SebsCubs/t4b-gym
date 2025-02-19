import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Mac specific?
import datetime
import pandas as pd
from dateutil.tz import gettz
import sys
# Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = os.path.join(uppath(os.path.abspath(__file__), 4), "Twin4Build")
    sys.path.append(file_path)
import twin4build as tb
import twin4build.utils.plot.plot as plot
from twin4build.utils.uppath import uppath
import numpy as np
import cProfile, io, pstats
from twin4build.utils.rsetattr import rsetattr
import matplotlib.pyplot as plt
sys.setrecursionlimit(2000)  # You can adjust this number as needed

def fcn(self):
    '''
        The fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''

    vent_airflow_sensor = tb.SensorSystem(id="vent_airflow_sensor", saveSimulationResult=True)
    vent_power_sensor = tb.SensorSystem(id="vent_power_sensor", saveSimulationResult=True)
    
    # Add AHU fan
    supply_fan = tb.FanSystem(id="supply_fan", saveSimulationResult=True)
    self.add_connection(vent_airflow_sensor, supply_fan, "airFlowRateIn", "airFlowRate")
    self.add_connection(supply_fan, vent_power_sensor, "Power", "measuredPower")

def get_model(id=None, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn
    model = tb.Model(id="fan_estimation_model", saveSimulationResult=True)
    model.load(fcn=fcn_, create_signature_graphs=False, validate_model=True, verbose=True, force_config_update=True)
    if id is not None:
        model.id = id
    return model

def run():
    stepSize = 600  # Seconds
    
    startTime = datetime.datetime(year=2024, month=1, day=3, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=1, day=4, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = get_model()

    simulator = tb.Simulator()

    simulator.simulate(model = model,
                       startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize)

    print("Simulation completed successfully!")


if __name__ == "__main__":
    run()