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


def custom_map_function(self):
    input_signal = self.input["inputSignal"].get()
    #heating valve position is 0-1, if the input signal is positiv, the valve is open with the same value but clamped at 1
    if input_signal > 0:
        self.output["heatingValvePosition"].set(min(input_signal, 1))
        self.output["coolingValvePosition"].set(0)
    else:
        self.output["heatingValvePosition"].set(0)
        self.output["coolingValvePosition"].set(min(-input_signal, 1))
    #Damper position is the absolute value of the input signal, clamped at 1
    damper_position = min(abs(input_signal), 1)
    self.output["supplyDamperPosition"].set(damper_position)
    #Return damper position is 1 - supply damper position
    self.output["returnDamperPosition"].set(1 - damper_position)


def fcn(self):
    '''
        The fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''

    vent_supply_air_temp_sensor = tb.SensorSystem(id="vent_supply_air_temp_sensor", saveSimulationResult=True)
    vent_airflow_sensor = tb.SensorSystem(id="vent_airflow_sensor", saveSimulationResult=True)
    vent_supply_damper_position_sensor = tb.SensorSystem(id="vent_supply_damper_position_sensor", saveSimulationResult=True)
    vent_return_damper_position_sensor = tb.SensorSystem(id="vent_return_damper_position_sensor", saveSimulationResult=True)
    vent_outdoor_air_temp_sensor = tb.SensorSystem(id="vent_outdoor_air_temp_sensor", saveSimulationResult=True)
    vent_power_sensor = tb.SensorSystem(id="vent_power_sensor", saveSimulationResult=True)
    supply_heating_coil_water_temp_sensor = tb.SensorSystem(id="supply_heating_coil_water_temp_sensor", saveSimulationResult=True)
    return_heating_coil_water_temp_sensor = tb.SensorSystem(id="return_heating_coil_water_temp_sensor", saveSimulationResult=True)
    supply_cooling_coil_water_temp_sensor = tb.SensorSystem(id="supply_cooling_coil_water_temp_sensor", saveSimulationResult=True)
    return_cooling_coil_water_temp_sensor = tb.SensorSystem(id="return_cooling_coil_water_temp_sensor", saveSimulationResult=True)

    # Add AHU fan
    supply_fan = tb.FanSystem(id="supply_fan", saveSimulationResult=True)
    self.add_connection(vent_airflow_sensor, supply_fan, "airFlowRateIn", "airFlowRate")
    self.add_connection(supply_fan, vent_power_sensor, "Power", "measuredPower")

    # Add AHU heating coil
    supply_heating_coil = tb.CoilPumpValveFMUSystem(id="[supply_heating_coil][heating_pump][heating_valve]", saveSimulationResult=True)
    self.add_connection(vent_outdoor_air_temp_sensor, supply_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(vent_airflow_sensor, supply_heating_coil, "airFlowRateIn", "airFlowRate")

    self.add_connection(supply_heating_coil_water_temp_sensor, supply_heating_coil, "supplyWaterTemperature", "supplyWaterTemperature")
    self.add_connection(supply_heating_coil, return_heating_coil_water_temp_sensor, "outletWaterTemperature", "inletWaterTemperature")

    # Add AHU cooling coil
    supply_cooling_coil = tb.CoilPumpValveFMUSystem(id="[supply_cooling_coil][cooling_pump][cooling_valve]", saveSimulationResult=True)
    self.add_connection(supply_heating_coil, supply_cooling_coil, "outletAirTemperature", "inletAirTemperature")
    self.add_connection(vent_airflow_sensor, supply_cooling_coil, "airFlowRateIn", "airFlowRate")

    self.add_connection(supply_cooling_coil, vent_supply_air_temp_sensor, "outletAirTemperature", "supplyAirTemperature")
    self.add_connection(supply_cooling_coil_water_temp_sensor, supply_cooling_coil, "supplyWaterTemperature", "supplyWaterTemperature")
    self.add_connection(supply_cooling_coil, return_cooling_coil_water_temp_sensor, "outletWaterTemperature", "inletWaterTemperature")

    # Add main dampers
    main_supply_damper = tb.DamperSystem(id="main_supply_damper", saveSimulationResult=True)
    #self.add_connection(main_supply_damper, vent_supply_damper_position_sensor, "damperPosition", "damperPosition")
    main_return_damper = tb.DamperSystem(id="main_return_damper", saveSimulationResult=True)
    #self.add_connection(main_return_damper, vent_return_damper_position_sensor, "damperPosition", "damperPosition")
    supply_air_temp_setpoint = tb.ScheduleSystem(id="supply_air_temp_setpoint", saveSimulationResult=True)
    supply_air_temp_controller = tb.PIControllerFMUSystem(id="supply_air_temp_controller", isReverse=False, saveSimulationResult=True)

    self.add_connection(supply_air_temp_setpoint, supply_air_temp_controller, "scheduleValue", "setpointValue")
    self.add_connection(vent_supply_air_temp_sensor, supply_air_temp_controller, "supplyAirTemperature", "actualValue")
     
    c3_control_map = tb.ControlSignalMapSystem(id="c3_control_map", saveSimulationResult=True)
    self.add_connection(supply_air_temp_controller, c3_control_map, "inputSignal", "actualValue")
    self.add_connection(c3_control_map, vent_supply_damper_position_sensor, "supplyDamperPosition", "damperPosition")
    self.add_connection(c3_control_map, vent_return_damper_position_sensor, "returnDamperPosition", "damperPosition")
    self.add_connection(c3_control_map, supply_heating_coil, "heatingValvePosition", "valvePosition")
    self.add_connection(c3_control_map, supply_cooling_coil, "coolingValvePosition", "valvePosition")
    
    # Replace the do_step method with your custom function
    c3_control_map.do_step = custom_map_function.__get__(c3_control_map, tb.ControlSignalMapSystem)

    

def get_model(id=None, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn
    model = tb.Model(id="only_ahu_model", saveSimulationResult=True)
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