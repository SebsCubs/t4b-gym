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

    """
    Components to add:
    -  AHU fan
    -  AHU heating coil
    -  AHU cooling coil
    -  AHU Dampers
    -  Flow junctions
 
    supply_flow_junction = tb.SupplyFlowJunctionSystem(id="supply_flow_junction",
                                                        saveSimulationResult=True)

    return_flow_junction = tb.ReturnFlowJunctionSystem(id="return_flow_junction",
                                                        saveSimulationResult=True)
    """ 
    
    # Add AHU fan
    supply_fan = tb.FanSystem(id="supply_fan", saveSimulationResult=True)
    self.add_connection(self.components["supply_flow_junction"], supply_fan, "airFlowRateIn", "airFlowRate")

    # Add AHU heating coil


    supply_heating_coil = tb.CoilPumpValveFMUSystem(id="[supply_heating_coil][heating_pump][heating_valve]", saveSimulationResult=True)
    self.add_connection(supply_heating_coil, self.components["vent_supply_air_temp_sensor"], "outletAirTemperature", "supplyAirTemperature")
    self.add_connection(self.components["supply_flow_junction"], supply_heating_coil, "airFlowRateIn", "airFlowRate")
    
    # Add AHU cooling coil
    supply_cooling_coil = tb.CoilPumpValveFMUSystem(id="[supply_cooling_coil][cooling_pump][cooling_valve]", saveSimulationResult=True)
    self.add_connection(supply_cooling_coil, supply_heating_coil, "outletAirTemperature", "inletAirTemperature")
    self.add_connection(self.components["supply_flow_junction"], supply_cooling_coil, "airFlowRateIn", "airFlowRate")

    # Add core heating coil
    core_reheating_coil = tb.CoilHeatingSystem(id="core_reheating_coil", saveSimulationResult=True)
    self.add_connection(self.components["core_temperature_heating_setpoint"], core_reheating_coil, "scheduleValue", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], core_reheating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(self.components["core_supply_damper"], core_reheating_coil, "airFlowRate", "airFlowRate")
    self.remove_connection(self.components["vent_supply_air_temp_sensor"], self.components["core"], "measuredValue", "supplyAirTemperature") #This connection was generated from the semantic model but is not needed
    self.add_connection(core_reheating_coil, self.components["core"], "outletAirTemperature", "supplyAirTemperature")

    #Add core sensors
    core_co2_sensor = tb.SensorSystem(id="core_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["core"], core_co2_sensor, "indoorCo2Concentration", "core_indoorCo2Concentration")
    core_supply_air_temp_sensor = tb.SensorSystem(id="core_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(core_reheating_coil, core_supply_air_temp_sensor, "outletAirTemperature", "core_supplyAirTemperature")
    core_supply_airflow_sensor = tb.SensorSystem(id="core_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(self.components["core_supply_damper"], core_supply_airflow_sensor, "airFlowRate", "core_supplyAirflow")
    

    #Add north heating coil
    north_heating_coil = tb.CoilHeatingSystem(id="north_heating_coil", saveSimulationResult=True)
    self.add_connection(self.components["north_temperature_heating_setpoint"], north_heating_coil, "scheduleValue", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], north_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(self.components["north_supply_damper"], north_heating_coil, "airFlowRate", "airFlowRate")
    self.add_connection(north_heating_coil, self.components["north"], "outletAirTemperature", "supplyAirTemperature")

    #Add north sensors
    north_co2_sensor = tb.SensorSystem(id="north_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["north"], north_co2_sensor, "indoorCo2Concentration", "north_indoorCo2Concentration")
    north_supply_air_temp_sensor = tb.SensorSystem(id="north_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(north_heating_coil, north_supply_air_temp_sensor, "outletAirTemperature", "north_supplyAirTemperature")
    north_supply_airflow_sensor = tb.SensorSystem(id="north_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(self.components["north_supply_damper"], north_supply_airflow_sensor, "airFlowRate", "north_supplyAirflow")

    #Add south heating coil
    south_heating_coil = tb.CoilHeatingSystem(id="south_heating_coil", saveSimulationResult=True)
    self.add_connection(self.components["south_temperature_heating_setpoint"], south_heating_coil, "scheduleValue", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], south_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(self.components["south_supply_damper"], south_heating_coil, "airFlowRate", "airFlowRate")
    self.add_connection(south_heating_coil, self.components["south"], "outletAirTemperature", "supplyAirTemperature")

    #Add south sensors
    south_co2_sensor = tb.SensorSystem(id="south_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["south"], south_co2_sensor, "indoorCo2Concentration", "south_indoorCo2Concentration")
    south_supply_air_temp_sensor = tb.SensorSystem(id="south_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(south_heating_coil, south_supply_air_temp_sensor, "outletAirTemperature", "south_supplyAirTemperature")
    south_supply_airflow_sensor = tb.SensorSystem(id="south_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(self.components["south_supply_damper"], south_supply_airflow_sensor, "airFlowRate", "south_supplyAirflow")

    #Add east heating coil
    east_heating_coil = tb.CoilHeatingSystem(id="east_heating_coil", saveSimulationResult=True)
    self.add_connection(self.components["east_temperature_heating_setpoint"], east_heating_coil, "scheduleValue", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], east_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(self.components["east_supply_damper"], east_heating_coil, "airFlowRate", "airFlowRate")
    self.add_connection(east_heating_coil, self.components["east"], "outletAirTemperature", "supplyAirTemperature")

    #Add east sensors
    east_co2_sensor = tb.SensorSystem(id="east_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["east"], east_co2_sensor, "indoorCo2Concentration", "east_indoorCo2Concentration")
    east_supply_air_temp_sensor = tb.SensorSystem(id="east_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(east_heating_coil, east_supply_air_temp_sensor, "outletAirTemperature", "east_supplyAirTemperature")
    east_supply_airflow_sensor = tb.SensorSystem(id="east_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(self.components["east_supply_damper"], east_supply_airflow_sensor, "airFlowRate", "east_supplyAirflow")

    #Add west heating coil
    west_heating_coil = tb.CoilHeatingSystem(id="west_heating_coil", saveSimulationResult=True)
    self.add_connection(self.components["west_temperature_heating_setpoint"], west_heating_coil, "scheduleValue", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], west_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(self.components["west_supply_damper"], west_heating_coil, "airFlowRate", "airFlowRate")
    self.add_connection(west_heating_coil, self.components["west"], "outletAirTemperature", "supplyAirTemperature")

    #Add west sensors
    west_co2_sensor = tb.SensorSystem(id="west_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["west"], west_co2_sensor, "indoorCo2Concentration", "west_indoorCo2Concentration")
    west_supply_air_temp_sensor = tb.SensorSystem(id="west_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(west_heating_coil, west_supply_air_temp_sensor, "outletAirTemperature", "west_supplyAirTemperature")
    west_supply_airflow_sensor = tb.SensorSystem(id="west_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(self.components["west_supply_damper"], west_supply_airflow_sensor, "airFlowRate", "west_supplyAirflow")

    #Add main dampers
    main_supply_damper = tb.DamperSystem(id="main_supply_damper", saveSimulationResult=True)
    mixing_damper = tb.DamperSystem(id="mixing_damper", saveSimulationResult=True)
    main_return_damper = tb.DamperSystem(id="main_return_damper", saveSimulationResult=True)

    #Add supply temperature setpoint
    supply_air_temp_setpoint = tb.ScheduleSystem(id="supply_air_temp_setpoint", saveSimulationResult=True)
    #Add controller for supply temperature setpoint
    supply_air_temp_controller = tb.PIControllerFMUSystem(id="supply_air_temp_controller", isReverse=False, saveSimulationResult=True)
    self.add_connection(supply_air_temp_setpoint, supply_air_temp_controller, "scheduleValue", "setpointValue")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], supply_air_temp_controller, "supplyAirTemperature", "actualValue")
    self.add_connection(supply_air_temp_controller, main_supply_damper, "inputSignal", "damperPosition")
    self.add_connection(supply_air_temp_controller, mixing_damper, "inputSignal", "damperPosition")
    self.add_connection(supply_air_temp_controller, main_return_damper, "inputSignal", "damperPosition")
    self.add_connection(supply_air_temp_controller, supply_heating_coil, "inputSignal", "valvePosition")
    self.add_connection(supply_air_temp_controller, supply_cooling_coil, "inputSignal", "valvePosition")
    
       
    
    
    

def get_model(id=None, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn
    model = tb.Model(id="five_rooms_only_template", saveSimulationResult=True)
    
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), "five_rooms_only_template.xlsm")
    model.load(semantic_model_filename=filename, fcn=fcn_, create_signature_graphs=False, validate_model=True, verbose=True, force_config_update=True)
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