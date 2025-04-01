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
    #Add core temp setpoint profiles
    core_temperature_heating_setpoint = tb.ScheduleSystem(id="core_temperature_heating_setpoint", saveSimulationResult=True)
    core_temperature_cooling_setpoint = tb.ScheduleSystem(id="core_temperature_cooling_setpoint", saveSimulationResult=True)

    #Add core temp controller
    core_temperature_heating_controller = tb.VAVReheatControllerSystem(id="core_temperature_heating_controller", saveSimulationResult=True)
    self.add_connection(core_temperature_heating_setpoint, core_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(core_temperature_cooling_setpoint, core_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(self.components["core"], core_temperature_heating_controller, "indoorTemperature", "roomTemp")
    #self.add_connection(core_temperature_heating_controller, self.components["core"], "supplyAirTemp", "supplyAirTemperature")

    #Add core dampers
    core_supply_damper = tb.DamperSystem(id="core_supply_damper", saveSimulationResult=True)
    core_exhaust_damper = tb.DamperSystem(id="core_exhaust_damper", saveSimulationResult=True)
    self.add_connection(core_temperature_heating_controller, core_supply_damper, "y_dam", "damperPosition")
    self.add_connection(core_temperature_heating_controller, core_exhaust_damper, "y_dam", "damperPosition")
    self.add_connection(core_supply_damper, self.components["core"], "airFlowRate", "airFlowRate")

    # Add core heating coil
    core_reheating_coil = tb.CoilHeatingSystem(id="core_reheating_coil", saveSimulationResult=True)
    self.add_connection(core_temperature_heating_controller, core_reheating_coil, "supplyAirTemp", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], core_reheating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(self.components["core_supply_damper"], core_reheating_coil, "airFlowRate", "airFlowRate")
    self.remove_connection(self.components["vent_supply_air_temp_sensor"], self.components["core"], "measuredValue", "supplyAirTemperature") #This connection was generated from the semantic model but is not needed
    self.add_connection(core_reheating_coil, self.components["core"], "outletAirTemperatureSetpoint", "supplyAirTemperature")

    #Add core sensors
    core_co2_sensor = tb.SensorSystem(id="core_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["core"], core_co2_sensor, "indoorCo2Concentration", "core_indoorCo2Concentration")
    core_supply_air_temp_sensor = tb.SensorSystem(id="core_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(core_reheating_coil, core_supply_air_temp_sensor, "outletAirTemperature", "core_supplyAirTemperature")
    core_supply_airflow_sensor = tb.SensorSystem(id="core_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(self.components["core_supply_damper"], core_supply_airflow_sensor, "airFlowRate", "core_supplyAirflow")
    core_supply_damper_position = tb.SensorSystem(id="core_supply_damper_position", saveSimulationResult=True)
    self.add_connection(self.components["core_supply_damper"], core_supply_damper_position, "damperPosition", "core_supplyDamperPosition")
    
    
    #Add north temp setpoint profiles
    north_temperature_heating_setpoint = tb.ScheduleSystem(id="north_temperature_heating_setpoint", saveSimulationResult=True)
    north_temperature_cooling_setpoint = tb.ScheduleSystem(id="north_temperature_cooling_setpoint", saveSimulationResult=True)

    #Add north temp controller
    north_temperature_heating_controller = tb.VAVReheatControllerSystem(id="north_temperature_heating_controller", saveSimulationResult=True)
    self.add_connection(north_temperature_heating_setpoint, north_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(north_temperature_cooling_setpoint, north_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(self.components["north"], north_temperature_heating_controller, "indoorTemperature", "roomTemp")
    self.add_connection(north_temperature_heating_controller, self.components["north"], "supplyAirTemp", "supplyAirTemperature")

    #Add north dampers
    north_supply_damper = tb.DamperSystem(id="north_supply_damper", saveSimulationResult=True)
    north_exhaust_damper = tb.DamperSystem(id="north_exhaust_damper", saveSimulationResult=True)
    self.add_connection(north_temperature_heating_controller, north_supply_damper, "y_dam", "damperPosition")
    self.add_connection(north_temperature_heating_controller, north_exhaust_damper, "y_dam", "damperPosition")
    self.add_connection(north_supply_damper, self.components["north"], "airFlowRate", "airFlowRate")

    #Add north heating coil
    north_heating_coil = tb.CoilHeatingSystem(id="north_heating_coil", saveSimulationResult=True)
    self.add_connection(north_temperature_heating_controller, north_heating_coil, "supplyAirTemp", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], north_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(north_supply_damper, north_heating_coil, "airFlowRate", "airFlowRate")
    

    #Add north sensors
    north_co2_sensor = tb.SensorSystem(id="north_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["north"], north_co2_sensor, "indoorCo2Concentration", "north_indoorCo2Concentration")
    north_supply_air_temp_sensor = tb.SensorSystem(id="north_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(north_heating_coil, north_supply_air_temp_sensor, "outletAirTemperature", "north_supplyAirTemperature")
    north_supply_airflow_sensor = tb.SensorSystem(id="north_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(north_supply_damper, north_supply_airflow_sensor, "airFlowRate", "north_supplyAirflow")
    north_supply_damper_position = tb.SensorSystem(id="north_supply_damper_position", saveSimulationResult=True)
    self.add_connection(north_supply_damper, north_supply_damper_position, "damperPosition", "north_supplyDamperPosition")

    #Add south temp setpoint profiles
    south_temperature_heating_setpoint = tb.ScheduleSystem(id="south_temperature_heating_setpoint", saveSimulationResult=True)
    south_temperature_cooling_setpoint = tb.ScheduleSystem(id="south_temperature_cooling_setpoint", saveSimulationResult=True)

    #Add south temp controller
    south_temperature_heating_controller = tb.VAVReheatControllerSystem(id="south_temperature_heating_controller", saveSimulationResult=True)
    self.add_connection(south_temperature_heating_setpoint, south_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(south_temperature_cooling_setpoint, south_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(self.components["south"], south_temperature_heating_controller, "indoorTemperature", "roomTemp")
    self.add_connection(south_temperature_heating_controller, self.components["south"], "supplyAirTemp", "supplyAirTemperature")

    #Add south dampers
    south_supply_damper = tb.DamperSystem(id="south_supply_damper", saveSimulationResult=True)
    south_exhaust_damper = tb.DamperSystem(id="south_exhaust_damper", saveSimulationResult=True)
    self.add_connection(south_temperature_heating_controller, south_supply_damper, "y_dam", "damperPosition")
    self.add_connection(south_temperature_heating_controller, south_exhaust_damper, "y_dam", "damperPosition")
    self.add_connection(south_supply_damper, self.components["south"], "airFlowRate", "airFlowRate")
    #Add south heating coil
    south_heating_coil = tb.CoilHeatingSystem(id="south_heating_coil", saveSimulationResult=True)
    self.add_connection(south_temperature_heating_controller, south_heating_coil, "supplyAirTemp", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], south_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(south_supply_damper, south_heating_coil, "airFlowRate", "airFlowRate")
    

    #Add south sensors
    south_co2_sensor = tb.SensorSystem(id="south_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["south"], south_co2_sensor, "indoorCo2Concentration", "south_indoorCo2Concentration")
    south_supply_air_temp_sensor = tb.SensorSystem(id="south_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(south_heating_coil, south_supply_air_temp_sensor, "outletAirTemperature", "south_supplyAirTemperature")
    south_supply_airflow_sensor = tb.SensorSystem(id="south_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(south_supply_damper, south_supply_airflow_sensor, "airFlowRate", "south_supplyAirflow")
    south_supply_damper_position = tb.SensorSystem(id="south_supply_damper_position", saveSimulationResult=True)
    self.add_connection(south_supply_damper, south_supply_damper_position, "damperPosition", "south_supplyDamperPosition")

    #Add east temp setpoint profiles
    east_temperature_heating_setpoint = tb.ScheduleSystem(id="east_temperature_heating_setpoint", saveSimulationResult=True)
    east_temperature_cooling_setpoint = tb.ScheduleSystem(id="east_temperature_cooling_setpoint", saveSimulationResult=True)

    #Add east temp controller
    east_temperature_heating_controller = tb.VAVReheatControllerSystem(id="east_temperature_heating_controller", saveSimulationResult=True)
    self.add_connection(east_temperature_heating_setpoint, east_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(east_temperature_cooling_setpoint, east_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(self.components["east"], east_temperature_heating_controller, "indoorTemperature", "roomTemp")
    self.add_connection(east_temperature_heating_controller, self.components["east"], "supplyAirTemp", "supplyAirTemperature")


    #Add east dampers
    east_supply_damper = tb.DamperSystem(id="east_supply_damper", saveSimulationResult=True)
    east_exhaust_damper = tb.DamperSystem(id="east_exhaust_damper", saveSimulationResult=True)
    self.add_connection(east_temperature_heating_controller, east_supply_damper, "y_dam", "damperPosition")
    self.add_connection(east_temperature_heating_controller, east_exhaust_damper, "y_dam", "damperPosition")
    self.add_connection(east_supply_damper, self.components["east"], "airFlowRate", "airFlowRate")
    #Add east heating coil
    east_heating_coil = tb.CoilHeatingSystem(id="east_heating_coil", saveSimulationResult=True)
    self.add_connection(east_temperature_heating_controller, east_heating_coil, "supplyAirTemp", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], east_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(east_supply_damper, east_heating_coil, "airFlowRate", "airFlowRate")

    #Add east sensors
    east_co2_sensor = tb.SensorSystem(id="east_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["east"], east_co2_sensor, "indoorCo2Concentration", "east_indoorCo2Concentration")
    east_supply_air_temp_sensor = tb.SensorSystem(id="east_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(east_heating_coil, east_supply_air_temp_sensor, "outletAirTemperature", "east_supplyAirTemperature")
    east_supply_airflow_sensor = tb.SensorSystem(id="east_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(east_supply_damper, east_supply_airflow_sensor, "airFlowRate", "east_supplyAirflow")
    east_supply_damper_position = tb.SensorSystem(id="east_supply_damper_position", saveSimulationResult=True)
    self.add_connection(east_supply_damper, east_supply_damper_position, "damperPosition", "east_supplyDamperPosition")

    #Add west temp setpoint profiles
    west_temperature_heating_setpoint = tb.ScheduleSystem(id="west_temperature_heating_setpoint", saveSimulationResult=True)
    west_temperature_cooling_setpoint = tb.ScheduleSystem(id="west_temperature_cooling_setpoint", saveSimulationResult=True)

    #Add west temp controller
    west_temperature_heating_controller = tb.VAVReheatControllerSystem(id="west_temperature_heating_controller", saveSimulationResult=True)
    self.add_connection(west_temperature_heating_setpoint, west_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(west_temperature_cooling_setpoint, west_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(self.components["west"], west_temperature_heating_controller, "indoorTemperature", "roomTemp")
    self.add_connection(west_temperature_heating_controller, self.components["west"], "supplyAirTemp", "supplyAirTemperature")

    #Add west dampers
    west_supply_damper = tb.DamperSystem(id="west_supply_damper", saveSimulationResult=True)
    west_exhaust_damper = tb.DamperSystem(id="west_exhaust_damper", saveSimulationResult=True)
    self.add_connection(west_temperature_heating_controller, west_supply_damper, "y_dam", "damperPosition")
    self.add_connection(west_temperature_heating_controller, west_exhaust_damper, "y_dam", "damperPosition")
    self.add_connection(west_supply_damper, self.components["west"], "airFlowRate", "airFlowRate")
    #Add west heating coil
    west_heating_coil = tb.CoilHeatingSystem(id="west_heating_coil", saveSimulationResult=True)
    self.add_connection(west_temperature_heating_controller, west_heating_coil, "supplyAirTemp", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], west_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(west_supply_damper, west_heating_coil, "airFlowRate", "airFlowRate")

    #Add west sensors
    west_co2_sensor = tb.SensorSystem(id="west_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["west"], west_co2_sensor, "indoorCo2Concentration", "west_indoorCo2Concentration")
    west_supply_air_temp_sensor = tb.SensorSystem(id="west_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(west_heating_coil, west_supply_air_temp_sensor, "outletAirTemperature", "west_supplyAirTemperature")
    west_supply_airflow_sensor = tb.SensorSystem(id="west_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(west_supply_damper, west_supply_airflow_sensor, "airFlowRate", "west_supplyAirflow")
    west_supply_damper_position = tb.SensorSystem(id="west_supply_damper_position", saveSimulationResult=True)
    self.add_connection(west_supply_damper, west_supply_damper_position, "damperPosition", "west_supplyDamperPosition")
    

    #Add supply junction
    supply_junction = tb.SupplyFlowJunctionSystem(id="supply_junction", saveSimulationResult=True)
    self.add_connection(core_supply_damper, supply_junction, "airFlowRate", "airFlowRateOut")
    self.add_connection(north_supply_damper, supply_junction, "airFlowRate", "airFlowRateOut")
    self.add_connection(south_supply_damper, supply_junction, "airFlowRate", "airFlowRateOut")
    self.add_connection(east_supply_damper, supply_junction, "airFlowRate", "airFlowRateOut")
    self.add_connection(west_supply_damper, supply_junction, "airFlowRate", "airFlowRateOut")
    
    vent_supply_airflow_sensor = tb.SensorSystem(id="vent_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(supply_junction, vent_supply_airflow_sensor, "airFlowRateIn", "supplyAirflow")

    #Connect return flow junction
    return_junction = self.components["exhaust_flow_junction"]
    self.add_connection(core_exhaust_damper, return_junction, "airFlowRate", "airFlowRateIn")
    self.add_connection(north_exhaust_damper, return_junction, "airFlowRate", "airFlowRateIn")
    self.add_connection(south_exhaust_damper, return_junction, "airFlowRate", "airFlowRateIn")
    self.add_connection(east_exhaust_damper, return_junction, "airFlowRate", "airFlowRateIn")
    self.add_connection(west_exhaust_damper, return_junction, "airFlowRate", "airFlowRateIn")

    vent_return_airflow_sensor = tb.SensorSystem(id="vent_return_airflow_sensor", saveSimulationResult=True)
    self.add_connection(return_junction, vent_return_airflow_sensor, "airFlowRateOut", "returnAirflow")
    vent_return_air_temp_sensor = tb.SensorSystem(id="vent_return_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(return_junction, vent_return_air_temp_sensor, "airTemperatureOut", "returnAirTemperature")

def get_model(id=None, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn
    model = tb.Model(id="five_rooms_only_template", saveSimulationResult=True)
    
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), r"semantic_models\five_rooms_no_contr.xlsm")
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

def print_parameter_results(model):
    """Print the resulting parameters for all rooms in a more organized way"""
    
    rooms = ['core', 'north', 'south', 'east', 'west']
    
    # Space model parameters
    print("SPACE MODEL PARAMETERS:")
    space_params = {
        'core': ['C_air', 'C_int', 'C_boundary', 'R_int', 'R_boundary', 'Q_occ_gain'],
        'other': ['C_wall', 'C_air', 'C_int', 'C_boundary', 'R_out', 'R_in', 'R_int', 
                 'R_boundary', 'f_wall', 'f_air', 'Q_occ_gain']
    }
    
    for room in rooms:
        print(f"\n{room.upper()}:")
        params = space_params['core'] if room == 'core' else space_params['other']
        for param in params:
            value = getattr(model.components[room], param)
            print(f"{param}: {value}")
    
    # Controller parameters
    print("\nCONTROLLERS:")
    controller_params = ['k_coo', 'ti_coo', 'k_hea', 'ti_hea']
    
    for room in rooms:
        print(f"\n{room.upper()}:")
        controller_id = f"{room}_temperature_heating_controller"
        for param in controller_params:
            value = getattr(model.components[controller_id], param)
            print(f"{param}: {value}")
    
    # Damper parameters
    print("\nDAMPERS:")
    damper_params = ['a', "nominalAirFlowRate"]
    
    for room in rooms:
        print(f"\n{room.upper()}:")
        damper_id = f"{room}_supply_damper"
        for param in damper_params:
            value = getattr(model.components[damper_id], param)
            if param == "nominalAirFlowRate":
                print(f"{param}: {value.hasValue}")
            else:
                print(f"{param}: {value}")


def parameter_estimation():
    """
    Checklist for the parameter estimation:
    [x] Gather and add the outdoor environment data
    [x] Define the target parameters
    [x] Define the target measuring devices
    [x] Define the list of required data points from the BOPTEST model
    [x] Acquire and pre process the data from the BOPTEST model
    [x] Fill in the data paths in the model_parameters/data_paths.json file
    [x] Update the component files with known parameters in the model_parameters folder
    [x] Transform the temperatures to celsius in the data
    [] Run the parameter estimation
    [] Load the estimation result
    [] Plot the results
    """
    stepSize = 600  # Seconds can go down to 30
    # Then set the startTime and endTime to a valid range
    startTime = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))

    model = get_model()

    ## Target parameters definition
    #CORE
    core_space = model.components["core"]
    core_supply_damper = model.components["core_supply_damper"]
    core_exhaust_damper = model.components["core_exhaust_damper"]
    core_temp_controller = model.components["core_temperature_heating_controller"]
    #NORTH
    north_space = model.components["north"]
    north_supply_damper = model.components["north_supply_damper"]
    north_exhaust_damper = model.components["north_exhaust_damper"]
    north_temp_controller = model.components["north_temperature_heating_controller"]
    #SOUTH
    south_space = model.components["south"]
    south_supply_damper = model.components["south_supply_damper"]
    south_exhaust_damper = model.components["south_exhaust_damper"]
    south_temp_controller = model.components["south_temperature_heating_controller"]
    #EAST
    east_space = model.components["east"]
    east_supply_damper = model.components["east_supply_damper"]
    east_exhaust_damper = model.components["east_exhaust_damper"]
    east_temp_controller = model.components["east_temperature_heating_controller"]
    #WEST
    west_space = model.components["west"]
    west_supply_damper = model.components["west_supply_damper"]
    west_exhaust_damper = model.components["west_exhaust_damper"]
    west_temp_controller = model.components["west_temperature_heating_controller"]

    dampers_list = [core_supply_damper, core_exhaust_damper, north_supply_damper, north_exhaust_damper, south_supply_damper, south_exhaust_damper, east_supply_damper, east_exhaust_damper, west_supply_damper, west_exhaust_damper]

    targetParameters = {"private": {
                                    "C_air": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 1e+5, "lb": 1e+4, "ub": 1e+7},                                    
                                    "C_boundary": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                    "R_boundary": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 0.1, "lb": 1e-2, "ub": 0.2},
                                    "Q_occ_gain": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 100, "lb": 0, "ub": 200},

                                    "C_wall": {"components": [north_space, south_space, east_space, west_space], "x0": 1e+5, "lb": 1e+4, "ub": 1e+7},
                                    "R_out": {"components": [north_space, south_space, east_space, west_space], "x0": 0.1, "lb": 1e-2, "ub": 0.2},
                                    "R_in": {"components": [north_space, south_space, east_space, west_space], "x0": 0.1, "lb": 1e-2, "ub": 0.2},
                                    "f_wall": {"components": [north_space, south_space, east_space, west_space], "x0": 1, "lb": 0, "ub": 3},
                                    "f_air": {"components": [north_space, south_space, east_space, west_space], "x0": 1, "lb": 0, "ub": 3},
                                    "k_coo": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 2e-4, "lb": 1e-5, "ub": 3},
                                    "ti_coo": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 3e-1, "lb": 1e-5, "ub": 3},
                                    "k_hea": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 2e-4, "lb": 1e-5, "ub": 3},
                                    "ti_hea": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 3e-1, "lb": 1e-5, "ub": 3},
                                    "nominalAirFlowRate.hasValue": {"components": dampers_list, "x0": 1.6, "lb": 1e-2, "ub": 5}, #0.0202
                                    },
                        "shared": {"C_int": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                    "R_int": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 0.1, "lb": 1e-2, "ub": 0.2},
                                    "T_boundary": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 20, "lb": 19, "ub": 23},
                                    "a": {"components": dampers_list, "x0": 5, "lb": 0.5, "ub": 8},
                                    "infiltration": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 0.001, "lb": 1e-4, "ub": 0.01},
                                    "CO2_occ_gain": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 0.001, "lb": 1e-4, "ub": 0.01},
                            }}
    
    """
    Parameters for each room:
    - Input & output damper:
        - a (shared) 
        - nominalAirFlowRate
    - PI controller Kp, Ki constants
    - Space model parameters:

      BuildingSpaceNoSH1AdjBoundaryOutdoorFMUSystem (north, south, east, west)

        C_supply 400
        C_wall - To be estimated
        C_air  - To be estimated
        C_int  - To be estimated (shared)
        C_boundary  - To be estimated
        R_out  - To be estimated
        R_in  - To be estimated
        R_int  - To be estimated (shared)
        R_boundary  - To be estimated
        f_wall  - To be estimated
        f_air  - To be estimated 
        Q_occ_gain - To be estimated (default 0)
        CO2_occ_gain  - To be estimated 8.18e-6?
        CO2_start (400 ppm)
        T_boundary  - To be estimated (shared)
        infiltration - To be estimated (shared)
        airVolume (Known from geometry)

        BuildingSpaceNoSH1AdjBoundaryFMUSystem (core)

        C_supply 400
        C_air  - To be estimated
        C_int - To be estimated (shared)    
        C_boundary - To be estimated
        R_int - To be estimated (shared)
        R_boundary  - To be estimated
        Q_occ_gain  - To be estimated (default 0 ?)
        CO2_occ_gain  - To be estimated (shared) (Where does it come from?) 8.18e-6 ?
        CO2_start (400 ppm)
        T_boundary - To be estimated (shared)
        infiltration - To be estimated (shared ?)
        airVolume (Known from geometry)
        
        Where the shared parameters to reduce estimation effort are:
        - C_int
        - R_int
        - T_boundary
        - a (For the dampers)


    Required data points:
    Common data points:
    [x]Supply air temperature (hvac_reaAhu_TSup_y)
    [x]Supply air flow rate (hvac_reaAhu_V_flow_sup_y) from supply junction
    [x]Return air flow rate (hvac_reaAhu_V_flow_ret_y) from return junction
    [x]Outdoor air temperature (weaSta_reaWeaTWetBul_y) Figure how to synchronize with the outdoor environment data
    Per room data points:
    [x]Supply air temperature (hvac_reaZonCor_TSup_y, hvac_reaZonNor_TSup_y, hvac_reaZonSou_TSup_y, hvac_reaZonEas_TSup_y, hvac_reaZonWes_TSup_y)
    [x]Supply air flow rate (hvac_reaZonCor_V_flow_y, hvac_reaZonNor_V_flow_y, hvac_reaZonSou_V_flow_y, hvac_reaZonEas_V_flow_y, hvac_reaZonWes_V_flow_y)
    [x]CO2 concentration (hvac_reaZonCor_CO2Zon_y, hvac_reaZonNor_CO2Zon_y, hvac_reaZonSou_CO2Zon_y, hvac_reaZonEas_CO2Zon_y, hvac_reaZonWes_CO2Zon_y)
    [x]Indoor air temperature (hvac_reaZonCor_TZon_y, hvac_reaZonNor_TZon_y, hvac_reaZonSou_TZon_y, hvac_reaZonEas_TZon_y, hvac_reaZonWes_TZon_y)
    [x]Supply damper position (hvac_oveZonActCor_yDam_u, hvac_oveZonActNor_yDam_u, hvac_oveZonActSou_yDam_u, hvac_oveZonActEas_yDam_u, hvac_oveZonActWes_yDam_u)
    [x]Heating setpoint (hvac_oveZonSupCor_TZonHeaSet_u, hvac_oveZonSupNor_TZonHeaSet_u, hvac_oveZonSupSou_TZonHeaSet_u, hvac_oveZonSupEas_TZonHeaSet_u, hvac_oveZonSupWes_TZonHeaSet_u)
    [x]Cooling setpoint (hvac_oveZonSupCor_TZonCooSet_u, hvac_oveZonSupNor_TZonCooSet_u, hvac_oveZonSupSou_TZonCooSet_u, hvac_oveZonSupEas_TZonCooSet_u, hvac_oveZonSupWes_TZonCooSet_u)
    [x](Forecast values) Occupancy (Occupancy[cor], Occupancy[nor], Occupancy[sou], Occupancy[eas], Occupancy[wes]) 

    Model outputs (measuring devices):
    - Indoor air temperature (core_indoor_air_temp_sensor, north_indoor_air_temp_sensor, south_indoor_air_temp_sensor, east_indoor_air_temp_sensor, west_indoor_air_temp_sensor)
    - Damper position (core_supply_damper_position, north_supply_damper_position, south_supply_damper_position, east_supply_damper_position, west_supply_damper_position)
    - Room supply air flow rate (core_supply_airflow_sensor, north_supply_airflow_sensor, south_supply_airflow_sensor, east_supply_airflow_sensor, west_supply_airflow_sensor)
    - CO2 concentration (core_co2_sensor, north_co2_sensor, south_co2_sensor, east_co2_sensor, west_co2_sensor)
    - Total supply air flow rate (vent_supply_airflow_sensor)

    """


    percentile = 2
    targetMeasuringDevices = {
                             model.components["vent_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["vent_return_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["vent_return_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["core_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["core_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1.5},
                             model.components["core_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["core_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             model.components["core_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["north_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["north_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1.5},
                             model.components["north_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["north_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1}, 
                             model.components["north_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["south_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["south_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1.5},
                             model.components["south_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["south_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             model.components["south_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["east_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["east_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1.5},
                             model.components["east_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["east_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             model.components["east_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["west_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["west_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1.5},
                             model.components["west_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["west_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             model.components["west_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             }  

    
    options = {
            "n_cores": 4,
            "ftol": 1e-10,
            "xtol": 1e-10,
            "gtol": 1e-10,
            "verbose": 2}
    estimator = tb.Estimator(model)
    estimator.estimate(targetParameters=targetParameters,
                        targetMeasuringDevices=targetMeasuringDevices,
                        startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize,
                        method="LS", #Use Least Squares instead
                        options=options,
                        verbose=True
                        )
    model.load_estimation_result(estimator.result_savedir_pickle)

    #Print the resulting parameters

    print("Resulting parameters:")
    print_parameter_results(model)

def load_and_print_parameters(filename):
    model = get_model()
    model.load_estimation_result(filename)

    #Print the resulting parameters

    print("Resulting parameters:")
    print_parameter_results(model)

if __name__ == "__main__":
    #parameter_estimation()
    #run()
    load_and_print_parameters(r"C:\Users\asces\OneDriveUni\Projects\RL_control\boptest_model\generated_files\models\five_rooms_only_template\model_parameters\estimation_results\LS_result\20250327_181329_ls.pickle")