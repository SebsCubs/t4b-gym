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
 
    #TODO: Evaluate if the room temperature control needs to be modified to be more compliant with the BOPTEST model

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
    core_supply_damper_position = tb.SensorSystem(id="core_supply_damper_position", saveSimulationResult=True)
    self.add_connection(self.components["core_supply_damper"], core_supply_damper_position, "damperPosition", "core_supplyDamperPosition")
    

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
    north_supply_damper_position = tb.SensorSystem(id="north_supply_damper_position", saveSimulationResult=True)
    self.add_connection(self.components["north_supply_damper"], north_supply_damper_position, "damperPosition", "north_supplyDamperPosition")

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
    south_supply_damper_position = tb.SensorSystem(id="south_supply_damper_position", saveSimulationResult=True)
    self.add_connection(self.components["south_supply_damper"], south_supply_damper_position, "damperPosition", "south_supplyDamperPosition")

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
    east_supply_damper_position = tb.SensorSystem(id="east_supply_damper_position", saveSimulationResult=True)
    self.add_connection(self.components["east_supply_damper"], east_supply_damper_position, "damperPosition", "east_supplyDamperPosition")

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
    west_supply_damper_position = tb.SensorSystem(id="west_supply_damper_position", saveSimulationResult=True)
    self.add_connection(self.components["west_supply_damper"], west_supply_damper_position, "damperPosition", "west_supplyDamperPosition")

    

def get_model(id=None, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn
    model = tb.Model(id="five_rooms_only_template", saveSimulationResult=True)
    
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), r"semantic_models\five_rooms_only_template.xlsm")
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

def parameter_estimation(verbose=False):
    """
    Checklist for the parameter estimation:
    - Pre process the data to be in the same time period as the available outdoor environment data
    - Define the target parameters
    - Define the target measuring devices
    - Define the list of required data points from the BOPTEST model
    - Acquire and pre process the data from the BOPTEST model
    - Run the parameter estimation
    - Load the estimation result
    - Plot the results
    """
    stepSize = 600  # Seconds can go down to 30
    #TODO: Pre process the data to be in the same time period as the available outdoor environment data
    # Then set the startTime and endTime to a valid range
    startTime = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))

    model = get_model(id="five_rooms_model")

    ## Target parameters definition
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

    """
    #CORE
    core_space = model.components["core"]
    core_supply_damper = core_space.components["core_supply_damper"]
    core_exhaust_damper = core_space.components["core_exhaust_damper"]
    core_temp_controller = core_space.components["core_temperature_heating_controller"]
    #NORTH
    north_space = model.components["north"]
    north_supply_damper = north_space.components["north_supply_damper"]
    north_exhaust_damper = north_space.components["north_exhaust_damper"]
    north_temp_controller = north_space.components["north_temperature_heating_controller"]
    #SOUTH
    south_space = model.components["south"]
    south_supply_damper = south_space.components["south_supply_damper"]
    south_exhaust_damper = south_space.components["south_exhaust_damper"]
    south_temp_controller = south_space.components["south_temperature_heating_controller"]
    #EAST
    east_space = model.components["east"]
    east_supply_damper = east_space.components["east_supply_damper"]
    east_exhaust_damper = east_space.components["east_exhaust_damper"]
    east_temp_controller = east_space.components["east_temperature_heating_controller"]
    #WEST
    west_space = model.components["west"]
    west_supply_damper = west_space.components["west_supply_damper"]
    west_exhaust_damper = west_space.components["west_exhaust_damper"]
    west_temp_controller = west_space.components["west_temperature_heating_controller"]

    dampers_list = [core_supply_damper, core_exhaust_damper, north_supply_damper, north_exhaust_damper, south_supply_damper, south_exhaust_damper, east_supply_damper, east_exhaust_damper, west_supply_damper, west_exhaust_damper]

    targetParameters = {"private": {
                                    "C_air": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 1e+5, "lb": 1e+4, "ub": 1e+7},                                    
                                    "C_boundary": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                    "R_boundary": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 0.1, "lb": 1e-2, "ub": 0.2},
                                    "Q_occ_gain": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 100, "lb": 0, "ub": 200},

                                    "C_wall": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 1e+5, "lb": 1e+4, "ub": 1e+7},
                                    "R_out": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 0.1, "lb": 1e-2, "ub": 0.2},
                                    "R_in": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 0.1, "lb": 1e-2, "ub": 0.2},
                                    "f_wall": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 1, "lb": 0, "ub": 3},
                                    "f_air": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 1, "lb": 0, "ub": 3},
                                    "Kp": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 2e-4, "lb": 1e-5, "ub": 3},
                                    "Ti": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 3e-1, "lb": 1e-5, "ub": 3},
                                    "m_flow_nominal": {"components": dampers_list, "x0": 1.6, "lb": 1e-2, "ub": 5}, #0.0202
                                    },
                        "shared": {"C_int": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                    "R_int": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 0.1, "lb": 1e-2, "ub": 0.2},
                                    "T_boundary": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 20, "lb": 19, "ub": 23},
                                    "a": {"components": dampers_list, "x0": 5, "lb": 0.5, "ub": 8},
                                    "infiltration": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 0.001, "lb": 1e-4, "ub": 0.01},
                                    "CO2_occ_gain": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 0.001, "lb": 1e-4, "ub": 0.01},
                            }}
    

    """
    Required data points:
    Common data points:
    [x]Supply air temperature (hvac_reaAhu_TSup_y)
    [x]Supply air flow rate (hvac_reaAhu_V_flow_sup_y) from supply junction
    [x]Return air flow rate (hvac_reaAhu_V_flow_ret_y) from return junction
    [x]Outdoor air temperature (weaSta_reaWeaTWetBul_y) Figure how to synchronize with the outdoor environment data
    Per room data points:
    []Supply air temperature (core_supply_air_temp_sensor, north_supply_air_temp_sensor, south_supply_air_temp_sensor, east_supply_air_temp_sensor, west_supply_air_temp_sensor)
    []Supply air flow rate (core_supply_airflow_sensor, north_supply_airflow_sensor, south_supply_airflow_sensor, east_supply_airflow_sensor, west_supply_airflow_sensor)
    []CO2 concentration (core_co2_sensor, north_co2_sensor, south_co2_sensor, east_co2_sensor, west_co2_sensor)
    []Indoor air temperature (core_indoor_air_temp_sensor, north_indoor_air_temp_sensor, south_indoor_air_temp_sensor, east_indoor_air_temp_sensor, west_indoor_air_temp_sensor)
    []Heating setpoint (core_temperature_heating_setpoint, north_temperature_heating_setpoint, south_temperature_heating_setpoint, east_temperature_heating_setpoint, west_temperature_heating_setpoint)
    []Cooling setpoint (core_temperature_cooling_setpoint, north_temperature_cooling_setpoint, south_temperature_cooling_setpoint, east_temperature_cooling_setpoint, west_temperature_cooling_setpoint)
    []Occupancy (core_occupancy_profile, north_occupancy_profile, south_occupancy_profile, east_occupancy_profile, west_occupancy_profile)

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

                             model.components["core_indoor_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["core_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1.5},
                             model.components["core_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["core_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},

                             model.components["north_indoor_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["north_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1.5},
                             model.components["north_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["north_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1}, 

                             model.components["south_indoor_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["south_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1.5},
                             model.components["south_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["south_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},

                             model.components["east_indoor_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["east_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1.5},
                             model.components["east_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["east_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},

                             model.components["west_indoor_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["west_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1.5},
                             model.components["west_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["west_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
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

    if verbose:
        # Print fan parameters
        print("\nSupply Fan Parameters:")
        print(f"c1: {supply_fan.c1}")
        print(f"c2: {supply_fan.c2}")
        print(f"c3: {supply_fan.c3}")
        print(f"c4: {supply_fan.c4}")
        print(f"nominalPowerRate: {supply_fan.nominalPowerRate.hasValue}")

        # Print damper parameters
        print("\nMain Supply Damper Parameters:")
        print(f"a: {main_supply_damper.a}")
        print(f"nominalAirFlowRate: {main_supply_damper.nominalAirFlowRate.hasValue}")

        print("\nMain Return Damper Parameters:")
        print(f"a: {main_return_damper.a}")
        print(f"nominalAirFlowRate: {main_return_damper.nominalAirFlowRate.hasValue}")

        print("\nMain Mixing Damper Parameters:")
        print(f"a: {main_mixing_damper.a}")
        print(f"nominalAirFlowRate: {main_mixing_damper.nominalAirFlowRate.hasValue}")

        # Print coil parameters


    return model
    


if __name__ == "__main__":
    run()