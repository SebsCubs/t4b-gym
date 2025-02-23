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


def do_step(self, secondTime=None, dateTime=None, stepSize=None):
    input_signal = self.input["actualValue"].get()[0]
    #heating valve position is 0-1, if the input signal is positiv, the valve is open with the same value but clamped at 1
    if input_signal > 0:
        self.output["heatingValvePosition"].set(min(input_signal, 1))
    else:
        self.output["heatingValvePosition"].set(0)



def fcn(self):
    '''
        The fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''

    vent_supply_air_temp_sensor = tb.SensorSystem(id="vent_supply_air_temp_sensor", saveSimulationResult=True)
    vent_airflow_sensor = tb.SensorSystem(id="vent_airflow_sensor", saveSimulationResult=True)
    vent_outdoor_air_temp_sensor = tb.SensorSystem(id="vent_outdoor_air_temp_sensor", saveSimulationResult=True)
    supply_heating_coil_water_temp_sensor = tb.SensorSystem(id="supply_heating_coil_water_temp_sensor", saveSimulationResult=True)
    return_heating_coil_water_temp_sensor = tb.SensorSystem(id="return_heating_coil_water_temp_sensor", saveSimulationResult=True)
    vent_heated_air_temp_sensor = tb.SensorSystem(id="vent_heated_air_temp_sensor", saveSimulationResult=True)



    # Add AHU heating coil
    supply_heating_coil = tb.CoilPumpValveFMUSystem(id="[supply_heating_coil][heating_pump][heating_valve]", saveSimulationResult=True)
    self.add_connection(vent_outdoor_air_temp_sensor, supply_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(vent_airflow_sensor, supply_heating_coil, "airFlowRateIn", "airFlowRate")
    self.add_connection(supply_heating_coil, vent_heated_air_temp_sensor, "outletAirTemperature", "heatedAirTemperature")
    self.add_connection(supply_heating_coil_water_temp_sensor, supply_heating_coil, "supplyWaterTemperature", "supplyWaterTemperature")
    self.add_connection(supply_heating_coil, return_heating_coil_water_temp_sensor, "outletWaterTemperature", "inletWaterTemperature")

    supply_air_temp_setpoint = tb.ScheduleSystem(id="supply_air_temp_setpoint", saveSimulationResult=True)
    supply_air_temp_controller = tb.PIControllerFMUSystem(kp=1, Ti=1,
        id="supply_air_temp_controller", isReverse=False, saveSimulationResult=True)

    self.add_connection(supply_air_temp_setpoint, supply_air_temp_controller, "scheduleValue", "setpointValue")
    self.add_connection(vent_supply_air_temp_sensor, supply_air_temp_controller, "supplyAirTemperature", "actualValue")
     
    c3_control_map = tb.ControlSignalMapSystem(id="c3_control_map", saveSimulationResult=True)
    self.add_connection(supply_air_temp_controller, c3_control_map, "inputSignal", "actualValue")
    self.add_connection(c3_control_map, supply_heating_coil, "heatingValvePosition", "valvePosition")
    
    # Replace the do_step method with your custom function
    c3_control_map.do_step = do_step.__get__(c3_control_map, tb.ControlSignalMapSystem)

    supply_heating_coil.m1_flow_nominal = 2.9
    supply_heating_coil.m2_flow_nominal = 6.713
    supply_heating_coil.tau1 = 18.41226242542397
    supply_heating_coil.tau2 = 10.64284456709903
    supply_heating_coil.tau_m = 13.976768318598277
    supply_heating_coil.nominalUa.hasValue = 2568.0665571292134
    supply_heating_coil.mFlowValve_nominal = 2.9
    supply_heating_coil.flowCoefficient.hasValue = 1
    supply_heating_coil.mFlowPump_nominal = 2.9
    supply_heating_coil.KvCheckValve = 1
    supply_heating_coil.dp1_nominal = 20000
    supply_heating_coil.dpPump = 45000
    supply_heating_coil.dpSystem = 10000
    supply_heating_coil.dpFixedSystem = 10000
    supply_heating_coil.tau_w_inlet = 1
    supply_heating_coil.tau_w_outlet = 1
    supply_heating_coil.tau_air_outlet = 1


def get_model(id=None, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn
    model = tb.Model(id="only_vent_coil_model", saveSimulationResult=True)
    model.load(fcn=fcn_, create_signature_graphs=False, validate_model=True, verbose=True, force_config_update=True)
    if id is not None:
        model.id = id
    return model

def estimate_parameters(verbose=False):
    stepSize = 600  # Seconds can go down to 30

    startTime = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=2, day=12, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = get_model()

    supply_fan = model.components["supply_fan"]
    supply_heating_coil = model.components["[supply_heating_coil][heating_pump][heating_valve]"]
    supply_cooling_coil = model.components["[supply_cooling_coil][cooling_pump][cooling_valve]"]
    pi_controller = model.components["supply_air_temp_controller"]

    
    targetParameters = {"private": {"c1": {"components": [supply_fan], "x0": 0, "lb": -10, "ub": 10}, 
                                    "c2": {"components": [supply_fan], "x0": 1, "lb": -10, "ub": 10}, 
                                    "c3": {"components": [supply_fan], "x0": 3, "lb": -10, "ub": 10}, 
                                    "c4": {"components": [supply_fan], "x0": 0, "lb": -10, "ub": 10}, 
                                    "nominalPowerRate.hasValue": {"components": [supply_fan], "x0": 500, "lb": 100, "ub": 10000}, 
                                    "tau1": {"components": [supply_heating_coil, supply_cooling_coil], "x0": 18.41226242542397, "lb": 1, "ub": 50}, 
                                    "tau2": {"components": [supply_heating_coil, supply_cooling_coil], "x0": 10.64284456709903, "lb": 1, "ub": 50}, 
                                    "tau_m": {"components": [supply_heating_coil, supply_cooling_coil], "x0": 13.976768318598277, "lb": 1, "ub": 50}, 
                                    "nominalUa.hasValue": {"components": [supply_heating_coil, supply_cooling_coil], "x0": 2568.0665571292134, "lb": 0, "ub": 10000}, 
                                    "flowCoefficient.hasValue": {"components": [supply_heating_coil, supply_cooling_coil], "x0": 1, "lb": 0, "ub": 10}, 
                                    "KvCheckValve": {"components": [supply_heating_coil, supply_cooling_coil], "x0": 1, "lb": 0, "ub": 100}, 
                                    "dp1_nominal": {"components": [supply_heating_coil, supply_cooling_coil], "x0": 100, "lb": 0, "ub": 100000}, 
                                    "dpSystem": {"components": [supply_heating_coil, supply_cooling_coil], "x0": 100, "lb": 0, "ub": 100000}, 
                                    "dpFixedSystem": {"components": [supply_heating_coil, supply_cooling_coil], "x0": 100, "lb": 0, "ub": 100000}, 
                                    "kp": {"components": [pi_controller], "x0": 1, "lb": -10, "ub": 10}, 
                                    "Ti": {"components": [pi_controller], "x0": 1, "lb": -10, "ub": 10}, 
                                    }}
    
    percentile = 2

    targetMeasuringDevices = {model.components["vent_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["vent_power_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 500},
                             model.components["return_heating_coil_water_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["return_cooling_coil_water_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["vent_supply_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             model.components["vent_return_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             }

    
    options = {
            "n_cores": 2,
            "ftol": 1e-12,
            "xtol": 1e-12,
            "gtol": 1e-15,
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
        
        # Print heating coil parameters
        print("\nHeating Coil Parameters:")
        print(f"tau1: {supply_heating_coil.tau1}")
        print(f"tau2: {supply_heating_coil.tau2}")
        print(f"tau_m: {supply_heating_coil.tau_m}")
        print(f"nominalUa: {supply_heating_coil.nominalUa.hasValue}")
        print(f"flowCoefficient: {supply_heating_coil.flowCoefficient.hasValue}")
        print(f"KvCheckValve: {supply_heating_coil.KvCheckValve}")
        print(f"dp1_nominal: {supply_heating_coil.dp1_nominal}")
        print(f"dpSystem: {supply_heating_coil.dpSystem}")
        print(f"dpFixedSystem: {supply_heating_coil.dpFixedSystem}")
        
        # Print cooling coil parameters
        print("\nCooling Coil Parameters:")
        print(f"tau1: {supply_cooling_coil.tau1}")
        print(f"tau2: {supply_cooling_coil.tau2}")
        print(f"tau_m: {supply_cooling_coil.tau_m}")
        print(f"nominalUa: {supply_cooling_coil.nominalUa.hasValue}")
        print(f"flowCoefficient: {supply_cooling_coil.flowCoefficient.hasValue}")
        print(f"KvCheckValve: {supply_cooling_coil.KvCheckValve}")
        print(f"dp1_nominal: {supply_cooling_coil.dp1_nominal}")
        print(f"dpSystem: {supply_cooling_coil.dpSystem}")
        print(f"dpFixedSystem: {supply_cooling_coil.dpFixedSystem}")
        
        # Print PI controller parameters
        print("\nPI Controller Parameters:")
        print(f"kp: {pi_controller.kp}")
        print(f"Ti: {pi_controller.Ti}")



def run():
    stepSize = 600  # Seconds can go down to 30

    startTime = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=2, day=12, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = get_model()

    simulator = tb.Simulator()

    simulator.simulate(model = model,
                       startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize)
    
    print("Simulation completed successfully!")

    temp_sensor = "vent_heated_air_temp_sensor"
    setpoint = "supply_air_temp_setpoint"
    # Temperature plot for room {room_id}
    fig, axes = plot.plot_component(
        simulator,
        components_1axis=[
            (temp_sensor, 'heatedAirTemperature'),
            (setpoint, 'scheduleValue'),
        ],
        ylabel_1axis='Duct Temperature [°C]',
        show=False  
    )
    lines = axes[0].get_lines()
    axes[0].legend(lines, [
        'Actual Temperature',
        'Original Setpoint'
    ])
    plt.title(f'Duct Temperature')
    plt.show()




if __name__ == "__main__":
    run()