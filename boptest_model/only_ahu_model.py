import os
import datetime
from dateutil.tz import gettz
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' 
# Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = os.path.join(uppath(os.path.abspath(__file__), 4), "Twin4Build")
    sys.path.append(file_path)

import twin4build as tb
import twin4build.utils.plot.plot as plot
from twin4build.utils.uppath import uppath
import matplotlib.pyplot as plt


def fcn(self):
    '''
        The fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''


    vent_supply_air_temp_sensor = tb.SensorSystem(id="vent_supply_air_temp_sensor", saveSimulationResult=True)
    vent_mixed_air_temp_sensor = tb.SensorSystem(id="vent_mixed_air_temp_sensor", saveSimulationResult=True)
    vent_airflow_sensor = tb.SensorSystem(id="vent_airflow_sensor", saveSimulationResult=True)
    vent_supply_damper_setpoint = tb.ScheduleSystem(id="vent_supply_damper_setpoint", saveSimulationResult=True)
    vent_return_damper_setpoint = tb.ScheduleSystem(id="vent_return_damper_setpoint", saveSimulationResult=True)
    vent_mixing_damper_setpoint = tb.ScheduleSystem(id="vent_mixing_damper_setpoint", saveSimulationResult=True)
    vent_return_air_temp_sensor = tb.SensorSystem(id="vent_return_air_temp_sensor", saveSimulationResult=True)
    vent_return_airflow_sensor = tb.SensorSystem(id="vent_return_airflow_sensor", saveSimulationResult=True)
    vent_outdoor_air_temp_sensor = tb.SensorSystem(id="vent_outdoor_air_temp_sensor", saveSimulationResult=True)
    vent_power_sensor = tb.SensorSystem(id="vent_power_sensor", saveSimulationResult=True)

    heating_coil_temperature_setpoint = tb.ScheduleSystem(id="heating_coil_temperature_setpoint", saveSimulationResult=True)
    cooling_coil_temperature_setpoint = tb.ScheduleSystem(id="cooling_coil_temperature_setpoint", saveSimulationResult=True)

    # Add AHU fan
    supply_fan = tb.FanSystem(id="supply_fan", saveSimulationResult=True)
    self.add_connection(vent_airflow_sensor, supply_fan, "airFlowRateIn", "airFlowRate")
    self.add_connection(supply_fan, vent_power_sensor, "Power", "power")

    # Add AHU heating coil
    supply_heating_coil = tb.CoilHeatingSystem(id="supply_heating_coil", saveSimulationResult=True)
    self.add_connection(vent_mixed_air_temp_sensor, supply_heating_coil, "mixedAirTemperature", "inletAirTemperature")
    self.add_connection(vent_airflow_sensor, supply_heating_coil, "airFlowRateIn", "airFlowRate")
    self.add_connection(heating_coil_temperature_setpoint, supply_heating_coil, "scheduleValue", "outletAirTemperatureSetpoint")

    # Add AHU cooling coil
    supply_cooling_coil = tb.CoilCoolingSystem(id="supply_cooling_coil", saveSimulationResult=True)
    self.add_connection(supply_heating_coil, supply_cooling_coil, "outletAirTemperature", "inletAirTemperature")
    self.add_connection(vent_airflow_sensor, supply_cooling_coil, "airFlowRateIn", "airFlowRate")
    self.add_connection(cooling_coil_temperature_setpoint, supply_cooling_coil, "scheduleValue", "outletAirTemperatureSetpoint")
    self.add_connection(supply_cooling_coil, vent_supply_air_temp_sensor, "outletAirTemperature", "supplyAirTemperature")

    # Add main dampers
    main_supply_damper = tb.DamperSystem(id="main_supply_damper", saveSimulationResult=True)
    self.add_connection(vent_supply_damper_setpoint, main_supply_damper, "scheduleValue", "damperPosition")
    main_return_damper = tb.DamperSystem(id="main_return_damper", saveSimulationResult=True)
    self.add_connection(vent_return_damper_setpoint, main_return_damper, "scheduleValue", "damperPosition")
    main_mixing_damper = tb.DamperSystem(id="main_mixing_damper", saveSimulationResult=True)
    self.add_connection(vent_mixing_damper_setpoint, main_mixing_damper, "scheduleValue", "damperPosition")

    # Add supply flow junction
    supply_flow_junction_for_return = tb.SupplyFlowJunctionSystem(id="supply_flow_junction_for_return", saveSimulationResult=True)
    self.add_connection(supply_flow_junction_for_return, vent_return_airflow_sensor, "airFlowRateIn", "returnAirFlowRate")
    self.add_connection(main_return_damper, supply_flow_junction_for_return, "airFlowRate", "airFlowRateOut")
    self.add_connection(main_mixing_damper, supply_flow_junction_for_return, "airFlowRate", "airFlowRateOut")

    # Add return flow junction
    return_flow_junction_for_supply = tb.ReturnFlowJunctionSystem(id="return_flow_junction", saveSimulationResult=True)
    self.add_connection(main_supply_damper, return_flow_junction_for_supply, "airFlowRate", "airFlowRateIn")
    self.add_connection(vent_outdoor_air_temp_sensor, return_flow_junction_for_supply, "airTemperatureIn", "airTemperatureIn")
    self.add_connection(main_mixing_damper, return_flow_junction_for_supply, "airFlowRate", "airFlowRateIn")
    self.add_connection(vent_return_air_temp_sensor, return_flow_junction_for_supply, "returnAirTemperature", "airTemperatureIn")
    self.add_connection(return_flow_junction_for_supply, vent_airflow_sensor, "airFlowRateOut", "airFlowRateIn")
    self.add_connection(return_flow_junction_for_supply, vent_mixed_air_temp_sensor, "airTemperatureOut", "mixedAirTemperature")

    


def get_model(id="default_model_id", fcn_=None):
    if fcn_ is None:
        fcn_ = fcn
    model = tb.Model(id=id, saveSimulationResult=True)
    model.load(fcn=fcn_, create_signature_graphs=False, validate_model=True, verbose=True, force_config_update=True)
    if id is not None:
        model.id = id
    return model

def estimate_parameters(verbose=False):
    stepSize = 600  # Seconds can go down to 30

    startTime = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = get_model(id="only_ahu_model")

    supply_fan = model.components["supply_fan"]
    main_supply_damper = model.components["main_supply_damper"]
    main_return_damper = model.components["main_return_damper"]
    main_mixing_damper = model.components["main_mixing_damper"]

    
    targetParameters = {"private": {"c1": {"components": [supply_fan], "x0": 0, "lb": -10, "ub": 10}, 
                                    "c2": {"components": [supply_fan], "x0": 1, "lb": -10, "ub": 10}, 
                                    "c3": {"components": [supply_fan], "x0": 3, "lb": -10, "ub": 10}, 
                                    "c4": {"components": [supply_fan], "x0": 0, "lb": -10, "ub": 10}, 
                                    "nominalPowerRate.hasValue": {"components": [supply_fan], "x0": 500, "lb": 100, "ub": 10000}, 
                                    "a": {"components": [main_supply_damper, main_return_damper, main_mixing_damper], "x0": 1, "lb": 0.0001, "ub": 5}, 
                                    "nominalAirFlowRate.hasValue": {"components": [main_supply_damper, main_return_damper, main_mixing_damper], "x0": 1, "lb": 0.0001, "ub": 5}, 
                                    }}
    
    percentile = 2

    targetMeasuringDevices = {
                             model.components["vent_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["vent_mixed_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             #model.components["vent_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             #model.components["vent_return_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             #model.components["vent_return_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             #model.components["vent_outdoor_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["vent_power_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 500},
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

def run():
    stepSize = 600  # Seconds can go down to 30

    startTime = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=2, day=12, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = get_model(id="only_ahu_model_simulation")
    
    model.load_estimation_result(r"C:\Users\asces\OneDriveUni\Projects\RL_control\boptest_model\generated_files\models\only_ahu_model\model_parameters\estimation_results\LS_result\20250314_153712_ls.pickle")

    simulator = tb.Simulator()

    simulator.simulate(model = model,
                       startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize)
    
    print("Simulation completed successfully!")

    temp_sensor = "vent_supply_air_temp_sensor"
    supply_cooling_coil = "supply_cooling_coil"
    setpoint = "cooling_coil_temperature_setpoint"
    # Temperature plot
    fig, axes = plot.plot_component(
        simulator,
        components_1axis=[
            (temp_sensor, 'supplyAirTemperature'),
            (setpoint, 'scheduleValue'),
            (supply_cooling_coil, 'outletAirTemperature'),
        ],
        ylabel_1axis='Duct Temperature [°C]',
        show=False  
    )
    lines = axes[0].get_lines()
    axes[0].legend(lines, [
        'Actual Temperature',
        'Original Setpoint',
        'Cooling Coil Outlet Temperature'
    ])
    plt.title(f'Duct Temperature')
    plt.show()

    #Fan power plot
    supply_fan = "supply_fan"
    original_power = "vent_power_sensor"
    #Fan power plot
    fig, axes = plot.plot_component(
        simulator,
        components_1axis=[
            (supply_fan, 'Power'),
            (original_power, 'power'),
        ],
        ylabel_1axis='Fan Power [W]',
        show=False
    )
    lines = axes[0].get_lines()
    axes[0].legend(lines, [
        'Estimated Power',
        'Original Power'
    ])
    plt.title(f'Fan Power')
    plt.show()

if __name__ == "__main__":
    estimate_parameters(verbose=True)
    #run()