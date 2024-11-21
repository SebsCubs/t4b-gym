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
    supply_water_temperature_schedule = tb.PiecewiseLinearScheduleSystem(
        weekDayRulesetDict = {
                "ruleset_default_value": {"X": [-12, 5, 20],
                                          "Y": [60, 50, 20]}},
            saveSimulationResult = True,
        id="supply_water_temperature_schedule")
    outdoor_environment = self.get_component_by_class(self.component_dict, tb.OutdoorEnvironmentSystem)[0]
    self.add_connection(outdoor_environment, supply_water_temperature_schedule, "outdoorTemperature", "outdoorTemperature")
    spaces = self.get_component_by_class(self.component_dict, tb.BuildingSpace1AdjBoundaryFMUSystem)
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace2AdjBoundaryFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace11AdjBoundaryFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace1AdjBoundaryOutdoorFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace2AdjBoundaryOutdoorFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace11AdjBoundaryOutdoorFMUSystem))
    for space in spaces:
        self.add_connection(supply_water_temperature_schedule, space, 
                            "scheduleValue", "supplyWaterTemperature")

def get_model(id=None, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn
    model = tb.Model(id="model_1st_floor", saveSimulationResult=True)
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), "fan_flow_configuration_template_DP37_full_no_cooling.xlsm")
    model.load(semantic_model_filename=filename, fcn=fcn_, create_signature_graphs=False, validate_model=True, verbose=False, force_config_update=True)
    if id is not None:
        model.id = id
    return model

"""def get_model_60(id=None, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn
    model = tb.Model(id="model_1st_floor_60", saveSimulationResult=True)
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), "configuration_template_DP37_full_no_cooling.xlsm")
    model.load(semantic_model_filename=filename, fcn=fcn_, create_signature_graphs=True, validate_model=True)
    if id is not None:
        model.id = id
    return model
"""
def run():
    stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2024, month=1, day=5, hour=0, minute=0, second=0,
                                  tzinfo=gettz("Europe/Copenhagen"))

    endTime = datetime.datetime(year=2024, month=1, day=6, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = get_model()

    """simulator = tb.Simulator()

    simulator.simulate(model = model,
                       startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize)"""

    """list = [1]*144

    valve = model.component_dict["029A_space_heater_valve"].savedOutput["valvePosition"]

    print(valve)

    model.component_dict["029A_space_heater_valve"].savedInput["valvePosition"] = list

    valve = model.component_dict["029A_space_heater_valve"].savedOutput["valvePosition"]

    print(valve)"""

    monitor = tb.Monitor(model) #Compares the simulation results with the expected results
    monitor.monitor(startTime=startTime,
                       endTime=endTime,
                        stepSize=stepSize,
                        show=True, 
                        subset=None)
    
    """plot.plot_component(monitor.simulator, 
                        components_1axis=[("supply_flow_junction","airFlowRateIn")], show=True)"""


    # profiler = cProfile.Profile()
    # profiler.enable()

    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240810_125036_mcmc.npz" #After new x0
    # model.load_estimation_result(loaddir)

    """evaluator = tb.Evaluator()

    simulator = tb.Simulator()

    simulator.simulate(model = model,
                       startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize)

    df = simulator.get_simulation_readings()"""

    #plot.plot_space_CO2(model = model, simulator = simulator, space_id = "[029A][029A_space_heater]", show=True)

    #co2_029 = evaluator.get_kpi(df, "029A_co2_sensor", "T", model=model)

    

    """monitor.monitor(startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize,
                        show=True,
                        subset=None,
                        titles=None)
"""

    """monitor = tb.Monitor(model)

    subset = ["BTA006"]
    component = model.component_dict["[HF04_heat_exchanger][HF04_heat_exchanger (return)][HF04_heat_exchanger (supply)]"]
    parameter_values = [0.75, 0.78, 0.8, 0.82, 0.85, 0.87]
    parameter_name = ["eps_100_h"]
    for i in range(len(parameter_values)):
         titles = [f"{s}: {parameter_name} = {parameter_values[i]}" for s in subset]
         for name in parameter_name:
            rsetattr(component, name, parameter_values[i])
            monitor.monitor(startTime=startTime,
                         endTime=endTime,
                         stepSize=stepSize,
                         show=True,
                         subset=subset,
                         titles=titles)"""


    #plt.figure()
    #plt.plot(monitor.simulator.dateTimeSteps, model.component_dict["[cooling_coil][cooling_coil (airside)][heating_coil][heating_coil (airside)]"].savedOutput["heatingPower"], color="red")
    #plt.show()

    """simulator = tb.Simulator(model)
    simulator.simulate(model, startTime=startTime, endTime=endTime, stepSize=stepSize)

    plot.plot_space_temperature_fmu(model, simulator, "[007A][007A_space_heater]")
    plt.figure()
    plt.plot(simulator.dateTimeSteps, model.component_dict["007A_occupancy_profile"].savedOutput["scheduleValue"], color="green")
    plt.figure()
    plt.plot(simulator.dateTimeSteps, model.component_dict["020A_occupancy_profile"].savedOutput["scheduleValue"], color="red")
    plt.figure()
    plt.plot(simulator.dateTimeSteps, model.component_dict["029A_occupancy_profile"].savedOutput["scheduleValue"], color="blue")
    plt.figure()
    plt.plot(simulator.dateTimeSteps, model.component_dict["035A_occupancy_profile"].savedOutput["scheduleValue"], color="blue")
    plt.show()
    """
    
    # profiler.disable()
    # out = io.StringIO()
    # pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(100)
    # plot.plot_outdoor_environment(model, monitor.simulator, firstAxisylim=[5, 30])
    # space_name = "[029A][029A_space_heater]"
    # temperature_heating_controller_name = "029A_temperature_heating_controller"

    # plot.plot_space_temperature_fmu(model, monitor.simulator, space_name)
    # plot.plot_space_CO2_fmu(model, monitor.simulator, space_name)
    # plot.plot_temperature_controller(model, monitor.simulator, temperature_heating_controller_name, show=True)


if __name__ == "__main__":
    run()