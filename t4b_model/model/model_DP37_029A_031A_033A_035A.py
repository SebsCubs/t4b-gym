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
    spaces = self.get_component_by_class(self.component_dict, tb.BuildingSpaceFMUSystem)
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace1AdjFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace2AdjFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace1AdjBoundaryFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace2AdjBoundaryFMUSystem))
    for space in spaces:
        self.add_connection(supply_water_temperature_schedule, space, 
                            "scheduleValue", "supplyWaterTemperature")
        
def get_model():
    model = tb.Model(id="model_coupled", saveSimulationResult=True)
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), "configuration_template_DP37_coupled_no_cooling.xlsm")
    model.load_model(semantic_model_filename=filename, infer_connections=True, fcn=fcn, create_signature_graphs=False, validate_model=False)
    return model

def run():
    stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2023, month=11, day=20, hour=0, minute=0, second=0,
                                  tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2023, month=12, day=12, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = get_model()
    monitor = tb.Monitor(model) #Compares the simulation results with the expected results
    monitor.monitor(startTime=startTime,
                    endTime=endTime,
                    stepSize=stepSize,
                    show=False)
    plot.plot_outdoor_environment(model, monitor.simulator, firstAxisylim=[5, 30])
    space_name = "[029A][029A_space_heater]"
    temperature_heating_controller_name = "029A_temperature_heating_controller"

    plot.plot_space_temperature_fmu(model, monitor.simulator, space_name)
    plot.plot_space_CO2_fmu(model, monitor.simulator, space_name)
    plot.plot_temperature_controller(model, monitor.simulator, temperature_heating_controller_name, show=True)

if __name__ == "__main__":
    run()