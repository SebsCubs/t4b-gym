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

    simulator = tb.Simulator()

    simulator.simulate(model = model,
                       startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize)

    print("Simulation completed successfully!")

    # Plot the results using plot_component
    space_id = '[020B][020B_space_heater]'
    
    #print the total energy consumption
    energy = np.array(model.components[space_id].savedOutput['spaceHeaterPower'])
    print(f"Total energy consumption: {energy.sum()} Wh")

    #Print the deviation from the setpoint
    deviation = np.array(model.components[space_id].savedOutput['indoorTemperature']) - 21
    print(f"Total deviation from setpoint: {deviation.sum()} °C")

    # Temperature plot
    plot.plot_component(
        simulator,
        components_1axis=[(space_id, 'indoorTemperature'),("020B_temperature_heating_setpoint", 'scheduleValue')],
        ylabel_1axis='Room Temperature [°C] (Actual and Setpoint)',
        show=True
    )

    # CO2 plot
    plot.plot_component(
        simulator,
        components_1axis=[(space_id, 'indoorCo2Concentration'),("020B_co2_setpoint", 'scheduleValue')],
        ylabel_1axis='CO2 Concentration [ppm] (Actual and Setpoint)',
        show=True
    )

    # power plot
    plot.plot_component(
        simulator,
        components_1axis=[(space_id, 'spaceHeaterPower')],
        ylabel_1axis='Space Heater Power [W] (Actual)',
        show=True
    )

        # 020B occupancy plot
    plot.plot_component(
        simulator,
        components_1axis=[("020B_occupancy_profile", 'scheduleValue')],
        ylabel_1axis='Occupancy 020B (Actual)',
        show=True
    )


if __name__ == "__main__":
    run()