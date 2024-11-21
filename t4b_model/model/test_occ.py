import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Mac specific?
import datetime
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

    spaces = self.get_component_by_class(self.component_dict, tb.BuildingSpaceFMUSystem)
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace1AdjFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace2AdjFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace0AdjBoundaryFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace1AdjBoundaryFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace2AdjBoundaryFMUSystem))
    for space in spaces:
        self.add_connection(supply_water_temperature_schedule, space, 
                            "scheduleValue", "supplyWaterTemperature")
    self.add_connection(outdoor_environment, supply_water_temperature_schedule, "outdoorTemperature", "outdoorTemperature")

    # self.remove_component(self.component_dict["020B_occupancy_profile"])

    s = tb.SensorSystem(id="co2_sensor", filename=self.component_dict["020B_co2_sensor"].filename, datecolumn=2, valuecolumn=4)
    occ = tb.OccupancySystem(id="020B_occupancy_profile", airVolume=100)


    supply_damper = self.component_dict["020B_room_supply_damper"]
    exhaust_damper = self.component_dict["020B_room_exhaust_damper"]

    self.add_connection(supply_damper, occ, "airFlowRate", "supplyAirFlowRate")
    self.add_connection(exhaust_damper, occ, "airFlowRate", "exhaustAirFlowRate")
    self.add_connection(s, occ, "measuredValue", "indoorCO2Concentration")

    space = self.component_dict["[020B][020B_space_heater]"]
    self.add_connection(occ, space,
                            "occupancy", "numberOfPeople")
    
def get_model():
    model = tb.Model(id="model_test_occ", saveSimulationResult=True)
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), "configuration_template_DP37_020B_no_cooling.xlsm")
    model.load_model(semantic_model_filename=filename, infer_connections=True, fcn=fcn, create_signature_graphs=False, validate_model=False)
    return model


def run():
    stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
                                  tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2023, month=12, day=7, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = get_model()
    simulator = tb.Simulator(model)
    simulator.simulate(model,
                       startTime=startTime,
                  endTime=endTime,
                  stepSize=stepSize)
    import matplotlib.pyplot as plt
    plt.plot(simulator.dateTimeSteps, model.component_dict["occupancy"].savedOutput)
    plt.show()
    
if __name__ == "__main__":
    run()