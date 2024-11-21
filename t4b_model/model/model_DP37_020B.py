import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Mac specific?
import datetime
from dateutil.tz import gettz
import sys

# Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = r"C:\Users\jabj\Documents\python\Twin4Build"
    sys.path.append(file_path)
    
import twin4build as tb
import twin4build.utils.plot.plot as plot
from twin4build.utils.uppath import uppath

def fcn(self):
    supply_water_temperature_schedule = tb.PiecewiseLinearScheduleSystem(
        weekDayRulesetDict = {
                "ruleset_default_value": {"X": [-12, 5, 20],
                                          "Y": [60, 50, 20]}},
            saveSimulationResult = True,
        id="supply_water_temperature_schedule")
    outdoor_environment = self.get_component_by_class(self.component_dict, tb.OutdoorEnvironmentSystem)[0]
    self.add_connection(outdoor_environment, supply_water_temperature_schedule, "outdoorTemperature", "outdoorTemperature")
    spaces = self.get_component_by_class(self.component_dict, tb.BuildingSpace0AdjBoundaryOutdoorFMUSystem)

    for space in spaces:
        self.add_connection(supply_water_temperature_schedule, space, 
                            "scheduleValue", "supplyWaterTemperature")
    
def get_model():
    model = tb.Model(id="model_020B", saveSimulationResult=True)
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), "configuration_template_DP37_020B_no_cooling.xlsm")
    model.load(semantic_model_filename=filename, fcn=fcn, create_signature_graphs=False, validate_model=True, force_config_update=True)
    return model


def run():
    stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
                                  tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2023, month=12, day=7, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = get_model()


    simulator = tb.Simulator(model)
    simulator.simulate(model, startTime=startTime, endTime=endTime, stepSize=stepSize)
    plot.plot_damper(model, simulator, "020B_room_supply_damper", show=False)
    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.plot(simulator.dateTimeSteps, model.component_dict["020B_occupancy_profile"].savedOutput["numberOfPeople"], label="occ", color="green")
    # plt.plot(simulator.dateTimeSteps, model.component_dict["020B_room_supply_damper"].savedOutput["airFlowRate"], label="supply")
    # plt.twinx()
    # plt.plot(simulator.dateTimeSteps, model.component_dict["co2_sensor"].savedOutput["measuredValue"], label="co2", color="red")
    # plt.legend()
    # plt.show()

    plot.plot_outdoor_environment(model, simulator, firstAxisylim=[5, 30])
    space_name = "[020B][020B_space_heater]"
    temperature_heating_controller_name = "020B_temperature_heating_controller"
    plot.plot_space_temperature_fmu(model, simulator, space_name)
    plot.plot_space_CO2_fmu(model, simulator, space_name)
    plot.plot_temperature_controller(model, simulator, temperature_heating_controller_name, show=True)



    monitor = tb.Monitor(model)
    monitor.monitor(startTime=startTime,
                    endTime=endTime,
                    stepSize=stepSize,
                    show=False)
    
    


    #########################################
    print(model.component_dict["[020B][020B_space_heater]"].savedOutput["spaceHeaterPower"])
    #########################################




if __name__ == "__main__":
    run()