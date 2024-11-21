import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Mac specific?
import datetime
import pandas as pd
from dateutil.tz import gettz
import sys

# Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = os.path.join(uppath(os.path.abspath(__file__), 4), "BuildingEnergyModel")
    sys.path.append(file_path)
from twin4build.model.model import Model
from twin4build.simulator.simulator import Simulator
import twin4build.utils.plot.plot as plot
from twin4build.utils.schedule.schedule_system import ScheduleSystem
from twin4build.utils.piecewise_linear_schedule import PiecewiseLinearScheduleSystem
from twin4build.utils.uppath import uppath
import json_data_handling


def fcn(self):
    '''
        The fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''

    # import data from json files:

    json_file_path = os.path.join(uppath(os.path.abspath(__file__), 1),
                                  "concatenatedSchedules.json")

    supply_air_temperature_file = os.path.join(uppath(os.path.abspath(__file__), 1),
                                  "schedulesForEachRoom/HF04_schedule.json")

    supply_water_temperature_file = os.path.join(uppath(os.path.abspath(__file__), 1),
                                  "schedulesForEachRoom/HD01_schedule")

    schedules = json_data_handling.read_json_file(json_file_path)
    schedule_type_lst = json_data_handling.extract_schedule_names(schedules)

    occupancy_schedule_instances = {}
    indoor_temperature_setpoint_schedule_instances = {}

    for schedule_id in schedule_type_lst['occupancy_schedule_names']:
        instance = ScheduleSystem(**(json_data_handling.extract_schedule_from_json(json_file_path, schedule_id, 'occupancy')))
        self._add_component(instance)
        occupancy_schedule_instances[schedule_id] = instance

    for schedule_id in schedule_type_lst['indoor_temperature_setpoint_schedule_names']:
        instance = ScheduleSystem(**(json_data_handling.extract_schedule_from_json(json_file_path, schedule_id, 'indoor_temperature_setpoint')))
        self._add_component(instance)
        indoor_temperature_setpoint_schedule_instances[schedule_id] = instance

    supply_water_temperature_setpoint_schedule = PiecewiseLinearScheduleSystem(**(json_data_handling.extract_schedule_from_json_single_file(supply_water_temperature_file, "supply_water_temperature_schedule")))
    supply_air_temperature_setpoint_schedule = ScheduleSystem(**(json_data_handling.extract_schedule_from_json_single_file(supply_air_temperature_file, "supply_air_temperature_schedule")))

    self._add_component(supply_water_temperature_setpoint_schedule)
    self._add_component(supply_air_temperature_setpoint_schedule)
    # initial_temperature = 21
    # custom_initial_dict = {"OE20-601b-2": {"indoorTemperature": initial_temperature}}
    # self.set_custom_initial_dict(custom_initial_dict)


def export_csv(simulator):
    model = simulator.model
    df_input = pd.DataFrame()
    df_output = pd.DataFrame()
    df_input.insert(0, "time", simulator.dateTimeSteps)
    df_output.insert(0, "time", simulator.dateTimeSteps)

    for component in model.component_dict.values():
        for property_, arr in component.savedInput.items():
            column_name = f"{component.id} ||| {property_}"
            df_input = df_input.join(pd.DataFrame({column_name: arr}))

        for property_, arr in component.savedOutput.items():
            column_name = f"{component.id} ||| {property_}"
            df_output = df_output.join(pd.DataFrame({column_name: arr}))

    df_measuring_devices = simulator.get_simulation_readings()

    df_input.set_index("time").to_csv("input.csv")
    df_output.set_index("time").to_csv("output.csv")
    df_measuring_devices.set_index("time").to_csv("measuring_devices.csv")


def run():
    stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0,
                                  tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2022, month=1, day=7, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = Model(id="model_4_rooms", saveSimulationResult=True)
    filename = os.path.join(uppath(os.path.abspath(__file__), 4), "BuildingEnergyModel", "twin4build", "test", "data",
                            "time_series_data", "weather_DMI.csv")
    model.add_outdoor_environment(filename=filename)
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), "configuration_template_DP37_extended.xlsm")
    model.load_model(semantic_model_filename=filename, infer_connections=True, fcn=fcn)

    simulator = Simulator()
    simulator.simulate(model,
                       stepSize=stepSize,
                       startTime=startTime,
                       endTime=endTime)
    # export_csv(simulator)

    space_name = "033A"
    space_heater_name = "033A_space_heater"
    temperature_controller_name = "033A_temperature_controller"
    CO2_controller_name = "033A_co2_controller"
    damper_name = "033A_room_supply_damper"

    plot.plot_space_temperature(model, simulator, space_name)
    plot.plot_space_CO2(model, simulator, space_name)
    plot.plot_outdoor_environment(model, simulator)
    plot.plot_space_heater(model, simulator, space_heater_name)
    plot.plot_space_heater_energy(model, simulator, space_heater_name)
    plot.plot_temperature_controller(model, simulator, temperature_controller_name)
    plot.plot_CO2_controller_rulebased(model, simulator, CO2_controller_name)
    plot.plot_space_wDELTA(model, simulator, space_name)
    plot.plot_heating_coil(model, simulator, "heating_coil")
    plot.plot_damper(model, simulator, damper_name, show=True)


if __name__ == "__main__":
    run()