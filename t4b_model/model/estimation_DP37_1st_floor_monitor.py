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
from model_DP37_1st_floor import get_model
from twin4build.utils.rsetattr import rsetattr 
import matplotlib.pyplot as plt

def run():
    stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2023, month=12, day=1, hour=0, minute=0, second=0,
                                  tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2023, month=12, day=12, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = get_model()

    space_007A = model.component_dict["[007A][007A_space_heater]"]
    space_008A = model.component_dict["008A"]
    space_011A = model.component_dict["[011A][011A_space_heater]"]
    space_012A = model.component_dict["[012A][012A_space_heater]"]
    space_013A = model.component_dict["[013A][013A_space_heater]"]
    space_015A = model.component_dict["[015A][015A_space_heater]"]
    space_020A = model.component_dict["[020A][020A_space_heater]"]
    space_020B = model.component_dict["[020B][020B_space_heater]"]
    space_029A = model.component_dict["[029A][029A_space_heater]"]
    space_031A = model.component_dict["[031A][031A_space_heater]"]
    space_033A = model.component_dict["[033A][033A_space_heater]"]
    space_035A = model.component_dict["[035A][035A_space_heater]"]

    spaces_list = [space_007A, space_008A, space_011A, space_012A, space_013A, space_015A, space_020A, space_020B,
                   space_029A, space_031A, space_033A, space_035A]

    heating_controller_007A = model.component_dict["007A_temperature_controller"]
    heating_controller_011A = model.component_dict["011A_temperature_controller"]
    heating_controller_012A = model.component_dict["012A_temperature_controller"]
    heating_controller_013A = model.component_dict["013A_temperature_controller"]
    heating_controller_015A = model.component_dict["015A_temperature_controller"]
    heating_controller_020A = model.component_dict["020A_temperature_controller"]
    heating_controller_020B = model.component_dict["020B_temperature_controller"]
    heating_controller_029A = model.component_dict["029A_temperature_heating_controller"]
    heating_controller_031A = model.component_dict["031A_temperature_heating_controller"]
    heating_controller_033A = model.component_dict["033A_temperature_heating_controller"]
    heating_controller_035A = model.component_dict["035A_temperature_heating_controller"]

    heating_controller_list = [heating_controller_007A, heating_controller_011A, heating_controller_012A,
                               heating_controller_013A, heating_controller_015A, heating_controller_020A, heating_controller_020B,
                               heating_controller_029A, heating_controller_031A, heating_controller_033A, heating_controller_035A]

    space_heater_valve_007A = model.component_dict["007A_space_heater_valve"]
    space_heater_valve_011A = model.component_dict["011A_space_heater_valve"]
    space_heater_valve_012A = model.component_dict["012A_space_heater_valve"]
    space_heater_valve_013A = model.component_dict["013A_space_heater_valve"]
    space_heater_valve_015A = model.component_dict["015A_space_heater_valve"]
    space_heater_valve_020A = model.component_dict["020A_space_heater_valve"]
    space_heater_valve_020B = model.component_dict["020B_space_heater_valve"]
    space_heater_valve_029A = model.component_dict["029A_space_heater_valve"]
    space_heater_valve_031A = model.component_dict["031A_space_heater_valve"]
    space_heater_valve_033A = model.component_dict["033A_space_heater_valve"]
    space_heater_valve_035A = model.component_dict["035A_space_heater_valve"]

    space_heater_valve_list = [space_heater_valve_007A, space_heater_valve_011A, space_heater_valve_012A,
                               space_heater_valve_013A, space_heater_valve_015A, space_heater_valve_020A, space_heater_valve_020B,
                               space_heater_valve_029A, space_heater_valve_031A, space_heater_valve_033A, space_heater_valve_035A]

    supply_damper_007A = model.component_dict["007A_room_supply_damper"]
    supply_damper_008A = model.component_dict["008A_room_supply_damper"]
    supply_damper_011A = model.component_dict["011A_room_supply_damper"]
    supply_damper_012A = model.component_dict["012A_room_supply_damper"]
    supply_damper_013A = model.component_dict["013A_room_supply_damper"]
    supply_damper_015A = model.component_dict["015A_room_supply_damper"]
    supply_damper_020A = model.component_dict["020A_room_supply_damper"]
    supply_damper_020B = model.component_dict["020B_room_supply_damper"]
    supply_damper_029A = model.component_dict["029A_room_supply_damper"]
    supply_damper_031A = model.component_dict["031A_room_supply_damper"]
    supply_damper_033A = model.component_dict["033A_room_supply_damper"]
    supply_damper_035A = model.component_dict["035A_room_supply_damper"]

    exhaust_damper_007A = model.component_dict["007A_room_exhaust_damper"]
    exhaust_damper_008A = model.component_dict["008A_room_exhaust_damper"]
    exhaust_damper_011A = model.component_dict["011A_room_exhaust_damper"]
    exhaust_damper_012A = model.component_dict["012A_room_exhaust_damper"]
    exhaust_damper_013A = model.component_dict["013A_room_exhaust_damper"]
    exhaust_damper_015A = model.component_dict["015A_room_exhaust_damper"]
    exhaust_damper_020A = model.component_dict["020A_room_exhaust_damper"]
    exhaust_damper_020B = model.component_dict["020B_room_exhaust_damper"]
    exhaust_damper_029A = model.component_dict["029A_room_exhaust_damper"]
    exhaust_damper_031A = model.component_dict["031A_room_exhaust_damper"]
    exhaust_damper_033A = model.component_dict["033A_room_exhaust_damper"]
    exhaust_damper_035A = model.component_dict["035A_room_exhaust_damper"]

    damper_list = [supply_damper_007A, exhaust_damper_007A, supply_damper_008A, exhaust_damper_008A, supply_damper_011A,
                   exhaust_damper_011A, supply_damper_012A, exhaust_damper_012A, supply_damper_013A, exhaust_damper_013A,
                   supply_damper_015A, exhaust_damper_015A, supply_damper_020A, exhaust_damper_020A, supply_damper_020B,
                   exhaust_damper_020B, supply_damper_029A, exhaust_damper_029A, supply_damper_031A, exhaust_damper_031A,
                   supply_damper_033A, exhaust_damper_033A, supply_damper_035A, exhaust_damper_035A]






    # spaces_list_space_heater = spaces_list.copy()
    # spaces_list_space_heater.remove(space_008A)

    airVolumes = np.array([s.airVolume for s in spaces_list])
    height = 2.7
    floorAreas = airVolumes/height
    thickness = 0.3
    specific_heat_capacity_wall = 1e+6 #J/m3
    wallAreas = (floorAreas)**(0.5)*height
    C_wall_x0 = wallAreas*thickness*specific_heat_capacity_wall
    furniture_mass = 30#kg/m2
    specific_heat_capacity_furniture = 2000
    C_air_x0 = (airVolumes*1000+furniture_mass*floorAreas*specific_heat_capacity_furniture)
    C_boundary_x0 = floorAreas*specific_heat_capacity_wall*thickness/3

    U_ins = 0.18 #W/m2K
    R_out_x0 = (1/(wallAreas*U_ins))/2
    R_in_x0 = (1/(wallAreas*U_ins))/2

    infiltrations_x0 = airVolumes*0.3/space_007A.airVolume

    print("C_wall_x0:")
    for C_wall_x0_, c in zip(C_wall_x0, spaces_list):
        print(f"{c.id}: {C_wall_x0_} J/K")
    print("C_air_x0:")
    for C_air_x0_, c in zip(C_air_x0, spaces_list):
        print(f"{c.id}: {C_air_x0_} J/K")
    print("C_boundary_x0:")
    for C_boundary_x0_, c in zip(C_boundary_x0, spaces_list):
        print(f"{c.id}: {C_boundary_x0_} J/K")
    print("R_out_x0:")
    for R_out_x0_, c in zip(R_out_x0, spaces_list):
        print(f"{c.id}: {R_out_x0_} K/W")
    print("R_in_x0:")
    for R_in_x0_, c in zip(R_in_x0, spaces_list):
        print(f"{c.id}: {R_in_x0_} K/W")
    

    print("C_wall_x0: ", C_wall_x0)
    print("C_air_x0: ", C_air_x0)
    print("C_boundary_x0: ", C_boundary_x0)
    print("R_out_x0: ", R_out_x0)
    print("R_in_x0: ", R_in_x0)
    print("inf_x0: ", infiltrations_x0)

    R_boundary_x0 = (1/(floorAreas*U_ins))

    print("R_booundary x0", R_boundary_x0)


    monitor = tb.Monitor(model) #Compares the simulation results with the expected results

    # profiler = cProfile.Profile()
    # profiler.enable()

    # subset = ["007A_temperature_sensor", "011A_temperature_sensor", "011A_valve_position_sensor", "012A_temperature_sensor", "013A_temperature_sensor", "015A_temperature_sensor", "020A_temperature_sensor", "020B_temperature_sensor", "029A_temperature_sensor", "031A_temperature_sensor", "033A_temperature_sensor", "035A_temperature_sensor"] #How many plots to show. If only one plot is needed, the list should contain only one name
    subset = ["029A_temperature_sensor", "029A_valve_position_sensor"] #How many plots to show. If only one plot is needed, the list should contain only one name

    
    component = model.component_dict["[029A][029A_space_heater]"]
    components = spaces_list
    components = [component, model.component_dict["[031A][031A_space_heater]"], model.component_dict["[033A][033A_space_heater]"], model.component_dict["[035A][035A_space_heater]"]]
    v = R_in_x0[8]
    parameter_values = [0.0004, 0.001, 0.01, 0.02, 0.05, 0.1] #This list can be as long as needed. A plot will be created for each value in the list
    print("parameter_values: ", parameter_values)
    parameter_name = ["R_int"] # Make a list with 1 name if only one parameter is to be changed. All parameters in this list will be set to the same value for each simulation.
    for i in range(len(parameter_values)):
        titles = [f"{s}: {parameter_name} = {parameter_values[i]}" for s in subset]
        for name in parameter_name:
            for component in components:
                rsetattr(component, name, parameter_values[i])
        monitor.monitor(startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize,
                        show=False,
                        subset=subset,
                        titles=titles,
                        draw_anomaly_signals=False)
        
    plot.plot_outdoor_environment(model, monitor.simulator, firstAxisylim=[-5, 10])
    plt.figure()
    plt.plot(monitor.simulator.dateTimeSteps, model.component_dict["029A_occupancy_profile"].savedOutput["scheduleValue"], color="green")
    plt.show()
if __name__ == "__main__":
    run()