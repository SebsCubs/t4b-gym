import matplotlib.pyplot as plt
import matplotlib
import pickle
import math
import numpy as np
import os
import datetime
import sys
import corner
import seaborn as sns
import copy
from dateutil import tz
from matplotlib.colors import LinearSegmentedColormap
if __name__ == '__main__':
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = os.path.join(uppath(os.path.abspath(__file__), 4), "Twin4Build")
    sys.path.append(file_path)
from twin4build.utils.uppath import uppath
import twin4build as tb
# from model_DP37_020B_newconnect import fcn
import twin4build.utils.plot.plot as plot
from model_DP37_029A_031A_033A_035A import get_model


def test_load_emcee_chain():

    stepSize = 600
    model = get_model()
    simulator = tb.Simulator(model)

    space_029A = model.component_dict["[029A][029A_space_heater]"]
    space_031A = model.component_dict["[031A][031A_space_heater]"]
    space_033A = model.component_dict["[033A][033A_space_heater]"]
    space_035A = model.component_dict["[035A][035A_space_heater]"]
    heating_controller_029A = model.component_dict["029A_temperature_heating_controller"]
    heating_controller_031A = model.component_dict["031A_temperature_heating_controller"]
    heating_controller_033A = model.component_dict["033A_temperature_heating_controller"]
    heating_controller_035A = model.component_dict["035A_temperature_heating_controller"]
    space_heater_valve_029A = model.component_dict["029A_space_heater_valve"]
    space_heater_valve_031A = model.component_dict["031A_space_heater_valve"]
    space_heater_valve_033A = model.component_dict["033A_space_heater_valve"]
    space_heater_valve_035A = model.component_dict["035A_space_heater_valve"]
    supply_damper_029A = model.component_dict["029A_room_supply_damper"]
    supply_damper_031A = model.component_dict["031A_room_supply_damper"]
    supply_damper_033A = model.component_dict["033A_room_supply_damper"]
    supply_damper_035A = model.component_dict["035A_room_supply_damper"]
    exhaust_damper_029A = model.component_dict["029A_room_exhaust_damper"]
    exhaust_damper_031A = model.component_dict["031A_room_exhaust_damper"]
    exhaust_damper_033A = model.component_dict["033A_room_exhaust_damper"]
    exhaust_damper_035A = model.component_dict["035A_room_exhaust_damper"]

    targetParameters = {"private": {
        "C_wall": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
        "C_air": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
        "C_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4,
                       "ub": 1e+6},
        "R_out": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.01, "lb": 1e-5, "ub": 0.05},
        "R_in": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.01, "lb": 1e-5, "ub": 0.05},
        "R_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.01, "lb": 0.001,
                       "ub": 0.05},
        "f_wall": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1, "lb": 0, "ub": 2},
        "f_air": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1, "lb": 0, "ub": 2},
        "m_flow_nominal_sh": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.0202, "lb": 1e-3,
                              "ub": 0.5},
        "Q_flow_nominal_sh": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1000, "lb": 100,
                              "ub": 10000},
        "n_sh": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1.24, "lb": 1, "ub": 2},
        "Kp": {"components": [heating_controller_029A, heating_controller_031A, heating_controller_033A,
                              heating_controller_035A], "x0": 2e-4, "lb": 1e-5, "ub": 3},
        "Ti": {"components": [heating_controller_029A, heating_controller_031A, heating_controller_033A,
                              heating_controller_035A], "x0": 3e-1, "lb": 1e-5, "ub": 3},
        "m_flow_nominal": {"components": [space_heater_valve_029A, space_heater_valve_031A, space_heater_valve_033A,
                                          space_heater_valve_035A], "x0": 0.0202, "lb": 1e-3, "ub": 0.3},  # 0.0202
        "flowCoefficient.hasValue": {
            "components": [space_heater_valve_029A, space_heater_valve_031A, space_heater_valve_033A,
                           space_heater_valve_035A], "x0": 8.7, "lb": 1, "ub": 100},
        "dpFixed_nominal": {"components": [space_heater_valve_029A, space_heater_valve_031A, space_heater_valve_033A,
                                           space_heater_valve_035A], "x0": 1e-6, "lb": 0, "ub": 10000}
        },
                        "shared": {
                            "C_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5,
                                           "lb": 1e+4, "ub": 1e+6},
                            "R_int": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.01,
                                      "lb": 0.001, "ub": 0.05},
                            "nominalAirFlowRate.hasValue": {"components": [supply_damper_029A, exhaust_damper_029A],
                                                            "x0": 0.1, "lb": 0.08, "ub": 0.3},
                            "nominalAirFlowRate.hasValue": {"components": [supply_damper_031A, exhaust_damper_031A],
                                                            "x0": 0.1, "lb": 0.08, "ub": 0.3},
                            "nominalAirFlowRate.hasValue": {"components": [supply_damper_033A, exhaust_damper_033A],
                                                            "x0": 0.1, "lb": 0.08, "ub": 0.3},
                            "nominalAirFlowRate.hasValue": {"components": [supply_damper_035A, exhaust_damper_035A],
                                                            "x0": 0.1, "lb": 0.08, "ub": 0.3},
                            "a": {"components": [supply_damper_029A, exhaust_damper_029A, supply_damper_031A,
                                                 exhaust_damper_031A, supply_damper_033A, exhaust_damper_033A,
                                                 supply_damper_035A, exhaust_damper_035A], "x0": 5, "lb": 0.5, "ub": 8},
                            "T_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 20,
                                           "lb": 19, "ub": 23},
                            }}

    percentile = 2

    loaddir = r"D:\Twin4Build\RemoteResults\chain_logs\20240607_130514_fac6_temp1.pickle"
    #loaddir = r"D:\Twin4Build\RemoteResults\chain_logs\20240605_094822_6fac_temp2.pickle"
    model.load_estimation_result(loaddir)

    #plot_logl_plot(model=model)
    plot.trace_plot(model=model, n_subplots =20, save_plot = True, max_cols = 4)
    #plot.corner_plot(model = model, subsample_factor=1000, save_plot = False, param_blocks=10)


if __name__=="__main__":
    test_load_emcee_chain()