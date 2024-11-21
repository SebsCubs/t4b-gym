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
    
    # flat_attr_list = [r"$\overline{\dot{m}}_{c,w}$", r"$\overline{\dot{m}}_{c,a}$", r"$\tau_w$", r"$\tau_a$", r"$\tau_m$", r"$\overline{UA}$", r"$\overline{\dot{m}}_{v,w}$", r"$\overline{\dot{m}}_{cv,w}$", r"$K_{cv}$", r"$\Delta P_{s,res}$", r"$\overline{\Delta P}_{c}$", r"$\Delta P_{p}$", r"$\Delta P_{s}$", r"$c_1$", r"$c_2$", r"$c_3$", r"$c_4$", r"$f_{tot}$", r"$K_P$", r"$T_I$", r"$T_D$"]

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


    targetParameters = {"private": {"C_wall": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                    "C_air": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                    "C_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                    "R_out": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.01, "lb": 1e-5, "ub": 0.05},
                                    "R_in": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.01, "lb": 1e-5, "ub": 0.05},
                                    "R_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.01, "lb": 0.001, "ub": 0.05},
                                    "f_wall": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1, "lb": 0, "ub": 2},
                                    "f_air": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1, "lb": 0, "ub": 2},
                                    "m_flow_nominal_sh": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.0202, "lb": 1e-3, "ub": 0.5},
                                    "Q_flow_nominal_sh": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1000, "lb": 100, "ub": 10000},
                                    "n_sh": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1.24, "lb": 1, "ub": 2},
                                    "Kp": {"components": [heating_controller_029A, heating_controller_031A, heating_controller_033A, heating_controller_035A], "x0": 2e-4, "lb": 1e-5, "ub": 3},
                                    "Ti": {"components": [heating_controller_029A, heating_controller_031A, heating_controller_033A, heating_controller_035A], "x0": 3e-1, "lb": 1e-5, "ub": 3},
                                    "m_flow_nominal": {"components": [space_heater_valve_029A, space_heater_valve_031A, space_heater_valve_033A, space_heater_valve_035A], "x0": 0.0202, "lb": 1e-3, "ub": 0.3}, #0.0202
                                    "flowCoefficient.hasValue": {"components": [space_heater_valve_029A, space_heater_valve_031A, space_heater_valve_033A, space_heater_valve_035A], "x0": 8.7, "lb": 1, "ub": 100},
                                    "dpFixed_nominal": {"components": [space_heater_valve_029A, space_heater_valve_031A, space_heater_valve_033A, space_heater_valve_035A], "x0": 1e-6, "lb": 0, "ub": 10000}
                                    },
                        "shared": {"C_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                    "R_int": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.01, "lb": 0.001, "ub": 0.05},
                                    "nominalAirFlowRate.hasValue": {"components": [supply_damper_029A, exhaust_damper_029A], "x0": 0.1, "lb": 0.08, "ub": 0.3},
                                    "nominalAirFlowRate.hasValue": {"components": [supply_damper_031A, exhaust_damper_031A], "x0": 0.1, "lb": 0.08, "ub": 0.3},
                                    "nominalAirFlowRate.hasValue": {"components": [supply_damper_033A, exhaust_damper_033A], "x0": 0.1, "lb": 0.08, "ub": 0.3},
                                    "nominalAirFlowRate.hasValue": {"components": [supply_damper_035A, exhaust_damper_035A], "x0": 0.1, "lb": 0.08, "ub": 0.3},
                                    "a": {"components": [supply_damper_029A, exhaust_damper_029A, supply_damper_031A, exhaust_damper_031A, supply_damper_033A, exhaust_damper_033A, supply_damper_035A, exhaust_damper_035A], "x0": 5, "lb": 0.5, "ub": 8},
                                    "T_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 20, "lb": 19, "ub": 23},
                        }}

    percentile = 2
    targetMeasuringDevices = {model.component_dict["029A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["029A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["031A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["031A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["033A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["033A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["035A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["035A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},}

    colors = sns.color_palette("deep")
    blue = colors[0]
    orange = colors[1]
    green = colors[2]
    red = colors[3]
    purple = colors[4]
    brown = colors[5]
    pink = colors[6]
    grey = colors[7]
    beis = colors[8]
    sky_blue = colors[9]
    plot.load_params()

    do_analysis_plots = False #############################################
    assume_uncorrelated_noise = True

    if do_analysis_plots:
        do_iac_plot = True
        do_logl_plot = True
        do_trace_plot = True
        do_jump_plot = True
        do_corner_plot = True
        do_inference = False
    else:
        do_iac_plot = False
        do_logl_plot = False
        do_trace_plot = False
        do_jump_plot = False
        do_corner_plot = False
        do_inference = True

    
    do_swap_plot = False
    
    assert (do_iac_plot and do_inference)!=True



    # DP37 case
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240510_135243.pickle") #uniform a=5, 1-day, err=0.1 ###########################################

    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240512_130803.pickle") #summer
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240512_145401.pickle") #summer
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240513_065422.pickle") #summer
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240513_121837.pickle") #winter
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240513_132926.pickle") #winter
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240513_132926.pickle") #winter
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240513_154625.pickle") #winter
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240514_073900.pickle") #winter
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240514_073900.pickle") #winter

    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240526_134413.pickle") #winter
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240527_163226.pickle") #winter
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240528_104929.pickle") #winter
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240528_164745.pickle") #winter
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240530_124121.pickle") #winter
    
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240603_081113.pickle") #winter
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240604_103744.pickle") #winter
    loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240605_094822.pickle") #winter
    # loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240613_084127.pickle") #winter dec2 - dec10, 2temp, 6facwalker
    # loaddir = os.path.join(r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\remote_results\chain_logs\chain_logs", "20240614_134812.pickle") #winter dec2 - dec10, 2temp, 6facwalker



    with open(loaddir, 'rb') as handle:
        result = pickle.load(handle)



    result["chain.T"] = 1/result["chain.betas"] ##################################
    
    burnin = 2899#int(result["chain.x"].shape[0])-3000 #100
    #########################################
    list_ = ["integratedAutoCorrelatedTime"]#, "chain.jumps_accepted", "chain.jumps_proposed", "chain.swaps_accepted", "chain.swaps_proposed"]
    for key in list_:
        result[key] = np.array(result[key])
    #########################################

    vmin = np.min(result["chain.betas"])
    vmax = np.max(result["chain.betas"])

    print(result["chain.x"].shape)


# #################################################################
    logl = result["chain.logl"]
    logl[np.abs(logl)>1e+9] = np.nan
    print(logl[:,0,:].max())
    indices = np.where(logl[:,0,:] == np.nanmax(logl[:,0,:]))
    print(indices)
    s0 = indices[0][0]
    s1 = indices[1][0]
    a = result["chain.x"][s0, 0, s1, :]
    a = np.resize(a, (1,1,1,a.shape[0]))
    result["chain.x"] = a
    for key in result.keys():
        # if key not in list_:
        if isinstance(result[key], list):
            result[key] = np.array(result[key])
    print(result["chain.x"].shape)


    records_array = np.array(result["theta_mask"])
    vals, inverse, count = np.unique(records_array, return_inverse=True,
                              return_counts=True)
    idx_vals_repeated = np.where(count > 1)[0]
    vals_repeated = vals[idx_vals_repeated]
    rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
    _, inverse_rows = np.unique(rows, return_index=True)
    res = np.split(cols, inverse_rows[1:])
    d_idx = []
    for i in res:
        d_idx.extend(list(i[1:]))
    attr_list = np.array(result["component_attr"])
    attr_list = np.delete(attr_list, np.array(d_idx).astype(int)) #res is an array of duplicates, so its size should always be larger than 1
    # print("-----------------")
    # print(result["chain.x"].shape)
    # print(result["chain.x"][0,0,0,:])
    # for v, attr in zip(result["chain.x"][0,0,0,:], attr_list):
    #     print(attr, v)
########################################################

    ndim = result["chain.x"].shape[3]
    ntemps = result["chain.x"].shape[1]
    nwalkers = result["chain.x"].shape[2] #Round up to nearest even number and multiply by 2


    cm_sb = sns.diverging_palette(210, 0, s=50, l=50, n=ntemps, center="dark") #vlag_r
    cm_sb_rev = list(reversed(cm_sb))
    cm_mpl = LinearSegmentedColormap.from_list("seaborn", cm_sb)#, N=ntemps)
    cm_mpl_rev = LinearSegmentedColormap.from_list("seaborn_rev", cm_sb_rev, N=ntemps)

    startTime_test1 = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test1 = datetime.datetime(year=2023, month=12, day=31, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test2 = datetime.datetime(year=2022, month=2, day=2, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test2 = datetime.datetime(year=2022, month=2, day=2, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test3 = datetime.datetime(year=2022, month=2, day=8, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test3 = datetime.datetime(year=2022, month=2, day=8, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test4 = datetime.datetime(year=2022, month=2, day=9, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test4 = datetime.datetime(year=2022, month=2, day=9, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test5 = datetime.datetime(year=2022, month=2, day=10, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))  
    endTime_test5 = datetime.datetime(year=2022, month=2, day=10, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test6 = datetime.datetime(year=2022, month=2, day=11, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test6 = datetime.datetime(year=2022, month=2, day=11, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))

    startTime_test7 = datetime.datetime(year=2022, month=2, day=12, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test7 = datetime.datetime(year=2022, month=2, day=12, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test8 = datetime.datetime(year=2022, month=2, day=13, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test8 = datetime.datetime(year=2022, month=2, day=13, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test9 = datetime.datetime(year=2022, month=2, day=14, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test9 = datetime.datetime(year=2022, month=2, day=14, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test10 = datetime.datetime(year=2022, month=2, day=15, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test10 = datetime.datetime(year=2022, month=2, day=15, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test11 = datetime.datetime(year=2022, month=2, day=16, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test11 = datetime.datetime(year=2022, month=2, day=16, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test12 = datetime.datetime(year=2022, month=2, day=17, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test12 = datetime.datetime(year=2022, month=2, day=17, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))


    startTime_test = [startTime_test1]
    endTime_test = [endTime_test1]
    stepSize_test = [stepSize]


    

    n_par = result["n_par"]
    n_par_map = result["n_par_map"]
    print(result["n_par"])
    print(result["n_par_map"])


    if assume_uncorrelated_noise==False:
        for j, measuring_device in enumerate(targetMeasuringDevices):
            # print(n_par_map[measuring_device.id])
            for i in range(n_par_map[measuring_device.id]):
                if i==0:
                    s = f"$a_{str(j)}$"
                    s = r'{}'.format(s)
                    flat_attr_list.append(s)
                # elif i==1:
                #     s = r'$\gamma_{%.0f}$' % (j,)
                #     flat_attr_list.append(s)
                # elif i==2:
                #     s = r'$\mathrm{ln}P_{%.0f}$' % (j,)
                #     flat_attr_list.append(s)
                else:
                    s = r'$l_{%.0f,%.0f}$' % (j,i-1, )
                    flat_attr_list.append(s)



    if do_inference:
        model.load_estimation_result(loaddir)

        startTime_train1 = datetime.datetime(year=2022, month=2, day=1, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        endTime_train1 = datetime.datetime(year=2022, month=2, day=1, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        startTime_train2 = datetime.datetime(year=2022, month=2, day=2, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        endTime_train2 = datetime.datetime(year=2022, month=2, day=2, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        startTime_train3 = datetime.datetime(year=2022, month=2, day=3, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        endTime_train3 = datetime.datetime(year=2022, month=2, day=3, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        startTime_train4 = datetime.datetime(year=2022, month=2, day=4, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        endTime_train4 = datetime.datetime(year=2022, month=2, day=4, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        startTime_train5 = datetime.datetime(year=2022, month=2, day=5, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        endTime_train5 = datetime.datetime(year=2022, month=2, day=5, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        startTime_train6 = datetime.datetime(year=2022, month=2, day=6, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        endTime_train6 = datetime.datetime(year=2022, month=2, day=6, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        startTime_train7 = datetime.datetime(year=2022, month=2, day=7, hour=8, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        endTime_train7 = datetime.datetime(year=2022, month=2, day=7, hour=22, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
        model.chain_log["startTime_train"] = [startTime_train1, startTime_train2, startTime_train3, startTime_train4, startTime_train5, startTime_train6, startTime_train7]
        model.chain_log["endTime_train"] = [endTime_train1, endTime_train2, endTime_train3, endTime_train4, endTime_train5, endTime_train6, endTime_train7]
        model.chain_log["stepSize_train"] = [stepSize, stepSize, stepSize, stepSize, stepSize, stepSize, stepSize]
        # parameter_chain = parameter_chain[::100,:,:]

        del result
        # ylims = ([20, 23], [19.5, 25], [None, None], [0,1], [19.5, 28])
        ylims = ([0,1], [19.5, 28.5], [19.5, 25.5], [20, 23.5], [None, None])
        
        fig, axes = simulator.run_emcee_inference(model, targetParameters, targetMeasuringDevices, startTime_test, endTime_test, stepSize_test, assume_uncorrelated_noise=assume_uncorrelated_noise, burnin=burnin)
        plt.show()
        ylabels = [r"$u_v [1]$", r"$T_{c,w,in} [^\circ\!C]$", r"$T_{c,w,out} [^\circ\!C]$", r"$T_{c,a,out} [^\circ\!C]$", r"$\dot{P}_f [W]$"]
        # fig.subplots_adjust(hspace=0.3)
        # fig.set_size_inches((15,10))
        for ax, ylabel, ylim in zip(axes, ylabels, ylims):
            # ax.legend(loc="center left", bbox_to_anchor=(1,0.5), prop={'size': 12})
            # pos = ax.get_position()
            # pos.x0 = 0.15       # for example 0.2, choose your value
            # pos.x1 = 0.99       # for example 0.2, choose your value

            # ax.set_position(pos)
            ax.tick_params(axis='y', labelsize=10)
            # ax.locator_params(axis='y', nbins=3)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.text(-0.07, 0.5, ylabel, fontsize=14, rotation="horizontal", ha="right", transform=ax.transAxes)
            ax.xaxis.label.set_color("black")

        
        # axes[3].plot(simulator.dateTimeSteps, model.component_dict["Supply air temperature setpoint"].savedOutput["scheduleValue"], color="blue", label="setpoint", linewidth=0.5)
        # axes[3].plot(simulator.dateTimeSteps, model.component_dict["fan inlet air temperature sensor"].get_physical_readings(startTime, endTime, stepSize)[0:-1], color="green", label="inlet air", linewidth=0.5)
        fig.savefig(r'C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\LBNL_inference_plot.png', dpi=300)
        # ax.plot(simulator.dateTimeSteps, simulator.model.component_dict[])


    

    if do_inference==False:
        assert len(flat_attr_list) == ndim, f"Number of parameters in flat_attr_list ({len(flat_attr_list)}) does not match number of parameters in chain.x ({ndim})"
        
        plt.rcParams['mathtext.fontset'] = 'cm'



        if assume_uncorrelated_noise==False:
            attr_list_model = flat_attr_list[:-n_par]
            attr_list_noise = flat_attr_list[-n_par:]
            flat_attr_list__ = [attr_list_model, attr_list_noise]
            list_ = ["chain.x"]
            result_model = result.copy()
            result_noise = result.copy()
            for key in list_:
                result_key = result[key]
                result_model[key] = result_key[...,:-n_par]
                result_noise[key] = result_key[...,-n_par:]
            result_list = [result_model, result_noise]
        else:
            flat_attr_list__ = [flat_attr_list]
            result_list = [result]

        if do_jump_plot:
            fig_jump, ax_jump = plt.subplots(layout='compressed')
            fig_jump.set_size_inches((17, 12))
            fig_jump.suptitle("Jumps", fontsize=20)
            # n_checkpoints = result["chain.jumps_proposed"].shape[0]
            # for i_checkpoint in range(n_checkpoints):
            #     for i in range(ntemps):
            #         ax_jump.scatter([i_checkpoint]*nwalkers, result["chain.jumps_accepted"][i_checkpoint,i,:]/result["chain.jumps_proposed"][i_checkpoint,i,:], color=cm_sb[i], s=20, alpha=1)

            n_it = result["chain.jump_acceptance"].shape[0]
            # for i_walker in range(nwalkers):
            for i in range(ntemps):
                if i==0: #######################################################################
                    ax_jump.plot(range(n_it), result["chain.jump_acceptance"][:,i], color=cm_sb[i])
        if do_logl_plot:
            fig_logl, ax_logl = plt.subplots(layout='compressed')
            fig_logl.set_size_inches((17/4, 12/4))
            fig_logl.suptitle("Log-likelihood", fontsize=20)
            # logl = np.abs(result_["chain.logl"])
            logl = result["chain.logl"]
            logl[np.abs(logl)>1e+9] = np.nan
            
            indices = np.where(logl[:,0,:] == np.nanmax(logl[:,0,:]))
            print(logl[:,0,:].max())
            s0 = indices[0][0]
            s1 = indices[1][0]
            print("logl_max: ", logl[s0,0,s1])
            # print("x_max: ", result["chain.x"][s0, 0, s1, :])
            
            n_it = result["chain.logl"].shape[0]
            for i_walker in range(nwalkers):
                for i in range(ntemps):
                    if i==0: #######################################################################
                        ax_logl.plot(range(n_it), logl[:,i,i_walker], color=cm_sb[i])
                        # ax_logl.set_yscale('log')

        for ii, (flat_attr_list_, result_) in enumerate(zip(flat_attr_list__, result_list)):
            nparam = len(flat_attr_list_)
            ncols = 3
            nrows = math.ceil(nparam/ncols)
            print(nparam, ncols, nrows)


            ndim = result_["chain.x"].shape[3]
            ntemps = result_["chain.x"].shape[1]
            nwalkers = result_["chain.x"].shape[2] #Round up to nearest even number and multiply by 2
            
            # cm = plt.get_cmap('RdYlBu', ntemps)
            # cm_sb = sns.color_palette("vlag_r", n_colors=ntemps, center="dark") #vlag_r
            cm_sb = sns.diverging_palette(210, 0, s=50, l=50, n=ntemps, center="dark") #vlag_r
            cm_sb_rev = list(reversed(cm_sb))
            cm_mpl = LinearSegmentedColormap.from_list("seaborn", cm_sb)#, N=ntemps)
            cm_mpl_rev = LinearSegmentedColormap.from_list("seaborn_rev", cm_sb_rev, N=ntemps)

            fig_trace_beta, axes_trace = plt.subplots(nrows=nrows, ncols=ncols, layout='compressed')
            fig_trace_beta.set_size_inches((17, 12))
            

            # list_ = ["chain.logl", "chain.logP", "chain.x", "chain.betas"]
            # for key in list_:
            #     for i, arr in enumerate(result[key]):
            #         result[key][i] = arr[-nsample_checkpoint:]
                
            # for key in result.keys():
            #     result[key] = np.concatenate(result[key],axis=0)
                # result["chain.jumps_accepted"].append(chain.jumps_accepted)
                # result["chain.jumps_proposed"].append(chain.jumps_proposed)
                # result["chain.logl"].append(chain.logl)
                # result["chain.logP"].append(chain.logP)
                # result["chain.swaps_accepted"].append(chain.swaps_accepted)
                # result["chain.swaps_proposed"].append(chain.swaps_proposed)
                # result["chain.x"].append(chain.x)
                # result["chain.betas"].append(chain.betas)

            # vmin = np.min(result["chain.T"])
            # vmax = np.max(result["chain.T"])
            
            



            if do_iac_plot:
                fig_iac = fig_trace_beta
                axes_iac = copy.deepcopy(axes_trace)
                for j, attr in enumerate(flat_attr_list_):
                    row = math.floor(j/ncols)
                    col = int(j-ncols*row)
                    axes_iac[row, col] = axes_trace[row, col].twinx()
                # fig_iac, axes_iac = plt.subplots(nrows=nrows, ncols=ncols, layout='compressed')
                # fig_iac.set_size_inches((17, 12))
                # fig_iac.suptitle("Integrated AutoCorrelated Time", fontsize=20)
                iac = result_["integratedAutoCorrelatedTime"][:-1]
                n_it = iac.shape[0]
                for i in range(ntemps):
                    beta = result_["chain.betas"][:, i]
                    for j, attr in enumerate(flat_attr_list_):
                        row = math.floor(j/ncols)
                        col = int(j-ncols*row)
                        
                        if ntemps>1:
                            sc = axes_iac[row, col].plot(range(n_it), iac[:,i,j], color=red, alpha=1, zorder=1)
                        else:
                            sc = axes_iac[row, col].plot(range(n_it), iac[:,i,j], color=red, alpha=1, zorder=1)
                
                # add heristic tau = N/50 line
                heuristic_line = np.arange(n_it)/20
                for j, attr in enumerate(flat_attr_list_):
                    row = math.floor(j/ncols)
                    col = int(j-ncols*row)
                    axes_iac[row, col].plot(range(n_it), heuristic_line, color="black", linewidth=1, linestyle="dashed", alpha=1, label=r"$\tau=N/50$")
                    axes_iac[row, col].set_ylim([0-0.05*iac.max(), iac.max()+0.05*iac.max()])
                # fig_iac.legend()
                
            

            
            if do_trace_plot:
                
                chain_logl = result_["chain.logl"]
                bool_ = chain_logl<-5e+9
                chain_logl[bool_] = np.nan
                chain_logl[np.isnan(chain_logl)] = np.nanmin(chain_logl)

                for nt in reversed(range(ntemps)):
                    for nw in range(nwalkers):
                        x = result_["chain.x"][:, nt, nw, :]
                        T = result_["chain.T"][:, nt]
                        beta = result_["chain.betas"][:, nt]
                        logl = chain_logl[:, nt, nw]
                        # alpha = (max_alpha-min_alpha)*(logl-logl_min)/(logl_max-logl_min) + min_alpha
                        # alpha = (max_alpha-min_alpha)*(T-vmin)/(vmax-vmin) + min_alpha
                        # alpha = (max_alpha-min_alpha)*(beta-vmin)/(vmax-vmin) + min_alpha
                        # Trace plots
                        
                        
                        for j, attr in enumerate(flat_attr_list_):
                            row = math.floor(j/ncols)
                            col = int(j-ncols*row)
                            # sc = axes_trace[row, col].scatter(range(x[:,j].shape[0]), x[:,j], c=T, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), s=0.3, cmap=cm_mpl, alpha=0.1)
                            if ntemps>1:
                                
                                sc = axes_trace[row, col].scatter(range(x[:,j].shape[0]), x[:,j], c=beta, vmin=vmin, vmax=vmax, s=0.3, cmap=cm_mpl_rev, alpha=0.1)
                            else:
                                sc = axes_trace[row, col].scatter(range(x[:,j].shape[0]), x[:,j], s=0.3, color=cm_sb[0], alpha=0.1)
                                
                            axes_trace[row, col].axvline(burnin, color="black", linewidth=1, alpha=0.8)#, linestyle="--")

                            # if plotted==False:
                            #     axes_trace[row, col].text(x_left+dx/2, 0.44, 'Burnin', ha='center', va='center', rotation='horizontal', fontsize=15, transform=axes_trace[row, col].transAxes)
                            #     axes_trace[row, col].arrow(x_right, 0.5, -dx, 0, head_width=0.1, head_length=0.05, color="black", transform=axes_trace[row, col].transAxes)
                            #     axes_trace[row, col].arrow(x_left, 0.5, dx, 0, head_width=0.1, head_length=0.05, color="black", transform=axes_trace[row, col].transAxes)
                            #     axes_trace[row, col].set_ylabel(attr, fontsize=20)
                            #     plotted = True



                x_left = 0.1
                x_mid_left = 0.515
                x_right = 0.9
                x_mid_right = 0.58
                dx_left = x_mid_left-x_left
                dx_right = x_right-x_mid_right

                fontsize = 12
                for j, attr in enumerate(flat_attr_list_):
                    row = math.floor(j/ncols)
                    col = int(j-ncols*row)
                    axes_trace[row, col].axvline(burnin, color="black", linestyle=":", linewidth=1.5, alpha=0.5)
                    y = np.array([-np.inf, np.inf])
                    x1 = -burnin
                    x2 = burnin
                    axes_trace[row, col].fill_betweenx(y, x1, x2=0)
                    axes_trace[row, col].text(x_left+dx_left/2, 0.44, 'Burn-in', ha='center', va='center', rotation='horizontal', fontsize=fontsize, transform=axes_trace[row, col].transAxes)
                    # axes_trace[row, col].arrow(x_mid_left, 0.5, -dx_left, 0, head_width=0.1, head_length=0.05, color="black", transform=axes_trace[row, col].transAxes)
                    # axes_trace[row, col].arrow(x_left, 0.5, dx_left, 0, head_width=0.1, head_length=0.05, color="black", transform=axes_trace[row, col].transAxes)

                    axes_trace[row, col].text(x_mid_right+dx_right/2, 0.44, 'Posterior', ha='center', va='center', rotation='horizontal', fontsize=fontsize, transform=axes_trace[row, col].transAxes)
                    # axes_trace[row, col].arrow(x_right, 0.5, -dx_right, 0, head_width=0.1, head_length=0.05, color="black", transform=axes_trace[row, col].transAxes)
                    # axes_trace[row, col].arrow(x_mid_right, 0.5, dx_right, 0, head_width=0.1, head_length=0.05, color="black", transform=axes_trace[row, col].transAxes)
                    axes_trace[row, col].set_ylabel(attr, fontsize=20)
                    axes_trace[row, col].ticklabel_format(style='plain', useOffset=False)

                    # arrow = axes_trace[row, col].annotate('', 
                    #                                     xy =(x_left, 0.5),
                    #                                     xytext =(x_mid_left, 0.5), 
                    #                                     arrowprops = dict(
                    #                                         arrowstyle="|-|,widthA=0.7, widthB=0.7"
                    #                                     ))
                    
                    # arrow = axes_trace[row, col].annotate('', 
                    #                                     xy =(x_mid_right, 0.5),
                    #                                     xytext =(x_right, 0.5), 
                    #                                     arrowprops = dict(
                    #                                         arrowstyle="|-|,widthA=0.7, widthB=0.7"
                    #                                     ))
                                                    
                # fig_trace.legend(labels, loc='lower right', bbox_to_anchor=(1,-0.1), ncol=len(labels))#, bbox_transform=fig.transFigure)
                if ntemps>1:
                    cb = fig_trace_beta.colorbar(sc, ax=axes_trace)
                    cb.set_label(label=r"$T$", size=30)#, weight='bold')
                    cb.solids.set(alpha=1)
                    # fig_trace_beta.tight_layout()
                    dist = (vmax-vmin)/(ntemps)/2
                    tick_start = vmin+dist
                    tick_end = vmax-dist
                    tick_locs = np.linspace(tick_start, tick_end, ntemps)[::-1]
                    cb.set_ticks(tick_locs)
                    labels = list(result_["chain.T"][0,:])
                    inf_label = r"$\infty$"
                    labels[-1] = inf_label
                    ticklabels = [str(round(float(label), 1)) if isinstance(label, str)==False else label for label in labels] #round(x, 2)
                    cb.set_ticklabels(ticklabels, size=12)

                    for tick in cb.ax.get_yticklabels():
                        tick.set_fontsize(12)
                        txt = tick.get_text()
                        if txt==inf_label:
                            tick.set_fontsize(20)
                            # tick.set_text()
                            # tick.set_ha("center")
                            # tick.set_va("center_baseline")
                if ii==0:
                    fig_trace_beta.savefig(r'C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\LBNL_trace_plot.png', dpi=300)

            if do_swap_plot and ntemps>1:
                fig_swap, ax_swap = plt.subplots(layout='compressed')
                fig_swap.set_size_inches((17, 12))
                fig_swap.suptitle("Swaps", fontsize=20)
                n = ntemps-1
                for i in range(n):
                    if i==0: #######################################################################
                        ax_swap.plot(range(result_["chain.swaps_accepted"][:,i].shape[0]), result_["chain.swaps_accepted"][:,i]/result_["chain.swaps_proposed"][:,i], color=cm_sb[i])




            if do_corner_plot:
                # fig_corner, axes_corner = plt.subplots(nrows=ndim, ncols=ndim, layout='compressed')
                
                parameter_chain = result_["chain.x"][burnin:,0,:,:]
                parameter_chain = parameter_chain.reshape(parameter_chain.shape[0]*parameter_chain.shape[1],parameter_chain.shape[2])
                fig_corner = corner.corner(parameter_chain, fig=None, labels=flat_attr_list_, labelpad=-0.2, show_titles=True, color=cm_sb[0], plot_contours=True, bins=15, hist_bin_factor=5, max_n_ticks=3, quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 10, "ha": "left", "position": (0.03, 1.01)})
                fig_corner.set_size_inches((12, 12))
                pad = 0.025
                fig_corner.subplots_adjust(left=pad, bottom=pad, right=1-pad, top=1-pad, wspace=0.08, hspace=0.08)
                axes = fig_corner.get_axes()
                for ax in axes:
                    ax.set_xticks([], minor=True)
                    ax.set_xticks([])
                    ax.set_yticks([], minor=True)
                    ax.set_yticks([])

                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])

                median = np.median(parameter_chain, axis=0)
                corner.overplot_lines(fig_corner, median, color=red, linewidth=0.5)
                corner.overplot_points(fig_corner, median.reshape(1,median.shape[0]), marker="s", color=red)
                if ii==0:
                    fig_corner.savefig(r'C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\LBNL_corner_plot.png', dpi=300)
        # color = cm(1)
        # fig_trace_loglike, axes_trace_loglike = plt.subplots(nrows=1, ncols=1)
        # fig_trace_loglike.set_size_inches((17, 12))
        # fig_trace_loglike.suptitle("Trace plots of log likelihoods")
        # vmin = np.nanmin(-chain_logl)
        # vmax = np.nanmax(-chain_logl)
        # for nt in range(1):
        #     for nw in range(nwalkers):
        #         logl = chain_logl[:, nt, nw]
        #         axes_trace_loglike.scatter(range(logl.shape[0]), -logl, color=color, s=4, alpha=0.8)
        # axes_trace_loglike.set_yscale("log")
        # plt.show()
        
    plt.show()


if __name__=="__main__":
    test_load_emcee_chain()