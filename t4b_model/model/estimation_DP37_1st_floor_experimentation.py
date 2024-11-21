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
from model_DP37_1st_floor import get_model


def test_load_emcee_chain():
    stepSize = 600
    startTime_test1 = datetime.datetime(year=2023, month=12, day=2, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test1 = datetime.datetime(year=2023, month=12, day=12, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test = [startTime_test1]
    endTime_test = [endTime_test1]
    stepSize_test = [stepSize]

     
    model = get_model()
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/chain_logs/20240624_144846.pickle"
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240627_061702.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240701_153927.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240702_100055.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240702_154124.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240702_161942.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240703_102029.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240703_185538.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240703_202739.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240703_185538.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240703_205942.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240703_205942.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240706_121008.pickle"


    model.load_estimation_result(loaddir)

    ################
    result = model.chain_log
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
    print("-----------------")
    print(result["chain.x"].shape)
    print(result["chain.x"][0,0,0,:])
    for v, attr in zip(result["chain.x"][0,0,0,:], attr_list):
        print(attr, v)
    model.chain_log = result
    #####################



    percentile = 2
    targetMeasuringDevices = {  model.component_dict["007A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["007A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["007A_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["007A_co2_sensor"]: {"standardDeviation": 50/percentile, "scale_factor": 1},
                                model.component_dict["008A_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["011A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["011A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["011A_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["011A_co2_sensor"]: {"standardDeviation": 50/percentile, "scale_factor": 1},
                                model.component_dict["012A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["012A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["012A_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["012A_co2_sensor"]: {"standardDeviation": 50/percentile, "scale_factor": 1},
                                model.component_dict["013A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["013A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["013A_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["013A_co2_sensor"]: {"standardDeviation": 50/percentile, "scale_factor": 1},
                                model.component_dict["015A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["015A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["015A_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["015A_co2_sensor"]: {"standardDeviation": 50/percentile, "scale_factor": 1},
                                model.component_dict["020A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["020A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["020A_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["020A_co2_sensor"]: {"standardDeviation": 50/percentile, "scale_factor": 1},
                                model.component_dict["020B_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["020B_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["020B_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["020B_co2_sensor"]: {"standardDeviation": 50/percentile, "scale_factor": 1},
                                model.component_dict["029A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["029A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["029A_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["029A_co2_sensor"]: {"standardDeviation": 50/percentile, "scale_factor": 1},
                                model.component_dict["031A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["031A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["031A_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["031A_co2_sensor"]: {"standardDeviation": 50/percentile, "scale_factor": 1},
                                model.component_dict["033A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["033A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["033A_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["033A_co2_sensor"]: {"standardDeviation": 50/percentile, "scale_factor": 1},
                                model.component_dict["035A_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["035A_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                model.component_dict["035A_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["035A_co2_sensor"]: {"standardDeviation": 50/percentile, "scale_factor": 1}}
    


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





    spaces_list_outdoor = spaces_list.copy()
    spaces_list_outdoor.remove(space_020A)
    spaces_list_outdoor.remove(space_020B)

    # spaces_list_space_heater = spaces_list.copy()
    # spaces_list_space_heater.remove(space_008A)

    airVolumes = np.array([s.airVolume for s in spaces_list])
    airVolumes_outdoor = np.array([s.airVolume for s in spaces_list_outdoor])
    height = 2.7
    floorAreas = airVolumes/height
    floorAreas_outdoor = airVolumes_outdoor/height
    thickness = 0.3
    specific_heat_capacity_wall = 1e+6 #J/m3
    wallAreas = (floorAreas_outdoor)**(0.5)*height
    C_wall_x0 = wallAreas*thickness*specific_heat_capacity_wall
    furniture_mass = 30*2#kg/m2
    specific_heat_capacity_furniture = 2000
    C_air_x0 = airVolumes*1000+furniture_mass*floorAreas*specific_heat_capacity_furniture
    C_boundary_x0 = floorAreas*specific_heat_capacity_wall*thickness/3

    U_ins = 0.18 #W/m2K
    R_out_x0 = (1/(wallAreas*U_ins))/2
    R_in_x0 = (1/(wallAreas*U_ins))/2

    infiltrations_x0 = airVolumes*0.3/space_007A.airVolume

    print("C_wall_x0: ", C_wall_x0)
    print("C_air_x0: ", C_air_x0)
    print("C_boundary_x0: ", C_boundary_x0)
    print("R_out_x0: ", R_out_x0)
    print("R_in_x0: ", R_in_x0)


    aaa

    

    targetParameters = {"private": {"C_wall": {"components": space_007A, "x0": 1, "lb": 0, "ub": 2},
                                    }}

    C = 10000
    options = {"n_sample": 2, #500 #This is a test file, and we therefore only sample 2. Typically, we need at least 1000 samples before the chain converges. 
                "n_temperature": 1, #20 #Number of parallel chains/temperatures.
                "fac_walker": 2, #Scaling factor for the number of ensemble walkers per chain. This number is multiplied with the number of estimated to get the number of ensemble walkers per chain. Minimum is 2 (required by PTEMCEE).
                "prior": "uniform", #Prior distribution - "gaussian" is also implemented
                "walker_initialization": "gaussian", #Initialization of parameters - "gaussian" is also implemented
                "add_gp": False,
                "maxtasksperchild": 30,
                "n_save_checkpoint": 1,
                "n_cores": 2
                }
    
    print("-----------------------------")
    x = result["chain.x"]
    
    for i, (attr, v) in enumerate(zip(attr_list, x[0,0,0,:])):
        print(i, attr, v)
    estimator = tb.Estimator(model)
    estimator.estimate(targetParameters=targetParameters,
                            targetMeasuringDevices=targetMeasuringDevices,
                            startTime=startTime_test,
                            endTime=endTime_test,
                            stepSize=stepSize,
                            n_initialization_steps=100,
                            method="MCMC",
                            options=options) #)


    x[0,0,0,123] = 0.005
    logl = estimator._loglike_test(x[0,0,0,:])
    print("loglike: ", logl)
    print("-------------0.005----------------")
    for key, v in estimator.loglike_dict.items():
        print(key, v)

    x[0,0,0,123] = 0.01
    logl = estimator._loglike_test(x[0,0,0,:])
    print("loglike: ", logl)
    print("-------------0.01----------------")
    for key, v in estimator.loglike_dict.items():
        print(key, v)

    x[0,0,0,123] = 0.05
    logl = estimator._loglike_test(x[0,0,0,:])
    print("-------------0.05----------------")
    print("loglike: ", logl)
    for key, v in estimator.loglike_dict.items():
        print(key, v)

    x[0,0,0,123] = 0.285
    logl = estimator._loglike_test(x[0,0,0,:])
    print("loglike: ", logl)
    print("-------------0.285----------------")
    for key, v in estimator.loglike_dict.items():
        print(key, v)

    x[0,0,0,123] = 0.5
    logl = estimator._loglike_test(x[0,0,0,:])
    print("loglike: ", logl)
    print("-------------0.285----------------")
    for key, v in estimator.loglike_dict.items():
        print(key, v)

    model.chain_log["chain.x"] = x
    simulator = tb.Simulator(model)
    print(model.chain_log["chain.x"].shape)


    # plot.trace_plot(model=model, n_subplots=15, save_plot=True, max_cols = 3)
    # plot.corner_plot(model=model, subsample_factor=20, save_plot=True, param_blocks=10, burnin=800)
    fig, axes = simulator.run_emcee_inference(model, None, targetMeasuringDevices, startTime_test, endTime_test, stepSize_test, assume_uncorrelated_noise=True, burnin=0, show=True, single_plot=True)


if __name__=="__main__":
    test_load_emcee_chain()