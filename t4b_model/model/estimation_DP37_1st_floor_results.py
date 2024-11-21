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
from twin4build.utils.bayesian_inference import get_iac, get_iac_old
import pandas as pd


def test_load_emcee_chain():
    stepSize = 600
    startTime_test1 = datetime.datetime(year=2024, month=1, day=5, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test1 = datetime.datetime(year=2024, month=1, day=12, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test = [startTime_test1]
    endTime_test = [endTime_test1]
    stepSize_test = [stepSize]


    
    model = get_model()
    # model = tb.Model(id="model_1st_floor", saveSimulationResult=True)
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
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240706_142732.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240706_212154.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240707_192042.pickle"
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_065231.pickle" #New 020A and 020B with outdoor connection
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_071740.pickle" #New 020A and 020B with outdoor connection

    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_075837.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_091839.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_094033.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_100526.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_105155.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_111642.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_121946.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_130532.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_133612.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_135547.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_141501.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_150143.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_152033.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_154736.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_160358.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_161734.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_195622.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240708_213036.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240711_102339.pickle" #New 020A and 020B with outdoor connection
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240715_002326.pickle" #After new x0
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240717_074737.pickle" #After new x0
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240718_150520.pickle" #GP
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240719_125459.pickle" #New bounds
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240720_133602.pickle" #GP
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240722_132853.npz" #GP
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240723_200131.npz" #GP
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240724_103611.npz" #GP
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240724_153956.npz" #GP
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240724_185556.npz" #GP
    
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240724_215136.npz" #GP good
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240728_144747.pickle" #GP New
    # loaddir = r"D:/Twin4Build/RemoteResults/chain_logs/1st Floor/20240728_144747.npz"
    #loaddir = r"D:/Twin4Build/RemoteResults/chain_logs/1st Floor/20240724_215136.npz"
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240805_073252.npz" #GP New
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240806_122304.npz" #GP New #
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240807_063857.npz" #GP New #
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240807_125121.npz" #GP New #
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240807_205408.npz" #GP New #
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240808_085154.npz" #GP New #
    # # loaddir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "models", "model_1st_floor", "model_parameters", "estimation_results", "chain_logs", "20240724_215136.npz")
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240808_152120.npz" #GP New #
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240810_125036.npz" #GP New #
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240814_061525.npz" #GP New #
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240815_070728.npz" #GP New #
    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240819_123307.npz" #GP New #

    
    model.load_estimation_result(loaddir)

    

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

    ##########################################################
    # result = model.chain_log
    # logl = result["chain.logl"]
    # logl[np.abs(logl)>1e+9] = np.nan
    # print(logl[:,0,:].max())
    # indices = np.where(logl[:,0,:] == np.nanmax(logl[:,0,:]))
    # print(indices)
    # s0 = indices[0][0]
    # s1 = indices[1][0]
    # a = result["chain.x"][s0, 0, s1, :]
    # a = np.resize(a, (1,1,1,a.shape[0]))
    # result["chain.x"] = a
    # # for key in result.keys():
    # #     # if key not in list_:
    # #     if isinstance(result[key], list):
    # #         result[key] = np.array(result[key])
    # print(result["chain.x"].shape)
    # records_array = np.array(result["theta_mask"])
    # vals, inverse, count = np.unique(records_array, return_inverse=True,
    #                           return_counts=True)
    # idx_vals_repeated = np.where(count > 1)[0]
    # vals_repeated = vals[idx_vals_repeated]
    # rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
    # _, inverse_rows = np.unique(rows, return_index=True)
    # res = np.split(cols, inverse_rows[1:])
    # d_idx = []
    # for i in res:
    #     d_idx.extend(list(i[1:]))
    # attr_list = np.array(result["component_attr"])
    # attr_list = np.delete(attr_list, np.array(d_idx).astype(int)) #res is an array of duplicates, so its size should always be larger than 1
    # id_list = np.array(result["component_id"])
    # id_list = np.delete(id_list, np.array(d_idx).astype(int))
    # print("-----------------")
    # print(result["chain.x"].shape)
    # print(result["chain.x"][0,0,0,:])
    # for v, attr, id in zip(result["chain.x"][0,0,0,:len(attr_list)], attr_list, id_list):
    #     print(id, attr, v)
    # model.chain_log = result
    ##########################################################
 
    subset = ["[012A][012A_space_heater]"]
    # chain = model.chain_log["chain.x"]
    # chain = chain[:,0,:,0].reshape((chain.shape[0],1,chain.shape[2],1))
    # iac,idx = get_iac_old(chain, interval=10)
    # plt.plot(idx, iac[:,0,0])
    # plt.show()
    # plot.logl_plot(model, show=True)
    # plot.trace_plot(model, save_plot=False, subset=subset, show=True, plot_title = "Trace Plot <Space 4>", burnin = 0)
    labels = [r"$C_{w}$", r"$C_a$", r"$C_{bou}$", r"$R_{out}$", r"$R_{in}$", r"$R_{bou}$", r"$f_{w}$", r"$f_{a}$", r"$\dot{m}_{inf}$", r"$\dot{Q}_{occ,gain}$", r"$T_{bou}$", r"$C_{int}$", r"$R_{int}$"]
    # labels = None
    # plot.corner_plot(model, subset=subset, burnin=1000, save_plot=True, show=False, labels=labels)

    
    startTime_train = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_train = datetime.datetime(year=2024, month=1, day=4, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    # model.chain_log["startTime_train"] = [startTime_train]
    # model.chain_log["endTime_train"] = [endTime_train]
    simulator = tb.Simulator(model)


    # ylabels = [r"$u_v [1]$", r"$T_z [^{\circ}]$C", r"$u_d [1]$", r"$C_z$ [ppm]"]
    result = simulator.bayesian_inference(model, startTime_test, endTime_test, stepSize_test, targetMeasuringDevices=targetMeasuringDevices, n_initialization_steps=188, assume_uncorrelated_noise=False, burnin=1000, n_cores=1, n_samples_max=1)
    
    subsets = [["007A_valve_position_sensor", "007A_temperature_sensor", "007A_damper_position_sensor", "007A_co2_sensor"],
        ["008A_damper_position_sensor"],
        ["011A_valve_position_sensor", "011A_temperature_sensor", "011A_damper_position_sensor", "011A_co2_sensor"],
        ["012A_valve_position_sensor", "012A_temperature_sensor", "012A_damper_position_sensor", "012A_co2_sensor"],
        ["013A_valve_position_sensor", "013A_temperature_sensor", "013A_damper_position_sensor", "013A_co2_sensor"],
        ["015A_valve_position_sensor", "015A_temperature_sensor", "015A_damper_position_sensor", "015A_co2_sensor"],
        ["020A_valve_position_sensor", "020A_temperature_sensor", "020A_damper_position_sensor", "020A_co2_sensor"],
        ["020B_valve_position_sensor", "020B_temperature_sensor", "020B_damper_position_sensor", "020B_co2_sensor"],
        ["029A_valve_position_sensor", "029A_temperature_sensor", "029A_damper_position_sensor", "029A_co2_sensor"],
        ["031A_valve_position_sensor", "031A_temperature_sensor", "031A_damper_position_sensor", "031A_co2_sensor"],
        ["033A_valve_position_sensor", "033A_temperature_sensor", "033A_damper_position_sensor", "033A_co2_sensor"],
        ["035A_valve_position_sensor", "035A_temperature_sensor", "035A_damper_position_sensor", "035A_co2_sensor"]]
    

    ylabels = {}
    for c in targetMeasuringDevices:
        id = c.id
        if "valve" in id.lower():
            ylabels[id] = r"$u_v$ [1]"
        elif "temperature" in id.lower():
            ylabels[id] = r"$T_z$ [$^\circ$C]"
        elif "damper" in id.lower():
            ylabels[id] = r"$u_d$ [1]"
        elif "co2" in id.lower():
            ylabels[id] = r"$C_z$ [ppm]"

    
    def ViewSubset(subset = subsets, maketable: bool = True, sortbySensorType = True, save_plot: bool = False, show_plot: bool = False, tableCSVname: str = "Table1.csv"):
        if maketable:
            metricsList = []
            if sortbySensorType:
                # ValveList = [{'ID': 'Valve_Position_sensor', 'MAE': "", 'RMSE': "", 'PI95': "", 'PI90':"", 'PI85': "", 'PI80': "", 'PI75': "", 'PI70': "", 'PI65': "", 'PI60': "", 'PI55': "", 'PI50': ""}]
                # DamperList = [{'ID': 'Damper_Position_sensor', 'MAE': "", 'RMSE': "", 'PI95': "", 'PI90':"", 'PI85': "", 'PI80': "", 'PI75': "", 'PI70': "", 'PI65': "", 'PI60': "", 'PI55': "", 'PI50': ""}]
                # TemperaturList = [{'ID': 'Temperature_sensor', 'MAE': "", 'RMSE': "", 'PI95': "", 'PI90':"", 'PI85': "", 'PI80': "", 'PI75': "", 'PI70': "", 'PI65': "", 'PI60': "", 'PI55': "", 'PI50': ""}]
                # Co2List = [{'ID': 'CO2_sensor', 'MAE': "", 'RMSE': "", 'PI95': "", 'PI90':"", 'PI85': "", 'PI80': "", 'PI75': "", 'PI70': "", 'PI65': "", 'PI60': "", 'PI55': "", 'PI50': ""}]

                # sensorLists = [TemperaturList, Co2List, ValveList, DamperList]

                for i,subsets in enumerate(subset):
                    fig, axes, mList = plot.plot_bayesian_inference(result["values"], result["time"], result["ydata"], show=False, subset=subsets, single_plot=False, save_plot=save_plot, addmodel=True, addmodelinterval=False, addnoisemodel=True, addnoisemodelinterval=True, addMetrics=False, summarizeMetrics = True, ylabels=ylabels)
                    
                    
                        # DamperList = [{'ID': 'Damper_Position_sensor', 'MAE': "", 'RMSE': "", 'PI95': "", 'PI90':"", 'PI85': "", 'PI80': "", 'PI75': "", 'PI70': "", 'PI65': "", 'PI60': "", 'PI55': "", 'PI50': ""}]
                        # TemperaturList = [{'ID': 'Temperature_sensor', 'MAE': "", 'RMSE': "", 'PI95': "", 'PI90':"", 'PI85': "", 'PI80': "", 'PI75': "", 'PI70': "", 'PI65': "", 'PI60': "", 'PI55': "", 'PI50': ""}]
                        # Co2List = [{'ID': 'CO2_sensor', 'MAE': "", 'RMSE': "", 'PI95': "", 'PI90':"", 'PI85': "", 'PI80': "", 'PI75': "", 'PI70': "", 'PI65': "", 'PI60': "", 'PI55': "", 'PI50': ""}]

                    for item in mList:
                        if "valve" in item['ID'].lower():
                            if i==0:
                                t = {k: "" for k,v in item.copy().items()}
                                t["ID"] = "Valve_Position_sensor"
                                ValveList = [t]
                                
                            ValveList.append(item)
                        elif "temperature" in item['ID'].lower():
                            if i==0:
                                t = {k: "" for k,v in item.copy().items()}
                                t["ID"] = "Temperature_sensor"
                                TemperaturList = [t]
                            TemperaturList.append(item)
                        elif "damper" in item['ID'].lower():
                            if i==0:
                                t = {k: "" for k,v in item.copy().items()}
                                t["ID"] = "Damper_Position_sensor"
                                DamperList = [t]
                            DamperList.append(item)
                        elif "co2" in item['ID'].lower():
                            if i==0:
                                t = {k: "" for k,v in item.copy().items()}
                                t["ID"] = "CO2_sensor"
                                Co2List = [t]
                            Co2List.append(item)
                    sensorLists = [TemperaturList, Co2List, ValveList, DamperList]

                for sensorList in sensorLists:
                    metricsList += sensorList
            
            else:
                for subsets in subset:
                    fig, axes, mList = plot.plot_bayesian_inference(result["values"], result["time"], result["ydata"], show=False, subset=subset, single_plot=False, save_plot=save_plot, addmodel=False, addmodelinterval=False, addnoisemodel=True, addnoisemodelinterval=True, addMetrics=False, summarizeMetrics=True, ylabels=ylabels)
                    metricsList += mList
            
            df_metrics = pd.DataFrame(metricsList)
            df_metrics.to_csv(tableCSVname, index=False)
        
        else:
            for subsets in subset:
                fig, axes = plot.plot_bayesian_inference(result["values"], result["time"], result["ydata"], show=show_plot, subset=subset, single_plot=False, save_plot=save_plot, addmodel=False, addmodelinterval=False, addnoisemodel=True, addnoisemodelinterval=True, addMetrics=False, summarizeMetrics = False, ylabels=ylabels)
        
        return
    
    ViewSubset(subset=subsets, maketable=True, sortbySensorType=True, tableCSVname="test3.csv", save_plot=True)
    
if __name__=="__main__":
    test_load_emcee_chain()