import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Mac specific?
import datetime
import pandas as pd
from dateutil.tz import gettz
import sys
import george

# Only for testing before distributing package
if __name__ == '__main__':
    file_path = r"C:\Users\jabj\Documents\python\Twin4Build"
    sys.path.append(file_path)
import twin4build as tb
import twin4build.utils.plot.plot as plot
from twin4build.utils.uppath import uppath
import numpy as np
from model_DP37_020B import get_model
import matplotlib.pyplot as plt

def run():
    stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
                                  tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2023, month=12, day=7, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = get_model()
    space = model.component_dict["[020B][020B_space_heater]"]
    heating_controller = model.component_dict["020B_temperature_heating_controller"]
    space_heater_valve = model.component_dict["020B_space_heater_valve"]
    supply_damper = model.component_dict["020B_room_supply_damper"]
    exhaust_damper = model.component_dict["020B_room_exhaust_damper"]

    targetParameters = {"private": {"C_wall": {"components": [space], "x0": 1.5e+6, "lb": 1e+6, "ub": 15e+6}, #1.5e+6
                                    "C_air": {"components": [space], "x0": 3e+6, "lb": 1e+4, "ub": 15e+6}, #3e+6
                                    "C_boundary": {"components": [space], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6}, #1e+5
                                    "R_out": {"components": [space], "x0": 0.2, "lb": 1e-3, "ub": 0.5}, #0.2
                                    "R_in": {"components": [space], "x0": 0.2, "lb": 1e-3, "ub": 0.5}, #0.2
                                    "R_boundary": {"components": [space], "x0": 0.005, "lb": 1e-3, "ub": 0.3}, #0.005
                                    "f_wall": {"components": [space], "x0": 1, "lb": 0, "ub": 2}, #1
                                    "f_air": {"components": [space], "x0": 1, "lb": 0, "ub": 2}, #1
                                    "kp": {"components": [heating_controller], "x0": 1e-3, "lb": 1e-6, "ub": 3}, #1e-3
                                    "Ti": {"components": [heating_controller], "x0": 3, "lb": 1e-5, "ub": 10}, #3
                                    "m_flow_nominal": {"components": [space_heater_valve], "x0": 0.0202, "lb": 1e-3, "ub": 0.5}, #0.0202
                                    "dpFixed_nominal": {"components": [space_heater_valve], "x0": 2000, "lb": 0, "ub": 10000}, #2000
                                    "T_boundary": {"components": [space], "x0": 20, "lb": 19, "ub": 24}, #20
                                    "a": {"components": [supply_damper, exhaust_damper], "x0": 2, "lb": 0.5, "ub": 8}, #2
                                    "infiltration": {"components": [space], "x0": 0.001, "lb": 1e-4, "ub": 0.3}, #0.001
                                    "Q_occ_gain": {"components": [space], "x0": 100, "lb": 10, "ub": 1000}, #100
                                    }}
    percentile = 2
    targetMeasuringDevices = {model.component_dict["020B_valve_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                model.component_dict["020B_temperature_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                                model.component_dict["020B_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                                model.component_dict["020B_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                }
    
    # Options for the MCMC estimation method. If the options argument is not supplied or None is supplied, default options are applied.  
    options = {"n_sample": 20, #500 #This is a test file, and we therefore only sample 2. Typically, we need at least 1000 samples before the chain converges. 
                "n_temperature": 1, #20 #Number of parallel chains/temperatures.
                "fac_walker": 8, #Scaling factor for the number of ensemble walkers per chain. This number is multiplied with the number of estimated to get the number of ensemble walkers per chain. Minimum is 2 (required by PTEMCEE).
                "prior": "uniform", #Prior distribution - "gaussian" is also implemented
                "walker_initialization": "gaussian", #Initialization of parameters - "gaussian" is also implemented
                "add_gp": False,
                "n_cores":6,
                "n_save_checkpoint": 1
                }
    estimator = tb.Estimator(model)
    estimator.estimate(targetParameters=targetParameters,
                        targetMeasuringDevices=targetMeasuringDevices,
                        startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize,
                        n_initialization_steps=288,
                        method="LS",
                        options=options #
                        )
    model.load_estimation_result(estimator.result_savedir_pickle)



    # filename = r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\Twin4build-Case-Studies\DP37\model\generated_files\models\model_020B\model_parameters\estimation_results\LS_result\20241016_105247_ls.pickle"
    # model.load_estimation_result(filename)


    print("SOLUTION")
    print("C_wall: ", space.C_wall)
    print("C_air: ", space.C_air)
    print("C_boundary: ", space.C_boundary)
    print("R_out: ", space.R_out)
    print("R_in: ", space.R_in)
    print("R_boundary: ", space.R_boundary)
    print("f_wall: ", space.f_wall)
    print("f_air: ", space.f_air)
    print("Q_occ_gain: ", space.Q_occ_gain)
    print("kp: ", heating_controller.kp)
    print("Ti: ", heating_controller.Ti)
    print("m_flow_nominal: ", space_heater_valve.m_flow_nominal)
    print("dpFixed_nominal: ", space_heater_valve.dpFixed_nominal)
    print("T_boundary: ", space.T_boundary)
    print("a: ", supply_damper.a)
    print("a: ", exhaust_damper.a)
    print("infiltration: ", space.infiltration)
    res = estimator._res_fun_ls(model.result["result.x"]).reshape((estimator.n_timesteps, len(estimator.targetMeasuringDevices)))
    print(np.sum(res**2, axis=0))
    print("res: ", res)
    aa
    # monitor = tb.Monitor(model) #Compares the simulation results with the expected results
    # monitor.monitor(startTime=startTime,
    #                     endTime=endTime,
    #                     stepSize=stepSize,
    #                     show=False)
    

    simulator = tb.Simulator(model)
    simulator.simulate(model,
                       startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize)

    def generate_data(t, measuring_device):
        N = t.size
        gp = george.GP(0.1 * george.kernels.ExpSquaredKernel(3.3))
        y = gp.sample(t)
        y += np.array(next(iter(model.component_dict[measuring_device].savedOutput.values())))
        yerr = 0.05 + 0.05 * np.random.rand(N)
        y += yerr * np.random.randn(N)
        return t, y, yerr
    
    def save_data(t, measuring_device, path):
        path_ = os.path.join(path, f"{measuring_device}.csv")
        df = pd.DataFrame({"time": t, "value": np.array(next(iter(model.component_dict[measuring_device].savedOutput.values())))})
        df.set_index("time", inplace=True)
        df.to_csv(path_)

    measuring_devices = ["020B_valve_position_sensor", "020B_temperature_sensor", "020B_co2_sensor", "020B_damper_position_sensor", "020B_temperature_heating_setpoint", "BTA004"]
    for measuring_device in measuring_devices:
        # t, y, yerr = generate_data(np.array(simulator.secondTimeSteps), measuring_device.id)
        save_data(simulator.dateTimeSteps, measuring_device, r"C:\Users\jabj\OneDrive - Syddansk Universitet\PhD_Project_Jakob\Twin4build\python\BuildingEnergyModel\Twin4build-Case-Studies\DP37\model\example_data")

        # plt.figure()
        # plt.plot(t, y, label=measuring_device.id + " - GENERATED")
        # plt.plot(t, next(iter(model.component_dict[measuring_device.id].savedOutput.values())), label=measuring_device.id + " - TRUTH")
    
    # plt.legend()
    # plt.show()

    startTime_test = [datetime.datetime(year=2024, month=1, day=5, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))]
    endTime_test = [datetime.datetime(year=2024, month=1, day=12, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))]
    stepSize_test = [600]


    # simulator = tb.Simulator(model)
    # result = simulator.bayesian_inference(model, startTime_test, endTime_test, stepSize_test, targetMeasuringDevices=targetMeasuringDevices, n_initialization_steps=188, assume_uncorrelated_noise=True, burnin=1, n_cores=1, n_samples_max=1)
    # estimator.chain_savedir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "models", "test_estimator_wbypass", "model_parameters", "estimation_results", "chain_logs", "20240307_130004.pickle")

    # ylabels = {}
    # for c in targetMeasuringDevices:
    #     id = c.id
    #     if "valve" in id.lower():
    #         ylabels[id] = r"$u_v$ [1]"
    #     elif "temperature" in id.lower():
    #         ylabels[id] = r"$T_z$ [$^\circ$C]"
    #     elif "damper" in id.lower():
    #         ylabels[id] = r"$u_d$ [1]"
    #     elif "co2" in id.lower():
    #         ylabels[id] = r"$C_z$ [ppm]"
    # fig, axes = plot.plot_bayesian_inference(result["values"], result["time"], result["ydata"], show=True, single_plot=False, save_plot=True, addmodel=True, addmodelinterval=False, addnoisemodel=False, addnoisemodelinterval=False, addMetrics=False, summarizeMetrics=False, ylabels=ylabels)
    
    
if __name__ == "__main__":
    run()