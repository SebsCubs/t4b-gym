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
import json

def run():
    stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2023, month=12, day=2, hour=0, minute=0, second=0,
                                  tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2023, month=12, day=7, hour=0, minute=0, second=0,
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
    

    file_path = os.path.join(uppath(os.path.abspath(__file__), 1), "json_lists_of_parameters.json")

    with open(file_path, 'r') as f:
        data = json.load(f)

    (
    C_wall_list,
    C_wall_list_lb,
    C_wall_list_ub,
    C_boundary_list,
    C_boundary_list_lb,
    C_boundary_list_ub,
    C_air_list,
    C_air_list_lb,
    C_air_list_ub,
    R_out_list,
    R_out_list_lb,
    R_out_list_ub,
    R_in_list,
    R_in_list_lb,
    R_in_list_ub,
    R_boundary_list,
    R_boundary_list_lb,
    R_boundary_list_ub,
    f_wall_list,
    f_wall_list_lb,
    f_wall_list_ub,
    f_air_list,
    f_air_list_lb,
    f_air_list_ub,
    infiltration_list,
    infiltration_list_lb,
    infiltration_list_ub,
    T_boundary_list,
    T_boundary_list_lb,
    T_boundary_list_ub,
    Q_occ_gain_list,
    Q_occ_gain_list_lb,
    Q_occ_gain_list_ub
    ) = (
    data["C_wall_list"],
    data["C_wall_list_lb"],
    data["C_wall_list_ub"],
    data["C_boundary_list"],
    data["C_boundary_list_lb"],
    data["C_boundary_list_ub"],
    data["C_air_list"],
    data["C_air_list_lb"],
    data["C_air_list_ub"],
    data["R_out_list"],
    data["R_out_list_lb"],
    data["R_out_list_ub"],
    data["R_in_list"],
    data["R_in_list_lb"],
    data["R_in_list_ub"],
    data["R_boundary_list"],
    data["R_boundary_list_lb"],
    data["R_boundary_list_ub"],
    data["f_wall_list"],
    data["f_wall_list_lb"],
    data["f_wall_list_ub"],
    data["f_air_list"],
    data["f_air_list_lb"],
    data["f_air_list_ub"],
    data["infiltration_list"],
    data["infiltration_list_lb"],
    data["infiltration_list_ub"],
    data["T_boundary_list"],
    data["T_boundary_list_lb"],
    data["T_boundary_list_ub"],
    data["Q_occ_gain_list"],
    data["Q_occ_gain_list_lb"],
    data["Q_occ_gain_list_ub"],
    )


    targetParameters = {"private": {"C_wall": {"components": spaces_list, "x0": list(C_wall_list), "lb": list(C_wall_list_lb), "ub": list(C_wall_list_ub)},
                                    "C_air": {"components": spaces_list, "x0": list(C_air_list), "lb": list(C_air_list_lb), "ub": list(C_air_list_ub)},
                                    "C_boundary": {"components": spaces_list, "x0": list(C_boundary_list), "lb": list(C_boundary_list_lb), "ub": list(C_boundary_list_ub)},
                                    "R_out": {"components": spaces_list, "x0": list(R_out_list), "lb": list(R_out_list_lb), "ub": list(R_out_list_ub)},
                                    "R_in": {"components": spaces_list, "x0": list(R_in_list), "lb": list(R_in_list_lb), "ub": list(R_in_list_ub)},
                                    "R_boundary": {"components": spaces_list, "x0": list(R_boundary_list), "lb": list(R_boundary_list_lb), "ub": list(R_boundary_list_ub)},
                                    "f_wall": {"components": spaces_list, "x0": list(f_wall_list), "lb": list(f_wall_list_lb), "ub": list(f_wall_list_ub)},
                                    "f_air": {"components": spaces_list, "x0": list(f_air_list), "lb": list(f_air_list_lb), "ub": list(f_air_list_ub)},
                                    "kp": {"components": heating_controller_list, "x0": 0.001, "lb": 1e-5, "ub": 3},
                                    "Ti": {"components": heating_controller_list, "x0": 3, "lb": 1e-5, "ub": 5},
                                    "m_flow_nominal": {"components": space_heater_valve_list, "x0": 0.02, "lb": 1e-3, "ub": 0.1}, #0.0202
                                    "infiltration": {"components": spaces_list, "x0": list(infiltration_list), "lb": list(infiltration_list_lb), "ub": list(infiltration_list_ub)},
                                    "Q_occ_gain": {"components": spaces_list, "x0": Q_occ_gain_list, "lb": Q_occ_gain_list_lb, "ub": Q_occ_gain_list_ub},
                                    "T_boundary": {"components": spaces_list, "x0": list(T_boundary_list), "lb": list(T_boundary_list_lb), "ub": list(T_boundary_list_ub)}
                                    },
                        "shared": {"C_int": {"components": spaces_list, "x0": 25006.836341524, "lb": 12503.418170762, "ub": 50013.672683048},
                                    "R_int": {"components": spaces_list, "x0": 0.0004338679264954746, "lb": 0.0002169339632477373, "ub": 0.0008677358529909492},
                                    "a": {"components": damper_list, "x0": 2, "lb": 0.5, "ub": 8},
                                    "dpFixed_nominal": {"components": space_heater_valve_list, "x0": 2000, "lb": 0, "ub": 10000}
                        }}

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
    

    # loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240808_152120.npz" #After new x0
    # model.load_estimation_result(loaddir)
    
    estimator = tb.Estimator(model)
    estimator.chain_savedir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "models", "model_1st_floor", "model_parameters", "estimation_results", "chain_logs", "20240814_061525.npz")
    model.load_estimation_result(estimator.chain_savedir)

    # Options for the PTEMCEE estimation method. If the options argument is not supplied or None is supplied, default options are applied.  
    options = {"n_sample": 10000, #500 #This is a test file, and we therefore only sample 2. Typically, we need at least 1000 samples before the chain converges. 
                "n_temperature": 1, #20 #Number of parallel chains/temperatures.
                "fac_walker": 2, #Scaling factor for the number of ensemble walkers per chain. This number is multiplied with the number of estimated to get the number of ensemble walkers per chain. Minimum is 2 (required by PTEMCEE).
                "prior": "uniform", #Prior distribution - "gaussian" is also implemented
                "walker_initialization": "sample_hypercube", #sample 
                "add_gp": True,
                "gp_input_type": "closest",
                "gp_add_time": False,
                "gp_max_inputs": 7,
                "maxtasksperchild": 30,
                "n_save_checkpoint": 30,
                # "n_cores": 1
                }
    estimator.estimate(targetParameters=targetParameters,
                        targetMeasuringDevices=targetMeasuringDevices,
                        startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize,
                        n_initialization_steps=188,
                        method="MCMC",
                        options=options,
                        )
    # estimator.chain_savedir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "models", "test_estimator_wbypass", "model_parameters", "estimation_results", "chain_logs", "20240307_130004.pickle")
    
    if add_gp:
        model.load_estimation_result(estimator.chain_savedir)
        options = {"n_sample": 1000, #This is a test file, and we therefore only sample 2. Typically, we need at least 1000 samples before the chain converges. 
                    "n_temperature": 5, #Number of parallel chains/temperatures.
                    "fac_walker": 2, #Scaling factor for the number of ensemble walkers per chain. This number is multiplied with the number of estimated to get the number of ensemble walkers per chain. Minimum is 2 (required by PTEMCEE).
                    "prior": "uniform", #Prior distribution - "gaussian" is also implemented
                    "model_walker_initialization": "sample", 
                    "noise_walker_initialization": "uniform",
                    "add_gp": True,
                    }
        estimator.estimate(targetParameters=targetParameters,
                            targetMeasuringDevices=targetMeasuringDevices,
                            startTime=startTime,
                            endTime=endTime,
                            stepSize=stepSize,
                            n_initialization_steps=100,
                            method="MCMC",
                            options=options #
                            )

        model.load_estimation_result(estimator.chain_savedir)
        options = {"n_sample": 20000, #This is a test file, and we therefore only sample 2. Typically, we need at least 1000 samples before the chain converges. 
                    "n_temperature": 1, #Number of parallel chains/temperatures.
                    "fac_walker": 4, #Scaling factor for the number of ensemble walkers per chain. This number is multiplied with the number of estimated to get the number of ensemble walkers per chain. Minimum is 2 (required by PTEMCEE).
                    "prior": "uniform", #Prior distribution - "gaussian" is also implemented
                    "walker_initialization": "sample_hypercube", 
                    "add_gp": True,
                    }
        estimator.estimate(targetParameters=targetParameters,
                            targetMeasuringDevices=targetMeasuringDevices,
                            startTime=startTime,
                            endTime=endTime,
                            stepSize=stepSize,
                            n_initialization_steps=100,
                            method="MCMC",
                            options=options #
                            )
        
if __name__ == "__main__":
    run()