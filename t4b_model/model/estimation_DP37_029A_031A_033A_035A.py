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
from model_DP37_029A_031A_033A_035A import get_model


def run():
    stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2023, month=11, day=27, hour=0, minute=0, second=0,
                                  tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2023, month=12, day=7, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    model = get_model()
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


    targetParameters = {"private": {"C_wall": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+7},
                                    "C_air": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+7},
                                    "C_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                    "R_out": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.1, "lb": 1e-2, "ub": 0.2},
                                    "R_in": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.1, "lb": 1e-2, "ub": 0.2},
                                    "R_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.1, "lb": 1e-2, "ub": 0.2},
                                    "f_wall": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1, "lb": 0, "ub": 3},
                                    "f_air": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1, "lb": 0, "ub": 3},
                                    "m_flow_nominal_sh": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.0202, "lb": 1e-3, "ub": 0.1},
                                    "Q_flow_nominal_sh": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1000, "lb": 100, "ub": 5000},
                                    "Kp": {"components": [heating_controller_029A, heating_controller_031A, heating_controller_033A, heating_controller_035A], "x0": 2e-4, "lb": 1e-5, "ub": 3},
                                    "Ti": {"components": [heating_controller_029A, heating_controller_031A, heating_controller_033A, heating_controller_035A], "x0": 3e-1, "lb": 1e-5, "ub": 3},
                                    "m_flow_nominal": {"components": [space_heater_valve_029A, space_heater_valve_031A, space_heater_valve_033A, space_heater_valve_035A], "x0": 0.0202, "lb": 1e-3, "ub": 0.1}, #0.0202
                                    },
                        "shared": {"C_int": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 1e+5, "lb": 1e+4, "ub": 1e+6},
                                    "R_int": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 0.1, "lb": 1e-2, "ub": 0.2},
                                    "a": {"components": [supply_damper_029A, exhaust_damper_029A, supply_damper_031A, exhaust_damper_031A, supply_damper_033A, exhaust_damper_033A, supply_damper_035A, exhaust_damper_035A], "x0": 5, "lb": 0.5, "ub": 8},
                                    "T_boundary": {"components": [space_029A, space_031A, space_033A, space_035A], "x0": 20, "lb": 19, "ub": 23},
                                    "dpFixed_nominal": {"components": [space_heater_valve_029A, space_heater_valve_031A, space_heater_valve_033A, space_heater_valve_035A], "x0": 1e-6, "lb": 0, "ub": 10000}
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
    
    # Options for the PTEMCEE estimation method. If the options argument is not supplied or None is supplied, default options are applied.  
    options = {"n_sample": 10000, #500 #This is a test file, and we therefore only sample 2. Typically, we need at least 1000 samples before the chain converges. 
                "n_temperature": 2, #20 #Number of parallel chains/temperatures.
                "fac_walker": 6, #Scaling factor for the number of ensemble walkers per chain. This number is multiplied with the number of estimated to get the number of ensemble walkers per chain. Minimum is 2 (required by PTEMCEE).
                "prior": "uniform", #Prior distribution - "gaussian" is also implemented
                "walker_initialization": "uniform", #Initialization of parameters - "gaussian" is also implemented
                "add_gp": False,
                # "n_cores": 1
                }
    estimator = tb.Estimator(model)
    estimator.estimate(targetParameters=targetParameters,
                        targetMeasuringDevices=targetMeasuringDevices,
                        startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize,
                        n_initialization_steps=288,
                        method="MCMC",
                        options=options #
                        )
    # estimator.chain_savedir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "models", "test_estimator_wbypass", "model_parameters", "estimation_results", "chain_logs", "20240307_130004.pickle")
    
    add_gp = False
    if add_gp:
        model.load_estimation_result(estimator.chain_savedir)
        options = {"n_sample": 250, #This is a test file, and we therefore only sample 2. Typically, we need at least 1000 samples before the chain converges. 
                    "n_temperature": 5, #Number of parallel chains/temperatures.
                    "fac_walker": 2, #Scaling factor for the number of ensemble walkers per chain. This number is multiplied with the number of estimated to get the number of ensemble walkers per chain. Minimum is 2 (required by PTEMCEE).
                    "prior": "uniform",
                    "model_walker_initialization": "sample", #Prior distribution - "gaussian" is also implemented
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
                    "prior": "uniform",
                    "walker_initialization": "sample_hypercube", #Prior distribution - "gaussian" is also implemented
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