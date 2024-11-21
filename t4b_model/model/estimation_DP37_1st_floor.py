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



        # spaces_list = [space_007A, space_008A, space_011A, space_012A, space_013A, space_015A, space_020A, space_020B,
                #    space_029A, space_031A, space_033A, space_035A]
    coeffs_R_out_in = {space: 1 for space in spaces_list}
    coeffs_R_out_in[space_007A] = 0.1
    coeffs_R_out_in[space_011A] = 0.1
    coeffs_R_out_in[space_012A] = 0.1
    coeffs_R_out_in[space_013A] = 0.1
    coeffs_R_out_in[space_015A] = 0.1
    coeffs_R_out_in[space_020A] = 0.1
    coeffs_R_out_in[space_020B] = 0.1
    coeffs_R_out_in[space_029A] = 0.1
    coeffs_R_out_in[space_031A] = 0.1
    coeffs_R_out_in[space_033A] = 0.1
    coeffs_R_out_in[space_035A] = 0.1
    coeffs_R_out_in = np.array([coeffs_R_out_in[space] for space in spaces_list])


    coeffs_R_boundary = {space: 1 for space in spaces_list}
    coeffs_R_boundary[space_007A] = 0.
    coeffs_R_boundary[space_008A] = 0.1
    coeffs_R_boundary[space_011A] = 0.05
    coeffs_R_boundary[space_012A] = 0.1
    coeffs_R_boundary[space_013A] = 0.1
    coeffs_R_boundary[space_015A] = 0.1
    coeffs_R_boundary[space_020A] = 0.3
    coeffs_R_boundary[space_020B] = 0.3
    coeffs_R_boundary[space_029A] = 0.5
    coeffs_R_boundary[space_031A] = 0.5
    coeffs_R_boundary[space_033A] = 0.5
    coeffs_R_boundary[space_035A] = 0.5
    coeffs_R_boundary = np.array([coeffs_R_boundary[space] for space in spaces_list])


    coeffs_C_air = {space: 1 for space in spaces_list}
    coeffs_C_air[space_007A] = 10
    coeffs_C_air[space_011A] = 5
    coeffs_C_air[space_012A] = 7
    coeffs_C_air[space_013A] = 20
    coeffs_C_air[space_015A] = 2
    coeffs_C_air[space_020A] = 2
    coeffs_C_air[space_020B] = 2
    coeffs_C_air[space_029A] = 2
    coeffs_C_air[space_031A] = 2
    coeffs_C_air[space_033A] = 2
    coeffs_C_air[space_035A] = 2
    coeffs_C_air = np.array([coeffs_C_air[space] for space in spaces_list])


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
    C_air_x0 = (airVolumes*1000+furniture_mass*floorAreas*specific_heat_capacity_furniture)*coeffs_C_air
    C_boundary_x0 = floorAreas*specific_heat_capacity_wall*thickness/3

    U_ins = 0.18 #W/m2K
    R_out_x0 = (1/(wallAreas*U_ins))/2*coeffs_R_out_in
    R_in_x0 = (1/(wallAreas*U_ins))/2*coeffs_R_out_in

    infiltrations_x0 = airVolumes*0.3/space_007A.airVolume

    print("C_wall_x0: ", C_wall_x0)
    print("C_air_x0: ", C_air_x0)
    print("C_boundary_x0: ", C_boundary_x0)
    print("R_out_x0: ", R_out_x0)
    print("R_in_x0: ", R_in_x0)
    print("inf_x0: ", infiltrations_x0)

    R_boundary_x0 = (1/(floorAreas*U_ins))*coeffs_R_boundary

    print("R_booundary x0", R_boundary_x0)

    aaa

    infiltrations_x0 = [0.01, 0.001, 0.005, 0.33, 0.009, 0.005, 0.004, 0.03, 0.089, 0.01, 0.005, 0.007]
    infiltrations_x0_lb = [0.01/2, 0.001/2, 0.005/2, 0.33/2, 0.009/2, 0.005/2, 0.004/2, 0.03/2, 0.089/2, 0.01/2, 0.005/2, 0.007/2]
    infiltrations_x0_ub = [0.01*2, 0.001*2, 0.005*2, 0.33*2, 0.009*2, 0.005*2, 0.004*2, 0.03*2, 0.089*2, 0.01*2, 0.005*2, 0.007*2]

    T_boundary_list = [22,22,22,18,21,22,20,22,22.5,23.5,21,19]

    targetParameters = {"private": {"C_wall": {"components": spaces_list, "x0": list(C_wall_x0), "lb": list(C_wall_x0/2), "ub": list(C_wall_x0*2)},
                                    "C_air": {"components": spaces_list, "x0": list(C_air_x0), "lb": list(C_air_x0/2), "ub": list(C_air_x0*2)},
                                    "C_boundary": {"components": spaces_list, "x0": list(C_boundary_x0), "lb": list(C_boundary_x0/2), "ub": list(C_boundary_x0*2)},
                                    "R_out": {"components": spaces_list, "x0": list(R_out_x0), "lb": list(R_out_x0/2), "ub": list(R_out_x0*2)},
                                    "R_in": {"components": spaces_list, "x0": list(R_in_x0), "lb": list(R_in_x0/2), "ub": list(R_in_x0*2)},
                                    "R_boundary": {"components": spaces_list, "x0": list(R_boundary_x0), "lb": list(R_boundary_x0/2), "ub": list(R_boundary_x0*2)},
                                    "f_wall": {"components": spaces_list, "x0": 1, "lb": 0, "ub": 3},
                                    "f_air": {"components": spaces_list, "x0": 1, "lb": 0, "ub": 3},
                                    "kp": {"components": heating_controller_list, "x0": 0.001, "lb": 1e-5, "ub": 3},
                                    "Ti": {"components": heating_controller_list, "x0": 3, "lb": 1e-5, "ub": 5},
                                    "m_flow_nominal": {"components": space_heater_valve_list, "x0": 0.02, "lb": 1e-3, "ub": 0.1}, #0.0202
                                    "infiltration": {"components": spaces_list, "x0": list(infiltrations_x0), "lb": list(infiltrations_x0_lb), "ub": list(infiltrations_x0_ub)},
                                    "Q_occ_gain": {"components": spaces_list, "x0": 100, "lb": 0, "ub": 200},
                                    "T_boundary": {"components": spaces_list, "x0": list(T_boundary_list), "lb": 17, "ub": 24},
                                    },
                        "shared": {"C_int": {"components": spaces_list, "x0": 1e+6, "lb": 1e+4, "ub": 2e+6},
                                    "R_int": {"components": spaces_list, "x0": 0.05, "lb": 1e-2, "ub": 0.1},
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
    estimator = tb.Estimator(model)
    # estimator.chain_savedir = os.path.join(uppath(os.path.abspath(__file__), 1), "generated_files", "models", "model_1st_floor", "model_parameters", "estimation_results", "chain_logs", "20240702_100055.pickle")
    # model.load_estimation_result(estimator.chain_savedir)
    # Options for the PTEMCEE estimation method. If the options argument is not supplied or None is supplied, default options are applied.  
    options = {"n_sample": 2, #500 #This is a test file, and we therefore only sample 2. Typically, we need at least 1000 samples before the chain converges. 
                "n_temperature": 1, #20 #Number of parallel chains/temperatures.
                "fac_walker": 2, #Scaling factor for the number of ensemble walkers per chain. This number is multiplied with the number of estimated to get the number of ensemble walkers per chain. Minimum is 2 (required by PTEMCEE).
                "prior": "uniform", #Prior distribution - "gaussian" is also implemented
                "walker_initialization": "gaussian", #Initialization of parameters - "gaussian" is also implemented
                "add_gp": True,
                "maxtasksperchild": 30,
                "n_save_checkpoint": 1,
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
    
    add_gp = False
    if add_gp:
        model.load_estimation_result(estimator.chain_savedir)
        options = {"n_sample": 1000, #This is a test file, and we therefore only sample 2. Typically, we need at least 1000 samples before the chain converges. 
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