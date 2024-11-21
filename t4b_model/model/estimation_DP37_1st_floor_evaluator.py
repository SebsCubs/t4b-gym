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
from model_DP37_1st_floor import get_model, get_model_60
from model_DP37_1st_floor_added_space_heater import get_model_added_space_heater, get_model_added_space_heater_60



def test_load_emcee_chain():
    stepSize = 600
    startTime_test1 = datetime.datetime(year=2024, month=1, day=5, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test1 = datetime.datetime(year=2024, month=1, day=19, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test = [startTime_test1]
    endTime_test = [endTime_test1]
    stepSize_test = [stepSize]

    model = get_model(id="\#1: Baseline")
    model_60 = get_model_60(id=r"\#2: Baseline, $T_{w,sup}=65^{\circ}$C")
    # model_replace = get_model("\#2: Replace")
    model_add1 = get_model_added_space_heater("\#3: Add same")
    model_add2 = get_model_added_space_heater("\#4: Add large")
    model_60_add = get_model_added_space_heater_60(r"\#5: Add same, $T_{w,sup}=65^{\circ}$C")

    



    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240715_002326.pickle" #After new x0
    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240728_144747.pickle" #After new x0


    model.load_estimation_result(loaddir)
    print(model.chain_log["chain.x"].shape)
    # model_replace.load_estimation_result(chain_log=model.chain_log)
    # model_replace.chain_log["component_id"] = [i.replace("[012A][012A_space_heater]", "[012A][012A_space_heater][012A_space_heater_added]") for i in model_replace.chain_log["component_id"]]
    
    model_add1.load_estimation_result(chain_log=model.chain_log)
    model_add1.chain_log["component_id"] = [i.replace("[012A][012A_space_heater]", "[012A][012A_space_heater][012A_space_heater_added]") for i in model_add1.chain_log["component_id"]]
    
    model_add2.load_estimation_result(chain_log=model.chain_log)
    model_add2.chain_log["component_id"] = [i.replace("[012A][012A_space_heater]", "[012A][012A_space_heater][012A_space_heater_added]") for i in model_add2.chain_log["component_id"]]
    
    model_60.load_estimation_result(chain_log=model.chain_log)

    model_60_add.load_estimation_result(chain_log=model.chain_log)
    model_60_add.chain_log["component_id"] = [i.replace("[012A][012A_space_heater]", "[012A][012A_space_heater][012A_space_heater_added]") for i in model_60_add.chain_log["component_id"]]

    
    percentile = 2
    targetMeasuringDevices = {  "007A_valve_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "007A_temperature_sensor": {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                "007A_damper_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "007A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "008A_damper_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "011A_valve_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "011A_temperature_sensor": {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                "011A_damper_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "011A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "012A_valve_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "012A_temperature_sensor": {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                "012A_damper_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "012A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "013A_valve_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "013A_temperature_sensor": {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                "013A_damper_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "013A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "015A_valve_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "015A_temperature_sensor": {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                "015A_damper_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "015A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "020A_valve_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "020A_temperature_sensor": {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                "020A_damper_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "020A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "020B_valve_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "020B_temperature_sensor": {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                "020B_damper_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "020B_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "029A_valve_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "029A_temperature_sensor": {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                "029A_damper_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "029A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "031A_valve_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "031A_temperature_sensor": {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                "031A_damper_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "031A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "033A_valve_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "033A_temperature_sensor": {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                "033A_damper_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "033A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "035A_valve_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "035A_temperature_sensor": {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                                "035A_damper_position_sensor": {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                                "035A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "007A_space_heater_meter": {"standardDeviation": 100/percentile, "scale_factor": 1},
                                "011A_space_heater_meter": {"standardDeviation": 100/percentile, "scale_factor": 1},
                                "012A_space_heater_meter": {"standardDeviation": 100/percentile, "scale_factor": 1},
                                "013A_space_heater_meter": {"standardDeviation": 100/percentile, "scale_factor": 1},
                                "015A_space_heater_meter": {"standardDeviation": 100/percentile, "scale_factor": 1},
                                "020A_space_heater_meter": {"standardDeviation": 100/percentile, "scale_factor": 1},
                                "020B_space_heater_meter": {"standardDeviation": 100/percentile, "scale_factor": 1},
                                "029A_space_heater_meter": {"standardDeviation": 100/percentile, "scale_factor": 1},
                                "031A_space_heater_meter": {"standardDeviation": 100/percentile, "scale_factor": 1},
                                "033A_space_heater_meter": {"standardDeviation": 100/percentile, "scale_factor": 1},
                                "035A_space_heater_meter": {"standardDeviation": 100/percentile, "scale_factor": 1},}

    

    evaluator = tb.Evaluator()
    models = [model, model_60, model_add1, model_add2, model_60_add] ##################################

    # c11 = model_replace.component_dict["[012A][012A_space_heater]"]
    # c12 = model_replace.component_dict["012A_space_heater_valve"]
    # q1 = 747
    # m1 = q1/((55-40)*4180)
    # parameters = [q1, 1.3255, m1]
    # component_list = [c11, c11, c12]
    # attr_list = ["Q_flow_nominal_sh", "n_sh", "m_flow_nominal"]
    # model_replace.set_parameters_from_array(parameters, component_list, attr_list)


    # c11 = model_replace.component_dict["[012A][012A_space_heater][012A_space_heater_added]"]
    # q1 = 1
    # parameters = [q1]
    # component_list = [c11]
    # attr_list = ["Q_flow_nominal_sh1"]
    # model_replace.set_parameters_from_array(parameters, component_list, attr_list)
    # c11 = model_replace.component_dict["[012A][012A_space_heater][012A_space_heater_added]"]
    # c12 = model_replace.component_dict["012A_space_heater_valve_added"]
    # q1 = 747
    # m1 = q1/((55-40)*4180)
    # parameters = [q1, 1.3255, m1, 0]
    # component_list = [c11, c11, c12]
    # attr_list = ["Q_flow_nominal_sh2", "n_sh2", "m_flow_nominal", "dpFixed_nominal"]
    # model_replace.set_parameters_from_array(parameters, component_list, attr_list)

    c11 = model_add1.component_dict["[012A][012A_space_heater][012A_space_heater_added]"]
    c12 = model_add1.component_dict["012A_space_heater_valve_added"]
    q1 = 432
    m1 = q1/((55-40)*4180)
    parameters = [q1, 1.2940, m1, 0]
    component_list = [c11, c11, c12]
    attr_list = ["Q_flow_nominal_sh2", "n_sh2", "m_flow_nominal", "dpFixed_nominal"]
    model_add1.set_parameters_from_array(parameters, component_list, attr_list)

    c11 = model_add2.component_dict["[012A][012A_space_heater][012A_space_heater_added]"]
    c12 = model_add2.component_dict["012A_space_heater_valve_added"]
    q1 = 747
    m1 = q1/((55-40)*4180)
    parameters = [q1, 1.3255, m1, 0]
    component_list = [c11, c11, c12]
    attr_list = ["Q_flow_nominal_sh2", "n_sh2", "m_flow_nominal", "dpFixed_nominal"]
    model_add2.set_parameters_from_array(parameters, component_list, attr_list)

    c11 = model_60_add.component_dict["[012A][012A_space_heater][012A_space_heater_added]"]
    c12 = model_60_add.component_dict["012A_space_heater_valve_added"]
    q1 = 432
    m1 = q1/((55-40)*4180)
    parameters = [q1, 1.2940, m1, 0]
    component_list = [c11, c11, c12]
    attr_list = ["Q_flow_nominal_sh2", "n_sh2", "m_flow_nominal", "dpFixed_nominal"]
    model_60_add.set_parameters_from_array(parameters, component_list, attr_list)


    measuring_device_name_map = {"007A_temperature_sensor": "Space 1",
                                    "011A_temperature_sensor": "Space 3",
                                    "012A_temperature_sensor": "Space 4",
                                    "013A_temperature_sensor": "Space 5",
                                    "015A_temperature_sensor": "Space 6",
                                    "020A_temperature_sensor": "Space 7",
                                    "020B_temperature_sensor": "Space 8",
                                    "029A_temperature_sensor": "Space 9",
                                    "031A_temperature_sensor": "Space 10",
                                    "033A_temperature_sensor": "Space 11",
                                    "035A_temperature_sensor": "Space 12",}

    options = {"targetMeasuringDevices": targetMeasuringDevices, 
               "assume_uncorrelated_noise": True, 
               "burnin": 1000, 
               "n_cores": 1,
               "n_samples_max": 2,
               "limit": 99,
               "seed": 42}
    measurement_devices = ["007A_temperature_sensor", 
                           "011A_temperature_sensor", 
                           "012A_temperature_sensor", 
                           "013A_temperature_sensor", 
                           "015A_temperature_sensor", 
                           "020A_temperature_sensor", 
                           "020B_temperature_sensor", 
                           "029A_temperature_sensor", 
                           "031A_temperature_sensor", 
                           "033A_temperature_sensor", 
                           "035A_temperature_sensor",
                            "007A_space_heater_meter",
                           "011A_space_heater_meter",
                            "012A_space_heater_meter",
                            "013A_space_heater_meter",
                            "015A_space_heater_meter",
                            "020A_space_heater_meter",
                            "020B_space_heater_meter",
                            "029A_space_heater_meter",
                            "031A_space_heater_meter",
                            "033A_space_heater_meter",
                            "035A_space_heater_meter"]
    # measurement_devices = ["012A_space_heater_meter"]
    evaluator.evaluate(startTime=startTime_test,
                        endTime=endTime_test,
                        stepSize=stepSize_test,
                        models=models,
                        measuring_devices=measurement_devices,
                        evaluation_metrics=["T"]*len(measurement_devices),
                        # evaluation_metrics=["T"],
                        method="bayesian_inference",
                        measuring_device_name_map=measuring_device_name_map,
                        include_measured=False,
                        single_plot=True,
                        options=options,)


if __name__=="__main__":
    test_load_emcee_chain()