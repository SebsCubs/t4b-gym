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
import cProfile, io, pstats
from model_DP37_1st_floor import get_model
from dateutil import tz


def fcn(self):
    '''
        The fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''
    supply_water_temperature_schedule = tb.PiecewiseLinearScheduleSystem(
        weekDayRulesetDict = {
                "ruleset_default_value": {"X": [-12, 5, 20],
                                          "Y": [60, 50, 20]}},
            saveSimulationResult = True,
        id="supply_water_temperature_schedule")
    outdoor_environment = self.get_component_by_class(self.component_dict, tb.OutdoorEnvironmentSystem)[0]
    self.add_connection(outdoor_environment, supply_water_temperature_schedule, "outdoorTemperature", "outdoorTemperature")
    spaces = self.get_component_by_class(self.component_dict, tb.BuildingSpaceFMUSystem)
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace1AdjBoundaryFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace2AdjBoundaryFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace11AdjBoundaryFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace1AdjBoundaryOutdoorFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace2SH1AdjBoundaryOutdoorFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace2AdjBoundaryOutdoorFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace11AdjBoundaryOutdoorFMUSystem))
    for space in spaces:
        self.add_connection(supply_water_temperature_schedule, space, 
                            "scheduleValue", "supplyWaterTemperature")


def get_model_added_space_heater(id):
    model = tb.Model(id="model_1st_floor_added_space_heater_sc1", saveSimulationResult=True)
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), "configuration_template_DP37_full_no_cooling_added_space_heater.xlsm")
    model.load_model(semantic_model_filename=filename, infer_connections=True, fcn=fcn, create_object_graph=False, create_signature_graphs=False, create_system_graph=False, validate_model=False)
    model.id = id
    return model


def get_model_added_space_heater_60(id, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn
    model = tb.Model(id="model_1st_floor_added_space_heater_sc2", saveSimulationResult=True)
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), "configuration_template_DP37_full_no_cooling_added_space_heater.xlsm")
    model.load_model(semantic_model_filename=filename, infer_connections=True, fcn=fcn_, create_object_graph=False, create_signature_graphs=False, create_system_graph=False, validate_model=False)
    model.id = id
    return model

def run():

    stepSize = 600
    startTime_test1 = datetime.datetime(year=2024, month=1, day=5, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    endTime_test1 = datetime.datetime(year=2024, month=1, day=12, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))
    startTime_test = [startTime_test1]
    endTime_test = [endTime_test1]
    stepSize_test = [stepSize]
    """stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2023, month=12, day=2, hour=0, minute=0, second=0,
                                  tzinfo=gettz("Europe/Copenhagen"))

    endTime = datetime.datetime(year=2023, month=12, day=12, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))"""
    model_baseline = get_model()
    model_sc1 = get_model_added_space_heater_sc1()
    model_sc2 = get_model_added_space_heater_sc2()
    

    loaddir = r"C:/Users/jabj/OneDrive - Syddansk Universitet/PhD_Project_Jakob/Twin4build/python/BuildingEnergyModel/remote_results/chain_logs/20240728_144747.npz" #GP New
    model_baseline.load_estimation_result(loaddir)
    model_sc1.load_estimation_result(loaddir)
    model_sc2.load_estimation_result(loaddir)
    l = model_sc1.chain_log["component_id"]
    l = [w.replace('[012A][012A_space_heater]', '[012A][012A_space_heater][012A_space_heater_added]') for w in l]
    model_sc1.chain_log["component_id"] = l

    l = model_sc2.chain_log["component_id"]
    l = [w.replace('[012A][012A_space_heater]', '[012A][012A_space_heater][012A_space_heater_added]') for w in l]
    model_sc2.chain_log["component_id"] = l

    models = [model_baseline, model_sc1, model_sc2]

    # model_sc2 = get_model_added_space_heater()
    # model_sc3 = get_model_added_space_heater()

    parameters = [400, 1.28, 0.02]
    obj_space = model_sc2.component_dict["[012A][012A_space_heater][012A_space_heater_added]"]
    obj_valve = model_sc2.component_dict["012A_space_heater_valve_added"]
    component_list = [obj_space, obj_space, obj_valve]
    attr_list = ["Q_flow_nominal_sh2", "n_sh2"]
    model_sc1.set_parameters_from_array(parameters, component_list, attr_list)

    parameters = [800, 1.28, 0.05]
    obj_space = model_sc2.component_dict["[012A][012A_space_heater][012A_space_heater_added]"]
    obj_valve = model_sc2.component_dict["012A_space_heater_valve_added"]
    component_list = [obj_space, obj_space, obj_valve]
    attr_list = ["Q_flow_nominal_sh2", "n_sh2"]
    model_sc2.set_parameters_from_array(parameters, component_list, attr_list)


    
    percentile = 2
    targetMeasuringDevices = {  "007A_valve_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "007A_temperature_sensor": {"standardDeviation": 0.05/percentile, "scale_factor": 1},
                                "007A_damper_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "007A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "008A_damper_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "011A_valve_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "011A_temperature_sensor": {"standardDeviation": 0.05/percentile, "scale_factor": 1},
                                "011A_damper_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "011A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "012A_valve_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "012A_temperature_sensor": {"standardDeviation": 0.05/percentile, "scale_factor": 1},
                                "012A_damper_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "012A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "013A_valve_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "013A_temperature_sensor": {"standardDeviation": 0.05/percentile, "scale_factor": 1},
                                "013A_damper_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "013A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "015A_valve_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "015A_temperature_sensor": {"standardDeviation": 0.05/percentile, "scale_factor": 1},
                                "015A_damper_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "015A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "020A_valve_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "020A_temperature_sensor": {"standardDeviation": 0.05/percentile, "scale_factor": 1},
                                "020A_damper_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "020A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "020B_valve_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "020B_temperature_sensor": {"standardDeviation": 0.05/percentile, "scale_factor": 1},
                                "020B_damper_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "020B_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "029A_valve_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "029A_temperature_sensor": {"standardDeviation": 0.05/percentile, "scale_factor": 1},
                                "029A_damper_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "029A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "031A_valve_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "031A_temperature_sensor": {"standardDeviation": 0.05/percentile, "scale_factor": 1},
                                "031A_damper_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "031A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "033A_valve_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "033A_temperature_sensor": {"standardDeviation": 0.05/percentile, "scale_factor": 1},
                                "033A_damper_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "033A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1},
                                "035A_valve_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "035A_temperature_sensor": {"standardDeviation": 0.05/percentile, "scale_factor": 1},
                                "035A_damper_position_sensor": {"standardDeviation": 0.001/percentile, "scale_factor": 1},
                                "035A_co2_sensor": {"standardDeviation": 50/percentile, "scale_factor": 1}}

    
    evaluator = tb.Evaluator()
    options = {"targetMeasuringDevices": targetMeasuringDevices,
               "assume_uncorrelated_noise": False,
               "burnin": 530,
               "n_cores": 2,
               "n_samples_max": 3,
               "limit": 99}

    evaluator.evaluate(startTime=startTime_test,
                        endTime=endTime_test,
                        stepSize=stepSize_test,
                        models=models,
                        measuring_devices=["012A_temperature_sensor", "020A_temperature_sensor", "020B_temperature_sensor", "029A_temperature_sensor", "035A_temperature_sensor"],
                        evaluation_metrics=["T", "T", "T", "T", "T"],
                        method="bayesian_inference",
                        options=options)
    




if __name__ == "__main__":
    run()