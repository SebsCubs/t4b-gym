import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Mac specific?
import datetime
from datetime import timedelta
import pandas as pd
from dateutil.tz import gettz
import sys
# Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = os.path.join(uppath(os.path.abspath(__file__), 4), "Twin4Build")
    sys.path.append(file_path)
import twin4build as tb
from twin4build.utils.uppath import uppath
import numpy as np
import matplotlib.pyplot as plt

model_output_points = [
    {
        'component_id': 'core_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonCor_TZon_y_processed.csv'
    },
    {
        'component_id': 'core_co2_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonCor_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'north_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonNor_TZon_y_processed.csv'
    },
    {
        'component_id': 'north_co2_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonNor_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'south_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonSou_TZon_y_processed.csv'
    },
    {
        'component_id': 'south_co2_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonSou_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'east_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonEas_TZon_y_processed.csv'
    },
    {
        'component_id': 'east_co2_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonEas_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'west_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonWes_TZon_y_processed.csv'
    },
    {
        'component_id': 'west_co2_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonWes_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'vent_supply_airflow_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_V_flow_sup_y_processed.csv'
    },
    {
        'component_id': 'vent_return_air_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_TRet_y_processed.csv'
    },
    {
        'component_id': 'vent_return_airflow_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_V_flow_ret_y_processed.csv'
    },
    {
        'component_id': 'vent_supply_air_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_TSup_y_processed.csv'
    },
    {
        'component_id': 'supply_fan',
        'output_value': 'Power',
        'csv_path': 'C:/repos/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_PFanSup_y_processed.csv'
    }
]

def envelope_fcn(self):
    '''
        The envelope_fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
    '''
    core = tb.BuildingSpaceNoSH1AdjBoundaryFMUSystem(id="core", saveSimulationResult=True)
    north = tb.BuildingSpaceNoSH1AdjBoundaryOutdoorFMUSystem(id="north", saveSimulationResult=True)
    south = tb.BuildingSpaceNoSH1AdjBoundaryOutdoorFMUSystem(id="south", saveSimulationResult=True)
    east = tb.BuildingSpaceNoSH1AdjBoundaryOutdoorFMUSystem(id="east", saveSimulationResult=True)
    west = tb.BuildingSpaceNoSH1AdjBoundaryOutdoorFMUSystem(id="west", saveSimulationResult=True)
    outdoor_environment = tb.OutdoorEnvironmentSystem(filename=r"C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/outdoor_env_data.csv",id="outdoor_environment", saveSimulationResult=True)

    #Outdoor environment connections
    self.add_connection(outdoor_environment, north, "outdoorTemperature", "outdoorTemperature")
    self.add_connection(outdoor_environment, north, "globalIrradiation", "globalIrradiation")
    self.add_connection(outdoor_environment, north, "outdoorCo2Concentration", "outdoorCo2Concentration")

    self.add_connection(outdoor_environment, south, "outdoorTemperature", "outdoorTemperature")
    self.add_connection(outdoor_environment, south, "globalIrradiation", "globalIrradiation")
    self.add_connection(outdoor_environment, south, "outdoorCo2Concentration", "outdoorCo2Concentration")

    self.add_connection(outdoor_environment, east, "outdoorTemperature", "outdoorTemperature")
    self.add_connection(outdoor_environment, east, "globalIrradiation", "globalIrradiation")
    self.add_connection(outdoor_environment, east, "outdoorCo2Concentration", "outdoorCo2Concentration")

    self.add_connection(outdoor_environment, west, "outdoorTemperature", "outdoorTemperature")
    self.add_connection(outdoor_environment, west, "globalIrradiation", "globalIrradiation")
    self.add_connection(outdoor_environment, west, "outdoorCo2Concentration", "outdoorCo2Concentration")

    #Occupancy profiles
    core_occupancy_profile = tb.ScheduleSystem(id="core_occupancy_profile", saveSimulationResult=True) 
    self.add_connection(core_occupancy_profile, core, "scheduleValue", "numberOfPeople")
    north_occupancy_profile = tb.ScheduleSystem(id="north_occupancy_profile", saveSimulationResult=True)
    self.add_connection(north_occupancy_profile, north, "scheduleValue", "numberOfPeople")
    south_occupancy_profile = tb.ScheduleSystem(id="south_occupancy_profile", saveSimulationResult=True)
    self.add_connection(south_occupancy_profile, south, "scheduleValue", "numberOfPeople")
    east_occupancy_profile = tb.ScheduleSystem(id="east_occupancy_profile", saveSimulationResult=True)
    self.add_connection(east_occupancy_profile, east, "scheduleValue", "numberOfPeople")
    west_occupancy_profile = tb.ScheduleSystem(id="west_occupancy_profile", saveSimulationResult=True)
    self.add_connection(west_occupancy_profile, west, "scheduleValue", "numberOfPeople")

    #Adjacency connections
    self.add_connection(core, south, "indoorTemperature", "indoorTemperature_adj1")
    self.add_connection(core, north, "indoorTemperature", "indoorTemperature_adj1")
    self.add_connection(north, core, "indoorTemperature", "indoorTemperature_adj1")
    self.add_connection(core, west, "indoorTemperature", "indoorTemperature_adj1")
    self.add_connection(core, east, "indoorTemperature", "indoorTemperature_adj1")

    #Add core sensors
    core_co2_sensor = tb.SensorSystem(id="core_co2_sensor", saveSimulationResult=True)
    #core_supply_air_temp_sensor = tb.SensorSystem(id="core_supply_air_temp_sensor", saveSimulationResult=True)
    #core_supply_airflow_sensor = tb.SensorSystem(id="core_supply_airflow_sensor", saveSimulationResult=True)
    core_indoor_temp_sensor = tb.SensorSystem(id="core_indoor_temp_sensor", saveSimulationResult=True)
    self.add_connection(core, core_co2_sensor, "indoorCo2Concentration", "measuredValue")
    #self.add_connection(core_supply_airflow_sensor, core, "airFlowRate", "airFlowRate")
    #self.add_connection(core_supply_air_temp_sensor, core, "supplyAirTemperature", "supplyAirTemperature")
    self.add_connection(core, core_indoor_temp_sensor, "indoorTemperature", "measuredValue")


    #Add north sensors
    north_co2_sensor = tb.SensorSystem(id="north_co2_sensor", saveSimulationResult=True)
    self.add_connection(north, north_co2_sensor, "indoorCo2Concentration", "measuredValue")
    #north_supply_air_temp_sensor = tb.SensorSystem(id="north_supply_air_temp_sensor", saveSimulationResult=True)
    #self.add_connection(north_supply_air_temp_sensor, north, "supplyAirTemperature", "supplyAirTemperature")
    #north_supply_airflow_sensor = tb.SensorSystem(id="north_supply_airflow_sensor", saveSimulationResult=True)
    #self.add_connection(north_supply_airflow_sensor, north, "airFlowRate", "airFlowRate")
    north_indoor_temp_sensor = tb.SensorSystem(id="north_indoor_temp_sensor", saveSimulationResult=True)
    self.add_connection(north, north_indoor_temp_sensor, "indoorTemperature", "measuredValue")

     #Add south sensors
    south_co2_sensor = tb.SensorSystem(id="south_co2_sensor", saveSimulationResult=True)
    self.add_connection(south, south_co2_sensor, "indoorCo2Concentration", "measuredValue")
    #south_supply_air_temp_sensor = tb.SensorSystem(id="south_supply_air_temp_sensor", saveSimulationResult=True)
    #self.add_connection(south_supply_air_temp_sensor, south, "supplyAirTemperature", "supplyAirTemperature")
    #south_supply_airflow_sensor = tb.SensorSystem(id="south_supply_airflow_sensor", saveSimulationResult=True)
    #self.add_connection(south_supply_airflow_sensor, south, "airFlowRate", "airFlowRate")
    south_indoor_temp_sensor = tb.SensorSystem(id="south_indoor_temp_sensor", saveSimulationResult=True)
    self.add_connection(south, south_indoor_temp_sensor, "indoorTemperature", "measuredValue")


    #Add east sensors
    east_co2_sensor = tb.SensorSystem(id="east_co2_sensor", saveSimulationResult=True)
    self.add_connection(east, east_co2_sensor, "indoorCo2Concentration", "measuredValue")
    #east_supply_air_temp_sensor = tb.SensorSystem(id="east_supply_air_temp_sensor", saveSimulationResult=True)
    #self.add_connection(east_supply_air_temp_sensor, east, "supplyAirTemperature", "supplyAirTemperature")
    #east_supply_airflow_sensor = tb.SensorSystem(id="east_supply_airflow_sensor", saveSimulationResult=True)
    #self.add_connection(east_supply_airflow_sensor, east, "airFlowRate", "airFlowRate")
    east_indoor_temp_sensor = tb.SensorSystem(id="east_indoor_temp_sensor", saveSimulationResult=True)
    self.add_connection(east, east_indoor_temp_sensor, "indoorTemperature", "measuredValue")
    

    #Add west sensors
    west_co2_sensor = tb.SensorSystem(id="west_co2_sensor", saveSimulationResult=True)
    self.add_connection(west, west_co2_sensor, "indoorCo2Concentration", "measuredValue")
    #west_supply_air_temp_sensor = tb.SensorSystem(id="west_supply_air_temp_sensor", saveSimulationResult=True)
    #self.add_connection(west_supply_air_temp_sensor, west, "supplyAirTemperature", "supplyAirTemperature")
    #west_supply_airflow_sensor = tb.SensorSystem(id="west_supply_airflow_sensor", saveSimulationResult=True)
    #self.add_connection(west_supply_airflow_sensor, west, "airFlowRate", "airFlowRate")
    west_indoor_temp_sensor = tb.SensorSystem(id="west_indoor_temp_sensor", saveSimulationResult=True)
    self.add_connection(west, west_indoor_temp_sensor, "indoorTemperature", "measuredValue")

def vavs_fcn(self):
    '''
        The vavs_fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
    '''
    reheat_coils_supply_water_temperature = tb.SensorSystem(id="reheat_coils_supply_water_temperature", saveSimulationResult=True)
    #reheat_coils_supply_air_temperature = tb.SensorSystem(id="reheat_coils_supply_air_temperature", saveSimulationResult=True)

    #Add core components
    core_temperature_heating_setpoint = tb.ScheduleSystem(id="core_temperature_heating_setpoint", saveSimulationResult=True)
    core_temperature_cooling_setpoint = tb.ScheduleSystem(id="core_temperature_cooling_setpoint", saveSimulationResult=True)
    core_co2_setpoint = tb.ScheduleSystem(id="core_co2_setpoint", saveSimulationResult=True)
    core_temperature_heating_controller = tb.VAVReheatControllerSystem(id="core_temperature_heating_controller", rat_v_flo_min=0.15, saveSimulationResult=True)
    core_reheat_control_sensor = tb.SensorSystem(id="core_reheat_control_sensor", saveSimulationResult=True)
    core_supply_air_temp_sensor = tb.SensorSystem(id="core_supply_air_temp_sensor", saveSimulationResult=True)    
    core_supply_damper_position_sensor = tb.SensorSystem(id="core_supply_damper_position_sensor", saveSimulationResult=True)

    #core_indoor_temp_sensor = tb.SensorSystem(id="core_indoor_temp_sensor", saveSimulationResult=True)

    core_reheat_coil = tb.CoilPumpValveFMUSystem(m2_flow_nominal = 4.4966688*1.225, 
                                                  tau_w_inlet = 1,
                                                  tau_w_outlet = 1,
                                                  tau_air_outlet = 1,
                                                  id="core_reheat_coil", 
                                                  saveSimulationResult=True)
    core_supply_damper = tb.DamperSystem(nominalAirFlowRate=tb.Measurement(hasValue=4.4966688*1.225), id="core_supply_damper", saveSimulationResult=True)
    core_reheat_coil_return_water_temperature = tb.SensorSystem(id="core_reheat_coil_return_water_temperature", saveSimulationResult=True)
    
    #Add core connections
    self.add_connection(core_temperature_heating_setpoint, core_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(core_temperature_cooling_setpoint, core_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(core_co2_setpoint, core_temperature_heating_controller, "scheduleValue", "co2setpointValue")
    

       
    self.add_connection(core_temperature_heating_controller, core_reheat_control_sensor, "y_valve", "measuredValue")
    self.add_connection(core_reheat_control_sensor, core_reheat_coil, "measuredValue", "valvePosition") 
    self.add_connection(core_supply_damper, core_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, core_reheat_coil, "measuredValue", "supplyWaterTemperature")
    
    self.add_connection(core_temperature_heating_controller, core_supply_damper_position_sensor, "y_dam", "measuredValue")
    self.add_connection(core_supply_damper_position_sensor, core_supply_damper, "measuredValue", "damperPosition")

    self.add_connection(core_reheat_coil, core_supply_air_temp_sensor, "outletAirTemperature", "measuredValue")
    self.add_connection(core_reheat_coil, core_reheat_coil_return_water_temperature, "inletWaterTemperature", "measuredValue")

    #Connection to envelope
    self.add_connection(self.components["core_indoor_temp_sensor"], core_temperature_heating_controller, "measuredValue", "roomTemp")
    self.add_connection(core_supply_air_temp_sensor, self.components["core"], "measuredValue", "supplyAirTemperature")
    self.add_connection(core_supply_damper, self.components["core"], "airFlowRate", "airFlowRate")

    #Add north components
    north_temperature_heating_setpoint = tb.ScheduleSystem(id="north_temperature_heating_setpoint", saveSimulationResult=True)
    north_temperature_cooling_setpoint = tb.ScheduleSystem(id="north_temperature_cooling_setpoint", saveSimulationResult=True)
    north_co2_setpoint = tb.ScheduleSystem(id="north_co2_setpoint", saveSimulationResult=True)
    north_temperature_heating_controller = tb.VAVReheatControllerSystem(id="north_temperature_heating_controller", rat_v_flo_min=0.15, saveSimulationResult=True)
    north_reheat_control_sensor = tb.SensorSystem(id="north_reheat_control_sensor", saveSimulationResult=True)
    north_supply_air_temp_sensor = tb.SensorSystem(id="north_supply_air_temp_sensor", saveSimulationResult=True)    
    north_supply_damper_position_sensor = tb.SensorSystem(id="north_supply_damper_position_sensor", saveSimulationResult=True)
    #north_indoor_temp_sensor = tb.SensorSystem(id="north_indoor_temp_sensor", saveSimulationResult=True)
    north_reheat_coil = tb.CoilPumpValveFMUSystem(m2_flow_nominal = 0.947948667*1.225, 
                                                  tau_w_inlet = 1,
                                                  tau_w_outlet = 1,
                                                  tau_air_outlet = 1,
                                                  id="north_reheat_coil", 
                                                  saveSimulationResult=True)
    north_supply_damper = tb.DamperSystem(nominalAirFlowRate=tb.Measurement(hasValue=0.947948667*1.225), id="north_supply_damper", saveSimulationResult=True)
    north_reheat_coil_return_water_temperature = tb.SensorSystem(id="north_reheat_coil_return_water_temperature", saveSimulationResult=True)
    
    #Add north connections
    self.add_connection(north_temperature_heating_setpoint, north_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(north_temperature_cooling_setpoint, north_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(north_co2_setpoint, north_temperature_heating_controller, "scheduleValue", "co2setpointValue")

    self.add_connection(north_temperature_heating_controller, north_reheat_control_sensor, "y_valve", "measuredValue")
    self.add_connection(north_reheat_control_sensor, north_reheat_coil, "measuredValue", "valvePosition")
    self.add_connection(north_supply_damper, north_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, north_reheat_coil, "measuredValue", "supplyWaterTemperature")
    
    self.add_connection(north_temperature_heating_controller, north_supply_damper_position_sensor, "y_dam", "measuredValue")
    self.add_connection(north_supply_damper_position_sensor, north_supply_damper, "measuredValue", "damperPosition")
    
    self.add_connection(north_reheat_coil, north_supply_air_temp_sensor, "outletAirTemperature", "measuredValue")
    self.add_connection(north_reheat_coil, north_reheat_coil_return_water_temperature, "inletWaterTemperature", "measuredValue")

    #Connection to envelope
    self.add_connection(self.components["north_indoor_temp_sensor"], north_temperature_heating_controller, "measuredValue", "roomTemp")
    self.add_connection(north_supply_air_temp_sensor, self.components["north"], "measuredValue", "supplyAirTemperature")
    self.add_connection(north_supply_damper, self.components["north"], "airFlowRate", "airFlowRate")    

    #Add south components
    south_temperature_heating_setpoint = tb.ScheduleSystem(id="south_temperature_heating_setpoint", saveSimulationResult=True)
    south_temperature_cooling_setpoint = tb.ScheduleSystem(id="south_temperature_cooling_setpoint", saveSimulationResult=True)
    south_co2_setpoint = tb.ScheduleSystem(id="south_co2_setpoint", saveSimulationResult=True)
    south_temperature_heating_controller = tb.VAVReheatControllerSystem(id="south_temperature_heating_controller", rat_v_flo_min=0.15, saveSimulationResult=True)
    south_reheat_control_sensor = tb.SensorSystem(id="south_reheat_control_sensor", saveSimulationResult=True)
    south_supply_air_temp_sensor = tb.SensorSystem(id="south_supply_air_temp_sensor", saveSimulationResult=True)    
    south_supply_damper_position_sensor = tb.SensorSystem(id="south_supply_damper_position_sensor", saveSimulationResult=True)
    #south_indoor_temp_sensor = tb.SensorSystem(id="south_indoor_temp_sensor", saveSimulationResult=True)
    south_reheat_coil = tb.CoilPumpValveFMUSystem(m2_flow_nominal = 0.947948667*1.225, 
                                                  tau_w_inlet = 1,
                                                  tau_w_outlet = 1,
                                                  tau_air_outlet = 1,
                                                  id="south_reheat_coil", 
                                                  saveSimulationResult=True)
    south_supply_damper = tb.DamperSystem(nominalAirFlowRate=tb.Measurement(hasValue=0.947948667*1.225), id="south_supply_damper", saveSimulationResult=True)
    south_reheat_coil_return_water_temperature = tb.SensorSystem(id="south_reheat_coil_return_water_temperature", saveSimulationResult=True)

    #Add south connections
    self.add_connection(south_temperature_heating_setpoint, south_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(south_temperature_cooling_setpoint, south_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(south_co2_setpoint, south_temperature_heating_controller, "scheduleValue", "co2setpointValue")

    self.add_connection(south_temperature_heating_controller, south_reheat_control_sensor, "y_valve", "measuredValue")
    self.add_connection(south_reheat_control_sensor, south_reheat_coil, "measuredValue", "valvePosition")
    self.add_connection(south_supply_damper, south_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, south_reheat_coil, "measuredValue", "supplyWaterTemperature")
    
    self.add_connection(south_temperature_heating_controller, south_supply_damper_position_sensor, "y_dam", "measuredValue")
    self.add_connection(south_supply_damper_position_sensor, south_supply_damper, "measuredValue", "damperPosition")
    
    self.add_connection(south_reheat_coil, south_supply_air_temp_sensor, "outletAirTemperature", "measuredValue")
    self.add_connection(south_reheat_coil, south_reheat_coil_return_water_temperature, "inletWaterTemperature", "measuredValue")

    #Connection to envelope
    self.add_connection(self.components["south_indoor_temp_sensor"], south_temperature_heating_controller, "measuredValue", "roomTemp")
    self.add_connection(south_supply_air_temp_sensor, self.components["south"], "measuredValue", "supplyAirTemperature")
    self.add_connection(south_supply_damper, self.components["south"], "airFlowRate", "airFlowRate")

    #Add east components
    east_temperature_heating_setpoint = tb.ScheduleSystem(id="east_temperature_heating_setpoint", saveSimulationResult=True)
    east_temperature_cooling_setpoint = tb.ScheduleSystem(id="east_temperature_cooling_setpoint", saveSimulationResult=True)
    east_co2_setpoint = tb.ScheduleSystem(id="east_co2_setpoint", saveSimulationResult=True)
    east_temperature_heating_controller = tb.VAVReheatControllerSystem(id="east_temperature_heating_controller", rat_v_flo_min=0.15, saveSimulationResult=True)
    east_reheat_control_sensor = tb.SensorSystem(id="east_reheat_control_sensor", saveSimulationResult=True)
    east_supply_air_temp_sensor = tb.SensorSystem(id="east_supply_air_temp_sensor", saveSimulationResult=True)    
    east_supply_damper_position_sensor = tb.SensorSystem(id="east_supply_damper_position_sensor", saveSimulationResult=True)
    #east_indoor_temp_sensor = tb.SensorSystem(id="east_indoor_temp_sensor", saveSimulationResult=True)
    east_reheat_coil = tb.CoilPumpValveFMUSystem(m2_flow_nominal = 0.9001996*1.225, 
                                                  tau_w_inlet = 1,
                                                  tau_w_outlet = 1,
                                                  tau_air_outlet = 1,
                                                  id="east_reheat_coil", 
                                                  saveSimulationResult=True)
    east_supply_damper = tb.DamperSystem(nominalAirFlowRate=tb.Measurement(hasValue=0.9001996*1.225), id="east_supply_damper", saveSimulationResult=True)
    east_reheat_coil_return_water_temperature = tb.SensorSystem(id="east_reheat_coil_return_water_temperature", saveSimulationResult=True)

    #Add east connections
    self.add_connection(east_temperature_heating_setpoint, east_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(east_temperature_cooling_setpoint, east_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(east_co2_setpoint, east_temperature_heating_controller, "scheduleValue", "co2setpointValue")

    self.add_connection(east_temperature_heating_controller, east_reheat_control_sensor, "y_valve", "measuredValue")
    self.add_connection(east_reheat_control_sensor, east_reheat_coil, "measuredValue", "valvePosition")
    self.add_connection(east_supply_damper, east_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, east_reheat_coil, "measuredValue", "supplyWaterTemperature")
    
    self.add_connection(east_temperature_heating_controller, east_supply_damper_position_sensor, "y_dam", "measuredValue")
    self.add_connection(east_supply_damper_position_sensor, east_supply_damper, "measuredValue", "damperPosition")
    
    self.add_connection(east_reheat_coil, east_supply_air_temp_sensor, "outletAirTemperature", "measuredValue")
    self.add_connection(east_reheat_coil, east_reheat_coil_return_water_temperature, "inletWaterTemperature", "measuredValue")

    #Connection to envelope
    self.add_connection(self.components["east_indoor_temp_sensor"], east_temperature_heating_controller, "measuredValue", "roomTemp")
    self.add_connection(east_supply_air_temp_sensor, self.components["east"], "measuredValue", "supplyAirTemperature")
    self.add_connection(east_supply_damper, self.components["east"], "airFlowRate", "airFlowRate")

    #Add west components
    west_temperature_heating_setpoint = tb.ScheduleSystem(id="west_temperature_heating_setpoint", saveSimulationResult=True)
    west_temperature_cooling_setpoint = tb.ScheduleSystem(id="west_temperature_cooling_setpoint", saveSimulationResult=True)
    west_co2_setpoint = tb.ScheduleSystem(id="west_co2_setpoint", saveSimulationResult=True)
    west_temperature_heating_controller = tb.VAVReheatControllerSystem(id="west_temperature_heating_controller", rat_v_flo_min=0.15, saveSimulationResult=True)
    west_supply_air_temp_sensor = tb.SensorSystem(id="west_supply_air_temp_sensor", saveSimulationResult=True)    
    west_supply_damper_position_sensor = tb.SensorSystem(id="west_supply_damper_position_sensor", saveSimulationResult=True)
    #west_indoor_temp_sensor = tb.SensorSystem(id="west_indoor_temp_sensor", saveSimulationResult=True)
    west_reheat_control_sensor = tb.SensorSystem(id="west_reheat_control_sensor", saveSimulationResult=True)
    west_reheat_coil = tb.CoilPumpValveFMUSystem(m2_flow_nominal = 0.700155244*1.225, 
                                                  tau_w_inlet = 1,
                                                  tau_w_outlet = 1,
                                                  tau_air_outlet = 1,
                                                  id="west_reheat_coil", 
                                                  saveSimulationResult=True)
    west_supply_damper = tb.DamperSystem(nominalAirFlowRate=tb.Measurement(hasValue=0.700155244*1.225), id="west_supply_damper", saveSimulationResult=True)
    west_reheat_coil_return_water_temperature = tb.SensorSystem(id="west_reheat_coil_return_water_temperature", saveSimulationResult=True)

    #Add west connections
    self.add_connection(west_temperature_heating_setpoint, west_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(west_temperature_cooling_setpoint, west_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(west_co2_setpoint, west_temperature_heating_controller, "scheduleValue", "co2setpointValue")    

    self.add_connection(west_temperature_heating_controller, west_reheat_control_sensor, "y_valve", "measuredValue")
    self.add_connection(west_reheat_control_sensor, west_reheat_coil, "measuredValue", "valvePosition")
    self.add_connection(west_supply_damper, west_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, west_reheat_coil, "measuredValue", "supplyWaterTemperature")
    
    self.add_connection(west_temperature_heating_controller, west_supply_damper_position_sensor, "y_dam", "measuredValue")
    self.add_connection(west_supply_damper_position_sensor, west_supply_damper, "measuredValue", "damperPosition")
    
    self.add_connection(west_reheat_coil, west_supply_air_temp_sensor, "outletAirTemperature", "measuredValue")
    self.add_connection(west_reheat_coil, west_reheat_coil_return_water_temperature, "inletWaterTemperature", "measuredValue")

    #Connection to envelope
    self.add_connection(self.components["west_indoor_temp_sensor"], west_temperature_heating_controller, "measuredValue", "roomTemp")
    self.add_connection(west_supply_air_temp_sensor, self.components["west"], "measuredValue", "supplyAirTemperature")
    self.add_connection(west_supply_damper, self.components["west"], "airFlowRate", "airFlowRate")

def ahu_fcn(self):
    """
    Assuming the supply and return junctions and their sensors are already added to the model
    """
    vent_supply_air_temp_sensor = tb.SensorSystem(id="vent_supply_air_temp_sensor", saveSimulationResult=True)
    vent_mixed_air_temp_sensor = tb.SensorSystem(id="vent_mixed_air_temp_sensor", saveSimulationResult=True)
    vent_airflow_sensor = tb.SensorSystem(id="vent_airflow_sensor", saveSimulationResult=True)
    vent_supply_damper_setpoint = tb.ScheduleSystem(id="vent_supply_damper_setpoint", saveSimulationResult=True)
    #vent_return_damper_setpoint = tb.ScheduleSystem(id="vent_return_damper_setpoint", saveSimulationResult=True) #They have the same value
    vent_mixing_damper_setpoint = tb.ScheduleSystem(id="vent_mixing_damper_setpoint", saveSimulationResult=True)
    #vent_return_air_temp_sensor = tb.SensorSystem(id="vent_return_air_temp_sensor", saveSimulationResult=True)
    vent_return_airflow_sensor_ahu = tb.SensorSystem(id="vent_return_airflow_sensor_ahu", saveSimulationResult=True)
    vent_outdoor_air_temp_sensor = tb.SensorSystem(id="vent_outdoor_air_temp_sensor", saveSimulationResult=True)
    vent_power_sensor = tb.SensorSystem(id="vent_power_sensor", saveSimulationResult=True)

    heating_coil_temperature_setpoint = tb.ScheduleSystem(id="heating_coil_temperature_setpoint", saveSimulationResult=True)
    #cooling_coil_temperature_setpoint = tb.ScheduleSystem(id="cooling_coil_temperature_setpoint", saveSimulationResult=True) #They have the same value
    supply_air_temp_setpoint_sensor = tb.SensorSystem(id="supply_air_temp_setpoint_sensor", saveSimulationResult=True) # To allow the RL agent to override the setpoint
    self.add_connection(heating_coil_temperature_setpoint, supply_air_temp_setpoint_sensor, "scheduleValue", "measuredValue")

    # Add AHU fan
    supply_fan = tb.FanSystem(id="supply_fan", saveSimulationResult=True)
    self.add_connection(self.components["vent_return_airflow_sensor"], supply_fan, "measuredValue", "airFlowRate")
    self.add_connection(supply_fan, vent_power_sensor, "Power", "measuredValue")

    # Add AHU heating coil
    supply_heating_coil = tb.CoilHeatingSystem(id="supply_heating_coil", saveSimulationResult=True)
    self.add_connection(vent_mixed_air_temp_sensor, supply_heating_coil, "measuredValue", "inletAirTemperature")
    self.add_connection(self.components["vent_supply_airflow_sensor"], supply_heating_coil, "measuredValue", "airFlowRate")
    self.add_connection(supply_air_temp_setpoint_sensor, supply_heating_coil, "measuredValue", "outletAirTemperatureSetpoint")

    # Add AHU cooling coil
    supply_cooling_coil = tb.CoilCoolingSystem(id="supply_cooling_coil", saveSimulationResult=True)
    self.add_connection(supply_heating_coil, supply_cooling_coil, "outletAirTemperature", "inletAirTemperature")
    self.add_connection(self.components["vent_supply_airflow_sensor"], supply_cooling_coil, "measuredValue", "airFlowRate")
    self.add_connection(supply_air_temp_setpoint_sensor, supply_cooling_coil, "measuredValue", "outletAirTemperatureSetpoint")
    self.add_connection(supply_cooling_coil, vent_supply_air_temp_sensor, "outletAirTemperature", "measuredValue")

    # Add main dampers
    main_supply_damper = tb.DamperSystem(id="main_supply_damper", saveSimulationResult=True)
    self.add_connection(vent_supply_damper_setpoint, main_supply_damper, "scheduleValue", "damperPosition")
    main_return_damper = tb.DamperSystem(id="main_return_damper", saveSimulationResult=True)
    self.add_connection(vent_supply_damper_setpoint, main_return_damper, "scheduleValue", "damperPosition")
    main_mixing_damper = tb.DamperSystem(id="main_mixing_damper", saveSimulationResult=True)
    self.add_connection(vent_mixing_damper_setpoint, main_mixing_damper, "scheduleValue", "damperPosition")

    # Add supply flow junction
    supply_flow_junction_for_return = tb.SupplyFlowJunctionSystem(id="supply_flow_junction_for_return", saveSimulationResult=True)
    self.add_connection(supply_flow_junction_for_return, vent_return_airflow_sensor_ahu, "airFlowRateIn", "measuredValue")
    self.add_connection(main_supply_damper, supply_flow_junction_for_return, "airFlowRate", "airFlowRateOut")
    self.add_connection(main_mixing_damper, supply_flow_junction_for_return, "airFlowRate", "airFlowRateOut")

    # Add return flow junction
    return_flow_junction_for_supply = tb.ReturnFlowJunctionSystem(id="return_flow_junction_for_supply", saveSimulationResult=True)
    self.add_connection(main_supply_damper, return_flow_junction_for_supply, "airFlowRate", "airFlowRateIn")
    self.add_connection(vent_outdoor_air_temp_sensor, return_flow_junction_for_supply, "measuredValue", "airTemperatureIn")
    self.add_connection(main_mixing_damper, return_flow_junction_for_supply, "airFlowRate", "airFlowRateIn")

    self.add_connection(return_flow_junction_for_supply, vent_airflow_sensor, "airFlowRateOut", "measuredValue")
    self.add_connection(return_flow_junction_for_supply, vent_mixed_air_temp_sensor, "airTemperatureOut", "measuredValue")

    #Add connections to the rooms
    self.add_connection(self.components["vent_return_air_temp_sensor"], return_flow_junction_for_supply, "measuredValue", "airTemperatureIn")
    #reheat coils supply air temperature
    #self.add_connection(return_flow_junction_for_supply, self.components["reheat_coils_supply_air_temperature"], "airTemperatureOut", "measuredValue")

def fcn(self):
    '''
        The fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''

    envelope_fcn(self)
    vavs_fcn(self)


    #Add supply junction
    supply_junction = tb.SupplyFlowJunctionSystem(id="supply_junction", saveSimulationResult=True)
    self.add_connection(self.components["core_supply_damper"], supply_junction, "airFlowRate", "airFlowRateOut")
    self.add_connection(self.components["north_supply_damper"], supply_junction, "airFlowRate", "airFlowRateOut")
    self.add_connection(self.components["south_supply_damper"], supply_junction, "airFlowRate", "airFlowRateOut")
    self.add_connection(self.components["east_supply_damper"], supply_junction, "airFlowRate", "airFlowRateOut")
    self.add_connection(self.components["west_supply_damper"], supply_junction, "airFlowRate", "airFlowRateOut")
    
    vent_supply_airflow_sensor = tb.SensorSystem(id="vent_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(supply_junction, vent_supply_airflow_sensor, "airFlowRateIn", "measuredValue")

    #Connect return flow junction 
    return_junction = tb.ReturnFlowJunctionSystem(id="return_junction", saveSimulationResult=True)
    self.add_connection(self.components["core_supply_damper"], return_junction, "airFlowRate", "airFlowRateIn")
    self.add_connection(self.components["north_supply_damper"], return_junction, "airFlowRate", "airFlowRateIn")
    self.add_connection(self.components["south_supply_damper"], return_junction, "airFlowRate", "airFlowRateIn")
    self.add_connection(self.components["east_supply_damper"], return_junction, "airFlowRate", "airFlowRateIn")
    self.add_connection(self.components["west_supply_damper"], return_junction, "airFlowRate", "airFlowRateIn")

    self.add_connection(self.components["core"], return_junction, "indoorTemperature", "airTemperatureIn")
    self.add_connection(self.components["north"], return_junction, "indoorTemperature", "airTemperatureIn")
    self.add_connection(self.components["south"], return_junction, "indoorTemperature", "airTemperatureIn")
    self.add_connection(self.components["east"], return_junction, "indoorTemperature", "airTemperatureIn")
    self.add_connection(self.components["west"], return_junction, "indoorTemperature", "airTemperatureIn")

    vent_return_airflow_sensor = tb.SensorSystem(id="vent_return_airflow_sensor", saveSimulationResult=True)
    self.add_connection(return_junction, vent_return_airflow_sensor, "airFlowRateOut", "measuredValue")
    vent_return_air_temp_sensor = tb.SensorSystem(id="vent_return_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(return_junction, vent_return_air_temp_sensor, "airTemperatureOut", "measuredValue")

    ahu_fcn(self)

    #Add connection to reheat coils supply air temperature
    self.add_connection(self.components["vent_supply_air_temp_sensor"], self.components["core_reheat_coil"], "measuredValue", "inletAirTemperature")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], self.components["north_reheat_coil"], "measuredValue", "inletAirTemperature")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], self.components["south_reheat_coil"], "measuredValue", "inletAirTemperature")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], self.components["east_reheat_coil"], "measuredValue", "inletAirTemperature")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], self.components["west_reheat_coil"], "measuredValue", "inletAirTemperature")


def get_model(id=None, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn

    if id is not None:
        model = tb.Model(id=id, saveSimulationResult=True)
    else:
        model = tb.Model(id="rooms_and_ahu_model_no_id", saveSimulationResult=True)
    
    model.load(fcn=fcn_, create_signature_graphs=False, validate_model=True, verbose=False, force_config_update=True)
    return model
 

def run(model = None):
    stepSize = 600  # Seconds
    
    time_periods = [
        # Typical heat day: January 11-25, 2024
        (datetime.datetime(year=2024, month=1, day=11, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
            datetime.datetime(year=2024, month=1, day=25, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))),
        
        # Mix day: March 17 - March 31, 2024
        (datetime.datetime(year=2024, month=3, day=17, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
            datetime.datetime(year=2024, month=3, day=31, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))),
        
        # Typical cool day: May 17-31, 2024
        (datetime.datetime(year=2024, month=5, day=17, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
            datetime.datetime(year=2024, month=5, day=31, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")))
    ]     

    startTime = time_periods[0][0]
    endTime = time_periods[0][1]

    if model is None:
        raise ValueError("No model provided to run(). Please provide a model instance.")

    simulator = tb.Simulator()

    # Use keyword arguments to avoid type annotation issues
    simulator.simulate(
        model=model,
        startTime=startTime,
        endTime=endTime,
        stepSize=stepSize
    )

    print("Simulation completed successfully!")

    return simulator

def print_parameter_results(model):
    """Print the resulting parameters for all rooms in a more organized way"""
    
    rooms = ['core', 'north', 'south', 'east', 'west']
    
    # Space model parameters
    print("SPACE MODEL PARAMETERS:")
    space_params = {
        'core': ['C_supply', 'C_air', 'C_int', 'C_boundary', 'R_int', 'R_boundary', 'Q_occ_gain', 
                 'CO2_occ_gain', 'CO2_start', 'airVolume', 'T_boundary', 'infiltration'],
        'other': ['C_supply', 'C_wall', 'C_air', 'C_int', 'C_boundary', 'R_out', 'R_in', 'R_int', 
                 'R_boundary', 'f_wall', 'f_air', 'Q_occ_gain', 'CO2_occ_gain', 'CO2_start', 'airVolume', 'T_boundary', 'infiltration']
    }
    
    for room in rooms:
        print(f"\n{room.upper()}:")
        params = space_params['core'] if room == 'core' else space_params['other']
        for param in params:
            value = getattr(model.components[room], param)
            print(f"{param}: {value}")

    #Fan parameters
    print("\nFAN PARAMETERS:")
    print(f"c1: {model.components['supply_fan'].c1}")
    print(f"c2: {model.components['supply_fan'].c2}")
    print(f"c3: {model.components['supply_fan'].c3}")
    print(f"c4: {model.components['supply_fan'].c4}")
    print(f"nominalPowerRate.hasValue: {model.components['supply_fan'].nominalPowerRate.hasValue}")
    print(f"nominalAirFlowRate.hasValue: {model.components['supply_fan'].nominalAirFlowRate.hasValue}")

def parameter_estimation():
    """
    Checklist for the parameter estimation:
    [x] Gather and add the outdoor environment data
    [x] Define the target parameters
    [x] Define the target measuring devices
    [x] Define the list of required data points from the BOPTEST model
    [x] Acquire and pre process the data from the BOPTEST model
    [x] Fill in the data paths in the model_parameters/data_paths.json file
    [x] Update the component files with known parameters in the model_parameters folder
    [x] Transform the temperatures to celsius in the data
    [] Run the parameter estimation
    [] Load the estimation result
    [] Plot the results
    """
    stepSize = 60  # Seconds can go down to 30
    # Then set the startTime and endTime to a valid range
    startTime = datetime.datetime(year=2024, month=1, day=11, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=1, day=25, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))

    envelope_filepath = r"C:\repos\RL_control\boptest_model\generated_files\models\only_rooms_estimation\model_parameters\estimation_results\LS_result\mix_day_most_accurate_08042025.pickle"
    vavs_filepath = r"C:\repos\RL_control\boptest_model\generated_files\models\vav_controllers_param_est\model_parameters\estimation_results\LS_result\20250506_095811_ls.pickle"
    ahu_filepath = r"C:\repos\RL_control\boptest_model\generated_files\models\only_ahu_model\model_parameters\estimation_results\LS_result\20250314_163600_ls.pickle"
    parameter_filenames = {"envelope": envelope_filepath, "vavs": vavs_filepath, "ahu": ahu_filepath}
    # Load model with estimated parameters and run simulation
    model = get_model(id="rooms_and_ahu_estimation")
    model.load_estimation_result(parameter_filenames["envelope"])
    model.load_estimation_result(parameter_filenames["vavs"])
    model.load_estimation_result(parameter_filenames["ahu"])
    
    
    north = model.components["north"]
    north.C_boundary = 94407.559
    north.Q_occ_gain = 150 #224.04964129088137

    east = model.components["east"]
    east.C_boundary = 31197951.98026053
    east.Q_occ_gain = 150 #232.5238692961377

    south = model.components["south"]
    south.Q_occ_gain = 150 #232.5238692961377

    west = model.components["west"]
    west.Q_occ_gain = 150 #232.5238692961377

    ## Target parameters definition
    supply_fan = model.components["supply_fan"]
    #Space parameters are estimated separately
    #CORE
    core_supply_damper = model.components["core_supply_damper"]
    core_temp_controller = model.components["core_temperature_heating_controller"]
    #NORTH
    north_supply_damper = model.components["north_supply_damper"]
    north_temp_controller = model.components["north_temperature_heating_controller"]
    #SOUTH
    south_supply_damper = model.components["south_supply_damper"]
    south_temp_controller = model.components["south_temperature_heating_controller"]
    #EAST
    east_supply_damper = model.components["east_supply_damper"]
    east_temp_controller = model.components["east_temperature_heating_controller"]
    #WEST
    west_supply_damper = model.components["west_supply_damper"] 
    west_temp_controller = model.components["west_temperature_heating_controller"]

    dampers_list = [core_supply_damper, north_supply_damper, south_supply_damper, east_supply_damper, west_supply_damper]

    targetParameters = {"private": {
                                    "nominalPowerRate.hasValue": {"components": [supply_fan], "x0": 3.5, "lb": 1e-2, "ub": 10},
                                    "c1": {"components": [supply_fan], "x0": 0.5, "lb": -15, "ub": 15},
                                    "c2": {"components": [supply_fan], "x0": 0.5, "lb": -15, "ub": 15},
                                    "c3": {"components": [supply_fan], "x0": 0.5, "lb": -15, "ub": 15},
                                    "c4": {"components": [supply_fan], "x0": 0.5, "lb": -15, "ub": 15},
                                    "k_coo": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 1, "lb": 1e-5, "ub": 10},
                                    "ti_coo": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 1, "lb": 1e-5, "ub": 10},
                                    "k_hea": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 1, "lb": 1e-5, "ub": 10},
                                    "ti_hea": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 1, "lb": 1e-5, "ub": 10},
                                    "nominalAirFlowRate.hasValue": {"components": dampers_list, "x0": 3.5, "lb": 1e-2, "ub": 10},
                                    "a": {"components": dampers_list, "x0": 6.74, "lb": 0.5, "ub": 8}
                                    }
                        }

    percentile = 2
    targetMeasuringDevices = {
                             model.components["vent_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 10},
                             model.components["vent_return_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 10},
                             model.components["vent_return_air_temp_sensor"]: {"standardDeviation": 1/percentile, "scale_factor": 20},
                             model.components["vent_power_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 1000},

                             model.components["core_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["core_supply_damper_position_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                             model.components["core_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["core_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["north_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["north_supply_damper_position_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                             model.components["north_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["north_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["south_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["south_supply_damper_position_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                             model.components["south_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["south_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["east_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["east_supply_damper_position_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                             model.components["east_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["east_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["west_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["west_supply_damper_position_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 1},
                             model.components["west_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["west_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             }  

    
    options = {
            "n_cores": 8,
            "ftol": 1e-10,
            "xtol": 1e-10,
            "gtol": 1e-10,
            "max_nfev": 90,
            "verbose": 2}
    estimator = tb.Estimator(model)
    estimator.estimate(targetParameters=targetParameters,
                        targetMeasuringDevices=targetMeasuringDevices,
                        startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize,
                        method="LS", #Use Least Squares instead
                        options=options,
                        verbose=True
                        )
    model.load_estimation_result(estimator.result_savedir_pickle)

    #Print the resulting parameters

    print("Resulting parameters saved: ", estimator.result_savedir_pickle)
    print_parameter_results(model)

    return estimator.result_savedir_pickle

def fan_parameter_estimation():
    stepSize = 60  # Seconds can go down to 30
    # Then set the startTime and endTime to a valid range
    startTime = datetime.datetime(year=2024, month=1, day=11, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=1, day=25, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))

    envelope_filepath = r"C:\repos\RL_control\boptest_model\generated_files\models\only_rooms_estimation\model_parameters\estimation_results\LS_result\mix_day_most_accurate_08042025.pickle"
    vavs_filepath = r"C:\repos\RL_control\boptest_model\generated_files\models\vav_controllers_param_est\model_parameters\estimation_results\LS_result\20250506_095811_ls.pickle"
    ahu_filepath = r"C:\repos\RL_control\boptest_model\generated_files\models\only_ahu_model\model_parameters\estimation_results\LS_result\20250314_163600_ls.pickle"


    parameter_filenames = {"envelope": envelope_filepath, "vavs": vavs_filepath, "ahu": ahu_filepath}
    # Load model with estimated parameters and run simulation
    model = get_model(id="rooms_and_ahu_estimation")
    model.load_estimation_result(parameter_filenames["envelope"])
    model.load_estimation_result(parameter_filenames["vavs"])
    model.load_estimation_result(parameter_filenames["ahu"])
    
    
    north = model.components["north"]
    north.C_boundary = 94407.559
    north.Q_occ_gain = 150 #224.04964129088137

    east = model.components["east"]
    east.C_boundary = 31197951.98026053
    east.Q_occ_gain = 150 #232.5238692961377

    south = model.components["south"]
    south.Q_occ_gain = 150 #232.5238692961377

    west = model.components["west"]
    west.Q_occ_gain = 150 #232.5238692961377

    model.load_estimation_result(r"C:\repos\RL_control\boptest_model\generated_files\models\rooms_and_ahu_estimation\model_parameters\estimation_results\LS_result\20251026_151452_ls.pickle")
    
    ## Target parameters definition
    supply_fan = model.components["supply_fan"]
    #Space parameters are estimated separately


    targetParameters = {"private": {
                                    "nominalPowerRate.hasValue": {"components": [supply_fan], "x0": 3.5, "lb": 1e-2, "ub": 50},
                                    "c1": {"components": [supply_fan], "x0": 0.5, "lb": -30, "ub": 100},
                                    "c2": {"components": [supply_fan], "x0": 0.5, "lb": -30, "ub": 100},
                                    "c3": {"components": [supply_fan], "x0": 0.5, "lb": -30, "ub": 100},
                                    "c4": {"components": [supply_fan], "x0": 0.5, "lb": -30, "ub": 100}
                                    }
                        }

    percentile = 2
    targetMeasuringDevices = {
                             model.components["vent_supply_airflow_sensor"]: {"standardDeviation": 1/percentile, "scale_factor": 10},
                             model.components["vent_power_sensor"]: {"standardDeviation": 100/percentile, "scale_factor": 1000},
                             }  

    
    options = {
            "n_cores": 8,
            "ftol": 1e-10,
            "xtol": 1e-10,
            "gtol": 1e-10,
            "max_nfev": 90,
            "verbose": 2}
    estimator = tb.Estimator(model)
    estimator.estimate(targetParameters=targetParameters,
                        targetMeasuringDevices=targetMeasuringDevices,
                        startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize,
                        method="LS", #Use Least Squares instead
                        options=options,
                        verbose=True
                        )
    model.load_estimation_result(estimator.result_savedir_pickle)

    #Print the resulting parameters

    print("Resulting parameters saved: ", estimator.result_savedir_pickle)
    print_parameter_results(model)

    return estimator.result_savedir_pickle
    
def rooms_parameter_estimation():
    stepSize = 60  # Seconds can go down to 30
    # Then set the startTime and endTime to a valid range
    startTime = datetime.datetime(year=2024, month=1, day=11, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=1, day=25, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))

    envelope_filepath = r"C:\repos\RL_control\boptest_model\generated_files\models\only_rooms_estimation\model_parameters\estimation_results\LS_result\mix_day_most_accurate_08042025.pickle"
    vavs_filepath = r"C:\repos\RL_control\boptest_model\generated_files\models\vav_controllers_param_est\model_parameters\estimation_results\LS_result\20250506_095811_ls.pickle"
    ahu_filepath = r"C:\repos\RL_control\boptest_model\generated_files\models\only_ahu_model\model_parameters\estimation_results\LS_result\20250314_163600_ls.pickle"


    parameter_filenames = {"envelope": envelope_filepath, "vavs": vavs_filepath, "ahu": ahu_filepath}
    # Load model with estimated parameters and run simulation
    model = get_model(id="rooms_and_ahu_estimation")
    model.load_estimation_result(parameter_filenames["envelope"])
    model.load_estimation_result(parameter_filenames["vavs"])
    model.load_estimation_result(parameter_filenames["ahu"])
    
    
    north = model.components["north"]
    north.C_boundary = 94407.559
    north.Q_occ_gain = 150 #224.04964129088137

    east = model.components["east"]
    east.C_boundary = 31197951.98026053
    east.Q_occ_gain = 150 #232.5238692961377

    south = model.components["south"]
    south.Q_occ_gain = 150 #232.5238692961377

    west = model.components["west"]
    west.Q_occ_gain = 150 #232.5238692961377
    

    model.load_estimation_result(r"C:\repos\RL_control\boptest_model\generated_files\models\rooms_and_ahu_estimation\model_parameters\estimation_results\LS_result\20251028_094629_ls.pickle")

    ## Target parameters definition
    #CORE
    core_space = model.components["core"]
    #NORTH
    north_space = model.components["north"]
    #SOUTH
    south_space = model.components["south"]
    #EAST
    east_space = model.components["east"]
    #WEST
    west_space = model.components["west"]

    targetParameters = {"private": {
                                    "C_wall": {"components": [north_space, south_space, east_space, west_space], "x0": 5476780, "lb": 1e+4, "ub": 1e+8},
                                    "C_air": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 3698995, "lb": 1e+4, "ub": 1e+8},                                    
                                    "C_boundary": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 1e6, "lb": 1e+4, "ub": 1e+8},
                                    "C_int": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 17381232, "lb": 1e+4, "ub": 1e+8},
                                    "R_out": {"components": [north_space, south_space, east_space, west_space], "x0": 0.019, "lb": 1e-2, "ub": 0.5},
                                    "R_in": {"components": [north_space, south_space, east_space, west_space], "x0": 0.015, "lb": 1e-4, "ub": 0.5},
                                    "f_wall": {"components": [north_space, south_space, east_space, west_space], "x0": 0.75, "lb": 0, "ub": 6},
                                    "f_air": {"components": [north_space, south_space, east_space, west_space], "x0": 0.45, "lb": 0, "ub": 6},
                                    "R_int": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 0.015, "lb": 1e-5, "ub": 0.2},
                                    "Q_occ_gain": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 200, "lb": 100, "ub": 350},
                                    },
                        "shared": {
                                    "CO2_occ_gain": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 8.18e-6, "lb": 1e-10, "ub": 0.1},
                                    "infiltration": {"components": [core_space, north_space, south_space, east_space, west_space], "x0": 1e-5, "lb": 1e-10, "ub": 0.01}
                            }}
     

    percentile = 2
    targetMeasuringDevices = {
                             model.components["core_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["core_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},

                             model.components["north_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["north_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},

                             model.components["south_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["south_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},

                             model.components["east_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["east_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},

                             model.components["west_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["west_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             }  

    
    options = {
                "n_cores": 8,
                "ftol": 1e-10,
                "xtol": 1e-10,
                "gtol": 1e-10,
                "max_nfev": 90,
                "verbose": 2
            }
    estimator = tb.Estimator(model)
    estimator.estimate(
                        targetParameters=targetParameters,
                        targetMeasuringDevices=targetMeasuringDevices,
                        startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize,
                        method="LS", #Use Least Squares instead
                        options=options,
                        verbose=True
                    )
    model.load_estimation_result(estimator.result_savedir_pickle)
    print("Resulting parameters:")
    print_parameter_results(model)
    print("Results saved in: ", estimator.result_savedir_pickle)
    return estimator.result_savedir_pickle    

def load_and_print_parameters(filename):
    model = get_model()
    model.load_estimation_result(filename)

    #Print the resulting parameters

    print("Resulting parameters:")
    print_parameter_results(model)

def load_model_parameters(model_id="rooms_and_ahu_estimation"):
    """Load model with all estimated parameters from various estimation results.
    
    Args:
        model_id: ID for the model instance
        
    Returns:
        Model with all parameters loaded
    """
    envelope_filepath = r"C:\repos\RL_control\boptest_model\generated_files\models\only_rooms_estimation\model_parameters\estimation_results\LS_result\mix_day_most_accurate_08042025.pickle"
    vavs_filepath = r"C:\repos\RL_control\boptest_model\generated_files\models\vav_controllers_param_est\model_parameters\estimation_results\LS_result\20250506_095811_ls.pickle"
    ahu_filepath = r"C:\repos\RL_control\boptest_model\generated_files\models\only_ahu_model\model_parameters\estimation_results\LS_result\20250314_163600_ls.pickle"
    parameter_filenames = {"envelope": envelope_filepath, "vavs": vavs_filepath, "ahu": ahu_filepath}
    
    # Load model with estimated parameters
    model = get_model(id=model_id)
    model.load_estimation_result(parameter_filenames["envelope"])
    model.load_estimation_result(parameter_filenames["vavs"])
    model.load_estimation_result(parameter_filenames["ahu"])
    
    # Apply manual parameter adjustments
    north = model.components["north"]
    north.C_boundary = 94407.559
    north.Q_occ_gain = 150 #224.04964129088137

    east = model.components["east"]
    east.C_boundary = 31197951.98026053
    east.Q_occ_gain = 150 #232.5238692961377

    south = model.components["south"]
    south.Q_occ_gain = 150 #232.5238692961377

    west = model.components["west"]
    west.Q_occ_gain = 150 #232.5238692961377
    
    # Load additional estimation results
    model.load_estimation_result(r"C:\repos\RL_control\boptest_model\generated_files\models\rooms_and_ahu_estimation\model_parameters\estimation_results\LS_result\damper_params_13_06.pickle")
    model.load_estimation_result(r"C:\repos\RL_control\boptest_model\generated_files\models\rooms_and_ahu_estimation\model_parameters\estimation_results\LS_result\fan_params_13_06.pickle")
    model.load_estimation_result(r"C:\repos\RL_control\boptest_model\generated_files\models\rooms_and_ahu_estimation\model_parameters\estimation_results\LS_result\20251026_151452_ls.pickle")
    model.load_estimation_result(r"C:\repos\RL_control\boptest_model\generated_files\models\rooms_and_ahu_estimation\model_parameters\estimation_results\LS_result\20251028_094629_ls.pickle")
    model.load_estimation_result(r"C:\repos\RL_control\boptest_model\generated_files\models\rooms_and_ahu_estimation\model_parameters\estimation_results\LS_result\20251028_103636_ls.pickle")
    return model

def parameter_evaluation(data_points, save_plots=False):
    """Evaluate model parameters by comparing simulation results with real data.
    
    Args:
        data_points: List of dictionaries containing:
            - component_id: ID of the component to extract simulation data from
            - output_value: Name of the output value from component's saved outputs
            - csv_path: Path to CSV file containing real data
        parameter_filename: Path to pickle file containing estimated parameters
    """
    """
    #Example of data_points format:
    data_points = [
        {
            'component_id': 'core_indoor_temp_sensor',
            'output_value': 'measuredValue',
            'csv_path': 'path/to/core_temp_data.csv'
        },
        # Add more data points as needed
    ]
    """
    
    # Load model with all estimated parameters
    model = load_model_parameters()

    print("Resulting parameters:")
    print_parameter_results(model)

    simulator = run(model)
    stepSize = simulator.stepSize
    plotting_stepSize = 600
    
    # Create comparison plots for each data point
    for data in data_points:
        component_id = data['component_id']
        output_value = data['output_value'] 
        csv_path = data['csv_path']
        
        # Get simulation results
        sim_data = simulator.model.components[component_id].savedOutput[output_value]
        sim_times = simulator.dateTimeSteps
        sim_df = pd.Series(data=sim_data, index=sim_times)
        # Convert timezone without changing the actual timestamps
        sim_df.index = sim_df.index.tz_convert('Europe/Copenhagen')

        # Read real data
        real_data = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        real_data.index = real_data.index.tz_localize('Europe/Copenhagen')

        # First resample both to the common timestep
        sim_df = sim_df.resample(pd.Timedelta(seconds=plotting_stepSize)).mean()
        real_data = real_data.resample(pd.Timedelta(seconds=plotting_stepSize)).mean()

        # Find overlapping time period
        start_time = max(sim_df.index.min(), real_data.index.min())
        end_time = min(sim_df.index.max(), real_data.index.max())
        
        # Slice both datasets to the overlapping period
        sim_df = sim_df[start_time:end_time]
        real_data = real_data[start_time:end_time]
        
        # Now align the data (should be minimal or no interpolation needed since both are on same timestep)
        real_data = real_data.reindex(sim_df.index, method='nearest')
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        plt.plot(sim_df.index, sim_df.values, label='Simulation', linewidth=2)
        plt.plot(real_data.index, real_data.values, label='Real Data', linewidth=2, alpha=0.7)
        
        plt.title(f'Comparison for {component_id} - {output_value}')
        plt.xlabel('Time')
        plt.ylabel(output_value)
        plt.legend()
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Calculate and display error metrics
        mse = np.mean((sim_df.values - real_data.values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(sim_df.values - real_data.values))
        
        plt.figtext(0.15, 0.95, 
                   f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}',
                   bbox=dict(facecolor='white', alpha=0.8))
        if save_plots:
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/{component_id}_{output_value}_comparison.png')
        else:
            plt.show()
        


def load_model_and_params(model_id="rooms_and_ahu_model"):
    """Load model with all estimated parameters.
    
    Args:
        model_id: ID for the model instance
        
    Returns:
        Model with all parameters loaded
    """
    return load_model_parameters(model_id)

# ============================================================
# Model Evaluation Suite - Publication-Quality Metrics & Plots
# ============================================================

SIGNAL_METADATA = {
    'core_indoor_temp_sensor':   {'label': 'Core Zone Temp.',       'unit': '°C',  'category': 'zone_temperature'},
    'core_co2_sensor':           {'label': 'Core Zone CO₂',        'unit': 'ppm', 'category': 'zone_co2'},
    'north_indoor_temp_sensor':  {'label': 'North Zone Temp.',      'unit': '°C',  'category': 'zone_temperature'},
    'north_co2_sensor':          {'label': 'North Zone CO₂',       'unit': 'ppm', 'category': 'zone_co2'},
    'south_indoor_temp_sensor':  {'label': 'South Zone Temp.',      'unit': '°C',  'category': 'zone_temperature'},
    'south_co2_sensor':          {'label': 'South Zone CO₂',       'unit': 'ppm', 'category': 'zone_co2'},
    'east_indoor_temp_sensor':   {'label': 'East Zone Temp.',       'unit': '°C',  'category': 'zone_temperature'},
    'east_co2_sensor':           {'label': 'East Zone CO₂',        'unit': 'ppm', 'category': 'zone_co2'},
    'west_indoor_temp_sensor':   {'label': 'West Zone Temp.',       'unit': '°C',  'category': 'zone_temperature'},
    'west_co2_sensor':           {'label': 'West Zone CO₂',        'unit': 'ppm', 'category': 'zone_co2'},
    'vent_supply_airflow_sensor':{'label': 'Supply Airflow',        'unit': 'kg/s','category': 'ahu'},
    'vent_return_air_temp_sensor':{'label': 'Return Air Temp.',     'unit': '°C',  'category': 'ahu'},
    'vent_return_airflow_sensor':{'label': 'Return Airflow',        'unit': 'kg/s','category': 'ahu'},
    'vent_supply_air_temp_sensor':{'label': 'Supply Air Temp.',     'unit': '°C',  'category': 'ahu'},
    'supply_fan':                {'label': 'Supply Fan Power',      'unit': 'W',   'category': 'ahu'},
}

CATEGORY_COLORS = {
    'zone_temperature': '#D32F2F',
    'zone_co2':         '#388E3C',
    'ahu':              '#1565C0',
}

CATEGORY_LABELS = {
    'zone_temperature': 'Zone Temperatures',
    'zone_co2':         'Zone CO₂ Concentrations',
    'ahu':              'AHU Signals',
}


def set_publication_style():
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 17,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })


def compute_signal_metrics(sim_values, real_values):
    """Compute comprehensive accuracy metrics between simulated and measured signals.

    Metrics follow ASHRAE Guideline 14-2014 definitions where applicable.
    """
    residuals = sim_values - real_values
    n = len(sim_values)
    mean_real = np.mean(real_values)
    std_real = np.std(real_values, ddof=1)

    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    mbe = np.mean(residuals)
    max_ae = np.max(np.abs(residuals))

    cv_rmse = (rmse / abs(mean_real)) * 100 if mean_real != 0 else np.inf
    nmbe = (mbe / abs(mean_real)) * 100 if mean_real != 0 else np.inf

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((real_values - mean_real) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    pearson_r = np.corrcoef(sim_values, real_values)[0, 1]

    # Willmott index of agreement (d)
    d_den = np.sum((np.abs(sim_values - mean_real) + np.abs(real_values - mean_real)) ** 2)
    willmott_d = 1 - (ss_res / d_den) if d_den > 0 else 0.0

    return {
        'N': n,
        'Mean (ref)': mean_real,
        'Std (ref)': std_real,
        'RMSE': rmse,
        'CV-RMSE (%)': cv_rmse,
        'MAE': mae,
        'MBE': mbe,
        'NMBE (%)': nmbe,
        'Max AE': max_ae,
        'R²': r_squared,
        'Pearson r': pearson_r,
        'Willmott d': willmott_d,
    }


def prepare_evaluation_data(data_points, plotting_step_size=600):
    """Load model, run simulation, and return aligned (sim, reference) pairs with metrics."""
    model = load_model_parameters()
    simulator = run(model)

    results = []
    for dp in data_points:
        cid = dp['component_id']
        oval = dp['output_value']
        csv_path = dp['csv_path']

        sim_raw = simulator.model.components[cid].savedOutput[oval]
        sim_df = pd.Series(data=sim_raw, index=simulator.dateTimeSteps, name='Surrogate (T4B)')
        sim_df.index = sim_df.index.tz_convert('Europe/Copenhagen')

        ref_raw = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        ref_raw.index = ref_raw.index.tz_localize('Europe/Copenhagen')
        ref_series = ref_raw.iloc[:, 0].rename('Reference (BOPTEST)')

        sim_df = sim_df.resample(pd.Timedelta(seconds=plotting_step_size)).mean()
        ref_series = ref_series.resample(pd.Timedelta(seconds=plotting_step_size)).mean()

        t0 = max(sim_df.index.min(), ref_series.index.min())
        t1 = min(sim_df.index.max(), ref_series.index.max())
        sim_df = sim_df[t0:t1]
        ref_series = ref_series[t0:t1]
        ref_series = ref_series.reindex(sim_df.index, method='nearest')

        valid = ~(sim_df.isna() | ref_series.isna())
        sim_df, ref_series = sim_df[valid], ref_series[valid]

        meta = SIGNAL_METADATA.get(cid, {'label': cid, 'unit': '-', 'category': 'other'})

        results.append({
            'component_id': cid,
            'output_value': oval,
            'sim': sim_df,
            'ref': ref_series,
            'metrics': compute_signal_metrics(sim_df.values, ref_series.values),
            'meta': meta,
        })
    return results


def plot_time_series_overlay(result, save_dir):
    """Per-signal overlay with residual subplot."""
    meta = result['meta']
    m = result['metrics']
    sim, ref = result['sim'], result['ref']

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    ax_top.plot(ref.index, ref.values, color='#1565C0', label='Reference (BOPTEST)',
                linewidth=1.4, alpha=0.85)
    ax_top.plot(sim.index, sim.values, color='#D32F2F', label='Surrogate (T4B)',
                linewidth=1.4, alpha=0.85)
    ax_top.set_ylabel(f"{meta['label']}  [{meta['unit']}]")
    ax_top.legend(loc='upper right', framealpha=0.9)

    box_text = (f"CV-RMSE = {m['CV-RMSE (%)']:.2f}%\n"
                f"NMBE    = {m['NMBE (%)']:.2f}%\n"
                f"R²      = {m['R²']:.4f}")
    ax_top.text(0.015, 0.96, box_text, transform=ax_top.transAxes, va='top',
                fontsize=13, family='monospace',
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))

    residuals = sim.values - ref.values
    ax_bot.fill_between(sim.index, residuals, 0, alpha=0.45, color='#7B1FA2')
    ax_bot.axhline(0, color='black', lw=0.8, ls='--')
    ax_bot.set_ylabel(f"Error  [{meta['unit']}]")
    ax_bot.set_xlabel('Time')
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    fig.savefig(os.path.join(save_dir, f"{result['component_id']}_overlay.png"))
    plt.close(fig)


def plot_scatter_comparison(result, save_dir):
    """Per-signal scatter plot with 1:1 line and linear regression."""
    meta = result['meta']
    m = result['metrics']
    sim_v, ref_v = result['sim'].values, result['ref'].values

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(ref_v, sim_v, alpha=0.25, s=10, color='#1565C0', edgecolors='none', rasterized=True)

    all_v = np.concatenate([ref_v, sim_v])
    pad = (all_v.max() - all_v.min()) * 0.04
    lo, hi = all_v.min() - pad, all_v.max() + pad
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='1 : 1')

    slope, intercept = np.polyfit(ref_v, sim_v, 1)
    xr = np.linspace(lo, hi, 100)
    ax.plot(xr, intercept + slope * xr, color='#D32F2F', lw=1.5,
            label=f'Fit: y = {slope:.3f}x + {intercept:.2f}')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel(f"Reference (BOPTEST)  [{meta['unit']}]")
    ax.set_ylabel(f"Surrogate (T4B)  [{meta['unit']}]")
    ax.set_title(meta['label'])
    ax.legend(loc='upper left')

    box_text = f"R² = {m['R²']:.4f}\nr  = {m['Pearson r']:.4f}"
    ax.text(0.97, 0.03, box_text, transform=ax.transAxes, va='bottom', ha='right',
            fontsize=13, family='monospace',
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{result['component_id']}_scatter.png"))
    plt.close(fig)


def plot_multi_panel_scatter(all_results, save_dir):
    """3-row x 5-col scatter grid: zone temps | zone CO₂ | AHU signals."""
    ordered_ids = [
        'core_indoor_temp_sensor', 'north_indoor_temp_sensor', 'south_indoor_temp_sensor',
        'east_indoor_temp_sensor', 'west_indoor_temp_sensor',
        'core_co2_sensor', 'north_co2_sensor', 'south_co2_sensor',
        'east_co2_sensor', 'west_co2_sensor',
        'vent_supply_airflow_sensor', 'vent_return_airflow_sensor',
        'vent_return_air_temp_sensor', 'vent_supply_air_temp_sensor', 'supply_fan',
    ]
    lookup = {r['component_id']: r for r in all_results}
    ordered = [lookup[cid] for cid in ordered_ids if cid in lookup]

    nrows, ncols = 3, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 14))
    axes_flat = axes.flatten()

    row_labels = ['Zone Temperatures', 'Zone CO₂ Concentrations', 'AHU Signals']
    row_colors = ['#D32F2F', '#388E3C', '#1565C0']

    for idx, r in enumerate(ordered):
        ax = axes_flat[idx]
        sv, rv = r['sim'].values, r['ref'].values
        row = idx // ncols
        ax.scatter(rv, sv, alpha=0.2, s=6, color=row_colors[row], edgecolors='none', rasterized=True)

        all_v = np.concatenate([rv, sv])
        pad = (all_v.max() - all_v.min()) * 0.05
        lo, hi = all_v.min() - pad, all_v.max() + pad
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_title(r['meta']['label'], fontsize=12)

        m = r['metrics']
        ax.text(0.04, 0.94, f"R²={m['R²']:.3f}", transform=ax.transAxes, va='top',
                fontsize=10, bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.7))

        if idx % ncols == 0:
            ax.set_ylabel(f"T4B  [{r['meta']['unit']}]", fontsize=13)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel(f"BOPTEST  [{r['meta']['unit']}]", fontsize=13)

    for unused_idx in range(len(ordered), nrows * ncols):
        axes_flat[unused_idx].set_visible(False)

    for row_idx, label in enumerate(row_labels):
        fig.text(0.005, 1 - (row_idx + 0.5) / nrows, label, va='center', ha='left',
                 fontsize=15, fontweight='bold', rotation=90, color=row_colors[row_idx])

    plt.tight_layout(rect=[0.025, 0, 1, 1])
    fig.savefig(os.path.join(save_dir, 'multi_panel_scatter.png'))
    plt.close(fig)


def plot_grouped_overlay(results_subset, group_label, save_dir):
    """Stacked time-series overlay for one category of signals."""
    n = len(results_subset)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results_subset):
        meta = r['meta']
        ax.plot(r['ref'].index, r['ref'].values, color='#1565C0', label='BOPTEST', lw=1.3, alpha=0.85)
        ax.plot(r['sim'].index, r['sim'].values, color='#D32F2F', label='T4B', lw=1.3, alpha=0.85)
        ax.set_ylabel(f"{meta['label']}\n[{meta['unit']}]", fontsize=13)
        ax.legend(loc='upper right', ncol=2, fontsize=11)

        m = r['metrics']
        ax.text(0.015, 0.90, f"CV-RMSE={m['CV-RMSE (%)']:.1f}%   R²={m['R²']:.3f}",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.7))

    axes[-1].set_xlabel('Time', fontsize=15)
    fig.autofmt_xdate(rotation=30)
    fig.suptitle(group_label, fontsize=18, y=1.005)
    plt.tight_layout()
    safe_name = group_label.lower().replace(' ', '_').replace('₂', '2')
    fig.savefig(os.path.join(save_dir, f'grouped_{safe_name}.png'), bbox_inches='tight')
    plt.close(fig)


def plot_metrics_summary(all_results, save_dir):
    """Bar charts of CV-RMSE, NMBE, and R² with ASHRAE Guideline 14 thresholds."""
    labels = [r['meta']['label'] for r in all_results]
    cats = [r['meta']['category'] for r in all_results]
    colors = [CATEGORY_COLORS.get(c, '#757575') for c in cats]

    cv_rmse = [r['metrics']['CV-RMSE (%)'] for r in all_results]
    nmbe = [r['metrics']['NMBE (%)'] for r in all_results]
    r2 = [r['metrics']['R²'] for r in all_results]
    x = np.arange(len(labels))

    # --- CV-RMSE ---
    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(x, cv_rmse, color=colors, edgecolor='white', linewidth=0.6)
    ax.axhline(30, color='red', ls='--', lw=1.2, label='ASHRAE Guideline 14 limit (30 %)')
    ax.set_ylabel('CV-RMSE  (%)')
    ax.set_title('Coefficient of Variation of RMSE — All Signals')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=12)
    ax.legend(fontsize=13)
    for b, v in zip(bars, cv_rmse):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                f'{v:.1f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'summary_cv_rmse.png'))
    plt.close(fig)

    # --- NMBE ---
    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(x, nmbe, color=colors, edgecolor='white', linewidth=0.6)
    ax.axhline(10, color='red', ls='--', lw=1.2, label='ASHRAE Guideline 14 limit (±10 %)')
    ax.axhline(-10, color='red', ls='--', lw=1.2)
    ax.set_ylabel('NMBE  (%)')
    ax.set_title('Normalized Mean Bias Error — All Signals')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=12)
    ax.legend(fontsize=13)
    for b, v in zip(bars, nmbe):
        offset = 0.3 if v >= 0 else -0.6
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + offset,
                f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'summary_nmbe.png'))
    plt.close(fig)

    # --- R² ---
    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(x, r2, color=colors, edgecolor='white', linewidth=0.6)
    ax.axhline(0.75, color='orange', ls='--', lw=1.2, label='Good-fit threshold (0.75)')
    ax.set_ylabel('R²')
    ax.set_title('Coefficient of Determination — All Signals')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=12)
    ax.set_ylim([min(0, min(r2) - 0.05), 1.05])
    ax.legend(fontsize=13)
    for b, v in zip(bars, r2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'summary_r_squared.png'))
    plt.close(fig)


def plot_error_distributions(all_results, save_dir):
    """Histograms of normalised residuals grouped by category."""
    categories = {}
    for r in all_results:
        categories.setdefault(r['meta']['category'], []).append(r)

    for cat, group in categories.items():
        n = len(group)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True)
        if n == 1:
            axes = [axes]
        color = CATEGORY_COLORS.get(cat, '#757575')
        for ax, r in zip(axes, group):
            residuals = r['sim'].values - r['ref'].values
            ax.hist(residuals, bins=40, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
            ax.axvline(0, color='black', lw=0.8, ls='--')
            ax.set_xlabel(f"Error  [{r['meta']['unit']}]")
            ax.set_title(r['meta']['label'], fontsize=12)
        axes[0].set_ylabel('Count')
        fig.suptitle(f"Residual Distributions — {CATEGORY_LABELS.get(cat, cat)}",
                     fontsize=16, y=1.02)
        plt.tight_layout()
        safe = cat.replace(' ', '_')
        fig.savefig(os.path.join(save_dir, f'error_dist_{safe}.png'), bbox_inches='tight')
        plt.close(fig)


def generate_metrics_table(all_results, save_dir):
    """Print and save a comprehensive metrics summary with ASHRAE compliance check."""
    rows = []
    for r in all_results:
        row = {'Signal': r['meta']['label'], 'Unit': r['meta']['unit'],
               'Category': r['meta']['category']}
        row.update(r['metrics'])
        rows.append(row)
    df = pd.DataFrame(rows)

    display_cols = ['Signal', 'Unit', 'RMSE', 'CV-RMSE (%)', 'MAE', 'MBE',
                    'NMBE (%)', 'Max AE', 'R²', 'Pearson r', 'Willmott d']
    df_display = df[display_cols]
    df_display.to_csv(os.path.join(save_dir, 'metrics_summary.csv'), index=False, float_format='%.4f')

    sep = '=' * 130
    print(f"\n{sep}")
    print("MODEL EVALUATION METRICS SUMMARY")
    print(sep)
    print(df_display.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(sep)

    print("\nAGGREGATE METRICS BY CATEGORY:")
    print('-' * 90)
    agg_cols = ['RMSE', 'CV-RMSE (%)', 'NMBE (%)', 'MAE', 'R²', 'Willmott d']
    for cat in df['Category'].unique():
        subset = df[df['Category'] == cat]
        print(f"\n  {CATEGORY_LABELS.get(cat, cat).upper()}:")
        for col in agg_cols:
            vals = subset[col]
            print(f"    {col:16s}  mean={vals.mean():9.4f}   std={vals.std():9.4f}   "
                  f"min={vals.min():9.4f}   max={vals.max():9.4f}")

    print(f"\n{'─' * 90}")
    print("ASHRAE GUIDELINE 14-2014 COMPLIANCE  (hourly: CV-RMSE < 30 %, |NMBE| < 10 %)")
    print('─' * 90)
    all_pass = True
    for _, row in df.iterrows():
        cv_ok = abs(row['CV-RMSE (%)']) < 30
        nmbe_ok = abs(row['NMBE (%)']) < 10
        ok = cv_ok and nmbe_ok
        all_pass = all_pass and ok
        tag = 'PASS' if ok else 'FAIL'
        print(f"  {row['Signal']:28s}  CV-RMSE={row['CV-RMSE (%)']:7.2f}% {'✓' if cv_ok else '✗'}   "
              f"NMBE={row['NMBE (%)']:7.2f}% {'✓' if nmbe_ok else '✗'}   → {tag}")
    verdict = 'ALL SIGNALS PASS' if all_pass else 'SOME SIGNALS FAIL'
    print(f"\n  Overall: {verdict}")
    print(sep)

    return df


def model_evaluation_suite(data_points, save_dir='evaluation_results'):
    """Run the complete evaluation pipeline and produce publication-ready outputs.

    Steps
    -----
    1. Load model with calibrated parameters and run forward simulation.
    2. Align simulated outputs with BOPTEST reference data.
    3. Compute per-signal statistical metrics (CV-RMSE, NMBE, R², …).
    4. Generate individual time-series overlays with residual subplots.
    5. Generate individual and multi-panel scatter plots.
    6. Generate grouped overlay figures by signal category.
    7. Generate summary bar charts with ASHRAE Guideline 14 thresholds.
    8. Generate residual distribution histograms.
    9. Print and export a metrics summary table with ASHRAE compliance.
    """
    set_publication_style()
    sub_dirs = {
        'overlays': os.path.join(save_dir, 'overlays'),
        'scatter':  os.path.join(save_dir, 'scatter'),
        'grouped':  os.path.join(save_dir, 'grouped'),
        'errors':   os.path.join(save_dir, 'error_distributions'),
    }
    for d in [save_dir, *sub_dirs.values()]:
        os.makedirs(d, exist_ok=True)

    print("Step 1/7 — Loading model & running simulation …")
    all_results = prepare_evaluation_data(data_points)

    print("Step 2/7 — Computing metrics & generating summary table …")
    metrics_df = generate_metrics_table(all_results, save_dir)

    print("Step 3/7 — Generating per-signal time-series overlays …")
    for r in all_results:
        plot_time_series_overlay(r, sub_dirs['overlays'])

    print("Step 4/7 — Generating scatter plots (individual + multi-panel) …")
    for r in all_results:
        plot_scatter_comparison(r, sub_dirs['scatter'])
    plot_multi_panel_scatter(all_results, sub_dirs['scatter'])

    print("Step 5/7 — Generating grouped overlay figures …")
    by_cat = {}
    for r in all_results:
        by_cat.setdefault(r['meta']['category'], []).append(r)
    for cat, group in by_cat.items():
        plot_grouped_overlay(group, CATEGORY_LABELS.get(cat, cat), sub_dirs['grouped'])

    print("Step 6/7 — Generating summary bar charts …")
    plot_metrics_summary(all_results, save_dir)

    print("Step 7/7 — Generating residual distribution plots …")
    plot_error_distributions(all_results, sub_dirs['errors'])

    print(f"\nAll outputs saved to:  {os.path.abspath(save_dir)}/")
    return all_results, metrics_df


if __name__ == "__main__":
    # parameter_estimation()
    # rooms_parameter_estimation()
    # parameter_evaluation(data_points=model_output_points, save_plots=True)
    model_evaluation_suite(data_points=model_output_points, save_dir='evaluation_results')
