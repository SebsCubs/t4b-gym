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
from twin4build.utils.uppath import uppath
import numpy as np
import matplotlib.pyplot as plt

model_output_points = [
    {
        'component_id': 'core_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonCor_TZon_y_processed.csv'
    },
    {
        'component_id': 'core_co2_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonCor_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'north_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonNor_TZon_y_processed.csv'
    },
    {
        'component_id': 'north_co2_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonNor_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'south_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonSou_TZon_y_processed.csv'
    },
    {
        'component_id': 'south_co2_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonSou_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'east_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonEas_TZon_y_processed.csv'
    },
    {
        'component_id': 'east_co2_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonEas_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'west_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonWes_TZon_y_processed.csv'
    },
    {
        'component_id': 'west_co2_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonWes_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'vent_supply_airflow_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_V_flow_sup_y_processed.csv'
    },
    {
        'component_id': 'vent_return_air_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_TRet_y_processed.csv'
    },
    {
        'component_id': 'vent_return_airflow_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_V_flow_ret_y_processed.csv'
    },
    {
        'component_id': 'vent_supply_air_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_TSup_y_processed.csv'
    },
    {
        'component_id': 'supply_fan',
        'output_value': 'Power',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_PFanSup_y_processed.csv'
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
    outdoor_environment = tb.OutdoorEnvironmentSystem(filename="C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/outdoor_env_data.csv",id="outdoor_environment", saveSimulationResult=True)

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
    reheat_coils_supply_air_temperature = tb.SensorSystem(id="reheat_coils_supply_air_temperature", saveSimulationResult=True)

    #Add core components
    core_temperature_heating_setpoint = tb.ScheduleSystem(id="core_temperature_heating_setpoint", saveSimulationResult=True)
    core_temperature_cooling_setpoint = tb.ScheduleSystem(id="core_temperature_cooling_setpoint", saveSimulationResult=True)
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
    

       
    self.add_connection(core_temperature_heating_controller, core_reheat_control_sensor, "y_valve", "measuredValue")
    self.add_connection(core_reheat_control_sensor, core_reheat_coil, "measuredValue", "valvePosition") 
    self.add_connection(core_supply_damper, core_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, core_reheat_coil, "measuredValue", "supplyWaterTemperature")
    self.add_connection(reheat_coils_supply_air_temperature, core_reheat_coil, "measuredValue", "inletAirTemperature")

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

    self.add_connection(north_temperature_heating_controller, north_reheat_control_sensor, "y_valve", "measuredValue")
    self.add_connection(north_reheat_control_sensor, north_reheat_coil, "measuredValue", "valvePosition")
    self.add_connection(north_supply_damper, north_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, north_reheat_coil, "measuredValue", "supplyWaterTemperature")
    self.add_connection(reheat_coils_supply_air_temperature, north_reheat_coil, "measuredValue", "inletAirTemperature")

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
    

    self.add_connection(south_temperature_heating_controller, south_reheat_control_sensor, "y_valve", "measuredValue")
    self.add_connection(south_reheat_control_sensor, south_reheat_coil, "measuredValue", "valvePosition")
    self.add_connection(south_supply_damper, south_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, south_reheat_coil, "measuredValue", "supplyWaterTemperature")
    self.add_connection(reheat_coils_supply_air_temperature, south_reheat_coil, "measuredValue", "inletAirTemperature")

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
    

    self.add_connection(east_temperature_heating_controller, east_reheat_control_sensor, "y_valve", "measuredValue")
    self.add_connection(east_reheat_control_sensor, east_reheat_coil, "measuredValue", "valvePosition")
    self.add_connection(east_supply_damper, east_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, east_reheat_coil, "measuredValue", "supplyWaterTemperature")
    self.add_connection(reheat_coils_supply_air_temperature, east_reheat_coil, "measuredValue", "inletAirTemperature")

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


    self.add_connection(west_temperature_heating_controller, west_reheat_control_sensor, "y_valve", "measuredValue")
    self.add_connection(west_reheat_control_sensor, west_reheat_coil, "measuredValue", "valvePosition")
    self.add_connection(west_supply_damper, west_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, west_reheat_coil, "measuredValue", "supplyWaterTemperature")
    self.add_connection(reheat_coils_supply_air_temperature, west_reheat_coil, "measuredValue", "inletAirTemperature")

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

    # Add AHU fan
    supply_fan = tb.FanSystem(id="supply_fan", saveSimulationResult=True)
    self.add_connection(vent_airflow_sensor, supply_fan, "measuredValue", "airFlowRate")
    self.add_connection(supply_fan, vent_power_sensor, "Power", "measuredValue")

    # Add AHU heating coil
    supply_heating_coil = tb.CoilHeatingSystem(id="supply_heating_coil", saveSimulationResult=True)
    self.add_connection(vent_mixed_air_temp_sensor, supply_heating_coil, "measuredValue", "inletAirTemperature")
    self.add_connection(vent_airflow_sensor, supply_heating_coil, "measuredValue", "airFlowRate")
    self.add_connection(heating_coil_temperature_setpoint, supply_heating_coil, "scheduleValue", "outletAirTemperatureSetpoint")

    # Add AHU cooling coil
    supply_cooling_coil = tb.CoilCoolingSystem(id="supply_cooling_coil", saveSimulationResult=True)
    self.add_connection(supply_heating_coil, supply_cooling_coil, "outletAirTemperature", "inletAirTemperature")
    self.add_connection(vent_airflow_sensor, supply_cooling_coil, "measuredValue", "airFlowRate")
    self.add_connection(heating_coil_temperature_setpoint, supply_cooling_coil, "scheduleValue", "outletAirTemperatureSetpoint")
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
    self.add_connection(return_flow_junction_for_supply, self.components["reheat_coils_supply_air_temperature"], "airTemperatureOut", "measuredValue")

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


def get_model(id=None, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn

    if id is not None:
        model = tb.Model(id=id, saveSimulationResult=True)
    else:
        model = tb.Model(id="rooms_and_ahu_model_no_id", saveSimulationResult=True)
    
    model.load(fcn=fcn_, create_signature_graphs=False, validate_model=True, verbose=True, force_config_update=True)
    return model



def run(model = None):
    stepSize = 60  # Seconds
    
    startTime = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    if model is None:
        model = get_model()

    simulator = tb.Simulator()

    simulator.simulate(model = model,
                       startTime=startTime,
                        endTime=endTime,
                        stepSize=stepSize)

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
    startTime = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))

    model = get_model()

    ## Target parameters definition
    #Space parameters are estimated separately
    #CORE
    core_supply_damper = model.components["core_supply_damper"]
    core_exhaust_damper = model.components["core_exhaust_damper"]
    core_temp_controller = model.components["core_temperature_heating_controller"]
    #NORTH
    north_supply_damper = model.components["north_supply_damper"]
    north_exhaust_damper = model.components["north_exhaust_damper"]
    north_temp_controller = model.components["north_temperature_heating_controller"]
    #SOUTH
    south_supply_damper = model.components["south_supply_damper"]
    south_exhaust_damper = model.components["south_exhaust_damper"]
    south_temp_controller = model.components["south_temperature_heating_controller"]
    #EAST
    east_supply_damper = model.components["east_supply_damper"]
    east_exhaust_damper = model.components["east_exhaust_damper"]
    east_temp_controller = model.components["east_temperature_heating_controller"]
    #WEST
    west_supply_damper = model.components["west_supply_damper"]
    west_exhaust_damper = model.components["west_exhaust_damper"]
    west_temp_controller = model.components["west_temperature_heating_controller"]

    dampers_list = [core_supply_damper, core_exhaust_damper, north_supply_damper, north_exhaust_damper, south_supply_damper, south_exhaust_damper, east_supply_damper, east_exhaust_damper, west_supply_damper, west_exhaust_damper]

    targetParameters = {"private": {
                                    "k_coo": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 1, "lb": 1e-5, "ub": 10},
                                    "ti_coo": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 1, "lb": 1e-5, "ub": 10},
                                    "k_hea": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 1, "lb": 1e-5, "ub": 10},
                                    "ti_hea": {"components": [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller], "x0": 1, "lb": 1e-5, "ub": 10},
                                    "nominalAirFlowRate.hasValue": {"components": dampers_list, "x0": 3.5, "lb": 1e-2, "ub": 10},
                                    "a": {"components": dampers_list, "x0": 6.74, "lb": 0.5, "ub": 8}
                                    }
                        }
    
    """
    Parameters for each room:
    - Input & output damper:
        - a (shared) 
        - nominalAirFlowRate
    - PI controller Kp, Ki constants
    Required data points:
    Common data points:
    [x]Supply air temperature (hvac_reaAhu_TSup_y)
    [x]Supply air flow rate (hvac_reaAhu_V_flow_sup_y) from supply junction
    [x]Return air flow rate (hvac_reaAhu_V_flow_ret_y) from return junction
    [x]Outdoor air temperature (weaSta_reaWeaTWetBul_y) Figure how to synchronize with the outdoor environment data
    Per room data points:
    [x]Supply air temperature (hvac_reaZonCor_TSup_y, hvac_reaZonNor_TSup_y, hvac_reaZonSou_TSup_y, hvac_reaZonEas_TSup_y, hvac_reaZonWes_TSup_y)
    [x]Supply air flow rate (hvac_reaZonCor_V_flow_y, hvac_reaZonNor_V_flow_y, hvac_reaZonSou_V_flow_y, hvac_reaZonEas_V_flow_y, hvac_reaZonWes_V_flow_y)
    [x]CO2 concentration (hvac_reaZonCor_CO2Zon_y, hvac_reaZonNor_CO2Zon_y, hvac_reaZonSou_CO2Zon_y, hvac_reaZonEas_CO2Zon_y, hvac_reaZonWes_CO2Zon_y)
    [x]Indoor air temperature (hvac_reaZonCor_TZon_y, hvac_reaZonNor_TZon_y, hvac_reaZonSou_TZon_y, hvac_reaZonEas_TZon_y, hvac_reaZonWes_TZon_y)
    [x]Supply damper position (hvac_oveZonActCor_yDam_u, hvac_oveZonActNor_yDam_u, hvac_oveZonActSou_yDam_u, hvac_oveZonActEas_yDam_u, hvac_oveZonActWes_yDam_u)
    [x]Heating setpoint (hvac_oveZonSupCor_TZonHeaSet_u, hvac_oveZonSupNor_TZonHeaSet_u, hvac_oveZonSupSou_TZonHeaSet_u, hvac_oveZonSupEas_TZonHeaSet_u, hvac_oveZonSupWes_TZonHeaSet_u)
    [x]Cooling setpoint (hvac_oveZonSupCor_TZonCooSet_u, hvac_oveZonSupNor_TZonCooSet_u, hvac_oveZonSupSou_TZonCooSet_u, hvac_oveZonSupEas_TZonCooSet_u, hvac_oveZonSupWes_TZonCooSet_u)
    [x](Forecast values) Occupancy (Occupancy[cor], Occupancy[nor], Occupancy[sou], Occupancy[eas], Occupancy[wes]) 

    Model outputs (measuring devices):
    - Indoor air temperature (core_indoor_air_temp_sensor, north_indoor_air_temp_sensor, south_indoor_air_temp_sensor, east_indoor_air_temp_sensor, west_indoor_air_temp_sensor)
    - Damper position (core_supply_damper_position, north_supply_damper_position, south_supply_damper_position, east_supply_damper_position, west_supply_damper_position)
    - Room supply air flow rate (core_supply_airflow_sensor, north_supply_airflow_sensor, south_supply_airflow_sensor, east_supply_airflow_sensor, west_supply_airflow_sensor)
    - CO2 concentration (core_co2_sensor, north_co2_sensor, south_co2_sensor, east_co2_sensor, west_co2_sensor)
    - Total supply air flow rate (vent_supply_airflow_sensor)

    """


    percentile = 2
    targetMeasuringDevices = {
                             model.components["vent_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 10},
                             model.components["vent_return_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 10},
                             model.components["vent_return_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["core_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["core_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 3},
                             model.components["core_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["core_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             model.components["core_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["north_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["north_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 3},
                             model.components["north_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["north_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1}, 
                             model.components["north_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["south_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["south_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 3},
                             model.components["south_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["south_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             model.components["south_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["east_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["east_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 3},
                             model.components["east_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["east_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             model.components["east_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},

                             model.components["west_indoor_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["west_supply_airflow_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 3},
                             model.components["west_co2_sensor"]: {"standardDeviation": 10/percentile, "scale_factor": 400},
                             model.components["west_supply_damper_position"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
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

def load_and_print_parameters(filename):
    model = get_model()
    model.load_estimation_result(filename)

    #Print the resulting parameters

    print("Resulting parameters:")
    print_parameter_results(model)

def parameter_evaluation(data_points, parameter_filenames:dict, save_plots=False):
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
        #plt.show()


def load_model_and_params(model_id="rooms_and_ahu_model"):
    envelope_filepath = r"C:\Users\asces\OneDriveUni\Projects\RL_control\boptest_model\generated_files\models\only_rooms_estimation\model_parameters\estimation_results\LS_result\mix_day_most_accurate_08042025.pickle"
    vavs_filepath = r"C:\Users\asces\OneDriveUni\Projects\RL_control\boptest_model\generated_files\models\vav_controllers_param_est\model_parameters\estimation_results\LS_result\20250506_095811_ls.pickle"
    ahu_filepath = r"C:\Users\asces\OneDriveUni\Projects\RL_control\boptest_model\generated_files\models\only_ahu_model\model_parameters\estimation_results\LS_result\20250314_163600_ls.pickle"
    parameter_filenames = {"envelope": envelope_filepath, "vavs": vavs_filepath, "ahu": ahu_filepath}
    # Load model with estimated parameters and run simulation
    model = get_model(id=model_id)
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

    return model

if __name__ == "__main__":
    envelope_filepath = r"C:\Users\asces\OneDriveUni\Projects\RL_control\boptest_model\generated_files\models\only_rooms_estimation\model_parameters\estimation_results\LS_result\mix_day_most_accurate_08042025.pickle"
    vavs_filepath = r"C:\Users\asces\OneDriveUni\Projects\RL_control\boptest_model\generated_files\models\vav_controllers_param_est\model_parameters\estimation_results\LS_result\20250506_095811_ls.pickle"
    ahu_filepath = r"C:\Users\asces\OneDriveUni\Projects\RL_control\boptest_model\generated_files\models\only_ahu_model\model_parameters\estimation_results\LS_result\20250314_163600_ls.pickle"
    parameter_filenames = {"envelope": envelope_filepath, "vavs": vavs_filepath, "ahu": ahu_filepath}
    parameter_evaluation(model_output_points, parameter_filenames, save_plots=True)

    