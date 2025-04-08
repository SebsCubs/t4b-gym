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
        'component_id': 'vent_supply_airflow_sensor',
        'output_value': 'supplyAirflow',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_V_flow_sup_y_processed.csv'
    },
    {
        'component_id': 'vent_return_airflow_sensor',
        'output_value': 'returnAirflow',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_V_flow_ret_y_processed.csv'
    },
    {
        'component_id': 'vent_return_air_temp_sensor',
        'output_value': 'returnAirTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaAhu_TRet_y_processed.csv'
    },
    {
        'component_id': 'core_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonCor_TZon_y_processed.csv'
    },
    {
        'component_id': 'core_supply_airflow_sensor',
        'output_value': 'core_supplyAirflow',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonCor_V_flow_y_processed.csv'
    },
    {
        'component_id': 'core_co2_sensor',
        'output_value': 'core_indoorCo2Concentration',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonCor_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'core_supply_damper_position',
        'output_value': 'core_supplyDamperPosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActCor_yDam_u_processed.csv'
    },
    {
        'component_id': 'core_supply_air_temp_sensor',
        'output_value': 'core_supplyAirTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonCor_TSup_y_processed.csv'
    },
    {
        'component_id': 'north_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonNor_TZon_y_processed.csv'
    },
    {
        'component_id': 'north_supply_airflow_sensor',
        'output_value': 'north_supplyAirflow',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonNor_V_flow_y_processed.csv'
    },
    {
        'component_id': 'north_co2_sensor',
        'output_value': 'north_indoorCo2Concentration',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonNor_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'north_supply_damper_position',
        'output_value': 'north_supplyDamperPosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActNor_yDam_u_processed.csv'
    },
    {
        'component_id': 'north_supply_air_temp_sensor',
        'output_value': 'north_supplyAirTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonNor_TSup_y_processed.csv'
    },
    {
        'component_id': 'south_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonSou_TZon_y_processed.csv'
    },
    {
        'component_id': 'south_supply_airflow_sensor',
        'output_value': 'south_supplyAirflow',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonSou_V_flow_y_processed.csv'
    },
    {
        'component_id': 'south_co2_sensor',
        'output_value': 'south_indoorCo2Concentration',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonSou_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'south_supply_damper_position',
        'output_value': 'south_supplyDamperPosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActSou_yDam_u_processed.csv'
    },
    {
        'component_id': 'south_supply_air_temp_sensor',
        'output_value': 'south_supplyAirTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonSou_TSup_y_processed.csv'
    },
    {
        'component_id': 'east_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonEas_TZon_y_processed.csv'
    },
    {
        'component_id': 'east_supply_airflow_sensor',
        'output_value': 'east_supplyAirflow',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonEas_V_flow_y_processed.csv'
    },
    {
        'component_id': 'east_co2_sensor',
        'output_value': 'east_indoorCo2Concentration',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonEas_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'east_supply_damper_position',
        'output_value': 'east_supplyDamperPosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActEas_yDam_u_processed.csv'
    },
    {
        'component_id': 'east_supply_air_temp_sensor',
        'output_value': 'east_supplyAirTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonEas_TSup_y_processed.csv'
    },
    {
        'component_id': 'west_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonWes_TZon_y_processed.csv'
    },
    {
        'component_id': 'west_supply_airflow_sensor',
        'output_value': 'west_supplyAirflow',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonWes_V_flow_y_processed.csv'
    },
    {
        'component_id': 'west_co2_sensor',
        'output_value': 'west_indoorCo2Concentration',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonWes_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'west_supply_damper_position',
        'output_value': 'west_supplyDamperPosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActWes_yDam_u_processed.csv'
    },
    {
        'component_id': 'west_supply_air_temp_sensor',
        'output_value': 'west_supplyAirTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonWes_TSup_y_processed.csv'
    }
]

def fcn(self):
    '''
        The fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''
    #Add core temp setpoint profiles
    core_temperature_heating_setpoint = tb.ScheduleSystem(id="core_temperature_heating_setpoint", saveSimulationResult=True)
    core_temperature_cooling_setpoint = tb.ScheduleSystem(id="core_temperature_cooling_setpoint", saveSimulationResult=True)

    #Add core temp controller
    core_temperature_heating_controller = tb.VAVReheatControllerSystem(id="core_temperature_heating_controller", rat_v_flo_min=0.01, saveSimulationResult=True)
    self.add_connection(core_temperature_heating_setpoint, core_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(core_temperature_cooling_setpoint, core_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(self.components["core"], core_temperature_heating_controller, "indoorTemperature", "roomTemp")
    #self.add_connection(core_temperature_heating_controller, self.components["core"], "supplyAirTemp", "supplyAirTemperature")

    #Add core dampers
    core_supply_damper = tb.DamperSystem(id="core_supply_damper", saveSimulationResult=True)
    core_exhaust_damper = tb.DamperSystem(id="core_exhaust_damper", saveSimulationResult=True)
    self.add_connection(core_temperature_heating_controller, core_supply_damper, "y_dam", "damperPosition")
    self.add_connection(core_temperature_heating_controller, core_exhaust_damper, "y_dam", "damperPosition")
    self.add_connection(core_supply_damper, self.components["core"], "airFlowRate", "airFlowRate")

    # Add core heating coil
    core_reheating_coil = tb.CoilHeatingSystem(id="core_reheating_coil", saveSimulationResult=True)
    self.add_connection(core_temperature_heating_controller, core_reheating_coil, "supplyAirTemp", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], core_reheating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(self.components["core_supply_damper"], core_reheating_coil, "airFlowRate", "airFlowRate")
    self.remove_connection(self.components["vent_supply_air_temp_sensor"], self.components["core"], "measuredValue", "supplyAirTemperature") #This connection was generated from the semantic model but is not needed
    self.add_connection(core_reheating_coil, self.components["core"], "outletAirTemperatureSetpoint", "supplyAirTemperature")

    #Add core sensors
    core_co2_sensor = tb.SensorSystem(id="core_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["core"], core_co2_sensor, "indoorCo2Concentration", "core_indoorCo2Concentration")
    core_supply_air_temp_sensor = tb.SensorSystem(id="core_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(core_reheating_coil, core_supply_air_temp_sensor, "outletAirTemperature", "core_supplyAirTemperature")
    core_supply_airflow_sensor = tb.SensorSystem(id="core_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(self.components["core_supply_damper"], core_supply_airflow_sensor, "airFlowRate", "core_supplyAirflow")
    core_supply_damper_position = tb.SensorSystem(id="core_supply_damper_position", saveSimulationResult=True)
    self.add_connection(self.components["core_supply_damper"], core_supply_damper_position, "damperPosition", "core_supplyDamperPosition")
    
    
    #Add north temp setpoint profiles
    north_temperature_heating_setpoint = tb.ScheduleSystem(id="north_temperature_heating_setpoint", saveSimulationResult=True)
    north_temperature_cooling_setpoint = tb.ScheduleSystem(id="north_temperature_cooling_setpoint", saveSimulationResult=True)

    #Add north temp controller
    north_temperature_heating_controller = tb.VAVReheatControllerSystem(id="north_temperature_heating_controller", rat_v_flo_min=0.01, saveSimulationResult=True)
    self.add_connection(north_temperature_heating_setpoint, north_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(north_temperature_cooling_setpoint, north_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(self.components["north"], north_temperature_heating_controller, "indoorTemperature", "roomTemp")
    self.add_connection(north_temperature_heating_controller, self.components["north"], "supplyAirTemp", "supplyAirTemperature")

    #Add north dampers
    north_supply_damper = tb.DamperSystem(id="north_supply_damper", saveSimulationResult=True)
    north_exhaust_damper = tb.DamperSystem(id="north_exhaust_damper", saveSimulationResult=True)
    self.add_connection(north_temperature_heating_controller, north_supply_damper, "y_dam", "damperPosition")
    self.add_connection(north_temperature_heating_controller, north_exhaust_damper, "y_dam", "damperPosition")
    self.add_connection(north_supply_damper, self.components["north"], "airFlowRate", "airFlowRate")

    #Add north heating coil
    north_heating_coil = tb.CoilHeatingSystem(id="north_heating_coil", saveSimulationResult=True)
    self.add_connection(north_temperature_heating_controller, north_heating_coil, "supplyAirTemp", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], north_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(north_supply_damper, north_heating_coil, "airFlowRate", "airFlowRate")
    

    #Add north sensors
    north_co2_sensor = tb.SensorSystem(id="north_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["north"], north_co2_sensor, "indoorCo2Concentration", "north_indoorCo2Concentration")
    north_supply_air_temp_sensor = tb.SensorSystem(id="north_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(north_heating_coil, north_supply_air_temp_sensor, "outletAirTemperature", "north_supplyAirTemperature")
    north_supply_airflow_sensor = tb.SensorSystem(id="north_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(north_supply_damper, north_supply_airflow_sensor, "airFlowRate", "north_supplyAirflow")
    north_supply_damper_position = tb.SensorSystem(id="north_supply_damper_position", saveSimulationResult=True)
    self.add_connection(north_supply_damper, north_supply_damper_position, "damperPosition", "north_supplyDamperPosition")

    #Add south temp setpoint profiles
    south_temperature_heating_setpoint = tb.ScheduleSystem(id="south_temperature_heating_setpoint", saveSimulationResult=True)
    south_temperature_cooling_setpoint = tb.ScheduleSystem(id="south_temperature_cooling_setpoint", saveSimulationResult=True)

    #Add south temp controller
    south_temperature_heating_controller = tb.VAVReheatControllerSystem(id="south_temperature_heating_controller", rat_v_flo_min=0.01, saveSimulationResult=True)
    self.add_connection(south_temperature_heating_setpoint, south_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(south_temperature_cooling_setpoint, south_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(self.components["south"], south_temperature_heating_controller, "indoorTemperature", "roomTemp")
    self.add_connection(south_temperature_heating_controller, self.components["south"], "supplyAirTemp", "supplyAirTemperature")

    #Add south dampers
    south_supply_damper = tb.DamperSystem(id="south_supply_damper", saveSimulationResult=True)
    south_exhaust_damper = tb.DamperSystem(id="south_exhaust_damper", saveSimulationResult=True)
    self.add_connection(south_temperature_heating_controller, south_supply_damper, "y_dam", "damperPosition")
    self.add_connection(south_temperature_heating_controller, south_exhaust_damper, "y_dam", "damperPosition")
    self.add_connection(south_supply_damper, self.components["south"], "airFlowRate", "airFlowRate")
    #Add south heating coil
    south_heating_coil = tb.CoilHeatingSystem(id="south_heating_coil", saveSimulationResult=True)
    self.add_connection(south_temperature_heating_controller, south_heating_coil, "supplyAirTemp", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], south_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(south_supply_damper, south_heating_coil, "airFlowRate", "airFlowRate")
    

    #Add south sensors
    south_co2_sensor = tb.SensorSystem(id="south_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["south"], south_co2_sensor, "indoorCo2Concentration", "south_indoorCo2Concentration")
    south_supply_air_temp_sensor = tb.SensorSystem(id="south_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(south_heating_coil, south_supply_air_temp_sensor, "outletAirTemperature", "south_supplyAirTemperature")
    south_supply_airflow_sensor = tb.SensorSystem(id="south_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(south_supply_damper, south_supply_airflow_sensor, "airFlowRate", "south_supplyAirflow")
    south_supply_damper_position = tb.SensorSystem(id="south_supply_damper_position", saveSimulationResult=True)
    self.add_connection(south_supply_damper, south_supply_damper_position, "damperPosition", "south_supplyDamperPosition")

    #Add east temp setpoint profiles
    east_temperature_heating_setpoint = tb.ScheduleSystem(id="east_temperature_heating_setpoint", saveSimulationResult=True)
    east_temperature_cooling_setpoint = tb.ScheduleSystem(id="east_temperature_cooling_setpoint", saveSimulationResult=True)

    #Add east temp controller
    east_temperature_heating_controller = tb.VAVReheatControllerSystem(id="east_temperature_heating_controller", rat_v_flo_min=0.01, saveSimulationResult=True)
    self.add_connection(east_temperature_heating_setpoint, east_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(east_temperature_cooling_setpoint, east_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(self.components["east"], east_temperature_heating_controller, "indoorTemperature", "roomTemp")
    self.add_connection(east_temperature_heating_controller, self.components["east"], "supplyAirTemp", "supplyAirTemperature")


    #Add east dampers
    east_supply_damper = tb.DamperSystem(id="east_supply_damper", saveSimulationResult=True)
    east_exhaust_damper = tb.DamperSystem(id="east_exhaust_damper", saveSimulationResult=True)
    self.add_connection(east_temperature_heating_controller, east_supply_damper, "y_dam", "damperPosition")
    self.add_connection(east_temperature_heating_controller, east_exhaust_damper, "y_dam", "damperPosition")
    self.add_connection(east_supply_damper, self.components["east"], "airFlowRate", "airFlowRate")
    #Add east heating coil
    east_heating_coil = tb.CoilHeatingSystem(id="east_heating_coil", saveSimulationResult=True)
    self.add_connection(east_temperature_heating_controller, east_heating_coil, "supplyAirTemp", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], east_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(east_supply_damper, east_heating_coil, "airFlowRate", "airFlowRate")

    #Add east sensors
    east_co2_sensor = tb.SensorSystem(id="east_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["east"], east_co2_sensor, "indoorCo2Concentration", "east_indoorCo2Concentration")
    east_supply_air_temp_sensor = tb.SensorSystem(id="east_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(east_heating_coil, east_supply_air_temp_sensor, "outletAirTemperature", "east_supplyAirTemperature")
    east_supply_airflow_sensor = tb.SensorSystem(id="east_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(east_supply_damper, east_supply_airflow_sensor, "airFlowRate", "east_supplyAirflow")
    east_supply_damper_position = tb.SensorSystem(id="east_supply_damper_position", saveSimulationResult=True)
    self.add_connection(east_supply_damper, east_supply_damper_position, "damperPosition", "east_supplyDamperPosition")

    #Add west temp setpoint profiles
    west_temperature_heating_setpoint = tb.ScheduleSystem(id="west_temperature_heating_setpoint", saveSimulationResult=True)
    west_temperature_cooling_setpoint = tb.ScheduleSystem(id="west_temperature_cooling_setpoint", saveSimulationResult=True)

    #Add west temp controller
    west_temperature_heating_controller = tb.VAVReheatControllerSystem(id="west_temperature_heating_controller", rat_v_flo_min=0.01, saveSimulationResult=True)
    self.add_connection(west_temperature_heating_setpoint, west_temperature_heating_controller, "scheduleValue", "heatingsetpointValue")
    self.add_connection(west_temperature_cooling_setpoint, west_temperature_heating_controller, "scheduleValue", "coolingsetpointValue")
    self.add_connection(self.components["west"], west_temperature_heating_controller, "indoorTemperature", "roomTemp")
    self.add_connection(west_temperature_heating_controller, self.components["west"], "supplyAirTemp", "supplyAirTemperature")

    #Add west dampers
    west_supply_damper = tb.DamperSystem(id="west_supply_damper", saveSimulationResult=True)
    west_exhaust_damper = tb.DamperSystem(id="west_exhaust_damper", saveSimulationResult=True)
    self.add_connection(west_temperature_heating_controller, west_supply_damper, "y_dam", "damperPosition")
    self.add_connection(west_temperature_heating_controller, west_exhaust_damper, "y_dam", "damperPosition")
    self.add_connection(west_supply_damper, self.components["west"], "airFlowRate", "airFlowRate")
    #Add west heating coil
    west_heating_coil = tb.CoilHeatingSystem(id="west_heating_coil", saveSimulationResult=True)
    self.add_connection(west_temperature_heating_controller, west_heating_coil, "supplyAirTemp", "outletAirTemperatureSetpoint")
    self.add_connection(self.components["vent_supply_air_temp_sensor"], west_heating_coil, "supplyAirTemperature", "inletAirTemperature")
    self.add_connection(west_supply_damper, west_heating_coil, "airFlowRate", "airFlowRate")

    #Add west sensors
    west_co2_sensor = tb.SensorSystem(id="west_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["west"], west_co2_sensor, "indoorCo2Concentration", "west_indoorCo2Concentration")
    west_supply_air_temp_sensor = tb.SensorSystem(id="west_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(west_heating_coil, west_supply_air_temp_sensor, "outletAirTemperature", "west_supplyAirTemperature")
    west_supply_airflow_sensor = tb.SensorSystem(id="west_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(west_supply_damper, west_supply_airflow_sensor, "airFlowRate", "west_supplyAirflow")
    west_supply_damper_position = tb.SensorSystem(id="west_supply_damper_position", saveSimulationResult=True)
    self.add_connection(west_supply_damper, west_supply_damper_position, "damperPosition", "west_supplyDamperPosition")
    

    #Add supply junction
    supply_junction = tb.SupplyFlowJunctionSystem(id="supply_junction", saveSimulationResult=True)
    self.add_connection(core_supply_damper, supply_junction, "airFlowRate", "airFlowRateOut")
    self.add_connection(north_supply_damper, supply_junction, "airFlowRate", "airFlowRateOut")
    self.add_connection(south_supply_damper, supply_junction, "airFlowRate", "airFlowRateOut")
    self.add_connection(east_supply_damper, supply_junction, "airFlowRate", "airFlowRateOut")
    self.add_connection(west_supply_damper, supply_junction, "airFlowRate", "airFlowRateOut")
    
    vent_supply_airflow_sensor = tb.SensorSystem(id="vent_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(supply_junction, vent_supply_airflow_sensor, "airFlowRateIn", "supplyAirflow")

    #Connect return flow junction 
    return_junction = self.components["exhaust_flow_junction"]
    self.add_connection(core_exhaust_damper, return_junction, "airFlowRate", "airFlowRateIn")
    self.add_connection(north_exhaust_damper, return_junction, "airFlowRate", "airFlowRateIn")
    self.add_connection(south_exhaust_damper, return_junction, "airFlowRate", "airFlowRateIn")
    self.add_connection(east_exhaust_damper, return_junction, "airFlowRate", "airFlowRateIn")
    self.add_connection(west_exhaust_damper, return_junction, "airFlowRate", "airFlowRateIn")

    vent_return_airflow_sensor = tb.SensorSystem(id="vent_return_airflow_sensor", saveSimulationResult=True)
    self.add_connection(return_junction, vent_return_airflow_sensor, "airFlowRateOut", "returnAirflow")
    vent_return_air_temp_sensor = tb.SensorSystem(id="vent_return_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(return_junction, vent_return_air_temp_sensor, "airTemperatureOut", "returnAirTemperature")

def get_model(id=None, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn
    model = tb.Model(id="five_rooms_only_template", saveSimulationResult=True)
    
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), r"semantic_models\five_rooms_no_contr.xlsm")
    model.load(semantic_model_filename=filename, fcn=fcn_, create_signature_graphs=False, validate_model=True, verbose=True, force_config_update=True)
   
    model.load_estimation_result(r"C:\Users\asces\OneDriveUni\Projects\RL_control\boptest_model\generated_files\models\only_rooms_estimation\model_parameters\estimation_results\LS_result\mix_day_most_accurate_08042025.pickle")
    if id is not None:
        model.id = id
    return model

def run(model = None):
    stepSize = 60  # Seconds
    
    startTime = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    if model is None:
        model = get_model()
        #model._connect() #Neccessary with this fcn function modifying the model so much

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
        'core': ['C_air', 'C_int', 'C_boundary', 'R_int', 'R_boundary', 'Q_occ_gain', 'infiltration', 'T_boundary'],
        'other': ['C_wall', 'C_air', 'C_int', 'C_boundary', 'R_out', 'R_in', 'R_int', 
                 'R_boundary', 'f_wall', 'f_air', 'Q_occ_gain', 'infiltration', 'T_boundary']
    }
    
    for room in rooms:
        print(f"\n{room.upper()}:")
        params = space_params['core'] if room == 'core' else space_params['other']
        for param in params:
            value = getattr(model.components[room], param)
            print(f"{param}: {value}")
    
    # Controller parameters
    print("\nCONTROLLERS:")
    controller_params = ['k_coo', 'ti_coo', 'k_hea', 'ti_hea']
    
    for room in rooms:
        print(f"\n{room.upper()}:")
        controller_id = f"{room}_temperature_heating_controller"
        for param in controller_params:
            value = getattr(model.components[controller_id], param)
            print(f"{param}: {value}")
    
    # Damper parameters
    print("\nDAMPERS:")
    damper_params = ['a', "nominalAirFlowRate"]
    
    for room in rooms:
        print(f"\n{room.upper()}:")
        damper_id = f"{room}_supply_damper"
        for param in damper_params:
            value = getattr(model.components[damper_id], param)
            if param == "nominalAirFlowRate":
                print(f"{param}: {value.hasValue}")
            else:
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

def parameter_evaluation(data_points, parameter_filename, save_plots=False):
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
    model = get_model(id="five_rooms_only_template")
    model.load_estimation_result(parameter_filename)
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
        plt.show()
 
if __name__ == "__main__":
    parameter_filename = parameter_estimation()
    #parameter_filename = r"C:\Users\asces\OneDriveUni\Projects\RL_control\boptest_model\generated_files\models\five_rooms_only_template\model_parameters\estimation_results\LS_result\20250401_145631_ls.pickle"
    parameter_evaluation(model_output_points, parameter_filename, save_plots=True)