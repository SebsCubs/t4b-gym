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
        'component_id': 'core_supply_damper_position_sensor',
        'output_value': 'core_supplyDamperPosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActCor_yDam_u_processed.csv'
    },
    {
        'component_id': 'core_supply_air_temp_sensor',
        'output_value': 'core_supplyAirTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonCor_TSup_y_processed.csv'
    },
    {
        'component_id': 'core_reheat_control_sensor',
        'output_value': 'valvePosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActCor_yReaHea_u_processed.csv'
    },
    {
        'component_id': 'core_reheat_coil_return_water_temperature',
        'output_value': 'core_inletWaterTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/coils_outlet_water_temperature.csv'
    },
    {
        'component_id': 'north_supply_damper_position_sensor',
        'output_value': 'north_supplyDamperPosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActNor_yDam_u_processed.csv'
    },
    {
        'component_id': 'north_supply_air_temp_sensor',
        'output_value': 'north_supplyAirTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonNor_TSup_y_processed.csv'
    },
    {
        'component_id': 'north_reheat_control_sensor',
        'output_value': 'valvePosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActNor_yReaHea_u_processed.csv'
    },
    {
        'component_id': 'north_reheat_coil_return_water_temperature',
        'output_value': 'north_inletWaterTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/coils_outlet_water_temperature.csv'
    },
    {
        'component_id': 'south_supply_damper_position_sensor',
        'output_value': 'south_supplyDamperPosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActSou_yDam_u_processed.csv'
    },
    {
        'component_id': 'south_supply_air_temp_sensor',
        'output_value': 'south_supplyAirTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonSou_TSup_y_processed.csv'
    },
    {
        'component_id': 'south_reheat_control_sensor',
        'output_value': 'valvePosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActSou_yReaHea_u_processed.csv'
    },
    {
        'component_id': 'south_reheat_coil_return_water_temperature',
        'output_value': 'south_inletWaterTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/coils_outlet_water_temperature.csv'
    },
    {
        'component_id': 'east_supply_damper_position_sensor',
        'output_value': 'east_supplyDamperPosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActEas_yDam_u_processed.csv'
    },
    {
        'component_id': 'east_supply_air_temp_sensor',
        'output_value': 'east_supplyAirTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonEas_TSup_y_processed.csv'
    },
    {
        'component_id': 'east_reheat_control_sensor',
        'output_value': 'valvePosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActEas_yReaHea_u_processed.csv'
    },
    {
        'component_id': 'east_reheat_coil_return_water_temperature',
        'output_value': 'east_inletWaterTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/coils_outlet_water_temperature.csv'
    },
    {
        'component_id': 'west_supply_damper_position_sensor',
        'output_value': 'west_supplyDamperPosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActWes_yDam_u_processed.csv'
    },
    {
        'component_id': 'west_supply_air_temp_sensor',
        'output_value': 'west_supplyAirTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonWes_TSup_y_processed.csv'
    },
    {
        'component_id': 'west_reheat_control_sensor',
        'output_value': 'valvePosition',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_oveZonActWes_yReaHea_u_processed.csv'
    },
    {
        'component_id': 'west_reheat_coil_return_water_temperature',
        'output_value': 'west_inletWaterTemperature',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/coils_outlet_water_temperature.csv'
    }
]

def fcn(self):
    '''
        The fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
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
    core_indoor_temp_sensor = tb.SensorSystem(id="core_indoor_temp_sensor", saveSimulationResult=True)
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
    self.add_connection(core_indoor_temp_sensor, core_temperature_heating_controller, "indoorTemperature", "roomTemp")

       
    self.add_connection(core_temperature_heating_controller, core_reheat_control_sensor, "y_valve", "valvePosition")
    self.add_connection(core_reheat_control_sensor, core_reheat_coil, "valvePosition", "valvePosition") 
    self.add_connection(core_supply_damper, core_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, core_reheat_coil, "supplyWaterTemperature", "supplyWaterTemperature")
    self.add_connection(reheat_coils_supply_air_temperature, core_reheat_coil, "inletAirTemperature", "inletAirTemperature")

    self.add_connection(core_temperature_heating_controller, core_supply_damper_position_sensor, "y_dam", "core_supplyDamperPosition")
    self.add_connection(core_supply_damper_position_sensor, core_supply_damper, "core_supplyDamperPosition", "damperPosition")

    self.add_connection(core_reheat_coil, core_supply_air_temp_sensor, "outletAirTemperature", "core_supplyAirTemperature")
    self.add_connection(core_reheat_coil, core_reheat_coil_return_water_temperature, "inletWaterTemperature", "core_inletWaterTemperature")

    #Add north components
    north_temperature_heating_setpoint = tb.ScheduleSystem(id="north_temperature_heating_setpoint", saveSimulationResult=True)
    north_temperature_cooling_setpoint = tb.ScheduleSystem(id="north_temperature_cooling_setpoint", saveSimulationResult=True)
    north_temperature_heating_controller = tb.VAVReheatControllerSystem(id="north_temperature_heating_controller", rat_v_flo_min=0.15, saveSimulationResult=True)
    north_reheat_control_sensor = tb.SensorSystem(id="north_reheat_control_sensor", saveSimulationResult=True)
    north_supply_air_temp_sensor = tb.SensorSystem(id="north_supply_air_temp_sensor", saveSimulationResult=True)    
    north_supply_damper_position_sensor = tb.SensorSystem(id="north_supply_damper_position_sensor", saveSimulationResult=True)
    north_indoor_temp_sensor = tb.SensorSystem(id="north_indoor_temp_sensor", saveSimulationResult=True)
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
    self.add_connection(north_indoor_temp_sensor, north_temperature_heating_controller, "indoorTemperature", "roomTemp")

    self.add_connection(north_temperature_heating_controller, north_reheat_control_sensor, "y_valve", "valvePosition")
    self.add_connection(north_reheat_control_sensor, north_reheat_coil, "valvePosition", "valvePosition")
    self.add_connection(north_supply_damper, north_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, north_reheat_coil, "supplyWaterTemperature", "supplyWaterTemperature")
    self.add_connection(reheat_coils_supply_air_temperature, north_reheat_coil, "inletAirTemperature", "inletAirTemperature")

    self.add_connection(north_temperature_heating_controller, north_supply_damper_position_sensor, "y_dam", "north_supplyDamperPosition")
    self.add_connection(north_supply_damper_position_sensor, north_supply_damper, "north_supplyDamperPosition", "damperPosition")
    
    self.add_connection(north_reheat_coil, north_supply_air_temp_sensor, "outletAirTemperature", "north_supplyAirTemperature")
    self.add_connection(north_reheat_coil, north_reheat_coil_return_water_temperature, "inletWaterTemperature", "north_inletWaterTemperature")
    #Add south components
    south_temperature_heating_setpoint = tb.ScheduleSystem(id="south_temperature_heating_setpoint", saveSimulationResult=True)
    south_temperature_cooling_setpoint = tb.ScheduleSystem(id="south_temperature_cooling_setpoint", saveSimulationResult=True)
    south_temperature_heating_controller = tb.VAVReheatControllerSystem(id="south_temperature_heating_controller", rat_v_flo_min=0.15, saveSimulationResult=True)
    south_reheat_control_sensor = tb.SensorSystem(id="south_reheat_control_sensor", saveSimulationResult=True)
    south_supply_air_temp_sensor = tb.SensorSystem(id="south_supply_air_temp_sensor", saveSimulationResult=True)    
    south_supply_damper_position_sensor = tb.SensorSystem(id="south_supply_damper_position_sensor", saveSimulationResult=True)
    south_indoor_temp_sensor = tb.SensorSystem(id="south_indoor_temp_sensor", saveSimulationResult=True)
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
    self.add_connection(south_indoor_temp_sensor, south_temperature_heating_controller, "indoorTemperature", "roomTemp")

    self.add_connection(south_temperature_heating_controller, south_reheat_control_sensor, "y_valve", "valvePosition")
    self.add_connection(south_reheat_control_sensor, south_reheat_coil, "valvePosition", "valvePosition")
    self.add_connection(south_supply_damper, south_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, south_reheat_coil, "supplyWaterTemperature", "supplyWaterTemperature")
    self.add_connection(reheat_coils_supply_air_temperature, south_reheat_coil, "inletAirTemperature", "inletAirTemperature")

    self.add_connection(south_temperature_heating_controller, south_supply_damper_position_sensor, "y_dam", "south_supplyDamperPosition")
    self.add_connection(south_supply_damper_position_sensor, south_supply_damper, "south_supplyDamperPosition", "damperPosition")
    
    self.add_connection(south_reheat_coil, south_supply_air_temp_sensor, "outletAirTemperature", "south_supplyAirTemperature")
    self.add_connection(south_reheat_coil, south_reheat_coil_return_water_temperature, "inletWaterTemperature", "south_inletWaterTemperature")

    #Add east components
    east_temperature_heating_setpoint = tb.ScheduleSystem(id="east_temperature_heating_setpoint", saveSimulationResult=True)
    east_temperature_cooling_setpoint = tb.ScheduleSystem(id="east_temperature_cooling_setpoint", saveSimulationResult=True)
    east_temperature_heating_controller = tb.VAVReheatControllerSystem(id="east_temperature_heating_controller", rat_v_flo_min=0.15, saveSimulationResult=True)
    east_reheat_control_sensor = tb.SensorSystem(id="east_reheat_control_sensor", saveSimulationResult=True)
    east_supply_air_temp_sensor = tb.SensorSystem(id="east_supply_air_temp_sensor", saveSimulationResult=True)    
    east_supply_damper_position_sensor = tb.SensorSystem(id="east_supply_damper_position_sensor", saveSimulationResult=True)
    east_indoor_temp_sensor = tb.SensorSystem(id="east_indoor_temp_sensor", saveSimulationResult=True)
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
    self.add_connection(east_indoor_temp_sensor, east_temperature_heating_controller, "indoorTemperature", "roomTemp")

    self.add_connection(east_temperature_heating_controller, east_reheat_control_sensor, "y_valve", "valvePosition")
    self.add_connection(east_reheat_control_sensor, east_reheat_coil, "valvePosition", "valvePosition")
    self.add_connection(east_supply_damper, east_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, east_reheat_coil, "supplyWaterTemperature", "supplyWaterTemperature")
    self.add_connection(reheat_coils_supply_air_temperature, east_reheat_coil, "inletAirTemperature", "inletAirTemperature")

    self.add_connection(east_temperature_heating_controller, east_supply_damper_position_sensor, "y_dam", "east_supplyDamperPosition")
    self.add_connection(east_supply_damper_position_sensor, east_supply_damper, "east_supplyDamperPosition", "damperPosition")
    
    self.add_connection(east_reheat_coil, east_supply_air_temp_sensor, "outletAirTemperature", "east_supplyAirTemperature")
    self.add_connection(east_reheat_coil, east_reheat_coil_return_water_temperature, "inletWaterTemperature", "east_inletWaterTemperature")

    #Add west components
    west_temperature_heating_setpoint = tb.ScheduleSystem(id="west_temperature_heating_setpoint", saveSimulationResult=True)
    west_temperature_cooling_setpoint = tb.ScheduleSystem(id="west_temperature_cooling_setpoint", saveSimulationResult=True)
    west_temperature_heating_controller = tb.VAVReheatControllerSystem(id="west_temperature_heating_controller", rat_v_flo_min=0.15, saveSimulationResult=True)
    west_supply_air_temp_sensor = tb.SensorSystem(id="west_supply_air_temp_sensor", saveSimulationResult=True)    
    west_supply_damper_position_sensor = tb.SensorSystem(id="west_supply_damper_position_sensor", saveSimulationResult=True)
    west_indoor_temp_sensor = tb.SensorSystem(id="west_indoor_temp_sensor", saveSimulationResult=True)
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
    self.add_connection(west_indoor_temp_sensor, west_temperature_heating_controller, "indoorTemperature", "roomTemp")

    self.add_connection(west_temperature_heating_controller, west_reheat_control_sensor, "y_valve", "valvePosition")
    self.add_connection(west_reheat_control_sensor, west_reheat_coil, "valvePosition", "valvePosition")
    self.add_connection(west_supply_damper, west_reheat_coil, "airFlowRate", "airFlowRate")
    self.add_connection(reheat_coils_supply_water_temperature, west_reheat_coil, "supplyWaterTemperature", "supplyWaterTemperature")
    self.add_connection(reheat_coils_supply_air_temperature, west_reheat_coil, "inletAirTemperature", "inletAirTemperature")

    self.add_connection(west_temperature_heating_controller, west_supply_damper_position_sensor, "y_dam", "west_supplyDamperPosition")
    self.add_connection(west_supply_damper_position_sensor, west_supply_damper, "west_supplyDamperPosition", "damperPosition")
    
    self.add_connection(west_reheat_coil, west_supply_air_temp_sensor, "outletAirTemperature", "west_supplyAirTemperature")
    self.add_connection(west_reheat_coil, west_reheat_coil_return_water_temperature, "inletWaterTemperature", "west_inletWaterTemperature")

def get_model(id=None, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn
    model = tb.Model(id="vav_controllers_param_est", saveSimulationResult=True)
    
    model.load(fcn=fcn_, create_signature_graphs=False, validate_model=True, verbose=True, force_config_update=True)
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
    
    #Coil parameters    
    print("\nCOILS:")
    coil_params = ['m1_flow_nominal', 
                   'm2_flow_nominal', 
                   'tau1', 
                   'tau2', 
                   'tau_m', 
                   'nominalUa',
                   'mFlowValve_nominal', 
                   'flowCoefficient', 
                   'mFlowPump_nominal',
                   'KvCheckValve', 
                   'dp1_nominal', 
                   'dpPump', 
                   'dpSystem', 
                   'dpFixedSystem',
                   'tau_w_inlet',
                   'tau_w_outlet',
                   'tau_air_outlet']
    for room in rooms:
        print(f"\n{room.upper()}:")
        coil_id = f"{room}_reheat_coil"
        for param in coil_params:
            value = getattr(model.components[coil_id], param)
            if param == "nominalUa" or param == "flowCoefficient":
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
    core_temp_controller = model.components["core_temperature_heating_controller"]
    core_supply_damper = model.components["core_supply_damper"]
    core_reheat_coil = model.components["core_reheat_coil"]
    #NORTH
    north_temp_controller = model.components["north_temperature_heating_controller"]
    north_supply_damper = model.components["north_supply_damper"]
    north_reheat_coil = model.components["north_reheat_coil"]
    #SOUTH
    south_temp_controller = model.components["south_temperature_heating_controller"]
    south_supply_damper = model.components["south_supply_damper"]
    south_reheat_coil = model.components["south_reheat_coil"]
    #EAST
    east_temp_controller = model.components["east_temperature_heating_controller"]
    east_supply_damper = model.components["east_supply_damper"]
    east_reheat_coil = model.components["east_reheat_coil"]
    #WEST
    west_temp_controller = model.components["west_temperature_heating_controller"]
    west_supply_damper = model.components["west_supply_damper"]
    west_reheat_coil = model.components["west_reheat_coil"]

    dampers_list = [core_supply_damper, north_supply_damper, south_supply_damper, east_supply_damper, west_supply_damper]
    coils_list = [core_reheat_coil, north_reheat_coil, south_reheat_coil, east_reheat_coil, west_reheat_coil]
    controllers_list = [core_temp_controller, north_temp_controller, south_temp_controller, east_temp_controller, west_temp_controller]
    
    targetParameters = {"private": {
                                    "k_coo": {"components": controllers_list, "x0": 1, "lb": 1e-5, "ub": 10},
                                    "ti_coo": {"components": controllers_list, "x0": 1, "lb": 1e-5, "ub": 10},
                                    "k_hea": {"components": controllers_list, "x0": 1, "lb": 1e-5, "ub": 10},
                                    "ti_hea": {"components": controllers_list, "x0": 1, "lb": 1e-5, "ub": 10},

                                    "a": {"components": dampers_list, "x0": 1, "lb": 1e-5, "ub": 10},
                                    
                                    "m1_flow_nominal": {"components": coils_list, "x0": 2.794192, "lb": 1e-5, "ub": 10},
                                    "tau1": {"components": coils_list, "x0": 11.74021, "lb": 1, "ub": 50}, 
                                    "tau2": {"components": coils_list, "x0": 4.76927, "lb": 1, "ub": 50}, 
                                    "tau_m": {"components": coils_list, "x0": 16.072749, "lb": 1, "ub": 50}, 
                                    "nominalUa.hasValue": {"components": coils_list, "x0": 1637.3321, "lb": 0, "ub": 10000}, 
                                    "mFlowValve_nominal": {"components": coils_list, "x0": 0.6770, "lb": 0, "ub": 10},
                                    "flowCoefficient.hasValue": {"components": coils_list, "x0": 1, "lb": 0, "ub": 10}, 
                                    "mFlowPump_nominal": {"components": coils_list, "x0": 0.0026929, "lb": 0, "ub": 10},
                                    "KvCheckValve": {"components": coils_list, "x0": 1, "lb": 0, "ub": 100}, 
                                    "dp1_nominal": {"components": coils_list, "x0": 32127.83069, "lb": 0, "ub": 100000}, 
                                    "dpPump": {"components": coils_list, "x0": 1347.48462, "lb": 0, "ub": 100000},
                                    "dpSystem": {"components": coils_list, "x0": 7198.2045, "lb": 0, "ub": 100000}, 
                                    "dpFixedSystem": {"components": coils_list, "x0": 34616.7005, "lb": 0, "ub": 100000}, 
                                    }
                        }
    
    """
    Parameters for each room:
    - PI controller Kp, Ki constants
    - Damper params
    - Re-heat coil params

    Required data points:
    [x]Supply air temperature (hvac_reaZonCor_TSup_y, hvac_reaZonNor_TSup_y, hvac_reaZonSou_TSup_y, hvac_reaZonEas_TSup_y, hvac_reaZonWes_TSup_y)
    [x]Supply air flow rate (hvac_reaZonCor_V_flow_y, hvac_reaZonNor_V_flow_y, hvac_reaZonSou_V_flow_y, hvac_reaZonEas_V_flow_y, hvac_reaZonWes_V_flow_y)
    [x]Indoor air temperature (hvac_reaZonCor_TZon_y, hvac_reaZonNor_TZon_y, hvac_reaZonSou_TZon_y, hvac_reaZonEas_TZon_y, hvac_reaZonWes_TZon_y)
    [x]Supply damper position (hvac_oveZonActCor_yDam_u, hvac_oveZonActNor_yDam_u, hvac_oveZonActSou_yDam_u, hvac_oveZonActEas_yDam_u, hvac_oveZonActWes_yDam_u)
    [x]Heating setpoint (hvac_oveZonSupCor_TZonHeaSet_u, hvac_oveZonSupNor_TZonHeaSet_u, hvac_oveZonSupSou_TZonHeaSet_u, hvac_oveZonSupEas_TZonHeaSet_u, hvac_oveZonSupWes_TZonHeaSet_u)
    [x]Cooling setpoint (hvac_oveZonSupCor_TZonCooSet_u, hvac_oveZonSupNor_TZonCooSet_u, hvac_oveZonSupSou_TZonCooSet_u, hvac_oveZonSupEas_TZonCooSet_u, hvac_oveZonSupWes_TZonCooSet_u)
    []Reheat control signal (hvac_oveZonActCor_yReaHea_u, hvac_oveZonActNor_yReaHea_u, hvac_oveZonActSou_yReaHea_u, hvac_oveZonActEas_yReaHea_u, hvac_oveZonActWes_yReaHea_u)
    []Re-heat coil inlet water temperature (hvac_reaAhu_THeaCoiSup_y) or constant 45 degrees
    []Re-heat coil inlet air temperature (hvac_reaAhu_TSup_y) or constant 35 degrees
    
    Model outputs (measuring devices):
    - Damper position (core_supply_damper_position, north_supply_damper_position, south_supply_damper_position, east_supply_damper_position, west_supply_damper_position)
    - Room supply air flow rate (core_supply_airflow_sensor, north_supply_airflow_sensor, south_supply_airflow_sensor, east_supply_airflow_sensor, west_supply_airflow_sensor)
    """


    percentile = 2
    targetMeasuringDevices = {

                             model.components["core_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["core_supply_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             model.components["core_reheat_control_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             
                             model.components["north_supply_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1}, 
                             model.components["north_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["north_reheat_control_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},

                             model.components["south_supply_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             model.components["south_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["south_reheat_control_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},

                             model.components["east_supply_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             model.components["east_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["east_reheat_control_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},

                             model.components["west_supply_damper_position_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
                             model.components["west_supply_air_temp_sensor"]: {"standardDeviation": 0.1/percentile, "scale_factor": 20},
                             model.components["west_reheat_control_sensor"]: {"standardDeviation": 0.01/percentile, "scale_factor": 1},
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

    

    # Set west reheat coil parameters the results from the param estimation for west break the FMU
    # Results: C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/generated_files/models/vav_controllers_param_est/model_parameters/estimation_results/LS_result/20250416_122059_ls.pickle

    west_coil = model.components["west_reheat_coil"]
    west_coil.m1_flow_nominal = 2.489581121812883
    west_coil.m2_flow_nominal = 1.10274451
    west_coil.tau1 = 11.740211593307396
    west_coil.tau2 = 4.769272915298189
    west_coil.tau_m = 16.072749602786455
    west_coil.nominalUa.hasValue = 1637.3321140160149
    west_coil.mFlowValve_nominal = 0.6770486544227082
    west_coil.flowCoefficient.hasValue = 1.0000000000013491
    west_coil.mFlowPump_nominal = 0.0026929261007314904
    west_coil.KvCheckValve = 0.9999999999994005
    west_coil.dp1_nominal = 32127.830691396135
    west_coil.dpPump = 1347.484623168432
    west_coil.dpSystem = 7198.204507093697
    west_coil.dpFixedSystem = 34616.700590515815
    west_coil.tau_w_inlet = 1.0
    west_coil.tau_w_outlet = 1.0
    west_coil.tau_air_outlet = 1.0

    
    
    simulator = run(model)
    stepSize = simulator.stepSize
    plotting_stepSize = 300
    
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
    #parameter_filename = parameter_estimation()
    parameter_filename = r"C:\Users\asces\OneDriveUni\Projects\RL_control\boptest_model\generated_files\models\vav_controllers_param_est\model_parameters\estimation_results\LS_result\20250416_122059_ls.pickle"
    parameter_evaluation(model_output_points, parameter_filename, save_plots=True)