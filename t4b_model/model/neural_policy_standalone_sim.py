#Only code version
import twin4build as tb
import datetime
import twin4build.examples.utils as utils
import numpy as np
import torch.nn as nn
import torch
import json
from dateutil.tz import gettz 
import twin4build.utils.plot.plot as plot
import twin4build.utils.input_output_types as tps
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # Add the grandparent directory to the system path

# Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = os.path.join(uppath(os.path.abspath(__file__), 4), "Twin4Build")
    sys.path.append(file_path)

from RL_Algos.networks import PolicyNetwork

sys.setrecursionlimit(2000)

def insert_neural_policy_in_fcn(self:tb.Model, input_output_dictionary, policy_path=None):
        """
        Example of an input_output_dictionary:

        "[020B][020B_space_heater]": {
            "indoorTemperature": {
                "min": 0,
                "max": 40,
                "description": "Room 020B indoor temperature"
            },
            "indoorCo2Concentration": {
                "min": 0,
                "max": 4000,
                "description": "Room 020B indoor CO2 concentration"
            }
        }

        The outputs are the setpoints overrided from the input components that are setpoints, identified by the "scheduleValue" key.
        """
        # Validate schema - will raise error if invalid
        utils.validate_schema(input_output_dictionary)

        #Create the controller
        input_size = sum(len(signals) for signals in input_output_dictionary["input"].values())
        output_size = 5 + 2 + 1 #TODO: Make this dynamic

        policy = PolicyNetwork(input_size, output_size, action_bound=1.0)

        #Load the policy model
        if policy_path is not None:
            policy.load_state_dict(torch.load(policy_path))

        neural_policy_controller = tb.NeuralPolicyControllerSystem(
            input_size = input_size,
            output_size = output_size,
            input_output_schema = input_output_dictionary,
            policy_model = policy,
            saveSimulationResult = True,
            id = "neural_controller"
        )

        #Find the existing connections to the setpoint-receiving components

        setpoint_delivery_components = []
        for input_component_key in input_output_dictionary["input"]:
            for input_signal_key in input_output_dictionary["input"][input_component_key]:
                if input_signal_key == "scheduleValue":
                    setpoint_delivery_components.append(input_component_key)

        #1. Find the existing connections to the setpoint-receiving components
        #2. Remove the existing connections
        #3. Store the setpoint-receiving components
        setpoint_connections = {component_key: {} for component_key in setpoint_delivery_components}
        for setpoint_component_key in setpoint_delivery_components:
            sender_component = self.components[setpoint_component_key]

            for connection_point in sender_component.connectedThrough[0].connectsSystemAt:
                receiver_component = connection_point.connectionPointOf
                setpoint_receiver_signal_key = connection_point.receiverPropertyName
                receiver_component_key = receiver_component.id
                setpoint_connections[setpoint_component_key][receiver_component_key] = setpoint_receiver_signal_key

            #Deleting one connection deletes all of them in this case, since they all have "scheduleValue" as the sender signal key    
            self.remove_connection_multiple_receivers(sender_component, "scheduleValue")


        # Add the connections to the setpoint-receiving components
        for sender_key, receivers in setpoint_connections.items():
            for receiver_key, receiver_signal in receivers.items():
                output_signal_key = f"{sender_key}_{receiver_key}_input_signal"
                neural_policy_controller.output[output_signal_key] = tps.Scalar()
                self.add_connection(
                    neural_policy_controller,
                    self.components[receiver_key],
                    output_signal_key,
                    receiver_signal
                )

        #Add the input connections
        for component_key in input_output_dictionary["input"]:
            try:
                sender_component = self.components[component_key]
            except KeyError:
                print(f"Could not find component {component_key}")
                continue
            receiving_component = neural_policy_controller
            for input_signal_key in input_output_dictionary["input"][component_key]:    
                self.add_connection(
                    sender_component,
                    receiving_component,
                    input_signal_key,
                    "actualValue"
                )

        # Define a custom initial dictionary for the NeuralController outputs:
        custom_initial = {"neural_controller": {f"{component_key}": tps.Scalar(0) for component_key in neural_policy_controller.output.keys()}}
        self.set_custom_initial_dict(custom_initial)

        return neural_policy_controller
        

def fcn(self):
    '''
        The fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''
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
    outdoor_environment = self.get_component_by_class(self.components, tb.OutdoorEnvironmentSystem)[0]
    self.add_connection(outdoor_environment, supply_water_temperature_schedule, "outdoorTemperature", "outdoorTemperature")
    spaces = self.get_component_by_class(self.components, tb.BuildingSpace1AdjBoundaryFMUSystem)
    spaces.extend(self.get_component_by_class(self.components, tb.BuildingSpace2AdjBoundaryFMUSystem))
    spaces.extend(self.get_component_by_class(self.components, tb.BuildingSpace11AdjBoundaryFMUSystem))
    spaces.extend(self.get_component_by_class(self.components, tb.BuildingSpace1AdjBoundaryOutdoorFMUSystem))
    spaces.extend(self.get_component_by_class(self.components, tb.BuildingSpace2AdjBoundaryOutdoorFMUSystem))
    spaces.extend(self.get_component_by_class(self.components, tb.BuildingSpace11AdjBoundaryOutdoorFMUSystem))
    for space in spaces:
        self.add_connection(supply_water_temperature_schedule, space, 
                            "scheduleValue", "supplyWaterTemperature")
        
    #Load the input/output dictionary from the file policy_input_output.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "policy_input_output.json")
    with open(json_path) as f:
        input_output_dictionary = json.load(f)

    neural_policy_controller = insert_neural_policy_in_fcn(self, input_output_dictionary, policy_path= None)

if __name__ == "__main__":
    # Create a new model
    model = tb.Model(id="neural_policy_1stfloor", saveSimulationResult=True)

    filename = os.path.join(uppath(os.path.abspath(__file__), 1), "fan_flow_configuration_template_DP37_full_no_cooling.xlsm")

    model.load(semantic_model_filename=filename, fcn=fcn, create_signature_graphs=False, validate_model=True, verbose=False, force_config_update=True)

    #model.components["neural_controller"].policy.load_state_dict(torch.load(r"C:\Users\asces\OneDriveUni\Projects\Adrenalin_BOPTEST_Challenge\RL_control\best_policy.pth"))
    model.components["neural_controller"].policy.load_state_dict(torch.load(r"C:\Users\asces\OneDriveUni\Projects\Adrenalin_BOPTEST_Challenge\RL_control\t4b_model\model\best_policy.pth"))
    #Run a simulation
    stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2024, month=1, day=5, hour=0, minute=0, second=0,
                                  tzinfo=gettz("Europe/Copenhagen"))

    endTime = datetime.datetime(year=2024, month=1, day=6, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))

    simulator = tb.Simulator()
    simulator.simulate(model, startTime=startTime, endTime=endTime, stepSize=stepSize)
    print("Simulation completed successfully!")

    # Plot the results using plot_component
    space_id = '[020B][020B_space_heater]'
    
    #print the total energy consumption
    energy = np.array(model.components[space_id].savedOutput['spaceHeaterPower'])
    print(f"Total energy consumption: {energy.sum()} Wh")

    #Print the deviation from the setpoint
    deviation = np.array(model.components[space_id].savedOutput['indoorTemperature']) - 21
    print(f"Total deviation from setpoint: {deviation.sum()} °C")

    # Temperature plot
    plot.plot_component(
        simulator,
        components_1axis=[(space_id, 'indoorTemperature'),("neural_controller", '020B_temperature_heating_setpoint_[020B_co2_controller][020B_damper_heating_controller]_input_signal'),("neural_controller", '020B_temperature_heating_setpoint_020B_temperature_controller_input_signal')],
        ylabel_1axis='Room Temperature [°C] (Actual and Setpoint)',
        show=True
    )

    # CO2 plot
    plot.plot_component(
        simulator,
        components_1axis=[(space_id, 'indoorCo2Concentration'),("neural_controller", '020B_co2_setpoint_[020B_co2_controller][020B_damper_heating_controller]_input_signal')],
        ylabel_1axis='CO2 Concentration [ppm] (Actual and Setpoint)',
        show=True
    )

    # Duct temperature plot
    plot.plot_component(
        simulator,
        components_1axis=[("supply_air_setpoint", 'scheduleValue'), ("neural_controller", "supply_air_setpoint_heating_coil_controller_input_signal"),],
        ylabel_1axis='Duct Temperature [°C] (Original setpoint and Neural Controller setpoint)',
        show=True
    )
    
    # 020B occupancy plot
    plot.plot_component(
        simulator,
        components_1axis=[("020B_occupancy_profile", 'scheduleValue')],
        ylabel_1axis='Occupancy 020B (Actual)',
        show=True
    )

