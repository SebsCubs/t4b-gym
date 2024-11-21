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
from RL_Algos.networks import PolicyNetwork

sys.setrecursionlimit(2000)

def insert_neural_policy_in_fcn(self:tb.Model, input_output_dictionary, policy_path=None):
        """
        The input/output dictionary contains information on the input and output signals of the controller.
        These signals must match the component and signal keys to replace in the model
        The input dictionary will have items like this:
            "component_key": {
                "component_output_signal_key": {
                    "min": 0,
                    "max": 1,
                    "description": "Description of the signal"
                }
            }
        Whilst the output items will have a similar structure but for the output signals:
            "component_key": {
                "component_input_signal_key": {
                    "min": 0,
                    "max": 1,
                    "description": "Description of the signal"
                }
            }
        Note that the input signals must contain the key for the output compoenent signal and the output signals must contain the key for the input component signal

        This function instantiates the controller and adds it to the model.
        Then it goes through the input dictionary adding connection to the input signals
        Then it goes through the output dictionary finding the corresponding existing connections, deleting the existing connections and adding the new connections
        """
        try:
            utils.validate_schema(input_output_dictionary)
        except Exception as e:
            print("Validation error:", e)
            return

        #Create the controller
        input_size = len(input_output_dictionary["input"])
        output_size = len(input_output_dictionary["output"])

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
            if input_output_dictionary["input"][input_component_key]["signal_key"] == "scheduleValue":
                setpoint_delivery_components.append(input_component_key)

        #1. Find the existing connections to the setpoint-receiving components
        #2. Remove the existing connections
        #3. Store the setpoint-receiving components
        setpoint_connections = {component_key: {} for component_key in setpoint_delivery_components}
        for setpoint_component_key in setpoint_delivery_components:
            sender_component = self.component_dict[setpoint_component_key]
            try:
                for connection_point in sender_component.connectedThrough[0].connectsSystemAt:
                    receiver_component = connection_point.connectionPointOf
                    setpoint_receiver_signal_key = connection_point.receiverPropertyName
                    receiver_component_key = receiver_component.id
                    setpoint_connections[setpoint_component_key][receiver_component_key] = setpoint_receiver_signal_key

                #Deleting one connection deletes all of them in this case, since they all have "scheduleValue" as the sender signal key    
                self.remove_connection(sender_component, receiver_component, "scheduleValue", setpoint_receiver_signal_key)
            except Exception as e:
                print(f"Could not find connection for {setpoint_component_key} and {input_output_dictionary['input'][setpoint_component_key]['signal_key']}")

        # Define the output dictionary for the NeuralController using a dictionary comprehension
        # Add the connections to the setpoint-receiving components
        for sender_key, receivers in setpoint_connections.items():
            for receiver_key, receiver_signal in receivers.items():
                output_signal_key = f"{sender_key}_{receiver_key}_input_signal"
                neural_policy_controller.output[output_signal_key] = tps.Scalar()
                self.add_connection(
                    neural_policy_controller,
                    self.component_dict[receiver_key],
                    output_signal_key,
                    receiver_signal
                )

        # Define a custom initial dictionary for the NeuralController outputs:
        custom_initial = {"neural_controller": {f"{component_key}_input_signal": tps.Scalar(0) for component_key in setpoint_connections.keys()}}
        self.set_custom_initial_dict(custom_initial)
       
        #Add the input connections
        
        for component_key in input_output_dictionary["input"]:
            try:
                sender_component = self.component_dict[component_key]
            except KeyError:
                print(f"Could not find component {component_key}")
                continue
            receiving_component = neural_policy_controller
            self.add_connection(
                sender_component,
                receiving_component,
                input_output_dictionary["input"][component_key]["signal_key"],
                "actualValue"
            )


        


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
    spaces = self.get_component_by_class(self.component_dict, tb.BuildingSpace1AdjBoundaryFMUSystem)
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace2AdjBoundaryFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace11AdjBoundaryFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace1AdjBoundaryOutdoorFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace2AdjBoundaryOutdoorFMUSystem))
    spaces.extend(self.get_component_by_class(self.component_dict, tb.BuildingSpace11AdjBoundaryOutdoorFMUSystem))
    for space in spaces:
        self.add_connection(supply_water_temperature_schedule, space, 
                            "scheduleValue", "supplyWaterTemperature")
        
    #Load the input/output dictionary from the file policy_input_output.json
    with open(r"C:\Users\asces\OneDriveUni\Projects\Adrenalin_BOPTEST_Challenge\RL_control\t4b_model\policy_training\policy_input_output.json") as f:
        input_output_dictionary = json.load(f)

    insert_neural_policy_in_fcn(self, input_output_dictionary, policy_path= None)

if __name__ == "__main__":
    # Create a new model
    model = tb.Model(id="neural_policy_1stfloor", saveSimulationResult=True)
    filename = r"C:\Users\asces\OneDriveUni\Projects\Adrenalin_BOPTEST_Challenge\RL_control\t4b_model\model\fan_flow_configuration_template_DP37_full_no_cooling.xlsm"
    model.load(semantic_model_filename=filename, fcn=fcn, verbose=False)

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
    energy = np.array(model.component_dict[space_id].savedOutput['spaceHeaterPower'])
    print(f"Total energy consumption: {energy.sum()} Wh")

    #Print the deviation from the setpoint
    deviation = np.array(model.component_dict[space_id].savedOutput['indoorTemperature']) - 21
    print(f"Total deviation from setpoint: {deviation.sum()} °C")

    # Temperature plot
    plot.plot_component(
        simulator,
        components_1axis=[(space_id, 'indoorTemperature')],
        components_2axis=[(space_id, 'outdoorTemperature')],
        ylabel_1axis='Room Temperature [°C]',
        ylabel_2axis='Outdoor Temperature [°C]',
        show=True
    )

    # CO2 plot
    plot.plot_component(
        simulator,
        components_1axis=[(space_id, 'indoorCo2Concentration')],
        components_2axis=[(space_id, 'airFlowRate')],
        ylabel_1axis='CO2 Concentration [ppm]',
        ylabel_2axis='Air Flow Rate [m³/s]',
        show=True
    )

    # power plot
    plot.plot_component(
        simulator,
        components_1axis=[(space_id, 'spaceHeaterPower')],
        ylabel_1axis='Space Heater Power [W]',
        show=True
    )

    #plot the CO2 setpoint
    plot.plot_component(
        simulator,
        components_1axis=[("neural_controller", '020B_co2_controller_input_signal')],
        ylabel_1axis='CO2 Setpoint [ppm]',
        show=True
    )

    #plot the temperature setpoint
    plot.plot_component(
        simulator,
        components_1axis=[("neural_controller", '020B_temperature_heating_controller_input_signal')],
        ylabel_1axis='Temperature Setpoint [°C]',
        show=True
    )
