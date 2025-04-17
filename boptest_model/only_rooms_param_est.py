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
        'output_value': 'core_indoorCo2Concentration',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonCor_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'north_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonNor_TZon_y_processed.csv'
    },
    {
        'component_id': 'north_co2_sensor',
        'output_value': 'north_indoorCo2Concentration',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonNor_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'south_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonSou_TZon_y_processed.csv'
    },
    {
        'component_id': 'south_co2_sensor',
        'output_value': 'south_indoorCo2Concentration',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonSou_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'east_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonEas_TZon_y_processed.csv'
    },
    {
        'component_id': 'east_co2_sensor',
        'output_value': 'east_indoorCo2Concentration',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonEas_CO2Zon_y_processed.csv'
    },
    {
        'component_id': 'west_indoor_temp_sensor',
        'output_value': 'measuredValue',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonWes_TZon_y_processed.csv'
    },
    {
        'component_id': 'west_co2_sensor',
        'output_value': 'west_indoorCo2Concentration',
        'csv_path': 'C:/Users/asces/OneDriveUni/Projects/RL_control/boptest_model/boptest_handler/data/merged_data/hvac_reaZonWes_CO2Zon_y_processed.csv'
    }
]

def fcn(self):
    '''
        The fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''

    #Add core sensors
    core_co2_sensor = tb.SensorSystem(id="core_co2_sensor", saveSimulationResult=True)
    core_supply_air_temp_sensor = tb.SensorSystem(id="core_supply_air_temp_sensor", saveSimulationResult=True)
    core_supply_airflow_sensor = tb.SensorSystem(id="core_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(self.components["core"], core_co2_sensor, "indoorCo2Concentration", "core_indoorCo2Concentration")
    self.add_connection(core_supply_airflow_sensor, self.components["core"], "airFlowRate", "airFlowRate")
    self.add_connection(core_supply_air_temp_sensor, self.components["core"], "supplyAirTemperature", "supplyAirTemperature")
    #self.remove_connection(self.components["vent_supply_air_temp_sensor"], self.components["core"], "measuredValue", "supplyAirTemperature") #This connection was generated from the semantic model but is not needed
    #self.remove_component(self.components["vent_supply_air_temp_sensor"]) #This component was generated from the semantic model but is not needed


    #Add north sensors
    north_co2_sensor = tb.SensorSystem(id="north_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["north"], north_co2_sensor, "indoorCo2Concentration", "north_indoorCo2Concentration")
    north_supply_air_temp_sensor = tb.SensorSystem(id="north_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(north_supply_air_temp_sensor, self.components["north"], "supplyAirTemperature", "supplyAirTemperature")
    north_supply_airflow_sensor = tb.SensorSystem(id="north_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(north_supply_airflow_sensor, self.components["north"], "airFlowRate", "airFlowRate")

     #Add south sensors
    south_co2_sensor = tb.SensorSystem(id="south_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["south"], south_co2_sensor, "indoorCo2Concentration", "south_indoorCo2Concentration")
    south_supply_air_temp_sensor = tb.SensorSystem(id="south_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(south_supply_air_temp_sensor, self.components["south"], "supplyAirTemperature", "supplyAirTemperature")
    south_supply_airflow_sensor = tb.SensorSystem(id="south_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(south_supply_airflow_sensor, self.components["south"], "airFlowRate", "airFlowRate")

    #Add east sensors
    east_co2_sensor = tb.SensorSystem(id="east_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["east"], east_co2_sensor, "indoorCo2Concentration", "east_indoorCo2Concentration")
    east_supply_air_temp_sensor = tb.SensorSystem(id="east_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(east_supply_air_temp_sensor, self.components["east"], "supplyAirTemperature", "supplyAirTemperature")
    east_supply_airflow_sensor = tb.SensorSystem(id="east_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(east_supply_airflow_sensor, self.components["east"], "airFlowRate", "airFlowRate")
    
    #Add west sensors
    west_co2_sensor = tb.SensorSystem(id="west_co2_sensor", saveSimulationResult=True)
    self.add_connection(self.components["west"], west_co2_sensor, "indoorCo2Concentration", "west_indoorCo2Concentration")
    west_supply_air_temp_sensor = tb.SensorSystem(id="west_supply_air_temp_sensor", saveSimulationResult=True)
    self.add_connection(west_supply_air_temp_sensor, self.components["west"], "supplyAirTemperature", "supplyAirTemperature")
    west_supply_airflow_sensor = tb.SensorSystem(id="west_supply_airflow_sensor", saveSimulationResult=True)
    self.add_connection(west_supply_airflow_sensor, self.components["west"], "airFlowRate", "airFlowRate")


def get_model(id=None, fcn_=None):
    if fcn_ is None:
        fcn_ = fcn
    model = tb.Model(id="only_rooms_estimation", saveSimulationResult=True)
    
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), r"semantic_models\five_rooms_no_junctions.xlsm")
    model.load(semantic_model_filename=filename, fcn=fcn_, create_signature_graphs=False, validate_model=True, verbose=True, force_config_update=True)
    if id is not None:
        model.id = id

    #model.remove_component(model.components["vent_supply_air_temp_sensor"]) #This component was generated from the semantic model but is not needed
    #model.draw_system_graph()
    return model

def run(model = None):
    stepSize = 60  # Seconds
    
    startTime = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=2, day=12, hour=0, minute=0, second=0,
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
    stepSize = 60  # Seconds can go down to 30
    # Then set the startTime and endTime to a valid range
    startTime = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))
    endTime = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))

    model = get_model(id="only_rooms_estimation")

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
    
    """
    Parameters for each room:
    - Input & output damper:
        - a (shared) 
        - nominalAirFlowRate
    - PI controller Kp, Ki constants
    - Space model parameters:

      BuildingSpaceNoSH1AdjBoundaryOutdoorFMUSystem (north, south, east, west)

        C_supply 400
        C_wall - To be estimated
        C_air  - To be estimated
        C_int  - To be estimated (shared)
        C_boundary  - To be estimated
        R_out  - To be estimated
        R_in  - To be estimated
        R_int  - To be estimated (shared)
        R_boundary  - 100 (high value)
        f_wall  - To be estimated
        f_air  - To be estimated 
        Q_occ_gain - To be estimated (default 0)
        CO2_occ_gain  - To be estimated 8.18e-6?
        CO2_start (400 ppm)
        T_boundary  - 20 (default value, since R_boundary is high)
        infiltration - To be estimated (shared)
        airVolume (Known from geometry)

        BuildingSpaceNoSH1AdjBoundaryFMUSystem (core)

        C_supply 400
        C_air  - To be estimated
        C_int - To be estimated (shared)    
        C_boundary - To be estimated
        R_int - To be estimated (shared)
        R_boundary  - 100 (high value)
        Q_occ_gain  - To be estimated (default 0 ?)
        CO2_occ_gain  - To be estimated (shared) (Where does it come from?) 8.18e-6 ?
        CO2_start (400 ppm)
        T_boundary - 20 (default value, since R_boundary is high)
        infiltration - To be estimated (shared ?)
        airVolume (Known from geometry)
        
        Where the shared parameters to reduce estimation effort are:
        - C_int
        - R_int
        - T_boundary


    Required data points:
    Per room data points:
    [x]Supply air temperature (hvac_reaZonCor_TSup_y, hvac_reaZonNor_TSup_y, hvac_reaZonSou_TSup_y, hvac_reaZonEas_TSup_y, hvac_reaZonWes_TSup_y)
    [x]Supply air flow rate (hvac_reaZonCor_V_flow_y, hvac_reaZonNor_V_flow_y, hvac_reaZonSou_V_flow_y, hvac_reaZonEas_V_flow_y, hvac_reaZonWes_V_flow_y)
    [x]CO2 concentration (hvac_reaZonCor_CO2Zon_y, hvac_reaZonNor_CO2Zon_y, hvac_reaZonSou_CO2Zon_y, hvac_reaZonEas_CO2Zon_y, hvac_reaZonWes_CO2Zon_y)
    [x]Indoor air temperature (hvac_reaZonCor_TZon_y, hvac_reaZonNor_TZon_y, hvac_reaZonSou_TZon_y, hvac_reaZonEas_TZon_y, hvac_reaZonWes_TZon_y)
    [x](Forecast values) Occupancy (Occupancy[cor], Occupancy[nor], Occupancy[sou], Occupancy[eas], Occupancy[wes]) 

    Model outputs (measuring devices):
    - Indoor air temperature (core_indoor_air_temp_sensor, north_indoor_air_temp_sensor, south_indoor_air_temp_sensor, east_indoor_air_temp_sensor, west_indoor_air_temp_sensor)
    - CO2 concentration (core_co2_sensor, north_co2_sensor, south_co2_sensor, east_co2_sensor, west_co2_sensor)
    """
    

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
    model = get_model(id="only_rooms_estimation")
    model.load_estimation_result(parameter_filename)


    

    """
    CORE:
        C_supply: 400.0
        C_air: 10564828.200215876
        C_int: 6852402.107907902
        C_boundary: 1221217.5316119944
        R_int: 0.00031542152549524235
        R_boundary: 100.0
        Q_occ_gain: 251.2805298698468
        CO2_occ_gain: 1.1067941581940439e-05
        CO2_start: 400.0
        airVolume: 2698.00128
        T_boundary: 20.0
        infiltration: 1.0000000000000001e-07
    
    """
    model.components["core"].Q_occ_gain = 255
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
    #filepath = parameter_estimation()
    filepath = r"C:\Users\asces\OneDriveUni\Projects\RL_control\boptest_model\generated_files\models\only_rooms_estimation\model_parameters\estimation_results\LS_result\mix_day_most_accurate_08042025.pickle"
    parameter_evaluation(model_output_points, filepath, save_plots=True)
