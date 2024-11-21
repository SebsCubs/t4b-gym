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



def fcn(self):
    '''
        The fcn() function adds connections between components in a system model,
        creates a schedule object, and adds it to the component dictionary.
        The test() function sets simulation parameters and runs a simulation of the system
        model using the Simulator() class. It then generates several plots of the simulation results using functions from the plot module.
    '''


def get_model_added_space_heater():
    model = tb.Model(id="cut_added_space_heater", saveSimulationResult=True)
    filename = os.path.join(uppath(os.path.abspath(__file__), 1), "cut_added_space_heater.xlsm")
    model.load_model(semantic_model_filename=filename, infer_connections=True, fcn=fcn, create_signature_graphs=False, validate_model=False)
    return model

def run():

    """stepSize = 600  # Seconds
    startTime = datetime.datetime(year=2023, month=12, day=2, hour=0, minute=0, second=0,
                                  tzinfo=gettz("Europe/Copenhagen"))

    endTime = datetime.datetime(year=2023, month=12, day=12, hour=0, minute=0, second=0,
                                tzinfo=gettz("Europe/Copenhagen"))"""
    model = get_model_added_space_heater()

    # monitor = tb.Monitor(model) #Compares the simulation results with the expected results

    # profiler = cProfile.Profile()
    # profiler.enable()

    # monitor.monitor(startTime=startTime,
    #                 endTime=endTime,
    #                 stepSize=stepSize,
    #                 show=False)

    """simulator = tb.Simulator(model)
    simulator.simulate(model, startTime=startTime, endTime=endTime, stepSize=stepSize)

    plot.plot_space_CO2_fmu(model, simulator, "[013A][013A_space_heater]")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(simulator.dateTimeSteps, model.component_dict["007A_occupancy_profile"].savedOutput["scheduleValue"], color="green")
    plt.figure()
    plt.plot(simulator.dateTimeSteps, model.component_dict["020A_occupancy_profile"].savedOutput["scheduleValue"], color="red")
    plt.figure()
    plt.plot(simulator.dateTimeSteps, model.component_dict["029A_occupancy_profile"].savedOutput["scheduleValue"], color="blue")
    plt.figure()
    plt.plot(simulator.dateTimeSteps, model.component_dict["035A_occupancy_profile"].savedOutput["scheduleValue"], color="blue")
    plt.show()"""
    
    # simulator = tb.Simulator(model)
    # simulator.simulate(model, startTime=startTime, endTime=endTime, stepSize=stepSize)
    
    # profiler.disable()
    # out = io.StringIO()
    # pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(100)
    # plot.plot_outdoor_environment(model, monitor.simulator, firstAxisylim=[5, 30])
    # space_name = "[029A][029A_space_heater]"
    # temperature_heating_controller_name = "029A_temperature_heating_controller"

    # plot.plot_space_temperature_fmu(model, monitor.simulator, space_name)
    # plot.plot_space_CO2_fmu(model, monitor.simulator, space_name)
    # plot.plot_temperature_controller(model, monitor.simulator, temperature_heating_controller_name, show=True)


if __name__ == "__main__":
    run()