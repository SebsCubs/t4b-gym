import numpy as np
import pandas as pd
import math

""" in the following script, will a function, for determining tthe fan flow be defined.
The flow will be based on the pressure difference, over the fan and the K-faktor, using
the function: flow [m3/h] = k_faktor * (sqrt(∆p))/h [Pa/h]"""

def fan_flow(k_faktor: float, pressure: list):
    flow = []
    index = 0
    for i in range(len(pressure)):
        if pressure[index] == '(null)':
            flow.append('null')
            index = index+1
        else:
            pressure_value = float(pressure[index])
            flow_value = math.sqrt(pressure_value)*k_faktor
            flow.append(flow_value)
            index=index+1
    return flow


df_supply_fan_pressure = pd.read_csv("/Users/augustthomsen/Desktop/UNI/SDU-mmmi/Twin4build-Case-Studies/DP37/data/Time series/HF04/OD095_01_032A_J95_HF04_BPA002_S1.plc_SENSOR_VALUE.csv", index_col=2)
supply_fan_pressure = df_supply_fan_pressure['vValue'].tolist()
k_value = 355
supply_fan_flow = fan_flow(k_value, supply_fan_pressure)

indexList = df_supply_fan_pressure.index.tolist()

df_supply_fan_flow = pd.DataFrame(supply_fan_flow)
df_supply_fan_flow.index = indexList
df_supply_fan_flow.columns = ['supply fan flow [m^3/h]']

df_exhaust_fan_pressure = pd.read_csv("/Users/augustthomsen/Desktop/UNI/SDU-mmmi/Twin4build-Case-Studies/DP37/data/Time series/HF04/OD095_01_032A_J95_HF04_BPA004_S1.plc_SENSOR_VALUE.csv", index_col=2)
exhaust_fan_pressure = df_exhaust_fan_pressure['vValue'].tolist()
k_value = 355
exhaust_fan_flow = fan_flow(k_value, exhaust_fan_pressure)

indexList = df_exhaust_fan_pressure.index.tolist()

df_exhaust_fan_flow = pd.DataFrame(supply_fan_flow)
df_exhaust_fan_flow.index = indexList
df_exhaust_fan_flow.columns = ['exhaust fan flow [m^3/h]']

df_exhaust_fan_flow.to_csv("HF04_exhaust_fan_flow.csv")
df_supply_fan_flow.to_csv("HF04_supply_fan_flow.csv")

