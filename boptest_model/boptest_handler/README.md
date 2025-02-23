This folder contains the code for running simulations from BOPTEST and harvesting data from the model.
In order to model the multizone_office_simple_air testcase, the following points are used for parameter estimation:

- hvac_reaAhu_TSup_y [K] [min=None, max=None]: Supply air temperature measurement for AHU
- hvac_reaAhu_V_flow_sup_y [m3/s] [min=None, max=None]: Supply air flowrate measurement for AHU
- weaSta_reaWeaTWetBul_y [K] [min=None, max=None]: Wet bulb temperature measurement
- hvac_oveAhu_TSupSet_u [K] [min=285.15, max=313.15]: Supply air temperature setpoint for AHU
- hvac_oveAhu_yOA_u [1] [min=0.0, max=1.0]: Outside air damper position setpoint for AHU
- hvac_oveAhu_yRet_u [1] [min=0.0, max=1.0]: Return air damper position setpoint for AHU
- hvac_reaAhu_PFanSup_y [W] [min=None, max=None]: Electrical power measurement of supply fan for AHU

Parameters for the t4b model:
- fan coefficients: airflow to power consumption
- pump coefficients: flowrate to power consumption
- cooling coil coefficients: flowrate to cooling capacity
- heating coil coefficients: flowrate to heating capacity
- check valve coefficients: just a big value to make it fully open
- valve maxFlowRate: same as pump nominal flowrate
- Dampers: position to flowrate to get the curve coefficients

Specific for the coil parameters:
```json
{
    "parameters": {
        "m1_flow_nominal": null, // COIL flowrate for water side [kg/s]
        "m2_flow_nominal": null, // COIL flowrate for air side [kg/s]
        "tau1": null, // COIL time constant at nominal flow for water side [s]
        "tau2": null, // COIL time constant at nominal flow for air side [s]
        "tau_m": null, // COIL time constant of metal at nominal UA value [s]
        "nominalUa.hasValue": null, // COIL Thermal conductance at nominal flow, used to compute heat capacity [W/K]
        "mFlowValve_nominal": null, // VALVE nominal flowrate [kg/s]
        "flowCoefficient.hasValue": null, // VALVE Kv (metric) flow coefficient [m3/h/(bar)^(1/2)]
        "mFlowPump_nominal": null, // PUMP nominal flowrate [kg/s]
        "KvCheckValve": null, // CHECKVALVE Kv value
        "dp1_nominal": null, // COIL water pressure difference [Pa]
        "dpPump": null, // PUMP pressure drop [Pa]
        "dpSystem": null, // SYSTEM pressure drop [Pa]
        "dpFixedSystem": null, // VALVEOUTLET pressure drop [Pa]
        "tau_w_inlet": 1, // time constant of the water inlet
        "tau_w_outlet": 1, // time constant of the water outlet
        "tau_air_outlet": 1 // time constant of the air outlet
    }
}
```

Specific for the fan parameters:
```json
{
    "parameters": {
        "c1": null, // FAN coefficient 1
        "c2": null, // FAN coefficient 2
        "c3": null, // FAN coefficient 3
        "c4": null, // FAN coefficient 4
        "nominalAirFlowRate.hasValue": null, // FAN nominal flowrate [kg/s]
        "nominalPowerRate.hasValue": null // FAN nominal power consumption [W]
    }
}
```
The parameters to estimate are:
- All fan coefficients
- Fan power rate
- All coil tau's
- Coil UA values
- Valve Kv values
- Coil DP1 values
- DP system and DP fixed system values
- Check valve Kv value

### Estimation of the parameters

#### Only AHU model

To perform and estimation of the parameters, the input and output of the AHU model is provided. 

#### Input

- Supply air temperature measurement for AHU (hvac_reaAhu_TSup_y)(vent_supply_air_temp_sensor)
- Supply air flowrate measurement for AHU (hvac_reaAhu_V_flow_sup_y)(vent_airflow_sensor)
- Wet bulb temperature measurement (weaSta_reaWeaTWetBul_y)(vent_outdoor_air_temp_sensor)
- Supply air temperature setpoint for AHU (hvac_oveAhu_TSupSet_u)(supply_air_temp_setpoint)
- Fan power measurement for AHU (hvac_reaAhu_PFanSup_y)(vent_power_sensor)
- Heating supply water temperature measurement (hvac_reaAhu_THeaCoiSup_y)(supply_heating_coil_water_temp_sensor)
- Heating return water temperature measurement (hvac_reaAhu_THeaCoiRet_y)(return_heating_coil_water_temp_sensor)
- Cooling supply water temperature measurement (hvac_reaAhu_TCooCoiSup_y)(supply_cooling_coil_water_temp_sensor)
- Cooling return water temperature measurement (hvac_reaAhu_TCooCoiRet_y)(return_cooling_coil_water_temp_sensor)
- Outside air damper position setpoint for AHU (hvac_oveAhu_yOA_u)(vent_supply_damper_position_sensor)
- Return air damper position setpoint for AHU (hvac_oveAhu_yRet_u)(vent_return_damper_position_sensor)
- Heating valve position measurement (hvac_oveAhu_yHea_u)(heating_valve_position_sensor)
- Cooling valve position measurement (hvac_oveAhu_yCoo_u)(cooling_valve_position_sensor)

Providing all inputs will allow the model to estimate the parameters.

With parameters estimated, the model can be used to predict:
- Supply air temperature measurement for AHU (hvac_reaAhu_TSup_y)(vent_supply_air_temp_sensor)
- Fan power measurement for AHU (hvac_reaAhu_PFanSup_y)(vent_power_sensor)
- Heating and cooling coil power measurement (supply_heating_coil_power_sensor, supply_cooling_coil_power_sensor)

Providing these inputs:

- Supply air flowrate measurement for AHU (hvac_reaAhu_V_flow_sup_y)(vent_airflow_sensor)
- Wet bulb temperature measurement (weaSta_reaWeaTWetBul_y)(vent_outdoor_air_temp_sensor)
- Supply air temperature setpoint for AHU (hvac_oveAhu_TSupSet_u)(supply_air_temp_setpoint)





