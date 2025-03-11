# Replacing the baseline control with a simpler controller for model calibration

The baseline controller aims to make it easier to calibrate the model of the AHU and the different zones. The controller is defined in the `baseline_control.py` file.

The controller has only two operation modes:
- Occupied mode: the controller is in occupied mode when the zone is occupied.
- Unoccupied mode: the controller is in unoccupied mode when the zone is unoccupied.

With no economizer, controlling supply air temperature, supply air static pressure a constant indoor air flow rate. And simple PI controllers for the room temperature setpoint tracking. 

Because of this, the control signals to be overridden are:

Setpoints: 
- hvac_oveAhu_TSupSet_u [K] [min=285.15, max=313.15]: Supply air temperature setpoint for AHU
- hvac_oveAhu_dpSet_u [Pa] [min=50.0, max=410.0]: Supply duct pressure setpoint for AHU
- hvac_oveAhu_yFan_u [1] [min=0.0, max=1.0]: Supply fan speed setpoint for AHU
- hvac_oveAhu_yOA_u [1] [min=0.0, max=1.0]: Outside air damper position setpoint for AHU
- hvac_oveAhu_yRet_u [1] [min=0.0, max=1.0]: Return air damper position setpoint for AHU

Control signals:
- hvac_oveAhu_yCoo_u [1] [min=0.0, max=1.0]: Cooling coil valve control signal for AHU
- hvac_oveAhu_yHea_u [1] [min=0.0, max=1.0]: Heating coil valve control signal for AHU
- hvac_oveAhu_yPumCoo_u [1] [min=0.0, max=1.0]: Cooling coil pump control signal for AHU
- hvac_oveAhu_yPumHea_u [1] [min=0.0, max=1.0]: Heating coil pump control signal for AHU

Room specific signals (repeats for each zone):
- hvac_oveZonActCor_yDam_u [1] [min=0.0, max=1.0]: Damper position setpoint for zone cor
- hvac_oveZonActCor_yReaHea_u [1] [min=0.0, max=1.0]: Reheat control signal for zone cor
- hvac_oveZonSupCor_TZonCooSet_u [K] [min=285.15, max=313.15]: Zone air temperature cooling setpoint for zone cor
- hvac_oveZonSupCor_TZonHeaSet_u [K] [min=285.15, max=313.15]: Zone air temperature heating setpoint for zone cor

## Custom strategy:

The custom controller would override all the setpints and control signals (Although the control signals should be enough, because of the hidden states of the building, all of the available signals should be overridden). This exercise is effectively building a new controller with limited information.
- Occupied mode: From 7:00 to 17:00
- Unoccupied mode: From 17:00 to 7:00

Setpoints:
- hvac_oveAhu_TSupSet_u: 12 degrees always
- hvac_oveAhu_dpSet_u: 50 Pa unoccupied, 400 Pa occupied
- hvac_oveAhu_yFan_u: PI controller with setpoint hvac_oveAhu_dpSet_u and feedback hvac_reaAhu_dp_sup_y (fan discharge static pressure)
- hvac_oveAhu_yOA_u: 0 unoccupied, 1 occupied
- hvac_oveAhu_yRet_u: 1 unoccupied, 0 occupied

Control signals:
- hvac_oveAhu_yCoo_u: PI controller with setpoint hvac_oveAhu_TSupSet_u and feedback hvac_reaAhu_TSup_y 
- hvac_oveAhu_yHea_u: PI controller with setpoint hvac_oveAhu_TSupSet_u and feedback hvac_reaAhu_TSup_y 
- hvac_oveAhu_yPumCoo_u: 1 if hvac_oveAhu_yCoo_u > 0.1, 0 otherwise
- hvac_oveAhu_yPumHea_u: 1 if hvac_oveAhu_yHea_u > 0.1, 0 otherwise

Room controllers can be left unchanged, while validating the first version of the model. 


