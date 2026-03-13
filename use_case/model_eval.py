import sys
import os
import datetime
from dateutil.tz import gettz 
from gymnasium.core import Wrapper
import gymnasium as gym
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import twin4build as tb 
from tqdm import tqdm
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(MAIN_DIR)

from boptest_model.rooms_and_ahu_model import load_model_and_params

POLICY_CONFIG_PATH = os.path.join(SCRIPT_DIR, "policy_input_output.json")
device = 'cpu'



def get_baseline(model):
        stepSize = 600 #Seconds
        #Define the range of available data
        # Three time periods for the three cases
        time_periods = [
            # Typical heat day: January 11-25, 2024
            (datetime.datetime(year=2024, month=1, day=11, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
             datetime.datetime(year=2024, month=1, day=25, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))),
            
            # Mix day: March 17 - March 31, 2024
            (datetime.datetime(year=2024, month=3, day=17, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
             datetime.datetime(year=2024, month=3, day=31, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))),
            
            # Typical cool day: May 17-31, 2024
            (datetime.datetime(year=2024, month=5, day=17, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")),
             datetime.datetime(year=2024, month=5, day=31, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen")))
        ]        

        simulator = tb.Simulator()

        simulator.simulate(model, time_periods[0][0], time_periods[0][1], stepSize)

        plot_results_publication(simulator, save_dir='plots_baseline')

        return simulator



def test_model(env, model):
        stepSize = 600 #Seconds
        #Define the range of available data
        start_time = datetime.datetime(year=2024, month=1, day=11, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))
        #end_time = datetime.datetime(year=2024, month=1, day=15, hour=0, minute=0, second=0, tzinfo=gettz("Europe/Copenhagen"))        
        episode_length = int(3600*24*15 / stepSize)
        warmup_period = 0

        # Set a fixed start time
        if isinstance(env,Wrapper): 
                env.unwrapped.random_start = False
                env.unwrapped.global_start_time = start_time
                env.unwrapped.episode_length = episode_length
                env.unwrapped.warmup_period = warmup_period
        else:
                env.random_start   = False
                env.global_start_time   = start_time
                env.episode_length  = episode_length
                env.warmup_period = warmup_period

        # Reset environment
        obs, _ = env.reset()
        
        # Simulation loop
        done = False
        observations = [obs]
        rewards = []
        print('Simulating...')

        # Create progress bar
        pbar = tqdm(total=episode_length, desc="Simulation Progress")
        
        while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                observations.append(obs)
                rewards.append(reward)
                done = (terminated or truncated)
                pbar.update(1)
        
        pbar.close()
        plot_results_publication(env.unwrapped.simulator, rewards, save_dir='plots')

        return observations, rewards




def thermal_discomfort(T, T_low, T_high, dt_seconds):
    """
    Compute D(t0, tf) = (1/N) * sum_z ∫ |s_z(t)| dt   (in K·h/zone)
    where s_z(t) is the deviation outside the comfort band [T_low, T_high].

    Parameters
    ----------
    T : array-like, shape (Tsteps, Nz)
        Indoor temperatures per time step and zone.
    T_low, T_high : array-like (broadcastable to T)
        Lower/upper setpoints per time step and zone.
        Can be scalars, 1D (Tsteps,), or 2D (Tsteps, Nz).
    dt_seconds : float
        Sampling period in seconds.

    Returns
    -------
    D : float
        Thermal discomfort averaged over zones (K·h/zone).
    per_zone_D : ndarray, shape (Nz,)
        Thermal discomfort per zone (K·h).
    """
    T = np.asarray(T, dtype=float)
    T_low = np.asarray(T_low, dtype=float)
    T_high = np.asarray(T_high, dtype=float)

    # Slack outside the band (inside the band → 0)
    slack_below = np.maximum(T_low - T, 0.0)
    slack_above = np.maximum(T - T_high, 0.0)
    slack = slack_below + slack_above  # magnitude in K at each step

    # Discrete integral over time: sum(slack) * Δt, then convert s→h
    per_zone_D = np.nansum(slack, axis=0) * (dt_seconds / 3600.0)  # K·h per zone
    D = np.nanmean(per_zone_D)  # average over zones → K·h/zone
    return D, per_zone_D

def iaq_violation_building(CO2, CO2_thr, dt_seconds, area_m2):
    """
    Compute building-level CO₂ violation in ppm·h/m².

    Φ = ( Σ_z ∫ max(CO2_z(t) - CO2_thr_z(t), 0) dt ) / ( Σ_z A_z )

    Parameters
    ----------
    CO2 : array-like, shape (Tsteps, Nz)
        Measured CO₂ concentrations (ppm).
    CO2_thr : array-like (broadcastable to CO2)
        Threshold(s) (ppm). Can be scalar, (Tsteps,), or (Tsteps, Nz).
    dt_seconds : float
        Sampling period (seconds).
    area_m2 : float
        Building floor area (m²).

    Returns
    -------
    Phi_building : float
        Total building CO₂ violation, units ppm·h/m².
    per_zone_Phi : ndarray, shape (Nz,)
        Contribution of each zone, units ppm·h (not normalized).
    """
    CO2 = np.asarray(CO2, dtype=float)
    CO2_thr = np.asarray(CO2_thr, dtype=float)

    # Positive deviation above threshold
    phi = np.maximum(CO2 - CO2_thr, 0.0)  # ppm

    # Integrate over time → ppm·h per zone
    per_zone_Phi = np.nansum(phi, axis=0) * (dt_seconds / 3600.0)

    # Weighted building-level normalization by total floor area
    Phi_building = np.sum(per_zone_Phi) / area_m2

    return Phi_building, per_zone_Phi

def calculate_energy_boptest_style(power_data, time_data, area_m2):
    """
    Calculate energy consumption using BOPTEST-style trapezoidal integration.
    
    Parameters
    ----------
    power_data : array-like
        Power consumption data (W) - can be 1D or 2D array
    time_data : array-like
        Time data (seconds) - should match the time dimension of power_data
    area_m2 : float
        Building floor area (m²)
        
    Returns
    -------
    energy_kwh : float
        Total energy consumption in kWh
    energy_kwh_m2 : float
        Energy consumption per m² in kWh/m²
    """
    power_data = np.asarray(power_data, dtype=float)
    time_data = np.asarray(time_data, dtype=float)
    
    # If power_data is 2D, sum across all sources
    if power_data.ndim > 1:
        power_data = np.sum(power_data, axis=0)
    
    # Use trapezoidal integration: ∫ P(t) dt
    energy_joules = np.trapezoid(power_data, time_data)
    
    # Convert J to kWh: 1 kWh = 3.6e6 J, so multiply by 2.77778e-7
    energy_kwh = energy_joules * 2.77778e-7
    
    # Normalize by floor area
    energy_kwh_m2 = energy_kwh / area_m2
    
    return energy_kwh, energy_kwh_m2


# ============================================================
# Publication-Quality Plotting for RL Control Evaluation
# ============================================================

ZONE_ORDER = ['core', 'north', 'south', 'east', 'west']
ZONE_COLORS = {
    'core': '#D32F2F', 'north': '#1565C0', 'south': '#2E7D32',
    'east': '#F57F17', 'west': '#7B1FA2',
}
ZONE_LABELS = {
    'core': 'Core', 'north': 'North', 'south': 'South',
    'east': 'East', 'west': 'West',
}
ENERGY_COLORS = {
    'fan': '#1565C0', 'heating_coil': '#D32F2F',
    'cooling_coil': '#2196F3', 'reheat_coils': '#FF9800',
}

ACTION_LABELS = {
    ('supply_air_temp_setpoint_sensor', 'measuredValue'):  'AHU Supply Temp. Setpoint [°C]',
    ('core_temperature_heating_controller', 'heatingsetpointValue'):   'Core Heating SP [°C]',
    ('core_temperature_heating_controller', 'coolingsetpointValue'):   'Core Cooling SP [°C]',
    ('core_supply_damper_position_sensor', 'measuredValue'):           'Core Damper Pos. [–]',
    ('north_temperature_heating_controller', 'heatingsetpointValue'):  'North Heating SP [°C]',
    ('north_temperature_heating_controller', 'coolingsetpointValue'):  'North Cooling SP [°C]',
    ('north_supply_damper_position_sensor', 'measuredValue'):          'North Damper Pos. [–]',
    ('south_temperature_heating_controller', 'heatingsetpointValue'):  'South Heating SP [°C]',
    ('south_temperature_heating_controller', 'coolingsetpointValue'):  'South Cooling SP [°C]',
    ('south_supply_damper_position_sensor', 'measuredValue'):          'South Damper Pos. [–]',
    ('east_temperature_heating_controller', 'heatingsetpointValue'):   'East Heating SP [°C]',
    ('east_temperature_heating_controller', 'coolingsetpointValue'):   'East Cooling SP [°C]',
    ('east_supply_damper_position_sensor', 'measuredValue'):           'East Damper Pos. [–]',
    ('west_temperature_heating_controller', 'heatingsetpointValue'):   'West Heating SP [°C]',
    ('west_temperature_heating_controller', 'coolingsetpointValue'):   'West Cooling SP [°C]',
    ('west_supply_damper_position_sensor', 'measuredValue'):           'West Damper Pos. [–]',
}


def _set_pub_style():
    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 17,
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 1.5,
    })


def _make_ts(data, times, step_size):
    """Create a timezone-converted, resampled pandas Series."""
    s = pd.Series(data=data, index=times)
    s.index = s.index.tz_convert('Europe/Copenhagen')
    return s.resample(pd.Timedelta(seconds=step_size)).mean()


def _compute_reheat_power(simulator, zone):
    """Compute reheat coil thermal power [W] for a single zone."""
    cp_air = 1005
    tol = 1e-5
    m_dot = np.asarray(simulator.model.components[f"{zone}_reheat_coil"].savedInput["airFlowRate"])
    T_in  = np.asarray(simulator.model.components[f"{zone}_reheat_coil"].savedInput["inletAirTemperature"])
    T_out = np.asarray(simulator.model.components[f"{zone}_reheat_coil"].savedOutput["outletAirTemperature"])
    Q = np.zeros_like(m_dot)
    mask = (m_dot > tol) & ((T_out - T_in) > 1.0)
    Q[mask] = m_dot[mask] * cp_air * (T_out[mask] - T_in[mask])
    return Q


def plot_results_publication(simulator, rewards=None, plotting_stepSize=600,
                             save_dir='plots'):
    """Generate publication-quality figures for RL control evaluation.

    Figures produced
    ----------------
    1.  Zone temperatures with shaded comfort bands  (5-panel)
    2.  All-zone temperature comparison              (single panel)
    3.  Zone CO₂ concentrations with setpoint lines  (5-panel)
    4.  AHU signals overview                         (6-panel)
    5.  HVAC energy breakdown — stacked area
    6.  Cumulative energy consumption
    7.  Per-zone reheat coil power                   (5-panel)
    8.  KPI summary bar charts (thermal discomfort, IAQ, energy)
    9.  Comfort-violation timeline
    10. RL agent actions by category                 (multi-panel)
    11. Rewards curve (if available)
    12. Outdoor air temperature
    """
    _set_pub_style()
    os.makedirs(save_dir, exist_ok=True)

    sim_times = simulator.dateTimeSteps
    dt_s = (sim_times[1] - sim_times[0]).total_seconds()
    ts_idx = pd.DatetimeIndex(sim_times).tz_convert('Europe/Copenhagen')
    zones = ZONE_ORDER
    total_floor_area = 1662.664

    # ── Figure 1: Zone temperatures + comfort bands (5-panel) ─────────
    fig, axes = plt.subplots(len(zones), 1, figsize=(14, 3.2 * len(zones)),
                             sharex=True)
    for ax, zone in zip(axes, zones):
        temp = _make_ts(
            simulator.model.components[f"{zone}_indoor_temp_sensor"].savedOutput["measuredValue"],
            sim_times, plotting_stepSize)
        heat_sp = _make_ts(
            simulator.model.components[f"{zone}_temperature_heating_setpoint"].savedOutput["scheduleValue"],
            sim_times, plotting_stepSize)
        cool_sp = _make_ts(
            simulator.model.components[f"{zone}_temperature_cooling_setpoint"].savedOutput["scheduleValue"],
            sim_times, plotting_stepSize)

        ax.fill_between(heat_sp.index, heat_sp.values, cool_sp.values,
                        alpha=0.18, color='#4CAF50', label='Comfort band')
        ax.plot(heat_sp.index, heat_sp.values, color='#4CAF50', ls='--', lw=1.0, alpha=0.6)
        ax.plot(cool_sp.index, cool_sp.values, color='#4CAF50', ls='--', lw=1.0, alpha=0.6)
        ax.plot(temp.index, temp.values, color=ZONE_COLORS[zone], lw=1.5,
                label=f'{ZONE_LABELS[zone]} temp.')
        ax.set_ylabel('Temp.  [°C]')
        ax.legend(loc='upper right', ncol=2, fontsize=11)
        ax.set_title(f'{ZONE_LABELS[zone]} Zone', fontsize=14, fontweight='bold', loc='left')

    axes[-1].set_xlabel('Time')
    fig.autofmt_xdate(rotation=30)
    fig.suptitle('Zone Temperatures and Comfort Bands', fontsize=18, y=1.005)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'zone_temperatures_comfort.png'),
                bbox_inches='tight')
    plt.close(fig)

    # ── Figure 2: All-zone temperature comparison (single panel) ──────
    fig, ax = plt.subplots(figsize=(14, 5))
    for zone in zones:
        temp = _make_ts(
            simulator.model.components[f"{zone}_indoor_temp_sensor"].savedOutput["measuredValue"],
            sim_times, plotting_stepSize)
        ax.plot(temp.index, temp.values, color=ZONE_COLORS[zone], lw=1.3,
                label=ZONE_LABELS[zone])
    ref_heat = _make_ts(
        simulator.model.components["core_temperature_heating_setpoint"].savedOutput["scheduleValue"],
        sim_times, plotting_stepSize)
    ref_cool = _make_ts(
        simulator.model.components["core_temperature_cooling_setpoint"].savedOutput["scheduleValue"],
        sim_times, plotting_stepSize)
    ax.fill_between(ref_heat.index, ref_heat.values, ref_cool.values,
                    alpha=0.12, color='#4CAF50', label='Comfort band (Core)')
    ax.set_ylabel('Temperature  [°C]')
    ax.set_xlabel('Time')
    ax.set_title('All-Zone Temperature Comparison')
    ax.legend(ncol=3, fontsize=12)
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'zone_temperatures_comparison.png'))
    plt.close(fig)

    # ── Figure 3: Zone CO₂ with setpoint (5-panel) ───────────────────
    fig, axes = plt.subplots(len(zones), 1, figsize=(14, 3.2 * len(zones)),
                             sharex=True)
    for ax, zone in zip(axes, zones):
        co2 = _make_ts(
            simulator.model.components[f"{zone}_co2_sensor"].savedOutput["measuredValue"],
            sim_times, plotting_stepSize)
        co2_sp = _make_ts(
            simulator.model.components[f"{zone}_co2_setpoint"].savedOutput["scheduleValue"],
            sim_times, plotting_stepSize)
        ax.plot(co2.index, co2.values, color=ZONE_COLORS[zone], lw=1.5,
                label=f'{ZONE_LABELS[zone]} CO₂')
        ax.plot(co2_sp.index, co2_sp.values, color='#B71C1C', ls='--', lw=1.2,
                alpha=0.7, label='Setpoint')
        ax.set_ylabel('CO₂  [ppm]')
        ax.legend(loc='upper right', ncol=2, fontsize=11)
        ax.set_title(f'{ZONE_LABELS[zone]} Zone', fontsize=14,
                     fontweight='bold', loc='left')

    axes[-1].set_xlabel('Time')
    fig.autofmt_xdate(rotation=30)
    fig.suptitle('Zone CO₂ Concentrations', fontsize=18, y=1.005)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'zone_co2_overview.png'),
                bbox_inches='tight')
    plt.close(fig)

    # ── Figure 4: AHU signals overview ────────────────────────────────
    ahu_signals = [
        ('vent_supply_air_temp_sensor', 'measuredValue', 'Supply Air Temp.', '°C'),
        ('vent_return_air_temp_sensor', 'measuredValue', 'Return Air Temp.', '°C'),
        ('vent_mixed_air_temp_sensor',  'measuredValue', 'Mixed Air Temp.',  '°C'),
        ('vent_supply_airflow_sensor',  'measuredValue', 'Supply Airflow',   'kg/s'),
        ('vent_return_airflow_sensor',  'measuredValue', 'Return Airflow',   'kg/s'),
        ('vent_power_sensor',           'measuredValue', 'Fan Power',        'W'),
    ]
    n_ahu = len(ahu_signals)
    ahu_palette = ['#D32F2F', '#1565C0', '#00838F', '#2E7D32', '#F57F17', '#7B1FA2']
    fig, axes = plt.subplots(n_ahu, 1, figsize=(14, 3 * n_ahu), sharex=True)
    for ax, (cid, oval, label, unit), clr in zip(axes, ahu_signals, ahu_palette):
        s = _make_ts(simulator.model.components[cid].savedOutput[oval],
                     sim_times, plotting_stepSize)
        ax.plot(s.index, s.values, color=clr, lw=1.5, label=label)
        ax.set_ylabel(f'{label}\n[{unit}]', fontsize=13)
        ax.legend(loc='upper right', fontsize=11)

    axes[-1].set_xlabel('Time')
    fig.autofmt_xdate(rotation=30)
    fig.suptitle('AHU Signals', fontsize=18, y=1.005)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'ahu_overview.png'), bbox_inches='tight')
    plt.close(fig)

    # ── Energy computation ────────────────────────────────────────────
    fan_power = np.asarray(
        simulator.model.components["vent_power_sensor"].savedOutput["measuredValue"])
    supply_heating_pwr = np.asarray(
        simulator.model.components["supply_heating_coil"].savedOutput["Power"])
    supply_cooling_pwr = np.asarray(
        simulator.model.components["supply_cooling_coil"].savedOutput["Power"])

    zone_reheat_pwr = {}
    total_reheat = np.zeros_like(fan_power)
    for zone in zones:
        q = _compute_reheat_power(simulator, zone)
        zone_reheat_pwr[zone] = q
        total_reheat += q

    total_hvac = fan_power + supply_heating_pwr + supply_cooling_pwr + total_reheat

    # ── Figure 5: Stacked energy breakdown ────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    ts_fan    = _make_ts(fan_power, sim_times, plotting_stepSize)
    ts_heat   = _make_ts(supply_heating_pwr, sim_times, plotting_stepSize)
    ts_cool   = _make_ts(supply_cooling_pwr, sim_times, plotting_stepSize)
    ts_reheat = _make_ts(total_reheat, sim_times, plotting_stepSize)
    idx = ts_fan.index

    ax.fill_between(idx, 0, ts_fan.values, alpha=0.65,
                    color=ENERGY_COLORS['fan'], label='Fan')
    b = ts_fan.values.copy()
    ax.fill_between(idx, b, b + ts_heat.values, alpha=0.65,
                    color=ENERGY_COLORS['heating_coil'], label='AHU Heating Coil')
    b = b + ts_heat.values
    ax.fill_between(idx, b, b + ts_cool.values, alpha=0.65,
                    color=ENERGY_COLORS['cooling_coil'], label='AHU Cooling Coil')
    b = b + ts_cool.values
    ax.fill_between(idx, b, b + ts_reheat.values, alpha=0.65,
                    color=ENERGY_COLORS['reheat_coils'], label='Zone Reheat Coils')

    ax.set_ylabel('Power  [W]')
    ax.set_xlabel('Time')
    ax.set_title('HVAC Power Consumption Breakdown')
    ax.legend(loc='upper right', fontsize=13)
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'energy_breakdown_stacked.png'))
    plt.close(fig)

    # ── Figure 6: Cumulative energy ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    cum_fan    = np.cumsum(fan_power) * dt_s / 3.6e6
    cum_heat   = np.cumsum(supply_heating_pwr) * dt_s / 3.6e6
    cum_cool   = np.cumsum(supply_cooling_pwr) * dt_s / 3.6e6
    cum_reheat = np.cumsum(total_reheat) * dt_s / 3.6e6
    cum_total  = cum_fan + cum_heat + cum_cool + cum_reheat

    ax.plot(ts_idx, cum_total, 'k', lw=2, label='Total HVAC')
    ax.plot(ts_idx, cum_fan,    color=ENERGY_COLORS['fan'], lw=1.3, ls='--', label='Fan')
    ax.plot(ts_idx, cum_heat,   color=ENERGY_COLORS['heating_coil'], lw=1.3, ls='--', label='AHU Heating')
    ax.plot(ts_idx, cum_cool,   color=ENERGY_COLORS['cooling_coil'], lw=1.3, ls='--', label='AHU Cooling')
    ax.plot(ts_idx, cum_reheat, color=ENERGY_COLORS['reheat_coils'], lw=1.3, ls='--', label='Reheat Coils')

    ax.set_ylabel('Cumulative Energy  [kWh]')
    ax.set_xlabel('Time')
    ax.set_title('Cumulative HVAC Energy Consumption')
    ax.legend(fontsize=13)
    fig.autofmt_xdate(rotation=30)
    final_kwh = cum_total[-1]
    ax.text(0.98, 0.05,
            f"Total: {final_kwh:.1f} kWh  ({final_kwh / total_floor_area:.3f} kWh/m²)",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=13,
            bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'cumulative_energy.png'))
    plt.close(fig)

    # ── Figure 7: Per-zone reheat coil power ──────────────────────────
    fig, axes = plt.subplots(len(zones), 1, figsize=(14, 3 * len(zones)),
                             sharex=True)
    for ax, zone in zip(axes, zones):
        qs = _make_ts(zone_reheat_pwr[zone], sim_times, plotting_stepSize)
        ax.fill_between(qs.index, 0, qs.values, alpha=0.45, color=ZONE_COLORS[zone])
        ax.plot(qs.index, qs.values, color=ZONE_COLORS[zone], lw=1.2)
        avg_q = np.mean(zone_reheat_pwr[zone])
        ax.set_ylabel('Power  [W]')
        ax.set_title(f'{ZONE_LABELS[zone]} Reheat Coil  (avg {avg_q:.0f} W)',
                     fontsize=14, fontweight='bold', loc='left')

    axes[-1].set_xlabel('Time')
    fig.autofmt_xdate(rotation=30)
    fig.suptitle('Zone Reheat Coil Power', fontsize=18, y=1.005)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'zone_reheat_power.png'),
                bbox_inches='tight')
    plt.close(fig)

    # ── KPI computation ───────────────────────────────────────────────
    zone_temps = np.array([
        simulator.model.components[f"{z}_indoor_temp_sensor"].savedOutput["measuredValue"]
        for z in zones]).T
    zone_heat_sp = np.array([
        simulator.model.components[f"{z}_temperature_heating_setpoint"].savedOutput["scheduleValue"]
        for z in zones]).T
    zone_cool_sp = np.array([
        simulator.model.components[f"{z}_temperature_cooling_setpoint"].savedOutput["scheduleValue"]
        for z in zones]).T
    zone_co2_arr = np.array([
        simulator.model.components[f"{z}_co2_sensor"].savedOutput["measuredValue"]
        for z in zones]).T
    zone_co2_sp = np.array([
        simulator.model.components[f"{z}_co2_setpoint"].savedOutput["scheduleValue"]
        for z in zones]).T

    tdis_tot, per_zone_tdis = thermal_discomfort(
        zone_temps, zone_heat_sp, zone_cool_sp, dt_s)
    iad_tot, per_zone_iad = iaq_violation_building(
        zone_co2_arr, zone_co2_sp, dt_s, total_floor_area)

    time_arr = np.arange(len(fan_power)) * dt_s
    fan_kwh, fan_kwh_m2       = calculate_energy_boptest_style(fan_power, time_arr, total_floor_area)
    heat_kwh, heat_kwh_m2     = calculate_energy_boptest_style(supply_heating_pwr, time_arr, total_floor_area)
    cool_kwh, cool_kwh_m2     = calculate_energy_boptest_style(supply_cooling_pwr, time_arr, total_floor_area)
    reheat_kwh, reheat_kwh_m2 = calculate_energy_boptest_style(total_reheat, time_arr, total_floor_area)
    total_kwh, total_kwh_m2   = calculate_energy_boptest_style(total_hvac, time_arr, total_floor_area)

    sep = '=' * 80
    print(f"\n{sep}")
    print("PERFORMANCE KPI SUMMARY")
    print(sep)
    print(f"\n  Energy (ener_tot):     {total_kwh_m2:.3f} kWh/m²  ({total_kwh:.1f} kWh)")
    print(f"    Fan:                 {fan_kwh_m2:.3f} kWh/m²  ({fan_kwh:.1f} kWh)")
    print(f"    AHU Heating Coil:    {heat_kwh_m2:.3f} kWh/m²  ({heat_kwh:.1f} kWh)")
    print(f"    AHU Cooling Coil:    {cool_kwh_m2:.3f} kWh/m²  ({cool_kwh:.1f} kWh)")
    print(f"    Zone Reheat Coils:   {reheat_kwh_m2:.3f} kWh/m²  ({reheat_kwh:.1f} kWh)")
    print(f"\n  Thermal discomfort (tdis_tot):  {tdis_tot:.2f} K·h/zone")
    for i, z in enumerate(zones):
        print(f"    {z:8s}: {per_zone_tdis[i]:7.2f} K·h")
    print(f"\n  IAQ violation (iad_tot):  {iad_tot:.2f} ppm·h/zone")
    for i, z in enumerate(zones):
        print(f"    {z:8s}: {per_zone_iad[i]:7.2f} ppm·h")
    print(sep)

    # ── Figure 8: KPI summary (3-panel bar chart) ─────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(zones))
    zlabels = [ZONE_LABELS[z] for z in zones]
    zcolors = [ZONE_COLORS[z] for z in zones]

    bars = ax1.bar(x, per_zone_tdis, color=zcolors, edgecolor='white', lw=0.6)
    ax1.set_ylabel('Thermal Discomfort  [K·h]')
    ax1.set_title(f'Thermal Discomfort  (avg {tdis_tot:.2f} K·h/zone)')
    ax1.set_xticks(x); ax1.set_xticklabels(zlabels)
    for b, v in zip(bars, per_zone_tdis):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=11)

    bars = ax2.bar(x, per_zone_iad, color=zcolors, edgecolor='white', lw=0.6)
    ax2.set_ylabel('IAQ Violation  [ppm·h]')
    ax2.set_title(f'CO₂ Violations  (iad_tot {iad_tot:.2f} ppm·h/zone)')
    ax2.set_xticks(x); ax2.set_xticklabels(zlabels)
    for b, v in zip(bars, per_zone_iad):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=11)

    e_labels = ['Fan', 'AHU\nHeat', 'AHU\nCool', 'Reheat\nCoils']
    e_vals = [fan_kwh_m2, heat_kwh_m2, cool_kwh_m2, reheat_kwh_m2]
    e_colors = list(ENERGY_COLORS.values())
    bars = ax3.bar(np.arange(len(e_labels)), e_vals, color=e_colors,
                   edgecolor='white', lw=0.6)
    ax3.set_ylabel('Energy  [kWh/m²]')
    ax3.set_title(f'Energy Breakdown  (total {total_kwh_m2:.3f} kWh/m²)')
    ax3.set_xticks(np.arange(len(e_labels))); ax3.set_xticklabels(e_labels)
    for b, v in zip(bars, e_vals):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'kpi_summary.png'))
    plt.close(fig)

    # ── Figure 9: Comfort-violation timeline ──────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, zone in enumerate(zones):
        violation = (np.maximum(zone_heat_sp[:, i] - zone_temps[:, i], 0)
                     + np.maximum(zone_temps[:, i] - zone_cool_sp[:, i], 0))
        ax.plot(ts_idx, violation, color=ZONE_COLORS[zone], lw=0.9, alpha=0.85,
                label=ZONE_LABELS[zone])

    ax.set_ylabel('Temperature Deviation  [K]')
    ax.set_xlabel('Time')
    ax.set_title('Comfort Band Violations Over Time')
    ax.legend(loc='upper right', ncol=5, fontsize=11)
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'comfort_violations_timeline.png'))
    plt.close(fig)

    # ── Figure 10: RL actions by category ─────────────────────────────
    try:
        with open(POLICY_CONFIG_PATH, 'r') as f:
            policy_config = json.load(f)

        action_items = []
        for cid, acts in policy_config['actions'].items():
            for _aname, acfg in acts.items():
                action_items.append((cid, acfg['signal_key']))

        if action_items:
            ahu_items = [(c, k) for c, k in action_items
                         if 'supply_air_temp' in c.lower()]
            dmp_items = [(c, k) for c, k in action_items
                         if 'damper' in c.lower()]
            sp_items = [(c, k) for c, k in action_items
                        if (c, k) not in ahu_items and (c, k) not in dmp_items]

            def _plot_action_group(items, title, fname):
                if not items:
                    return
                n = len(items)
                fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=True)
                if n == 1:
                    axes = [axes]
                colors = plt.cm.Set2(np.linspace(0, 1, max(n, 3)))
                for ax, (cid, skey), clr in zip(axes, items, colors):
                    raw = simulator.model.components[cid].savedInput[skey]
                    s = _make_ts(raw, sim_times, plotting_stepSize)
                    label = ACTION_LABELS.get((cid, skey), f'{cid} / {skey}')
                    ax.plot(s.index, s.values, color=clr, lw=1.3)
                    ax.set_ylabel(label, fontsize=12)
                axes[-1].set_xlabel('Time')
                fig.autofmt_xdate(rotation=30)
                fig.suptitle(title, fontsize=18, y=1.005)
                plt.tight_layout()
                fig.savefig(os.path.join(save_dir, fname), bbox_inches='tight')
                plt.close(fig)

            _plot_action_group(ahu_items,
                               'RL Actions — AHU Supply Temperature',
                               'actions_ahu_supply_temp.png')
            _plot_action_group(sp_items,
                               'RL Actions — Zone Setpoints',
                               'actions_zone_setpoints.png')
            _plot_action_group(dmp_items,
                               'RL Actions — Damper Positions',
                               'actions_damper_positions.png')

    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # ── Figure 11: Rewards ────────────────────────────────────────────
    if rewards is not None and len(rewards) > 0:
        r = np.asarray(rewards, dtype=float).flatten()
        steps = np.arange(len(r))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax1.plot(steps, r, color='#1565C0', lw=0.8, alpha=0.5, label='Per-step')
        window = max(1, min(50, len(r) // 5))
        if window > 1:
            r_smooth = pd.Series(r).rolling(window, center=True).mean().values
            ax1.plot(steps, r_smooth, color='#D32F2F', lw=2,
                     label=f'{window}-step moving avg.')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards')
        ax1.legend(fontsize=13)

        ax2.plot(steps, np.cumsum(r), color='#2E7D32', lw=2)
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_xlabel('Step')
        ax2.set_title(f'Cumulative Reward  (total: {np.sum(r):.2f})')
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, 'rewards.png'))
        plt.close(fig)

    # ── Figure 12: Outdoor air temperature ────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    outdoor = _make_ts(
        simulator.model.components["vent_outdoor_air_temp_sensor"].savedOutput["measuredValue"],
        sim_times, plotting_stepSize)
    ax.plot(outdoor.index, outdoor.values, color='#00838F', lw=1.5)
    ax.set_ylabel('Temperature  [°C]')
    ax.set_xlabel('Time')
    ax.set_title('Outdoor Air Temperature')
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'outdoor_temperature.png'))
    plt.close(fig)

    # ── Figure 13: Compact dashboard (2×2) ────────────────────────────
    fig, ((d1, d2), (d3, d4)) = plt.subplots(2, 2, figsize=(18, 12))

    for zone in zones:
        temp = _make_ts(
            simulator.model.components[f"{zone}_indoor_temp_sensor"].savedOutput["measuredValue"],
            sim_times, plotting_stepSize)
        d1.plot(temp.index, temp.values, color=ZONE_COLORS[zone], lw=1.2,
                label=ZONE_LABELS[zone])
    d1.fill_between(ref_heat.index, ref_heat.values, ref_cool.values,
                    alpha=0.12, color='#4CAF50')
    d1.set_ylabel('Temperature  [°C]')
    d1.set_title('Zone Temperatures')
    d1.legend(ncol=5, fontsize=10, loc='upper right')

    for zone in zones:
        co2 = _make_ts(
            simulator.model.components[f"{zone}_co2_sensor"].savedOutput["measuredValue"],
            sim_times, plotting_stepSize)
        d2.plot(co2.index, co2.values, color=ZONE_COLORS[zone], lw=1.2,
                label=ZONE_LABELS[zone])
    d2.set_ylabel('CO₂  [ppm]')
    d2.set_title('Zone CO₂ Concentrations')
    d2.legend(ncol=5, fontsize=10, loc='upper right')

    ts_total = _make_ts(total_hvac, sim_times, plotting_stepSize)
    d3.fill_between(ts_total.index, 0, ts_total.values, alpha=0.4, color='#455A64')
    d3.plot(ts_total.index, ts_total.values, color='#263238', lw=1.2)
    d3.set_ylabel('Power  [W]')
    d3.set_title(f'Total HVAC Power  (total {total_kwh_m2:.3f} kWh/m²)')

    bar_labels = [ZONE_LABELS[z] for z in zones]
    bar_x = np.arange(len(zones))
    d4.bar(bar_x - 0.2, per_zone_tdis, 0.35, color='#EF5350', label='Thermal [K·h]')
    d4.bar(bar_x + 0.2, per_zone_iad, 0.35, color='#42A5F5', label='IAQ [ppm·h]')
    d4.set_xticks(bar_x); d4.set_xticklabels(bar_labels)
    d4.set_title('Per-Zone Discomfort & IAQ Violations')
    d4.legend(fontsize=12)

    for a in [d1, d2, d3]:
        a.tick_params(axis='x', rotation=30)
    fig.suptitle('Control Performance Dashboard', fontsize=20, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'dashboard.png'), bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll publication plots saved to: {os.path.abspath(save_dir)}/")


def plot_baseline_vs_rl(baseline_simulator, rl_simulator,
                        plotting_stepSize=600, save_dir='plots_comparison'):
    """Overlay baseline and RL control results for direct comparison.

    Figures produced
    ----------------
    1.  Per-zone indoor temperature (5-panel, each zone individually)
    2.  AHU supply air temperature
    3.  Controller action signals by category (individual panels)
    """
    _set_pub_style()
    os.makedirs(save_dir, exist_ok=True)

    bl_times = baseline_simulator.dateTimeSteps
    rl_times = rl_simulator.dateTimeSteps
    zones = ZONE_ORDER
    BL_COLOR = '#757575'

    # ── Per-zone temperature comparison (5-panel) ─────────────────────
    fig, axes = plt.subplots(len(zones), 1, figsize=(14, 3.5 * len(zones)),
                             sharex=True)
    for ax, zone in zip(axes, zones):
        bl_temp = _make_ts(
            baseline_simulator.model.components[f"{zone}_indoor_temp_sensor"]
            .savedOutput["measuredValue"], bl_times, plotting_stepSize)
        rl_temp = _make_ts(
            rl_simulator.model.components[f"{zone}_indoor_temp_sensor"]
            .savedOutput["measuredValue"], rl_times, plotting_stepSize)

        heat_sp = _make_ts(
            baseline_simulator.model.components[f"{zone}_temperature_heating_setpoint"]
            .savedOutput["scheduleValue"], bl_times, plotting_stepSize)
        cool_sp = _make_ts(
            baseline_simulator.model.components[f"{zone}_temperature_cooling_setpoint"]
            .savedOutput["scheduleValue"], bl_times, plotting_stepSize)

        ax.fill_between(heat_sp.index, heat_sp.values, cool_sp.values,
                        alpha=0.15, color='#4CAF50', label='Comfort band')
        ax.plot(heat_sp.index, heat_sp.values, color='#4CAF50', ls='--',
                lw=0.8, alpha=0.5)
        ax.plot(cool_sp.index, cool_sp.values, color='#4CAF50', ls='--',
                lw=0.8, alpha=0.5)
        ax.plot(bl_temp.index, bl_temp.values, color=BL_COLOR, lw=1.5,
                alpha=0.8, label='Baseline')
        ax.plot(rl_temp.index, rl_temp.values, color=ZONE_COLORS[zone],
                lw=1.5, label='RL Control')

        ax.set_ylabel('Temp. [°C]')
        ax.legend(loc='upper right', ncol=3, fontsize=11)
        ax.set_title(f'{ZONE_LABELS[zone]} Zone', fontsize=14,
                     fontweight='bold', loc='left')

    axes[-1].set_xlabel('Time')
    fig.autofmt_xdate(rotation=30)
    fig.suptitle('Zone Temperatures — Baseline vs RL Control',
                 fontsize=18, y=1.005)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'compare_zone_temperatures.png'),
                bbox_inches='tight')
    plt.close(fig)

    # ── Individual per-zone temperature figures ─────────────────────────
    for zone in zones:
        fig, ax = plt.subplots(figsize=(14, 5))
        bl_temp = _make_ts(
            baseline_simulator.model.components[f"{zone}_indoor_temp_sensor"]
            .savedOutput["measuredValue"], bl_times, plotting_stepSize)
        rl_temp = _make_ts(
            rl_simulator.model.components[f"{zone}_indoor_temp_sensor"]
            .savedOutput["measuredValue"], rl_times, plotting_stepSize)
        heat_sp = _make_ts(
            baseline_simulator.model.components[f"{zone}_temperature_heating_setpoint"]
            .savedOutput["scheduleValue"], bl_times, plotting_stepSize)
        cool_sp = _make_ts(
            baseline_simulator.model.components[f"{zone}_temperature_cooling_setpoint"]
            .savedOutput["scheduleValue"], bl_times, plotting_stepSize)

        ax.fill_between(heat_sp.index, heat_sp.values, cool_sp.values,
                        alpha=0.15, color='#4CAF50', label='Comfort band')
        ax.plot(heat_sp.index, heat_sp.values, color='#4CAF50', ls='--',
                lw=0.8, alpha=0.5)
        ax.plot(cool_sp.index, cool_sp.values, color='#4CAF50', ls='--',
                lw=0.8, alpha=0.5)
        ax.plot(bl_temp.index, bl_temp.values, color=BL_COLOR, lw=1.5,
                alpha=0.8, label='Baseline')
        ax.plot(rl_temp.index, rl_temp.values, color=ZONE_COLORS[zone],
                lw=1.5, label='RL Control')

        ax.set_ylabel('Temperature [°C]')
        ax.set_xlabel('Time')
        ax.set_title(f'{ZONE_LABELS[zone]} Zone Temperature — Baseline vs RL Control')
        ax.legend(loc='upper right', ncol=3, fontsize=12)
        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f'compare_temp_{zone}.png'))
        plt.close(fig)

    # ── AHU supply air temperature ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    bl_sat = _make_ts(
        baseline_simulator.model.components["vent_supply_air_temp_sensor"]
        .savedOutput["measuredValue"], bl_times, plotting_stepSize)
    rl_sat = _make_ts(
        rl_simulator.model.components["vent_supply_air_temp_sensor"]
        .savedOutput["measuredValue"], rl_times, plotting_stepSize)

    ax.plot(bl_sat.index, bl_sat.values, color=BL_COLOR, lw=1.5,
            alpha=0.8, label='Baseline')
    ax.plot(rl_sat.index, rl_sat.values, color='#D32F2F', lw=1.5,
            label='RL Control')
    ax.set_ylabel('Supply Air Temp. [°C]')
    ax.set_xlabel('Time')
    ax.set_title('AHU Supply Air Temperature — Baseline vs RL Control')
    ax.legend(fontsize=13)
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'compare_ahu_supply_temp.png'))
    plt.close(fig)

    # ── Controller action signals ─────────────────────────────────────
    try:
        with open(POLICY_CONFIG_PATH, 'r') as f:
            policy_config = json.load(f)

        action_items = []
        for cid, acts in policy_config['actions'].items():
            for _aname, acfg in acts.items():
                action_items.append((cid, acfg['signal_key']))

        if action_items:
            ahu_items = [(c, k) for c, k in action_items
                         if 'supply_air_temp' in c.lower()]
            dmp_items = [(c, k) for c, k in action_items
                         if 'damper' in c.lower()]
            sp_items = [(c, k) for c, k in action_items
                        if (c, k) not in ahu_items and (c, k) not in dmp_items]

            def _plot_cmp_group(items, title, fname):
                if not items:
                    return
                n = len(items)
                fig, axes_g = plt.subplots(n, 1, figsize=(14, 3 * n),
                                           sharex=True)
                if n == 1:
                    axes_g = [axes_g]
                for ax_g, (cid, skey) in zip(axes_g, items):
                    bl_raw = baseline_simulator.model.components[cid].savedInput[skey]
                    rl_raw = rl_simulator.model.components[cid].savedInput[skey]
                    bl_s = _make_ts(bl_raw, bl_times, plotting_stepSize)
                    rl_s = _make_ts(rl_raw, rl_times, plotting_stepSize)
                    label = ACTION_LABELS.get((cid, skey), f'{cid} / {skey}')
                    zone_prefix = cid.split('_')[0]
                    rl_clr = ZONE_COLORS.get(zone_prefix, '#1565C0')

                    ax_g.plot(bl_s.index, bl_s.values, color=BL_COLOR,
                              lw=1.3, alpha=0.8, label='Baseline')
                    ax_g.plot(rl_s.index, rl_s.values, color=rl_clr,
                              lw=1.3, label='RL Control')
                    ax_g.set_ylabel(label, fontsize=11)
                    ax_g.legend(loc='upper right', ncol=2, fontsize=10)

                axes_g[-1].set_xlabel('Time')
                fig.autofmt_xdate(rotation=30)
                fig.suptitle(title, fontsize=18, y=1.005)
                plt.tight_layout()
                fig.savefig(os.path.join(save_dir, fname),
                            bbox_inches='tight')
                plt.close(fig)

            def _plot_cmp_individual(items):
                for cid, skey in items:
                    bl_raw = baseline_simulator.model.components[cid].savedInput[skey]
                    rl_raw = rl_simulator.model.components[cid].savedInput[skey]
                    bl_s = _make_ts(bl_raw, bl_times, plotting_stepSize)
                    rl_s = _make_ts(rl_raw, rl_times, plotting_stepSize)
                    label = ACTION_LABELS.get((cid, skey), f'{cid} / {skey}')
                    zone_prefix = cid.split('_')[0]
                    rl_clr = ZONE_COLORS.get(zone_prefix, '#1565C0')

                    fig, ax = plt.subplots(figsize=(14, 5))
                    ax.plot(bl_s.index, bl_s.values, color=BL_COLOR,
                            lw=1.5, alpha=0.8, label='Baseline')
                    ax.plot(rl_s.index, rl_s.values, color=rl_clr,
                            lw=1.5, label='RL Control')
                    ax.set_ylabel(label, fontsize=14)
                    ax.set_xlabel('Time')
                    ax.set_title(f'{label} — Baseline vs RL Control')
                    ax.legend(loc='upper right', fontsize=12)
                    fig.autofmt_xdate(rotation=30)
                    plt.tight_layout()
                    safe_name = f'{cid}_{skey}'.replace(' ', '_')
                    fig.savefig(os.path.join(save_dir,
                                f'compare_action_{safe_name}.png'))
                    plt.close(fig)

            _plot_cmp_group(ahu_items,
                            'AHU Supply Temp. Setpoint — Baseline vs RL',
                            'compare_actions_ahu_supply_temp.png')
            _plot_cmp_group(sp_items,
                            'Zone Setpoints — Baseline vs RL',
                            'compare_actions_zone_setpoints.png')
            _plot_cmp_group(dmp_items,
                            'Damper Positions — Baseline vs RL',
                            'compare_actions_damper_positions.png')

            _plot_cmp_individual(ahu_items)
            _plot_cmp_individual(sp_items)
            _plot_cmp_individual(dmp_items)

            # ── Compact 2-panel per-room figures ──────────────────────
            CLR_RL_TEMP     = '#E53935'
            CLR_BL_DARK     = '#333333'
            CLR_RL_HEAT_SP  = '#D84315'
            CLR_RL_COOL_SP  = '#0D47A1'
            CLR_RL_DAMPER   = '#1565C0'
            CLR_COMFORT     = '#43A047'

            for zone in zones:
                fig, (ax_top, ax_bot) = plt.subplots(
                    2, 1, figsize=(14, 8.5), sharex=True,
                    gridspec_kw={'height_ratios': [3, 1.2]})

                # ── Top: temperature + comfort band + setpoint actions ──
                bl_temp = _make_ts(
                    baseline_simulator.model.components[f"{zone}_indoor_temp_sensor"]
                    .savedOutput["measuredValue"], bl_times, plotting_stepSize)
                rl_temp = _make_ts(
                    rl_simulator.model.components[f"{zone}_indoor_temp_sensor"]
                    .savedOutput["measuredValue"], rl_times, plotting_stepSize)
                heat_sp_sched = _make_ts(
                    baseline_simulator.model.components[f"{zone}_temperature_heating_setpoint"]
                    .savedOutput["scheduleValue"], bl_times, plotting_stepSize)
                cool_sp_sched = _make_ts(
                    baseline_simulator.model.components[f"{zone}_temperature_cooling_setpoint"]
                    .savedOutput["scheduleValue"], bl_times, plotting_stepSize)

                heat_cid = f'{zone}_temperature_heating_controller'
                cool_cid = heat_cid
                damp_cid = f'{zone}_supply_damper_position_sensor'

                bl_heat_sp = _make_ts(
                    baseline_simulator.model.components[heat_cid]
                    .savedInput['heatingsetpointValue'], bl_times, plotting_stepSize)
                rl_heat_sp = _make_ts(
                    rl_simulator.model.components[heat_cid]
                    .savedInput['heatingsetpointValue'], rl_times, plotting_stepSize)
                bl_cool_sp = _make_ts(
                    baseline_simulator.model.components[cool_cid]
                    .savedInput['coolingsetpointValue'], bl_times, plotting_stepSize)
                rl_cool_sp = _make_ts(
                    rl_simulator.model.components[cool_cid]
                    .savedInput['coolingsetpointValue'], rl_times, plotting_stepSize)

                ax_top.fill_between(heat_sp_sched.index,
                                    heat_sp_sched.values, cool_sp_sched.values,
                                    alpha=0.22, color=CLR_COMFORT,
                                    label='Comfort band')
                ax_top.plot(heat_sp_sched.index, heat_sp_sched.values,
                            color=CLR_COMFORT, ls='--', lw=0.9, alpha=0.6)
                ax_top.plot(cool_sp_sched.index, cool_sp_sched.values,
                            color=CLR_COMFORT, ls='--', lw=0.9, alpha=0.6)

                ax_top.plot(bl_temp.index, bl_temp.values, color=CLR_BL_DARK,
                            ls='--', lw=1.8, label='Baseline temp.')
                ax_top.plot(rl_temp.index, rl_temp.values, color=CLR_RL_TEMP,
                            lw=2.0, label='RL temp.')

                ax_top.plot(bl_heat_sp.index, bl_heat_sp.values,
                            color=CLR_RL_HEAT_SP, ls=':', lw=1.2, alpha=0.55,
                            label='Baseline heat. SP')
                ax_top.plot(rl_heat_sp.index, rl_heat_sp.values,
                            color=CLR_RL_HEAT_SP, lw=1.4,
                            label='RL heating SP')
                ax_top.plot(bl_cool_sp.index, bl_cool_sp.values,
                            color=CLR_RL_COOL_SP, ls=':', lw=1.2, alpha=0.55,
                            label='Baseline cool. SP')
                ax_top.plot(rl_cool_sp.index, rl_cool_sp.values,
                            color=CLR_RL_COOL_SP, lw=1.4,
                            label='RL cooling SP')

                ax_top.set_ylabel('Temperature [°C]')
                ax_top.legend(loc='upper right', ncol=4, fontsize=10,
                              framealpha=0.9)
                ax_top.set_title(
                    f'{ZONE_LABELS[zone]} Zone — Temperature & Setpoints',
                    fontsize=15, fontweight='bold', loc='left')

                # ── Bottom: damper position ─────────────────────────────
                bl_damp = _make_ts(
                    baseline_simulator.model.components[damp_cid]
                    .savedInput['measuredValue'], bl_times, plotting_stepSize)
                rl_damp = _make_ts(
                    rl_simulator.model.components[damp_cid]
                    .savedInput['measuredValue'], rl_times, plotting_stepSize)

                ax_bot.plot(bl_damp.index, bl_damp.values, color=CLR_BL_DARK,
                            ls='--', lw=1.6, label='Baseline')
                ax_bot.plot(rl_damp.index, rl_damp.values, color=CLR_RL_DAMPER,
                            lw=1.8, label='RL Control')
                ax_bot.set_ylabel('Damper Pos. [–]')
                ax_bot.set_xlabel('Time')
                ax_bot.legend(loc='upper right', ncol=2, fontsize=11)
                ax_bot.set_title('Damper Position', fontsize=13,
                                 fontweight='bold', loc='left')

                fig.autofmt_xdate(rotation=30)
                plt.tight_layout()
                fig.savefig(os.path.join(save_dir,
                            f'zone_control_{zone}.png'),
                            bbox_inches='tight')
                plt.close(fig)

    except (FileNotFoundError, json.JSONDecodeError):
        pass

    print(f"\nAll comparison plots saved to: {os.path.abspath(save_dir)}/")


def plot_results(simulator: tb.Simulator, rewards = None, plotting_stepSize=600, save_plots=False):
        # Convert actions and rewards to pandas DataFrames
        if rewards is not None:
                rewards_df = pd.DataFrame(rewards)

        """
        What do I want to plot?
        - Temperature in the rooms
        - Temp setpoints in the rooms
        - CO2 in the rooms
        - Total energy consumption (Coils consumption + Fan consumption)
        - Some actions taken by the model
        - The rewards for the episode
        """
        model_output_points = [
            {
                'component_id': 'core_indoor_temp_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'core_co2_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'core_temperature_heating_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'core_temperature_cooling_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'north_indoor_temp_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'north_co2_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'north_temperature_heating_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'north_temperature_cooling_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'south_indoor_temp_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'south_co2_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'south_temperature_heating_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'south_temperature_cooling_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'east_indoor_temp_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'east_co2_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'east_temperature_heating_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'east_temperature_cooling_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'west_indoor_temp_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'west_co2_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'west_temperature_heating_setpoint',
                'output_value': 'scheduleValue'
            },
            {
                'component_id': 'west_temperature_cooling_setpoint',
                'output_value': 'scheduleValue'
            },
     
        ]
        
        # Create plots for each room
        rooms = ['core', 'north', 'south', 'east', 'west']
        for room in rooms:
            # Get indices for this room's components
            base_idx = rooms.index(room) * 4
            temp_sensor = model_output_points[base_idx]
            co2_sensor = model_output_points[base_idx + 1]
            heating_setpoint = model_output_points[base_idx + 2]
            cooling_setpoint = model_output_points[base_idx + 3]

            # Get simulation data for temperature and setpoints
            temp_data = simulator.model.components[temp_sensor['component_id']].savedOutput[temp_sensor['output_value']]
            heating_data = simulator.model.components[heating_setpoint['component_id']].savedOutput[heating_setpoint['output_value']]
            cooling_data = simulator.model.components[cooling_setpoint['component_id']].savedOutput[cooling_setpoint['output_value']]
            sim_times = simulator.dateTimeSteps

            # Create temperature plot
            plt.figure(figsize=(12, 6))
            temp_df = pd.Series(data=temp_data, index=sim_times)
            heating_df = pd.Series(data=heating_data, index=sim_times)
            cooling_df = pd.Series(data=cooling_data, index=sim_times)

            # Convert timezone without changing the actual timestamps
            temp_df.index = temp_df.index.tz_convert('Europe/Copenhagen')
            heating_df.index = heating_df.index.tz_convert('Europe/Copenhagen')
            cooling_df.index = cooling_df.index.tz_convert('Europe/Copenhagen')

            # Resample to common timestep
            temp_df = temp_df.resample(pd.Timedelta(seconds=plotting_stepSize)).mean()
            heating_df = heating_df.resample(pd.Timedelta(seconds=plotting_stepSize)).mean()
            cooling_df = cooling_df.resample(pd.Timedelta(seconds=plotting_stepSize)).mean()

            plt.plot(temp_df.index, temp_df.values, label='Indoor Temperature', linewidth=2)
            plt.plot(heating_df.index, heating_df.values, label='Heating Setpoint', linestyle='--', linewidth=2)
            plt.plot(cooling_df.index, cooling_df.values, label='Cooling Setpoint', linestyle='--', linewidth=2)
            
            plt.title(f'{room.capitalize()} Room - Temperature and Setpoints')
            plt.xlabel('Time')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_plots:
                    os.makedirs('plots', exist_ok=True)
                    plt.savefig(f'plots/{room}_temperature_setpoints.png')
            #plt.show()

            # Create CO2 plot
            plt.figure(figsize=(12, 6))
            co2_data = simulator.model.components[co2_sensor['component_id']].savedOutput[co2_sensor['output_value']]
            co2_df = pd.Series(data=co2_data, index=sim_times)
            co2_df.index = co2_df.index.tz_convert('Europe/Copenhagen')
            co2_df = co2_df.resample(pd.Timedelta(seconds=plotting_stepSize)).mean()

            plt.plot(co2_df.index, co2_df.values, label='CO2 Concentration', linewidth=2)
            plt.title(f'{room.capitalize()} Room - CO2 Concentration')
            plt.xlabel('Time')
            plt.ylabel('CO2 (ppm)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_plots:
                    os.makedirs('plots', exist_ok=True)
                    plt.savefig(f'plots/{room}_co2.png')
            #plt.show()


        #Create plots for the AHU quantities
        ahu_quantities = [
            {
                'component_id': 'vent_supply_air_temp_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'vent_mixed_air_temp_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'vent_supply_airflow_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'vent_return_airflow_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'vent_power_sensor',
                'output_value': 'measuredValue'
            },
            {
                'component_id': 'vent_return_air_temp_sensor',
                'output_value': 'measuredValue'
            }
        ]
        for quantity in ahu_quantities:
            plt.figure(figsize=(12, 6))
            quantity_data = simulator.model.components[quantity['component_id']].savedOutput[quantity['output_value']]
            quantity_df = pd.Series(data=quantity_data, index=sim_times)
            quantity_df.index = quantity_df.index.tz_convert('Europe/Copenhagen')
            quantity_df = quantity_df.resample(pd.Timedelta(seconds=plotting_stepSize)).mean()
            plt.plot(quantity_df.index, quantity_df.values, label=quantity['component_id'], linewidth=2)
            plt.title(f'{quantity["component_id"]} - {quantity["output_value"]}')
            plt.xlabel('Time')
            plt.ylabel(quantity['output_value'])
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_plots:
                os.makedirs('plots', exist_ok=True)
                plt.savefig(f'plots/{quantity["component_id"]}_{quantity["output_value"]}.png')
            #plt.show()

        #Calculate temperature violation penalty with 1-degree deadband
        step_size_seconds = 600  # Simulation step size in seconds
        
        # Core room temperature violations
        core_temperature = np.array(simulator.model.components["core_indoor_temp_sensor"].savedOutput["measuredValue"])
        core_heating_temperature_setpoint = np.array(simulator.model.components["core_temperature_heating_setpoint"].savedOutput["scheduleValue"])
        core_cooling_temperature_setpoint = np.array(simulator.model.components["core_temperature_cooling_setpoint"].savedOutput["scheduleValue"])
        
        # Apply 1-degree deadband: upper bound = cooling_setpoint + 1, lower bound = heating_setpoint - 1
        core_upper_bound = core_cooling_temperature_setpoint + 1
        core_lower_bound = core_heating_temperature_setpoint - 1
        core_violations = (core_temperature > core_upper_bound) | (core_temperature < core_lower_bound)
        core_temp_set_violation_seconds = np.sum(core_violations) * step_size_seconds
        print(f"Core temp set violation: {core_temp_set_violation_seconds} seconds")

        # North room temperature violations
        north_temperature = np.array(simulator.model.components["north_indoor_temp_sensor"].savedOutput["measuredValue"])
        north_heating_temperature_setpoint = np.array(simulator.model.components["north_temperature_heating_setpoint" ].savedOutput["scheduleValue"])
        north_cooling_temperature_setpoint = np.array(simulator.model.components["north_temperature_cooling_setpoint"].savedOutput["scheduleValue"])
        
        north_upper_bound = north_cooling_temperature_setpoint + 1
        north_lower_bound = north_heating_temperature_setpoint - 1
        north_violations = (north_temperature > north_upper_bound) | (north_temperature < north_lower_bound)
        north_temp_set_violation_seconds = np.sum(north_violations) * step_size_seconds
        print(f"North temp set violation: {north_temp_set_violation_seconds} seconds")

        # East room temperature violations
        east_temperature = np.array(simulator.model.components["east_indoor_temp_sensor"].savedOutput["measuredValue"])
        east_heating_temperature_setpoint = np.array(simulator.model.components["east_temperature_heating_setpoint"].savedOutput["scheduleValue"])
        east_cooling_temperature_setpoint = np.array(simulator.model.components["east_temperature_cooling_setpoint"].savedOutput["scheduleValue"])
        
        east_upper_bound = east_cooling_temperature_setpoint + 1
        east_lower_bound = east_heating_temperature_setpoint - 1
        east_violations = (east_temperature > east_upper_bound) | (east_temperature < east_lower_bound)
        east_temp_set_violation_seconds = np.sum(east_violations) * step_size_seconds
        print(f"East temp set violation: {east_temp_set_violation_seconds} seconds")

        # South room temperature violations
        south_temperature = np.array(simulator.model.components["south_indoor_temp_sensor"].savedOutput["measuredValue"])
        south_heating_temperature_setpoint = np.array(simulator.model.components["south_temperature_heating_setpoint"].savedOutput["scheduleValue"])
        south_cooling_temperature_setpoint = np.array(simulator.model.components["south_temperature_cooling_setpoint"].savedOutput["scheduleValue"])
        
        south_upper_bound = south_cooling_temperature_setpoint + 1
        south_lower_bound = south_heating_temperature_setpoint - 1
        south_violations = (south_temperature > south_upper_bound) | (south_temperature < south_lower_bound)
        south_temp_set_violation_seconds = np.sum(south_violations) * step_size_seconds
        print(f"South temp set violation: {south_temp_set_violation_seconds} seconds")

        # West room temperature violations
        west_temperature = np.array(simulator.model.components["west_indoor_temp_sensor"].savedOutput["measuredValue"])
        west_heating_temperature_setpoint = np.array(simulator.model.components["west_temperature_heating_setpoint"].savedOutput["scheduleValue"])
        west_cooling_temperature_setpoint = np.array(simulator.model.components["west_temperature_cooling_setpoint"].savedOutput["scheduleValue"])
        
        west_upper_bound = west_cooling_temperature_setpoint + 1
        west_lower_bound = west_heating_temperature_setpoint - 1
        west_violations = (west_temperature > west_upper_bound) | (west_temperature < west_lower_bound)
        west_temp_set_violation_seconds = np.sum(west_violations) * step_size_seconds
        print(f"West temp set violation: {west_temp_set_violation_seconds} seconds")

        # Total temperature violation time in seconds
        total_temp_violation_seconds = (core_temp_set_violation_seconds + north_temp_set_violation_seconds + 
                                      east_temp_set_violation_seconds + south_temp_set_violation_seconds + 
                                      west_temp_set_violation_seconds)
        print(f"Total temperature violation time: {total_temp_violation_seconds} seconds")



        #Room VAV coils power consumption
        zones = ['core', 'north', 'east', 'south', 'west']
        coils_power_consumption = []
        average_coils_power_consumption = []
        for zone in zones:
            airflow_rate = np.array(simulator.model.components[f"{zone}_reheat_coil"].savedInput["airFlowRate"])
            inlet_air_temp = np.array(simulator.model.components[f"{zone}_reheat_coil"].savedInput["inletAirTemperature"])
            outlet_air_temp = np.array(simulator.model.components[f"{zone}_reheat_coil"].savedOutput["outletAirTemperature"])
            
            tol = 1e-5
            specificHeatCapacityAir = 1005 #J/kg/K
            
            # Initialize Q array with zeros
            Q = np.zeros_like(airflow_rate)
            
            # Create mask for valid conditions
            valid_flow_mask = airflow_rate > tol
            # Add a deadband of 1 degree Celsius for heating
            heating_mask = (outlet_air_temp - inlet_air_temp) > 1.0
            if np.any(~heating_mask):
                print(f"Warning: Heating coil in {zone} is cooling the air")
            
            # Calculate power only where both conditions are met
            combined_mask = valid_flow_mask & heating_mask
            Q[combined_mask] = (airflow_rate[combined_mask] * 
                               specificHeatCapacityAir * 
                               (outlet_air_temp[combined_mask] - inlet_air_temp[combined_mask]))
            
            # Check for NaN values
            if np.any(np.isnan(Q)):
                raise ValueError("Q contains NaN values")
            
            # Calculate average power consumption
            avg_Q = np.average(Q)
            average_coils_power_consumption.append(avg_Q)
            
            coils_power_consumption.append(Q)
            

            print(f" Zone: {zone} - reheat coil average power consumption: {average_coils_power_consumption[-1]:.1f} W")

            #plot the zone coil power consumption
            plt.figure(figsize=(12, 6))
            plt.plot(sim_times, Q, label=f'{zone} - Reheat Coil Power', linewidth=2)
            plt.title(f'{zone} - Reheat Coil Power')
            plt.xlabel('Time')
            plt.ylabel('Power (W)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_plots:
                os.makedirs('plots', exist_ok=True)
                plt.savefig(f'plots/{zone}_reheat_coil_power.png')

            #Plot inlet and outlet air temperature
            plt.figure(figsize=(12, 6))
            plt.plot(sim_times, inlet_air_temp, label=f'{zone} - Inlet Air Temperature', linewidth=2)
            plt.plot(sim_times, outlet_air_temp, label=f'{zone} - Outlet Air Temperature', linewidth=2)
            plt.title(f'{zone} - Inlet and Outlet Air Temperature')
            plt.xlabel('Time')
            plt.ylabel('Air Temperature (°C)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_plots:
                os.makedirs('plots', exist_ok=True)
                plt.savefig(f'plots/{zone}_coil_inlet_and_outlet_air_temperature.png')

        # Calculate the energy consumption
        core_outlet_water_temperature = np.array(simulator.model.components["core_reheat_coil"].savedOutput["outletWaterTemperature"])
        north_outlet_water_temperature = np.array(simulator.model.components["north_reheat_coil"].savedOutput["outletWaterTemperature"])
        east_outlet_water_temperature = np.array(simulator.model.components["east_reheat_coil"].savedOutput["outletWaterTemperature"])
        south_outlet_water_temperature = np.array(simulator.model.components["south_reheat_coil"].savedOutput["outletWaterTemperature"])
        west_outlet_water_temperature = np.array(simulator.model.components["west_reheat_coil"].savedOutput["outletWaterTemperature"])

        inlet_water_temperature = np.array(simulator.model.components["reheat_coils_supply_water_temperature"].savedOutput["measuredValue"])
        #rooms water temp difference:
        core_room_water_temp_difference = abs(core_outlet_water_temperature - inlet_water_temperature)
        north_room_water_temp_difference = abs(north_outlet_water_temperature - inlet_water_temperature)
        east_room_water_temp_difference = abs(east_outlet_water_temperature - inlet_water_temperature)
        south_room_water_temp_difference = abs(south_outlet_water_temperature - inlet_water_temperature)
        west_room_water_temp_difference = abs(west_outlet_water_temperature - inlet_water_temperature)

        room_water_temp_difference_penalty = (core_room_water_temp_difference + north_room_water_temp_difference + east_room_water_temp_difference + south_room_water_temp_difference + west_room_water_temp_difference)

        print(f"Average room water temp difference: {np.average(room_water_temp_difference_penalty):.2f} °C")
        
        #plot the room water temp difference
        plt.figure(figsize=(12, 6))
        plt.plot(sim_times, room_water_temp_difference_penalty, label='Room Water Temp Difference', linewidth=2)
        plt.title('Room Water Temp Difference')
        plt.xlabel('Time')
        plt.ylabel('Water Temp Difference (°C)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/room_water_temp_difference.png')
        #plt.show()

        # AHU power consumption
        fan_power = np.array(simulator.model.components["vent_power_sensor"].savedOutput["measuredValue"])
        print(f"Average fan power: {np.average(fan_power):.1f} W")
        supply_cooling_coil_power = np.array(simulator.model.components["supply_cooling_coil"].savedOutput["Power"])
        print(f"Average supply cooling coil power: {np.average(supply_cooling_coil_power):.1f} W")
        supply_heating_coil_power = np.array(simulator.model.components["supply_heating_coil"].savedOutput["Power"])
        print(f"Average supply heating coil power: {np.average(supply_heating_coil_power):.1f} W")
        ahu_power_consumption_penalty = fan_power + supply_cooling_coil_power + supply_heating_coil_power
        print(f"Average total AHU power consumption: {np.average(ahu_power_consumption_penalty):.1f} W")
        outdoor_air_temperature = np.array(simulator.model.components["vent_outdoor_air_temp_sensor"].savedOutput["measuredValue"])

        # Sum coils power consumption element-wise to get total coils power at each time step
        total_coils_power = np.sum(coils_power_consumption, axis=0)
        total_hvac_power_consumption = fan_power + supply_cooling_coil_power + supply_heating_coil_power + total_coils_power
        print(f"Average total HVAC power consumption: {np.average(total_hvac_power_consumption):.1f} W")

        pre_heated_air_temperature = np.array(simulator.model.components["supply_heating_coil"].savedOutput["outletAirTemperature"])
        print(f"Average pre-heated air temperature: {np.average(pre_heated_air_temperature):.1f} °C")
        pre_cooled_air_temperature = np.array(simulator.model.components["supply_cooling_coil"].savedOutput["outletAirTemperature"])
        print(f"Average pre-cooled air temperature: {np.average(pre_cooled_air_temperature):.1f} °C")

        #plot the fan power consumption
        plt.figure(figsize=(12, 6))
        plt.plot(sim_times, fan_power, label='Fan Power Consumption', linewidth=2)
        plt.title('Fan Power Consumption')
        plt.xlabel('Time')
        plt.ylabel('AHU Power Consumption (W)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/fan_power.png')
        #plt.show()

        #plot the supply cooling and heating coil power consumption
        plt.figure(figsize=(12, 6))
        plt.plot(sim_times, supply_cooling_coil_power, label='Supply Cooling Coil Power', linewidth=2)
        plt.plot(sim_times, supply_heating_coil_power, label='Supply Heating Coil Power', linewidth=2)
        plt.title('Supply Coils Power Consumption')
        plt.xlabel('Time')
        plt.ylabel('Power Consumption (W)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/supply_coils_power.png')
        #plt.show()

        #Plot the pre-heated and pre-cooled air temperature
        plt.figure(figsize=(12, 6))
        plt.plot(sim_times, pre_heated_air_temperature, label='Pre-Heated Air Temperature', linewidth=2)
        plt.plot(sim_times, pre_cooled_air_temperature, label='Pre-Cooled Air Temperature', linewidth=2)
        plt.title('Pre-Heated and Pre-Cooled Air Temperature')
        plt.xlabel('Time')
        plt.ylabel('Air Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/pre_heated_and_pre_cooled_air_temperature.png')
        #plt.show()

        #Plot the outdoor air temperature
        plt.figure(figsize=(12, 6))
        plt.plot(sim_times, outdoor_air_temperature, label='Outdoor Air Temperature', linewidth=2)
        plt.title('Outdoor Air Temperature')
        plt.xlabel('Time')
        plt.ylabel('Outdoor Air Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/outdoor_air_temperature.png')
        #plt.show()

        #Plot some actions
        #Load the policy_input_output.json file and get the component ids and the signal keys for the actions
        with open(POLICY_CONFIG_PATH, 'r') as f:
                policy_config = json.load(f)

        #Get the component ids and the signal keys for the actions
        component_ids = []
        signal_keys = []
        for component_id, actions in policy_config['actions'].items():
            for action_name, action_config in actions.items():
                component_ids.append(component_id)
                signal_keys.append(action_config['signal_key'])

        #Get the actions from the simulator
        actions = [simulator.model.components[component_id].savedInput[signal_key] for component_id, signal_key in zip(component_ids, signal_keys)]

        #Plot each action separately
        for action, component_id, signal_key in zip(actions, component_ids, signal_keys):
            plt.figure(figsize=(12, 6))
            plt.plot(sim_times, action, label=f'{component_id} - {signal_key}', linewidth=2)
            plt.title(f'Action: {component_id} - {signal_key}')
            plt.xlabel('Time')
            plt.ylabel('Action Value')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            # Format Y-axis to show actual numbers instead of scientific notation
            plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
            plt.tight_layout()
            if save_plots:
                os.makedirs('plots', exist_ok=True)
                plt.savefig(f'plots/action_{component_id}_{signal_key}.png')
            #plt.show()

        #Calculate and print BOPTESTs KPIs

        total_floor_area = 1662.664 #m2
        step_size_seconds = 600 #s

        #ener_tot - BOPTEST style calculation with trapezoidal integration
        # Create time array for integration (same length as power data)
        time_array = np.arange(0, len(total_hvac_power_consumption)) * step_size_seconds
        
        # Calculate energy for individual sources (BOPTEST style)
        fan_energy_kwh, fan_energy_kwh_m2 = calculate_energy_boptest_style(fan_power, time_array, total_floor_area)
        cooling_coil_energy_kwh, cooling_coil_energy_kwh_m2 = calculate_energy_boptest_style(supply_cooling_coil_power, time_array, total_floor_area)
        heating_coil_energy_kwh, heating_coil_energy_kwh_m2 = calculate_energy_boptest_style(supply_heating_coil_power, time_array, total_floor_area)
        reheat_coils_energy_kwh, reheat_coils_energy_kwh_m2 = calculate_energy_boptest_style(total_coils_power, time_array, total_floor_area)
        
        # Calculate total energy using BOPTEST-style trapezoidal integration
        energ_tot_kwh, energ_tot_kwh_m2 = calculate_energy_boptest_style(
            total_hvac_power_consumption, time_array, total_floor_area
        )
        
        
        print(f"Total energy consumption: {energ_tot_kwh:.3f} kWh")
        print(f"  Fan energy: {fan_energy_kwh_m2:.3f} kWh/m2 ({fan_energy_kwh:.3f} kWh)")
        print(f"  Cooling coil energy: {cooling_coil_energy_kwh_m2:.1f} kWh/m2 ({cooling_coil_energy_kwh:.1f} kWh)")
        print(f"  Heating coil energy: {heating_coil_energy_kwh_m2:.1f} kWh/m2 ({heating_coil_energy_kwh:.1f} kWh)")
        print(f"  Reheat coils energy: {reheat_coils_energy_kwh_m2:.1f} kWh/m2 ({reheat_coils_energy_kwh:.1f} kWh)")
        
        print(f"ener_tot: {energ_tot_kwh_m2:.3f} kWh/m2")
        #tdis_tot

        zone_temperatures = []
        zone_temperatures_low = []
        zone_temperatures_high = []
        zones = ['core', 'north', 'east', 'south', 'west']
        for zone in zones:
            zone_temperatures.append(simulator.model.components[f"{zone}_indoor_temp_sensor"].savedOutput["measuredValue"])
            zone_temperatures_low.append(simulator.model.components[f"{zone}_temperature_heating_setpoint"].savedOutput["scheduleValue"])
            zone_temperatures_high.append(simulator.model.components[f"{zone}_temperature_cooling_setpoint"].savedOutput["scheduleValue"])
        
        # Convert to numpy arrays and transpose to (Tsteps, Nz) format
        zone_temperatures = np.array(zone_temperatures).T  # Shape: (Tsteps, 5)
        zone_temperatures_low = np.array(zone_temperatures_low).T  # Shape: (Tsteps, 5)
        zone_temperatures_high = np.array(zone_temperatures_high).T  # Shape: (Tsteps, 5)
        
        tdis_tot, per_zone_tdis = thermal_discomfort(zone_temperatures, zone_temperatures_low, zone_temperatures_high, step_size_seconds)
        
        print(f"Total thermal discomfort per zone: {per_zone_tdis} K·h")
        print(f"tdis_tot: {tdis_tot:.1f} K·h/zone")
        
        # Print per-zone breakdown
        zones = ['core', 'north', 'east', 'south', 'west']
        for i, zone in enumerate(zones):
            print(f"  {zone}: {per_zone_tdis[i]:.1f} K·h")


        #idis_tot
        CO2_measurements = []
        CO2_setpoints = []
        for zone in zones:
            CO2_measurements.append(simulator.model.components[f"{zone}_co2_sensor"].savedOutput["measuredValue"])
            CO2_setpoints.append(simulator.model.components[f"{zone}_co2_setpoint"].savedOutput["scheduleValue"])
        
        CO2_measurements = np.array(CO2_measurements).T
        CO2_setpoints = np.array(CO2_setpoints).T

        iad_tot, per_zone_iad = iaq_violation_building(CO2_measurements, CO2_setpoints, step_size_seconds, total_floor_area)
        print(f"Total IAQ violation per zone: {per_zone_iad} ppm·h")
        
        
        # Print per-zone breakdown
        for i, zone in enumerate(zones):
            print(f"  {zone}: {per_zone_iad[i]:.1f} ppm·h")

        print(f"iad_tot: {iad_tot:.1f} ppm·h/zone")


if __name__ == "__main__":
        model = load_model_and_params()
        get_baseline(model)