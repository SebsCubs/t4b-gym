# RL_control Repository — Architecture and Main Components

This document describes the architectural characteristics of the RL_control repository, its main packages, and the principal classes that implement building control and reinforcement learning (RL) training.

---

## 1. Repository Overview

<p align="center">
  <img src="./RL_control_with_T4B.png" alt="RL_control Architecture with Twin4Build" style="width:100%;"/>
</p>


**RL_control** is a reinforcement learning environment for **building HVAC control**, built on top of the **Twin4Build** (T4B) framework. It provides:

- A **Gymnasium-compatible environment** for training RL agents (e.g. PPO, A2C, BC).
- A **Twin4Build-based building model** (multi-zone office with AHU and VAVs) used as the simulation backend.
- **BOPTEST-oriented tooling** for testing controllers against a REST API (e.g. Dockerized FMU).
- **Use-case scripts** for training, evaluation, expert data, and behavioral cloning.

The design separates: (1) the **physics/simulation** (Twin4Build model), (2) the **RL interface** (Gym env + simulator), and (3) **applications** (use cases, BOPTEST interface).

---

## 2. Top-Level Structure

```
RL_control/
├── boptest_model/          # Building model definition and BOPTEST integration
├── t4b_gym/               # Gymnasium environment wrapping Twin4Build
├── use_case/              # Training, evaluation, and analysis scripts
└── docs/                  # Documentation (e.g. this file)
```

- **boptest_model**: Building model (envelope, VAVs, AHU), parameter loading, and BOPTEST handler (REST client, controllers, KPIs).
- **t4b_gym**: Core RL interface — `GymSimulator`, `T4BGymEnv`, and observation/action wrappers.
- **use_case**: Entry points for PPO/BC training, model evaluation, expert trajectories, and log analysis.

---

## 3. Main Architectural Layers

### 3.1 Simulation Layer (Twin4Build)

- **Twin4Build** (`tb`) provides the building simulation: `tb.Model`, `tb.Simulator`, and component systems (rooms, sensors, AHU, VAVs, etc.).
- The repository does **not** define the Twin4Build library; it **uses** it and extends it via:
  - **Model construction**: `get_model()` + `fcn()` in `rooms_and_ahu_model.py` (and variants in `only_ahu_model.py`, `5_rooms_model.py`, etc.).
  - **Simulation driver for RL**: `GymSimulator`, which extends `tb.Simulator` to accept control inputs and expose observations at each step.

### 3.2 RL Environment Layer (t4b_gym)

- **GymSimulator** (extends `tb.Simulator`): Single-step simulation with:
  - Registered **control inputs** (component_id + input name).
  - Registered **observation outputs** (component_id + output name).
  - Optional **baseline_mode** where control inputs are not overridden (for baseline controller comparison).
- **T4BGymEnv** (extends `gym.Env`): Standard Gymnasium API:
  - `reset()` → initial observation and info.
  - `step(action)` → observation, reward, terminated, truncated, info.
  - Action/observation spaces built from an **IO config JSON** (actions, observations, optional time_embeddings, forecasts).
- **Wrappers**: `NormalizedObservationWrapper`, `NormalizedActionWrapper` for scaling observations and actions (e.g. to [-1, 1]).

### 3.3 Building Model Layer (boptest_model)

- **Model construction**: A single **composite function** `fcn(self)` (bound to the model) calls:
  - **envelope_fcn(self)**: Building spaces (core, north, south, east, west), outdoor environment, occupancy schedules, adjacency, and zone sensors (temperature, CO2).
  - **vavs_fcn(self)**: Per-zone VAV reheat (setpoints, controllers, reheat coils, dampers, sensors).
  - **ahu_fcn(self)**: AHU (supply/return fans, heating/cooling coils, mixing, supply temperature and airflow sensors).
- **get_model(id, fcn_)**: Creates a `tb.Model`, then `model.load(fcn=fcn_, ...)` to build the graph. The default `fcn` is the full rooms + AHU composition.
- **Parameter loading**: `load_model_parameters()` / `load_model_and_params()` load estimated parameters from pickle files (envelope, VAVs, AHU, etc.) and apply them to the model; optional manual overrides (e.g. `Q_occ_gain`, `C_boundary`) are applied in code.

### 3.4 BOPTEST Interface Layer (boptest_handler)

- **interface**: REST client to a BOPTEST service (e.g. Docker container). `control_test()` / `control_test_with_points()` implement the loop: initialize → advance with `u` → get `y` → compute next `u` via a controller.
- **controllers**: Pluggable controllers loaded by name (e.g. `controllers.baseline`, `controllers.pid`). **Controller** is a thin wrapper that uses `importlib` to load a module and expose `initialize`, `compute_control`, and optionally `update_forecasts` / `get_forecast_parameters`.
- **custom_kpi**: **CustomKPI** loads KPI logic from JSON (file, class, data_points) and processes streaming measurement data for custom metrics.

### 3.5 Use-Case / Application Layer (use_case)

- **multizone_simple_air_RL_control.py**: Defines a custom reward (comfort + energy), builds `T4BGymEnv` (with optional custom reward subclass), wraps with normalizers, and runs PPO training or testing.
- **multizone_simple_air_RL_BC_control.py**: Behavioral cloning and BC-tuned PPO; custom callbacks and action wrappers (e.g. robust normalization, clipping).
- **run_ppo_with_autoencoder.py** / **run_ppo_without_autoencoder.py**: PPO entry points with/without autoencoder observation compression.
- **pretrain_with_expert.py**: Expert data, autoencoder, and policies (e.g. `AutoencoderMlpPolicy`) for pretraining.
- **model_eval.py**: Evaluation helpers (e.g. `test_model(env, model)`, `plot_results(simulator, rewards, ...)`).
- **record_expert_trajectories.py** / **expert_trajectories/**: Recording and preprocessing expert trajectories for imitation learning.

---

## 4. Principal Classes and Their Roles

### 4.1 t4b_gym (Gymnasium environment)

| Class | File | Role |
|-------|------|------|
| **GymSimulator** | `t4b_gym/t4b_gym_env.py` | Extends `tb.Simulator`. Holds `control_inputs` and `observation_outputs`; in `_do_component_timestep` applies control inputs (unless `baseline_mode`), runs component step, records observed outputs. Implements `initialize_simulation`, `step_simulation`, `get_observations`, `add_control_input`, `add_observation_output`, and `_populate_from_json` for IO config. |
| **T4BGymEnv** | `t4b_gym/t4b_gym_env.py` | Gymnasium environment. Owns a `GymSimulator`, builds `action_space` and `observation_space` from IO config (and optional time_embeddings, forecasts). `reset()` sets episode window (fixed or random), initializes simulator, returns `_get_obs()`. `step(action)` applies action (or baseline), gets observations, reward via `get_reward()` (override in subclasses). Supports `forecast_horizon`, `excluding_periods`, `random_start`. |
| **NormalizedObservationWrapper** | `t4b_gym/t4b_gym_env.py` | Gymnasium `ObservationWrapper` that scales observations to [-1, 1] using bounds from the inner env’s observation space. |
| **NormalizedActionWrapper** | `t4b_gym/t4b_gym_env.py` | Gymnasium `ActionWrapper` that maps [-1, 1] actions to the inner env’s action bounds. |

### 4.2 boptest_model (Building model)

| Concept | File | Role |
|--------|------|------|
| **fcn / envelope_fcn / vavs_fcn / ahu_fcn** | `boptest_model/rooms_and_ahu_model.py` | Composite model builder: `fcn(self)` calls `envelope_fcn`, `vavs_fcn`, `ahu_fcn` to add components and connections to the model. Envelope: spaces, outdoor env, occupancy, sensors. VAVs: setpoints, VAV reheat controllers, coils, dampers. AHU: fans, coils, mixing, sensors. |
| **get_model(id, fcn_)** | `boptest_model/rooms_and_ahu_model.py` | Creates `tb.Model(id=id)`, then `model.load(fcn=fcn_, ...)` to build the system. Default `fcn` is the full rooms+AHU composition. |
| **load_model_parameters(model_id)** / **load_model_and_params(model_id)** | `boptest_model/rooms_and_ahu_model.py` | Load a model via `get_model(id=model_id)` and apply estimated parameters from pickle files (envelope, VAVs, AHU, etc.) and any hard-coded overrides. |
| **run(model)** | `boptest_model/rooms_and_ahu_model.py` | Runs a standard Twin4Build `Simulator` (not GymSimulator) over a fixed period; used for offline simulation and parameter evaluation. |
| **parameter_evaluation(data_points, save_plots)** | `boptest_model/rooms_and_ahu_model.py` | Loads model with parameters, runs simulation, compares selected outputs to CSV data and optionally saves comparison plots. |

### 4.3 boptest_handler (BOPTEST and controllers)

| Class | File | Role |
|-------|------|------|
| **Controller** | `boptest_handler/controllers/controller.py` | Generic controller loader. Takes a module path (e.g. `controllers.baseline`) and `use_forecast`; uses `importlib` to load the module and expose `initialize`, `compute_control`, and optionally `update_forecasts`, `get_forecast_parameters`. |
| **CustomKPI** | `boptest_handler/custom_kpi/custom_kpi_calculator.py` | Loads KPI definition from config (name, kpi_file, kpi_class, data_points), instantiates the KPI class, and provides `processing_data(data)` and `calculation()` for streaming custom KPI computation. |
| **BoptestDebugger** | `boptest_handler/service_debugger.py` | Debugging support for the BOPTEST service. |
| Concrete controllers | `boptest_handler/controllers/baseline.py`, `pid.py`, `pidTwoZones.py`, `sup.py` | Implement `initialize()`, `compute_control(y, forecasts)` (and optionally forecast hooks) for use with the REST interface. |

### 4.4 use_case (Training and evaluation)

| Class / function | File | Role |
|------------------|------|------|
| **T4BGymEnvCustomReward** (example) | `use_case/multizone_simple_air_RL_control.py` | Subclass of `T4BGymEnv` that implements `get_reward()` combining zone temperature violations, reheat coil power, and AHU power. |
| **test_model(env, model)** | `use_case/model_eval.py` | Runs a trained RL model in the environment and returns observation and reward trajectories. |
| **plot_results(simulator, rewards, ...)** | `use_case/model_eval.py` | Plots simulation results and optionally reward; can save figures. |
| **TrainingLogAnalyzer** | `use_case/analyze_training_logs.py` | Analyzes training logs (e.g. TensorBoard / CSV). |
| **AutoencoderWrapper**, **AutoencoderMlpPolicy** | `use_case/pretrain_with_expert.py` | Observation wrapper and policy that use an autoencoder for compressed observations; used in pretraining with expert data. |
| BC-related callbacks and wrappers | `use_case/multizone_simple_air_RL_BC_control.py` | e.g. `AdaptKLtoBC`, `ActionBoundsMonitorCallback`, `RobustNormalizedActionWrapper`, `ActionClippingWrapper` for BC and safe action handling. |

---

## 5. Data and Configuration Flow

### 5.1 IO configuration (actions and observations)

- A **JSON IO config file** (e.g. `policy_input_output.json`, `policy_input_output_co2sets.json`) defines:
  - **actions**: For each component and signal, `signal_key` (input name), `min`, `max`, and optional description.
  - **observations**: Same for outputs.
  - **time_embeddings** (optional): e.g. `time_of_day`, `day_of_week`, `month_of_year` — added as sin/cos features.
  - **forecasts** (optional): e.g. outdoor temperature, irradiation; and/or schedule-based forecasts with a horizon.
- **GymSimulator._populate_from_json()** registers control inputs and observation outputs and stores bounds.
- **T4BGymEnv** uses these bounds to build `action_space` and `observation_space` (Box), and `_get_obs()` assembles the observation vector (model outputs + time embeddings + forecast series).

### 5.2 Parameter and model data

- **Building model parameters**: Stored in pickle files under `boptest_model/generated_files/models/.../estimation_results/...`. Loaded by `load_model_parameters()` / `load_model_and_params()`.
- **Reference data for validation**: CSV paths are listed in `model_output_points` in `rooms_and_ahu_model.py`; used by `parameter_evaluation()` to compare simulation vs. real data and generate plots (e.g. in `plots/`, `plots_cool/`, `plots_mix/`).

---

## 6. Design Patterns and Characteristics

- **Separation of model and env**: The Twin4Build model is built once (e.g. via `load_model_and_params()`); the same model instance is passed into `T4BGymEnv` and thus into `GymSimulator`. Control and observation sets are defined by the IO config, not by the model file.
- **Composition of model**: The building model is composed from three sub-graphs (envelope, VAVs, AHU) via a single `fcn(self)` passed to `model.load()`, keeping topology in one place.
- **Pluggable controllers**: BOPTEST controllers are specified by module path; the **Controller** class delegates to the loaded module’s `initialize` and `compute_control` (and optionally forecast) functions.
- **Baseline mode**: `GymSimulator.baseline_mode` and `T4BGymEnv(baseline_mode=True)` allow running the same simulation with the built-in schedule/controller logic without overriding with RL actions, useful for comparison.
- **Reward customization**: Reward is not fixed in the core env; subclasses of `T4BGymEnv` override `get_reward(observations, action)` (e.g. comfort + energy in multizone_simple_air_RL_control).
- **Wrappers**: Observation and action normalization are implemented as Gymnasium wrappers so the same env can be used raw or normalized for different algorithms.

---

## 7. Entry Points (Summary)

- **RL training (T4B, no BOPTEST)**: Use `load_model_and_params()` and `T4BGymEnv` with an IO config; wrap with normalizers; train with PPO/A2C/BC from `use_case` scripts (e.g. `multizone_simple_air_RL_control.py`, `run_ppo_with_autoencoder.py`).
- **BOPTEST testing**: Run the BOPTEST service (e.g. Docker), then use `control_test()` or `control_test_with_points()` from `boptest_handler.interface` with a controller module (e.g. `controllers.baseline`) and optional custom KPI config.
- **Model validation**: Use `parameter_evaluation(model_output_points, save_plots=True)` in `rooms_and_ahu_model.py` to compare simulation outputs to CSV data and generate plots.
- **Offline simulation**: Use `run(model)` in `rooms_and_ahu_model.py` with a pre-loaded model for non-RL simulation.

This architecture keeps the Twin4Build model and BOPTEST interface reusable while providing a single, consistent Gymnasium API for RL development and evaluation.
