# T4B Gym Library Refactor Plan

## Goal

Turn the current repository into a reusable library for exposing Twin4Build models as Gymnasium environments, while preserving the current multizone HVAC workflow as an example use case rather than the architectural center of the codebase.

The intended outcome is:

- `t4b_gym` becomes a small, clean library.
- `boptest_model` remains an example Twin4Build model package.
- `use_case` becomes example training and evaluation scripts built on top of the library API.
- New Twin4Build models can be used without editing the environment internals.

## Current State Summary

The repository already contains a strong prototype:

- `GymSimulator` provides step-by-step Twin4Build simulation control.
- `T4BGymEnv` provides a Gymnasium-compatible environment.
- JSON configuration defines actions and observations.
- wrappers normalize actions and observations.
- the multizone office model demonstrates an end-to-end RL workflow.

The main limitations are:

- reward logic is embedded in custom environment subclasses inside `use_case/`
- forecast logic in the environment assumes specific model structures and names
- component IDs from the multizone model leak into the RL logic
- configuration is only loosely validated
- library packaging is incomplete
- some model code still uses absolute paths and example-specific assumptions

## Target Architecture

The refactor should produce four clean layers:

1. Core environment layer
   - generic Twin4Build stepping and Gymnasium API
2. Configuration and adapter layer
   - validated environment config
   - model adapters for reading and writing signals
3. Reward and forecast layer
   - pluggable reward terms
   - pluggable forecast providers
4. Example and application layer
   - multizone HVAC example
   - PPO/A2C/BC training scripts

## Proposed Package Structure

Suggested target layout:

```text
RL_control/
├── t4b_gym/
│   ├── __init__.py
│   ├── env.py
│   ├── simulator.py
│   ├── wrappers.py
│   ├── config.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── direct_component.py
│   │   ├── schedule.py
│   │   └── dataframe.py
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── composite.py
│   │   ├── comfort.py
│   │   ├── air_quality.py
│   │   └── energy.py
│   ├── forecasts/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── schedule.py
│   │   └── dataframe.py
│   └── testing/
├── boptest_model/
├── use_case/
└── docs/
```

## Refactor Phases

## Phase 1: Stabilize the Current Core

Objective: preserve current behavior while making the code safe to evolve.

Tasks:

- split `t4b_gym/t4b_gym_env.py` into focused modules:
  - `simulator.py`
  - `env.py`
  - `wrappers.py`
- add a clean public API in `t4b_gym/__init__.py`
- make observation ordering deterministic and explicit
- make action ordering deterministic and explicit
- clean up the `step_simulation()` contract so the return type is unambiguous
- ensure `reset()` and `step()` remain Gymnasium-compliant
- keep a compatibility import path temporarily if needed

Definition of done:

- the old single-file implementation is split without changing current user-facing behavior
- tests still pass for the current simple test model

## Phase 2: Introduce a Validated Configuration Layer

Objective: replace loosely structured JSON usage with a library-level config model.

Tasks:

- create config models for:
  - `ActionSpec`
  - `ObservationSpec`
  - `ForecastSpec`
  - `TimeEmbeddingSpec`
  - `EpisodeConfig`
  - `EnvConfig`
- support loading from the current JSON format first
- validate:
  - component references
  - input/output names
  - lower and upper bounds
  - duplicate signal keys
  - forecast horizon consistency
- keep backward compatibility with the current `policy_input_output*.json` files during migration

Definition of done:

- the environment consumes a validated config object rather than raw JSON dictionaries internally
- invalid configs fail early with clear error messages

## Phase 3: Introduce a Model Adapter Abstraction

Objective: remove model-specific semantics from the core environment.

Tasks:

- define a `ModelAdapter` interface with methods such as:
  - `validate(model, config)`
  - `write_action(model, action_spec, value)`
  - `read_observation(model, observation_spec)`
  - `get_forecast(model, forecast_spec, current_step, horizon, step_size)`
- implement `DirectComponentAdapter`
  - supports the current `component_id + signal_name` access pattern
- implement adapter support for schedule-backed values
- implement adapter support for dataframe-backed values
- update the environment so it talks only to the adapter, not directly to model internals

Definition of done:

- the environment no longer hardcodes assumptions like a specific outdoor component name
- a new Twin4Build model can be integrated by supplying config plus an adapter

## Phase 4: Extract Reward Logic into Reusable Components

Objective: stop requiring custom environment subclasses for each control task.

Tasks:

- define a `RewardFn` interface
- implement a `CompositeReward`
- create reusable reward terms such as:
  - `TemperatureComfortPenalty`
  - `CO2Penalty`
  - `EnergyPenalty`
  - `ActionPenalty`
  - `ActionChangePenalty`
  - `ConstraintViolationPenalty`
- allow reward terms to read named signals or derived metrics from the adapter layer
- migrate current reward logic from `use_case/` into reusable reward components

Definition of done:

- users can compose a reward without subclassing `T4BGymEnv`
- the current multizone HVAC reward can be reproduced from reusable pieces

## Phase 5: Generalize Forecast Handling

Objective: remove hardcoded forecast extraction logic from the environment.

Tasks:

- define a `ForecastProvider` abstraction
- implement:
  - `ScheduleForecastProvider`
  - `DataFrameForecastProvider`
  - optional `StaticForecastProvider`
- make each forecast entry in config declare how it should be resolved
- move weather, occupancy, and schedule forecast logic out of `_get_obs()`
- support future extension to forecast APIs or custom scenario generators

Definition of done:

- `_get_obs()` does not contain model-specific forecast logic
- forecasts are provider-driven and reusable

## Phase 6: Define a Small Public Library API

Objective: make the library easy to understand and use.

Target API style:

```python
from t4b_gym import Twin4BuildEnv, EnvConfig
from t4b_gym.adapters import DirectComponentAdapter
from t4b_gym.rewards import CompositeReward, EnergyPenalty

config = EnvConfig.from_json("env.json")
adapter = DirectComponentAdapter()
reward_fn = CompositeReward([...])

env = Twin4BuildEnv(
    model=model,
    config=config,
    adapter=adapter,
    reward_fn=reward_fn,
)
```

Tasks:

- expose stable public imports from `t4b_gym/__init__.py`
- hide internal helper classes where possible
- document one recommended environment construction pattern
- keep the public API smaller than the internal implementation surface

Definition of done:

- README examples use the new API
- users can understand environment construction in a few lines

## Phase 7: Move Existing Project Logic into Examples

Objective: make the current multizone workflow a reference implementation rather than core architecture.

Tasks:

- migrate:
  - `use_case/multizone_simple_air_RL_control.py`
  - `use_case/multizone_simple_air_RL_A2C.py`
  - `use_case/multizone_simple_air_RL_BC_control.py`
- keep them as example scripts built on the library API
- replace custom reward subclasses with reward composition
- keep current JSON configs during transition, then optionally migrate to richer config files later

Suggested example layout:

```text
examples/
├── multizone_hvac/
│   ├── env_config.json
│   ├── train_ppo.py
│   ├── train_a2c.py
│   └── train_bc.py
└── simple_co2_control/
    └── train_ppo.py
```

Definition of done:

- the multizone office use case runs without depending on environment subclassing
- the simple test model also has a minimal example

## Phase 8: Package the Project Properly

Objective: make `t4b_gym` installable and library-like.

Tasks:

- add `pyproject.toml`
- define package metadata
- define runtime and optional development dependencies
- make `pip install -e .` work
- ensure imports work from outside the repository root
- keep Twin4Build as a documented dependency or optional dependency depending on packaging constraints

Definition of done:

- the library can be installed in editable mode
- imports do not depend on manual `sys.path` manipulation

## Phase 9: Expand Tests Around the New Abstractions

Objective: verify the refactor without relying only on the multizone example.

Tasks:

- keep the current simple test model as a lightweight integration test
- split tests by responsibility:
  - config validation tests
  - adapter tests
  - reward composition tests
  - forecast provider tests
  - environment stepping tests
  - wrapper normalization tests
- add deterministic ordering tests for actions and observations
- test `random_start` and `excluding_periods`

Definition of done:

- the core package can be validated without running the full multizone workflow
- tests cover the new extension points directly

## Phase 10: Clean Up Example-Specific Portability Issues

Objective: remove assumptions that make the repository hard to reuse across machines and models.

Tasks:

- remove absolute file paths from model code
- isolate calibrated model loading from the generic environment package
- treat `boptest_model/` as an example model family, not as a required part of the env design
- document what a new model must provide to work with the library

Definition of done:

- the generic environment package has no hidden dependency on the multizone office model
- example model code is clearly separated from reusable library code

## Recommended Implementation Order

To reduce risk, implement the refactor in this order:

1. Split `t4b_gym_env.py` into `env.py`, `simulator.py`, and `wrappers.py`
2. Add config objects and validation
3. Make action and observation ordering deterministic
4. Introduce the adapter abstraction
5. Introduce forecast providers
6. Introduce reward abstractions and reusable reward terms
7. Migrate multizone use cases to the new API
8. Add packaging and improve documentation
9. Expand automated tests
10. Clean up portability issues in example model code

## Definition of Success

The refactor is successful when all of the following are true:

- the core environment contains no multizone-specific component assumptions
- a new Twin4Build model can be used through config plus an adapter
- rewards are injectable or composable rather than implemented only by subclassing the env
- forecasts are provider-based rather than hardcoded in the environment
- the current multizone HVAC workflow still works as an example
- the package can be installed and imported cleanly

## Near-Term Deliverables

If the work is done incrementally, the most practical first milestone is:

- split the environment file
- add config objects
- add deterministic signal ordering
- preserve the current JSON format
- keep the current multizone PPO example working

That first milestone would already make the codebase much easier to maintain, and it would create the right foundation for the more important abstraction work that follows.
