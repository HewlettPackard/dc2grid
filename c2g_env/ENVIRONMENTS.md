# C2G-Bench — Environment & Simulator Reference

This document is the authoritative reference for C2G-Bench.

**Sections 2–3** describe the two Gymnasium **environments** (`C2GFastEnv`, `C2GMacroEnv`) — their action/observation spaces, reward functions, and Python API.  
**Sections 4–10** describe the seven physics **simulators** that both environments share — governing equations, parameters, and numerical methods.  
Section 11 is the configuration reference; Section 12 shows how to add a new scenario.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [C2GFastEnv — Low-Level Hardware Controller](#2-c2gfastenv)
3. [C2GMacroEnv — High-Level Market Orchestrator](#3-c2gmacroenv)
4. [Simulator: Thermal Twin](#4-thermal-twin)
5. [Simulator: BESS (Battery Energy Storage)](#5-bess)
6. [Simulator: Electrical Chain](#6-electrical-chain)
7. [Simulator: Macro-Grid Signal](#7-macro-grid-signal)
8. [Simulator: Weather](#8-weather)
9. [Simulator: Workload Orchestrator](#9-workload-orchestrator)
10. [Configuration Reference](#10-configuration-reference)
11. [Adding a New Scenario](#11-adding-a-new-scenario)

---

## 1. Architecture Overview

C2G-Bench exposes two Gymnasium environments that operate at different timescales:

```
C2GMacroEnv  (15-minute ticks)
│   Observation : 17-D aggregated from 180 sub-steps
│   Action      : 2-D  [commit_norm, bess_target]
│
│   calls inner_action_fn() 180 times per macro-step
│
└── C2GFastEnv  (5-second ticks)
        Observation : 17-D normalised
        Action      : 4-D  [throttle, pump_A, hvac, bess]
        │
        └── Six physics engines (all unconditionally stable):
                ThermalTwin · BESSModel · DatacenterElectrical
                MacroGridSignal · WeatherLoader
                WorkloadOrchestrator
```

Both environments share the same six physics engines and the same `config.yaml`.

> **Formal MDP specification:** For the full two-level MDP tuple definition — state/action spaces, discount factors, terminal conditions, and the Semi-MDP framing — see [README.md §5.0](../README.md#50-formal-mdp-specification).

---

## 2. C2GFastEnv

**File:** `c2g_env/env_low_level.py`  
**Class:** `C2GFastEnv`  
**Gymnasium ID:** `c2g-fast-v0`  
**Timestep:** 5 seconds  
**Episode length:** 17,280 ticks = 24 hours

### 2.1 Action Space

`Box(low, high, shape=(4,), dtype=float32)`

| Index | Name | Range | Physical meaning |
|-------|------|-------|-----------------|
| 0 | `throttle_batch` | [0, 1] | DVFS throttle on schedulable batch jobs. 0 = full speed, 1 = fully throttled (zero batch power). GenAI/DLRM rigid loads are **not** affected. |
| 1 | `pump_speed_A` | [0, 1] | CDU circulating pump speed (Zone A). Minimum clamped to 0.15 to maintain minimum server-blade flow. |
| 2 | `hvac_effort` | [0, 1] | Zone B HVAC fan + chiller effort. 1 = full HVAC power (50 MW). |
| 3 | `bess_dispatch` | [-1, 1] | BESS dispatch: −1 = full charge (50 MW into battery), +1 = full discharge (50 MW to grid). |

### 2.2 Observation Space

`Box(low, high, shape=(17,), dtype=float32)`

| Index | Name | Range | Formula | Description |
|-------|------|-------|---------|-------------|
| 0 | `temp_A_norm` | [0, 2] | $T_A / T_\text{safe}$ | Zone A temperature normalised by silicon limit |
| 1 | `temp_B_norm` | [0, 2] | $T_B / T_\text{safe}$ | Zone B temperature normalised by silicon limit |
| 2 | `bess_soc` | [0, 1] | $\text{SOC}$ | Battery state of charge |
| 3 | `p_base_norm` | [0, 1] | $P_\text{base} / P_\text{norm}$ | Rigid IT load fraction (GenAI + DLRM) |
| 4 | `p_flex_nom_norm` | [0, 1] | $P_\text{flex,nom} / P_\text{norm}$ | Schedulable batch at full throttle |
| 5 | `p_facility_norm` | [0, 2] | $P_\text{facility} / P_\text{norm}$ | Total facility power draw |
| 6 | `regd_signal` | [-1, 1] | AR(1) process | Normalised RegD regulation signal |
| 7 | `lmp_norm` | [0, 1] | $\text{LMP} / \text{LMP}_\text{max}$ | Localised marginal price |
| 8 | `grid_load_norm` | [0, 1] | $L / L_\text{max}$ | Regional grid load stress indicator |
| 9 | `is_spike` | {0, 1} | binary | GenAI serving spike flag |
| 10 | `prev_throttle` | [0, 1] | $a_{t-1}[0]$ | Previous DVFS throttle (action memory) |
| 11 | `prev_pump_speed` | [0, 1] | $a_{t-1}[1]$ | Previous pump speed (action memory) |
| 12 | `pue_norm` | [0, 2] | $\text{PUE} / 2$ | Power Usage Effectiveness normalised |
| 13 | `T_amb_norm` | [0, 1] | $T_\text{amb} / 50$ | Ambient temperature |
| 14 | `freq_dev_norm` | [-1, 1] | $\Delta f / 0.5$ | Grid frequency deviation (swing equation) |
| 15 | `v_pcc_pu` | [0, 1.1] | Thévenin | PCC voltage in per-unit |
| 16 | `backlog_norm` | [0, 1] | $q / q_\text{max}$ | Workload backlog fraction (FIFO queue) |

### 2.3 Reward Function

Seven-term scalar reward at every 5-second tick. See [README.md §5.3](../README.md#53-the-neurips-evaluation-metric-the-tracking-reward) for the full term-by-term breakdown, coefficient rationale, and lever hierarchy.

```math
\mathcal{R}_t = \alpha \cdot u_\text{thr} - \beta \cdot \frac{|\Delta P_\text{demand} - \Delta P_\text{actual}|}{P_\text{norm}} - \gamma \cdot (T - T_\text{warn})^{+} - \delta_\text{soc} \cdot \mathbf{1}\text{soc} - \delta_f \cdot (|\Delta f| - 0.2)^{+} - \delta_v \cdot \varepsilon_v - \delta_q \cdot \frac{Q_\text{backlog}}{P_\text{flex,max}}
```
where:
- $(x)^{+} = \max(0, x)$  
- $u_\text{thr} = 1 - \text{throttle}$ (fraction of batch workload served)  
- $\Delta P_\text{actual} = P_\text{flex,served} + P_\text{BESS,actual}$  
- $\varepsilon_v = (0.95 - v_\text{pcc})^{+} + (v_\text{pcc} - 1.05)^{+}$  
- $\mathbf{1}\text{soc} = 1$ if $\text{SOC} < \text{SOC}_\text{min} + 0.02$, else 0  
- $Q_\text{backlog}$ — FIFO queue depth [kW]; $P_\text{flex,max}$ — peak flexible IT capacity [kW]

**Coefficients** (source of truth: `c2g_env/config.yaml`):

| Symbol | Config key | Default | Meaning |
|--------|-----------|---------|---------|
| $\alpha$ | `reward.alpha` | 1.0 | Throughput incentive |
| $\beta$ | `reward.beta` | 2.0 | Tracking error penalty |
| $\gamma$ | `reward.gamma_thermal` | 5.0 | Thermal penalty (per °C above $T_\text{warn}$) |
| $\delta_\text{soc}$ | `reward.soc_penalty` | 0.5 | BESS near-empty penalty (per tick) |
| $\delta_f$ | `reward.delta_freq_penalty` | 2.0 | Frequency penalty (per Hz beyond dead-band) |
| $\delta_v$ | `reward.delta_volt_penalty` | 5.0 | Voltage penalty (per pu outside [0.95, 1.05]) |
| $\delta_q$ | `reward.sla_backlog_penalty` | 2.0 | SLA backlog penalty (per unit normalised queue depth) |

### 2.4 Termination Conditions

The episode ends **immediately** (not truncated) on any hard fault:

| Fault | Condition | Physical meaning |
|-------|-----------|-----------------|
| Thermal | $T_A > 35°C$ or $T_B > 35°C$ | Silicon over-temperature → server shutdown |
| Frequency | $|\Delta f| > 0.5$ Hz | UFLS / over-frequency generator trip relay |
| Voltage | $v_\text{pcc} < 0.90$ pu | Under-voltage relay at PCC |

The episode **truncates** (without fault) after 17,280 ticks (24 hours).

### 2.5 Usage

```python
from c2g_env import C2GFastEnv

env = C2GFastEnv(scenario="default", seed=42)
obs, info = env.reset()

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

**`info` dictionary keys** (returned every step):

| Key | Type | Description |
|-----|------|-------------|
| `tick` | int | Current episode tick |
| `temp_A` | float | Zone A temperature [°C] |
| `temp_B` | float | Zone B temperature [°C] |
| `bess_soc` | float | Battery SOC [0, 1] |
| `p_facility_mw` | float | Total facility power [MW] |
| `p_total_it_mw` | float | Total IT power [MW] |
| `pue` | float | Power Usage Effectiveness |
| `tracking_err_kw` | float | Regulation tracking error [kW] |
| `delta_p_actual_kw` | float | Actual power change [kW] |
| `delta_p_demanded_kw` | float | Grid demanded power change [kW] |
| `lmp` | float | LMP [$/MWh] |
| `regd_signal` | float | RegD signal [-1, 1] |
| `f_grid_hz` | float | Grid frequency [Hz] |
| `freq_dev_hz` | float | Frequency deviation [Hz] |
| `v_pcc_pu` | float | PCC voltage [pu] |
| `thermal_fault` | bool | Thermal termination triggered |
| `freq_fault` | bool | Frequency termination triggered |
| `voltage_fault` | bool | Voltage termination triggered |
| `scenario` | str | Active scenario name |

---

## 3. C2GMacroEnv

**File:** `c2g_env/env_high_level.py`  
**Class:** `C2GMacroEnv`  
**Timestep:** 15 minutes (= 180 × 5 s sub-steps)  
**Episode length:** 96 ticks = 24 hours

### 3.1 Action Space

`Box(low=[0,-1], high=[1,1], shape=(2,), dtype=float32)`

| Index | Name | Range | Meaning |
|-------|------|-------|---------|
| 0 | `commit_norm` | [0, 1] | Fraction of max regulation capacity (50 MW) to commit |
| 1 | `bess_target` | [-1, 1] | Mean BESS dispatch target for the 15-min window |

### 3.2 Observation Space

`Box(shape=(17,), dtype=float32)` — same 17-D structure as `C2GFastEnv` but **aggregated** over 180 sub-steps:

- `temp_A_norm`, `temp_B_norm` — **mean** over sub-steps
- `bess_soc` — **final** SOC at end of window
- `freq_dev_norm` — **mean** frequency deviation
- `v_pcc_pu` — **mean** PCC voltage
- All other fields — mean of sub-step values
- `backlog_norm` — **mean** backlog fraction over sub-steps

### 3.3 Reward Function

```math
\mathcal{R}_\text{macro} = \bar{\mathcal{R}}_\text{fast} + \lambda_\text{lmp} \cdot \text{LMP} \cdot P_\text{BESS,mean} - \lambda_\text{churn} \cdot |\Delta \text{commit}|
```
where $\bar{\mathcal{R}}_\text{fast}$ is the mean of the 180 sub-step rewards (see §2.3 for the fast-step reward formula). The LMP bonus and churn penalty are set in `conf/algo/ppo_macro.yaml`.

### 3.4 The `inner_action_fn` Interface

The macro environment can receive a callable that maps `(obs_17d) → action_4d`.  
This is the **Hierarchical RL interface** — the inner function is the trained low-level policy.

```python
from stable_baselines3 import PPO
from c2g_env import C2GMacroEnv

# Load a trained low-level policy
inner_model = PPO.load("trained_models/ppo_default_s42/final_model")
inner_fn = lambda obs: inner_model.predict(obs, deterministic=True)[0]

# Macro env uses that policy for its 180 sub-steps
env = C2GMacroEnv(scenario="default", inner_action_fn=inner_fn)
obs, _ = env.reset()
macro_action = env.action_space.sample()
obs, reward, term, trunc, info = env.step(macro_action)
```

If `inner_action_fn=None` (default), the macro env uses a **zero action** for sub-steps (for baseline evaluation only).

---

## 4. Simulator: Thermal Twin

**File:** `c2g_env/physics/thermal.py`  
**Class:** `ThermalTwin`

Models the thermal dynamics of two independently-cooled zones using **exact exponential integration** (unconditionally stable for any timestep).

### 4.1 Zone A — Liquid-Cooled (HPE Cray EX GPU Cluster)

**Governing ODE** (lumped-capacitance energy balance):

```math
C_A \frac{dT_A}{dt} = P_\text{IT,A} - K_\text{liq,eff}(T_A - T_\text{supply,A}) + K_\text{env,A}(T_\text{amb} - T_A)
```
where:
- $K_\text{liq,eff} = K_\text{liq} \cdot \max(u_{p,\min},\ u_p) \cdot f_\text{fault}$
- $f_\text{fault} \in [0, 1]$ — cooling fault factor (1.0 = normal, 0.6 = Scenario C)

**Exact solution** (per timestep $\Delta t$):

```math
T_A(t + \Delta t) = T_\text{eq} + (T_A(t) - T_\text{eq}) \cdot e^{-K_\text{total}\,\Delta t / C_A}
```
```math
T_\text{eq} = \frac{P_\text{IT,A} + K_\text{liq,eff} T_\text{supply,A} + K_\text{env,A} T_\text{amb}}{K_\text{liq,eff} + K_\text{env,A}}, \quad K_\text{total} = K_\text{liq,eff} + K_\text{env,A}
```
**CDU chiller COP** (degrades when supply temperature is lowered below design point):

```math
\text{COP}_\text{liq} = \text{COP}_\text{base,liq} \cdot \max\bigl(0.3,\ 1 - \beta_\text{liq}(T_\text{ref,A} - T_\text{supply,A})\bigr)
```
**Electrical power draws**:
- CDU chiller: $P_\text{cool,A} = K_\text{liq,eff}(T_A - T_\text{supply,A}) / \text{COP}_\text{liq}$  
- Circulating pump: $P_\text{pump} = P_\text{pump,max} \cdot \max(u_{p,\min},\ u_p)$

**Zone A Parameters:**

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Thermal capacitance | $C_A$ | 27,000 | MJ/°C |
| Liquid-loop heat transfer | $K_\text{liq}$ | 35.0 | MW/°C |
| Supply temperature (design) | $T_\text{ref,A}$ | 30.0 | °C |
| Base CDU COP | $\text{COP}_\text{base,liq}$ | 30.0 | — |
| COP degradation rate | $\beta_\text{liq}$ | 0.03 | /°C |
| Envelope coupling | $K_\text{env,A}$ | 0.5 | MW/°C |
| Max pump power | $P_\text{pump,max}$ | 1.5 | MW |
| Minimum pump speed | PUMP_MIN | 0.15 | — |
| Time constant (full speed) | $\tau_A = C_A/K_\text{liq}$ | ≈12.7 | min |

> **Thermal battery intuition:** Slowing the pump reduces $K_\text{liq,eff}$, causing $T_A$ to rise toward a higher equilibrium. This stores heat in the water-loop thermal mass — the agent can exploit this ~12.7 min time constant to absorb grid regulation signals without throttling workloads.

### 4.2 Zone B — Air-Cooled (HPE ProLiant Inference/DLRM)

**Governing ODE:**

```math
C_B \frac{dT_B}{dt} = P_\text{IT,B} - Q_\text{HVAC} + K_\text{env,B}(T_\text{amb} - T_B)
```
**HVAC cooling capacity** (three operating regimes):

```math
K_\text{eff} = K_\text{air} \cdot f_\text{fault} \cdot (0.3 + 0.7 \cdot u_\text{hvac})
```
```math
\text{COP}_\text{air} = \text{COP}_\text{base} \cdot \max(0.3,\ 1 - \alpha_\text{amb}(T_\text{amb} - 25)) \cdot \max(0.3,\ 1 - \beta_\text{air}(T_\text{ref,B} - T_\text{supply,B}))
```
| Regime | Condition | Effective ODE |
|--------|-----------|---------------|
| Below supply air | $T_B < T_\text{supply,B}$ | Ambient coupling only |
| Physics-limited | $T_B < T_\text{cross}$ | Full convective term $K_\text{eff}(T_B - T_\text{supply,B})$ |
| Capacity-limited | $T_B \geq T_\text{cross}$ | $Q_\text{HVAC} = P_\text{hvac} \cdot \text{COP}$ (constant) |

**Zone B Parameters:**

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Thermal capacitance | $C_B$ | 10,000 | MJ/°C |
| Full-fan air-side K | $K_\text{air}$ | 13.0 | MW/°C |
| Supply temperature (design) | $T_\text{ref,B}$ | 20.0 | °C |
| Base HVAC COP (25°C) | $\text{COP}_\text{base}$ | 3.5 | — |
| COP/ambient degradation | $\alpha_\text{amb}$ | 0.02 | /°C |
| COP/supply degradation | $\beta_\text{air}$ | 0.04 | /°C |
| Max HVAC electrical draw | $P_\text{hvac,max}$ | 50.0 | MW |
| Envelope coupling | $K_\text{env,B}$ | 0.5 | MW/°C |

### 4.3 Safety Thresholds

| Threshold | Value | Meaning |
|-----------|-------|---------|
| $T_\text{safe}$ | 35 °C | Silicon hard limit — episode terminates |
| $T_\text{warn,A}$ | 33 °C | Soft limit — thermal reward penalty begins |
| $T_\text{warn,B}$ | 33 °C | Soft limit — thermal reward penalty begins |
| ASHRAE Zone A supply range | [20, 40] °C | W3 allowable supply temperature |
| ASHRAE Zone B supply range | [15, 27] °C | A1 allowable supply temperature |

### References

[1] Incropera, F.P., et al. (2007) *Fundamentals of Heat and Mass Transfer*, 6th ed., Wiley (ISBN 978-0-471-45728-2). Ch. 5: lumped-capacitance ODE basis.  
[2] Moore, J.D., Chase, J.S., Ranganathan, P., Sharma, R. (2005) “Making Scheduling ‘Cool’: Temperature-Aware Workload Placement in Data Centers,” USENIX ATC 2005, pp. 61–75. <https://www.usenix.org/legacy/publications/library/proceedings/usenix05/tech/general/moore.html>  
[3] Tang, Q., Gupta, S.K.S., Varsamopoulos, G. (2008) “Energy-efficient Thermal-aware Task Scheduling for Homogeneous HPC Data Centers,” *IEEE Trans. Parallel Distrib. Syst.*, 19(11), 1458–1472. DOI: [10.1109/TPDS.2008.111](https://doi.org/10.1109/TPDS.2008.111)  
[4] ASHRAE TC 9.9 (2021) *Thermal Guidelines for Data Processing Environments*, 5th ed. (A1 = 15–27 °C, W3 = 5–40 °C). <https://www.ashrae.org/technical-resources/bookstore/datacom-series>  
[5] Patankar, S.V. (2010) “Airflow and Cooling in a Data Center,” *J. Heat Transfer*, 132(7), 073001. DOI: [10.1115/1.4000703](https://doi.org/10.1115/1.4000703)  
[6] Zimmermann, S., Meijer, I., Tiwari, M.K., Paredes, S., Michel, B., Poulikakos, D. (2012) “Aquasar: A hot water cooled data center with direct energy reuse,” *Energy*, 43(1), 237–245. DOI: [10.1016/j.energy.2012.04.037](https://doi.org/10.1016/j.energy.2012.04.037)  

---

## 5. Simulator: BESS

**File:** `c2g_env/physics/bess.py`  
**Class:** `BESSModel` (auto-selects backend)  
**Design spec:** 150 MWh / 50 MW Li-ion NMC (utility Megapack-class)

Two backend implementations with an identical public API:

| Backend | Selected when | Model |
|---------|--------------|-------|
| `_PySAMBESSModel` | `nrel-pysam` installed | NREL BatteryStateful — Shepherd voltage curve, I²R thermal, temperature-dependent capacity |
| `_SimpleBESSModel` | Default (no PySAM) | Equivalent-circuit with key non-linearities |

### 5.1 Round-Trip Efficiency (Pure-Python Backend)

```math
\eta(C\text{-rate}, \text{SOC}) = \max\!\left(0.70,\ \eta_\text{peak}^2 - k_C \cdot C^2 - k_\text{SOC} \cdot (\text{SOC} - 0.5)^2\right)
```
where $C = P / E_\text{nom}$ is the C-rate in h⁻¹.

### 5.2 SOC Dynamics

```math
\text{discharge:}\quad \Delta\text{SOC} = -\frac{P_\text{actual} \cdot \Delta t}{\eta \cdot E_\text{nom}}
```
```math
\text{charge:}\quad \Delta\text{SOC} = +\frac{|P_\text{actual}| \cdot \eta \cdot \Delta t}{E_\text{nom}}
```
### 5.3 SOC-Dependent Power Derating

Smooth derating near hard limits (avoids cliff-edge cutoffs):

```math
P_\text{discharge,max}(\text{SOC}) = P_\text{max} \cdot \min\!\left(1,\ \frac{\text{SOC} - \text{SOC}_\text{min}}{0.10}\right)
```
```math
P_\text{charge,max}(\text{SOC}) = P_\text{max} \cdot \min\!\left(1,\ \frac{\text{SOC}_\text{max} - \text{SOC}}{0.05}\right)
```
### 5.4 Capacity Fade

Linear calendar + cycle fade (1% per 1,000 equivalent full cycles):

```math
f_\text{age} \mathrel{+}= \frac{|P_\text{actual}| \cdot \Delta t}{E_\text{nom} \cdot 1000}
```
```math
E_\text{effective} = E_\text{nom} \cdot (1 - 0.2 \cdot f_\text{age})
```
### 5.5 BESS Parameters

| Parameter | Symbol | Value | Unit |
|-----------|--------|-------|------|
| Nominal energy | $E_\text{nom}$ | 150 | MWh |
| Max power | $P_\text{max}$ | 50 | MW |
| Min SOC (hard floor) | $\text{SOC}_\text{min}$ | 0.10 | — |
| Max SOC (hard ceil) | $\text{SOC}_\text{max}$ | 0.95 | — |
| Peak one-way efficiency | $\eta_\text{peak}$ | 0.97 | — |
| C-rate efficiency loss | $k_C$ | 0.008 | — |
| SOC efficiency loss | $k_\text{SOC}$ | 0.010 | — |
| Nominal pack voltage | $V_\text{nom}$ | 800 | V |
| Internal resistance | $R_\text{int}$ | 0.002 | Ω |

### References

[1] NREL PySAM BatteryStateful documentation. <https://nrel-pysam.readthedocs.io/en/main/modules/BatteryStateful.html>  
[2] Blair, N., DiOrio, N., Freeman, J., Gilman, P., Janzou, S. (2018) *System Advisor Model (SAM) General Description (Version 2017.9.5)*, NREL/TP-6A20-70414. <https://www.nrel.gov/docs/fy18osti/70414.pdf>  
[3] Xu, B., Zhao, J., Zheng, T., Litvinov, E., Kirschen, D.S. (2018) “Factoring the Cycle Aging Cost of Batteries Participating in Electricity Markets,” *IEEE Trans. Power Syst.*, 33(2), 2248–2259. DOI: [10.1109/TPWRS.2017.2733339](https://doi.org/10.1109/TPWRS.2017.2733339)  
[4] Shepherd, C.M. (1965) “Design of Primary and Secondary Cells: II . An Equation Describing Battery Discharge,” *J. Electrochem. Soc.*, 112(7), 657–664. DOI: [10.1149/1.2423659](https://doi.org/10.1149/1.2423659)  
[5] Wang, J., Liu, P., Hicks-Garner, J., et al. (2011) “Cycle-life model for graphite-LiFePO4 cells,” *J. Power Sources*, 196(8), 3942–3948. DOI: [10.1016/j.jpowsour.2010.11.134](https://doi.org/10.1016/j.jpowsour.2010.11.134)  
[6] Hesse, H.C., Schimpe, M., Kucevic, D., Jossen, A. (2017) “Lithium-Ion Battery Storage for the Grid,” *Energies*, 10(12), 2107. DOI: [10.3390/en10122107](https://doi.org/10.3390/en10122107)  

---

## 6. Simulator: Electrical Chain

**File:** `c2g_env/physics/electrical.py`  
**Class:** `DatacenterElectrical`

Models the full AC power flow from grid PCC to IT loads, including non-linear losses at each stage, PUE calculation, and PCC voltage.

### 6.1 IT Power Model

Non-linear server power vs. utilisation (superlinear exponent for GPU power):

```math
P_\text{IT,zone} = N_\text{racks} \cdot \left[P_\text{idle} + (P_\text{max} - P_\text{idle}) \cdot u^\alpha\right]
```
| Zone | Racks | $P_\text{idle}$ | $P_\text{max}$ | $\alpha$ |
|------|-------|----------------|----------------|---------|
| A (liquid, GPU) | 2,000 | 8 kW | 75 kW | 1.4 |
| B (air, CPU/inference) | 2,500 | 4 kW | 40 kW | 1.2 |

### 6.2 UPS Efficiency Model

```math
\eta_\text{UPS}(x) = \frac{\eta_\text{peak} \cdot x}{x + k_\text{loss}(1-x)^2 + k_\text{noload}}
```
where $x$ is the load fraction. Low load → poor efficiency (no-load losses dominate).

| Parameter | Zone A | Zone B |
|-----------|--------|--------|
| $\eta_\text{peak}$ | 0.97 | 0.96 |
| $k_\text{loss}$ | 0.03 | 0.04 |
| $k_\text{noload}$ | 0.005 | 0.008 |

### 6.3 Transformer Losses

```math
P_\text{loss,XFMR} = P_\text{iron} + P_\text{copper} \cdot \left(\frac{P_\text{load}}{S_\text{rated}}\right)^2
```
| Parameter | Value |
|-----------|-------|
| Rating $S_\text{rated}$ | 300 MVA |
| Iron (no-load) losses | 0.15 MW |
| Copper loss at rated | 0.6% |

### 6.4 PCC Voltage (Thévenin Model)

```math
v_\text{drop,pu} = P_\text{pu} \cdot R_\text{pu} + Q_\text{pu} \cdot X_\text{pu}
```
```math
v_\text{pcc,pu} = 1.0 - v_\text{drop,pu}
```
where $Z_\text{grid} = 0.04$ pu, X/R ratio = 10 ⟹ $X_\text{pu} \approx 0.0398$, $R_\text{pu} \approx 0.00398$.

**Voltage thresholds** (ANSI C84.1):

| Threshold | Value | Action |
|-----------|-------|--------|
| $V_\text{min,safe}$ | 0.95 pu | Reward penalty begins |
| $V_\text{max,safe}$ | 1.05 pu | Reward penalty begins |
| UV relay | 0.90 pu | Episode terminates |

### 6.5 Power Usage Effectiveness

```math
\text{PUE} = \frac{P_\text{facility}}{P_\text{IT,total}} = \frac{P_\text{IT} + P_\text{cool} + P_\text{UPS loss} + P_\text{XFMR loss} + P_\text{aux}}{P_\text{IT}}
```
Practical range: 1.25 (excellent) to 2.5+ (very poor, Zone B heat wave).

### References

[1] Barroso, L.A., Hölzle, U., Ranganathan, P. (2019) *The Datacenter as a Computer*, 3rd ed., Morgan & Claypool. <http://dx.doi.org/10.2200/S00516ED2V01Y201306CAC024>  
[2] Fan, X., Weber, W., Barroso, L.A. (2007) “Power Provisioning for a Warehouse-sized Computer,” *ACM ISCA 2007*, pp. 13–23. DOI: [10.1145/1273440.1250665](https://doi.org/10.1145/1273440.1250665)  
[3] IEEE Std 3006.8/4447 — Recommended Practice for Analyzing Reliability Data for Equipment Used in Industrial and Commercial Power Systems (PUE). <https://standards.ieee.org/ieee/3006.8/4447/>  
[4] ASHRAE TC 9.9 (2021) *Thermal Guidelines for Data Processing Environments*, 5th ed. <https://www.ashrae.org/>  
[5] Economou, D., Rivoire, S., Kozyrakis, C., Ranganathan, P. (2006) “Full-System Power Analysis and Modeling for Server Environments,” MoBS workshop, ISCA 2006. <https://csl.stanford.edu/~christos/publications/2006.mantis.mobs.slides.pdf>  
[6] Shehabi, A., Smith, S., Sartor, D., et al. (2016) *United States Data Center Energy Usage Report*, LBNL-1005775. <https://escholarship.org/content/qt84p772fc/qt84p772fc.pdf>  

---

## 7. Simulator: Macro-Grid Signal

**File:** `c2g_env/physics/macro_grid.py`  
**Class:** `MacroGridSignal`

Generates frequency regulation signals, wholesale LMP prices, and grid frequency dynamics calibrated to six real energy markets.

### 7.1 Regulation Signal — AR(1) Process

The RegD-inspired normalised signal $s_t \in [-1, 1]$:

```math
s_t = \rho \cdot s_{t-1} + \sigma \cdot \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,1)
```
**Energy neutrality** (matches PJM/AEMO settlement requirements): the signal is mean-corrected over rolling windows so it integrates to ≈ 0 over each settlement period.

**Regulation dispatch command** [kW]:

```math
\Delta P_\text{demanded} = s_t \cdot P_\text{committed} \cdot 1000
```
### 7.2 Grid Frequency Model (Swing Equation)

```math
\frac{df}{dt} = \frac{1}{2H}\left(\Delta P_\text{pu,supply} - \Delta P_\text{pu,demand}\right) - \frac{D}{2H} \cdot \Delta f + \sigma_f \varepsilon_t
```
Discretised per 5-second step with $H = 6$ s (system inertia), $D = 1.5$ (load damping), $\sigma_f = 0.005$ Hz.

**Frequency safety thresholds:**

| Threshold | Value | Relay |
|-----------|-------|-------|
| Dead-band | ±0.2 Hz | No penalty |
| Penalty trigger | ±0.2 Hz | $\delta_f \cdot (|\Delta f| - 0.2)^+$ penalty begins |
| UFLS/OFGT trip | ±0.5 Hz | Episode terminates |

### 7.3 LMP Proxy

```math
\text{LMP}(L) = \text{LMP}_\text{base} + \text{LMP}_\text{slope} \cdot \max(0, L - L_\text{median})
```
where $L$ is the regional zone load in MW.

### 7.4 Market Presets

| Market key | Region | Regulation product | $\rho$ | $\sigma$ | Window | LMP base |
|-----------|--------|--------------------|--------|---------|--------|----------|
| `nyiso_nyc` | New York City | AGC Secondary Reserve | 0.996 | 0.018 | 180 ticks | $45/MWh |
| `pjm_dom` | N. Virginia | RegD (Fast-Response) | 0.992 | 0.025 | 180 | $42/MWh |
| `caiso_pgae` | Bay Area | REGU/REGD | 0.994 | 0.020 | 180 | $48/MWh |
| `ercot_north` | Dallas–FW | ECRS | 0.990 | 0.030 | 180 | $38/MWh |
| `entso_de` | Frankfurt | FCR-N | 0.997 | 0.015 | 360 | €52/MWh |
| `aemo_nsw` | Sydney | Regulation FCAS | 0.9945 | 0.032 | 60 | AUD$55/MWh |

> **Note on ENTSO-E:** 30-minute FCR-N settlement → longer correlation window (360 ticks). Nominal frequency is 50 Hz rather than 60 Hz.

### References

[1] PJM Manual 12: *Balancing Operations*, Section 4 — RegD signal specification. <https://www.pjm.com/-/media/documents/manuals/m12.ashx>  
[2] NYISO Real-Time Actual Load data, 5-minute resolution. <https://www.nyiso.com/real-time-dashboard>  
[3] Hogan, W.W. (2002) “Electricity Market Restructuring: Reforms of Reforms,” *J. Regulatory Economics*, 21(1), 103–132. DOI: [10.1023/A:1013682825693](https://doi.org/10.1023/A:1013682825693)  
[4] Kundur, P. (1994) *Power System Stability and Control*, McGraw-Hill (ISBN 978-0070359581). Ch. 12 — Swing equation; damping coefficient D.  
[5] Schweppe, F.C., Caramanis, M.C., Tabors, R.D., Bohn, R.E. (1988) *Spot Pricing of Electricity*, Kluwer Academic (ISBN 978-0-89838-260-0).  
[6] Kirby, B.J. (2005) *Frequency Regulation Basics and Trends*, ORNL/TM-2004/291. DOI: [10.2172/885974](https://doi.org/10.2172/885974)  

## 8. Simulator: Weather

**File:** `c2g_env/physics/weather.py`  
**Class:** `WeatherLoader`

Supplies per-tick ambient temperature $T_\text{amb}$ to the thermal simulator.

**Two modes:**

| Mode | When | Source |
|------|------|--------|
| Real data | CSV found in `weather_dir` | NOAA ISD-Lite hourly observations, linearly interpolated to 5-s ticks |
| Synthetic | CSV not found | Calibrated sinusoidal model with market-specific climate parameters |

**Synthetic climate model:**

```math
T_\text{amb}(d, h) = \bar{T}_\text{annual} + A_\text{seasonal} \cdot \cos\!\left(\frac{2\pi(d - d_\text{peak})}{365}\right) + A_\text{diurnal} \cdot \cos\!\left(\frac{2\pi(h - h_\text{peak})}{24}\right) + \sigma_\text{noise} \varepsilon
```
**Market weather stations:**

| Market | Station | City |
|--------|---------|------|
| `nyiso_nyc` | NYC (725053) | Central Park |
| `pjm_dom` | DCA | Reagan National |
| `caiso_pgae` | SJC | San José Mineta |
| `ercot_north` | DFW | Dallas–FW Intl |
| `entso_de` | FRA | Frankfurt |
| `aemo_nsw` | BKT | Bankstown (Sydney) |

> **Southern hemisphere:** AEMO NSW seasonal phase is inverted (summer = January). The model handles this automatically via a market-specific `d_peak` offset.

### References

[1] Smith, A., Lott, J.N., Vose, R. (2011) “The Integrated Surface Database: Recent Developments and Partnerships,” *Bull. Am. Meteorol. Soc.*, 92(6), 704–708. DOI: [10.1175/2011BAMS3015.1](https://doi.org/10.1175/2011BAMS3015.1)  
[2] Parton, W.J., Logan, J.A. (1981) “A model for diurnal variation in soil and air temperature,” *Agricultural Meteorology*, 23, 205–216. DOI: [10.1016/0002-1571(81)90105-9](https://doi.org/10.1016/0002-1571(81)90105-9)  
[3] ASHRAE (2021) *ASHRAE Handbook — Fundamentals*, Ch. 14: Climatic Design Information. <https://www.ashrae.org/technical-resources/ashrae-handbook>  
[4] Lee, K.P., Chen, H.L. (2013) “Analysis of energy saving potential of air-side free cooling for data centers in worldwide climate zones,” *Energy and Buildings*, 64, 103–112. DOI: [10.1016/j.enbuild.2013.04.013](https://doi.org/10.1016/j.enbuild.2013.04.013)  

---

## 9. Simulator: Workload Orchestrator

**File:** `c2g_env/physics/workload.py`  
**Class:** `WorkloadOrchestrator`

Fuses three real Alibaba cluster trace types into a total IT power demand at each tick.

| Trace | Source | Role | Power model |
|-------|--------|------|-------------|
| Batch (2023) | Alibaba cluster trace v2023 | **Flexible** — fully throttleable by DVFS | $P_\text{flex}(t) \cdot (1 - \text{throttle})$ |
| DLRM (2025) | Alibaba inference trace v2025 | **Rigid** — SLA-protected | $P_\text{DLRM}(t)$ |
| GenAI (2026) | Alibaba GenAI serving v2026 | **Rigid + spikes** — non-throttleable | $P_\text{GenAI}(t) \cdot s_\text{spike}$ |
| Spot (2026) | Spot/preemptible workloads | **Flexible** — lowest priority | $P_\text{spot}(t)$ |

**Total IT power split across zones:**

```math
P_\text{base} = P_\text{DLRM} + P_\text{GenAI} \quad \text{(rigid, Zone A)}
```
```math
P_\text{flex,nom} = P_\text{batch} + P_\text{spot} \quad \text{(schedulable)}
```
```math
P_\text{IT,actual} = P_\text{base} + P_\text{flex,nom} \cdot (1 - \text{throttle})
```
**GenAI spike model:** Poisson-distributed serving bursts scaled by `genai_spike_scale` (1.0 = nominal, 1.8 = Scenario A). The `is_spike` observation flag is set during active bursts.

### References

[1] Weng, Q., Xiao, W., Yu, Y., et al. (2022) “MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters,” *USENIX NSDI 2022*. <https://www.usenix.org/conference/nsdi22/presentation/weng>  
[2] Guo, J., Chang, Z., Wang, S., et al. (2019) “Who Limits the Resource Efficiency of My Datacenter: An Analysis of Alibaba Datacenter Traces,” *ACM IWQoS 2019*. DOI: [10.1145/3326285.3329074](https://doi.org/10.1145/3326285.3329074)  
[3] Fan, X., Weber, W., Barroso, L.A. (2007) “Power Provisioning for a Warehouse-sized Computer,” *ACM ISCA 2007*, pp. 13–23. DOI: [10.1145/1273440.1250665](https://doi.org/10.1145/1273440.1250665)  
[4] Wierman, A., Liu, Z., Liu, I., Mohsenian-Rad, H. (2014) “Opportunities and Challenges for Data Center Demand Response,” *IEEE IGCC 2014*. DOI: [10.1109/IGCC.2014.7039172](https://doi.org/10.1109/IGCC.2014.7039172)  
[5] Narayanan, D., Shoeybi, M., Casper, J., et al. (2021) “Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM,” *SC 2021*. DOI: [10.1145/3458817.3476209](https://doi.org/10.1145/3458817.3476209)  

---

## 10. Configuration Reference

**File:** `c2g_env/config.yaml`

All parameters are loaded at environment construction time. Override via Hydra:

```bash
uv run python baselines/train_ppo.py scenario=scenario_b reward.beta=3.0
```

### 10.1 Global Parameters

```yaml
global:
  dt_seconds: 5          # Timestep [s] — must divide 3600
  episode_ticks: 17280   # 24-hour episodes (17280 × 5 s)
  trace_dir:    "data/processed/workload_traces"
  energy_dir:   "data/processed/energy"
  weather_dir:  "data/processed/weather"
  grid_market:  "nyiso_nyc"
```

### 10.2 Reward Weights

```yaml
reward:
  alpha:              1.0   # Throughput incentive
  beta:               2.0   # Tracking error penalty
  gamma_thermal:      5.0   # Thermal penalty (°C above T_warn)
  T_warn_A:          33.0   # Zone A soft limit [°C]
  T_warn_B:          33.0   # Zone B soft limit [°C]
  soc_penalty:        0.5   # BESS near-empty penalty
  delta_freq_penalty: 2.0   # Frequency penalty (per Hz beyond ±0.2 Hz)
  delta_volt_penalty: 5.0   # Voltage penalty (per pu outside [0.95, 1.05])
```

### 10.3 Scenario Parameters

```yaml
<scenario_key>:
  grid_market:       "nyiso_nyc"   # Market key → regulation + LMP calibration
  weather_market:    "nyiso_nyc"   # Weather station key (can differ)
  weather_driven:    true          # true = real NOAA data, false = static T_amb
  T_amb:             25.0          # Initial/fallback ambient temperature [°C]
  committed_mw:      15.0          # Regulation capacity committed [MW]
  bess_soc_init:     0.50          # BESS initial SOC
  cooling_fault:     false         # CDU pump fault injection
  cooling_fault_factor: 0.6        # Efficiency when fault=true (1.0 = nominal)
  genai_spike_scale: 1.0           # GenAI spike magnitude multiplier
  grid_stress_scale: 1.0           # RegD signal amplitude multiplier
```

### 10.4 Hydra Configuration Tree

```
conf/
├── config.yaml           # Top-level: algo, scenario, market, logging defaults
├── algo/
│   ├── ppo.yaml          # PPO hyperparameters (300k steps, lr=3e-4, γ=0.99)
│   ├── sac.yaml          # SAC hyperparameters
│   └── ppo_macro.yaml    # PPO for C2GMacroEnv (100k steps, lr=1e-4, γ=0.995)
├── scenario/
│   ├── default.yaml
│   ├── scenario_a.yaml
│   ├── scenario_b.yaml
│   └── scenario_c.yaml
├── market/
│   ├── nyiso_nyc.yaml, pjm_dom.yaml, caiso_pgae.yaml
│   ├── ercot_north.yaml, entso_de.yaml, aemo_nsw.yaml
└── logging/
    └── tensorboard.yaml
```

---

## 11. Adding a New Scenario

1. **Add a block to `c2g_env/config.yaml`:**

```yaml
my_scenario:
  name: "My Custom Scenario"
  description: "Short description."
  grid_market:    nyiso_nyc
  weather_market: nyiso_nyc
  weather_driven: false
  T_amb: 35.0
  committed_mw: 25.0
  bess_soc_init: 0.40
  cooling_fault: false
  genai_spike_scale: 1.3
  grid_stress_scale: 1.4
```

2. **Create the matching Hydra config `conf/scenario/my_scenario.yaml`:**

```yaml
name: my_scenario
env_id: my_scenario
description: "My Custom Scenario"
grid_market:    nyiso_nyc
weather_market: nyiso_nyc
weather_driven: false
T_amb: 35.0
committed_mw: 25.0
bess_soc_init: 0.40
cooling_fault: false
genai_spike_scale: 1.3
grid_stress_scale: 1.4
```

3. **Train and evaluate:**

```bash
uv run python baselines/train_ppo.py scenario=my_scenario market=pjm_dom
```

No code changes required — the environments load any scenario key found in `config.yaml`.
