# DC2Grid: Hierarchical AI Orchestration for Grid-Interactive Hyperscale Data Centers

![DC2Grid Architecture](figures/dc2grid_architecture.png)

---

## 1. Executive Summary

This project addresses the **"AI-Energy Paradox"** by transforming 250 MW+ hyperscale data centers from passive power consumers into active, grid-balancing assets. By establishing a formal **Energy System Handshake**, we enable data centers to provide wholesale Frequency Regulation, stabilizing the regional transmission grid in exchange for significant revenue and faster deployment permits.

We solve this using a **Hierarchical AI Orchestration** framework that bridges long-term energy market bidding (minutes/hours) and sub-second hardware physics. The framework evaluates the synergy between three critical control levers: **Throttling Batch Workloads (DVFS)**, **Modulating Cooling Thermal Inertia (CDU pump)**, and **Dispatching Battery Energy Storage (BESS)**. This project delivers a high-fidelity cyber-physical benchmark for NeurIPS 2026, at the frontier of autonomous, grid-interactive infrastructure.

---

## 2. Background: Grid Frequency Regulation and RegD

### What is the RegD signal?

Every power grid must keep its frequency exactly at **60 Hz** (US) or **50 Hz** (EU) at all times. When a generator trips offline or a large load turns on suddenly, frequency deviates. Grid operators use **Automatic Generation Control (AGC)** to recruit *fast-response providers* — assets that can inject or absorb power within seconds to correct the imbalance.

**FERC Order 755 (2011)** created a pay-for-performance market for exactly this. Instead of paying only for *available capacity* (MW committed), it mandates that grid operators also pay for *accuracy* — how precisely an asset tracks the real-time regulation signal. PJM (the largest US grid operator) implemented this as the **RegD** signal: the "D" stands for *dynamic*, meaning it is designed for fast-response resources such as batteries and flexible loads.

### How the RegD signal works

Every **2–5 seconds**, the grid operator broadcasts a normalized score:

```math
\text{RegD}(t) \in [-1,\, +1]
```

The sign convention is:

| Signal value  | Grid instruction                          | Data center must...                                |
| ------------- | ----------------------------------------- | -------------------------------------------------- |
| **+1**  | Grid has excess load — reduce grid draw  | Shed batch load, discharge BESS, or slow cooling   |
| **−1** | Grid has excess generation — absorb more | Increase batch load, charge BESS, or raise cooling |
| **0**   | Balanced                                  | Hold current power level                           |

The actual MW response required is:

```math
\Delta P_{\text{demanded}} = C_{\text{MW}} \times \text{RegD}(t)
```

where $C_{\text{MW}}$ (`committed_mw`) is the regulation capacity the data center has pre-contracted to the market for the current 15-minute settlement interval.

### Statistical properties (AR(1) model)

The RegD signal is statistically modelled as a **first-order autoregressive (AR(1)) process** — persistent but zero-mean. At the 5-minute scale it has autocorrelation ρ ≈ 0.80, which time-scales to ρ ≈ 0.997 at the 5-second simulation step used in DC2Grid. The signal averages to zero over a settlement period, meaning the data center neither gains nor loses net energy from providing regulation.

In `c2g_env/physics/macro_grid.py`:

```python
self._regd_state = rho * self._regd_state + sigma * noise  # AR(1)
regd = np.clip(self._regd_state, -1.0, 1.0)               # normalise to [-1,1]
```

### The performance score (mileage metric)

Under FERC Order 755, the *performance score* is the correlation between the demanded signal and the actual response. A score of 1.0 = perfect tracking; 0.0 = random; below a threshold (typically 0.75) results in zero payment and market suspension. This maps directly to the **β tracking term** in the DC2Grid reward function.

### Why a data center is uniquely suited

A 250 MW hyperscale facility has three fast-response levers unavailable to most grid assets:

1. **Batch compute DVFS** — schedulable HPC/AI training jobs can be throttled in milliseconds via CPU/GPU frequency scaling. Service capacity is capped at `p_flex_max × throttle` (~90 MW); unserved work is **deferred into a FIFO queue** (not dropped) and served when capacity recovers. Average queue delay is tracked via Little's Law and exposed in `obs[16]` (`backlog_norm`).
2. **BESS** — the on-site 15 MWh / 5 MW battery can charge or discharge at full rate in under 100 ms, providing the fastest regulation response.
3. **Thermal inertia (CDU pump)** — the liquid cooling loop acts as a thermal capacitor (τ ≈ 12.7 min). Slowing the pump briefly stores heat in the water loop without immediately raising server temperatures, providing ~5–10 MW of additional regulation headroom for short intervals.

These three levers in combination can follow a RegD signal far more accurately than a single-asset provider, while the hierarchical RL agent learns the optimal trade-off between grid revenue, compute throughput, and thermal safety.

---

## 3. Problem Statement: The "Handshake" Gap

Current data center management systems are "grid-blind": they optimize internal efficiency (PUE) while ignoring the real-time needs of the regional energy system.

- **The Grid Need:** Modern grids require large loads to respond to Frequency Regulation signals (e.g., PJM RegD) every 2–4 seconds to balance renewable energy volatility.
- **The Datacenter Barrier:** Standard AI controllers cannot track these high-speed signals because they do not account for the non-linear physics of liquid cooling, battery degradation, and the bursty nature of GenAI workloads.
- **The Objective:** Create a synergy where the data center matches the grid's power signal perfectly without violating hardware safety limits or AI training SLAs.

---

## 4. State-of-the-Art and Our Contribution

| SOTA                                                                                                | Gap                                                                            | Our Step Further                                                                                                |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| **Wang et al., 2019** — Proved DCs can follow grid signals using DVFS.                       | Used "dummy loads" to intentionally waste power to meet the signal.            | We use **BESS + thermal storage synergy** — no wasted power.                                             |
| **Fu et al., 2021** — Demonstrated cooling systems have "thermal inertia" for grid services. | Relies on classical MPC, which fails under unpredictable GenAI serving spikes. | We replace MPC with **Hierarchical RL** to handle extreme, non-linear volatility of Alibaba GenAI traces. |
| **Li et al., 2026** — Identifies the need for intelligent VPP aggregation.                   | Lacks a standardized, high-fidelity physical testbed for datacenters.          | We provide the **first 250 MW-scale evaluation testbed** with real data across 6 global energy markets.   |

---

## 5. Technical Solution: Hierarchical AI Orchestration

### 5.0. Formal MDP Specification

DC2Grid defines a **two-level hierarchical Markov Decision Process**. The two agents share no parameters and communicate only through the `inner_action_fn` interface.

#### Lower-Level MDP — C2GFastEnv (5-second ticks)

```math
M_{\text{low}} = (\mathcal{S},\, \mathcal{A},\, P,\, R,\, \gamma,\, T)
```

| Symbol                                  | Definition                                                              |
| --------------------------------------- | ----------------------------------------------------------------------- |
| $\mathcal{S} \subset \mathbb{R}^{18}$ | Normalised observation vector (see §5.2 for index definitions)         |
| $\mathcal{A} = [0,1]^3 \times [-1,1]$ | Continuous 4-D action: throttle, pump speed, HVAC effort, BESS dispatch |
| $P(s_{t+1} \mid s_t, a_t)$            | Deterministic physics step + stochastic AR(1) RegD signal (see §2)     |
| $R(s_t, a_t)$                         | 7-term scalar reward (see §5.3)                                        |
| $\gamma = 0.99$                       | Training discount; undiscounted episodic sum used for benchmark ranking |
| $T = 17{,}280$                        | Steps per episode (24 h at 5 s per step)                                |

The only stochasticity in $P$ arises from the AR(1) process driving RegD$(t)$. All physics engines (thermal, BESS, electrical) are **deterministic** given $(s_t, a_t)$. A fixed seed fully determines the trajectory.

**Terminal states:** the episode ends early on three hard constraints — thermal fault ($T > 35\,°\text{C}$), frequency fault ($|\Delta f| > 0.5\,\text{Hz}$), or voltage fault ($v_\text{pcc} < 0.90\,\text{pu}$).

#### Upper-Level Semi-MDP — C2GMacroEnv (15-minute ticks)

The macro agent is framed as a **Semi-MDP** (Sutton et al., 1999) with fixed option duration $K = 180$ sub-steps:

```math
M_{\text{macro}} = (\mathcal{S}_M,\, \mathcal{A}_M,\, P_M,\, R_M,\, \gamma_M,\, T_M)
```

| Symbol                                    | Definition                                                                                                                                                                          |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\mathcal{S}_M \subset \mathbb{R}^{19}$ | Aggregated sub-step states: component-wise means + SOC endpoint + extrema + market context                                                                                          |
| $\mathcal{A}_M = [0,1]^2$               | 2-D: `bid_mw_norm` (MW capacity to offer), `bid_price_norm` (asking price)                                                                                                      |
| $P_M$                                   | $K$ applications of the lower-level transition $P$                                                                                                                              |
| $R_M$                                   | $\lambda_{\text{rev}} \times \text{regulation revenue} + \bar{r}_K - \lambda_{\text{elec}} \times \text{electricity cost} - \lambda_{\text{churn}} \times |\Delta\text{bid\_mw}|$ |
| $\gamma_M = \gamma^K$                   | $0.99^{180} \approx 0.163$ effective discount per macro step                                                                                                                      |
| $T_M = 96$                              | Macro steps per episode (24 h $\div$ 15 min)                                                                                                                                      |

where

```math
\bar{r}_K = \frac{1}{K}\sum_{i=0}^{K-1} r_i
```

and $r_i$ is the 5-second reward at sub-step i. Thus, $\bar{r}_K$ is the mean of the 180 fast-step rewards in macro step $k$.

The macro agent never directly observes the 5-second physics — it sees only the aggregated $\mathcal{S}_M$. This induces **partial observability** at the macro level that the agent must compensate for through robust bidding policies.

### 5.1. Upper-Level Agent: The Market Orchestrator (15-min ticks)

Manages the "Business Handshake." Observes regional market prices, weather forecasts, and the Alibaba batch job queue.

> **Decision:** *"How much MW capacity should I bid to the grid operator, and at what price, for the next 15 minutes?"*

Grid operators (PJM, ERCOT, etc.) clear ancillary-service markets in **15-minute settlement intervals**. The MacroEnv implements a **3-phase market handshake** each macro step: (1) the grid posts its RMCP and residual regulation need, (2) the DC agent bids MW capacity at an asking price, and (3) the grid probabilistically accepts the bid via a sigmoid function. If rejected, the DC falls back to a standing Demand Response (DR) baseline contract. The macro agent's challenge is to bid optimally under uncertainty about the next 180 RegD ticks, the next GenAI spike, and how much thermal headroom will remain at the end of the interval. The correct strategy is context-dependent — bid aggressively when the BESS is full and LMP is high; bid conservatively when ambient temperature is near the thermal limit or SOC is low. The macro agent never sees individual 5-second ticks; it only receives an aggregated summary *after* the interval completes, making this a **partially observable** planning problem.

- **Action Space (2-D):** `[bid_mw_norm ∈ [0,1], bid_price_norm ∈ [0,1]]` — MW capacity to offer (mapped to `[0, committed_max_mw]`) and asking price (mapped to `[0, 2 × rmcp_max]`).
- **Observation Space (19-D):** Aggregated over 180 sub-steps:

  | Index | Name                    | Range    | Description                                     |
  | ----- | ----------------------- | -------- | ----------------------------------------------- |
  | 0     | `temp_A_mean`         | [0, 1]   | Mean Zone A temperature / T_safe                |
  | 1     | `temp_B_mean`         | [0, 1]   | Mean Zone B temperature / T_safe                |
  | 2     | `bess_soc_end`        | [0, 1]   | SOC at end of the macro-step                    |
  | 3     | `p_base_mean`         | [0, 1]   | Mean p_base_norm                                |
  | 4     | `p_facility_mean`     | [0, 2]   | Mean p_facility_norm                            |
  | 5     | `regd_mean`           | [0, 1]   | Mean \|regd_signal\|                           |
  | 6     | `lmp_mean`            | [0, 1]   | Mean lmp_norm                                   |
  | 7     | `grid_load_mean`      | [0, 1]   | Mean load_norm                                  |
  | 8     | `tracking_err_mean`   | [0, 2]   | Mean \|ΔP_demanded − ΔP_actual\| / norm        |
  | 9     | `is_spike_any`        | {0, 1}   | 1.0 if any sub-step had a GenAI spike           |
  | 10    | `thermal_headroom_A`  | [0, 1]   | (T_safe − T_A_max) / T_safe                    |
  | 11    | `thermal_headroom_B`  | [0, 1]   | (T_safe − T_B_max) / T_safe                    |
  | 12    | `bid_mw_prev_norm`    | [0, 1]   | Previous macro-action bid MW                    |
  | 13    | `bid_price_prev_norm` | [0, 1]   | Previous macro-action bid price                 |
  | 14    | `freq_dev_mean`       | [-1, 1]  | Mean normalised frequency deviation             |
  | 15    | `v_pcc_mean`          | [0, 1.1] | Mean PCC voltage (per-unit)                     |
  | 16    | `backlog_norm_mean`   | [0, 2]   | Mean batch queue depth / p_flex_max             |
  | 17    | `rmcp_norm`           | [0, 5]   | Grid's posted RMCP / rmcp_max                   |
  | 18    | `reg_need_norm`       | [0, 5]   | Grid's residual regulation need / committed_max |
- **Reward:** $R_{\text{macro}} = \lambda_{\text{rev}} \times \text{regulation\_revenue} / 1000 + \bar{r}_K - \lambda_{\text{elec}} \times \text{electricity\_cost} / 1000 - \lambda_{\text{churn}} \times |\text{bid\_mw\_now} - \text{bid\_mw\_prev}|$
  - $\lambda_{\text{rev}} = 1.0$, $\lambda_{\text{elec}} = 0.5$, $\lambda_{\text{churn}} = 0.05$ (all in `c2g_env/config.yaml`)

### 5.2. Lower-Level Agent: The Hardware Controller (5 s ticks)

Executes the physical "Handshake." Receives the real-time frequency regulation signal and uses **four physical levers**. The central difficulty is that these levers have fundamentally different dynamics, costs, and side-effects — the agent must learn to combine them in the right order:

| Lever                               | Response time            | Capacity             | Side-effect                                                         | Role                                                                          |
| ----------------------------------- | ------------------------ | -------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **BESS** `action[3]`        | <100 ms                  | 5 MW / 15 MWh        | Depletes; capacity fade over time                                   | **First resort** — fastest, zero-penalty, finite                       |
| **CDU pump** `action[1]`    | Minutes (τ ≈ 12.7 min) | ~30 MW equivalent    | Thermal inertia; slow and partially irreversible on short intervals | **Second resort** — exploits physics cheaply                           |
| **HVAC** `action[2]`        | Seconds                  | ~50 MW draw          | Affects Zone B only; draws additional facility power                | **Defensive** — prevents thermal fault, not a primary regulation lever |
| **IT throttle** `action[0]` | Milliseconds             | Up to full flex load | Accrues FIFO backlog; cuts throughput revenue                       | **Last resort** — immediate but highest SLA cost                       |

The optimal policy learns this hierarchy: use BESS first (fast, free), borrow thermal inertia second (slow, cheap), fall back to DVFS only when both are exhausted. This mirrors how FERC-paid fast-response providers operate in real ancillary-service markets.

| Lever                        | Action dim    | Range   | Effect                                                                      |
| ---------------------------- | ------------- | ------- | --------------------------------------------------------------------------- |
| **IT (DVFS)**          | `action[0]` | [0, 1]  | Throttles schedulable Alibaba batch jobs; GenAI/DLRM rigid loads unaffected |
| **Cooling (CDU pump)** | `action[1]` | [0, 1]  | Modulates liquid cooling pump speed, exploiting thermal inertia             |
| **HVAC**               | `action[2]` | [0, 1]  | Zone B air-side fan speed                                                   |
| **BESS**               | `action[3]` | [-1, 1] | Charge (−) / discharge (+) the 15 MWh battery                              |

- **Action Space (4-D, continuous):** `[throttle_batch, pump_speed_A, hvac_effort, bess_dispatch]`
- **Observation Space (18-D, normalised):** | Index | Name                  | Range    | Description                                                  |
  | ----- | --------------------- | -------- | ------------------------------------------------------------ |
  | 0     | `temp_A_norm`       | [0, 2]   | Zone A (liquid-cooled GPU) temperature / T_safe              |
  | 1     | `temp_B_norm`       | [0, 2]   | Zone B (air-cooled CPU) temperature / T_safe                 |
  | 2     | `bess_soc`          | [0, 1]   | Battery state of charge                                      |
  | 3     | `p_base_norm`       | [0, 1]   | Rigid IT load (GenAI + DLRM)                                 |
  | 4     | `p_flex_nom_norm`   | [0, 1]   | New batch arrivals this tick (trace demand)                  |
  | 5     | `p_facility_norm`   | [0, 2]   | Total facility power                                         |
  | 6     | `regd_signal`       | [-1, 1]  | Grid regulation signal (signed)                              |
  | 7     | `lmp_norm`          | [0, 1]   | Locational marginal price                                    |
  | 8     | `grid_load_norm`    | [0, 1]   | Regional grid load stress indicator                          |
  | 9     | `is_spike`          | {0, 1}   | GenAI serving spike flag                                     |
  | 10    | `prev_throttle`     | [0, 1]   | Previous DVFS throttle                                       |
  | 11    | `prev_pump_speed`   | [0, 1]   | Previous pump speed                                          |
  | 12    | `pue_norm`          | [0, 2]   | Current Power Usage Effectiveness                            |
  | 13    | `T_amb_norm`        | [0, 1]   | Ambient temperature                                          |
  | 14    | `freq_dev_norm`     | [-1, 1]  | Normalised grid frequency deviation (swing equation)         |
  | 15    | `v_pcc_pu`          | [0, 1.1] | PCC voltage in per-unit (Thévenin model)                    |
  | 16    | `backlog_norm`      | [0, 2]   | Deferred batch queue depth / p_flex_max (Little's Law queue) |
  | 17    | `committed_mw_norm` | [0, 1]   | Current DR commitment / committed_mw_max                     |

### 5.3. The NeurIPS Evaluation Metric: The Tracking Reward

The scalar reward received at every 5-second tick has **seven additive terms**:

```math
\begin{aligned}
\mathcal{R} =&\; \alpha \cdot u_{\text{thr}} \\
  &- \beta \cdot \frac{|\Delta P_{\text{demand}} - \Delta P_{\text{actual}}|}{P_{\text{norm}}} \\
  &- \gamma \cdot (T - T_{\text{warn}})^{+} \\
  &- \delta_{\text{soc}} \cdot \mathbf{1}_{\text{soc}} \\
  &- \delta_f \cdot (|\Delta f| - 0.2)^{+} \\
  &- \delta_v \cdot \varepsilon_v \\
  &- \delta_q \cdot \frac{Q_{\text{backlog}}}{P_{\text{flex,max}}}
\end{aligned}
```

where:

- $(x)^{+} = \max(0,\, x)$ — ReLU / hinge: only positive exceedances are penalised
- $u_{\text{thr}} \in [0,1]$ — DVFS throttle fraction; fraction of flexible batch capacity currently committed
- $\Delta P_{\text{demand}} = C_{\text{MW}} \times \text{RegD}(t)$ — MW change requested by the grid operator this tick
- $\Delta P_{\text{actual}} = P_{\text{flex,served}} + P_{\text{BESS,actual}}$ — MW change the DC actually delivered
- $P_{\text{norm}} = C_{\text{MW}} \times 1000$ — normalisation constant (converts tracking error to a [0, ~2] range)
- $T$ — temperature of the hotter of the two cooling zones (°C)
- $T_{\text{warn}} = 33\,°\text{C}$ — soft warning threshold; thermal penalty begins here, 2 °C before the hard trip
- $\mathbf{1}_{\text{soc}}$ — binary flag: 1 if BESS state-of-charge is below $\text{SOC}_{\min} + 2\%$ (i.e. below 12%), else 0
- $|\Delta f|$ — absolute grid frequency deviation (Hz) from the 60 Hz nominal
- $\varepsilon_v = (0.95 - v_{\text{pcc}})^{+} + (v_{\text{pcc}} - 1.05)^{+}$ — PCC voltage exceedance (pu) outside the ANSI C84.1 Range A band $[0.95, 1.05]$
- $Q_{\text{backlog}}$ — deferred batch work currently sitting in the FIFO queue (kW-equivalent)
- $P_{\text{flex,max}} \approx 90{,}000\,\text{kW}$ — peak flexible IT capacity at full throttle (1,200 racks × 75 kW)
- Coefficients (all in `config.yaml`): $\alpha{=}1.0$, $\beta{=}2.0$, $\gamma{=}5.0$, $\delta_{\text{soc}}{=}0.5$, $\delta_f{=}2.0$, $\delta_v{=}5.0$, $\delta_q{=}2.0$

#### Term-by-term breakdown

| # | Term                | Coefficient                   | What it measures                                                                                        | Why it matters                                                                                              |
| - | ------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 1 | Throughput          | $\alpha = 1.0$              | Fraction of max IT capacity actually committed ($u_{\text{thr}} \in [0,1]$)                           | Maximising revenue — the agent earns more for accepting more DFS workload                                  |
| 2 | RegD tracking       | $\beta = 2.0$               | Normalised absolute error between the FERC-requested power change and what the DC actually delivered    | The primary ancillary-service obligation — missing this is penalised twice as hard as raw throughput gains |
| 3 | Thermal overrun     | $\gamma = 5.0$              | Degrees above the warning threshold $T_{\text{warn}} = 33°$C for the hotter of the two cooling zones | Linear ramp long before the hard 35 °C trip; $\gamma$ is large enough to dominate at +1 °C overshoot    |
| 4 | BESS SoC            | $\delta_{\text{soc}} = 0.5$ | Binary flag: 1 if the battery state-of-charge falls below $\text{SOC}_{\min} + 2\%$ (12%)             | Flat per-tick penalty prevents the BESS from being stranded near empty when a RegD ramp arrives             |
| 5 | Frequency deviation | $\delta_f = 2.0$            | Frequency excursion beyond the ±0.2 Hz NERC dead-band                                                  | Proportional penalty that steepens as the grid approaches the ±0.5 Hz trip threshold                       |
| 6 | Voltage deviation   | $\delta_v = 5.0$            | One-sided penalty for PCC voltage outside [0.95, 1.05] pu                                               | Voltage violations are fast and dangerous; the large coefficient forces early corrective action             |
| 7 | SLA backlog         | $\delta_q = 2.0$            | FIFO queue depth normalised by peak flexible capacity $P_{\text{flex,max}}$                           | Deferred batch jobs accumulate in queue; this term penalises latency and incentivises draining the queue    |

#### Core tension: why the agent must balance throughput vs. tracking

Terms 1 and 2 are structurally opposed:

- **Higher throttle** ($u_{\text{thr}} \uparrow$) → more revenue from IT (term 1 ↑) but increases the power baseline, making it harder to deliver a downward RegD ramp accurately (term 2 ↓).
- **Lower throttle** ($u_{\text{thr}} \downarrow$) → improves tracking flexibility but sacrifices revenue and grows the backlog (term 7 ↓).

The optimal agent learns a **lever hierarchy**: use BESS charge/discharge first (zero-penalty, fast), then exploit thermal inertia of the cooling system (slow, cheap), and only fall back to DVFS throttling as a last resort. This mirrors real-world FERC-paid frequency regulation.

#### Coefficient scaling rationale

All coefficients are chosen so that terms land in the same numerical range under typical operation:

- $\alpha = 1$ → throughput at $u_{\text{thr}} = 0.8$ contributes $+0.8$ per tick
- $\beta = 2$ → a 40% normalised tracking error contributes $-0.8$ per tick
- $\gamma = 5$ → 1 °C overshoot contributes $-5$ per tick, dominating immediately
- $\delta_v = 5$ → 5% voltage sag contributes $-0.25$ per tick, matching the thermal scale

#### Tracking loop

The RegD tracking error is computed as:

```math
\Delta P_{\text{actual}} = P_{\text{flex,served}} + P_{\text{BESS,actual}}
```

where $P_{\text{flex,served}} = \min\!\left(Q_{\text{backlog}},\ P_{\text{flex,max}} \times u_{\text{thr}}\right)$ is the batch work actually served from the FIFO queue this tick, and $P_{\text{BESS,actual}}$ is the net BESS power after battery dynamics.

#### Cumulative reward scale (per 24-hour episode)

| Agent                          | Typical range       | Notes                            |
| ------------------------------ | ------------------- | -------------------------------- |
| Random policy                  | −15,000 to −5,000 | Frequent thermal & voltage trips |
| Rule-based (threshold control) | −2,000 to +500     | No backlog awareness             |
| PPO (trained, 5 M steps)       | +2,000 to +5,000    | Learns lever hierarchy           |
| Adversarial scenario C         | −5,000 to −1,000  | High ambient temp + price spike  |

**Termination** (episode ends immediately):

- Thermal fault: $T_A > 35°$C or $T_B > 35°$C
- Frequency fault: $|f - f_{\text{nom}}| > 0.5$ Hz (UFLS / over-frequency trip)
- Voltage fault: $v_{\text{pcc}} < 0.90$ pu (under-voltage relay)

Episode truncates at 17,280 ticks (24 hours at 5 s).

## 6. Physics Engines

> **DC2Grid exposes exactly two Gymnasium environments** — `C2GFastEnv` and `C2GMacroEnv` — both registered under `gym.make()`. Everything below is *not* an environment: the six physics engines are internal simulation components with no `reset()/step()` or `observation_space/action_space` API. They are called exclusively by the two environments and are never exposed to an RL agent directly. If you want to interact with a physics engine in isolation (e.g. for unit testing or analysis), instantiate it directly from `c2g_env.physics.*`.

Six independent physics/data modules, all with exact-exponential or analytical solutions (unconditionally stable):

| Simulator                       | File              | Description                                                                                                                                                                                                                                       |
| ------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Workload Orchestrator** | `workload.py`   | Fuses Alibaba batch (2023), DLRM (2025), and GenAI (2026) traces into P_base + P_flex at 5-min resolution. FIFO queue model: unserved batch work defers rather than drops; exposes `backlog_kw` and `avg_delay_steps` (Little's Law) per step |
| **Thermal Twin**          | `thermal.py`    | Exact exponential ODE integration for dual-zone cooling (Zone A: HPE Cray EX liquid, Zone B: HPE ProLiant air)                                                                                                                                    |
| **Electrical Chain**      | `electrical.py` | Non-linear UPS/PDU/XFMR loss curves + PUE calculation                                                                                                                                                                                             |
| **BESS**                  | `bess.py`       | 15 MWh / 5 MW Li-ion NMC (pure-Python backend + optional PySAM) with C-rate η, SOC derating, capacity fade                                                                                                                                       |
| **Macro-Grid**            | `macro_grid.py` | AR(1) RegD signal + LMP proxy; calibrated for 6 global markets                                                                                                                                                                                    |
| **Weather**               | `weather.py`    | NOAA ISD-Lite real data or calibrated synthetic (6 climate profiles)                                                                                                                                                                              |

---

## 7. Data

### Real Datasets

| Dataset                   | Source                 | Markets/Zones                                             | Resolution        | Files   |
| ------------------------- | ---------------------- | --------------------------------------------------------- | ----------------- | ------- |
| **Workload traces** | Alibaba cluster traces | batch, DLRM, GenAI, spot                                  | 5-min             | 4 CSVs  |
| **Energy load**     | EIA, SMARD.de, AEMO    | NYISO (11 zones), PJM, CAISO, ERCOT, ENTSO-E DE, AEMO NSW | 5-min (resampled) | 16 CSVs |
| **Weather**         | NOAA ISD-Lite          | NYC, DCA, SJC, DFW, FRA, BKT                              | Hourly            | 7 CSVs  |

### 7.1. Workload Traces — Deep Dive

The benchmark fuses **three Alibaba production traces** to model the IT load of the 250 MW facility.
Each trace has a distinct statistical character, hardware zone assignment, and role in the control problem.

#### Trace Summary

| File                | Source                                                                                                                                               |   Duration   | Zone                  | Role                                             | Controllable?                                              |
| :------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------- | :-----------: | :-------------------- | :----------------------------------------------- | :--------------------------------------------------------- |
| `batch_v2023.csv` | [Alibaba GPU v2023 (`openb_pod_list_default.csv`)](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-pod-v2023)                      |    33 days    | A (GPU liquid-cooled) | `P_flex` — schedulable batch jobs             | ✅ DVFS throttle `action[0]` defers work into FIFO queue |
| `dlrm_v2025.csv`  | [Alibaba GPU v2025 (`disaggregated_DLRM_trace.csv`)](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2025)                    |    30 days    | B (CPU air-cooled)    | `P_base` — rigid DLRM inference serving       | ❌ Must be served regardless of grid state                 |
| `genai_v2026.csv` | [Alibaba v2026 GenAI (`qps.csv` and `pod_gpu_duty_cycle_anon.csv`)](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI) | 1 day (tiled) | A (GPU liquid-cooled) | `P_base` — rigid GenAI inference, spike-prone | ❌ Must be served; spikes set `obs[9]=1`                 |

> **`spot_v2026.csv`** is bundled but excluded from the current release — it requires an
> arrival-based preemptible scheduler not yet implemented.
>
> To reproduce these processed CSVs from the raw Alibaba data, see `preprocessing/workload_traces/`.
> All three files are loaded at startup by `c2g_env.physics.workload.WorkloadSimulator`.

---

#### Power Model

All three utilisation signals are translated to rack-level electrical power via the
**non-linear server power model** (Fan et al., ISCA 2007):

$$
P_{server}(u) = N_{racks} \times \bigl[ P_{idle} + (P_{max} - P_{idle}) \cdot u^{\alpha} \bigr]
$$

| Stream              | Racks | $P_{idle}$ | $P_{max}$ | $\alpha$            | Utilisation normaliser         |
| ------------------- | ----- | ------------ | ----------- | --------------------- | ------------------------------ |
| Batch (Zone A flex) | 1 200 | 8 kW/rack    | 25 kW/rack  | 1.4 (GPU superlinear) | `gpu_milli_request / 12 620` |
| GenAI (Zone A base) | 800   | 8 kW/rack    | 25 kW/rack  | 1.4                   | `avg_gpu_duty_cycle / 100`   |
| DLRM (Zone B base)  | 2 500 | 4 kW/rack    | 16 kW/rack  | 1.2 (CPU inference)   | `active_gpu_count / 227`     |

**Resulting power envelope** (30-day mean at default scenario):

| Stream                | Mean power         | Max power | Share of total IT |
| --------------------- | ------------------ | --------- | ----------------- |
| DLRM P_base (Zone B)  | ~21.7 MW           | 40.0 MW   | 56%               |
| Batch P_flex (Zone A) | ~10.1 MW           | 30.0 MW   | 26%               |
| GenAI P_base (Zone A) | ~6.8 MW            | 8.3 MW    | 18%               |
| **Total IT**    | **~38.5 MW** | —        | 100%              |

74% of IT power is rigid (`P_base`) — the agent's primary controllable lever is
batch throttling which covers only the remaining 26%.

---

#### Trace Characteristics

**`batch_v2023.csv` — Schedulable Batch (P_flex)**

- **Column:** `gpu_milli_request` (sum of GPU milli-cores requested per 5-min tick)
- **Statistics:** 78% of ticks have zero arrivals; mean utilisation ≈ 0.043; max = 12 620 gpu-milli
- **Nature:** Highly bursty. Jobs arrive sporadically with durations from 1–2 825 ticks (5 min to 9.8 days).
  Unserved work accumulates in a FIFO queue (tracked as `backlog_kw` and `avg_delay_steps` via Little's Law).
- **Agent implication:** DVFS throttle (`action[0]`) directly gates the batch service rate.
  Throttling below 1.0 reduces thermal load and peak grid draw at the cost of growing backlog.
  Reward term 1 (throughput) penalises low `action[0]`.

<p align="center">
  <img src="notebooks/fig_workload_distributions.png" width="95%" alt="Workload utilisation distributions"/>
</p>
<p align="center"><em>Utilisation distributions: batch is 78% zero (bursty); DLRM is near-Gaussian (always-on); GenAI is multimodal low-duty.</em></p>

---

**`dlrm_v2025.csv` — DLRM Inference (P_base, Zone B)**

- **Columns:** `active_gpu_count`, `active_cpu_cores`, `active_mem_gib`
- **Statistics:** Always non-zero (min=1 GPU); mean ≈ 101 GPUs; near-Gaussian distribution with a clear two-shift diurnal pattern.
- **Nature:** Continuous, predictable. DLRM (Deep Learning Recommendation Model) serving is the backbone of
  Zone B — it never drops below idle power. The 30-day trace captures weekday/weekend cycling clearly.
- **Agent implication:** Contributes the largest fixed baseload (~21.7 MW). The only thermal handle for
  Zone B is HVAC effort (`action[2]`); DLRM itself cannot be throttled.

---

**`genai_v2026.csv` — GenAI Serving (P_base, Zone A)**

- **Columns:** `total_qps`, `avg_gpu_duty_cycle`, `active_genai_pods`
- **Statistics:** 288 ticks (1 day) tiled cyclically; duty cycle mean ≈ 6.7%, max 24.4%; spike rate ≈ 25%
- **Nature:** Multimodal — most time near-idle, with sharp afternoon QPS bursts.
  Ticks where `avg_gpu_duty_cycle > P75 = 12.19%` are flagged as **spikes** (`obs[9] = 1`).
  GenAI runs on the same Zone A GPU racks as batch but with strict SLA priority.
- **Agent implication:** Spikes increase Zone A temperature rapidly (liquid cooling response time τ ≈ 13 min).
  The safety shield terminates episodes if `T_A > 35°C`. During spikes the agent must reduce batch load
  (`action[0]`) and possibly increase pump speed (`action[1]`) to prevent thermal fault.

<p align="center">
  <img src="notebooks/fig_workload_genai_spikes.png" width="95%" alt="GenAI spike analysis"/>
</p>
<p align="center"><em>Left: GenAI duty cycle with spike threshold (red dashes) and spike ticks (red dots). Right: spike probability peaks in afternoon hours.</em></p>

---

#### IT Power Breakdown

<p align="center">
  <img src="notebooks/fig_workload_power_breakdown.png" width="95%" alt="Stacked IT power breakdown"/>
</p>
<p align="center"><em>Stacked IT power (MW) over 30 days and 1-week zoom. The DLRM base (orange) dominates; batch flex (blue) provides the agent's only demand-side handle.</em></p>

---

#### Temporal Correlation

<p align="center">
  <img src="notebooks/fig_workload_acf.png" width="95%" alt="Autocorrelation function for all three traces"/>
</p>
<p align="center"><em>ACF up to 24 hours. DLRM is highly persistent (slow decay with 24-hour periodicity). Batch decorrelates fastest — it is the hardest to predict. GenAI reveals its 1-day tile boundary.</em></p>

The DLRM trace has the highest autocorrelation (predictable → MPC/rule-based works well for Zone B).
Batch is the most volatile (decorrelates within ~2 hours), making it the prime target for RL.

---

#### Batch Queue Dynamics

<p align="center">
  <img src="notebooks/fig_workload_queue_dynamics.png" width="75%" alt="Batch queue backlog under different throttle policies"/>
</p>
<p align="center"><em>Simulated backlog over 7 days at three throttle levels. At 50% throttle the queue stabilises near zero — the mean arrival rate is well within half-capacity. A completely off agent (throttle=0.3) accumulates ~10e3 MW equivalent backlog in 7 days.</em></p>

This reveals a key benchmark insight: **the batch queue is stable under mild throttle** (≥ 40%) because
the mean arrival rate (10.1 MW) is only 34% of full capacity (30 MW). The agent does not need to fully
commit compute to clear the queue; it has real headroom to throttle for grid regulation.

See [`notebooks/11_workload_deep_dive.ipynb`](notebooks/11_workload_deep_dive.ipynb) for full interactive analysis.

### 6 Global Energy Markets

| Market Key      | Region              | Grid Operator  | Energy Source | Weather Station    |
| --------------- | ------------------- | -------------- | ------------- | ------------------ |
| `nyiso_nyc`   | New York City       | NYISO          | NYISO OASIS   | NYC (Central Park) |
| `pjm_dom`     | Northern Virginia   | PJM            | EIA API       | DCA (Reagan Natl)  |
| `caiso_pgae`  | Bay Area / San Jose | CAISO          | EIA API       | SJC (Mineta Intl)  |
| `ercot_north` | Dallas–Fort Worth  | ERCOT          | EIA API       | DFW (DFW Intl)     |
| `entso_de`    | Frankfurt, Germany  | ENTSO-E / EPEX | SMARD.de      | FRA (Frankfurt)    |
| `aemo_nsw`    | Sydney, Australia   | AEMO / NEM     | AEMO CSVs     | BKT (Bankstown)    |

---

## 8. Evaluation Scenarios

DC2Grid ships four progressively harder 24-hour scenarios (17,280 ticks at 5 s each). Every scenario is fully deterministic when a fixed seed is set and can be combined with any of the six energy markets via a single Hydra override.

```bash
# Run any scenario × any market
uv run python baselines/train_ppo.py scenario=scenario_b market=ercot_north
```

### 8.1. Scene-setting: shared physics

All scenarios share the same underlying simulator stack and reward weights:

| Parameter           | Value                                                      | Meaning                                       |
| ------------------- | ---------------------------------------------------------- | --------------------------------------------- |
| Episode length      | 17,280 ticks                                               | 24 h × 3,600 s h⁻¹ ÷ 5 s tick⁻¹         |
| IT capacity         | 250 MW                                                     | Rigid (GenAI/DLRM) + flexible (Alibaba batch) |
| BESS                | 15 MWh / 5 MW                                              | NMC Li-ion, C-rate derating + capacity fade   |
| Cooling zones       | Zone A (liquid, HPE Cray EX) · Zone B (air, HPE ProLiant) |                                               |
| $T_{\text{safe}}$ | 35 °C                                                     | Silicon hard limit → immediate termination   |
| $T_{\text{warn}}$ | 33 °C                                                     | Soft threshold → thermal penalty begins      |
| Frequency UFLS      | ±0.5 Hz                                                   | Under/over-frequency relay → termination     |
| Voltage UV relay    | 0.90 pu                                                    | Under-voltage → termination                  |

---

### 8.2. `default` — Baseline Operations

> *"Can the agent learn to coordinate four physical levers under normal grid conditions?"*

The entry-level scenario. Ambient temperature is comfortable (25 °C, NYISO NYC summer), BESS starts at 50 % SOC, and the regulation signal has standard amplitude. No faults are injected. This is the recommended starting point for algorithm development and ablation studies.

| Parameter                  | Value                   |
| -------------------------- | ----------------------- |
| Market                     | NYISO NYC               |
| Ambient $T_{\text{amb}}$ | 25 °C (weather-driven) |
| Committed MW (max)         | 30 MW                   |
| BESS SOC₀                 | 50 %                    |
| GenAI spike scale          | 1.0× (nominal)         |
| Grid stress scale          | 1.0× (nominal)         |
| Cooling fault              | None                    |

**Primary challenge:** Learning the basic DVFS ↔ cooling ↔ BESS synergy to track the regulation signal while keeping temperatures below $T_{\text{warn}}$.

**Termination risk:** Low. An untrained random agent survives ≈ 40 % of the episode on average.

---

### 8.3. `scenario_a` — GenAI Crisis

> *"A viral model launch + a grid under-frequency event hit simultaneously. The agent must shed flexible load without starving the BESS."*

This scenario models a **Northern Virginia (PJM DOM)** summer day when a new GPT-class model goes viral. GenAI serving load spikes to **1.8× nominal**, consuming headroom that the agent would otherwise use for regulation. At the same time, the grid issues a sustained under-frequency signal, demanding active discharge. The agent must resolve the conflict between IT throughput and grid support.

| Parameter                  | Value           |
| -------------------------- | --------------- |
| Market                     | PJM DOM         |
| Ambient $T_{\text{amb}}$ | 30 °C (static) |
| Committed MW (max)         | 40 MW           |
| BESS SOC₀                 | 55 %            |
| GenAI spike scale          | **1.8×** |
| Grid stress scale          | **1.5×** |
| Cooling fault              | None            |

**Primary challenge:** IT vs. grid conflict. The GenAI rigid load is non-throttleable, so the agent must use BESS discharge and batch-job throttling simultaneously — but throttling reduces throughput reward $\alpha \cdot u_{\text{thr}}$, and over-discharging depletes the BESS.

**Termination risk:** Medium–High. Frequency faults are likely if the agent ignores the regulation signal. Thermal faults are possible if cooling is under-prioritised during spikes.

---

### 8.4. `scenario_b` — Thermal Squeeze

> *"Dallas in August: 40 °C ambient, a 30 MW commitment, and a cooling system pushed to its physical limits."*

This scenario targets **ERCOT North (DFW)** during a peak-summer heat wave. The 40 °C ambient temperature drives the cooling COP down by ≈ 30 %, meaning the pump must work harder to achieve the same heat rejection. The committed MW is raised to 30 MW, increasing the power swings the agent must track. GenAI load is nominal, but the thermal margin to $T_{\text{safe}}$ is extremely thin.

| Parameter                  | Value                     |
| -------------------------- | ------------------------- |
| Market                     | ERCOT North               |
| Ambient $T_{\text{amb}}$ | **40 °C** (static) |
| Committed MW (max)         | **60 MW**           |
| BESS SOC₀                 | 60 %                      |
| GenAI spike scale          | 1.0× (nominal)           |
| Grid stress scale          | 1.3×                     |
| Cooling fault              | None                      |

**Primary challenge:** Thermal constraint binding. The thermal penalty $\gamma \cdot (T - T_{\text{warn}})^{+}$ dominates the reward signal. The agent must learn aggressive pump-speed scheduling and accept reduced throughput to keep temperatures in the safe band.

**Termination risk:** Very High. A naive agent that ignores the pump lever will hit $T_{\text{safe}} = 35$ °C within the first hour. This scenario is the primary driver of thermal-safety research.

---

### 8.5. `scenario_c` — Battery Drain

> *"Western Sydney summer: the BESS starts nearly empty, the pump is failing, and the grid is stressed."*

This scenario represents a compounding failure in **AEMO NSW**. The BESS begins at only **15 % SOC** (near the 10 % hard floor), leaving almost no discharge capacity for regulation. A simulated CDU pump degradation reduces cooling efficiency to **60 % of nominal**, tightening the thermal margin. GenAI and grid stress are both elevated. The agent must simultaneously ration the BESS, compensate for degraded cooling, and track the regulation signal — with essentially no buffer.

| Parameter                  | Value                                        |
| -------------------------- | -------------------------------------------- |
| Market                     | AEMO NSW                                     |
| Ambient $T_{\text{amb}}$ | 32 °C (static)                              |
| Committed MW (max)         | 40 MW                                        |
| BESS SOC₀                 | **15 %**                               |
| GenAI spike scale          | 1.2×                                        |
| Grid stress scale          | 1.2×                                        |
| Cooling fault              | **Pump degradation (60 % efficiency)** |

**Primary challenge:** Resource scarcity under compound failure. The BESS SOC penalty $\delta_{\text{soc}}$ activates immediately. The agent must switch to DVFS-only regulation while the pump fault is active, and carefully trickle-charge the BESS when the regulation signal allows.

**Termination risk:** Extreme. This is the hardest scenario in the benchmark. A random agent terminates within ≈ 5 % of the episode on average.

---

### 8.6. Scenario × Market grid

All four scenarios can be combined with all six markets, yielding **24 distinct evaluation configurations**. Market selection changes the LMP profile, weather driver, and grid-stress statistics, while scenario selection changes the hardware stress and initial conditions:

|                | `nyiso_nyc` | `pjm_dom` | `caiso_pgae` | `ercot_north` | `entso_de` | `aemo_nsw` |
| -------------- | :-----------: | :---------: | :------------: | :-------------: | :----------: | :----------: |
| `default`    |  ★ default  |            |                |                |              |              |
| `scenario_a` |              | ★ default |                |                |              |              |
| `scenario_b` |              |            |                |   ★ default   |              |              |
| `scenario_c` |              |            |                |                |              |  ★ default  |

★ = default market for that scenario. Any other cell is a valid cross-market stress test.

```bash
# Example: Thermal Squeeze under European low-carbon prices
uv run python baselines/train_ppo.py scenario=scenario_b market=entso_de experiment.seed=1
```

---

## 9. Repository Structure

```
dc2grid/
├── pyproject.toml                       # uv/hatchling build + all dependencies
├── README.md
│
├── c2g_env/                             # The Core RL Environment
│   ├── __init__.py                      # Exports C2GFastEnv, C2GMacroEnv
│   ├── env_low_level.py                 # 5 s physics step — C2GFastEnv (18-D obs, 4-D act)
│   ├── env_high_level.py                # 15-min market step — C2GMacroEnv (19-D obs, 2-D act)
│   ├── ENVIRONMENTS.md                  # 📖 Full environment & simulator reference (equations, params)
│   ├── config.yaml                      # Centralised env configuration
│   ├── experiments/
│   │   ├── __init__.py                  # Exports ActionAblationFastEnv
│   │   └── action_ablation_env.py       # C2GFastEnv subclass for action-level ablation studies
│   └── physics/
│       ├── workload.py                  # Alibaba trace fusion (batch/DLRM/GenAI)
│       ├── thermal.py                   # Exact-exponential ODEs, dual-zone cooling
│       ├── electrical.py                # Non-linear UPS/PDU/XFMR loss + PUE
│       ├── bess.py                      # 15 MWh NMC BESS (pure-Python + PySAM)
│       ├── macro_grid.py                # AR(1) RegD + LMP proxy, 6 market presets
│       └── weather.py                   # NOAA ISD real data + synthetic climate, 6 presets
│
├── data/
│   └── processed/
│       ├── workload_traces/             # batch_v2023, dlrm_v2025, genai_v2026, spot_v2026
│       ├── energy/                      # 16 CSVs: 11 NYISO zones + PJM/CAISO/ERCOT/ENTSO-E/AEMO
│       └── weather/                     # 7 station CSVs: NYC, DCA, SJC, DFW, FRA, BKT, LONGIL + merged
│
├── conf/                                # Hydra configuration tree
│   ├── config.yaml                      # Top-level defaults (scenario, algo, market, logging)
│   ├── algo/                            # 19 algo configs: ppo, sac, ppo_macro, sac_macro, cpo,
│   │                                    #   ppo_lagrangian, cbf_ppo, hj_ppo, mpcsf_ppo, ha_c2g,
│   │                                    #   cbm_only, cbm_gate, cbm_shield, rule_macro_ppo, pid,
│   │                                    #   mpc_fast, mpc_macro, milp, shield_reward_shaping
│   ├── scenario/                        # default, scenario_a, scenario_b, scenario_c
│   ├── market/                          # nyiso_nyc, pjm_dom, caiso_pgae, ercot_north, entso_de, aemo_nsw
│   └── logging/                         # tensorboard.yaml
│
├── baselines/                           # NeurIPS Evaluation Agents
│   ├── _hydra_compat.py                 # Hydra 1.3.x compatibility patch for Python ≥ 3.14
│   ├── metrics_callback.py              # C2GMetricsCallback — per-episode CSV + TensorBoard
│   │
│   │  # ── Classical Controllers ───────────────────────────────────────────
│   ├── rule_based_mpc.py                # Threshold controller for C2GFastEnv (SB3-compatible)
│   ├── rule_based_macro.py              # Macro-level rule-based controller for C2GMacroEnv
│   ├── bang_bang.py                     # Bang-bang / hysteresis controller (floor baseline)
│   ├── pid_controller.py                # Multi-loop PID controller with anti-windup
│   │
│   │  # ── RL Training Scripts ─────────────────────────────────────────────
│   ├── train_sac.py                     # SB3 SAC (off-policy, auto entropy)
│   ├── train_hierarchical.py            # Two-phase sequential HRL pipeline (PPO inner)
│   ├── train_hierarchical_sac.py        # Two-phase HRL with SAC inner policy
│   ├── train_rule_macro_sac.py          # Rule-based macro + SAC inner policy
│   ├── train_lowsac_highrandom.py       # SAC lower + random macro (ablation)
│   ├── train_llm_agents.py              # LLM-guided agent training
│   │
│   └── safety/                          # HA safety methods + shielded training scripts (see §11)
│
├── evaluation/                          # Benchmark auditing & analysis
│   ├── run_benchmark.py                 # Standard benchmark: runs agents on all 4 scenarios
│   │                                    # Outputs: per-episode CSVs with cumulative power metrics
│   ├── run_ha_benchmark.py              # HA safety benchmark: 11-metric evaluation set
│   │                                    # Same cumulative power metrics as run_benchmark.py
│   ├── generate_plots.py                # Publication-ready PDF/PNG figures
│   ├── generate_ha_plots.py             # HA-specific: Pareto frontier, radar, violin plots
│   ├── plot_episode_traces.py           # Per-episode trace analysis with ablation filtering
│   ├── failure_analysis.py              # Failure-case categorisation for HA benchmark
│   └── statistical_analysis.py          # Bootstrap CIs, Welch's t-test, Cohen's d, LaTeX tables
│
├── scripts/                             # Data download & training utilities
│   ├── download_weather.py              # Open-Meteo ERA5 → 6 weather CSVs
│   ├── download_energy.py               # EIA + SMARD + AEMO → 5 energy CSVs
│   └── run_sweep.sh                     # Full training sweep (25 phases, ~270 jobs)
│
├── preprocessing/                       # Raw → processed data pipelines
│   ├── workload_traces/                 # process_v2023.py, process_v2025.py, process_v2026_genai.py
│   ├── energy/                          # process_energy.py (NYISO zone load)
│   └── weather/                         # download_noaa_isd.py
│
├── notebooks/                           # 11 Jupyter notebooks for exploration & visualisation
│   ├── 01_workload.ipynb                # Alibaba trace analysis
│   ├── 02_thermal.ipynb                 # Thermal model step response & steady-state
│   ├── 03_electrical_bess.ipynb         # Electrical chain + BESS cycling
│   ├── 04_macro_grid.ipynb              # RegD signal + LMP proxy
│   ├── 05_environments.ipynb            # Gym API demo, scenario comparison
│   ├── 06_weather.ipynb                 # Weather data: 6 markets, real vs. synthetic
│   ├── 07_energy_markets.ipynb          # Energy load: 6 markets, LDC, diurnal patterns
│   ├── 08_frequency_voltage.ipynb       # Grid frequency & PCC voltage safety signals
│   ├── 09_evaluation_scenarios.ipynb    # Scenario deep dive: params, rollouts, risk, reward
│   ├── 10_baselines_visualization.ipynb # Baseline agent comparison & visualisation
│   └── 11_workload_deep_dive.ipynb      # Workload queue dynamics & trace statistics
│
├── tests/                               # 531 tests (pytest)
│   ├── test_workload.py                 # 24 tests
│   ├── test_thermal.py                  # 32 tests
│   ├── test_electrical.py               # 27 tests
│   ├── test_macro_grid.py               # 30 tests
│   ├── test_weather.py                  # 23 tests
│   ├── test_gym_api.py                  # 72 tests (API compliance both envs)
│   ├── test_baselines.py                # 18 tests
│   ├── test_new_baselines.py            # 50 tests (classical + gradient-free baselines)
│   ├── test_frequency_voltage.py        # 31 tests (freq/voltage safety signals)
│   ├── test_hierarchical.py             # 22 tests (HRL, macro agents)
│   ├── test_safety_shield.py            # 24 tests (Simplex shield, wrappers)
│   ├── test_ha_safety.py                # 70 tests (3-tier HA safety methods)
│   ├── test_critical_bug_fixes.py       # 50 tests (regression tests)
│   ├── test_ablation.py                 # 18 tests (action ablation env)
│   ├── test_readme_smoke.py             # 13 tests (README code snippet validation)
│   ├── test_datalogging.py              # 7 tests (transition logging schema + 5 smoke tests)
│                                        #   Hardware vs macro column validation
│                                        #   5 CLI smoke tests: rule_macro, rule_based,
│                                        #   rule_based+BESS_ablation, ha_rule_based (variants)
│
└── figures/                             # Root-level figures (TensorBoard screenshot, etc.)
```

---

## 10. Quick Start

### Prerequisites

- **Python 3.11** (exact; `==3.11.*` in `pyproject.toml`)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone & install

```bash
git clone https://github.com/HewlettPackard/dc2grid.git
cd dc2grid
uv sync
uv sync --extra dev   # pytest, ruff, mypy
```

### Run the tests

```bash
uv run pytest tests/ -q
# 531 passed
```

### Train a single agent

```bash
# PPO — default scenario, 300k steps
uv run python baselines/train_ppo.py

# PPO — GenAI Crisis + PJM market
uv run python baselines/train_ppo.py scenario=scenario_a market=pjm_dom

# SAC — Thermal Squeeze
uv run python baselines/train_sac.py algo=sac scenario=scenario_b

# Hydra multirun — all scenarios × 3 seeds
uv run python baselines/train_ppo.py --multirun \
    scenario=default,scenario_a,scenario_b,scenario_c \
    experiment.seed=1,2,3

# Hierarchical RL — sequential two-phase pipeline
uv run python baselines/train_hierarchical.py

# Safety-shielded PPO (provable constraint satisfaction)
uv run python baselines/safety/train_shielded_ppo.py scenario=default

# Constrained RL — PPO-Lagrangian
uv run python baselines/safety/train_ppo_lagrangian.py scenario=default

# CPO — Constrained Policy Optimization
uv run python baselines/safety/train_cpo.py scenario=default

# CBF-shielded PPO (QP-based action projection)
uv run python baselines/safety/train_cbf_ppo.py scenario=default

# Full HA-C2G neuro-symbolic 3-layer architecture
uv run python baselines/safety/train_ha_c2g.py scenario=default
```

### Run the full benchmark sweep

```bash
# Dry-run first — prints all 48 jobs without executing anything:
bash scripts/run_sweep.sh --dry-run

# Full sweep (default: 4 parallel jobs):
bash scripts/run_sweep.sh

# Use more parallelism (208 cores available — 16 is safe):
MAX_PARALLEL=16 bash scripts/run_sweep.sh
```

The sweep runs in 25 phases:

| Phase | Jobs | What runs                                                 |
| ----- | ---- | --------------------------------------------------------- |
| 1     | 24   | Rule-Based + Random evaluation only (no training, ~5 min) |
| 2     | 12   | PPO training (300k steps) + evaluation                    |
| 3     | 12   | SAC training (200k steps) + evaluation                    |
| 4     | 12   | Macro Rule-Based evaluation                               |
| 5     | 12   | PPO-Macro training (100k steps) + evaluation              |
| 6     | 12   | HRL sequential training (300k + 100k) + evaluation        |
| 7     | 36   | Bang-Bang, PID, MPC evaluation (no training)              |
| 8     | 24   | MPC-Macro & MILP evaluation (no training)                 |
| 9     | 12   | PPO-Lagrangian training (300k) + evaluation               |
| 10    | 12   | CBF-PPO training (300k) + evaluation                      |
| 11    | 12   | HJ-PPO training (300k) + evaluation                       |
| 12    | 12   | MPC-SF-PPO training (300k) + evaluation                   |
| 13    | 12   | CPO training (300k) + evaluation                          |
| 14    | 12   | Shield-Reward-Shaping training (300k) + evaluation        |
| 15    | 12   | HA-C2G neuro-symbolic training (300k) + evaluation        |
| 16    | 12   | CBM-Only ablation training (300k)                         |
| 19    | 12   | CBM+Gate ablation training (300k)                         |
| 20    | 12   | CBM+Shield ablation training (300k)                       |
| 21    | 1    | HA Benchmark evaluation (11 metrics, 5 episodes)          |
| 22    | 1    | Summary table + LaTeX rows                                |
| 23    | 1    | Multi-seed HA benchmark (10 seeds × 5 episodes)          |
| 24    | 1    | Statistical analysis (CIs + significance tests)           |
| 25    | 1    | Failure-case analysis                                     |

The sweep produces a per-run results CSV (one row per run, upserted on re-runs) and a summary CSV (mean and standard deviation across seeds). Output destinations are defined in [scripts/run_sweep.sh](scripts/run_sweep.sh).

### Run benchmark evaluation directly

Use the evaluation runners when you want targeted experiments instead of the full sweep.
The `--fixed-action` setting allows granular control experiments by pinning selected actuators to analyst-chosen setpoints.

Unless `--output` is provided, the benchmark runner generates a CSV destination using the algorithm, scenario, controller type, and any fixed-action ablation. For example, a hardware PPO evaluation in `scenario_b` with BESS fixed at `0.5` receives an ablation-tagged CSV. Here `agent_type` denotes the transition-logging and output suffix for the evaluated controller, and can be `hardware`, `macro`, or `hardware_ha`.

---

#### Standard benchmark runner

Runs any combination of agents across all four evaluation scenarios and writes per-episode metrics to CSV.

```bash
# Classical hardware controllers (no trained models needed)
uv run evaluation/run_benchmark.py --agents rule_based bang_bang pid random

# SAC low-level agent (requires a trained model)
uv run evaluation/run_benchmark.py --agents sac --scenarios default scenario_b \
    --hw-model-dir trained_models/BestCheckpoints/sac_default_s100

# Hierarchical combos: rule-based macro + hardware controller
uv run evaluation/run_benchmark.py --agents rule_macro+sac rule_macro+rule_based \
    rule_macro+pid rule_macro+bang_bang rule_macro+random \
    --hw-model-dir trained_models/BestCheckpoints/sac_default_s100

# RL macro (Phase 2) + frozen SAC low-level
uv run evaluation/run_benchmark.py --agents sac_macro+sac \
    --macro-model-dir trained_models/BestCheckpoints/sac_macro_default_s100 \
    --hw-model-dir trained_models/BestCheckpoints/sac_default_s100

# LLM macro + hardware controller (requires a running vLLM server)
uv run evaluation/run_benchmark.py --agents llm_policy_macro+sac \
    --hw-model-dir trained_models/BestCheckpoints/sac_default_s100 \
    --llm-api-base http://localhost:8000/v1

# With transition logging (per-step CSV traces)
uv run evaluation/run_benchmark.py --agents rule_macro+sac --record_transitions \
    --hw-model-dir trained_models/BestCheckpoints/sac_default_s100
```

SAC agents select default checkpoint directories by algorithm, scenario, and seed. The examples above use bundled, Git-tracked SAC checkpoints via `--hw-model-dir` and `--macro-model-dir`; use these options to select other model directories.

**Agents used in the paper:**

| Agent                  | Type     | Description                                                                               |
| ---------------------- | -------- | ----------------------------------------------------------------------------------------- |
| `random`             | hardware | Uniform random baseline (lower bound)                                                     |
| `bang_bang`          | hardware | Hysteresis on/off controller                                                              |
| `pid`                | hardware | Multi-loop PID with anti-windup                                                           |
| `rule_based`         | hardware | Threshold heuristic controller (`baselines/rule_based_mpc.py`)                          |
| `sac`                | hardware | Trained SAC low-level controller (Phase 1)                                                |
| `rule_macro`         | macro    | Rule-based macro bidding controller                                                       |
| `sac_macro`          | macro    | Trained SAC macro controller (Phase 2)                                                    |
| `llm_policy_macro`   | macro    | LLM macro controller (Qwen3-32B, ICRL)                                                    |
| `<macro>+<hardware>` | combo    | Macro agent paired with hardware agent, e.g. `rule_macro+sac`, `llm_policy_macro+pid` |

**Fixed-action ablations (Appendix M):**

Pin actuators to fixed setpoints to isolate each lever's contribution:

```bash
# Disable BESS (throttle + cooling only)
uv run evaluation/run_benchmark.py --agents rule_macro+sac \
    --fixed-action bess_dispatch=0.0 \
    --hw-model-dir trained_models/BestCheckpoints/sac_default_s100

# Disable BESS and fix cooling (throttle only)
uv run evaluation/run_benchmark.py --agents rule_macro+sac \
    --fixed-action bess_dispatch=0.0 \
    --fixed-action pump_speed_A=0.7 \
    --fixed-action hvac_effort=0.7 \
    --hw-model-dir trained_models/BestCheckpoints/sac_default_s100
```

Action bounds: `throttle_batch ∈ [0, 1]`, `pump_speed_A ∈ [0, 1]`, `hvac_effort ∈ [0, 1]`, `bess_dispatch ∈ [-1, 1]`. Values are validated and clipped to these ranges.

**Key CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--agents` | `rule_based bang_bang pid random` | One or more agent names (see table above) |
| `--scenarios` | all 4 | Subset of `default scenario_a scenario_b scenario_c` |
| `--n_episodes` | `5` | Episodes per agent × scenario combination |
| `--seed` | `100` | Starting RNG seed; episode `i` uses `seed + i` |
| `--hw-model-dir` | `None` | Model directory for the hardware/inner SAC agent |
| `--macro-model-dir` | `None` | Model directory for the macro-level SAC agent |
| `--output` | auto-generated | Output CSV destination; otherwise named by algorithm, scenario, controller type, and ablation |
| `--record_transitions` / `--no-record_transitions` | disabled | Write per-step state/action/reward traces to per-episode CSVs in the runner's log hierarchy |
| `--append` | `False` | Append rows to an existing CSV instead of overwriting |
| `--fixed-action <name>=<value>` | none | Pin an actuator to a fixed setpoint (repeatable) |
| `--llm-api-base` | `http://localhost:8000/v1` | vLLM / OpenAI-compatible server URL |
| `--llm-template-path` | `conf/chat_templates/run_benchmark_rbc+ICRL.yaml` | YAML prompt templates for LLM agents |
| `--llm-max-new-tokens` | `8192` | Maximum tokens per LLM generation step |
| `--llm-temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `--llm-no-thinking` | off | Disable `<think>` reasoning blocks |
| `--llm-context-num-steps` | `10` | ICRL rolling buffer size in past steps (paper uses 5; 0 = disabled) |
| `--llm-context-stride` | `1` | Store every *K*-th step in the ICRL buffer |
| `--llm-icrl-mode` | `autonomous` | ICRL instruction mode: `autonomous` (paper default), `preset`, or `exploit` |

**Output metrics:**

For hardware agents, each row in the output CSV contains:

| Column                | Description                                            |
| --------------------- | ------------------------------------------------------ |
| `mean_reward`       | Mean step reward over the episode                      |
| `total_reward`      | Sum of step rewards                                    |
| `tracking_rmse`     | RMSE of `ΔP_demanded − ΔP_actual` (kW)            |
| `thermal_viol_rate` | Fraction of ticks with temperature > T_warn (33 °C)   |
| `throughput_ratio`  | Mean `p_flex_served / p_flex_nom`                    |
| `bess_degradation`  | Cumulative capacity fade × 10⁴                       |
| `episode_length`    | Ticks completed (< 17 280 indicates early termination) |
| `survival_rate`     | Fraction of episodes surviving to 24 h                 |

For macro agents, additional columns include:

| Column                  | Description                                  |
| ----------------------- | -------------------------------------------- |
| `bid_acceptance_rate` | Fraction of 15-min bids accepted by the grid |
| `total_reg_revenue`   | Cumulative regulation revenue (USD)          |
| `mean_perf_score`     | Mean FERC performance score                  |
| `mean_committed_mw`   | Mean accepted MW commitment per interval     |

For hierarchical combo agents (`macro+hardware`), results are split into separate macro and hardware CSVs automatically, with a separate hardware-schema row for the inner controller enabling comparison with standalone hardware results.

When `--record_transitions` is enabled and the run-log root defined in [evaluation/run_benchmark.py](evaluation/run_benchmark.py) exists, benchmark logs are grouped by experiment identity, agent, scenario, and controller type. An adjacent JSON manifest records the effective plant and thermal overrides, seed, source fingerprint, checkpoint fingerprints where available, and invocation identity. Macro and hardware streams share the same episode context. Repeated invocations receive separate directories.

#### Multiple facility capacities (50 / 100 / 250 / 500 MW)

DC2Grid provides **capacity-generalization profiles** for the benchmark and full-environment thermal runner. Profiles live under `conf/plant_profiles/` and override rack counts, transformer rating, thermal masses, cooling coefficients, BESS energy/power, and standing DR commitments. The environment scales each scenario's commitment cap with effective IT nameplate capacity. Extensive quantities scale linearly with capacity; per-rack power, COP, C/K time constants, SOC bounds, and setpoints remain unchanged. This preserves thermal-model ratios under matched relative inputs, but does not imply identical closed-loop trajectories against an unchanged regional grid and market.

| Profile | Nameplate | Scale | Default-scenario `committed_mw_max` | Zone A × B racks | BESS (`E_NOM_MWH` / `P_MAX_MW`) |
|---------|-----------|-------|--------------------|------------------|---------------------------------|
| `plant_50mw` | 50 MW | 0.2× | 6.0 | 240+160 × 500 | 3.0 / 1.0 |
| `plant_100mw` | 100 MW | 0.4× | 12.0 | 480+320 × 1000 | 6.0 / 2.0 |
| `none` *(default)* | 250 MW | 1.0× | 30.0 | built-in (150 MW A + 100 MW B) | 15.0 / 5.0 |
| `plant_500mw` | 500 MW | 2.0× | 60.0 | 2400+1600 × 5000 | 30.0 / 10.0 |

`none` = the built-in 250 MW facility (no override). Profile names are discovered automatically from `conf/plant_profiles/*.yaml`.

**Benchmark evaluation at a chosen capacity** — pass `--plant_profile`:

```bash
# Evaluate the classical controllers on a 500 MW facility
uv run evaluation/run_benchmark.py --agents rule_based bang_bang pid \
    --plant_profile plant_500mw

# Sweep every capacity in one run; each output row is tagged with `plant_profile`
uv run evaluation/run_benchmark.py --agents sac_macro+sac \
    --macro-model-dir trained_models/BestCheckpoints/sac_macro_default_s100 \
    --hw-model-dir trained_models/BestCheckpoints/sac_default_s100 \
    --plant_profile all
```

`--plant_profile` accepts `none` (default), `plant_50mw`, `plant_100mw`, `plant_500mw`, or `all`. When `all` is used, every row in the output CSV carries a `plant_profile` column so results can be grouped by capacity.

**Thermal-sensitivity sweep at a chosen capacity** — the cross-scenario thermal runner takes `--plant-profile` (extensive sweep values are automatically rescaled to match the selected size):

Set `THERMAL_RESULTS_CSV` to a new output CSV destination for the selected capacity before running this example.

```bash
uv run python -m c2g_env.experiments.thermal_sensitivity_cross_scenario \
    --plant-profile plant_100mw \
    --macro-controller sac_macro \
    --controllers sac \
    --out "${THERMAL_RESULTS_CSV:?Set THERMAL_RESULTS_CSV to the output CSV for this capacity}"
```

**Hydra / training runs:** the same profiles are exposed as a Hydra config group (`conf/config.yaml` selects `plant_profiles: none` by default). The standalone SAC and rule-macro SAC trainers forward this group to their environments; the older PPO, hierarchical, and safety trainers do not yet do so. For a supported trainer:

```bash
uv run baselines/train_sac.py plant_profiles=plant_500mw
```

Standalone (non-Hydra) callers can pass the override dict directly: `C2GFastEnv(plant_overrides=load_plant_profile("plant_100mw"))`.

**Capacity-sensitive execution.** BESS actions use the effective battery power rating. Macro bidding and benchmark rule-controller gains use the effective commitment cap: the scenario's 250 MW reference cap multiplied by effective IT nameplate MW / 250. At 50/100/250/500 MW, the default-scenario caps are 6/12/30/60 MW, scenario A/C caps are 8/16/40/80 MW, and scenario B caps are 12/24/60/120 MW. The cap is recalculated on reset, including scenario changes. An explicit `committed_mw_max` plant override remains an absolute MW limit and takes precedence. Standing DR baselines remain profile-defined. Regional grid size, regulation demand, competitor participation, and market pricing parameters are not capacity-scaled; realized bids and market outcomes can still differ.

Tracking denominator floors are configured by `reward.tracking_floor_fraction` and `reward.macro_tracking_floor_fraction` in [c2g_env/config.yaml](c2g_env/config.yaml). Each fraction is multiplied by effective IT nameplate kW. Their defaults retain the previous 250 MW floors of 100 kW for hardware reward and 1 kW for macro tracking observations and performance scores. Temperature, SOC, and grid protection thresholds are unchanged.

**Result identity and resuming.** Thermal CSVs record `run_id`, `episode_id`, `plant_profile`, `facility_capacity_mw`, and `run_config` alongside the existing metrics. Resume decisions distinguish effective thermal overrides, plant settings, scenario, seed, controllers, checkpoint contents, environment configuration, and source fingerprints. Older summary-only CSV schemas are rejected for appending; retain those files and choose a new output path. These fixes change non-default-capacity behavior, so previous results should not be treated as equivalent reruns.

The thermal runner records complete hardware and macro transitions in separate compressed JSONL streams in its default recording directory, or the destination selected with `--record-dir`. Records include reset state, full `info` mappings, actual durations, and separate termination/truncation flags. Episode summaries include online scalar statistics, energy integrals, tracking error, actuator and battery metrics, and reported market totals. A manifest preserves effective parameters, observation/action maps, source snapshots, package versions, checkpoint identities, and file checksums. Resume requires finalized, checksum-verified recordings; incomplete attempts remain separate. This recording does not change the environment's existing market-accounting formulas.

Recording schema 3 also exports all `C2GMetricsCallback` hardware-episode metrics as named CSV columns and under `hardware.episode_metrics` in the JSON summary. These include reward components, temperatures, tracking RMSE, PUE, power, SOC, action means and standard deviations, backlog, and fault indicators. In the sweep, `ep/length` counts recorded hardware steps and `ep/survived` requires the configured hardware horizon without termination, rather than the callback's legacy 288-tick cutoff. `reward/mean_throughput` is the mean throughput reward component; `workload/mean_throughput_ratio` is the mean reported batch throttle; `workload/served_work_ratio` is total served flexible work divided by arrivals. Choose a fresh output path when moving from an older CSV schema.

The matched-pair workflow is configured by `thermal_sensitivity.cross_scenario.controller_pairs`: rule macro with rule hardware, and SAC macro with SAC hardware. The existing launch helper performs nominal/adverse validation at the smallest and largest capacities, recomputes metrics from raw records, and estimates storage before the full grid. Its `--plant-profile` option limits a worker to one capacity and runs validation once for that capacity. Give parallel workers separate `--artifacts` directories and non-overlapping capacity assignments. Keep source and configuration files unchanged during a running sweep.

Default benchmark CSV names include non-default profile selections. Hydra sweep subdirectories include the job number so different conditions with the same seed do not collide within a sweep.

**Condition-aware analysis.** Statistical summaries and comparisons retain `plant_profile`, `facility_capacity_mw`, `config_id`, and structured `thermal_overrides` when those columns are present. Baselines are matched within a condition. The statistical CLI writes separate LaTeX tables for multiple conditions, with condition metadata alongside them. Direct table generation requires an explicit condition when the input contains several.

The plot generator writes result sets into directories identified by condition, each containing a JSON metadata file. Direct plotting helpers reject mixed conditions; CSVs without condition columns retain the legacy output layout. These changes do not add new metrics or convert the thermal summary schema into the ordinary benchmark plotting schema.

#### High-assurance benchmark runner

```bash
uv run evaluation/run_ha_benchmark.py --agents simplex_ppo cbf_ppo hj_ppo
uv run evaluation/run_ha_benchmark.py --agents ha_c2g --scenarios default scenario_c --n_episodes 5
uv run evaluation/run_ha_benchmark.py --agents cbf_ppo --record_transitions
uv run evaluation/run_ha_benchmark.py --agents cbf_ppo --no-record_transitions
uv run evaluation/run_ha_benchmark.py --fixed-action bess_dispatch=0.0
uv run evaluation/run_ha_benchmark.py \
  --fixed-action hvac_effort=0.9 \
  --fixed-action bess_dispatch=0.0
```

Key options:

- `--agents`: HA agents to evaluate
- `--scenarios`: scenarios to run
- `--n_episodes`: number of episodes per agent/scenario
- `--seed`: starting seed
- `--model_dir`: optional override for trained model directory
- `--output`: output CSV path
- `--record_transitions` / `--no-record_transitions`: enable or disable per-step transition logging
- `--fixed-action action=value`: assign a fixed value to an action

Notes:

- These settings allow granular experimentation and control for high-assurance studies as well: you can evaluate whether a safety method still works when specific actuators are pinned to fixed operating points.
- The same continuous low-level action ranges apply here: `throttle_batch ∈ [0, 1]`, `pump_speed_A ∈ [0, 1]`, `hvac_effort ∈ [0, 1]`, and `bess_dispatch ∈ [-1, 1]`.
- Fixed-action overrides are applied inside the low-level environment before dynamics are applied.
- When enabled, transition logs are written to per-episode CSVs grouped by agent, scenario, and high-assurance controller type.

#### Plotting episode traces and statistics

After generating per-episode transition CSVs (via `--record_transitions` in benchmark runners), you can visualize per-step state, action, observation, and reward traces as aggregated statistics (mean and 99% confidence intervals across episodes). The logs are grouped by algorithm, scenario, controller type, and any ablation settings.

**Plot episode statistics**:

```bash
# Basic usage (no ablation)
uv run evaluation/plot_episode_traces.py --algoname bang_bang --scenario default --agent-type hardware

# With ablation filters (plots only episodes matching specific disabled/fixed actions)
uv run evaluation/plot_episode_traces.py \
  --algoname bang_bang \
  --fixed-action pump_speed_A=0.25 \
  --scenario default \
  --agent-type macro
```

**Outputs:** JPEG and PDF figures named by algorithm, scenario, controller type, and any ablation settings.

Each figure contains one subplot per state/reward column, shows the **mean** line (solid) with a **99% confidence band** (shaded area) computed across all matching episodes.

- State variables (blue), with 0–1 reference bounds shown as dashed lines
- Cumulative reward components (red)

### Download real-world data (optional — CSVs are bundled)

```bash
uv run python scripts/download_weather.py --year 2024
uv run python scripts/download_energy.py  --year 2024
```

### Monitor training with TensorBoard

All training scripts log scalar metrics (episode reward, episode length, thermal/tracking/SOC penalties, shield interventions) to TensorBoard within the Hydra output directory. Set `TENSORBOARD_LOG_DIR` to the output root to compare runs, or to one run's event-log directory to inspect it separately.

```bash
uv run tensorboard --logdir "${TENSORBOARD_LOG_DIR:?Set TENSORBOARD_LOG_DIR to the logs to inspect}"
```

Then open [http://localhost:6006](http://localhost:6006) in your browser.

<p align="center">
  <img src="figures/tensorboard_screenshot.jpg" width="90%" alt="TensorBoard dashboard showing PPO training curves: episode reward, episode length, and per-term reward components"/>
</p>

### Explore interactively

```bash
uv run jupyter lab notebooks/
```

> **Note:** The optional `nrel-pysam` BESS backend requires `uv pip install nrel-pysam`.
> The environment automatically falls back to the pure-Python `_SimpleBESSModel` if absent.

---

## 11. High-Assurance Safety Controllers

DC2Grid provides a comprehensive **3-tier high-assurance (HA) safety framework** for grid-interactive data center control. All tiers enforce the same **5 hard constraints (C1–C5)** and are evaluated with an **11-metric set** (6 standard + 5 HA-specific).

### Hard Constraints

| ID | Constraint                   | Threshold                 | Physical Meaning                 |
| -- | ---------------------------- | ------------------------- | -------------------------------- |
| C1 | $T_A < T_{\text{safe}}$    | 35 °C (margin 1 °C)     | Server room A thermal limit      |
| C2 | $T_B < T_{\text{safe}}$    | 35 °C (margin 1 °C)     | Server room B thermal limit      |
| C3 | SOC ∈ [SOC_min, SOC_max]    | [0.10, 0.95] (guard 0.03) | BESS operational envelope        |
| C4 | $|\Delta f| < 0.5$ Hz      | ±0.4 Hz trigger          | UFLS / over-frequency protection |
| C5 | $V_{\text{pcc}} > 0.90$ pu | 0.92 pu trigger           | Under-voltage relay threshold    |

### Tier 1 — Hard-Guarantee Methods (provable safety via optimisation)

| Method                       | Shield                              | Permissiveness  | Cost                      | File                                      |
| ---------------------------- | ----------------------------------- | --------------- | ------------------------- | ----------------------------------------- |
| **Simplex** [Sha 2001] | O(1) analytic worst-case bounds     | Conservative    | Negligible                | `baselines/safety/safety_shield.py`     |
| **CBF** [Ames 2019]    | QP projection into barrier-safe set | Moderate        | Low                       | `baselines/safety/cbf_shield.py`        |
| **HJ Reachability**    | Offline BRS + runtime override      | Moderate        | Offline high, runtime low | `baselines/safety/hj_shield.py`         |
| **MPC Safety Filter**  | Receding-horizon constrained NLP    | Most permissive | Highest online            | `baselines/safety/mpc_safety_filter.py` |

#### Simplex Shield — Three Usage Modes

```python
# 1. Standalone filter — works with ANY agent
from baselines.safety.safety_shield import SafetyShield
shield = SafetyShield()
safe_action, was_modified, info = shield.filter(raw_action, obs)

# 2. Gymnasium wrapper — agent trains inside safe manifold
from baselines.safety.safety_shield import ShieldedEnv
env = ShieldedEnv(C2GFastEnv(scenario="default"))

# 3. SB3-compatible agent wrapper — for evaluation
from baselines.safety.safety_shield import ShieldedAgent
safe_agent = ShieldedAgent(trained_agent, env)
```

#### Training with Tier 1 Shields

```bash
# Simplex-shielded PPO
uv run python baselines/safety/train_shielded_ppo.py scenario=default experiment.seed=42

# CBF-shielded PPO (QP-based, more permissive than Simplex)
uv run python baselines/safety/train_cbf_ppo.py scenario=default

# HJ reachability-shielded PPO (offline BRS computation)
uv run python baselines/safety/train_hj_ppo.py scenario=default

# MPC safety filter PPO (receding-horizon, most permissive)
uv run python baselines/safety/train_mpcsf_ppo.py scenario=default
```

### Tier 2 — Constrained RL (soft constraint satisfaction during training)

| Method                          | Mechanism                                          | File                                                |
| ------------------------------- | -------------------------------------------------- | --------------------------------------------------- |
| **PPO-Lagrangian**        | Adaptive Lagrange multipliers for 3 cost types     | `baselines/safety/train_ppo_lagrangian.py`        |
| **CPO** [Achiam 2017]     | Trust-region with conjugate gradient + line search | `baselines/safety/train_cpo.py`                   |
| **Shield Reward Shaping** | Fixed quadratic distance-to-boundary penalties     | `baselines/safety/train_shield_reward_shaping.py` |

### Tier 3 — Neuro-Symbolic HA-C2G Architecture

The full **HA-C2G** pipeline is a 3-layer neuro-symbolic architecture:

1. **Layer 1 — Concept Bottleneck Model** (`baselines/safety/concept_bottleneck.py`): Maps raw 17-D obs → ~10 interpretable safety concepts (thermal margins, SOC health, etc.)
2. **Layer 2 — Safe Projection Gate** (`baselines/safety/safe_projection.py`): Concept-guided differentiable projection that blends policy actions toward safe priors based on learned pass-through gates; applied consistently during training and evaluation for `ha_c2g` and `cbm_gate`
3. **Layer 3 — Physics Rule Shield**: In-the-loop Simplex shield with shield-penalty reward

**Ablation studies** isolate each layer's contribution:

| Variant                 | CBM | Gate | Shield | File                                     |
| ----------------------- | :-: | :--: | :----: | ---------------------------------------- |
| **HA-C2G (full)** | ✅ |  ✅  |   ✅   | `baselines/safety/train_ha_c2g.py`     |
| CBM-Only                | ✅ |  ❌  |   ❌   | `baselines/safety/train_cbm_only.py`   |
| CBM+Gate                | ✅ |  ✅  |   ❌   | `baselines/safety/train_cbm_gate.py`   |
| CBM+Shield              | ✅ |  ❌  |   ✅   | `baselines/safety/train_cbm_shield.py` |

**Proof trees** (`baselines/safety/proof_tree.py`) generate per-timestep hierarchical audit logs documenting which safety rules passed/failed and the sensor readings grounding each decision.

### HA Evaluation Metrics (11-D)

| Category | Metric                        | Description                                    |
| -------- | ----------------------------- | ---------------------------------------------- |
| Standard | `mean_reward`               | Mean episode reward                            |
| Standard | `tracking_rmse`             | RegD tracking RMSE                             |
| Standard | `thermal_viol_rate`         | Fraction of ticks with thermal violation       |
| Standard | `throughput_ratio`          | Fraction of max IT capacity served             |
| Standard | `bess_degradation`          | Battery capacity fade over episode             |
| Standard | `survival_rate`             | Fraction of episodes surviving to 24 h         |
| HA       | `hard_violation_rate`       | Rate of C1–C5 constraint violations           |
| HA       | `shield_intervention_rate`  | How often the shield overrides the agent       |
| HA       | `constraint_margin`         | Mean distance from nearest constraint boundary |
| HA       | `worst_case_margin`         | Minimum margin across all constraints          |
| HA       | `computational_overhead_ms` | Per-step shield compute time                   |

### Evaluation & Analysis Tools

```bash
# HA benchmark evaluation (11 metrics across all HA agents)
uv run evaluation/run_ha_benchmark.py --agents simplex_ppo cbf_ppo hj_ppo mpcsf_ppo ha_c2g

# HA-specific plots (Pareto frontier, radar, violin, LaTeX table)
uv run evaluation/generate_ha_plots.py

# Failure-case analysis (where/why/how often agents fail)
uv run evaluation/failure_analysis.py

# Statistical analysis (bootstrap CIs, Welch's t-test, Cohen's d)
uv run evaluation/statistical_analysis.py
```

---

## 12. Strategic Value

### For the Energy System

- **Renewable Integration:** Data centers absorb excess wind/solar, preventing curtailment.
- **Grid Stability:** The DC acts as a "shock absorber" for the transmission grid, reducing reliance on fossil-fuel peaker plants.

### For AI Research (NeurIPS 2026)

- **Cyber-Physical Benchmark:** The first high-fidelity, multi-market testbed for hierarchical RL on real infrastructure physics.
- **Six Global Markets:** NYISO, PJM, CAISO, ERCOT, ENTSO-E, AEMO — largest DC hubs on Earth.
- **DOE Genesis Alignment:** 250 MW–1 GW scale matches the US national AI infrastructure program.

---

## 13. Citation

```bibtex
@inproceedings{dc2grid2026,
  title     = {{DC2Grid}: Multi-Objective Hierarchical {RL} Benchmark for Grid-Interactive
               Data Centers that Relieve Power System Stress},
  author    = {Anonymous},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
  year      = {2026},
}
```

---

## 14. Figure Gallery

All figures are generated by the notebooks in `notebooks/` and can be reproduced by running `uv run jupyter lab notebooks/`.

---

### Workload Traces (`01_workload.ipynb`)

<p align="center">
  <img src="notebooks/fig_workload_timeseries.png" width="49%" alt="Workload time-series: batch, DLRM, GenAI, spot traces fused at 5-min resolution"/>
  <img src="notebooks/fig_workload_hist.png"       width="49%" alt="Workload histogram: power distribution per trace type"/>
</p>
<p align="center">
  <img src="notebooks/fig_workload_spikes.png"     width="49%" alt="GenAI serving spike characterisation (v2026 trace)"/>
  <img src="notebooks/fig_workload_dvfs.png"       width="49%" alt="DVFS throttle effect: schedulable batch reduction vs. throttle level"/>
</p>

*Left-to-right, top-to-bottom:* **Fused workload time-series** (rigid GenAI/DLRM + flexible batch + spot); **power histogram** per trace type; **GenAI spike characterisation** showing burst magnitude and inter-arrival distribution; **DVFS throttle curve** mapping throttle level to actual batch power reduction.

---

### Thermal Twin (`02_thermal.ipynb`)

<p align="center">
  <img src="notebooks/fig_thermal_step.png"       width="49%" alt="Thermal step response: Zone A and B temperature rise from cold start"/>
  <img src="notebooks/fig_thermal_ss.png"         width="49%" alt="Thermal steady-state: equilibrium temperature vs. pump speed"/>
</p>
<p align="center">
  <img src="notebooks/fig_thermal_cop.png"        width="49%" alt="Cooling COP vs. ambient temperature for Zone A and B"/>
  <img src="notebooks/fig_thermal_hvac_sweep.png" width="49%" alt="HVAC sweep: Zone B temperature vs. fan speed at varying loads"/>
</p>
<p align="center">
  <img src="notebooks/fig_thermal_fault.png"      width="60%" alt="Thermal fault injection: CDU pump degradation scenario"/>
</p>

**Step response** (cold-start to thermal equilibrium); **steady-state map** (temperature vs. pump speed); **COP degradation** with ambient temperature (ERCOT 40 °C peak visible); **HVAC parameter sweep**; **fault injection** showing temperature excursion under 60% pump efficiency (Scenario C).

---

### Electrical Chain & BESS (`03_electrical_bess.ipynb`)

<p align="center">
  <img src="notebooks/fig_elec_breakdown.png" width="49%" alt="Power breakdown: IT, cooling, UPS, PDU, transformer losses"/>
  <img src="notebooks/fig_elec_pue.png"       width="49%" alt="PUE vs. facility load and ambient temperature"/>
</p>
<p align="center">
  <img src="notebooks/fig_elec_ups.png"       width="49%" alt="UPS efficiency curve: non-linear loss model"/>
  <img src="notebooks/fig_bess_cycle.png"     width="49%" alt="BESS charge/discharge cycle: SOC, power, capacity fade"/>
</p>
<p align="center">
  <img src="notebooks/fig_bess_eta.png"       width="60%" alt="BESS round-trip efficiency vs. C-rate"/>
</p>

**Power breakdown** across the facility electrical chain (IT → UPS → PDU → transformer); **PUE surface** showing how ambient temperature and load interact; **UPS non-linear efficiency curve**; **BESS charge/discharge cycle** with SOC tracking and capacity fade; **round-trip efficiency** vs. C-rate.

---

### Macro-Grid Signal (`04_macro_grid.ipynb`)

<p align="center">
  <img src="notebooks/fig_grid_profile.png" width="49%" alt="24-hour RegD-inspired regulation signal profile"/>
  <img src="notebooks/fig_grid_regd.png"    width="49%" alt="RegD signal power spectral density and autocorrelation"/>
</p>
<p align="center">
  <img src="notebooks/fig_grid_lmp.png"    width="49%" alt="LMP proxy time-series for 6 markets"/>
  <img src="notebooks/fig_grid_acf.png"    width="49%" alt="Regulation signal autocorrelation function (AR(1) calibration)"/>
</p>

**24-hour regulation signal** (AR(1) calibrated per market); **power spectral density and statistics**; **LMP proxy** across 6 global markets showing diurnal and seasonal patterns; **ACF plot** confirming AR(1) calibration quality.

---

### Environment API & Rollouts (`05_environments.ipynb`)

<p align="center">
  <img src="notebooks/fig_fast_rollout.png"    width="49%" alt="C2GFastEnv 24-hour rollout: temperature, SOC, power, reward"/>
  <img src="notebooks/fig_fast_components.png" width="49%" alt="Reward component breakdown over episode"/>
</p>
<p align="center">
  <img src="notebooks/fig_fast_reward.png"     width="49%" alt="Step-reward distribution across policies"/>
  <img src="notebooks/fig_obs_coverage.png"    width="49%" alt="Observation space coverage: 17-D normalised ranges"/>
</p>
<p align="center">
  <img src="notebooks/fig_macro_rollout.png"   width="49%" alt="C2GMacroEnv 15-min rollout: bid MW, bid price, market interaction"/>
  <img src="notebooks/fig_scenario_rewards.png" width="49%" alt="Reward comparison across all 4 scenarios"/>
</p>

**C2GFastEnv 24-hour rollout** (temperature, SOC, power, reward traces); **reward component breakdown** (tracking error, thermal penalty, SOC penalty, freq/voltage penalties); **step-reward distribution**; **observation space coverage** showing all 16 dimensions are exercised; **C2GMacroEnv rollout** at 15-min resolution; **cross-scenario reward comparison**.

---

### Weather Data (`06_weather.ipynb`)

<p align="center">
  <img src="notebooks/fig_weather_all_markets.png" width="49%" alt="Temperature profiles for all 6 markets throughout the year"/>
  <img src="notebooks/fig_weather_annual.png"      width="49%" alt="Annual temperature distribution by market"/>
</p>
<p align="center">
  <img src="notebooks/fig_weather_diurnal.png"     width="49%" alt="Diurnal temperature pattern by market and season"/>
  <img src="notebooks/fig_weather_syn_vs_real.png" width="49%" alt="Synthetic vs. real NOAA ISD weather comparison"/>
</p>
<p align="center">
  <img src="notebooks/fig_weather_cop.png"         width="49%" alt="Implied cooling COP from weather data across markets"/>
  <img src="notebooks/fig_weather_normhist.png"    width="49%" alt="Normalised temperature histogram: 6 markets"/>
</p>
<p align="center">
  <img src="notebooks/fig_weather_sh_flip.png"     width="60%" alt="Southern hemisphere seasonal flip: AEMO NSW vs. northern markets"/>
</p>

**Annual temperature profiles** for all 6 markets (NYC, DCA, SJC, DFW, FRA, BKT); **annual distribution**; **diurnal patterns** by season; **synthetic vs. real NOAA ISD validation**; **implied COP** showing how weather drives cooling cost; **normalised histograms**; **southern hemisphere seasonal inversion** (AEMO NSW summer = January).

---

### Energy Markets (`07_energy_markets.ipynb`)

<p align="center">
  <img src="notebooks/fig_energy_annual.png"        width="49%" alt="Annual grid load profile for 6 markets"/>
  <img src="notebooks/fig_energy_diurnal.png"       width="49%" alt="Diurnal load pattern by market and season"/>
</p>
<p align="center">
  <img src="notebooks/fig_energy_ldc_lmp.png"       width="49%" alt="Load duration curve and LMP distribution per market"/>
  <img src="notebooks/fig_energy_macrogrid.png"     width="49%" alt="Macro-grid load stress indicator calibration"/>
</p>
<p align="center">
  <img src="notebooks/fig_energy_weather_joint.png" width="49%" alt="Joint weather-energy distribution: COP vs. LMP"/>
</p>

**Annual grid load** (NYISO 11-zone, PJM DOM, CAISO PG&E, ERCOT North, ENTSO-E DE, AEMO NSW); **diurnal patterns**; **load duration curves + LMP distribution**; **grid stress indicator calibration** used by `macro_grid.py`; **joint weather–energy distribution** (ambient temperature vs. LMP — key for thermal-economic co-optimisation).

---

### Evaluation Scenarios (`09_evaluation_scenarios.ipynb`)

<p align="center">
  <img src="notebooks/fig_scenarios_params.png"      width="49%" alt="Scenario parameter overview: 6 key dimensions across 4 scenarios"/>
  <img src="notebooks/fig_scenarios_radar.png"       width="49%" alt="Radar chart: normalised stress profile per scenario"/>
</p>
<p align="center">
  <img src="notebooks/fig_scenarios_rollouts.png"    width="99%" alt="2-hour episode rollout traces: temperature, SOC, frequency, voltage per scenario"/>
</p>
<p align="center">
  <img src="notebooks/fig_scenarios_termination.png" width="49%" alt="Termination risk: episode length distribution under random policy"/>
  <img src="notebooks/fig_scenarios_reward.png"      width="49%" alt="Cumulative reward comparison across 4 scenarios"/>
</p>
<p align="center">
  <img src="notebooks/fig_scenarios_market_grid.png" width="70%" alt="Scenario x Market: all 24 valid evaluation configurations"/>
</p>

**Parameter overview** (6 bar charts: T_amb, committed MW, BESS SOC₀, GenAI scale, grid stress, cooling efficiency); **radar chart** showing the overall stress fingerprint of each scenario; **2-hour rollout traces** across all 4 scenarios for 5 physical signals; **termination risk** under 30 random-policy episodes; **cumulative reward gap** between scenarios; **24-configuration grid** of all Scenario × Market pairings.

<p align="center">
  <img src="notebooks/fig_scenarios_temp.png" width="70%" alt="Scenario temperature comparison: Zone A and B across all 4 scenarios"/>
</p>

**Zone temperature comparison** across all 4 scenarios showing the thermal headroom difference driven by T_amb and committed MW settings.

---

## Thermal Sensitivity Tests

We test whether the benchmark's thermal-safety results depend on permissive aggregate thermal assumptions by sweeping the plant's uncalibrated thermal coefficients while holding the safety anchors fixed (warning at 33 °C,
termination at 35 °C). The study varies 11 `ThermalTwin` coefficients one at a time, plus four named coupled cases, over physically grounded grids annotated
against ASHRAE, NIST, and vendor references (HPE, CoolIT, Vertiv). It runs a 288-run reduced open/closed-loop harness and a 2,880-episode full-environment
cross-scenario evaluation (36 unique plant configurations × 4 scenarios × 5 seeds × 4 hardware controllers: bang-bang, PID, rule-based, and a frozen SAC
checkpoint).

Across the study, controller failures were confined to configurations in which the cooling was already provisioned below a not-recommended threshold, that is,
plants in which the bare model crosses 35 °C under maximum cooling at continuous nameplate load. Under nominal or ordinary cooling assumptions we observed no thermal or SLA fault from any controller. This supports that the safety margins reported are a property of the modeled plant rather than permissive coefficients.

See the full writeup, including the annotated parameter grids, provenance, and
result tables: [`c2g_env/experiments/thermal_sensitivity.md`](c2g_env/experiments/thermal_sensitivity.md).

Reproduce with:

```powershell
python -m c2g_env.experiments.thermal_sensitivity
python -m c2g_env.experiments.thermal_sensitivity_cross_scenario
```

## PPO vs SAC

We also trained and evaluated a PPO macro agent on the default scenario, paired with the same frozen RL low-level controller. Its tracking, revenue, and safety are broadly comparable to SAC, with no thermal violations under either algorithm, which suggests that the staged 2-Phase (Sec. 7, Main paper) result does not appear to hinge on a single choice of macro learner.

| Configuration                              | Agent | RMSE ↓ (kW) | Regulation Revenue ↑ ($) | Macro Reward ↑ | Low-Level Reward ↑ | Thermal Violations ↓ | BESS Degradation ↓ ($\times 10^{-4}$) |              |              |            |               |
| ------------------------------------------ | ----- | -----------: | -----------------------------------------------------------------------------------------------------------------------------------: | -----------: | -----------: | ---------: | ------------: |
| RL Phase 2: RL macro + frozen RL low-level | SAC   |    780 ± 26 |                                                                                                                         8,550 ± 553 | 0.94 ± 0.13 | 0.85 ± 0.13 | 0.0 ± 0.0 | 18.38 ± 1.74 |
| RL Phase 2: RL macro + frozen RL low-level | PPO   |    745 ± 34 |                                                                                                                         7,345 ± 688 | 0.92 ± 0.13 | 0.81 ± 0.17 | 0.0 ± 0.0 | 21.22 ± 2.25 |
