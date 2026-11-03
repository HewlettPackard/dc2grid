# Original SAC-related code and metric audit

This is a bounded audit of original commit `75500133`, the corrected baseline
`5c7e057c`, and the unchanged scientific implementation at `34e01546`.
The original links below target the actual original revision. Corrected links
target its byte-preserved public clean export. No claim is made that every
possible defect in the repository has been ruled out.

## Important correction to earlier wording

These were not fixes to Stable-Baselines3 SAC's actor/critic update equations.
The findings mix physical inconsistencies, accounting, observation timing,
trainer wiring, diagnostics and evaluation defects. Some changes deliberately
redefined the task rather than repairing code that disagreed with the paper:

- Original camera_ready.tex line326 explicitly defines throughput as alpha*theta.
  Replacing it by served dynamic work / dynamic capacity is a reward redesign.
- Lines393-401 explicitly use mean LMP times normalized facility power.
  The old code matches that specified proxy. Treating it as a true electricity
  bill is dimensionally/physically inconsistent; metered PCC integration also
  changes reward scale and meaning, not merely an SAC implementation detail.
- Standardizing the nominal reference to pump=HVAC=0.7 changes the response
  estimand; the former cooling proxy used a different pump reference.
- Wider training coverage, eight fits, early checkpoint selection and two-hour
  physical recovery are experimental choices, not original optimizer bugs.

No one-factor causal ablation attributes a particular MW or RMSE gain to each
change. Original/corrected reward totals are not interchangeable. Comparisons
between SAC and PID are fair only when both use the same declared protocol.

## Physical power, work and reward measurement

### P1: Original tracking credits throttle reduction even when the queue serves exactly the same work and electrical load; it also omits negative flex relief during backlog repayment.

**Classification:** confirmed implementation/physical-response inconsistency, not an SAC algorithm bug. **Status:** Fixed in corrected/current physical_v1. Deliberately retained by legacy_v0..

**Original:** [c2g_env/physics/workload.py:WorkloadOrchestrator.step:239](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/workload.py#L239); [c2g_env/env_low_level.py:C2GFastEnv.step:453](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L453); [c2g_env/env_low_level.py:C2GFastEnv.step:514](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L514)

**Corrected:** [c2g_env/physics/workload.py:WorkloadOrchestrator.step:248](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/workload.py#L248); [c2g_env/env_low_level.py:C2GFastEnv.step:568](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L568)

Changes reward_tracking, reported tracking error, frequency-deficit input, and any RMSE/performance settlement derived from them. The sign of metric bias depends on the demanded response; no universal improvement or isolated SAC-gain attribution follows.

**Evidence details:** see the corresponding machine-readable ledger.

Full equations, probes and test anchors: [the physical ledger](reports/original_physical_audit.json).

### P2: Original nominal rack power, including nonzero idle power, is enqueued and can be deferred as if it were batch work.

**Classification:** confirmed idle-power/work-accounting inconsistency. **Status:** Fixed when physical_accounting=True, as selected by physical_v1. Standalone WorkloadOrchestrator still defaults to physical_accounting=False for compatibility..

**Original:** [c2g_env/physics/workload.py:_rack_power_kw:68](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/workload.py#L68); [c2g_env/physics/workload.py:WorkloadOrchestrator.__init__:182](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/workload.py#L182); [c2g_env/physics/workload.py:WorkloadOrchestrator.step:239](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/workload.py#L239)

**Corrected:** [c2g_env/physics/workload.py:WorkloadOrchestrator.__init__:148](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/workload.py#L148); [c2g_env/physics/workload.py:WorkloadOrchestrator.step:248](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/workload.py#L248); [c2g_env/env_low_level.py:C2GFastEnv.reset:358](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L358)

Changes service capacity, queue/SLA penalty and observation, flex power, and thermal loading. Corrected kWh are still energy-equivalent workload proxies, not measured job counts/FLOPs.

**Evidence details:** see the corresponding machine-readable ledger.

Full equations, probes and test anchors: [the physical ledger](reports/original_physical_audit.json).

### P3: Original thermal heat uses raw queued-work power, whereas the electrical model clips inverted utilization and reinstates the server idle floor.

**Classification:** confirmed cross-component electrical/thermal inconsistency. **Status:** Fixed in physical_v1 through matched heat input and retained idle load; legacy_v0 retains raw heat input..

**Original:** [c2g_env/env_low_level.py:C2GFastEnv.step:468](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L468); [c2g_env/physics/electrical.py:DatacenterElectrical._inverse_rack_util:175](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/electrical.py#L175); [c2g_env/physics/electrical.py:DatacenterElectrical._server_power_mw:162](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/electrical.py#L162); [c2g_env/physics/thermal.py:ThermalTwin.step:230](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/thermal.py#L230)

**Corrected:** [c2g_env/env_low_level.py:C2GFastEnv.step:542](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L542); [c2g_env/env_low_level.py:C2GFastEnv._advance_plant:786](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L786)

Can change thermal warnings/faults, cooling demand, feasible actions, episode length and all accumulated metrics. This affects every controller using the environment, not SAC alone.

Not a blanket mismatch at every operating point: the inverse/forward round trip agrees when raw zone power is within the allowed rack-power range. Original kW-to-MW division itself is correct.

**Evidence:** P_base_A=6400 kW, P_base_B=10000 kW, N=9600 kW, u=0: original heat input=16.4 MW but electrical IT=26 MW (9.6 MW discrepancy). Corrected physical_v1 heat and electrical IT both equal 26 MW.

Full equations, probes and test anchors: [the physical ledger](reports/original_physical_audit.json).

### P4: Original 'actual response' is not the difference in complete modeled facility/grid electrical draw; it omits liquid-chiller and conversion-loss changes.

**Classification:** confirmed response-meter boundary inconsistency plus intentional reference-policy change. **Status:** Physical_v1 uses the complete modeled facility boundary; legacy_v0 retains the original proxy. Neither counterfactual is a directly measured external utility baseline..

**Original:** [c2g_env/env_low_level.py:C2GFastEnv.step:503](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L503); [c2g_env/physics/thermal.py:ThermalTwin.p_cool_nominal_mw:73](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/thermal.py#L73); [c2g_env/physics/thermal.py:ThermalTwin.step:246](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/thermal.py#L246); [c2g_env/physics/electrical.py:DatacenterElectrical.step:282](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/electrical.py#L282)

**Corrected:** [c2g_env/env_low_level.py:_REFERENCE_COOLING:95](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L95); [c2g_env/env_low_level.py:C2GFastEnv.reset:479](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L479); [c2g_env/env_low_level.py:C2GFastEnv.step:568](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L568)

RMSE in kW under the two reference/response definitions is not the same estimand. Gross p_facility_mw excluding BESS is not by itself a bug; using gross power as net import would be a separate evaluator/cost-accounting issue outside this audit.

**Evidence:** At u=1, pump=HVAC=float32(0.7), B=0 and identical initial state: original reports 450.0006139278412 kW response; corrected physical_v1 reports exactly zero relative to its new reference. The approximately 450 kW difference reflects the intentional pump-reference change from 1.0 to 0.7, not solely a bug fix.

Full equations, probes and test anchors: [the physical ledger](reports/original_physical_audit.json).

### D1: Original rewards commanded throughput/DVFS fraction; corrected physical_v1 rewards served dynamic-work fraction of hardware capacity. The reported throughput_ratio has yet another denominator.

**Classification:** intentional throughput objective/metric redesign; not a demonstrated SAC or paper-implementation bug. **Status:** New objective in physical_v1; original definition retained in legacy_v0..

**Original:** [c2g_env/env_low_level.py:module reward documentation:46](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L46); [c2g_env/env_low_level.py:C2GFastEnv.step:573](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L573); [c2g_env/config.yaml:reward:50](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/config.yaml#L50)

**Corrected:** [c2g_env/env_low_level.py:C2GFastEnv.step:640](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L640)

Training targets and summed reward scale change even for identical service. Original and corrected reward totals cannot be compared as if only the policy/optimizer changed.

The original source explicitly documents alpha*throttle. Parent verification of original ZIP camera_ready.tex:323-336, especially :326, confirms the paper explicitly defines alpha*theta with matching original weights. Conversion to actual served_dynamic_work/dynamic_capacity is a changed reward definition coupled to the physics correction, not an original paper/code violation or an incorrect SAC update. The config comment 'per unit ... served' is less precise than the actual documented equation.

**Evidence details:** see the corresponding machine-readable ledger.

Full equations, probes and test anchors: [the physical ledger](reports/original_physical_audit.json).

### N1: Original tracking already uses actual, SOC-limited BESS power, not the requested dispatch.

**Classification:** verified non-finding for the simple BESS backend. **Status:** verified non-finding.

**Original:** [c2g_env/env_low_level.py:C2GFastEnv.step:445](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L445); [c2g_env/physics/bess.py:_SimpleBESSModel.step:137](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/bess.py#L137)

**Corrected:** [c2g_env/env_low_level.py:C2GFastEnv.step:519](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L519); [c2g_env/physics/bess.py:_SimpleBESSModel.step:137](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/bess.py#L137)

The complete BESS source file is unchanged across original, corrected, and current. This does not certify the optional PySAM adapter or the backend used by historical reported runs.

**Evidence:** At dt=5 s and command +/-5 MW: SOC=0.10/0.95 gives zero actual response; SOC=0.105/0.945 gives approximately +/-0.25 MW. Both revisions' tracking response increments equal those actual outputs, not the commands.

Full equations, probes and test anchors: [the physical ledger](reports/original_physical_audit.json).

### A1: Original tracking is scored against the RegD signal the agent actually observed, not the newly generated next-observation signal. However, the same step's returned info['regd_signal'] contains that next signal, inconsistent with info['delta_p_demanded_kw'].

**Classification:** fast reward/action signal alignment non-finding; confirmed transition-info signal alignment inconsistency. **Status:** Reward already aligned to observed RegD in original. Transition-info mismatch fixed in physical_v1, retained by legacy_v0. No unconditional stale-commitment bug established by this scope..

**Original:** [c2g_env/env_low_level.py:C2GFastEnv._build_obs_at_reset:737](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L737); [c2g_env/env_low_level.py:C2GFastEnv.step:516](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L516); [c2g_env/env_low_level.py:C2GFastEnv._build_obs:706](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L706); [c2g_env/physics/macro_grid.py:MacroGridSignal.step:385](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/macro_grid.py#L385)

**Corrected:** [c2g_env/env_low_level.py:C2GFastEnv._build_obs_at_reset:872](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L872); [c2g_env/env_low_level.py:C2GFastEnv.step:581](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L581); [c2g_env/env_low_level.py:C2GFastEnv._build_obs:849](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L849)

Original tracking_err_kw itself is computed from the observed signal. The metadata mismatch can corrupt action/signal reconstruction or consumers that reconstruct tracking from info.regd_signal. Returned info LMP likewise comes from the newly generated grid state in original (:636), whereas physical_v1 retains current-grid LMP (:759); Table1 settlement use remains unverified.

**Evidence:** Executed the exact source assignments/info expressions with observed RegD=0.25, commitment=10 MW, next RegD=-0.5. Both revisions demand 2500 kW and show -0.5 in the next observation. Original info.regd_signal=-0.5; corrected physical_v1 info.regd_signal=0.25. Thus reconstructing demand from original info.regd_signal yields -5000 kW instead of the correctly scored +2500 kW.

Full equations, probes and test anchors: [the physical ledger](reports/original_physical_audit.json).

## High-level accounting, clock and information

### M01: Published normalized facility proxy is not an actual-dollar electricity bill

**Classification:** dimensional_accounting_inconsistency_if_published_proxy_is_reported_as_actual_cost. **Status:** fixed.

**Original:** [c2g_env/env_high_level.py:C2GMacroEnv.step:394](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L394); [c2g_env/env_low_level.py:C2GFastEnv._build_obs:699](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L699); [c2g_env/env_low_level.py:<module>:90](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L90)

**Corrected:** [c2g_env/env_high_level.py:C2GMacroEnv.step:427](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L427)

The original Table 1 macro reward can faithfully report the published proxy objective. Its electricity term is much smaller than a dollar-cost term with the same lambda, so changing to metered cost materially changes high-level SAC's training target and reward scale; it is not a same-objective implementation repair. Full-window gross regulation revenue is unchanged when contract and performance are held fixed.

Factor 250 compares the proxy's numeric value with dollar accounting under the default scale, full duration, constant price/no covariance, no observation clipping, and zero battery exchange. It does not prove a paper-formula error or estimate the actual Table 1 reward change. Claims that the original field is actual USD remain dimensionally inconsistent.

**Evidence:** 100 MW facility, zero battery, $50/MWh, 180 five-second samples: original electricity_cost field is approximately 5 in normalized proxy units, not an actual $5 bill; corrected metered cost is $1250. At lambda_elec=0.5, reward_elec changes from approximately -0.0025 to -0.625. Exact source methods were executed against deterministic interface stubs; the numerical counterexample is unchanged.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M02: Published facility-level proxy differs from signed-PCC electricity accounting

**Classification:** published_facility_proxy_to_pcc_objective_boundary_change. **Status:** fixed.

**Original:** [c2g_env/env_high_level.py:C2GMacroEnv.step:394](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L394)

**Corrected:** [c2g_env/env_high_level.py:C2GMacroEnv.step:427](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L427); [c2g_env/env_low_level.py:C2GFastEnv.step:763](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L763)

The corrected objective adds sensitivity to battery charging/discharging at the PCC that the published facility-only proxy deliberately does not represent. This changes economic incentives as well as accounting scope; it is not evidence of an original implementation-versus-paper mismatch. Facility relief and PCC/battery relief remain distinct. The cost-component change alone does not alter frozen-policy full-window gross regulation_revenue.

Omission of battery exchange is an error only if the proxy is claimed to be an actual PCC bill. The corrected model bills signed net PCC energy at one LMP; it is not a separately validated import/export tariff model.

**Evidence:** At constant 100 MW facility draw and $50/MWh for 15 minutes, corrected bills are $1000 for 20 MW discharge, $1250 for neutral battery, and $1500 for 20 MW charging. The original normalized proxy is approximately 5 for all three, not an actual $5 bill, because only facility_norm enters its explicitly specified expression.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M03: Published product-of-means proxy differs from interval price-power settlement

**Classification:** published_aggregation_proxy_to_metered_cost_objective_change. **Status:** fixed.

**Original:** [c2g_env/env_high_level.py:C2GMacroEnv.step:358](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L358); [c2g_env/env_high_level.py:C2GMacroEnv.step:394](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L394)

**Corrected:** [c2g_env/env_high_level.py:C2GMacroEnv.step:427](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L427)

The paper explicitly specifies the product of means, and the original code implements it. Replacing it with mean products changes the reward response to within-window scheduling, so this is an objective refinement rather than a paper-fidelity fix. If actual metered cost is the intended reporting quantity, denormalization alone is insufficient; covariance determines the remaining difference. Corrected settlement also avoids the observation's upper clipping.

No claim is made that Table 1 had the synthetic covariance used here. Omitting covariance is not an implementation bug relative to the published proxy; representing that proxy as exact metered expenditure would be inaccurate.

**Evidence:** Equal half-windows at (50 USD/MWh, 100 MW) and (100 USD/MWh, 200 MW), no battery: original normalized proxy is approximately 11.25. Converting its power scale to MW gives a product-of-means estimate of $2812.50, versus metered interval cost $3125. The $312.50 difference is h*Cov(price,power). These numerical source counterexamples remain unchanged.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M04: Early-terminated macro windows settled a full nominal duration

**Classification:** true_original_duration_bug. **Status:** fixed.

**Original:** [c2g_env/env_high_level.py:C2GMacroEnv.step:345](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L345); [c2g_env/env_high_level.py:C2GMacroEnv.step:383](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L383)

**Corrected:** [c2g_env/env_high_level.py:C2GMacroEnv.step:412](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L412)

Can over-credit partially completed or failed windows and alter high-level training returns. It cannot change settlement for 96 fully executed 180-step windows.

**Evidence:** One five-second step, rate 40, commitment 10 MW, performance 0.9: original gross revenue $90; corrected $0.50. Full 180-step windows pay $90 under both methods.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M05: Macro clock hardcoded 180 steps and floor-divided the episode horizon

**Classification:** true_original_timebase_and_tail_bug. **Status:** fixed.

**Original:** [c2g_env/env_high_level.py:<module>:87](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L87); [c2g_env/env_high_level.py:C2GMacroEnv.__init__:205](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L205); [c2g_env/env_high_level.py:C2GMacroEnv.step:302](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L302); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv.step:90](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/train_envs/rule_macro_wrapped.py#L90)

**Corrected:** [c2g_env/env_high_level.py:C2GMacroEnv.__init__:205](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L205); [c2g_env/env_high_level.py:C2GMacroEnv.step:331](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L331); [c2g_env/env_low_level.py:C2GFastEnv.__init__:282](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L282); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv.step:105](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/train_envs/rule_macro_wrapped.py#L105)

Supports consistent 15-minute decisions under nondefault valid dt and preserves final partial windows. The default 5-second, 17280-tick, 96-window configuration has neither a macro-length nor an integer-remainder error.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M06: Agents bid before seeing the offer actually used to clear their bid

**Classification:** true_original_causal_observation_bug. **Status:** fixed.

**Original:** [c2g_env/env_high_level.py:C2GMacroEnv.macro_obs_from_reset:583](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L583); [c2g_env/env_high_level.py:C2GMacroEnv.run_handshake:622](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L622); [c2g_env/env_high_level.py:C2GMacroEnv.step:369](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L369); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv._do_macro_handshake:104](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/train_envs/rule_macro_wrapped.py#L104)

**Corrected:** [c2g_env/env_high_level.py:C2GMacroEnv.reset:252](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L252); [c2g_env/env_high_level.py:C2GMacroEnv.step:285](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L285); [c2g_env/env_high_level.py:C2GMacroEnv.step:444](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L444); [c2g_env/env_high_level.py:C2GMacroEnv.run_handshake:684](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L684); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv._do_macro_handshake:117](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/train_envs/rule_macro_wrapped.py#L117); [c2g_env/physics/macro_grid.py:MacroGridSignal.step_rmcp:493](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/macro_grid.py#L493)

High-level SAC and rule macro decisions now condition on the actual posted price/need. Bid acceptance, selected commitments, performance and gross revenue can change. This fixes environment observability, not an SB3 algorithm defect.

The pending-offer protocol assumes no unrelated code mutates the grid's market state between posting and clearing. No violation of that assumption in the normal audited paths was found.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M07: Low-level actions at reset/window boundaries observed the old commitment

**Classification:** true_original_action_observation_timing_bug. **Status:** fixed.

**Original:** [c2g_env/env_high_level.py:C2GMacroEnv.step:279](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L279); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv.reset:72](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/train_envs/rule_macro_wrapped.py#L72); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv.step:87](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/train_envs/rule_macro_wrapped.py#L87)

**Corrected:** [c2g_env/env_high_level.py:C2GMacroEnv.step:305](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L305); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv.reset:86](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/train_envs/rule_macro_wrapped.py#L86); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv.step:99](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/train_envs/rule_macro_wrapped.py#L99)

Removes one-transition stale commitment input at each changed macro contract. Any low-level policy/controller depending on that feature can be affected, including SAC. The size and sign of any Table 1 tracking or revenue effect require matched trajectories.

**Evidence:** In the source-method stub test, the original first inner observation reports commitment_norm=0 despite the just-cleared 10/20=0.5 commitment. Corrected reports 0.5 before the first inner action.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M08: Rule wrapper fabricated its initial macro state and wrote load to the LMP index

**Classification:** true_original_reset_observation_bug. **Status:** fixed.

**Original:** [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv._build_macro_obs:139](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/train_envs/rule_macro_wrapped.py#L139); [c2g_env/env_high_level.py:C2GMacroEnv.macro_obs_from_reset:538](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L538)

**Corrected:** [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv.reset:74](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/train_envs/rule_macro_wrapped.py#L74); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv._build_macro_obs:157](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/train_envs/rule_macro_wrapped.py#L157)

The initial rule-macro training-wrapper decision was based on a fabricated cold start rather than the actual scenario state. Corrected wrapper and MacroEnv paths now agree on tested streams.

This is specifically a RuleMacroWrappedEnv reset defect, not evidence that the original bare C2GMacroEnv reset fabricated all hardware features.

**Evidence:** For a real interface observation with SOC .65, LMP_norm .7, load_norm .8, the exact original wrapper builder outputs SOC 0, LMP_norm .5, load_norm 0, and market fields 0. Corrected outputs .65, .7, .8 and the posted market fields. Inner index 7 is LMP; load is index 8.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M09: Rule wrapper ignored the wrapped environment's custom configuration path

**Classification:** true_original_configuration_bug. **Status:** fixed.

**Original:** [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv.__init__:48](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/train_envs/rule_macro_wrapped.py#L48)

**Corrected:** [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv.__init__:48](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/train_envs/rule_macro_wrapped.py#L48)

Conditional on use of a custom configuration with different handshake parameters. No effect is established for default-config Table 1 runs.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M10: Source data rows were treated as simulation ticks instead of timestamped load samples

**Classification:** true_original_calendar_replay_bug. **Status:** fixed.

**Original:** [c2g_env/physics/macro_grid.py:MacroGridSignal._load_zone:676](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/macro_grid.py#L676); [c2g_env/physics/macro_grid.py:MacroGridSignal.step:385](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/macro_grid.py#L385); [c2g_env/env_low_level.py:C2GFastEnv.reset:374](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L374)

**Corrected:** [c2g_env/physics/macro_grid.py:MacroGridSignal.__init__:336](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/macro_grid.py#L336); [c2g_env/physics/macro_grid.py:MacroGridSignal.load_at_tick:463](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/macro_grid.py#L463); [c2g_env/env_low_level.py:C2GFastEnv.reset:421](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L421)

Changes price/load/calendar dynamics and therefore observations, macro decisions, economic returns and the SAC training distribution. Nominal five-minute rows are compressed approximately 60-fold when consumed every five seconds, but irregular timestamps make the exact ratio data-dependent.

The data witness describes the generic unmodified loader at the beginning of the dataset. Historical evaluation drivers may have replaced/rebased arrays; they were outside this scope. Do not equate this witness with the exact source-calendar coverage of the historical PID episodes without driver evidence.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M11: One RNG coupled exogenous RegD to market and frequency call counts

**Classification:** true_original_stochastic_isolation_bug. **Status:** fixed.

**Original:** [c2g_env/physics/macro_grid.py:MacroGridSignal.__init__:318](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/macro_grid.py#L318); [c2g_env/physics/macro_grid.py:MacroGridSignal.step_rmcp:452](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/macro_grid.py#L452); [c2g_env/physics/macro_grid.py:MacroGridSignal.clear_bid:488](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/macro_grid.py#L488); [c2g_env/physics/macro_grid.py:MacroGridSignal._step_regd:529](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/macro_grid.py#L529); [c2g_env/physics/macro_grid.py:MacroGridSignal._step_frequency:551](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/macro_grid.py#L551)

**Corrected:** [c2g_env/physics/macro_grid.py:MacroGridSignal._reset_signal_rngs:457](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/macro_grid.py#L457); [c2g_env/physics/macro_grid.py:MacroGridSignal.step_rmcp:493](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/macro_grid.py#L493); [c2g_env/physics/macro_grid.py:MacroGridSignal.clear_bid:542](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/macro_grid.py#L542); [c2g_env/physics/macro_grid.py:MacroGridSignal._step_regd:580](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/macro_grid.py#L580); [c2g_env/physics/macro_grid.py:MacroGridSignal._step_frequency:626](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/macro_grid.py#L626)

Extra market/frequency calls or different wrapper schedules no longer change exogenous RegD draws. Separating streams also intentionally changes realizations relative to the original under the same numeric seed.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M12: The advertised RegD window energy neutrality is not implemented

**Classification:** true_original_signal_model_and_documentation_bug. **Status:** not fixed.

**Original:** [c2g_env/physics/macro_grid.py:MacroGridSignal._step_regd:521](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/macro_grid.py#L521)

**Corrected:** [c2g_env/physics/macro_grid.py:MacroGridSignal._step_regd:572](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/macro_grid.py#L572)

Even nominal unbiased RegD can produce sustained per-window energy imbalance, influencing storage state, tracking and learned policies. Its individual contribution to Table 1 or SAC-versus-PID differences is unmeasured.

The explicit load-correlated RegD bias is an intentional stress intervention and is separately non-neutral by design. This audit tests the simulator's neutrality claim, not the real-world legal/regulatory requirements of PJM.

**Evidence:** Exact original and corrected _step_regd methods, N=180 and deterministic raw=0.5 every tick: first 179 outputs are 0.5, last is 0, window mean is 0.49722222222222223. About 99.44% of the original block mean remains.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M13: RegD window length and AR calibration still assume a five-second tick

**Classification:** remaining_timebase_limitation. **Status:** not fixed.

**Original:** [c2g_env/physics/macro_grid.py:MacroGridSignal.__init__:295](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/macro_grid.py#L295); [c2g_env/physics/macro_grid.py:<module>:85](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/macro_grid.py#L85)

**Corrected:** [c2g_env/physics/macro_grid.py:MacroGridSignal.__init__:361](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/macro_grid.py#L361)

Not an explanation of a default 5-second NYC Table 1 result. It limits alternative-dt interpretation. Fixed five-second rho and sigma similarly change their physical-time meaning at another dt.

**Evidence:** With valid physical dt=10 seconds, MacroEnv correctly uses 90 steps per 900-second action, but the NYC RegD generator still has window_ticks=180, hence a 1800-second correction block. The standalone constructor's unchanged default dt=300 and window_ticks=180 represent 15 hours, not 15 minutes.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M14: Macro tracking observation could violate its declared Box upper bound

**Classification:** true_original_observation_space_contract_bug. **Status:** fixed.

**Original:** [c2g_env/env_high_level.py:C2GMacroEnv.__init__:165](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L165); [c2g_env/env_high_level.py:C2GMacroEnv.aggregate_macro_obs:464](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L464)

**Corrected:** [c2g_env/env_high_level.py:C2GMacroEnv.__init__:209](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L209)

Corrects the numeric upper bound used by observation_space.contains(), without changing the numerical tracking observation, gross revenue formula or an SB3 optimizer. It does not repair cached Box boundedness metadata: see the separately documented, newly introduced current limitation C01. Large observation/value scales remain a possible learning concern, not proof of an SB3 update bug.

The counterexample holds the interface floor fixed to isolate the numeric-bound bug and tests contains(), not sample(). Physical_v1 separately supplies a larger small-commitment floor; that low-level change is outside this audit.

**Evidence:** Exact aggregation functions return obs[8]=1000 for tracking error 1000 kW, commitment 0, and a 1-kW interface floor. The original/legacy upper bound is 2. Current physical_v1 accepts the resulting otherwise-valid observation; legacy rejects it via observation_space.contains.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

### M15: Declared handshake sigmoid_k_p knob is not consumed by clearing

**Classification:** remaining_configuration_limitation. **Status:** not fixed.

**Original:** [c2g_env/config.yaml:None:69](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/config.yaml#L69); [c2g_env/physics/macro_grid.py:MacroGridSignal.clear_bid:488](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/macro_grid.py#L488)

**Corrected:** [c2g_env/config.yaml:None:72](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/config.yaml#L72); [c2g_env/physics/macro_grid.py:MacroGridSignal.clear_bid:536](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/macro_grid.py#L536)

None established: default 0.15 already matches the literal, and no evidence was inspected that the original Table 1 varied this knob.

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

## New issue introduced by the correction, not an original defect

### C01: Corrected high-level Box retains a finite-upper-bound sampling flag after high[8] becomes infinity

**Classification:** introduced_current_observation_space_metadata_and_sampler_bug. **Status:** Recorded without changing core code, frozen checkpoints, or published physical_v1 semantics..

Full equations, probes and test anchors: [the macro ledger](reports/original_macro_audit.json).

Confirmed from the current environment source, the installed Gymnasium Box construction/sampling logic, and the saved checkpoint preflight metadata. In-place high[8] mutation leaves a stale bounded_above flag, so the sampler attempts a finite-interval uniform draw with an infinite endpoint.

**Current source:** [c2g_env/env_high_level.py:C2GMacroEnv.__init__:158](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L158); [c2g_env/env_high_level.py:C2GMacroEnv.__init__:166](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L166); [c2g_env/env_high_level.py:C2GMacroEnv.__init__:209](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_high_level.py#L209)

Generic code that samples the high-level observation space can fail; finite observation membership and checkpoint inference can still work. The defect concerns observation_space.sample(), not evidence of an action-space or SAC gradient-update defect. The finite preflight demonstrates a working deterministic forward pass only, not full-episode qualification or Table 1 performance.

The midpoint (low+high)/2 is infinite for a genuinely upper-unbounded coordinate even when Box metadata is correct. Thus midpoint infinity is a limitation of that synthetic probe, not independent evidence of damaged SAC weights or an update-rule failure. The incorrect sampling flag is the actual implementation defect.

**Causal limits:** This finding was introduced between the original and corrected code endpoints and must not be added to the list of original SAC failures. No evidence here shows that the observation-space sampler was used in historical SAC updates or that this metadata inconsistency caused training degradation. No contribution to the historical PID MW difference or Table 1 algorithm gap is established.

The completed Table 1 study used real finite observations, not sampled
observation-space inputs. All 35 service trajectories completed. This
does not repair or excuse the separate Box sampling defect.

## Original SAC training integration and provenance

### TRAIN-001: Original hierarchical SAC does not load or freeze Phase-1 SAC

**Classification:** manuscript_protocol_vs_training_wiring. **Status:** fixed.

The original main always creates _make_rule_based_inner_fn(), which constructs RuleBasedController, and passes that callable to both training and evaluation C2GMacroEnv instances. There is no SAC.load or inner-checkpoint selection in this original trainer. It cannot implement the stated Phase-2 SAC-on-frozen-Phase-1-SAC protocol through its provided entrypoint.

**Original:** [baselines/train_hierarchical_sac.py:_make_rule_based_inner_fn:63](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_hierarchical_sac.py#L63); [baselines/train_hierarchical_sac.py:_make_macro_env:81](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_hierarchical_sac.py#L81); [baselines/train_hierarchical_sac.py:train_macro_sac:99](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_hierarchical_sac.py#L99); [baselines/train_hierarchical_sac.py:main:193](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_hierarchical_sac.py#L193); [README.md:Phase-2 benchmark example:815](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/README.md#L815)

**Corrected:** [baselines/train_hierarchical_sac.py:_make_sac_inner_fn:71](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_hierarchical_sac.py#L71); [baselines/train_hierarchical_sac.py:main:212](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_hierarchical_sac.py#L212); [baselines/train_hierarchical_sac.py:train_macro_sac:161](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_hierarchical_sac.py#L161)

Corrected/current require +hrl.inner_checkpoint or explicit +hrl.inner_controller=rule_based. The loaded hardware policy's SHA-256 fingerprint is embedded in the macro training context.

**Limits:** This proves what the checked-in original trainer does, not which unpublished or historical command produced the shipped macro ZIP. The macro ZIP lacks an inner-policy identity. Its actual historical training partner is unproven. Different training seeds for high and low learners are not themselves a pairing bug. The original PPO-macro path can load SAC; do not generalize this missing SAC-macro path into a claim that all original frozen-RL hierarchies were impossible.

Full equations, probes and test anchors: [the training ledger](reports/original_training_audit.json).

### TRAIN-002: Shipped low checkpoint's s100 label disagrees with its stored seed

**Classification:** checkpoint_provenance_label. **Status:** unfixed.

trained_models/BestCheckpoints/sac_default_s100/final_model.zip has SB3 data.seed = 200, not 100. Its recorded transition count is 215000 and target _total_timesteps is 400000. The macro s100 ZIP declares seed 100 and 400000 transitions.

**Original:** [conf/experiments.yaml:shipped SAC checkpoint paths:100](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/conf/experiments.yaml#L100); [README.md:shipped low checkpoint examples:806](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/README.md#L806); [baselines/train_sac.py:train seed forwarding:109](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_sac.py#L109); [baselines/train_rule_macro_sac.py:train seed forwarding:147](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_rule_macro_sac.py#L147); [baselines/train_hierarchical_sac.py:train_macro_sac seed forwarding:144](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_hierarchical_sac.py#L144)

**Corrected:** [baselines/sac_provenance.py:bind_training_context:20](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/sac_provenance.py#L20); [baselines/train_sac.py:train:113](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_sac.py#L113); [baselines/train_rule_macro_sac.py:train:151](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_rule_macro_sac.py#L151); [baselines/train_hierarchical_sac.py:train_macro_sac:158](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_hierarchical_sac.py#L158)

Unfixed for the preserved original artifact and directory name. New context records the actual model.seed; it does not retroactively relabel the old ZIP or enforce that path text matches seed.

**Limits:** Stored seed is the checkpoint's declared seed, not a reconstruction of every historical RNG state. A file named final_model inside BestCheckpoints may deliberately be a selected intermediate checkpoint. Its 215000 transition count does not prove training stopped early or that a 400000-step budget was never completed. Do not replace an intended seed-100 result with this model without disclosing the stored seed, fingerprint, and selection provenance.

Full equations, probes and test anchors: [the training ledger](reports/original_training_audit.json).

### TRAIN-003: Bare hierarchical SAC invocation consumes the default PPO configuration

**Classification:** wrong_child_learner_configuration. **Status:** fixed.

Original conf/config.yaml selects rule_macro_ppo. The original hierarchical SAC usage omits algo=sac, while main forwards cfg.algo without validating its name. Thus the documented default supplies PPO net_arch keys pi/vf to SAC, along with a 30000000-step budget, learning_rate 3e-5, batch_size 64 and fixed ent_coef 0.005 instead of the SAC configuration.

**Original:** [conf/config.yaml:defaults:21](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/conf/config.yaml#L21); [conf/algo/rule_macro_ppo.yaml:algorithm settings:1](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/conf/algo/rule_macro_ppo.yaml#L1); [conf/algo/sac.yaml:algorithm settings:1](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/conf/algo/sac.yaml#L1); [baselines/train_hierarchical_sac.py:documented default invocation:13](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_hierarchical_sac.py#L13); [baselines/train_hierarchical_sac.py:main:173](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_hierarchical_sac.py#L173); [baselines/train_hierarchical_sac.py:train_macro_sac:125](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_hierarchical_sac.py#L125); [baselines/train_sac.py:train:93](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_sac.py#L93); [baselines/train_rule_macro_sac.py:train:131](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_rule_macro_sac.py#L131)

**Corrected:** [baselines/train_hierarchical_sac.py:documented explicit algorithm:13](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_hierarchical_sac.py#L13); [baselines/train_hierarchical_sac.py:main:189](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_hierarchical_sac.py#L189); [baselines/train_sac.py:train:58](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_sac.py#L58); [baselines/train_rule_macro_sac.py:train:95](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_rule_macro_sac.py#L95)

Fixed by explicit algorithm guards and corrected hierarchical usage. The shared root default is still rule_macro_ppo; corrected trainers do not silently auto-switch it.

**Limits:** The correctly documented low-level commands already include algo=sac; those valid invocations are not affected. This is not a claim that entropy 0.005 or a different learning rate is intrinsically wrong. The bug is passing a different algorithm's configuration through an incompatible child-learner interface. A full SB3 constructor/training run was not executed.

Full equations, probes and test anchors: [the training ledger](reports/original_training_audit.json).

### TRAIN-004: Original hierarchical plant profile and all three trainers' device setting are dropped

**Classification:** configuration_not_forwarded. **Status:** fixed.

The original hierarchical macro factory accepts only scenario, seed and inner_fn, so a Hydra plant_profiles override never reaches its child fast environment. All three original SAC constructors omit device even though the shared configuration explicitly sets device: cpu; SB3 therefore uses its own default auto selection.

**Original:** [conf/config.yaml:device contract:35](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/conf/config.yaml#L35); [conf/plant_profiles/plant_100mw.yaml:plant override contract:1](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/conf/plant_profiles/plant_100mw.yaml#L1); [baselines/train_hierarchical_sac.py:_make_macro_env:81](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_hierarchical_sac.py#L81); [baselines/train_hierarchical_sac.py:main:167](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_hierarchical_sac.py#L167); [baselines/train_hierarchical_sac.py:train_macro_sac:128](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_hierarchical_sac.py#L128); [baselines/train_sac.py:train SAC constructor:93](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_sac.py#L93); [baselines/train_rule_macro_sac.py:train SAC constructor:131](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_rule_macro_sac.py#L131)

**Corrected:** [baselines/train_hierarchical_sac.py:_make_macro_env:93](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_hierarchical_sac.py#L93); [baselines/train_hierarchical_sac.py:main:212](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_hierarchical_sac.py#L212); [baselines/train_hierarchical_sac.py:train_macro_sac:158](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_hierarchical_sac.py#L158); [baselines/train_sac.py:train:113](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_sac.py#L113); [baselines/train_rule_macro_sac.py:train:151](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_rule_macro_sac.py#L151)

Fixed in corrected/current SAC entrypoints.

**Limits:** Original train_sac.py and train_rule_macro_sac.py already forward plant_overrides correctly; do not extend the hierarchical plant-profile defect to those low trainers. The default plant_profiles=none run is not changed by the missing nondefault profile. Ignoring device=cpu matters when auto selects a different device. No CPU/GPU numerical performance comparison was performed.

Full equations, probes and test anchors: [the training ledger](reports/original_training_audit.json).

### TRAIN-005: Original macro SAC uses a low-level metrics callback on incompatible macro infos

**Classification:** training_callback_schema_mismatch. **Status:** fixed.

train_macro_sac instantiates C2GMetricsCallback, whose accumulators expect tick, temp_A, temp_B, bess_soc and tracking_err_kw. Original C2GMacroEnv.step instead emits macro_tick, temp_A_max/temp_B_max, bess_soc_end and mean_tracking_err, with last_inner_info nested. The callback never unwraps those fields and silently logs zeros for several hardware metrics and episode length.

**Original:** [baselines/train_hierarchical_sac.py:train_macro_sac:119](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_hierarchical_sac.py#L119); [baselines/train_hierarchical_sac.py:train_macro_sac callback list:149](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_hierarchical_sac.py#L149); [c2g_env/env_high_level.py:C2GMacroEnv.step info_out:410](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L410); [baselines/metrics_callback.py:_accumulate_step:290](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/metrics_callback.py#L290); [baselines/metrics_callback.py:_compute_episode_metrics:334](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/metrics_callback.py#L334); [baselines/metrics_callback.py:C2GMetricsCallback._on_step:473](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/metrics_callback.py#L473)

**Corrected:** [baselines/train_hierarchical_sac.py:train_macro_sac:133](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/train_hierarchical_sac.py#L133); [baselines/macro_metrics_callback.py:C2GMacroMetricsCallback._on_step:22](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/macro_metrics_callback.py#L22); [tests/test_sac_contracts.py:test_macro_callback_weights_fast_steps_and_records_faults:60](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/tests/test_sac_contracts.py#L60)

Fixed for the inspected hierarchical SAC path by a separate macro-schema callback with fast-step-weighted error accumulation and explicit terminal flags.

**Limits:** This is a training-log defect. SB3's reward and replay transition are not replaced by the callback's diagnostic zeros. The best-model EvalCallback uses environment rewards; this finding alone does not establish a corrupted checkpoint-selection score. No claim is made that Table 1 was computed from this callback or that all other macro training entrypoints were corrected.

Full equations, probes and test anchors: [the training ledger](reports/original_training_audit.json).

### TRAIN-006: Original low training logger counts early failures after tick 287 as survival

**Classification:** training_callback_horizon_contract. **Status:** fixed.

_compute_episode_metrics declares survival whenever tick >= 287, independent of the configured 17280-tick episode horizon and termination cause. An early thermal/frequency/voltage failure late enough in an episode can therefore be recorded as survived.

**Original:** [c2g_env/config.yaml:global horizon:29](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/config.yaml#L29); [baselines/metrics_callback.py:_accumulate_step:300](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/metrics_callback.py#L300); [baselines/metrics_callback.py:_compute_episode_metrics:347](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/metrics_callback.py#L347); [baselines/train_sac.py:train metrics callback:84](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_sac.py#L84); [baselines/train_rule_macro_sac.py:train metrics callback:122](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_rule_macro_sac.py#L122)

**Corrected:** [baselines/metrics_callback.py:_accumulate_step:304](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/metrics_callback.py#L304); [baselines/metrics_callback.py:_compute_episode_metrics:351](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/baselines/metrics_callback.py#L351); [c2g_env/env_low_level.py:C2GFastEnv.step terminal flags:727](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L727); [c2g_env/env_low_level.py:C2GFastEnv.step physical metadata:753](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L753); [tests/test_sac_contracts.py:test_low_level_callback_does_not_count_early_fault_as_survival:125](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/tests/test_sac_contracts.py#L125)

Fixed for corrected physical_v1 training infos. The old >=287 fallback remains when episode_ticks is absent, including intentionally preserved legacy-style infos.

**Limits:** The episode termination/replay behavior itself is not altered by this logging calculation. Do not infer that the manuscript's independently computed survival column is wrong without the separate evaluation-driver evidence.

Full equations, probes and test anchors: [the training ledger](reports/original_training_audit.json).

### TRAIN-007: Original rule-macro training wrapper fabricates reset context and returns stale commitment

**Classification:** training_wrapper_reset_observation_contract. **Status:** fixed.

At reset, _build_macro_obs fabricates a zero hardware observation, puts 0.5 in index 7 while calling it grid load, and discards the real reset observation. Actual Fast.GRID_LOAD is index 8. reset then runs a handshake that can change committed_mw but returns the previously obtained observation unchanged. At subsequent macro boundaries, step changes commitment before consuming an already chosen hardware action.

**Original:** [baselines/train_rule_macro_sac.py:make_env_fn:73](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/train_rule_macro_sac.py#L73); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv.reset:69](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/train_envs/rule_macro_wrapped.py#L69); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv.step:87](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/train_envs/rule_macro_wrapped.py#L87); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv._build_macro_obs:139](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/train_envs/rule_macro_wrapped.py#L139); [c2g_env/obs_indices.py:Fast observation indices:25](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/obs_indices.py#L25); [c2g_env/env_high_level.py:C2GMacroEnv.macro_obs_from_reset:565](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_high_level.py#L565)

**Corrected:** [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv.reset:71](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/train_envs/rule_macro_wrapped.py#L71); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv.step:94](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/train_envs/rule_macro_wrapped.py#L94); [c2g_env/train_envs/rule_macro_wrapped.py:RuleMacroWrappedEnv._build_macro_obs:157](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/train_envs/rule_macro_wrapped.py#L157); [tests/test_physical_contract.py:test_rule_wrapper_observes_commitment_before_action_and_matches_macro:186](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/tests/test_physical_contract.py#L186)

Fixed for physical_v1; legacy_v0 explicitly retains the old observation schedule. This finding is limited to the training wrapper interface, not market-clearing or physics correctness.

**Limits:** No actual handshake, simulator step, or regulation-bias campaign was executed. This is an observation/wiring defect, not evidence of replay-buffer mutation or corrupted timeout storage. The original trainers did not promise bias-randomized training; lack of such coverage is treated separately as design, not as this bug.

Full equations, probes and test anchors: [the training ledger](reports/original_training_audit.json).

## Evaluation and reporting defects

### E01: Macro RMSE averaged errors before squaring

Original macro-side summaries computed RMS of window-mean errors, not fast-tick RMSE; partial windows were equally weighted. The separate hardware summary already aggregated per-tick environment-reported error. Missing producing records prevent proving which column supplied the archived Table 1.

**Original:** [evaluation/run_benchmark.py:run_macro_episode:603](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L603); [evaluation/run_sachigh_saclow_benchmark.py:run_hierarchical_episode:179](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_sachigh_saclow_benchmark.py#L179)

Fixed physical-mode aggregation uses tracking_error_sq_sum/sub_steps_run. Do not claim signed-error cancellation if the source error is absolute.

### E02: First BESS aging increment omitted

Aging was final_age minus the first post-step age, rather than reset age, multiplied by 10000. It is the simulator's aging fraction, not battery-throughput MWh.

**Original:** [evaluation/run_benchmark.py:run_episode:424](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L424); [evaluation/run_benchmark.py:run_macro_episode:626](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L626)

Still present in the generic runner. The new separate Table 1 collector must use reset-to-service-end aging.

### E03: No-inner row lacked fast metric collection

The generic benchmark installed the fast instrumentation only when an inner controller existed, leaving no-hardware low reward, warning and aging uncollected.

**Original:** [evaluation/run_benchmark.py:run_macro_episode:485](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L485)

Still present in the generic runner. A new explicit fixed-setpoint callback can collect the same physical fields without changing the plant.

### E04: Hardware SD field names contain an extra underscore

Output keys such as tracking_rmse__std use two underscores, so consumers expecting _std can miss hardware SDs.

**Original:** [evaluation/run_benchmark.py:benchmark:902](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L902)

Still present in the generic exporter; the new study uses explicit metric/SD keys.

### E05: Random-controller action RNG was not an episode-seeded stream

Global Python/NumPy/Torch seeding does not seed a Gym action space. The sampled action space belongs to a seed-0 construction environment, not each evaluated episode.

**Original:** [evaluation/run_benchmark.py:RandomAgent:250](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L250); [evaluation/run_benchmark.py:benchmark:687](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L687)

Physical resets seed action spaces, but the generic construction-space stream is still not an independent per-episode policy stream. The new study seeds its case-local random controller explicitly.

### E06: Stateful classical controllers carried state across episodes

The original generic runner reused controllers without reset. PID integrators and bang-bang hysteresis can therefore carry prior-episode state into another seed.

**Original:** [evaluation/run_benchmark.py:benchmark:726](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L726); [baselines/pid_controller.py:_PIDLoop:65](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/pid_controller.py#L65)

Current runner resets controllers exposing reset(); bang-bang has no reset method. The new study constructs fresh controllers per case.

### E07: Macro survival counted calls, not successful fast completion

Termination during the last macro call can still satisfy the macro-call count. Full survival requires all fast ticks and no physical termination.

**Original:** [evaluation/run_benchmark.py:run_macro_episode:618](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L618)

Still present in the generic runner. It does not prove that the archived 100% survival claim was wrong; original raw episodes are unavailable.

### E08: PPO normalization restoration was not fail-closed

An absent normalization pickle could leave PPO unnormalized; the original restore constructed a fast environment even for a macro PPO policy.

**Original:** [evaluation/run_benchmark.py:load_sb3_agent:295](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L295)

Corrected loader chooses the appropriate environment class. No original-bidding PPO artifact was found for this requested rerun; economic fractional PPO is not a substitute.

### E09: LLM warm-up filtering changes the requested cohort

Warm-up filtering can remove evaluation episodes while the reported requested count stays unchanged; excessive skipping falls back to all episodes.

**Original:** [evaluation/run_benchmark.py:benchmark:857](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L857)

Do not use this behavior to construct a matched five-case table. No LLM row is run in the authorized seven-row study.

### E10: Table 2's legacy calendar shim is not timestamp-faithful

Repeating each NYC source row sixty times assumes exact five-minute intervals. The selected peak dates contain 302/299/294 valid rows, and their first 288 rows end at 22:45/23:00/23:30. Workload phase remained zero, and bias was patched after the reset observation.

**Original:** [evaluation/grid_relief.py:_align_calendar:91](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/grid_relief.py#L91); [evaluation/grid_relief.py:_run_episode:165](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/grid_relief.py#L165)

Physical-mode replay uses actual timestamps, advances workload start day, and applies bias before reset. This particular shim belongs to Table 2, not the nominal Table 1 runner.

### E11: Table 2 prefix comparisons concealed incomplete horizons

The original evaluator used a common available prefix and recomputed stress quantiles on that prefix, while revenue remained an episode total.

**Original:** [evaluation/grid_relief.py:_metrics:257](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/grid_relief.py#L257)

Physical-mode evaluation rejects incomplete matched comparisons and suppresses incomplete aggregate estimates. Do not infer this altered Table 1 if all its runs completed.

### E12: Thermal-warning boundary conventions differ

The benchmark uses normalized observations with >=33 C; the callback uses actual info temperatures with >33 C. A fraction of warning ticks is not the fraction of episodes with any warning.

**Original:** [evaluation/run_benchmark.py:run_episode:407](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L407); [evaluation/run_benchmark.py:run_macro_episode:505](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L505); [baselines/metrics_callback.py:_compute_episode_metrics:301](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/baselines/metrics_callback.py#L301)

Still distinct. The new study explicitly records its actual-temperature boundary and both interpretations where available.

### E13: Original Table 1 provenance is incomplete

No original Table 1 episode manifests, exact command, matching full result CSV, PPO weights/normalizer, or LLM execution receipt was recovered. Older summary CSVs do not match the published 229 +/-18 kW. Original README says macro SAC 780 +/-26, whereas camera_ready.tex says 780 +/-626.

**Original:** [README.md:reported macro SAC result:1282](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/README.md#L1282); [evaluation/run_benchmark.py:benchmark:829](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/evaluation/run_benchmark.py#L829)

The discrepancy cannot be adjudicated without producing records. New results must be labelled a corrected Table 1-style comparison, not an exact reproduction of all archival numbers.

## Remaining physical/model limitations

### R1: PCC voltage still depends on gross facility power/reactive demand, not actual BESS net import.

**Classification:** remaining model-coupling limitation. **Status:** see scoped qualification.

**Original:** [c2g_env/env_low_level.py:C2GFastEnv.step:538](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L538); [c2g_env/physics/electrical.py:DatacenterElectrical.step:325](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/electrical.py#L325)

**Corrected:** [c2g_env/env_low_level.py:C2GFastEnv.step:606](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L606); [c2g_env/physics/electrical.py:DatacenterElectrical.step:325](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/electrical.py#L325)

Battery support cannot influence the voltage penalty/fault through this interface. Frequency deficit does include BESS; it is incorrect to claim the entire grid model ignores battery response.

**Evidence:** Same fresh plant/action except BESS=0 versus +5 MW actual discharge: v_pcc is 0.9935073774422036 pu in both cases in both revisions. Corrected net import drops 5 MW and tracking response rises 5000 kW, but voltage does not change.

Full equations, probes and test anchors: [the physical ledger](reports/original_physical_audit.json).

### R2: Reported max_charge_mw uses a 0.05 SOC window; actual charge derating uses 0.10.

**Classification:** remaining BESS diagnostic inconsistency, not phantom tracking credit. **Status:** Unfixed in all three revisions..

**Original:** [c2g_env/physics/bess.py:_SimpleBESSModel.step:139](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/bess.py#L139)

**Corrected:** [c2g_env/physics/bess.py:_SimpleBESSModel.step:139](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/bess.py#L139)

The limit diagnostic overstates usable charging near SOC_MAX. The directly audited low-level tracking/reward path uses actual_power_mw and does not consume max_charge_mw, so no direct Table1 impact is established.

**Evidence:** At SOC=0.90, querying with zero power reports max_charge_mw approximately 5; immediately commanding that charge produces approximately -2.5 MW.

Full equations, probes and test anchors: [the physical ledger](reports/original_physical_audit.json).

### R3: Corrected work is a dynamic-power-derived energy proxy, not a job/FLOP completion model; queues reset at episode boundaries and legacy semantics remain selectable.

**Classification:** remaining deliberate abstractions and compatibility behavior. **Status:** Design limitations/compatibility choices, not newly discovered SAC algorithm defects..

**Original:** [c2g_env/physics/workload.py:WorkloadOrchestrator.__init__:182](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/workload.py#L182); [c2g_env/physics/workload.py:WorkloadOrchestrator.reset:276](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/workload.py#L276)

**Corrected:** [c2g_env/physics/workload.py:WorkloadOrchestrator.__init__:156](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/workload.py#L156); [c2g_env/physics/workload.py:WorkloadOrchestrator.step:248](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/workload.py#L248); [c2g_env/physics/workload.py:WorkloadOrchestrator.reset:290](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/physics/workload.py#L290); [c2g_env/env_low_level.py:C2GFastEnv.step:649](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L649)

There is a per-step backlog penalty, but no terminal clearance/payment in the audited low-level reward. Within-episode conservation does not prove service debt is repaid after reset. Holding a 300-second rate across 5-second steps is not itself a 60x work-arrival bug when integrated with dt.

Full equations, probes and test anchors: [the physical ledger](reports/original_physical_audit.json).

### R4: Original BESSModel auto-selects PySAM when importable; corrected physical_v1 explicitly selects _SimpleBESSModel. The PySAM adapter lacks the _age_frac attribute read by low-level info construction.

**Classification:** conditional original backend/interface issue; source-verified, optional backend not executed. **Status:** Physical_v1 avoids this optional path; adapter and legacy path remain unchanged. Do not attribute this condition to reported SAC results without runtime evidence..

**Original:** [c2g_env/physics/bess.py:backend selection:54](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/bess.py#L54); [c2g_env/physics/bess.py:_SimpleBESSModel:87](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/bess.py#L87); [c2g_env/physics/bess.py:_PySAMBESSModel:212](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/physics/bess.py#L212); [c2g_env/env_low_level.py:C2GFastEnv.step:675](https://github.com/HewlettPackard/dc2grid/blob/7550013331a865fa45684cb7cd973279312d4678/c2g_env/env_low_level.py#L675)

**Corrected:** [c2g_env/env_low_level.py:C2GFastEnv.reset:380](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L380); [c2g_env/env_low_level.py:C2GFastEnv.step:751](https://github.com/HewlettPackard/dc2grid/blob/repro/20261102/tracking-baseline/c2g_env/env_low_level.py#L751)

Adapters declare different capacities (simple: 15 MWh/5 MW; PySAM-labelled: 150 MWh/50 MW). Native PySAM units/runtime compatibility and historical backend provenance were not established. If the PySAM adapter reaches info construction, the missing _age_frac is incompatible with that access.

Full equations, probes and test anchors: [the physical ledger](reports/original_physical_audit.json).

## Claims that this audit does not support

- Original simple-BESS tracking already used actual, SOC-limited battery output.
- Original fast reward already used the RegD signal observed by the action;
  the confirmed signal-time mismatch was in returned transition telemetry.
- The 215k selected checkpoint versus a 400k training budget is not proof of
  undertraining; the seed200-versus-s100 artifact label is the proven mismatch.
- Source wiring proves what the checked-in SAC macro trainer does, not the
  unknown actual training partner of an unmarked historical model ZIP.
- No explaining sixfold UTD, replay/timeout corruption, or NaN mechanism was
  established in the previous tested instability studies.
- Gamma0.99 at five seconds gives an effective discounted horizon near8.3min.
  That and observation aliasing/reward-scale choices are design limitations,
  not by themselves incorrect SAC updates.
- Follow-up analysis-column collisions, SB3 kwargs mutation/provenance checks
  and final-post-optimizer evaluation changes must not be relabelled as
  original code defects. Their separate history is in the training ledger.

## Authorized Table 1 follow-up

The user chose existing corrected models only. A separate seven-row, five-case
comparison is complete: classical rule-macro baselines, selected low SAC1007
(250002 transitions), and selected macro SAC1007 (8004 transitions) paired with
that exact low checkpoint. No new training, economic-policy substitution or
LLM call is authorized. PPO/LLM rows remain not rerun, not filled with legacy
values inside a corrected table.

The reconstructed defaults are day0, zero bias and seeds100-104. They are
documented source defaults, not proven identities of the archived producing
runs. Day0 is a training start date, not proof of exact-cell or trajectory-level
in-distribution status. All main metrics use 24-hour service;
two-hour recovery and stronger physical qualification are separately reported.
Remaining frozen-physics limitations are disclosed, not silently patched in
an evaluation claimed to match the published Table2 checkpoint/protocol.

All 35 service cases completed; 31 passed the stronger supplemental gates.
The four failures are service-end SOC below 0.20 for the learned macro
paired with SAC low. All cases, including those four, closed two-hour
recovery. No failed case is dropped from the means.

The recorded training and native Table 1 configurations have identical
hashes. The Table 2 recovery configuration differs only in the three
relative-to-absolute data-directory strings, not market, plant or reward
settings. This rules out a documented static configuration mismatch, not
every possible runtime or state-distribution difference. See
[the exact configuration comparison](reports/training_configuration_comparison.json).

The new low SAC has lower mean tracking RMSE than PID, but slightly lower
revenue, greater aging and lower recovery-inclusive cash on this default
panel. The learned macro rejects every bid and is not an improved bidding
result. These findings do not contradict the separate biased peak-date
Table 2 comparison or identify the cause of the macro policy's behavior.

See [the audited numerical results](reports/table1_report.json),
[the standalone comparison PDF](publication/table1_corrected_comparison.pdf),
and [the study README](README.md) for definitions, scope and reproduction.

Original manuscripts, published releases and previous studies remain untouched.
