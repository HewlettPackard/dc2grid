import os, json, math
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import numpy as np
from c2g_env.env_high_level import C2GMacroEnv

DAY=196
def mk(bias, seed, inner=None):
    e=C2GMacroEnv(scenario="default", simulation_mode="physical_v1",
                  start_day=DAY, regd_load_bias=bias, inner_action_fn=inner)
    e.reset(seed=seed)
    return e

AUX = 0.5+1.5  # p_lighting_mw + p_network_mw (constant, NOT logged)
res={}

# --- Run A: default levers (1.0,0.7,0.7,0.0) via inner_action_fn ---
def defaults(obs,a): return np.array([1.0,0.7,0.7,0.0],dtype=np.float32)
subs=[]
e=mk(0.0,90001,inner=defaults)
macro=[]
for k in range(8):
    cb=[]
    e._sub_step_callback=lambda p,a,o,r,d,i: cb.append(i)
    o,rw,te,tr,inf=e.step(np.array([0.6,0.6],dtype=np.float32))
    macro.append((inf,cb))
    if te or tr: break

# identity checks on fast infos
fac_err=[];grid_err=[];rel_err=[]
for inf,cb in macro:
    for i in cb:
        comp=(i["p_it_A_mw"]+i["p_it_B_mw"]+i["p_chiller_mw"]+i["p_hvac_mw"]+i["p_pump_mw"]
              +i["p_conversion_loss_mw"]+AUX)
        fac_err.append(abs(comp-i["p_facility_mw"]))
        grid_err.append(abs((i["p_facility_mw"]-i["bess_actual_kw"]/1000.0)-i["p_grid_import_mw"]))
        rel_err.append(abs(i["facility_relief_kw"]))
res["P_fac_identity_max_abs_MW"]=max(fac_err)
res["P_grid_identity_max_abs_MW"]=max(grid_err)
res["defaults_facility_relief_max_abs_kW"]=max(rel_err)
res["n_fast_steps_checked"]=len(fac_err)

# --- settlement linearity test ---
lin=[]
for inf,cb in macro:
    C=inf["committed_mw"]; E=inf["mean_tracking_err"]; h=inf["settlement_hours"]; rate=inf["rate"]
    pred=rate*h*max(0.0, C-E/1000.0)
    lin.append(abs(pred-inf["regulation_revenue"]))
res["revenue_linear_form_max_abs_USD"]=max(lin)
res["sample_macro"]=[{k:float(macro[0][0][k]) for k in
    ("committed_mw","rate","perf_score","mean_tracking_err","regulation_revenue",
     "electricity_cost","settlement_hours","bid_mw")}]
res["macro_floor_kw"]=float(e._fast_env._macro_tracking_floor_kw)
res["committed_max_mw"]=float(e._committed_max_mw); res["dr_baseline_mw"]=float(e._dr_baseline_mw)

# --- electricity cost unit check ---
inf0,cb0=macro[0]
res["elec_cost_recompute"]=float(sum(i["lmp"]*i["p_grid_import_mw"]*i["duration_hours"] for i in cb0))
res["elec_cost_logged"]=float(inf0["electricity_cost"])

# --- BESS energy balance + nominal vs usable ---
b=e._fast_env._bess
res["bess_backend"]=type(b).__name__; res["E_NOM_MWH"]=float(b.E_NOM_MWH); res["P_MAX_MW"]=float(b.P_MAX_MW)
res["soc_frac"]=float(b.soc_fraction); res["soc_mwh_property"]=float(b.soc_mwh)
res["nominal_stored_MWh"]=float(b.soc_fraction*b.E_NOM_MWH); res["age_frac"]=float(b._age_frac)

# standalone BESS round-trip balance
from c2g_env.physics.bess import _SimpleBESSModel
m=_SimpleBESSModel(300.0); s0=m.soc_fraction*m.E_NOM_MWH; out=0.0; inn=0.0
rng=np.random.default_rng(0)
for t in range(200):
    p=float(rng.uniform(-5,5)); r=m.step(p); ap=r["actual_power_mw"]
    if ap>0: out+=ap*(300/3600.)
    else: inn+=-ap*(300/3600.)
s1=m.soc_fraction*m.E_NOM_MWH
res["bess_internal_delta_MWh"]=float(s1-s0)
res["bess_terminal_out_MWh"]=float(out); res["bess_terminal_in_MWh"]=float(inn)
res["bess_naive_balance_resid_MWh"]=float((s1-s0)-(inn-out))  # nonzero => losses, expected
res["bess_soc_mwh_vs_nominal_gap_MWh"]=float(m.soc_mwh-m.soc_fraction*m.E_NOM_MWH)

# --- queue balance invariant ---
arr=sum(i["work_arrived_kwh"] for _,cb in macro for i in cb)
srv=sum(i["work_served_kwh"] for _,cb in macro for i in cb)
bl=macro[-1][1][-1]["backlog_kw"]*(5/3600.)
res["queue_arrived_kwh"]=float(arr); res["queue_served_kwh"]=float(srv)
res["queue_terminal_backlog_kwh_equiv"]=float(bl)
res["queue_balance_resid_kwh"]=float(arr-srv-bl)

# --- matched-bias exogenous stream check (bias 0 vs 0.5, same seed) ---
def exo(bias):
    ee=mk(bias,90001,inner=defaults); recs=[]
    ee._sub_step_callback=lambda p,a,o,r,d,i: recs.append((i["lmp"],i["regd_signal"],i["grid_load_mw"]))
    mi=[]
    for k in range(4):
        _,_,_,_,inf=ee.step(np.array([0.6,0.6],dtype=np.float32)); mi.append((inf["rmcp_usd"],inf["bid_accepted"],inf["residual_mw"]))
    return recs,mi
r0,m0=exo(0.0); r5,m5=exo(0.5)
res["bias_lmp_identical"]=bool(all(abs(a[0]-b[0])<1e-12 for a,b in zip(r0,r5)))
res["bias_load_identical"]=bool(all(abs(a[2]-b[2])<1e-12 for a,b in zip(r0,r5)))
res["bias_rmcp_identical"]=bool(all(abs(a[0]-b[0])<1e-12 for a,b in zip(m0,m5)))
res["bias_regd_mean_0"]=float(np.mean([x[1] for x in r0]))
res["bias_regd_mean_05"]=float(np.mean([x[1] for x in r5]))
res["bias_regd_sat_frac_05"]=float(np.mean([abs(x[1])>=0.999 for x in r5]))

# --- IT saturation headroom (inverse-util clipping) ---
itA=[i["p_it_A_mw"] for _,cb in macro for i in cb]; itB=[i["p_it_B_mw"] for _,cb in macro for i in cb]
res["it_A_min_max_MW"]=[float(min(itA)),float(max(itA))]; res["it_A_floor_ceiling_MW"]=[16.0,150.0]
res["it_B_min_max_MW"]=[float(min(itB)),float(max(itB))]; res["it_B_floor_ceiling_MW"]=[10.0,100.0]
res["p_facility_range_MW"]=[float(min(i["p_facility_mw"] for _,cb in macro for i in cb)),
                            float(max(i["p_facility_mw"] for _,cb in macro for i in cb))]
print(json.dumps(res,indent=1,default=str))
