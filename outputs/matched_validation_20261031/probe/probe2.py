import os,json
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from c2g_env.env_high_level import C2GMacroEnv
res={}
def mk(inner,bias=0.0,seed=90001,day=196):
    e=C2GMacroEnv(scenario="default",simulation_mode="physical_v1",start_day=day,
                  regd_load_bias=bias,inner_action_fn=inner); e.reset(seed=seed); return e

# (A) reference path-independence across different inner policies
def polA(o,a): return np.array([1.0,0.7,0.7,0.0],dtype=np.float32)
def polB(o,a): return np.array([0.0,0.15,0.0,1.0],dtype=np.float32)   # aggressive shed
def trace(pol,nmac=6,bid=(0.6,0.6)):
    e=mk(pol); recs=[]
    e._sub_step_callback=lambda p,a,o,r,d,i: recs.append(
        (i["p_reference_mw"],i["p_facility_mw"],i["temp_A"],i["temp_B"],i["backlog_kw"],
         i["f_grid_hz"],i["p_it_A_mw"],i["p_chiller_mw"],i["p_hvac_mw"],i["p_pump_mw"]))
    out=[]
    for k in range(nmac):
        _,_,te,tr,inf=e.step(np.array(bid,dtype=np.float32)); out.append(inf)
        if te or tr: break
    return recs,out,e
rA,iA,eA=trace(polA); rB,iB,eB=trace(polB)
n=min(len(rA),len(rB))
res["p_reference_max_abs_diff_MW"]=float(max(abs(rA[i][0]-rB[i][0]) for i in range(n)))
res["p_facility_max_abs_diff_MW"]=float(max(abs(rA[i][1]-rB[i][1]) for i in range(n)))
res["polB_temp_A_end"]=rB[n-1][2]; res["polA_temp_A_end"]=rA[n-1][2]
res["polB_temp_B_end"]=rB[n-1][3]; res["polA_temp_B_end"]=rA[n-1][3]
res["polB_backlog_kw_end"]=rB[n-1][4]
res["T_safe"]=float(eA._fast_env._thermal.T_safe)
res["polB_relief_MW_mean"]=float(np.mean([rA[i][0]-rB[i][1] for i in range(n)]))

# (B) frequency dynamics / early-termination risk
g=eA._fast_env._grid
res["H_inertia"]=float(g._H_inertia); res["D_damping"]=float(g._D_damping)
res["f_noise_std"]=float(g._f_noise_std); res["load_median_mw"]=float(g._load_median)
res["episode_ticks"]=int(eA._fast_env._episode_ticks)
res["episode_macro_ticks"]=int(eA._episode_macro_ticks)
# do-nothing levers + max bid -> large persistent deficit
e=mk(polA); term=None; fs=[]
for k in range(96):
    _,_,te,tr,inf=e.step(np.array([1.0,1.0],dtype=np.float32))
    fs.append(inf["last_inner_info"]["f_grid_hz"])
    if te or tr: term=(k,te,tr,inf["committed_mw"],inf["last_inner_info"]["freq_fault"],
                       inf["last_inner_info"]["thermal_warning"]); break
res["maxbid_defaults_stop"]=term; res["maxbid_f_min"]=float(min(fs)); res["maxbid_f_max"]=float(max(fs))
res["maxbid_macro_steps_run"]=len(fs)

# (C) thermal debt energy equivalence constants
th=eA._fast_env._thermal
res["C_A_MJ_per_C"]=th.C_A; res["C_B_MJ_per_C"]=th.C_B
res["COP_liq"]=float(th.cop_liquid()); res["COP_air"]=float(th.cop_air())
res["MWh_elec_per_degC_zoneA"]=float(th.C_A/3600.0/th.cop_liquid())
res["MWh_elec_per_degC_zoneB"]=float(th.C_B/3600.0/th.cop_air())
res["p_cool_nominal_mw_property"]=float(th.p_cool_nominal_mw)
res["reference_cooling_actual_hvac_pump_mw"]=float(0.7*th.max_hvac_mw+th.P_PUMP_MAX_MW*0.7)
print(json.dumps(res,indent=1,default=str))
