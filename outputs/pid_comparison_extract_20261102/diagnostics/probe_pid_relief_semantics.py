"""Read-only reproduction of the archived PID comparison in both simulator modes."""

import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
PROJECT = Path("/workspace/IBM/temp/dc2grid-repos/dc2grid")
OUT = Path(__file__).parent / "pid_relief_attribution_20261102"
os.environ["MPLCONFIGDIR"] = str(OUT / "matplotlib")
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
from evaluation.grid_relief import _market_config, _metrics, _run_episode


def initialize(queue):
    import torch
    os.sched_setaffinity(0, {queue.get()})
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def episode(mode, hardware, day, seed, config):
    directory = OUT / mode / f"d{day}_s{seed}" / hardware
    _run_episode(
        hw=hardware, scenario="default", seed=seed, model_dir=None,
        macro="rule_macro", config_path=config, start_day=day,
        regd_bias=0.5, trace_dir=str(directory), simulation_mode=mode,
    )
    return str(directory)


def frame(directory):
    data = pd.read_csv(directory / "hardware.csv.gz")
    context = json.loads((directory / "episode.json").read_text())
    data.attrs.update(context["metric_context"])
    return data


def main():
    if OUT.exists():
        raise FileExistsError("Never replace an existing diagnostic reproduction")
    OUT.mkdir()
    configs = {}
    for mode in ("legacy_v0", "physical_v1"):
        directory = OUT / mode
        directory.mkdir()
        configs[mode] = _market_config("nyiso_nyc", str(directory), mode)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in range(8):
        queue.put(cpu)
    with ProcessPoolExecutor(
        max_workers=8, mp_context=context, initializer=initialize, initargs=(queue,),
    ) as pool:
        futures = [
            pool.submit(episode, mode, hardware, day, seed, configs[mode])
            for mode in configs for day in (207, 248, 249)
            for seed in range(100, 105) for hardware in ("pid", "none")
        ]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f"Completed {count}/{len(futures)} diagnostic episodes", flush=True)
    queue.close()
    queue.join_thread()
    results = []
    for mode in configs:
        for day in (207, 248, 249):
            for seed in range(100, 105):
                root = OUT / mode / f"d{day}_s{seed}"
                agent, baseline = frame(root / "pid"), frame(root / "none")
                metrics = _metrics(agent, baseline)
                load = agent.load_mw.to_numpy()
                threshold = agent.attrs.get("stress_threshold_mw", np.quantile(load, 0.9))
                stress = load >= threshold
                facility = baseline.p_fac_mw.to_numpy() - agent.p_fac_mw.to_numpy()
                cooling = baseline.p_cooling_mw.to_numpy() - agent.p_cooling_mw.to_numpy()
                actual_pcc = facility + agent.bess_actual_kw.to_numpy() / 1000
                demanded = agent.dp_dem_kw.to_numpy() / 1000
                result = {
                    "mode": mode, "day": day, "seed": seed,
                    "agent_ticks": len(agent), "baseline_ticks": len(baseline),
                    "reported_formula_relief_mw": float(metrics["dP_stress_mw"]),
                    "stress_cooling_relief_mw": float(cooling[stress].mean()),
                    "stress_other_facility_relief_mw": float((facility - cooling)[stress].mean()),
                    "stress_nominal_facility_mw": float(baseline.p_fac_mw.to_numpy()[stress].mean()),
                    "stress_actual_facility_mw": float(agent.p_fac_mw.to_numpy()[stress].mean()),
                    "stress_demanded_response_mw": float(demanded[stress].mean()),
                    "stress_true_metered_error_mw": float((actual_pcc - demanded)[stress].mean()),
                    "stress_battery_discharge_mw": float(agent.bess_actual_kw.to_numpy()[stress].mean() / 1000),
                    "stress_observed_regd": float(agent.regd_observed.to_numpy()[stress].mean()),
                    "stress_actual_commitment_mw": float(agent.committed_actual_mw.to_numpy()[stress].mean()),
                    "stress_observed_commitment_mw": float(agent.committed_observed_mw.to_numpy()[stress].mean()),
                    "stress_hvac_effort": float(agent.hvac_effort.to_numpy()[stress].mean()),
                    "stress_pump_speed": float(agent.pump_speed_A.to_numpy()[stress].mean()),
                    "stress_temp_A": float(agent.temp_A.to_numpy()[stress].mean()),
                    "stress_temp_B": float(agent.temp_B.to_numpy()[stress].mean()),
                    "stress_mean_grid_load_mw": float(load[stress].mean()),
                    "stress_tick_count": int(stress.sum()),
                }
                results.append(result)
    summary = {}
    for mode in configs:
        rows = [row for row in results if row["mode"] == mode]
        summary[mode] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0] if key not in {"mode", "day", "seed"}
        }
        summary[mode]["relief_case_sd_mw"] = float(np.std(
            [row["reported_formula_relief_mw"] for row in rows], ddof=1,
        ))
    (OUT / "summary.json").write_text(json.dumps({
        "role": "diagnostic_reproduction_not_new_publication_evidence",
        "cases": results, "summary": summary,
        "no_training_or_gain_changes": True,
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
