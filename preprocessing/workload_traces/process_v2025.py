# preprocessing/workload_traces/process_v2025.py

import pandas as pd
import numpy as np
import os

# Configuration
TICK_SECONDS = 300 
RAW_DATA_PATH = "data/raw/workload_traces/v2025/disaggregated_DLRM_trace.csv"
OUTPUT_PATH = "data/processed/workload_traces/dlrm_v2025.csv"

def preprocess_v2025():
    print(f"Loading Alibaba v2025 DLRM data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)

    # 1. Focus on Heterogeneous GPU Nodes (HN)
    # These are the primary power consumers in our model
    df_hn = df[df['role'] == 'HN'].copy()

    # 2. Normalize Timestamps
    # Filter out NaNs in scheduled/deletion times for this baseline
    df_hn = df_hn.dropna(subset=['scheduled_time', 'deletion_time'])
    
    start_time = df_hn['scheduled_time'].min()
    df_hn['start_tick'] = ((df_hn['scheduled_time'] - start_time) // TICK_SECONDS).astype(int)
    df_hn['end_tick'] = ((df_hn['deletion_time'] - start_time) // TICK_SECONDS).astype(int)

    # Ensure ticks are non-negative
    df_hn['start_tick'] = df_hn['start_tick'].clip(lower=0)
    df_hn['end_tick'] = df_hn['end_tick'].clip(lower=0)

    max_tick = df_hn['end_tick'].max()
    print(f"Total timeline duration: {max_tick} ticks")

    # 3. Efficient Concurrency Calculation using Difference Arrays
    # We create arrays to track the 'entry' and 'exit' of resources
    gpu_diff = np.zeros(max_tick + 2)
    cpu_diff = np.zeros(max_tick + 2)
    mem_diff = np.zeros(max_tick + 2)

    for _, row in df_hn.iterrows():
        s, e = int(row['start_tick']), int(row['end_tick'])
        if e >= s:
            gpu_diff[s] += row['gpu_request']
            gpu_diff[e + 1] -= row['gpu_request']
            
            cpu_diff[s] += row['cpu_request']
            cpu_diff[e + 1] -= row['cpu_request']
            
            mem_diff[s] += row['memory_request']
            mem_diff[e + 1] -= row['memory_request']

    # 4. Prefix Sum to get concurrent values at each tick
    concurrent_gpu = np.cumsum(gpu_diff)[:-1]
    concurrent_cpu = np.cumsum(cpu_diff)[:-1]
    concurrent_mem = np.cumsum(mem_diff)[:-1]

    # 5. Build Output DataFrame
    dlrm_series = pd.DataFrame({
        'tick': range(len(concurrent_gpu)),
        'active_gpu_count': concurrent_gpu,
        'active_cpu_cores': concurrent_cpu,
        'active_mem_gib': concurrent_mem
    })

    # 6. Trim to exactly 30 full days (30 * 288 = 8640 ticks)
    target_ticks = 30 * 288
    if len(dlrm_series) > target_ticks:
        dlrm_series = dlrm_series.iloc[:target_ticks].copy()

    # 7. Cyclic Smoothing (The Bridge)
    # Ensure Day 30 connects smoothly back to Day 1
    print(f"Creating cyclic bridge for the end of Day 30...")
    bridge_size = 12 
    last_idx = target_ticks - 1
    
    for col in ['active_gpu_count', 'active_cpu_cores', 'active_mem_gib']:
        val_start = dlrm_series.loc[0, col]
        val_end = dlrm_series.loc[last_idx - bridge_size, col]
        # Create the linear ramp
        bridge = np.linspace(val_end, val_start, num=bridge_size)
        dlrm_series.loc[last_idx - bridge_size + 1 :, col] = bridge

    # 8. Clean up and Save
    dlrm_series = dlrm_series.fillna(0)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    dlrm_series.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! Saved concurrent DLRM load to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_v2025()