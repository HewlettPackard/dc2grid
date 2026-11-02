# preprocessing/workload_traces/process_v2023.py

import pandas as pd
import numpy as np
import os

# Configuration
TICK_SECONDS = 300  # 5 Minutes
RAW_DATA_PATH = "data/raw/workload_traces/v2023/openb_pod_list_default.csv"
OUTPUT_PATH = "data/processed/workload_traces/batch_v2023.csv"

def preprocess_v2023():
    print(f"Loading raw Alibaba v2023 data...")
    df = pd.read_csv(RAW_DATA_PATH)

    # 1. Filter for Batch Workloads (Burstable and Best Effort)
    df = df[df['qos'].isin(['Burstable', 'BE'])]

    # 2. Use Absolute Ticks
    global_start = df['creation_time'].min()
    df['abs_tick'] = ((df['creation_time'] - global_start) // TICK_SECONDS).astype(int)
    
    # 3. Calculate Duration in Ticks
    df = df.dropna(subset=['deletion_time', 'creation_time'])
    df['duration_ticks'] = np.ceil((df['deletion_time'] - df['creation_time']) / TICK_SECONDS).astype(int)
    
    # 4. Aggregate Resource Requests
    df['total_gpu_milli'] = df['num_gpu'] * 1000 + df['gpu_milli'].fillna(0)

    # 5. Group by Tick
    batch_series = df.groupby('abs_tick').agg(
        gpu_milli_request=('total_gpu_milli', 'sum'),
        avg_duration_ticks=('duration_ticks', 'mean')
    ).reset_index()
    
    batch_series.rename(columns={'abs_tick': 'tick'}, inplace=True)

    # 6. --- COLD START REMOVAL ---
    # We remove the first 7 days (7 * 288 = 2016 ticks) to remove the flat behavior
    print("Removing the first 7 days (Cold Start)...")
    ticks_to_remove = 7 * 288
    batch_series = batch_series[batch_series['tick'] >= ticks_to_remove].copy()
    
    # Reset tick index to start at 0
    batch_series['tick'] = batch_series['tick'] - ticks_to_remove
    
    # 7. Trim to exactly 33 days (Original 40 - 7 = 33)
    # 33 days * 288 = 9504 ticks
    target_ticks = 33 * 288
    all_ticks = pd.DataFrame({'tick': range(target_ticks)})
    batch_series = all_ticks.merge(batch_series, on='tick', how='left').fillna(0)

    # 8. Cyclic Smoothing (The Bridge) at the end of Day 33
    print(f"Creating cyclic bridge for the end of Day 33...")
    bridge_size = 12 
    last_idx = target_ticks - 1
    
    for col in ['gpu_milli_request', 'avg_duration_ticks']:
        val_start = batch_series.loc[0, col]
        val_end = batch_series.loc[last_idx - bridge_size, col]
        bridge = np.linspace(val_end, val_start, num=bridge_size)
        batch_series.loc[last_idx - bridge_size + 1 :, col] = bridge

    # 9. Save to Processed
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    batch_series.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! Processed {len(batch_series)} ticks (33 days). Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_v2023()