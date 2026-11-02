# preprocessing/workload_traces/process_v2026.py

import pandas as pd
import numpy as np
import os

# Configuration aligned with README.md
TICK_SECONDS = 300
RAW_DIR = "data/raw/workload_traces/v2026_genai/"
QPS_PATH = os.path.join(RAW_DIR, "qps.csv")
DUTY_CYCLE_PATH = os.path.join(RAW_DIR, "pod_gpu_duty_cycle_anon.csv")
OUTPUT_PATH = "data/processed/workload_traces/genai_v2026.csv"

def preprocess_v2026():
    print("Loading Alibaba v2026 GenAI traces...")
    
    # 1. Process QPS
    print(f"Reading {QPS_PATH}...")
    df_qps = pd.read_csv(QPS_PATH)
    
    # Calculate absolute tick
    df_qps['abs_tick'] = (df_qps['timestamp_anon'] // TICK_SECONDS).astype(int)
    
    # Aggregate: Total QPS across the whole cluster per tick
    qps_agg = df_qps.groupby('abs_tick').agg(
        total_qps=('value', 'sum')
    ).reset_index()

    # 2. Process GPU Duty Cycle
    print(f"Reading {DUTY_CYCLE_PATH}...")
    df_duty = pd.read_csv(DUTY_CYCLE_PATH)
    
    # Calculate absolute tick
    df_duty['abs_tick'] = (df_duty['timestamp_anon'] // TICK_SECONDS).astype(int)
    
    # Aggregate: Average Duty Cycle across all active GenAI pods per tick
    duty_agg = df_duty.groupby('abs_tick').agg(
        avg_gpu_duty_cycle=('value', 'mean'),
        active_genai_pods=('container_ip', 'nunique') # How many pods are reporting
    ).reset_index()

    # 3. Merge the datasets on absolute tick
    print("Merging QPS and Duty Cycle data...")
    # Outer join to ensure we don't drop ticks where one metric might be missing
    genai_series = pd.merge(qps_agg, duty_agg, on='abs_tick', how='outer').fillna(0)
    
    # Sort by time
    genai_series = genai_series.sort_values('abs_tick').reset_index(drop=True)
    
    # Shift to start at Tick 0
    min_tick = genai_series['abs_tick'].min()
    genai_series['tick'] = genai_series['abs_tick'] - min_tick
    genai_series.drop(columns=['abs_tick'], inplace=True)
    
    # Ensure ticks are contiguous
    max_tick = genai_series['tick'].max()
    all_ticks = pd.DataFrame({'tick': range(int(max_tick) + 1)})
    genai_series = all_ticks.merge(genai_series, on='tick', how='left').fillna(0)
    
    # Assuming genai_series currently has 274 rows (0 to 273)
    target_ticks = 288
    current_len = len(genai_series)

    if current_len < target_ticks:
        print(f"Applying cyclic smoothing from {current_len} to {target_ticks} ticks...")
        
        # 1. Create a full 288-tick index
        full_index = pd.DataFrame({'tick': range(target_ticks)})
        genai_series = full_index.merge(genai_series, on='tick', how='left')
        
        # 2. To ensure a smooth "wrap-around", interpolate ALL metrics
        # Added 'active_genai_pods' to this list!
        cols_to_fix = ['total_qps', 'avg_gpu_duty_cycle', 'active_genai_pods']
        
        for col in cols_to_fix:
            if col not in genai_series.columns:
                continue
                
            val_start = genai_series.loc[0, col]
            val_end = genai_series.loc[current_len - 1, col]
            
            # Create a linear bridge for the missing indices [274 ... 287]
            bridge = np.linspace(val_end, val_start, num=(target_ticks - current_len + 2))
            
            # Fill the NaNs with the bridge values (excluding the first and last which overlap)
            genai_series.loc[current_len:, col] = bridge[1:-1]

        print("Success: GenAI trace expanded with smooth cyclic transition.")

    # Safety: Fill any remaining NaNs (though interpolation should have caught them)
    genai_series = genai_series.fillna(0)

    if 'active_genai_pods' in genai_series.columns:
        # Now astype(int) will work because there are no NaNs
        genai_series['active_genai_pods'] = genai_series['active_genai_pods'].round().astype(int)

    # 4. Save to Processed
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    genai_series.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! Processed {len(genai_series)} ticks. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_v2026()