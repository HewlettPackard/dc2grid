# preprocessing/process_energy.py
# data extracted from: https://mis.nyiso.com/public/P-58Blist.htm

import pandas as pd
import zipfile
import io
import os
import glob

# Configuration
RAW_DIR = "data/raw/energy/"
PROCESSED_DIR = "data/processed/energy/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_nyiso_data():
    # Find all 12 ZIP files
    zip_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.zip")))
    
    # We will store data in a dictionary: {zone_name: [list_of_dfs]}
    zone_data = {}

    print(f"Found {len(zip_files)} zip files. Starting extraction...")

    for zip_path in zip_files:
        print(f"Processing {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Each zip contains daily CSV files
            for csv_name in sorted(z.namelist()):
                if csv_name.endswith('.csv'):
                    with z.open(csv_name) as f:
                        # Read the daily CSV
                        df_day = pd.read_csv(io.BytesIO(f.read()))
                        
                        # Standardize column names (strip whitespace if any)
                        df_day.columns = df_day.columns.str.strip()
                        
                        # Get unique zones in this file
                        zones = df_day['Name'].unique()
                        
                        for zone in zones:
                            if zone not in zone_data:
                                zone_data[zone] = []
                            
                            # Extract only this zone's data
                            df_zone = df_day[df_day['Name'] == zone][['Time Stamp', 'Load']].copy()
                            zone_data[zone].append(df_zone)

    print("Consolidating and saving files...")
    
    for zone, df_list in zone_data.items():
        # Combine all days for this zone
        full_df = pd.concat(df_list, ignore_index=True)
        
        # Convert Time Stamp to datetime for sorting
        full_df['Time Stamp'] = pd.to_datetime(full_df['Time Stamp'])
        full_df = full_df.sort_values('Time Stamp')
        
        # Drop duplicates if any (overlap between files)
        full_df = full_df.drop_duplicates(subset=['Time Stamp'])
        full_df['Load'] = pd.to_numeric(full_df['Load'], errors='coerce')
        full_df['Load'] = full_df['Load'].interpolate(method='linear', limit_direction='both')
        full_df['Load'] = full_df['Load'].ffill().bfill()

        if full_df['Load'].isna().any():
            missing_count = int(full_df['Load'].isna().sum())
            raise ValueError(f"Zone {zone} still has {missing_count} missing load values after repair")
        
        # Create a clean filename (e.g., NYC.csv instead of N.Y.C..csv)
        clean_name = zone.replace('.', '').replace(' ', '_')
        output_path = os.path.join(PROCESSED_DIR, f"{clean_name}.csv")
        
        full_df.to_csv(output_path, index=False)
        print(f"Saved {zone} data to {output_path} ({len(full_df)} records)")

if __name__ == "__main__":
    process_nyiso_data()