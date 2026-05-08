# Merges reef site rankings with monthly workability statistics computed from daily WHACS
# probability CSV files, producing one success rate column per calendar month.

import pandas as pd
import numpy as np
import os
from glob import glob


def merge_rankings_with_probabilities(rankings_file, whacs_folder):
    print("Loading rankings data...")
    rankings_df = pd.read_csv(rankings_file)

    rankings_coords = rankings_df[['uniqueID', 'centroidX', 'centroiodY']].copy()
    rankings_coords = rankings_coords.rename(columns={'centroidX': 'lon', 'centroiodY': 'lat'})

    monthly_columns = [f'month_{i}_success_rate' for i in range(1, 13)]
    for col in monthly_columns:
        rankings_df[col] = np.nan

    csv_files = glob(os.path.join(whacs_folder, '*.csv'))

    if not csv_files:
        print(f"No CSV files found in {whacs_folder}")
        return rankings_df

    print(f"Found {len(csv_files)} CSV files to process...")

    monthly_data = {month: [] for month in range(1, 13)}

    for csv_file in csv_files:
        try:
            daily_df = pd.read_csv(csv_file)

            for month in daily_df['month'].unique():
                month_data = daily_df[daily_df['month'] == month].copy()
                monthly_data[month].append(month_data)

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    print("Calculating monthly success probabilities...")

    tolerance = 1e-2

    for month in range(1, 13):
        if not monthly_data[month]:
            print(f"No data found for month {month}")
            continue

        month_combined = pd.concat(monthly_data[month], ignore_index=True)

        location_success = month_combined.groupby(['lat', 'lon']).agg({
            'probability': ['count', lambda x: (x > 0.75).sum()]
        }).reset_index()

        location_success.columns = ['lat', 'lon', 'total_days', 'success_days']
        location_success['success_rate'] = location_success['success_days'] / location_success['total_days']

        for idx, row in rankings_df.iterrows():
            target_lat = row['centroiodY']
            target_lon = row['centroidX']

            lat_match = np.abs(location_success['lat'] - target_lat) < tolerance
            lon_match = np.abs(location_success['lon'] - target_lon) < tolerance
            match_idx = location_success[lat_match & lon_match].index

            if len(match_idx) > 0:
                success_rate = location_success.loc[match_idx[0], 'success_rate']
                rankings_df.at[idx, f'month_{month}_success_rate'] = success_rate

    print("Processing complete!")
    return rankings_df


def main():
    rankings_file = 'rankings.csv'
    whacs_folder = 'new_WHACS_Extracted_multiyear'
    output_file = 'rankings_with_monthly_probabilities_new.csv'

    if not os.path.exists(rankings_file):
        print(f"Error: {rankings_file} not found!")
        return

    if not os.path.exists(whacs_folder):
        print(f"Error: {whacs_folder} folder not found!")
        return

    result_df = merge_rankings_with_probabilities(rankings_file, whacs_folder)

    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    monthly_cols = [col for col in result_df.columns if col.startswith('month_') and col.endswith('_success_rate')]
    print(f"\nSummary of monthly success rates:")
    print(result_df[monthly_cols].describe())

    missing_data = result_df[monthly_cols].isnull().sum()
    if missing_data.sum() > 0:
        print(f"\nWarning: Some locations have missing monthly data:")
        for col, count in missing_data.items():
            if count > 0:
                print(f"  {col}: {count} missing values")


if __name__ == "__main__":
    main()
