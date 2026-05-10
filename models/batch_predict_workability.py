# Applies a trained workability model to all per-day WHACS CSV files, interpolating
# missing environmental values spatially and appending predicted success probabilities.

import pandas as pd
import numpy as np
import pickle
import os
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')


class WeatherDataProcessor:
    def __init__(self, csv_folder='new_WHACS_Extracted_multiyear', model_path='best_reef_prediction_model_randomforest.pkl'):
        self.csv_folder = csv_folder
        self.model_path = model_path
        self.model_info = None
        self.load_model()

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.model_info = pickle.load(f)
            print(f"Model loaded successfully: {self.model_info['model_name']}")
        except FileNotFoundError:
            print(f"Error: Model file '{self.model_path}' not found!")
            print("Please ensure you have trained and saved the model first.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def get_csv_files(self):
        if not os.path.exists(self.csv_folder):
            print(f"Error: Folder '{self.csv_folder}' not found!")
            return []

        csv_files = [f for f in os.listdir(self.csv_folder) if f.endswith('.csv')]

        def sort_key(filename):
            try:
                name = filename.replace('.csv', '')
                month, day = map(int, name.split('-'))
                return (month, day)
            except:
                return (999, 999)

        csv_files.sort(key=sort_key)
        return csv_files

    def interpolate_missing_values_spatial(self, df, feature_cols=['wave_height', 'u_wind', 'v_wind']):
        df_interpolated = df.copy()

        for feature in feature_cols:
            if feature not in df.columns:
                continue

            missing_mask = df[feature].isna()
            if not missing_mask.any():
                continue

            valid_mask = df[feature].notna()
            if not valid_mask.any():
                print(f"Warning: No valid data for {feature}, dropping all rows")
                continue

            valid_coords = df.loc[valid_mask, ['lon', 'lat']].values
            valid_values = df.loc[valid_mask, feature].values
            missing_coords = df.loc[missing_mask, ['lon', 'lat']].values

            if len(valid_coords) < 3:
                tree = KDTree(valid_coords)
                _, nearest_indices = tree.query(missing_coords)
                interpolated_values = valid_values[nearest_indices]
            else:
                try:
                    interpolated_values = griddata(
                        valid_coords, valid_values, missing_coords,
                        method='linear', fill_value=np.nan
                    )

                    nan_mask = np.isnan(interpolated_values)
                    if nan_mask.any():
                        nearest_values = griddata(
                            valid_coords, valid_values, missing_coords[nan_mask],
                            method='nearest'
                        )
                        interpolated_values[nan_mask] = nearest_values

                except Exception as e:
                    print(f"Interpolation failed for {feature}: {e}, using nearest neighbour")
                    tree = KDTree(valid_coords)
                    _, nearest_indices = tree.query(missing_coords)
                    interpolated_values = valid_values[nearest_indices]

            df_interpolated.loc[missing_mask, feature] = interpolated_values

            filled_count = missing_mask.sum()
            print(f"  Interpolated {filled_count} missing values for {feature}")

        return df_interpolated

    def predict_probabilities(self, df):
        if self.model_info is None:
            print("Error: No model loaded!")
            return df

        model = self.model_info['model']
        features = self.model_info['features']

        missing_features = [f for f in features if f not in df.columns]

        if missing_features:
            print(f"Warning: Missing features for prediction: {missing_features}")
            for feature in missing_features:
                if feature == 'month':
                    if 'date' in df.columns:
                        df['month'] = pd.to_datetime(df['date']).dt.month
                    else:
                        df['month'] = 1
                elif feature == 'wind_magnitude':
                    if 'u_wind' in df.columns and 'v_wind' in df.columns:
                        df['wind_magnitude'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)
                    else:
                        df['wind_magnitude'] = 0
                else:
                    df[feature] = 0

        if 'wind_magnitude' not in df.columns and 'u_wind' in df.columns and 'v_wind' in df.columns:
            df['wind_magnitude'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)

        if 'month' not in df.columns and 'date' in df.columns:
            df['month'] = pd.to_datetime(df['date']).dt.month

        X = df[features].copy()
        X = X.fillna(X.mean())

        try:
            probabilities = model.predict_proba(X)[:, 1]
            df['probability'] = probabilities

            print(f"  Added probabilities. Mean: {probabilities.mean():.3f}, "
                  f"Min: {probabilities.min():.3f}, Max: {probabilities.max():.3f}")

        except Exception as e:
            print(f"Error predicting probabilities: {e}")
            df['probability'] = 0.5

        return df

    def process_single_file(self, filename):
        filepath = os.path.join(self.csv_folder, filename)

        try:
            df = pd.read_csv(filepath)

            print(f"\nProcessing {filename}:")
            print(f"  Original shape: {df.shape}")

            missing_counts = df[['wave_height', 'u_wind', 'v_wind']].isna().sum()
            total_missing = missing_counts.sum()

            if total_missing > 0:
                print(f"  Missing values: wave_height={missing_counts['wave_height']}, "
                      f"u_wind={missing_counts['u_wind']}, v_wind={missing_counts['v_wind']}")

                df = self.interpolate_missing_values_spatial(df)

                remaining_missing = df[['wave_height', 'u_wind', 'v_wind']].isna().sum().sum()

                if remaining_missing > 0:
                    print(f"  Dropping {remaining_missing} rows with remaining missing values")
                    df = df.dropna(subset=['wave_height', 'u_wind', 'v_wind'])

            df = self.predict_probabilities(df)

            print(f"  Final shape: {df.shape}")

            df.to_csv(filepath, index=False)

            return df

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None

    def process_all_files(self):
        if self.model_info is None:
            print("Cannot process files: No model loaded!")
            return

        csv_files = self.get_csv_files()

        if not csv_files:
            print("No CSV files found!")
            return

        print(f"Found {len(csv_files)} CSV files to process")
        print(f"Processing files with interpolation and probability prediction...")

        successful_files = 0
        failed_files = 0

        for i, filename in enumerate(csv_files, 1):
            print(f"\n[{i}/{len(csv_files)}] Processing {filename}...")

            result = self.process_single_file(filename)

            if result is not None:
                successful_files += 1
            else:
                failed_files += 1

        print(f"\n{'=' * 60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Successfully processed: {successful_files} files")
        print(f"Failed to process: {failed_files} files")

        if successful_files > 0:
            print(f"\nSample results from first processed file:")
            sample_file = csv_files[0]
            sample_df = pd.read_csv(os.path.join(self.csv_folder, sample_file))
            print(f"Columns: {list(sample_df.columns)}")
            print(f"Sample probabilities: {sample_df['probability'].head().tolist()}")


def main():
    print("Starting weather data processing...")

    processor = WeatherDataProcessor()
    processor.process_all_files()


if __name__ == "__main__":
    main()
