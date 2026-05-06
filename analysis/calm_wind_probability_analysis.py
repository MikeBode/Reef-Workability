# Investigates model prediction behaviour for near-zero wind conditions, analysing
# training data to diagnose why the classifier returns low probability at calm wind speeds.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')


class TrainingDataAnalyzer:
    def __init__(self):
        self.ms_to_knots = 1.94384

    def load_and_preprocess_data(self):
        successful_visits = pd.read_csv('surveyData[63]WithWindWaveData_Final.csv')
        failed_visits = pd.read_csv('cots_withWindWaveData.csv')

        successful_visits['date'] = pd.to_datetime(successful_visits['date'])
        if 'Date' in failed_visits.columns:
            failed_visits['date'] = pd.to_datetime(failed_visits['Date'])
            failed_visits.drop('Date', axis=1, inplace=True)
        else:
            failed_visits['date'] = pd.to_datetime(failed_visits['date'])

        successful_visits['visit_status'] = 'Successful'
        failed_visits['visit_status'] = 'Failed'

        column_mapping = {
            'Reef': 'reefName',
            'Vessel': 'source',
            'Region': 'region',
        }
        failed_visits.rename(
            columns={k: v for k, v in column_mapping.items() if k in failed_visits.columns},
            inplace=True
        )

        for df in [successful_visits, failed_visits]:
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            df['quarter'] = df['date'].dt.quarter
            df['day_of_year'] = df['date'].dt.dayofyear

        common_columns = [
            'date', 'reefName', 'x', 'y', 'wave_height', 'u_wind', 'v_wind',
            'visit_status', 'month', 'year', 'quarter', 'day_of_year'
        ]

        if 'Days lost' in failed_visits.columns:
            failed_visits['days_lost'] = failed_visits['Days lost']
            common_columns.append('days_lost')
            successful_visits['days_lost'] = 0

        for col in common_columns:
            if col not in successful_visits.columns:
                successful_visits[col] = np.nan
            if col not in failed_visits.columns:
                failed_visits[col] = np.nan

        combined_df = pd.concat([
            successful_visits[common_columns],
            failed_visits[common_columns]
        ]).reset_index(drop=True)

        combined_df['wind_magnitude'] = np.sqrt(combined_df['u_wind']**2 + combined_df['v_wind']**2)
        combined_df = combined_df.dropna(subset=['wave_height', 'u_wind', 'v_wind'])
        combined_df = combined_df.sort_values('date').reset_index(drop=True)

        return combined_df

    def analyze_calm_conditions(self, combined_df):
        print("=" * 80)
        print("ANALYSIS OF CALM WIND CONDITIONS")
        print("=" * 80)

        combined_df['u_wind_knots'] = combined_df['u_wind'] * self.ms_to_knots
        combined_df['v_wind_knots'] = combined_df['v_wind'] * self.ms_to_knots
        combined_df['wind_speed_knots'] = combined_df['wind_magnitude'] * self.ms_to_knots

        calm_thresholds = [2, 5, 8, 10]

        print(f"Total dataset size: {len(combined_df)}")
        print(f"Successful visits: {len(combined_df[combined_df['visit_status'] == 'Successful'])}")
        print(f"Failed visits: {len(combined_df[combined_df['visit_status'] == 'Failed'])}")
        print()

        for threshold in calm_thresholds:
            calm_conditions = combined_df[combined_df['wind_speed_knots'] <= threshold]

            if len(calm_conditions) > 0:
                success_rate = (calm_conditions['visit_status'] == 'Successful').mean()
                print(f"Wind speed <= {threshold} knots:")
                print(f"  Total samples: {len(calm_conditions)}")
                print(f"  Successful: {len(calm_conditions[calm_conditions['visit_status'] == 'Successful'])}")
                print(f"  Failed: {len(calm_conditions[calm_conditions['visit_status'] == 'Failed'])}")
                print(f"  Success rate: {success_rate:.3f} ({success_rate * 100:.1f}%)")

                if len(calm_conditions) <= 10:
                    print(f"  All {threshold}-knot samples:")
                    for idx, row in calm_conditions.iterrows():
                        print(
                            f"    u={row['u_wind_knots']:.1f}, v={row['v_wind_knots']:.1f}, "
                            f"wave={row['wave_height']:.1f}m, status={row['visit_status']}"
                        )
                print()

        return combined_df

    def analyze_near_zero_winds(self, combined_df):
        print("=" * 80)
        print("ANALYSIS OF NEAR-ZERO WIND CONDITIONS")
        print("=" * 80)

        tolerance_ranges = [1, 2, 3, 5]

        for tolerance in tolerance_ranges:
            near_zero = combined_df[
                (abs(combined_df['u_wind_knots']) <= tolerance) &
                (abs(combined_df['v_wind_knots']) <= tolerance)
            ]

            if len(near_zero) > 0:
                success_rate = (near_zero['visit_status'] == 'Successful').mean()
                print(f"Wind components within +/-{tolerance} knots of (0,0):")
                print(f"  Total samples: {len(near_zero)}")
                print(f"  Success rate: {success_rate:.3f} ({success_rate * 100:.1f}%)")

                wave_stats = near_zero.groupby('visit_status')['wave_height'].agg(
                    ['count', 'mean', 'std', 'min', 'max']
                )
                print(f"  Wave height stats by visit status:")
                print(wave_stats.round(3))
                print()

    def analyze_data_balance_by_conditions(self, combined_df):
        print("=" * 80)
        print("CLASS BALANCE ANALYSIS BY ENVIRONMENTAL CONDITIONS")
        print("=" * 80)

        combined_df['wind_speed_bin'] = pd.cut(
            combined_df['wind_speed_knots'],
            bins=[0, 5, 10, 15, 20, 100],
            labels=['0-5kt', '5-10kt', '10-15kt', '15-20kt', '20+kt']
        )

        combined_df['wave_height_bin'] = pd.cut(
            combined_df['wave_height'],
            bins=[0, 0.5, 1.0, 1.5, 2.0, 10],
            labels=['0-0.5m', '0.5-1.0m', '1.0-1.5m', '1.5-2.0m', '2.0+m']
        )

        cross_tab = pd.crosstab(
            [combined_df['wind_speed_bin'], combined_df['wave_height_bin']],
            combined_df['visit_status'],
            margins=True
        )
        print("Cross-tabulation: Wind Speed vs Wave Height vs Visit Status")
        print(cross_tab)
        print()

        success_rates = combined_df.groupby(['wind_speed_bin', 'wave_height_bin']).agg({
            'visit_status': lambda x: (x == 'Successful').mean()
        }).round(3)
        success_rates.columns = ['Success_Rate']

        print("Success rates by environmental conditions:")
        print(success_rates)
        print()

        return combined_df

    def create_wind_distribution_plots(self, combined_df):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        successful = combined_df[combined_df['visit_status'] == 'Successful']
        failed = combined_df[combined_df['visit_status'] == 'Failed']

        axes[0, 0].hist(successful['u_wind_knots'], bins=30, alpha=0.7, label='Successful', color='blue')
        axes[0, 0].hist(failed['u_wind_knots'], bins=30, alpha=0.7, label='Failed', color='red')
        axes[0, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('U-wind (knots)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('U-wind Distribution')
        axes[0, 0].legend()

        axes[0, 1].hist(successful['v_wind_knots'], bins=30, alpha=0.7, label='Successful', color='blue')
        axes[0, 1].hist(failed['v_wind_knots'], bins=30, alpha=0.7, label='Failed', color='red')
        axes[0, 1].axvline(0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('V-wind (knots)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('V-wind Distribution')
        axes[0, 1].legend()

        axes[0, 2].hist(successful['wind_speed_knots'], bins=30, alpha=0.7, label='Successful', color='blue')
        axes[0, 2].hist(failed['wind_speed_knots'], bins=30, alpha=0.7, label='Failed', color='red')
        axes[0, 2].set_xlabel('Wind Speed (knots)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Wind Speed Distribution')
        axes[0, 2].legend()

        axes[1, 0].scatter(
            successful['u_wind_knots'], successful['v_wind_knots'],
            alpha=0.6, s=20, label='Successful', color='blue'
        )
        axes[1, 0].scatter(
            failed['u_wind_knots'], failed['v_wind_knots'],
            alpha=0.6, s=20, label='Failed', color='red'
        )
        axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.3)
        axes[1, 0].set_xlabel('U-wind (knots)')
        axes[1, 0].set_ylabel('V-wind (knots)')
        axes[1, 0].set_title('Wind Components Scatter Plot')
        axes[1, 0].legend()

        wind_bins = np.arange(0, 30, 2)
        success_by_wind = []
        wind_centers = []

        for i in range(len(wind_bins) - 1):
            wind_range = combined_df[
                (combined_df['wind_speed_knots'] >= wind_bins[i]) &
                (combined_df['wind_speed_knots'] < wind_bins[i + 1])
            ]
            if len(wind_range) > 0:
                success_rate = (wind_range['visit_status'] == 'Successful').mean()
                success_by_wind.append(success_rate)
                wind_centers.append((wind_bins[i] + wind_bins[i + 1]) / 2)

        axes[1, 1].bar(wind_centers, success_by_wind, width=1.5, alpha=0.7)
        axes[1, 1].set_xlabel('Wind Speed (knots)')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Success Rate by Wind Speed')
        axes[1, 1].set_ylim(0, 1)

        u_bins = np.linspace(-20, 15, 15)
        v_bins = np.linspace(-15, 20, 15)
        success_grid = np.zeros((len(v_bins) - 1, len(u_bins) - 1))
        count_grid = np.zeros((len(v_bins) - 1, len(u_bins) - 1))

        for i in range(len(u_bins) - 1):
            for j in range(len(v_bins) - 1):
                mask = (
                    (combined_df['u_wind_knots'] >= u_bins[i]) &
                    (combined_df['u_wind_knots'] < u_bins[i + 1]) &
                    (combined_df['v_wind_knots'] >= v_bins[j]) &
                    (combined_df['v_wind_knots'] < v_bins[j + 1])
                )
                subset = combined_df[mask]
                if len(subset) > 0:
                    success_grid[j, i] = (subset['visit_status'] == 'Successful').mean()
                    count_grid[j, i] = len(subset)
                else:
                    success_grid[j, i] = np.nan

        success_grid[count_grid < 3] = np.nan

        im = axes[1, 2].imshow(
            success_grid,
            extent=[u_bins[0], u_bins[-1], v_bins[0], v_bins[-1]],
            origin='lower', aspect='equal', cmap='RdBu', vmin=0, vmax=1
        )
        axes[1, 2].set_xlabel('U-wind (knots)')
        axes[1, 2].set_ylabel('V-wind (knots)')
        axes[1, 2].set_title('Success Rate Heatmap (Training Data)')
        plt.colorbar(im, ax=axes[1, 2])

        plt.tight_layout()
        plt.savefig('wind_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def analyze_model_predictions_vs_reality(self, combined_df, model_path='best_reef_model_temporal_2.pkl'):
        print("=" * 80)
        print("MODEL PREDICTIONS VS TRAINING DATA REALITY")
        print("=" * 80)

        try:
            with open(model_path, 'rb') as f:
                model_info = pickle.load(f)

            model = model_info['model']
            features = model_info['features']
            threshold = model_info['threshold']

            X = combined_df[features].fillna(combined_df[features].mean())
            y_true = (combined_df['visit_status'] == 'Successful').astype(int)

            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)

            combined_df['pred_proba'] = y_pred_proba
            combined_df['pred_class'] = y_pred

            print("Overall model performance on training data:")
            print(classification_report(y_true, y_pred, target_names=['Failed', 'Successful']))

            calm_conditions = combined_df[combined_df['wind_speed_knots'] <= 5]

            if len(calm_conditions) > 0:
                print(f"\nCalm conditions (<=5 knots) analysis:")
                print(f"Number of samples: {len(calm_conditions)}")
                print(f"Actual success rate: {(calm_conditions['visit_status'] == 'Successful').mean():.3f}")
                print(f"Average predicted probability: {calm_conditions['pred_proba'].mean():.3f}")
                print(f"Predicted success rate: {(calm_conditions['pred_class'] == 1).mean():.3f}")

                near_zero = calm_conditions[
                    (abs(calm_conditions['u_wind_knots']) <= 2) &
                    (abs(calm_conditions['v_wind_knots']) <= 2)
                ]

                if len(near_zero) > 0:
                    print(f"\nNear-zero wind samples (+/-2 knots):")
                    for idx, row in near_zero.head(10).iterrows():
                        print(
                            f"  u={row['u_wind_knots']:.1f}, v={row['v_wind_knots']:.1f}, "
                            f"wave={row['wave_height']:.1f}m, actual={row['visit_status']}, "
                            f"pred_prob={row['pred_proba']:.3f}"
                        )

            return combined_df

        except FileNotFoundError:
            print(f"Model file '{model_path}' not found.")
            return combined_df

    def suggest_fixes(self):
        print("=" * 80)
        print("SUGGESTED FIXES FOR THE (0,0) WIND ISSUE")
        print("=" * 80)

        fixes = [
            {
                "Issue": "Data Imbalance",
                "Description": "Very few samples near (0,0) wind conditions",
                "Solutions": [
                    "Use SMOTE or ADASYN to generate synthetic samples for underrepresented regions",
                    "Collect more data specifically for calm wind conditions",
                    "Use stratified sampling to ensure balanced representation across wind ranges"
                ]
            },
            {
                "Issue": "Model Bias",
                "Description": "Model learned that calm conditions are rare, not that they are bad",
                "Solutions": [
                    "Use class weights that account for environmental conditions",
                    "Train separate models for different wind speed ranges",
                    "Use ensemble methods with models trained on different subsets"
                ]
            },
            {
                "Issue": "Feature Engineering",
                "Description": "Current features may not capture the non-linear relationship",
                "Solutions": [
                    "Add polynomial features (u_wind^2, v_wind^2, u_wind*v_wind)",
                    "Create categorical bins for wind speed ranges",
                    "Add interaction terms between wind components and other variables"
                ]
            },
            {
                "Issue": "Threshold Selection",
                "Description": "Decision threshold may not be optimal for all conditions",
                "Solutions": [
                    "Use condition-specific thresholds",
                    "Optimize threshold using cost-sensitive metrics",
                    "Use calibration techniques like Platt scaling"
                ]
            },
            {
                "Issue": "Temporal Bias",
                "Description": "Calm conditions might be associated with specific time periods in training data",
                "Solutions": [
                    "Add more temporal features (season, time of day if available)",
                    "Use temporal cross-validation with multiple folds",
                    "Train models on different time periods separately"
                ]
            }
        ]

        for i, fix in enumerate(fixes, 1):
            print(f"{i}. {fix['Issue']}")
            print(f"   Description: {fix['Description']}")
            print(f"   Solutions:")
            for solution in fix['Solutions']:
                print(f"     - {solution}")
            print()


def main():
    analyzer = TrainingDataAnalyzer()

    print("Loading and preprocessing data...")
    combined_df = analyzer.load_and_preprocess_data()

    if combined_df is None:
        return

    combined_df = analyzer.analyze_calm_conditions(combined_df)
    analyzer.analyze_near_zero_winds(combined_df)
    combined_df = analyzer.analyze_data_balance_by_conditions(combined_df)

    print("Creating wind distribution plots...")
    analyzer.create_wind_distribution_plots(combined_df)
    analyzer.analyze_model_predictions_vs_reality(combined_df)
    analyzer.suggest_fixes()


if __name__ == "__main__":
    main()
