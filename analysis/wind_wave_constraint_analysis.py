# Calculates realistic wind component ranges for given wave heights from WHACS hindcast data,
# producing constraint tables, grid visualisations, and exportable Python constraint code.

import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class WindComponentConstraints:
    def __init__(self, whacs_folder='new_WHACS_Extracted'):
        self.whacs_folder = whacs_folder
        self.all_data = None
        self.constraints = {}

    def load_all_data(self, sample_size=None):
        print("Loading data from all CSV files...")

        csv_files = glob(os.path.join(self.whacs_folder, '*.csv'))

        if not csv_files:
            print(f"No CSV files found in {self.whacs_folder}")
            return None

        print(f"Found {len(csv_files)} CSV files")

        all_records = []
        total_files = len(csv_files)

        for i, csv_file in enumerate(csv_files):
            try:
                df = pd.read_csv(csv_file)

                if 'wind_magnitude' not in df.columns:
                    df['wind_magnitude'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)

                df['wind_direction'] = np.arctan2(df['v_wind'], df['u_wind']) * 180 / np.pi

                needed_cols = ['u_wind', 'v_wind', 'wave_height', 'wind_magnitude', 'wind_direction']
                available_cols = [col for col in needed_cols if col in df.columns]

                if len(available_cols) >= 3:
                    all_records.append(df[available_cols])

                if (i + 1) % 100 == 0 or i == total_files - 1:
                    print(f"  Processed {i + 1}/{total_files} files...")

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

        if not all_records:
            print("No valid data found!")
            return None

        combined_df = pd.concat(all_records, ignore_index=True)
        combined_df = combined_df.dropna()
        combined_df = combined_df[
            (combined_df['wave_height'] >= 0) &
            (combined_df['wind_magnitude'] >= 0) &
            (combined_df['wave_height'] < 50) &
            (combined_df['wind_magnitude'] < 100) &
            (np.abs(combined_df['u_wind']) < 100) &
            (np.abs(combined_df['v_wind']) < 100)
        ]

        if sample_size and len(combined_df) > sample_size:
            print(f"Sampling {sample_size:,} records from {len(combined_df):,} total records")
            combined_df = combined_df.sample(n=sample_size, random_state=42)

        print(f"Loaded {len(combined_df):,} valid records")
        print(f"U-wind range: {combined_df['u_wind'].min():.2f} - {combined_df['u_wind'].max():.2f} m/s")
        print(f"V-wind range: {combined_df['v_wind'].min():.2f} - {combined_df['v_wind'].max():.2f} m/s")
        print(f"Wave height range: {combined_df['wave_height'].min():.3f} - {combined_df['wave_height'].max():.2f} m")

        self.all_data = combined_df
        return combined_df

    def calculate_wind_components_for_wave(self, wave_height_bins=None, percentile_range=(10, 90)):
        if self.all_data is None:
            print("No data loaded. Run load_all_data() first.")
            return None

        lower_pct, upper_pct = percentile_range
        print(f"\nCalculating realistic wind component ranges ({lower_pct}th-{upper_pct}th percentiles)...")

        if wave_height_bins is None:
            max_wave = self.all_data['wave_height'].quantile(0.95)
            wave_height_bins = np.arange(0.5, max_wave + 0.5, 0.5)

        wind_components_for_wave = {}

        for wave_height in wave_height_bins:
            tolerance = 0.25
            mask = (
                (self.all_data['wave_height'] >= wave_height - tolerance) &
                (self.all_data['wave_height'] <= wave_height + tolerance)
            )

            wave_data = self.all_data[mask]

            if len(wave_data) > 50:
                u_wind_range = [
                    wave_data['u_wind'].quantile(lower_pct / 100),
                    wave_data['u_wind'].quantile(upper_pct / 100)
                ]
                v_wind_range = [
                    wave_data['v_wind'].quantile(lower_pct / 100),
                    wave_data['v_wind'].quantile(upper_pct / 100)
                ]

                wind_components_for_wave[wave_height] = {
                    'u_wind_range': u_wind_range,
                    'v_wind_range': v_wind_range,
                    'u_wind_mean': wave_data['u_wind'].mean(),
                    'u_wind_std': wave_data['u_wind'].std(),
                    'v_wind_mean': wave_data['v_wind'].mean(),
                    'v_wind_std': wave_data['v_wind'].std(),
                    'count': len(wave_data),
                    'wind_magnitude_range': [
                        wave_data['wind_magnitude'].quantile(lower_pct / 100),
                        wave_data['wind_magnitude'].quantile(upper_pct / 100)
                    ]
                }

                print(
                    f"  Wave {wave_height:.1f}m: U-wind [{u_wind_range[0]:.1f}, {u_wind_range[1]:.1f}] m/s, "
                    f"V-wind [{v_wind_range[0]:.1f}, {v_wind_range[1]:.1f}] m/s ({len(wave_data):,} samples)"
                )

        self.constraints['wind_components_for_wave'] = wind_components_for_wave
        return wind_components_for_wave

    def calculate_wave_height_for_wind_components(self, u_wind_bins=None, v_wind_bins=None, percentile_range=(10, 90)):
        if self.all_data is None:
            print("No data loaded. Run load_all_data() first.")
            return None

        lower_pct, upper_pct = percentile_range
        print(f"\nCalculating realistic wave heights for wind component combinations...")

        if u_wind_bins is None:
            u_wind_range = self.all_data['u_wind'].quantile([0.05, 0.95])
            u_wind_bins = np.arange(u_wind_range.iloc[0], u_wind_range.iloc[1] + 2, 2)

        if v_wind_bins is None:
            v_wind_range = self.all_data['v_wind'].quantile([0.05, 0.95])
            v_wind_bins = np.arange(v_wind_range.iloc[0], v_wind_range.iloc[1] + 2, 2)

        wave_for_wind_components = {}

        for u_wind in u_wind_bins:
            for v_wind in v_wind_bins:
                tolerance = 1.0
                mask = (
                    (self.all_data['u_wind'] >= u_wind - tolerance) &
                    (self.all_data['u_wind'] <= u_wind + tolerance) &
                    (self.all_data['v_wind'] >= v_wind - tolerance) &
                    (self.all_data['v_wind'] <= v_wind + tolerance)
                )

                wind_data = self.all_data[mask]

                if len(wind_data) > 20:
                    wave_range = [
                        wind_data['wave_height'].quantile(lower_pct / 100),
                        wind_data['wave_height'].quantile(upper_pct / 100)
                    ]

                    wave_for_wind_components[(u_wind, v_wind)] = {
                        'wave_range': wave_range,
                        'wave_mean': wind_data['wave_height'].mean(),
                        'wave_std': wind_data['wave_height'].std(),
                        'count': len(wind_data),
                        'wind_magnitude_mean': wind_data['wind_magnitude'].mean()
                    }

        self.constraints['wave_for_wind_components'] = wave_for_wind_components
        return wave_for_wind_components

    def create_wind_component_grid_constraints(self, u_wind_range=None, v_wind_range=None, grid_size=50):
        if self.all_data is None:
            print("No data loaded. Run load_all_data() first.")
            return None

        print(f"\nCreating grid-based wind component constraints...")

        if u_wind_range is None:
            u_wind_range = self.all_data['u_wind'].quantile([0.01, 0.99])
            u_wind_range = [u_wind_range.iloc[0], u_wind_range.iloc[1]]

        if v_wind_range is None:
            v_wind_range = self.all_data['v_wind'].quantile([0.01, 0.99])
            v_wind_range = [v_wind_range.iloc[0], v_wind_range.iloc[1]]

        u_wind_grid = np.linspace(u_wind_range[0], u_wind_range[1], grid_size)
        v_wind_grid = np.linspace(v_wind_range[0], v_wind_range[1], grid_size)

        wave_height_grid = np.zeros((grid_size, grid_size))
        data_density_grid = np.zeros((grid_size, grid_size))

        for i, u_wind in enumerate(u_wind_grid):
            for j, v_wind in enumerate(v_wind_grid):
                tolerance = 2.0
                mask = (
                    (np.abs(self.all_data['u_wind'] - u_wind) <= tolerance) &
                    (np.abs(self.all_data['v_wind'] - v_wind) <= tolerance)
                )

                nearby_data = self.all_data[mask]

                if len(nearby_data) > 5:
                    wave_height_grid[i, j] = nearby_data['wave_height'].median()
                    data_density_grid[i, j] = len(nearby_data)
                else:
                    wave_height_grid[i, j] = np.nan
                    data_density_grid[i, j] = 0

        self.constraints['grid_data'] = {
            'u_wind_grid': u_wind_grid,
            'v_wind_grid': v_wind_grid,
            'wave_height_grid': wave_height_grid,
            'data_density_grid': data_density_grid
        }

        print(f"Grid created: {grid_size}x{grid_size} points")
        print(f"U-wind range: {u_wind_range[0]:.1f} to {u_wind_range[1]:.1f} m/s")
        print(f"V-wind range: {v_wind_range[0]:.1f} to {v_wind_range[1]:.1f} m/s")

        return self.constraints['grid_data']

    def save_wind_component_data(self, output_file='wind_component_constraints.csv'):
        if 'wind_components_for_wave' not in self.constraints:
            print("No wind component constraints calculated.")
            return None

        print(f"\nSaving wind component data to {output_file}...")

        constraint_rows = []

        for wave_height, data in self.constraints['wind_components_for_wave'].items():
            constraint_rows.append({
                'wave_height': wave_height,
                'u_wind_min': data['u_wind_range'][0],
                'u_wind_max': data['u_wind_range'][1],
                'v_wind_min': data['v_wind_range'][0],
                'v_wind_max': data['v_wind_range'][1],
                'u_wind_mean': data['u_wind_mean'],
                'u_wind_std': data['u_wind_std'],
                'v_wind_mean': data['v_wind_mean'],
                'v_wind_std': data['v_wind_std'],
                'wind_magnitude_min': data['wind_magnitude_range'][0],
                'wind_magnitude_max': data['wind_magnitude_range'][1],
                'sample_count': data['count']
            })

        df = pd.DataFrame(constraint_rows)
        df.to_csv(output_file, index=False)

        print(f"Wind component constraints saved to: {output_file}")
        print(f"Data contains {len(df)} wave height bins")

        return df

    def save_grid_data(self, output_file='wind_component_grid.npz'):
        if 'grid_data' not in self.constraints:
            print("No grid data calculated.")
            return None

        grid_data = self.constraints['grid_data']

        np.savez(
            output_file,
            u_wind_grid=grid_data['u_wind_grid'],
            v_wind_grid=grid_data['v_wind_grid'],
            wave_height_grid=grid_data['wave_height_grid'],
            data_density_grid=grid_data['data_density_grid']
        )

        print(f"Grid data saved to: {output_file}")
        return output_file

    def plot_wind_component_analysis(self, save_plots=True):
        if self.all_data is None:
            print("No data loaded.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        plot_data = self.all_data
        if len(plot_data) > 50000:
            plot_data = plot_data.sample(n=50000, random_state=42)

        axes[0, 0].scatter(plot_data['u_wind'], plot_data['wave_height'], alpha=0.1, s=1, c='blue')
        axes[0, 0].set_xlabel('U-Wind (m/s)')
        axes[0, 0].set_ylabel('Wave Height (m)')
        axes[0, 0].set_title('U-Wind vs Wave Height')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(plot_data['v_wind'], plot_data['wave_height'], alpha=0.1, s=1, c='red')
        axes[0, 1].set_xlabel('V-Wind (m/s)')
        axes[0, 1].set_ylabel('Wave Height (m)')
        axes[0, 1].set_title('V-Wind vs Wave Height')
        axes[0, 1].grid(True, alpha=0.3)

        scatter = axes[0, 2].scatter(
            plot_data['u_wind'], plot_data['v_wind'],
            c=plot_data['wave_height'], alpha=0.3, s=1, cmap='viridis'
        )
        axes[0, 2].set_xlabel('U-Wind (m/s)')
        axes[0, 2].set_ylabel('V-Wind (m/s)')
        axes[0, 2].set_title('Wind Components (coloured by wave height)')
        plt.colorbar(scatter, ax=axes[0, 2], label='Wave Height (m)')
        axes[0, 2].grid(True, alpha=0.3)

        if 'wind_components_for_wave' in self.constraints:
            wave_heights = list(self.constraints['wind_components_for_wave'].keys())
            u_wind_ranges = [self.constraints['wind_components_for_wave'][wh]['u_wind_range'] for wh in wave_heights]
            v_wind_ranges = [self.constraints['wind_components_for_wave'][wh]['v_wind_range'] for wh in wave_heights]

            u_mins = [r[0] for r in u_wind_ranges]
            u_maxs = [r[1] for r in u_wind_ranges]
            v_mins = [r[0] for r in v_wind_ranges]
            v_maxs = [r[1] for r in v_wind_ranges]

            axes[1, 0].fill_between(wave_heights, u_mins, u_maxs, alpha=0.3, color='blue', label='U-wind range')
            axes[1, 0].fill_between(wave_heights, v_mins, v_maxs, alpha=0.3, color='red', label='V-wind range')
            axes[1, 0].set_xlabel('Wave Height (m)')
            axes[1, 0].set_ylabel('Wind Component (m/s)')
            axes[1, 0].set_title('Wind Component Ranges for Wave Heights')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(plot_data['wind_direction'], bins=36, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Wind Direction (degrees)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Wind Direction Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        if 'grid_data' in self.constraints:
            grid_data = self.constraints['grid_data']
            u_mesh, v_mesh = np.meshgrid(grid_data['u_wind_grid'], grid_data['v_wind_grid'])

            wave_grid_masked = np.ma.masked_where(
                grid_data['data_density_grid'].T <= 0,
                grid_data['wave_height_grid'].T
            )

            im = axes[1, 2].contourf(u_mesh, v_mesh, wave_grid_masked, levels=20, cmap='viridis')
            axes[1, 2].set_xlabel('U-Wind (m/s)')
            axes[1, 2].set_ylabel('V-Wind (m/s)')
            axes[1, 2].set_title('Wave Height Grid (U-wind vs V-wind)')
            plt.colorbar(im, ax=axes[1, 2], label='Wave Height (m)')
        else:
            axes[1, 2].text(0.5, 0.5, 'No grid data\nRun create_wind_component_grid_constraints()',
                            ha='center', va='center', transform=axes[1, 2].transAxes)

        plt.tight_layout()

        if save_plots:
            plt.savefig('wind_component_analysis.png', dpi=300, bbox_inches='tight')
            print("\nPlots saved as 'wind_component_analysis.png'")

        plt.show()

    def print_summary(self):
        if not self.constraints:
            print("No constraints calculated yet.")
            return

        print("\n" + "=" * 70)
        print("WIND COMPONENT CONSTRAINT SUMMARY")
        print("=" * 70)

        if 'wind_components_for_wave' in self.constraints:
            print("\nWIND COMPONENT RANGES FOR WAVE HEIGHTS:")
            for wave, data in sorted(self.constraints['wind_components_for_wave'].items()):
                u_range = data['u_wind_range']
                v_range = data['v_wind_range']
                print(
                    f"  {wave:.1f}m waves: U-wind [{u_range[0]:.1f}, {u_range[1]:.1f}] m/s, "
                    f"V-wind [{v_range[0]:.1f}, {v_range[1]:.1f}] m/s"
                )

        if self.all_data is not None:
            print(f"\nBased on analysis of {len(self.all_data):,} data points")
            print(f"U-wind range: {self.all_data['u_wind'].min():.1f} - {self.all_data['u_wind'].max():.1f} m/s")
            print(f"V-wind range: {self.all_data['v_wind'].min():.1f} - {self.all_data['v_wind'].max():.1f} m/s")
            print(f"Wave range: {self.all_data['wave_height'].min():.3f} - {self.all_data['wave_height'].max():.2f} m")


def main():
    print("CALCULATING WIND COMPONENT CONSTRAINTS")
    print("=" * 70)

    calculator = WindComponentConstraints('new_WHACS_Extracted')

    data = calculator.load_all_data(sample_size=1000000)

    if data is None:
        print("Failed to load data!")
        return

    calculator.calculate_wind_components_for_wave(percentile_range=(10, 90))
    calculator.calculate_wave_height_for_wind_components(percentile_range=(10, 90))
    calculator.create_wind_component_grid_constraints(grid_size=50)
    calculator.print_summary()
    calculator.plot_wind_component_analysis()
    calculator.save_wind_component_data()
    calculator.save_grid_data()

    return calculator


if __name__ == "__main__":
    calculator = main()
