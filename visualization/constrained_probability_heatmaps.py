# Generates seasonal probability heatmaps for reef survey workability across zonal and
# meridional wind space, applying physically realistic wind-wave constraints derived from
# historical WHACS hindcast data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')


class RealisticWindConstraints:
    def __init__(self):
        self.ms_to_knots = 1.94384

        wind_components_ms = {
            0.5: {'u_wind_range': [-5.8, 1.8], 'v_wind_range': [-4.2, 3.4]},
            1.0: {'u_wind_range': [-7.0, 0.6], 'v_wind_range': [-3.6, 5.3]},
            1.5: {'u_wind_range': [-8.3, -2.9], 'v_wind_range': [0.4, 7.0]},
            2.0: {'u_wind_range': [-9.3, -5.2], 'v_wind_range': [2.8, 8.2]},
            2.5: {'u_wind_range': [-10.2, -6.4], 'v_wind_range': [4.2, 9.2]},
            3.0: {'u_wind_range': [-11.1, -7.1], 'v_wind_range': [5.2, 10.3]}
        }

        self.wind_components_for_wave = {}
        for wave_height, ranges in wind_components_ms.items():
            self.wind_components_for_wave[wave_height] = {
                'u_wind_range': [r * self.ms_to_knots for r in ranges['u_wind_range']],
                'v_wind_range': [r * self.ms_to_knots for r in ranges['v_wind_range']]
            }

    def get_realistic_ranges(self, wave_height):
        available_heights = list(self.wind_components_for_wave.keys())
        closest_height = min(available_heights, key=lambda x: abs(x - wave_height))
        return self.wind_components_for_wave[closest_height]

    def interpolate_ranges(self, wave_height):
        wave_heights = sorted(self.wind_components_for_wave.keys())

        if wave_height <= wave_heights[0]:
            return self.wind_components_for_wave[wave_heights[0]]
        elif wave_height >= wave_heights[-1]:
            return self.wind_components_for_wave[wave_heights[-1]]

        for i in range(len(wave_heights) - 1):
            if wave_heights[i] <= wave_height <= wave_heights[i + 1]:
                lower_wh = wave_heights[i]
                upper_wh = wave_heights[i + 1]

                factor = (wave_height - lower_wh) / (upper_wh - lower_wh)

                lower_data = self.wind_components_for_wave[lower_wh]
                upper_data = self.wind_components_for_wave[upper_wh]

                u_min = lower_data['u_wind_range'][0] + factor * (upper_data['u_wind_range'][0] - lower_data['u_wind_range'][0])
                u_max = lower_data['u_wind_range'][1] + factor * (upper_data['u_wind_range'][1] - lower_data['u_wind_range'][1])
                v_min = lower_data['v_wind_range'][0] + factor * (upper_data['v_wind_range'][0] - lower_data['v_wind_range'][0])
                v_max = lower_data['v_wind_range'][1] + factor * (upper_data['v_wind_range'][1] - lower_data['v_wind_range'][1])

                return {
                    'u_wind_range': [u_min, u_max],
                    'v_wind_range': [v_min, v_max]
                }

        return self.get_realistic_ranges(wave_height)

    def create_realistic_mask(self, U_grid, V_grid, wave_height, tolerance=0.9):
        ranges = self.interpolate_ranges(wave_height)

        u_min, u_max = ranges['u_wind_range']
        v_min, v_max = ranges['v_wind_range']

        u_range_expanded = (u_max - u_min) * (1 - tolerance) / 2
        v_range_expanded = (v_max - v_min) * (1 - tolerance) / 2

        u_min_tol = u_min - u_range_expanded
        u_max_tol = u_max + u_range_expanded
        v_min_tol = v_min - v_range_expanded
        v_max_tol = v_max + v_range_expanded

        u_mask = (U_grid >= u_min_tol) & (U_grid <= u_max_tol)
        v_mask = (V_grid >= v_min_tol) & (V_grid <= v_max_tol)

        return u_mask & v_mask


class WeatherModelHeatmaps:
    def __init__(self, model_path='best_reef_model.pkl'):
        self.model_path = model_path
        self.model_info = None
        self.load_model()

        self.constraints = RealisticWindConstraints()

        self.ms_to_knots = 1.94384
        self.u_wind_range = (-15 * self.ms_to_knots, 15 * self.ms_to_knots)
        self.v_wind_range = (-15 * self.ms_to_knots, 15 * self.ms_to_knots)
        self.wave_height_range = (0, 3)

        self.u_wind_resolution = 50
        self.v_wind_resolution = 50

        self.month_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.model_info = pickle.load(f)
            print(f"Model loaded successfully: {self.model_info['model_name']}")
            print(f"Model features: {self.model_info['features']}")
            print(f"Optimal threshold: {self.model_info['threshold']:.2f}")
        except FileNotFoundError:
            print(f"Error: Model file '{self.model_path}' not found!")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True

    def create_wind_grid(self):
        u_wind_values = np.linspace(self.u_wind_range[0], self.u_wind_range[1], self.u_wind_resolution)
        v_wind_values = np.linspace(self.v_wind_range[0], self.v_wind_range[1], self.v_wind_resolution)

        U_grid, V_grid = np.meshgrid(u_wind_values, v_wind_values)

        return U_grid, V_grid, u_wind_values, v_wind_values

    def convert_knots_to_ms(self, wind_knots):
        return wind_knots / self.ms_to_knots

    def predict_probabilities_grid(self, U_grid, V_grid, wave_height, month):
        if self.model_info is None:
            print("Error: No model loaded!")
            return None

        model = self.model_info['model']
        features = self.model_info['features']

        u_flat = U_grid.flatten()
        v_flat = V_grid.flatten()

        u_flat_ms = self.convert_knots_to_ms(u_flat)
        v_flat_ms = self.convert_knots_to_ms(v_flat)

        n_points = len(u_flat_ms)
        data = {
            'u_wind': u_flat_ms,
            'v_wind': v_flat_ms,
            'wave_height': np.full(n_points, wave_height),
            'month': np.full(n_points, month)
        }

        data['wind_magnitude'] = np.sqrt(data['u_wind']**2 + data['v_wind']**2)

        for feature in features:
            if feature not in data:
                data[feature] = np.full(n_points, 0)

        df = pd.DataFrame(data)
        X = df[features].copy()
        X = X.fillna(X.mean())

        try:
            probabilities = model.predict_proba(X)[:, 1]
            prob_grid = probabilities.reshape(U_grid.shape)
            return prob_grid

        except Exception as e:
            print(f"Error predicting probabilities: {e}")
            return None

    def format_wind_labels(self, wind_values, wind_type='u'):
        labels = []
        for val in wind_values:
            if abs(val) < 1:
                labels.append("0")
            elif wind_type == 'u':
                if val > 0:
                    labels.append(f"{abs(val):.0f}E")
                else:
                    labels.append(f"{abs(val):.0f}W")
            else:
                if val > 0:
                    labels.append(f"{abs(val):.0f}N")
                else:
                    labels.append(f"{abs(val):.0f}S")
        return labels

    def create_single_heatmap(self, U_grid, V_grid, prob_grid, u_wind_values, v_wind_values,
                              wave_height, month, ax=None, show_colorbar=False,
                              show_labels=True, show_title=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        realistic_mask = self.constraints.create_realistic_mask(U_grid, V_grid, wave_height, tolerance=0.9)

        masked_prob_grid = prob_grid.copy()

        colors = [
            '#CC0000', '#FF4444', '#FF8888', '#FFCCCC', '#FFFFFF',
            '#CCCCFF', '#8888FF', '#4444FF', '#0000CC'
        ]

        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('rainbow_custom', colors, N=n_bins)

        im = ax.imshow(
            masked_prob_grid,
            extent=[self.u_wind_range[0], self.u_wind_range[1], self.v_wind_range[0], self.v_wind_range[1]],
            origin='lower', aspect='equal', cmap=cmap, vmin=0, vmax=1
        )

        if show_labels:
            ax.set_xlabel('Zonal Wind (knots)', fontsize=8)
            ax.set_ylabel('Meridional Wind (knots)', fontsize=8)

        if show_title:
            ax.set_title(f'{self.month_names[month - 1]}', fontsize=10)

        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Probability of Success', rotation=270, labelpad=15)

        realistic_prob_grid = prob_grid.copy()

        ax.contour(
            np.linspace(self.u_wind_range[0], self.u_wind_range[1], self.u_wind_resolution),
            np.linspace(self.v_wind_range[0], self.v_wind_range[1], self.v_wind_resolution),
            realistic_prob_grid, levels=[0.2, 0.4, 0.6, 0.8], colors='white', alpha=0.4, linewidths=0.5
        )

        boundary_grid = realistic_mask.astype(float)
        ax.contour(
            np.linspace(self.u_wind_range[0], self.u_wind_range[1], self.u_wind_resolution),
            np.linspace(self.v_wind_range[0], self.v_wind_range[1], self.v_wind_resolution),
            boundary_grid, levels=[0.5], colors='black', alpha=0.8, linewidths=3
        )

        u_tick_values = np.array([-30, -20, -10, 0, 10, 20, 30])
        v_tick_values = np.array([-30, -20, -10, 0, 10, 20, 30])

        u_tick_values = u_tick_values[
            (u_tick_values >= self.u_wind_range[0]) & (u_tick_values <= self.u_wind_range[1])
        ]
        v_tick_values = v_tick_values[
            (v_tick_values >= self.v_wind_range[0]) & (v_tick_values <= self.v_wind_range[1])
        ]

        ax.set_xticks(u_tick_values)
        ax.set_yticks(v_tick_values)

        ax.set_xticklabels(self.format_wind_labels(u_tick_values, 'u'), fontsize=8)
        ax.set_yticklabels(self.format_wind_labels(v_tick_values, 'v'), fontsize=8)

        ranges = self.constraints.interpolate_ranges(wave_height)
        u_range = ranges['u_wind_range']
        v_range = ranges['v_wind_range']

        constraint_text = (
            f"Realistic ranges:\n"
            f"Zonal: [{u_range[0]:.1f}, {u_range[1]:.1f}] kt\n"
            f"Meridional: [{v_range[0]:.1f}, {v_range[1]:.1f}] kt"
        )
        ax.text(0.02, 0.98, constraint_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=6, verticalalignment='top')

        return ax, im

    def generate_seasonal_comparison(self, save_plot=True):
        if self.model_info is None:
            print("Cannot generate heatmaps: No model loaded!")
            return

        months = [1, 4, 7, 10]
        wave_heights = [0.5, 2.0]

        U_grid, V_grid, u_wind_values, v_wind_values = self.create_wind_grid()

        fig, axes = plt.subplots(4, 2, figsize=(12, 16))

        print("Generating seasonal comparison heatmaps with realistic constraints...")

        colorbar_im = None

        for month_idx, month in enumerate(months):
            print(f"  Processing {self.month_names[month - 1]}...")

            for wave_idx, wave_height in enumerate(wave_heights):
                print(f"    Wave height: {wave_height}m")

                prob_grid = self.predict_probabilities_grid(U_grid, V_grid, wave_height, month)

                if prob_grid is None:
                    continue

                ax = axes[month_idx, wave_idx]

                ax_result, im = self.create_single_heatmap(
                    U_grid, V_grid, prob_grid, u_wind_values, v_wind_values,
                    wave_height, month, ax,
                    show_colorbar=False, show_labels=True, show_title=True
                )

                if colorbar_im is None:
                    colorbar_im = im

        axes[0, 0].text(0.5, 1.15, 'Wave Height = 0.5m', transform=axes[0, 0].transAxes,
                        fontsize=12, fontweight='bold', ha='center')
        axes[0, 1].text(0.5, 1.15, 'Wave Height = 2.0m', transform=axes[0, 1].transAxes,
                        fontsize=12, fontweight='bold', ha='center')

        if colorbar_im is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(colorbar_im, cax=cbar_ax)
            cbar.set_label('Probability of Success', rotation=270, labelpad=20, fontsize=14)

        legend_text = (
            "Areas outside the box represent unrealistic\n"
            "wind-wave combinations based\non historical data"
        )
        fig.text(0.02, 0.02, legend_text, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.subplots_adjust(left=0.08, right=0.90, top=0.93, bottom=0.08, hspace=0.35, wspace=0.25)

        if save_plot:
            filename = 'constrained_seasonal_heatmaps_knots.png'
            plt.savefig(filename, dpi=400, bbox_inches='tight', facecolor='white')
            print(f"\nSaved constrained seasonal comparison heatmap: {filename}")

        return fig

    def plot_constraint_boundaries(self, save_plot=True):
        fig, ax = plt.subplots(figsize=(10, 10))

        U_grid, V_grid, u_wind_values, v_wind_values = self.create_wind_grid()

        wave_heights = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

        for i, wave_height in enumerate(wave_heights):
            realistic_mask = self.constraints.create_realistic_mask(U_grid, V_grid, wave_height)
            boundary_grid = realistic_mask.astype(float)
            ax.contour(
                np.linspace(self.u_wind_range[0], self.u_wind_range[1], self.u_wind_resolution),
                np.linspace(self.v_wind_range[0], self.v_wind_range[1], self.v_wind_resolution),
                boundary_grid, levels=[0.5], colors=colors[i], linewidths=5,
                alpha=0.8, label=f'{wave_height}m waves'
            )

        ax.set_xlabel('Zonal Wind (knots)', fontsize=12)
        ax.set_ylabel('Meridional Wind (knots)', fontsize=12)
        ax.set_title('Realistic Wind Component Boundaries for Different Wave Heights', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(self.u_wind_range)
        ax.set_ylim(self.v_wind_range)
        ax.set_aspect('equal')

        u_tick_values = np.array([-30, -20, -10, 0, 10, 20, 30])
        v_tick_values = np.array([-30, -20, -10, 0, 10, 20, 30])

        u_tick_values = u_tick_values[
            (u_tick_values >= self.u_wind_range[0]) & (u_tick_values <= self.u_wind_range[1])
        ]
        v_tick_values = v_tick_values[
            (v_tick_values >= self.v_wind_range[0]) & (v_tick_values <= self.v_wind_range[1])
        ]

        ax.set_xticks(u_tick_values)
        ax.set_yticks(v_tick_values)
        ax.set_xticklabels(self.format_wind_labels(u_tick_values, 'u'))
        ax.set_yticklabels(self.format_wind_labels(v_tick_values, 'v'))

        if save_plot:
            filename = 'constraint_boundaries_knots.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved constraint boundaries plot: {filename}")

        return fig

    def format_wind_labels(self, wind_values, wind_type='u'):
        labels = []
        for val in wind_values:
            if abs(val) < 1:
                labels.append("0")
            elif wind_type == 'u':
                if val > 0:
                    labels.append(f"{abs(val):.0f}E")
                else:
                    labels.append(f"{abs(val):.0f}W")
            else:
                if val > 0:
                    labels.append(f"{abs(val):.0f}N")
                else:
                    labels.append(f"{abs(val):.0f}S")
        return labels


def main():
    generator = WeatherModelHeatmaps()

    print(f"\nConfiguration:")
    print(f"  Months: January, April, July, October")
    print(f"  Wave heights: 0.5m, 2.0m")
    print(f"  Zonal wind range: {generator.u_wind_range[0]:.1f} to {generator.u_wind_range[1]:.1f} knots")
    print(f"  Meridional wind range: {generator.v_wind_range[0]:.1f} to {generator.v_wind_range[1]:.1f} knots")
    print(f"  Realistic constraints: Applied based on historical data")

    print(f"\n{'=' * 60}")
    print("GENERATING CONSTRAINED SEASONAL COMPARISON HEATMAPS")
    print(f"{'=' * 60}")

    seasonal_fig = generator.generate_seasonal_comparison(save_plot=True)

    print(f"\n{'=' * 60}")
    print("GENERATING CONSTRAINT BOUNDARY VISUALIZATION")
    print(f"{'=' * 60}")

    generator.plot_constraint_boundaries(save_plot=True)


if __name__ == "__main__":
    main()
