# Simulates reef survey fleet scheduling over a 10-year period using daily probability
# matrices derived from WHACS hindcast data, computing the distribution of days lost
# to non-workable sea conditions across Monte Carlo runs.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from collections import defaultdict


class ReefSimulation:
    def __init__(self, data_folder="new_WHACS_Extracted_multiyear"):
        self.data_folder = data_folder
        self.probability_data = None
        self.reefs = None
        self.days_per_year = 365
        self.num_years = 10
        self.total_days = self.days_per_year * self.num_years
        self.num_boats = 6

    def load_probability_data(self):
        print("Loading probability data...")

        all_data = defaultdict(list)

        csv_files = glob(os.path.join(self.data_folder, "*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_folder}")

        for file_path in csv_files:
            filename = os.path.basename(file_path)
            month_day = filename.replace('.csv', '')

            try:
                df = pd.read_csv(file_path)
                if 'probability' not in df.columns:
                    print(f"Warning: 'probability' column not found in {filename}")
                    continue

                if 'lat' in df.columns and 'lon' in df.columns:
                    df['reef_id'] = df['lat'].astype(str) + '_' + df['lon'].astype(str)
                else:
                    df['reef_id'] = range(len(df))

                all_data[month_day].append(df[['reef_id', 'probability']])

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

        all_reefs = set()
        for month_day_data in all_data.values():
            for yearly_data in month_day_data:
                all_reefs.update(yearly_data['reef_id'].unique())

        self.reefs = sorted(list(all_reefs))
        print(f"Found {len(self.reefs)} unique reefs")

        W = np.zeros((self.days_per_year, len(self.reefs)))

        for month_day, yearly_data_list in all_data.items():
            try:
                month, day = map(int, month_day.split('-'))
                day_of_year = self._get_day_of_year(month, day)

                reef_probs = defaultdict(list)
                for yearly_data in yearly_data_list:
                    for _, row in yearly_data.iterrows():
                        reef_probs[row['reef_id']].append(row['probability'])

                for reef_id, probs in reef_probs.items():
                    if reef_id in self.reefs:
                        reef_idx = self.reefs.index(reef_id)
                        W[day_of_year - 1, reef_idx] = np.mean(probs)

            except Exception as e:
                print(f"Error processing {month_day}: {e}")
                continue

        self.probability_data = W
        print(f"Created probability matrix W: {W.shape}")
        return W

    def _get_day_of_year(self, month, day):
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return sum(days_in_month[:month - 1]) + day

    def create_boat_schedule_matrix(self):
        print("Creating boat schedule matrix...")

        travel_days = 3
        work_days_per_reef = 5
        reefs_per_cycle = 3
        total_work_days = work_days_per_reef * reefs_per_cycle
        off_days = 7
        cycle_length = travel_days + total_work_days + off_days

        B = np.zeros((self.total_days, len(self.reefs)))

        for boat in range(self.num_boats):
            start_offset = boat * (cycle_length // self.num_boats)

            for day in range(self.total_days):
                cycle_day = (day + start_offset) % cycle_length

                if travel_days <= cycle_day < travel_days + total_work_days:
                    work_day_in_cycle = cycle_day - travel_days
                    reef_number = work_day_in_cycle // work_days_per_reef

                    cycle_start_day = day - cycle_day
                    selected_reefs = np.random.choice(
                        len(self.reefs),
                        size=min(reefs_per_cycle, len(self.reefs)),
                        replace=False
                    )

                    if reef_number < len(selected_reefs):
                        B[day, selected_reefs[reef_number]] = 1

        return B

    def simulate_days_lost_distribution(self, num_simulations=1000, threshold=0.75):
        print(f"Running {num_simulations} simulations to calculate days lost distribution...")

        if self.probability_data is None:
            raise ValueError("Probability data not loaded. Call load_probability_data() first.")

        W_extended = np.tile(self.probability_data, (self.num_years, 1))

        percent_days_lost = np.zeros(num_simulations)

        for sim in range(num_simulations):
            if sim % 100 == 0:
                print(f"Simulation {sim + 1}/{num_simulations}")

            B = self.create_boat_schedule_matrix()
            BD = B * W_extended

            total_scheduled_days = np.sum(B > 0)
            workable_days = np.sum(BD > threshold)
            days_lost = total_scheduled_days - workable_days

            if total_scheduled_days > 0:
                percent_days_lost[sim] = (days_lost / total_scheduled_days) * 100
            else:
                percent_days_lost[sim] = 0

        return percent_days_lost

    def visualize_days_lost_distribution(self, percent_days_lost, threshold=0.75):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        n_bins = 30
        counts, bins, patches = ax.hist(
            percent_days_lost, bins=n_bins,
            alpha=0.7, color='steelblue',
            edgecolor='black', linewidth=0.5
        )

        ax.set_xlabel('Percent of days lost to unworkable conditions', fontsize=12)
        ax.set_ylabel('Percent of simulations', fontsize=12)
        ax.set_title(
            f'Distribution of Days Lost to Non-Workable Conditions\n'
            f'(Threshold: {threshold}, {len(percent_days_lost)} simulations)',
            fontsize=14, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)

        mean_loss = np.mean(percent_days_lost)
        median_loss = np.median(percent_days_lost)

        ax.axvline(mean_loss, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_loss:.1f}%')
        ax.axvline(median_loss, color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {median_loss:.1f}%')

        p25 = np.percentile(percent_days_lost, 25)
        p75 = np.percentile(percent_days_lost, 75)
        ax.axvline(p25, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'25th percentile: {p25:.1f}%')
        ax.axvline(p75, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'75th percentile: {p75:.1f}%')

        ax.legend()

        stats_text = (
            f"Statistics:\n"
            f"Mean: {mean_loss:.2f}%\n"
            f"Median: {median_loss:.2f}%\n"
            f"Std Dev: {np.std(percent_days_lost):.2f}%\n"
            f"Min: {np.min(percent_days_lost):.2f}%\n"
            f"Max: {np.max(percent_days_lost):.2f}%\n\n"
            f"Percentiles:\n"
            f"10th: {np.percentile(percent_days_lost, 10):.1f}%\n"
            f"25th: {p25:.1f}%\n"
            f"75th: {p75:.1f}%\n"
            f"90th: {np.percentile(percent_days_lost, 90):.1f}%"
        )

        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, fontfamily='monospace')

        plt.tight_layout()
        plt.show()

        print("\n=== DAYS LOST DISTRIBUTION ANALYSIS ===")
        print(f"Threshold: {threshold}")
        print(f"Number of simulations: {len(percent_days_lost)}")
        print(f"Fleet size: {self.num_boats} boats")
        print(f"Simulation period: {self.num_years} years ({self.total_days} days)")

        print(f"\nDistribution of percent days lost to non-workable conditions:")
        print(f"  Mean: {mean_loss:.2f}%")
        print(f"  Median: {median_loss:.2f}%")
        print(f"  Standard deviation: {np.std(percent_days_lost):.2f}%")
        print(f"  Minimum: {np.min(percent_days_lost):.2f}%")
        print(f"  Maximum: {np.max(percent_days_lost):.2f}%")

        print(f"\nPercentile breakdown:")
        for p in [5, 10, 25, 50, 75, 90, 95]:
            value = np.percentile(percent_days_lost, p)
            print(f"  {p}th percentile: {value:.2f}%")

        print(f"\nRisk analysis:")
        for high_threshold in [20, 30]:
            count = np.sum(percent_days_lost > high_threshold)
            print(f"  Probability of losing >{high_threshold}% of days: {count / len(percent_days_lost) * 100:.1f}%")

        for low_threshold in [5, 2]:
            count = np.sum(percent_days_lost < low_threshold)
            print(f"  Probability of losing <{low_threshold}% of days: {count / len(percent_days_lost) * 100:.1f}%")

        return percent_days_lost


def run_days_lost_analysis():
    sim = ReefSimulation("new_WHACS_Extracted_multiyear")

    try:
        W = sim.load_probability_data()
        percent_days_lost = sim.simulate_days_lost_distribution(num_simulations=5000, threshold=0.75)
        sim.visualize_days_lost_distribution(percent_days_lost, threshold=0.75)
        return percent_days_lost

    except Exception as e:
        print(f"Error during simulation: {e}")
        print("Please ensure your CSV files are in the correct format and location.")


if __name__ == "__main__":
    percent_days_lost = run_days_lost_analysis()
