import sys
import re
import gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from pathlib import Path
from multiprocessing import Pool
from collections import defaultdict

# SETTINGS
BASE_DIR = Path("/project/bios26211/alissa_eleonore/results/Gillespie")
RUN_PREFIX = "run_"

EXTINCTION_HEATMAP_DIR = BASE_DIR / "extinction_heatmap"
PREDATOR_HEATMAP_DIR = BASE_DIR / "heatmaps_predator"
PREY_HEATMAP_DIR = BASE_DIR / "heatmaps_prey"

EXTINCTION_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
PREDATOR_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
PREY_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

PATTERN = re.compile(r".*DH_(\d+\.\d+)_DL_(\d+\.\d+)")
SNAPSHOTS = np.arange(0, 101, 1)
TIME_RANGE = np.arange(20, 101, 1)  # we only want to plot after stabilizing 
num_workers = int(sys.argv[1])

# FILE PROCESSING
def process_csv(file_path_str):
    file_path = Path(file_path_str)
    match = PATTERN.search(file_path.name)
    if not match:
        return None

    DH = float(match.group(1))
    DL = float(match.group(2))

    try:
        with gzip.open(file_path, "rt") as f:
            df = pd.read_csv(f, usecols=["time", "prey", "predator"])
    except Exception as e:
        print("Skipping corrupted file:", file_path, e)
        return None

    times = df["time"].values
    prey = df["prey"].values
    predator = df["predator"].values

    extinct = int(prey[-1] == 0 or predator[-1] == 0)

    populations = {}
    for t in SNAPSHOTS:
        idx = np.searchsorted(times, t, side="right") - 1
        idx = max(idx, 0)
        populations[t] = (prey[idx], predator[idx])

    return (DH, DL, extinct, populations)

# EXTINCTION HEATMAP
def plot_extinction_heatmap(results):
    results_filtered = [r for r in results if r[0] != 0 and r[1] != 0]

    df_ext = pd.DataFrame(
        [(r[0], r[1], r[2]) for r in results_filtered],
        columns=["DH", "DL", "extinct"]
    )

    DH_values = sorted(df_ext["DH"].unique())
    DL_values = sorted(df_ext["DL"].unique())
    nH, nL = len(DH_values), len(DL_values)

    ext_prob = np.zeros((nH, nL))
    grouped = df_ext.groupby(["DH", "DL"])["extinct"].mean()

    for i, DH in enumerate(DH_values):
        for j, DL in enumerate(DL_values):
            if (DH, DL) in grouped:
                ext_prob[i, j] = grouped[(DH, DL)]

    plt.figure(figsize=(6, 5))
    im = plt.imshow(ext_prob, origin="lower", extent=[min(DL_values), max(DL_values), min(DH_values), max(DH_values)], aspect="auto", vmin=0, vmax=1, cmap="Reds")
    plt.colorbar(im, label="Extinction Probability")
    plt.xlabel("D_L")
    plt.ylabel("D_H")
    plt.title("Extinction Probability Across Runs")
    plt.tight_layout()
    plt.savefig(EXTINCTION_HEATMAP_DIR / "extinction_probability.png")
    plt.close()
    print("Extinction heatmap saved.")

# POPULATION HEATMAPS
def plot_population_heatmaps(results):
    results_filtered = [r for r in results if r[0] != 0 and r[1] != 0]

    data_dict = defaultdict(list)
    for DH, DL, _, populations in results_filtered:
        data_dict[(DH, DL)].append(populations)

    DH_values = sorted(set(k[0] for k in data_dict))
    DL_values = sorted(set(k[1] for k in data_dict))
    nH, nL = len(DH_values), len(DL_values)

    # compute average grids for all t 
    prey_grids = {}
    pred_grids = {}

    for t in TIME_RANGE:
        prey_grid = np.full((nH, nL), np.nan)
        pred_grid = np.full((nH, nL), np.nan)

        for i, DH in enumerate(DH_values):
            for j, DL in enumerate(DL_values):
                pops = data_dict.get((DH, DL))
                if not pops:
                    continue
                prey_vals = [p[t][0] for p in pops]
                pred_vals = [p[t][1] for p in pops]
                prey_grid[i, j] = np.mean(prey_vals)
                pred_grid[i, j] = np.mean(pred_vals)

        prey_grids[t] = prey_grid
        pred_grids[t] = pred_grid

    # get the global min/max of averaged populations for the color scale
    prey_min = min(np.nanmin(grid) for grid in prey_grids.values())
    prey_max = max(np.nanmax(grid) for grid in prey_grids.values())
    pred_min = min(np.nanmin(grid) for grid in pred_grids.values())
    pred_max = max(np.nanmax(grid) for grid in pred_grids.values())

    for t in TIME_RANGE:
        prey_grid = prey_grids[t]
        pred_grid = pred_grids[t]

        # PREY
        plt.figure(figsize=(6, 5))
        im = plt.imshow(prey_grid, origin="lower", extent=[min(DL_values), max(DL_values), min(DH_values), max(DH_values)], aspect="auto", cmap="Greens", vmin=prey_min, vmax=prey_max)
        plt.colorbar(im, label=f"Average prey population at t={t}")
        plt.xlabel("D_L")
        plt.ylabel("D_H")
        plt.title(f"Average prey population at t={t}")
        plt.tight_layout()
        plt.savefig(PREY_HEATMAP_DIR / f"prey_t{t}.png")
        plt.close()

        # PREDATOR
        plt.figure(figsize=(6, 5))
        im = plt.imshow(pred_grid, origin="lower", extent=[min(DL_values), max(DL_values), min(DH_values), max(DH_values)], aspect="auto", cmap="Oranges", vmin=pred_min, vmax=pred_max)
        plt.colorbar(im, label=f"Average predator population at t={t}")
        plt.xlabel("D_L")
        plt.ylabel("D_H")
        plt.title(f"Average predator population at t={t}")
        plt.tight_layout()
        plt.savefig(PREDATOR_HEATMAP_DIR / f"predator_t{t}.png")
        plt.close()

        if t % 10 == 0:
            print(f"Saved heatmaps for t={t}")


# MAIN
if __name__ == "__main__":
    all_csv_files = []
    missing_files_dict = defaultdict(list)

    for run_dir in sorted(BASE_DIR.glob(f"{RUN_PREFIX}*")):
        csv_dir = run_dir / "time_series_csv"
        if not csv_dir.exists():
            print(f"Directory missing: {csv_dir}")
            continue

        expected_files = list(csv_dir.glob("timeseries_DH_*_DL_*.csv.gz"))
        all_csv_files.extend(str(f) for f in expected_files if f.exists())

    print("Total CSV files found:", len(all_csv_files))

    chunksize = max(50, len(all_csv_files) // (num_workers * 4))
    with Pool(num_workers) as pool:
        results = pool.map(process_csv, all_csv_files, chunksize=chunksize)

    results = [r for r in results if r is not None]
    print("Processed files:", len(results))

    plot_extinction_heatmap(results)
    plot_population_heatmaps(results)
