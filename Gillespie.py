import numpy as np
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

COLOR_PALLETTE = ["#8fd7d7", "#00b0be", "#ff8ca1", "#f45f74", "#bdd373", "#98c127", "#ffcd8e", "#ffb255"]

POPULATION_MAX = 1_000_000  # threshold for population explosion. This didn't end up working very well.

def run_2d_diffusion_ssa(params):

    D_H, D_L, run = params

    # PARAMETERS
    L = 100 # lattice of 100
    T_max = 100 # we ended up redicuing this, initially we were running 300 but it as taking too long
    sigma = 1
    mu = 1
    lam = 0.5
    K = 100

    output_dir = f"/project/bios26211/alissa_eleonore/results/Gillespie_reduced_time/run_{run}"
    csv_path = f"{output_dir}/time_series_csv/timeseries_DH_{D_H:.2f}_DL_{D_L:.2f}.csv.gz"

    # skip if CSV already exists (we added this part after our first round of simulations since we were trying 
    # to re-simulate the scenarios that never finished)
    if os.path.exists(csv_path):
        print(f"Skipping D_H={D_H:.2f}, D_L={D_L:.2f}, run={run} (already exists)")
        return (D_H, D_L, None)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/snapshots", exist_ok=True)
    os.makedirs(f"{output_dir}/time_series_csv", exist_ok=True)
    os.makedirs(f"{output_dir}/time_series_plots", exist_ok=True)

    # INITIAL CONDITIONS 
    prey = np.ones((L, L), dtype=np.int32) # using int32 to take up less memory. We were worried about going over the 500GB
    pred = np.random.poisson(0.01, size=(L, L)).astype(np.int32)

    # RATE FUNCTION (putting this in a function so we don't have to recalculate it every time)
    def compute_site_rate(i, j):
        a = pred[i, j]
        b = prey[i, j]
        return (
            mu * a +
            lam * a * b +
            sigma * b +
            D_H * b +
            D_L * a
        )

    # initialize site rates
    site_rates = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            site_rates[i, j] = compute_site_rate(i, j)

    row_sums = site_rates.sum(axis=1)
    R_total = row_sums.sum()

    # SIMULATION STATE
    t = 0.0
    prey_history = []
    pred_history = []
    time_history = []

    #snapshot_times = [10, 20, 30, 40, 50, 100, 200, 300, 500]
    #snapshots = {}
    #next_snapshot_index = 0
    extinction_time = None
    step_counter = 0

    # GILLESPIE LOOP
    while t < T_max and R_total > 0:
        # get the time to the next event
        dt = -np.log(np.random.rand()) / R_total
        t += dt

        # select which site the event occurs at
        r = np.random.rand() * R_total  # we take a random number between 0 and R_total
        row_cumsum = np.cumsum(row_sums) # calculate each row's contribution to R_total
        i = np.searchsorted(row_cumsum, r)  # select the row r falls on as the row where an event occurs

        # now we need to get the mathing column
        r_local = r - (row_cumsum[i - 1] if i > 0 else 0.0)  # here, we shift our r so it’s relative to the start of row i. this gives a number between 0 and the total rate of row i
        col_cumsum = np.cumsum(site_rates[i]) # now we calculate cumulative sum of column rates in row i
        j = np.searchsorted(col_cumsum, r_local) # pick the column in row i where the random number falls

        # now we have the site the reaction occurs at
        a = pred[i, j] # predator at that site
        b = prey[i, j] # prey at that site

        # get the rates based on local prey and predator numbers
        r_death  = mu * a
        r_pred   = lam * a * b
        r_birth  = sigma * b
        r_diff_H = D_H * b
        r_diff_L = D_L * a

        # calculate the total local total rate
        local_total = r_death + r_pred + r_birth + r_diff_H + r_diff_L
        r2 = np.random.rand() * local_total

        changed_sites = [(i, j)]

        # APPLY EVENT
        if r2 < r_death:
            pred[i, j] -= 1
        elif r2 < r_death + r_pred:
            prey[i, j] -= 1
            pred[i, j] += 1
        elif r2 < r_death + r_pred + r_birth:
            prey[i, j] = min(prey[i, j] + 1, K)
        elif r2 < r_death + r_pred + r_birth + r_diff_H:
            if b > 0:
                di, dj = [(1,0),(-1,0),(0,1),(0,-1)][np.random.randint(4)]
                ni, nj = (i + di) % L, (j + dj) % L
                prey[i, j] -= 1
                prey[ni, nj] += 1
                changed_sites.append((ni, nj))
        else:
            if a > 0:
                di, dj = [(1,0),(-1,0),(0,1),(0,-1)][np.random.randint(4)]
                ni, nj = (i + di) % L, (j + dj) % L
                pred[i, j] -= 1
                pred[ni, nj] += 1
                changed_sites.append((ni, nj))
        # UPDATE ONLY THE AFFECTED SITES (this is for performance so it runs faster)
        for (x, y) in changed_sites:
            old_rate = site_rates[x, y]
            new_rate = compute_site_rate(x, y)
            site_rates[x, y] = new_rate
            row_sums[x] += (new_rate - old_rate)
            R_total += (new_rate - old_rate)

        # RECORD POPULATIONS
        prey_total = prey.sum()
        pred_total = pred.sum()
        prey_history.append(prey_total)
        pred_history.append(pred_total)
        time_history.append(t)
        step_counter += 1

        # CHECK FOR EXPLOSIONS OR EXTINCTION
        if prey_total > POPULATION_MAX or pred_total > POPULATION_MAX:
            extinction_time = t
            print(f"Population explosion detected at t={t:.2f}")
            break
        if prey_total == 0 or pred_total == 0:
            extinction_time = t
            print(f"Extinction detected at t={t:.2f}")
            break

        # SNAPSHOTS (was blowing up runtime too much so we stopped doing this) 
        """
        if next_snapshot_index < len(snapshot_times) and t >= snapshot_times[next_snapshot_index]:
            snap_time = snapshot_times[next_snapshot_index]
            state = np.zeros((L, L), dtype=np.int8)
            state[(prey > 0)] = 1
            state[(pred > 0)] += 2
            snapshots[snap_time] = state.copy()
            next_snapshot_index += 1
        """

    # FINAL CSV SAVE
    df = pd.DataFrame({
        "time": time_history,
        "prey": prey_history,
        "predator": pred_history
    })
    df.to_csv(csv_path, index=False, compression="gzip")

    # PLOT TIME SERIES
    plt.figure(figsize=(6, 4))
    plt.plot(time_history, prey_history, label="Prey", color=COLOR_PALLETTE[5])
    plt.plot(time_history, pred_history, label="Predator", color=COLOR_PALLETTE[7])
    plt.xlabel("Time (continuous)")
    plt.ylabel("Population")
    plt.title(f"SSA: D_H={D_H:.2f}, D_L={D_L:.2f}")
    if extinction_time is not None:
        plt.axvline(extinction_time, linestyle="--", color="black")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_series_plots/timeseries_DH_{D_H:.2f}_DL_{D_L:.2f}.png")
    plt.close()

    return (D_H, D_L, extinction_time)

if __name__ == "__main__":
    repeat_id = int(sys.argv[1])
    n_cores = int(sys.argv[2])

    diffusion_values = np.linspace(0, 1.0, 11)
    param_list = [(DH, DL, repeat_id) for DH in diffusion_values for DL in diffusion_values]

    with mp.Pool(n_cores) as pool:
        results = pool.map(run_2d_diffusion_ssa, param_list)

    print(f"Finished repeat {repeat_id}")
