import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

COLOR_PALLETTE = ["#8fd7d7", "#00b0be", "#ff8ca1", "#f45f74",
                  "#bdd373", "#98c127", "#ffcd8e", "#ffb255"]

def run_2d_faithful(save_path):
    """
    This function is the actual faithful implementation of the algorithm used to generate Figure 3,
    at least as far as we can be sure since we don't get all the details on the model. This is a type
    of Kinetic Monte-Carlo algorithm with local rate normalization and L^2 update attempts at every step. 
    However, they model diffusion only through birth and predation events, where hares give birth to neighboring
    lattice points and predation occurs in neighboring lattice points, so they do not have explicit diffusion 
    parameters. Therefore, we will implement a slightly altered version called run_2d_diffusion where we 
    add parameters D_H and D_L that represent lynx and hare diffusion respectively. 
    """
    L = 250
    MCS_total = 5000
    sigma = 0.1
    mu = 0.9
    lam = 1.0

    # initial conditions
    prey = np.ones((L, L), dtype=np.int32)
    pred = np.random.poisson(0.01, size=(L, L)).astype(np.int32)

    # snapshots
    snapshot_times = [17, 30, 71, 500, 2000, 4000, 4017, 4030, 4071, 4500, 5000]
    snapshots = {}

    # time series
    timeseries = {'prey': [], 'pred': []}

    # one MC step
    def step():
        for _ in range(L * L):
            i = np.random.randint(L)
            j = np.random.randint(L)
            # get the current predator-prey numbers
            a = pred[i, j]
            b = prey[i, j]
            # if the current location is empty we just keep going
            if a == 0 and b == 0:
                continue

            # reaction rates
            r_death = mu * a
            r_pred  = lam * a * b
            r_birth = sigma * b

            # sample reaction rates similar to Gillespie method, so with relative rates
            R = r_death + r_pred + r_birth
            if R == 0:
                continue
            r = np.random.rand() * R
            
            # predator death
            if r < r_death:
                pred[i, j] -= 1
            
            # predation
            elif r < r_death + r_pred:
                prey[i, j] -= 1
                di, dj = [(1,0),(-1,0),(0,1),(0,-1)][np.random.randint(4)]
                ni, nj = (i + di) % L, (j + dj) % L
                pred[ni, nj] += 1
            
            # prey reproduction
            else:
                di, dj = [(1,0),(-1,0),(0,1),(0,-1)][np.random.randint(4)]
                ni, nj = (i + di) % L, (j + dj) % L
                prey[ni, nj] += 1

    # simulation
    for t in range(1, MCS_total + 1):
        print(t)
        step()
        # record time series
        timeseries['prey'].append(prey.sum())
        timeseries['pred'].append(pred.sum())

        # record snapshots
        if t in snapshot_times:
            state = np.zeros((L, L), dtype=np.int8)
            state[(prey > 0)] = 1
            state[(pred > 0)] += 2
            snapshots[t] = state.copy()

    # save data 
    np.savez_compressed(save_path, snapshots=snapshots, prey_series=timeseries['prey'], pred_series=timeseries['pred'])

    return snapshots, timeseries


# Function to plot snapshots
def plot_snapshots(snapshots, path):
    """Plotting the snapshots to recreate Figure 3"""
    cmap = plt.cm.colors.ListedColormap([
        "black",            # empty
        COLOR_PALLETTE[5],  # prey
        COLOR_PALLETTE[7],  # predator
        COLOR_PALLETTE[1]   # both
    ])
    # only take the first four snapshots
    snapshot_items = list(snapshots.items())[:4]

    fig, axes = plt.subplots(1, len(snapshot_items), figsize=(4*len(snapshot_items), 5))
    if len(snapshot_items) == 1:
        axes = [axes]

    for ax, (t, state) in zip(axes, snapshot_items):

        ax.imshow(state, cmap=cmap)

        # time label 
        ax.text(0.03, 0.95, f"t = {t}", transform=ax.transAxes, color="white", fontsize=20, ha="left", va="top")
        ax.axis("off")

    plt.suptitle("2D Stochastic LV Model Snapshots")
    plt.tight_layout()

    plt.savefig(Path(path) / "snapshots.png", dpi=300)
    plt.close()

def plot_timeseries(prey_series, pred_series, path):
    """Plotting the timeseries of the simulated data"""
    fig, ax1 = plt.subplots(figsize=(8,5))

    # prey axis
    ax1.plot(prey_series[500:800], color=COLOR_PALLETTE[5])
    ax1.set_xlabel("Monte Carlo Step")
    ax1.set_ylabel("Prey Population")
    ax1.tick_params(axis='y')

    # predator axis
    ax2 = ax1.twinx()
    ax2.plot(pred_series[500:800], color=COLOR_PALLETTE[7])
    ax2.set_ylabel("Predator Population")
    ax2.tick_params(axis='y')

    # create a legend for both plots
    legend_elements = [
        Patch(facecolor="black", label="Empty"),
        Patch(facecolor=COLOR_PALLETTE[5], label="Prey"),
        Patch(facecolor=COLOR_PALLETTE[7], label="Predator"),
        Patch(facecolor=COLOR_PALLETTE[1], label="Both")
    ]

    # place legend below the plot
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))

    plt.title("Predator–Prey Dynamics Over Time")
    plt.tight_layout()
    plt.savefig(Path(path) / "timeseries.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_phase(prey_series, pred_series, path):
    """Plotting phase diagram"""
    plt.figure(figsize=(6,6))
    prey = prey_series[500:800]
    pred = pred_series[500:800]

    plt.plot(prey, pred, color=COLOR_PALLETTE[1], linewidth=1)

    # add direction arrow
    step = len(prey) // 100
    for i in range(step, len(prey)-1, step):
        plt.annotate(
            "",
            xy=(prey[i+1], pred[i+1]),
            xytext=(prey[i], pred[i]),
            arrowprops=dict(
                arrowstyle="->",
                color=COLOR_PALLETTE[1],
                lw=1
            )
        )

    plt.xlabel("Prey Population")
    plt.ylabel("Predator Population")
    plt.title("Predator–Prey Phase Plot")
    plt.tight_layout()
    plt.savefig(Path(path) / "phase_plot.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    path = "/Users/alissadomenig/Documents/Documents - Alissa’s MacBook Pro/BIOS/Math Methods/Final Project/Figures/Figure1/"
    run_2d_faithful(path)

    # load the saved simulation
    data = np.load(Path(path) / "data.npz", allow_pickle=True)
    snapshots = data["snapshots"].item()     
    prey_series = data["prey_series"]
    pred_series = data["pred_series"]

    # plot
    plot_snapshots(snapshots, path)
    plot_timeseries(prey_series, pred_series, path)
    plot_phase(prey_series, pred_series, path)
