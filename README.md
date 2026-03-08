# Modeling Predator-Prey Systems


There are three components of this code:

- **`MonteCarlo.py`**: Implements a faithful recreation of the algorithm described by Dobramysl et al. (https://arxiv.org/pdf/1708.07055) to simulate the Lotka–Volterra predator–prey dynamics on a 2D lattice using a time-stepped Monte Carlo approach.  
- **`Gillespie.py`**: Contains our continuous-time adaptation of the model, which incorporates explicit diffusion for both predator and prey populations, allowing exploration of different diffusive regimes.  
- **`Plotting.py`**: Provides functions for visualizing simulation results. Note that figures for the Monte Carlo simulations are generated directly within `MonteCarlo.py`, while `Plotting.py` is primarily used for analysis and visualizations of the continuous-time simulations.
