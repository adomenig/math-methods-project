# Modeling Predator-Prey Systems


There are three components of this code:

- **`MonteCarlo.py`**: Implements a faithful recreation of the algorithm described by Dobramysl et al. (https://arxiv.org/pdf/1708.07055) to simulate the Lotka–Volterra predator–prey dynamics on a 2D lattice using a time-stepped Monte Carlo approach.  
- **`Gillespie.py`**: Contains our continuous-time adaptation of the model, which incorporates explicit diffusion for both predator and prey populations which allows exploring different diffusive regimes. This code was run on the clusters, so it will need to be adapted to be run locally but the core logic of the Gillespie algorithm is still identical.
- **`Plotting.py`**: Provides functions for visualizing simulation results. Figures for the Monte Carlo simulations are generated directly within `MonteCarlo.py`, so `Plotting.py` is primarily used for analysis and visualizations of the continuous-time simulations. This code was run on the clusters. 
