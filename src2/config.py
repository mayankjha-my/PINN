"""
Configuration file for material properties and geometry
for SH-wave dispersion analysis using PINNs
"""

CONFIG = {

    # --------------------------------------------------
    # Functionally Graded Layer (FGPM)
    # --------------------------------------------------
    "LAYER": {
        "mu44_0": 4.35e9,          # Pa
        "mu66_0": 5e9,          # Pa
        "rho_0": 9890.0,           # kg/m^3
        "P_0": 4.0e7,              # Initial stress (Pa)
        "beta1": 0.007,            # FG parameter

        # Electromagnetic parameters
        "mu_e": 1,  # Magnetic permeability (H/m)
        "H0": 1.0,                  # Magnetic field intensity
        "phi": 0.78539816339        # Inclination angle (rad) = 45°
    },

    # --------------------------------------------------
    # Functionally Graded Half-Space (Substrate)
    # --------------------------------------------------
    "HALFSPACE": {
        "mu44_0": 5.3e9,           # Pa
        "mu66_0": 6.47e9,           # Pa
        "rho_0": 3400.0,            # kg/m^3
        "P_0": 4.0e7,               # Initial stress (Pa)
        "beta2": 0.007,             # FG parameter

        # Gravity
        "g": 9.81                   # m/s^2
    },

    # --------------------------------------------------
    # Geometry & Dispersion Settings
    # --------------------------------------------------
    "GEOMETRY": {
        "H": 1.0,                   # Non-dimensional layer thickness
        "L": 100.0,                  # Truncated half-space depth (10H)

        # Wavenumber sweep (non-dimensional)
        "k_min": 0.1,
        "k_max": 5.0,
        "num_k": 40
    },

    # --------------------------------------------------
    # Training Parameters
    # --------------------------------------------------
    "TRAINING": {
        "epochs": 20000,
        "learning_rate": 1e-3,
        "loss_weights": {
            "pde": 1.0,
            "bc": 10.0,
            "interface": 10.0
        }
    }
}
