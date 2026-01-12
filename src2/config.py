# """
# Configuration file for material properties and geometry
# for SH-wave dispersion analysis using PINNs
# """

# CONFIG = {

#     # --------------------------------------------------
#     # Functionally Graded Layer (FGPM)
#     # --------------------------------------------------
#     "LAYER": {
#         "mu44_0": 4.35e9,          # Pa
#         "mu66_0": 5e9,          # Pa
#         "rho_0": 9890.0,           # kg/m^3
#         "P_0": 4.0e7,              # Initial stress (Pa)
#         "beta1": 0.7,            # FG parameter

#         # Electromagnetic parameters
#         "mu_e": 100,  # Magnetic permeability (H/m)
#         "H0": 1.0,
#         "phi": 0.7853981633974483   # pi/4 in radians



#     },

#     # --------------------------------------------------
#     # Functionally Graded Half-Space (Substrate)
#     # --------------------------------------------------
#     "HALFSPACE": {
#         "mu44_0": 5.3e9,           # Pa
#         "mu66_0": 6.47e9,           # Pa
#         "rho_0": 3400.0,            # kg/m^3
#         "P_0": 4.0e7,               # Initial stress (Pa)
#         "beta2": 0.7,             # FG parameter

#         # Gravity
#         "g": 9.81                   # m/s^2
#     },

#     # --------------------------------------------------
#     # Geometry & Dispersion Settings
#     # --------------------------------------------------
#     "GEOMETRY": {
#         "H": 1.0,                   # Non-dimensional layer thickness
#         "L": 10.0,                  # Truncated half-space depth (10H)

#         # Wavenumber sweep (non-dimensional)
#         "k_min": 0.1,
#         "k_max": 1.0,
#         "num_k": 40
#     },

#     # --------------------------------------------------
#     # Training Parameters
#     # --------------------------------------------------
#     "TRAINING": {
#         "epochs": 20000,
#         "learning_rate": 5e-4,
#         "loss_weights": {
#     "pde": 1.0,
#     "bc": 0.1,
#     "interface": 0.01,
#     "far": 0.001
#     }

#     }
# }



# Reference values
H_ref = 1.0  # reference thickness (can be 1.0 for simplicity)
mu44_0_ref = 4.35e9  # reference shear modulus (Pa)
rho_0_ref = 9890.0   # reference density (kg/m^3)
c0_ref = (mu44_0_ref / rho_0_ref) ** 0.5  # reference velocity

CONFIG = {
    "LAYER": {
        "mu44_0": 1.0,  # non-dimensional (mu44_0 / mu44_0_ref)
        "mu66_0": 5e9 / mu44_0_ref,
        "rho_0": 1.0,   # non-dimensional (rho_0 / rho_0_ref)
        "P_0": 4.0e7 / mu44_0_ref,
        "beta1": 0.7 * H_ref,  # if beta1 has units 1/length, multiply by H_ref
        "mu_e": 100 / mu44_0_ref,
        "H0": 1.0 / H_ref,  # if H0 has units of length
        "phi": 0.7853981633974483
    },
    "HALFSPACE": {
        "mu44_0": 5.3e9 / mu44_0_ref,
        "mu66_0": 6.47e9 / mu44_0_ref,
        "rho_0": 3400.0 / rho_0_ref,
        "P_0": 4.0e7 / mu44_0_ref,
        "beta2": 0.7 * H_ref,
        "g": 9.81 * H_ref / c0_ref**2  # non-dimensional gravity
    },
    "GEOMETRY": {
        "H": 1.0,  # non-dimensional
        "L": 10.0, # non-dimensional
        "k_min": 0.1,  # already non-dimensional (k*H)
        "k_max": 1.0,
        "num_k": 40
    },
    #     # --------------------------------------------------
    # Training Parameters
    # --------------------------------------------------
    "TRAINING": {
        "epochs": 20000,
        "learning_rate": 5e-4,
        "loss_weights": {
    "pde": 1.0,
    "bc": 0.1,
    "interface": 0.01,
    "far": 0.001
    }

    }
}