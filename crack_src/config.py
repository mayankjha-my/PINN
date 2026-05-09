"""
Configuration file for thermo–piezoelectric crack problem using PINNs
"""

CONFIG = {

    # --------------------------------------------------
    # PIEZOELECTRIC MATERIAL PROPERTIES
    # --------------------------------------------------
    "MATERIAL": {

        # Elastic constants (Pa)
        "mu11": 1.0e10,
        "mu33": 1.0e10,
        "mu44": 0.5e10,
        "mu13": 0.3e10,

        # Piezoelectric constants (C/m^2)
        "e15": 12.0,
        "e31": -5.0,
        "e33": 15.0,

        # Dielectric constants (F/m)
        "eps11": 1.0e-8,
        "eps33": 1.0e-8,
    },


    # --------------------------------------------------
    # GEOMETRY (2D DOMAIN + CRACK)
    # --------------------------------------------------
    "GEOMETRY": {

        # Domain size
        "x1_min": -1.0,
        "x1_max": 1.0,

        "x3_min": -1.0,
        "x3_max": 1.0,

        # Crack definition
        "crack_length": 0.5,
        "crack_tip_x1": 0.5,
        "crack_tip_x3": 0.0,
    },


    # --------------------------------------------------
    # TRAINING PARAMETERS
    # --------------------------------------------------
    "TRAINING": {

        "epochs": 20000,
        "learning_rate": 5e-4,

        "loss_weights": {
            "pde": 1.0,
            "bc": 1.0,
            "crack": 1.0,
        }
    },


    # --------------------------------------------------
    # SAMPLING
    # --------------------------------------------------
    "SAMPLING": {

        "num_domain": 5000,
        "num_boundary": 1000,
        "num_crack": 1000,
    },

    
    "THERMAL": {
        "lambda0": 1.0,
        "T0": 1.0
    }
}
