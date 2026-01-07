"""
Configuration file for constants & material properties.
You will later replace placeholder values with actual data.
"""

CONFIG = {
    "FGPM": {
        "C44": None,
        "rho": None,
        "sigma22": None,
        "e15": None,
        "eps11": None,
        "alpha": None,
    },
    "HYDROGEL": {
        "C44": None,
        "rho": None,
        "sigma22": None,
        "eps11": None,
        "F": None,
        "Zf": None,
        "Cf": None,
    },
    "SUBSTRATE": {
        "rho": None,
        "sigma22": None,
    },
   "GEOMETRY": {
    "h1": 1.0,      # FGPM thickness
    "h2": 2.0,      # Hydrogel thickness
    "h3": 5.0,      # Substrate depth (optional)
    "y_min": -1.0,
    "y_max": 1.0,
    "t_min": 0.0,
    "t_max": 1.0
}

}
