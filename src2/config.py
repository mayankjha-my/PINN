"""
Configuration file for constants & material properties.
You will later replace placeholder values with actual data.
"""

CONFIG = {
    "FGPM": {
        "C44": 2.56*10**9,
        "rho": 7500,
        "sigma22": 4*10**7,
        "e15": 12.7,
        "eps11": 646*10**-11,
        "alpha": 0.007,
    },
    "HYDROGEL": {
        "C44": 384.61,
        "rho": 7280,
        "sigma22": 4,
        "eps11": 8.8542*10**-12,
        "F": 96485.3399,
        "Zf": 1,
        "Cf": 1.05,
    },
   "SUBSTRATE": {
    "rho": 1570,                # kg/m^3
    "sigma22": 4e7,             # Pa
    "C44": 3.98*10**9,               # Pa (example – use your material)
    "eta44": 4*10**7,             # Pa·s (example – use your material)
    "alpha3":0.007,
},

   "GEOMETRY": {
    "h1": 5.0,      # FGPM thickness
    "h2": 5.0,      # Hydrogel thickness
    "h3": 30.0,      # Substrate depth (optional)
    "y_min": -1.0,
    "y_max": 3.0,
    "t_min": 0.0,
    "t_max": 6.0
}

}
