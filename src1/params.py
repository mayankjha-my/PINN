import torch

params = {
    "fgpm": {
        "C44": 1.0,
        "sigma22": 0.0,
        "e15": 0.2,
        "eps11": 1.0,
        "rho": 1.0,
        "alpha": 0.0,
    },
    "hydrogel": {
        "C44": 0.8,
        "sigma22": 0.0,
        "rho": 1.0,
        "eps11": 1.0,
        "FZfCf": 0.1,
    },
   "substrate": {
    "rho": 1.0,
    "sigma22": 0.0,
    "G": 1.0,
    "eta": 0.2,   # <-- new
},
    "geom": {
        "h1": 1.0,
        "h2": 1.0,
    },
}
