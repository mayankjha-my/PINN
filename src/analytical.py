import numpy as np
from scipy.optimize import fsolve

# --- material parameters (set same as paper & PINN) ---
mu0 = 1.0
rho0 = 1.0
a = 1.0     # heterogeneity in layer
b = 0.5     # heterogeneity in halfspace
H = 1.0


def dispersion_equation(c, w):
    """
    This should encode your analytical dispersion function F(c,w)=0
    Replace the inside with your exact derived formula.
    """

    k = w / c

    # Example placeholder — replace with your Equation (26)
    # This ensures phase velocity depends on depth-varying mu
    F = np.tan(k * H) - np.sqrt((rho0 / mu0)) * c

    return F


def analytical_velocity(w):
    c0 = 1.0
    c = fsolve(dispersion_equation, c0, args=(w,))
    return c[0]
