import torch
from .utils import gradients


# --------------------------------------------------
# BC1: Crack faces (a < x3 < b)
# tau_11 = prescribed traction
# --------------------------------------------------
def crack_normal_stress_bc(model, x1, x3, params, tau0, tauT):

    x1 = x1.clone().detach().requires_grad_(True)
    x3 = x3.clone().detach().requires_grad_(True)

    X = torch.cat([x1, x3], dim=1)
    X.requires_grad_(True)

    U = model(X)

    u1 = U[:, 0:1]
    u3 = U[:, 1:2]
    phi = U[:, 2:3]

    # derivatives
    u1_x1 = gradients(u1, x1)
    u3_x3 = gradients(u3, x3)
    phi_x3 = gradients(phi, x3)

    # parameters
    mu11 = params["mu11"]
    mu13 = params["mu13"]
    e31  = params["e31"]
    kappa = params["kappa"]

    # stress
    T = U[:, 3:4]

    tau_11 = (
    mu11 * u1_x1
    + mu13 * u3_x3
    + e31 * phi_x3
    - kappa * T  
)

    # residual
    return tau_11 + tau0 + tauT(x3)


# --------------------------------------------------
# BC2: No displacement outside crack
# u1 = 0
# --------------------------------------------------
def displacement_bc(model, x1, x3):

    X = torch.cat([x1, x3], dim=1)
    U = model(X)

    u1 = U[:, 0:1]

    return u1


# --------------------------------------------------
# BC3: Shear-free
# tau_13 = 0
# --------------------------------------------------
def shear_bc(model, x1, x3, params):

    x1 = x1.clone().detach().requires_grad_(True)
    x3 = x3.clone().detach().requires_grad_(True)

    X = torch.cat([x1, x3], dim=1)
    X.requires_grad_(True)

    U = model(X)

    u1 = U[:, 0:1]
    u3 = U[:, 1:2]

    u1_x3 = gradients(u1, x3)
    u3_x1 = gradients(u3, x1)

    mu44 = params["mu44"]

    tau_13 = mu44 * (u1_x3 + u3_x1)

    return tau_13


# --------------------------------------------------
# BC4: Electric condition
# D1 = 0
# --------------------------------------------------
def electric_bc_D1(model, x1, x3, params):

    x1 = x1.clone().detach().requires_grad_(True)
    x3 = x3.clone().detach().requires_grad_(True)

    X = torch.cat([x1, x3], dim=1)
    X.requires_grad_(True)

    U = model(X)

    u1 = U[:, 0:1]
    u3 = U[:, 1:2]
    phi = U[:, 2:3]

    # derivatives
    u1_x1 = gradients(u1, x1)
    u3_x3 = gradients(u3, x3)
    phi_x1 = gradients(phi, x1)

    # parameters
    e15 = params["e15"]
    e31 = params["e31"]
    eps11 = params["eps11"]

    D1 = (
        e15 * u3_x1 if 'u3_x1' in locals() else 0
    ) + (
        e31 * u1_x1
        - eps11 * phi_x1
    )

    return D1


# --------------------------------------------------
# BC5: Top & Bottom surfaces
# tau_13 = 0, tau_33 = 0, D3 = 0
# --------------------------------------------------
def surface_bc(model, x1, x3, params):

    x1 = x1.clone().detach().requires_grad_(True)
    x3 = x3.clone().detach().requires_grad_(True)

    X = torch.cat([x1, x3], dim=1)
    X.requires_grad_(True)

    U = model(X)

    u1 = U[:, 0:1]
    u3 = U[:, 1:2]
    phi = U[:, 2:3]

    # derivatives
    u1_x3 = gradients(u1, x3)
    u3_x3 = gradients(u3, x3)
    u3_x1 = gradients(u3, x1)
    phi_x3 = gradients(phi, x3)

    # parameters
    mu44 = params["mu44"]
    mu33 = params["mu33"]
    mu13 = params["mu13"]
    e33  = params["e33"]
    eps33 = params["eps33"]

    # stresses
    tau_13 = mu44 * (u1_x3 + u3_x1)
    tau_33 = mu33 * u3_x3 + mu13 * gradients(u1, x1) + e33 * phi_x3

    # electric displacement
    D3 = -eps33 * phi_x3

    return tau_13, tau_33, D3
# --------------------------------------------------
# TEMPERATURE IC: T(x3,0) = 0
# --------------------------------------------------
def temperature_ic(model, x3, t0):

    X = torch.cat([x3, t0], dim=1)
    U = model(X)

    T = U[:, 3:4]

    return T


# --------------------------------------------------
# T(0,t) = T0
# --------------------------------------------------
def temperature_bc_left(model, x3_0, t, T0):

    X = torch.cat([x3_0, t], dim=1)
    U = model(X)

    T = U[:, 3:4]

    return T - T0


# --------------------------------------------------
# T(H,t) = 0
# --------------------------------------------------
def temperature_bc_right(model, x3_H, t):

    X = torch.cat([x3_H, t], dim=1)
    U = model(X)

    T = U[:, 3:4]

    return T