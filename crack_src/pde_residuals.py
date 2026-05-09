import torch

# --------------------------------------------------
# Gradient function (same style as your code)
# --------------------------------------------------
def gradients(u, x):
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]


# --------------------------------------------------
# Residual for PIEZOELECTRIC CRACK PROBLEM
# --------------------------------------------------
def residual_piezo(model, x1, x3, params):
    """
    PINN residual for coupled piezoelectric PDE system
    """

    x1 = x1.clone().detach().requires_grad_(True)
    x3 = x3.clone().detach().requires_grad_(True)

    # ---------------------------------
    # Forward pass
    # ---------------------------------
    X = torch.cat([x1, x3], dim=1)
    X.requires_grad_(True)

    U = model(X)

    u1 = U[:, 0:1]
    u3 = U[:, 1:2]
    phi = U[:, 2:3]

    # ---------------------------------
    # First derivatives
    # ---------------------------------
    u1_x1 = gradients(u1, x1)
    u1_x3 = gradients(u1, x3)

    u3_x1 = gradients(u3, x1)
    u3_x3 = gradients(u3, x3)

    phi_x1 = gradients(phi, x1)
    phi_x3 = gradients(phi, x3)

    # ---------------------------------
    # Second derivatives
    # ---------------------------------
    u1_x1x1 = gradients(u1_x1, x1)
    u1_x3x3 = gradients(u1_x3, x3)
    u1_x1x3 = gradients(u1_x1, x3)

    u3_x1x1 = gradients(u3_x1, x1)
    u3_x3x3 = gradients(u3_x3, x3)
    u3_x1x3 = gradients(u3_x1, x3)

    phi_x1x1 = gradients(phi_x1, x1)
    phi_x3x3 = gradients(phi_x3, x3)
    phi_x1x3 = gradients(phi_x1, x3)

    # ---------------------------------
    # Parameters
    # ---------------------------------
    mu11 = params["mu11"]
    mu33 = params["mu33"]
    mu44 = params["mu44"]
    mu13 = params["mu13"]

    e15 = params["e15"]
    e31 = params["e31"]
    e33 = params["e33"]

    eps11 = params["eps11"]
    eps33 = params["eps33"]

    # ---------------------------------
    # REAL PDE SYSTEM
    # ---------------------------------

    # Equation 1
    r1 = (
        mu11 * u1_x1x1
        + mu44 * u1_x3x3
        + (mu13 + mu44) * u3_x1x3
        + (e31 + e15) * phi_x1x3
    )

    # Equation 2
    r2 = (
        mu44 * u3_x1x1
        + mu33 * u3_x3x3
        + (mu13 + mu44) * u1_x1x3
        + e15 * phi_x1x1
        + e33 * phi_x3x3
    )

    # Equation 3 (electric)
    r3 = (
        e15 * u3_x1x1
        + e33 * u3_x3x3
        + (e15 + e31) * u1_x1x3
        - eps11 * phi_x1x1
        - eps33 * phi_x3x3
    )

    

    return r1, r2, r3



# --------------------------------------------------
# TEMPERATURE PDE (simple heat equation first)
# --------------------------------------------------
def residual_temperature(model, x3, t, params):

    x3 = x3.clone().detach().requires_grad_(True)
    t  = t.clone().detach().requires_grad_(True)

    X = torch.cat([x3, t], dim=1)
    X.requires_grad_(True)

    U = model(X)

    T = U[:, 3:4]   # fourth output

    # derivatives
    T_x3 = gradients(T, x3)
    T_x3x3 = gradients(T_x3, x3)

    T_t = gradients(T, t)

    lambda0 = params["lambda0"]

    rT = T_x3x3 - (1.0 / lambda0) * T_t

    return rT