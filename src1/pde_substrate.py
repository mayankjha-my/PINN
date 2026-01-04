import torch
from .utils import grad, grad2


def substrate_residual(model, x, y, t, p):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    out = model(torch.cat([x, y, t], dim=1))
    w = out[:, 4:5]
    phi = out[:, 5:6]

    rho = p["rho"]
    sigma = p["sigma22"]
    G = p["G"]
    eta = p["eta"]

    # --- spatial derivatives of displacement ---
    w_x = grad(w, x)
    w_y = grad(w, y)

    # --- mixed time derivatives (Kelvin–Voigt damping) ---
    w_xt = grad(w_x, t)
    w_yt = grad(w_y, t)

    # --- shear stresses ---
    tau_zx = G * w_x + eta * w_xt
    tau_zy = G * w_y + eta * w_yt

    # --- divergence of stresses ---
    tau_zx_x = grad(tau_zx, x)
    tau_zy_y = grad(tau_zy, y)

    # --- time acceleration term ---
    w_tt = grad2(w, t)

    # --- second spatial derivative along y (prestress term) ---
    w_yy = grad2(w, y)

    # --- electric field laplacian ---
    phi_xx = grad2(phi, x)
    phi_yy = grad2(phi, y)

    # --- mechanical residual ---
    mech = tau_zx_x + tau_zy_y + sigma * w_yy - rho * w_tt

    # --- electrical residual ---
    elec = phi_xx + phi_yy

    return mech.pow(2).mean(), elec.pow(2).mean()
