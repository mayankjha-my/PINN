import torch
from .utils import grad, grad2


def fgpm_residual(model, x, y, t, p):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    out = model(torch.cat([x, y, t], dim=1))
    w = out[:, 0:1]
    phi = out[:, 1:2]

    w_x = grad(w, x)
    w_y = grad(w, y)
    w_xx = grad2(w, x)
    w_yy = grad2(w, y)
    w_tt = grad2(w, t)

    phi_x = grad(phi, x)
    phi_y = grad(phi, y)
    phi_xx = grad2(phi, x)
    phi_yy = grad2(phi, y)

    C44 = p["C44"]
    sig0 = p["sigma22"]
    e15 = p["e15"]
    eps11 = p["eps11"]
    rho = p["rho"]
    a = p["alpha"]

    mech = (
        C44 * w_xx
        + (C44 + sig0) * w_yy
        + a * C44 * w_x
        + e15 * (phi_xx + phi_yy + a * phi_x)
        - rho * w_tt
    )

    elec = (
        e15 * (w_xx + w_yy + a * w_x)
        - eps11 * (phi_xx + phi_yy + a * phi_x)
    )

    return mech.pow(2).mean(), elec.pow(2).mean()
