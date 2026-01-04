import torch
from .utils import grad, grad2


def hydrogel_residual(model, x, y, t, p):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    out = model(torch.cat([x, y, t], dim=1))
    w = out[:, 2:3]
    phi = out[:, 3:4]

    w_xx = grad2(w, x)
    w_yy = grad2(w, y)
    w_tt = grad2(w, t)

    phi_xx = grad2(phi, x)
    phi_yy = grad2(phi, y)

    C44 = p["C44"]
    sigma = p["sigma22"]
    rho = p["rho"]
    rhs = p["FZfCf"] / p["eps11"]

    mech = (
        w_xx
        + ((C44 + sigma) / C44) * w_yy
        - (rho / C44) * w_tt
    )

    elec = phi_xx + phi_yy + rhs

    return mech.pow(2).mean(), elec.pow(2).mean()
