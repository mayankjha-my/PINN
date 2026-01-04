import torch
from .utils import grad


def top_surface_bc(model, p, x, y, t, mode="open"):

    # ---- enable gradients ----
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    out = model(torch.cat([x, y, t], dim=1))

    w1 = out[:, 0:1]
    phi1 = out[:, 1:2]

    # ---- mechanical traction-free ----
    wx = grad(w1, x)

    tau = p["C44"] * wx   # simplified tau_xz = C44 * dw/dx

    loss = tau.pow(2).mean()

    # ---- electric condition ----
    if mode == "open":
        Dx = grad(phi1, x)
        loss += Dx.pow(2).mean()

    elif mode == "short":
        loss += phi1.pow(2).mean()

    return loss


def interface_loss(model, x, y, t):

    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    out = model(torch.cat([x, y, t], dim=1))

    w1 = out[:, 0:1]
    phi1 = out[:, 1:2]

    w2 = out[:, 2:3]
    phi2 = out[:, 3:4]

    w3 = out[:, 4:5]
    phi3 = out[:, 5:6]

    loss = 0.0

    loss += (w1 - w2).pow(2).mean()
    loss += (phi1 - phi2).pow(2).mean()
    loss += (w2 - w3).pow(2).mean()
    loss += (phi2 - phi3).pow(2).mean()

    return loss
