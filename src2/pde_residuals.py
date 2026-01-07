import torch
from utils import gradients


def residual_medium1(model, xyt, params):
    """
    FGPM governing equations
    """

    xyt.requires_grad_(True)

    pred = model(xyt)
    w = pred[:, 0:1]
    phi = pred[:, 1:2]

    # first derivatives
    grads_w = gradients(w, xyt)
    grads_phi = gradients(phi, xyt)

    w_x = grads_w[:, 0:1]
    w_y = grads_w[:, 1:2]
    w_t = grads_w[:, 2:3]

    phi_x = grads_phi[:, 0:1]
    phi_y = grads_phi[:, 1:2]

    # second derivatives
    w_xx = gradients(w_x, xyt)[:, 0:1]
    w_yy = gradients(w_y, xyt)[:, 1:2]
    w_tt = gradients(w_t, xyt)[:, 2:3]

    phi_xx = gradients(phi_x, xyt)[:, 0:1]
    phi_yy = gradients(phi_y, xyt)[:, 1:2]

    # material constants
    C44 = params["C44"]
    rho = params["rho"]
    sig22 = params["sigma22"]
    e15 = params["e15"]
    eps11 = params["eps11"]
    alpha = params["alpha"]

    # ------------------ Residual 1 ------------------
    r1 = (
        C44 * w_xx
        + (C44 + sig22) * w_yy
        + alpha * C44 * w_x
        + e15 * (phi_xx + phi_yy + alpha * phi_x)
        - rho * w_tt
    )

    # ------------------ Residual 2 ------------------
    r2 = (
        e15 * (w_xx + w_yy + alpha * w_x)
        - eps11 * (phi_xx + phi_yy + alpha * phi_x)
    )

    return r1, r2


def residual_medium2(model, xyt, params):
    """
    Hydrogel governing equations
    """

    xyt.requires_grad_(True)

    pred = model(xyt)
    w = pred[:, 0:1]
    phi = pred[:, 1:2]

    # first & second derivatives
    grads_w = gradients(w, xyt)
    w_x = grads_w[:, 0:1]
    w_y = grads_w[:, 1:2]
    w_t = grads_w[:, 2:3]

    w_xx = gradients(w_x, xyt)[:, 0:1]
    w_yy = gradients(w_y, xyt)[:, 1:2]
    w_tt = gradients(w_t, xyt)[:, 2:3]

    grads_phi = gradients(phi, xyt)
    phi_x = grads_phi[:, 0:1]
    phi_y = grads_phi[:, 1:2]

    phi_xx = gradients(phi_x, xyt)[:, 0:1]
    phi_yy = gradients(phi_y, xyt)[:, 1:2]

    # material parameters
    C44 = params["C44"]
    rho = params["rho"]
    sig22 = params["sigma22"]
    eps11 = params["eps11"]

    F = params["F"]
    Zf = params["Zf"]
    Cf = params["Cf"]

    # ------------------ Residual 1 ------------------
    r1 = (
        w_xx
        + ((C44 + sig22) / C44) * w_yy
        - (rho / C44) * w_tt
    )

    # ------------------ Residual 2 ------------------
    rhs = -(F * Zf * Cf) / eps11

    r2 = phi_xx + phi_yy - rhs

    return r1, r2


def residual_medium3(model, xyt, params):
    """
    Viscoelastic substrate governing equations
    """

    xyt.requires_grad_(True)

    pred = model(xyt)
    w = pred[:, 0:1]
    phi = pred[:, 1:2]

    # derivatives
    grads_w = gradients(w, xyt)
    w_x = grads_w[:, 0:1]
    w_y = grads_w[:, 1:2]
    w_t = grads_w[:, 2:3]

    w_xx = gradients(w_x, xyt)[:, 0:1]
    w_yy = gradients(w_y, xyt)[:, 1:2]
    w_tt = gradients(w_t, xyt)[:, 2:3]

    grads_phi = gradients(phi, xyt)
    phi_x = grads_phi[:, 0:1]
    phi_y = grads_phi[:, 1:2]

    phi_xx = gradients(phi_x, xyt)[:, 0:1]
    phi_yy = gradients(phi_y, xyt)[:, 1:2]

    # material params
    rho = params["rho"]
    sig22 = params["sigma22"]

    # ------------------ Residual 1 ------------------
    # tau_zx and tau_zy will later be computed constitutively.
    # For now we use the PDE form:
    r1 = (
        w_xx
        + w_yy
        + (sig22) * w_yy
        - rho * w_tt
    )

    # ------------------ Residual 2 ------------------
    r2 = phi_xx + phi_yy

    return r1, r2
