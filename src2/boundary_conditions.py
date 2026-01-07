import torch
from utils import gradients


def top_surface_open_bc(model1, xt_top, material_params):
    """
    Electrically OPEN top surface:
    tau_xz = 0
    Dx = 0
    """

    xt_top.requires_grad_(True)

    pred = model1(xt_top)
    w = pred[:, 0:1]
    phi = pred[:, 1:2]

    grads_w = gradients(w, xt_top)
    w_x = grads_w[:, 0:1]

    grads_phi = gradients(phi, xt_top)
    phi_x = grads_phi[:, 0:1]

    C44 = material_params["C44"]
    e15 = material_params["e15"]
    eps11 = material_params["eps11"]

    # shear stress tau_xz
    tau_xz = C44 * w_x + e15 * phi_x

    # electric displacement
    Dx = e15 * w_x - eps11 * phi_x

    return tau_xz, Dx



def top_surface_short_bc(model1, xt_top, material_params):
    """
    Electrically SHORT top surface:
    tau_xz = 0
    phi = 0
    """

    xt_top.requires_grad_(True)

    pred = model1(xt_top)
    w = pred[:, 0:1]
    phi = pred[:, 1:2]

    grads_w = gradients(w, xt_top)
    w_x = grads_w[:, 0:1]

    C44 = material_params["C44"]
    e15 = material_params["e15"]

    tau_xz = C44 * w_x + e15 * phi

    return tau_xz, phi



def interface_fgpm_hydro(model1, model2, xyt_int):
    """
    Enforces at x = 0

    w1 = w2
    phi1 = phi2
    grad(w1)=grad(w2)
    grad(phi1)=grad(phi2)
    """

    xyt_int.requires_grad_(True)

    pred1 = model1(xyt_int)
    pred2 = model2(xyt_int)

    w1 = pred1[:, 0:1]
    phi1 = pred1[:, 1:2]

    w2 = pred2[:, 0:1]
    phi2 = pred2[:, 1:2]

    grad_w1 = gradients(w1, xyt_int)
    grad_w2 = gradients(w2, xyt_int)

    grad_phi1 = gradients(phi1, xyt_int)
    grad_phi2 = gradients(phi2, xyt_int)

    return (
        w1 - w2,
        phi1 - phi2,
        grad_w1 - grad_w2,
        grad_phi1 - grad_phi2
    )



def interface_hydro_substrate(model2, model3, xyt_int):
    """
    Same continuity at x = h2
    """

    xyt_int.requires_grad_(True)

    pred2 = model2(xyt_int)
    pred3 = model3(xyt_int)

    w2 = pred2[:, 0:1]
    phi2 = pred2[:, 1:2]

    w3 = pred3[:, 0:1]
    phi3 = pred3[:, 1:2]

    grad_w2 = gradients(w2, xyt_int)
    grad_w3 = gradients(w3, xyt_int)

    grad_phi2 = gradients(phi2, xyt_int)
    grad_phi3 = gradients(phi3, xyt_int)

    return (
        w2 - w3,
        phi2 - phi3,
        grad_w2 - grad_w3,
        grad_phi2 - grad_phi3
    )
