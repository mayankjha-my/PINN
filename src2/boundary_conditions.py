import torch
from .utils import gradients


def top_surface_open_bc(model1, xt_top, material_params):
    """
    Electrically OPEN top surface:
    tau_xz = 0
    Dx = 0
    """

    xt_top.requires_grad_(True)
    # extract x-coordinate at boundary
    x = xt_top[:, 0:1]


    pred = model1(xt_top)
    w = pred[:, 0:1]
    phi = pred[:, 1:2]

    grads_w = gradients(w, xt_top)
    w_x = grads_w[:, 0:1]

    grads_phi = gradients(phi, xt_top)
    phi_x = grads_phi[:, 0:1]

    # FGPM material parameters
    alpha = material_params["alpha"]

    C44   = material_params["C44"]   * torch.exp(alpha * x)
    e15   = material_params["e15"]   * torch.exp(alpha * x)
    eps11 = material_params["eps11"] * torch.exp(alpha * x)

    C0 = material_params["C44"]
    E0 = material_params["e15"]  
    # shear stress tau_xz
    tau_xz = (C44 * w_x + e15 * phi_x) / C0

    # electric displacement
    Dx     = (e15 * w_x - eps11 * phi_x) / E0


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

    grads_phi = gradients(phi, xt_top)
    phi_x = grads_phi[:, 0:1]


      # extract x-coordinate at boundary
    x = xt_top[:, 0:1]

    # FGPM material parameters
    alpha = material_params["alpha"]

    C44   = material_params["C44"]   * torch.exp(alpha * x)
    e15   = material_params["e15"]   * torch.exp(alpha * x)
    C0 = material_params["C44"]


    tau_xz = (C44 * w_x + e15 * phi_x)/C0

    return tau_xz, phi



def interface_fgpm_hydro(model1, model2, xyt_int):
    """
    Interface at x = 0
    C¹ continuity in the NORMAL direction only
    """

    xyt_int.requires_grad_(True)

    pred1 = model1(xyt_int)
    pred2 = model2(xyt_int)

    w1, phi1 = pred1[:, 0:1], pred1[:, 1:2]
    w2, phi2 = pred2[:, 0:1], pred2[:, 1:2]

    grads_w1 = gradients(w1, xyt_int)
    grads_w2 = gradients(w2, xyt_int)

    grads_phi1 = gradients(phi1, xyt_int)
    grads_phi2 = gradients(phi2, xyt_int)

    # ONLY x-derivative continuity
    dw_dx   = grads_w1[:, 0:1] - grads_w2[:, 0:1]
    dw_dy   = grads_w1[:, 1:2] - grads_w2[:, 1:2]
    dphi_dx = grads_phi1[:, 0:1] - grads_phi2[:, 0:1]
    dphi_dy = grads_phi1[:, 1:2] - grads_phi2[:, 1:2]


    return (
        w1 - w2,        # displacement continuity
        phi1 - phi2,    # potential continuity
        dw_dx + dw_dy,          # normal strain continuity
        dphi_dx + dphi_dy         # normal electric field continuity
    )



def interface_hydro_substrate(model2, model3, xyt_int):
    """
    Interface at x = h2
    C¹ continuity in the normal (x) direction only
    """

    xyt_int.requires_grad_(True)

    pred2 = model2(xyt_int)
    pred3 = model3(xyt_int)

    w2, phi2 = pred2[:, 0:1], pred2[:, 1:2]
    w3, phi3 = pred3[:, 0:1], pred3[:, 1:2]

    grads_w2 = gradients(w2, xyt_int)
    grads_w3 = gradients(w3, xyt_int)

    grads_phi2 = gradients(phi2, xyt_int)
    grads_phi3 = gradients(phi3, xyt_int)

    # ONLY x-derivative continuity
    dw_dx   = grads_w2[:, 0:1] - grads_w3[:, 0:1]
    dw_dy   = grads_w2[:, 1:2] - grads_w3[:, 1:2]
    dphi_dx = grads_phi2[:, 0:1] - grads_phi3[:, 0:1]
    dphi_dy = grads_phi2[:, 1:2] - grads_phi3[:, 1:2]


    return (
        w2 - w3,        # displacement continuity
        phi2 - phi3,    # potential continuity
        dw_dx + dw_dy,          # normal strain continuity
        dphi_dx + dphi_dy        # normal electric field continuity
    )
def substrate_far_field_bc(model3, xyt_far):
    """
    Half-space decay condition:
    w → 0, phi → 0 as x → +∞
    """

    pred = model3(xyt_far)
    w   = pred[:, 0:1]
    phi = pred[:, 1:2]

    return w, phi

