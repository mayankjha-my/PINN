import torch
from .utils import gradients


# --------------------------------------------------
# Top surface boundary condition (z = -1)
# tau_23 = 0
# --------------------------------------------------
def top_surface_bc(model_layer, z_top, material_params):
    """
    Stress-free top surface:
    tau_23 = mu44 * dV/dz = 0
    Applied to both real and imaginary parts
    """

    z_top.requires_grad_(True)

    pred = model_layer(z_top)
    scale = 1e-2
    V_R = scale * pred[:, 0:1]
    V_I = scale * pred[:, 1:2]

    V_R_z = gradients(V_R, z_top)
    V_I_z = gradients(V_I, z_top)

    beta1 = material_params["beta1"]
    mu44  = material_params["mu44_0"] * (1.0 + torch.sin(beta1 * z_top))

    tau_R = (mu44 * V_R_z)
    tau_I = mu44 * V_I_z

    return tau_R, tau_I


# --------------------------------------------------
# Interface between layer and half-space (z = 0)
# --------------------------------------------------
def interface_layer_halfspace(model_layer, model_half, z_int,
                              params_layer, params_half):
    """
    Interface conditions:
    V_R^l = V^h
    V_I^l = 0
    mu44^l * dV_R^l/dz = mu44^h * dV^h/dz
    mu44^l * dV_I^l/dz = 0
    """

    z_int.requires_grad_(True)

    # layer fields
    pred_l = model_layer(z_int)
    scale = 1e-2
    V_R = scale * pred_l[:, 0:1]
    V_I = scale * pred_l[:, 1:2]
    V_h = scale * model_half(z_int)
  

    # derivatives
    V_R_z = gradients(V_R, z_int)
    V_I_z = gradients(V_I, z_int)
    V_h_z = gradients(V_h, z_int)

    # graded shear moduli
    beta1 = params_layer["beta1"]
    beta2 = params_half["beta2"]

    mu44_l = params_layer["mu44_0"] * (1.0 + torch.sin(beta1 * z_int))
    mu44_h = params_half["mu44_0"] * (1.0 - torch.sin(beta2 * z_int))

    return (
        V_R - V_h,                      # displacement continuity
        V_I,                            # imaginary part vanishes
        mu44_l * V_R_z - mu44_h * V_h_z,  # stress continuity (real)
        mu44_l * V_I_z                  # imaginary stress = 0
    )


# --------------------------------------------------
# Far-field boundary condition (z = 10)
# --------------------------------------------------
def halfspace_far_field_bc(model_half, z_far):
    """
    Half-space decay condition:
    V -> 0 as z -> infinity (z = 10)
    """

    scale = 1e-2
    V = scale * model_half(z_far)


    return V
