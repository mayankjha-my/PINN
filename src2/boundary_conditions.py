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


    z_top = z_top.clone().detach().requires_grad_(True)


    pred = model_layer(z_top)
    V_R = pred[:, 0:1]
    V_I = pred[:, 1:2]

    V_R_z = gradients(V_R, z_top)
    V_I_z = gradients(V_I, z_top)
    mu44_0= material_params["mu44_0"]
    beta1 = material_params["beta1"]
    mu44 = mu44_0 * torch.exp(beta1 * z_top)
    tau_R = (mu44 * V_R_z)/mu44_0
    tau_I = (mu44 * V_I_z)/mu44_0

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

    z_int = z_int.clone().detach().requires_grad_(True)

    # layer fields
    pred_l = model_layer(z_int)
    pred_h = model_half(z_int)   # Half-space output

    
    V_R = pred_l[:, 0:1]
    V_I = pred_l[:, 1:2]
   # Half-space: handle both cases (1 or 2 outputs)
    if pred_h.shape[1] == 2:
        # Half-space outputs 2 values (complex)
        V_R_h =  pred_h[:, 0:1]  # Real part of half-space
        V_I_h =  pred_h[:, 1:2]  # Imag part of half-space
    else:
        # Half-space outputs 1 value (assumed real)
        V_R_h =  pred_h           # Real part = output
        V_I_h = torch.zeros_like(V_R_h)  # Imag part = 0
  

    # derivatives
    V_R_z = gradients(V_R, z_int)
    V_I_z = gradients(V_I, z_int)
    V_R_z_h = gradients(V_R_h, z_int)
    V_I_z_h = gradients(V_I_h, z_int) if pred_h.shape[1] == 2 else torch.zeros_like(V_R_z_h)

    # graded shear moduli
    mu44_l0 = params_layer["mu44_0"]
    beta1   = params_layer["beta1"]

    mu44_h0 = params_half["mu44_0"]
    beta2   = params_half["beta2"]

    mu44_l = mu44_l0 * torch.exp(beta1 * z_int)
    mu44_h = mu44_h0 * torch.exp(beta2 * z_int)

    return (
        V_R - V_R_h,                    # Real displacement continuity
        V_I - V_I_h,                    # Imag displacement continuity
        (mu44_l * V_R_z - mu44_h * V_R_z_h) / mu44_l0,  # Real stress continuity
        ((mu44_l /mu44_l0)* V_I_z - (mu44_h/mu44_l0) * V_I_z_h)   # Imag stress continuity
    )


# --------------------------------------------------
# Far-field boundary condition (z = 10)
# --------------------------------------------------
def halfspace_far_field_bc(model_half, z_far):
    """
    Half-space decay condition:
    V -> 0 as z -> infinity (z = 10)
    """
    
    V = model_half(z_far)


    return V
