import torch
import torch.nn as nn

from .pde_residuals import (
    residual_layer_coupled,
    residual_halfspace
)

from .boundary_conditions import (
    top_surface_bc,
    interface_layer_halfspace,
    halfspace_far_field_bc
)

mse = nn.MSELoss()

# --------------------------------------------------
# PDE loss
# --------------------------------------------------
def compute_pde_loss(
    model_layer,
    model_half,
    z_layer,
    z_half,
    params_layer,
    params_half,
    k,
    c
):
    """
    PDE residual loss for layer and half-space
    """

    rL_R, rL_I = residual_layer_coupled(
        model_layer, z_layer, params_layer, k, c
    )

    rH = residual_halfspace(
        model_half, z_half, params_half, k, c
    )

    loss_pde = (
    mse(rL_R, torch.zeros_like(rL_R)) / (rL_R.abs().mean().detach() + 1e-6) +
    mse(rL_I, torch.zeros_like(rL_I)) / (rL_I.abs().mean().detach() + 1e-6) +
    mse(rH,   torch.zeros_like(rH))   / (rH.abs().mean().detach()   + 1e-6)
   )


    return loss_pde


# --------------------------------------------------
# Top surface boundary loss
# --------------------------------------------------
def compute_top_surface_loss(model_layer, z_top, params_layer):
    """
    Stress-free top surface
    """

    tau_R, tau_I = top_surface_bc(model_layer, z_top, params_layer)

    loss_bc = (
        mse(tau_R, torch.zeros_like(tau_R)) +
        mse(tau_I, torch.zeros_like(tau_I))
    )

    return loss_bc


# --------------------------------------------------
# Interface loss (layer ↔ half-space)
# --------------------------------------------------
def compute_interface_loss(
    model_layer,
    model_half,
    z_int,
    params_layer,
    params_half
):
    """
    Continuity conditions at z = 0
    """

    r1, r2, r3, r4 = interface_layer_halfspace(
        model_layer, model_half, z_int,
        params_layer, params_half
    )

    loss_int = (
    mse(r1, torch.zeros_like(r1)) +
    mse(r2, torch.zeros_like(r2)) +
    0.1 * mse(r3, torch.zeros_like(r3)) +
    0.1 * mse(r4, torch.zeros_like(r4))
   )


    return loss_int


# --------------------------------------------------
# Far-field loss
# --------------------------------------------------
def compute_far_field_loss(model_half, z_far):
    """
    Half-space decay condition
    """

    V = halfspace_far_field_bc(model_half, z_far)

    loss_far = mse(V, torch.zeros_like(V))

    return loss_far


# --------------------------------------------------
# Total loss
# --------------------------------------------------
def total_loss(
    model_layer,
    model_half,
    z_layer,
    z_half,
    z_top,
    z_int,
    z_far,
    params_layer,
    params_half,
    k,
    c,
    w_pde = 1.0,
    w_bc  = 0.1,
    w_int = 0.01,
    w_far = 0.001

):
    """
    Total PINN loss for dispersion analysis
    """

    loss_pde = compute_pde_loss(
        model_layer,
        model_half,
        z_layer,
        z_half,
        params_layer,
        params_half,
        k,
        c
    )

    loss_bc = compute_top_surface_loss(
        model_layer, z_top, params_layer
    )

    loss_int = compute_interface_loss(
        model_layer,
        model_half,
        z_int,
        params_layer,
        params_half
    )

    loss_far = compute_far_field_loss(
        model_half, z_far
    )


    # Monotonicity penalty for phase velocity c
    # If c is a tensor or batch, penalize increases
    def monotonicity_penalty(c_value):
        # If c is a single value, penalty is zero
        if not hasattr(c_value, '__len__') or len(c_value) < 2:
            return 0.0
        diffs = c_value[1:] - c_value[:-1]
        penalty = torch.sum(torch.relu(diffs))
        return penalty

    # If you want to enforce monotonicity over a batch, pass a batch of c values
    # Here, c is a single value per call, so penalty is zero
    alpha = 0.1  # Weight for monotonicity penalty
    mono_penalty = monotonicity_penalty(torch.tensor([c]))

    loss_total = (
        w_pde * loss_pde +
        w_bc  * loss_bc +
        w_int * loss_int +
        w_far * loss_far +
        alpha * mono_penalty
    )

    return loss_total, {
        "pde": loss_pde.item(),
        "bc_top": loss_bc.item(),
        "interface": loss_int.item(),
        "far": loss_far.item(),
        "mono_penalty": mono_penalty.item() if hasattr(mono_penalty, 'item') else mono_penalty
    }
