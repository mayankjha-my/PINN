import torch
import torch.nn as nn

from .pde_residuals import residual_piezo, residual_temperature
from .boundary_conditions import (
    crack_normal_stress_bc,
    displacement_bc,
    shear_bc,
    electric_bc_D1,
    surface_bc,
    temperature_ic,
    temperature_bc_left,
    temperature_bc_right
)

mse = nn.MSELoss()

# --------------------------------------------------
# PDE LOSS
# --------------------------------------------------
def compute_pde_loss(model, x1, x3, params):

    r1, r2, r3 = residual_piezo(model, x1, x3, params)

    loss = (
        mse(r1, torch.zeros_like(r1)) +
        mse(r2, torch.zeros_like(r2)) +
        mse(r3, torch.zeros_like(r3))
    )

    return loss


# --------------------------------------------------
# TEMPERATURE PDE LOSS
# --------------------------------------------------
def compute_temperature_loss(model, x3, t, params):

    rT = residual_temperature(model, x3, t, params)

    return mse(rT, torch.zeros_like(rT))


# --------------------------------------------------
# CRACK BC LOSS
# --------------------------------------------------
def compute_crack_loss(model, x1, x3, params, tau0, tauT):

    r = crack_normal_stress_bc(model, x1, x3, params, tau0, tauT)

    return mse(r, torch.zeros_like(r))


# --------------------------------------------------
# DISPLACEMENT BC
# --------------------------------------------------
def compute_disp_loss(model, x1, x3):

    r = displacement_bc(model, x1, x3)

    return mse(r, torch.zeros_like(r))


# --------------------------------------------------
# SHEAR BC
# --------------------------------------------------
def compute_shear_loss(model, x1, x3, params):

    r = shear_bc(model, x1, x3, params)

    return mse(r, torch.zeros_like(r))


# --------------------------------------------------
# ELECTRIC BC
# --------------------------------------------------
def compute_electric_loss(model, x1, x3, params):

    r = electric_bc_D1(model, x1, x3, params)

    return mse(r, torch.zeros_like(r))


# --------------------------------------------------
# SURFACE BC
# --------------------------------------------------
def compute_surface_loss(model, x1, x3, params):

    r1, r2, r3 = surface_bc(model, x1, x3, params)

    return (
        mse(r1, torch.zeros_like(r1)) +
        mse(r2, torch.zeros_like(r2)) +
        mse(r3, torch.zeros_like(r3))
    )


# --------------------------------------------------
# TEMPERATURE BC LOSS
# --------------------------------------------------
def compute_temperature_bc(model, x3, t, params):

    T0 = params["T0"]

    loss_ic = mse(
        temperature_ic(model, x3, torch.zeros_like(t)),
        torch.zeros_like(x3)
    )

    loss_left = mse(
        temperature_bc_left(model, torch.zeros_like(x3), t, T0),
        torch.zeros_like(x3)
    )

    loss_right = mse(
        temperature_bc_right(model, x3, t),
        torch.zeros_like(x3)
    )

    return loss_ic + loss_left + loss_right


# --------------------------------------------------
# TOTAL LOSS
# --------------------------------------------------
def total_loss(
    model,
    x1,
    x3,
    t,
    params,
    tau0,
    tauT,
    weights
):

    loss_pde = compute_pde_loss(model, x1, x3, params)

    loss_temp = compute_temperature_loss(model, x3, t, params)

    loss_crack = compute_crack_loss(model, x1, x3, params, tau0, tauT)

    loss_disp = compute_disp_loss(model, x1, x3)

    loss_shear = compute_shear_loss(model, x1, x3, params)

    loss_elec = compute_electric_loss(model, x1, x3, params)

    loss_surface = compute_surface_loss(model, x1, x3, params)

    loss_temp_bc = compute_temperature_bc(model, x3, t, params)

    total = (
        weights["pde"] * loss_pde +
        weights["temp"] * loss_temp +
        weights["crack"] * loss_crack +
        weights["bc"] * (loss_disp + loss_shear + loss_elec + loss_surface) +
        weights["temp_bc"] * loss_temp_bc
    )

    return total, {
        "pde": loss_pde.item(),
        "temp": loss_temp.item(),
        "crack": loss_crack.item(),
        "bc": (loss_disp + loss_shear + loss_elec + loss_surface).item(),
        "temp_bc": loss_temp_bc.item()
    }