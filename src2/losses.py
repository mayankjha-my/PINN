import torch
import torch.nn as nn

from .pde_residuals import (
    residual_medium1,
    residual_medium2,
    residual_medium3
)

from .boundary_conditions import (
    top_surface_open_bc,
    top_surface_short_bc,
    interface_fgpm_hydro,
    interface_hydro_substrate,
    substrate_far_field_bc
)


mse = nn.MSELoss()


def compute_pde_loss(net1, net2, net3,
                     xyt_fgpm, xyt_hydro, xyt_sub,
                     params_fgpm, params_hydro, params_sub):

    r1_fgpm, r2_fgpm = residual_medium1(net1, xyt_fgpm, params_fgpm)
    r1_hydro, r2_hydro = residual_medium2(net2, xyt_hydro, params_hydro)
    r1_sub, r2_sub = residual_medium3(net3, xyt_sub, params_sub)
   

    loss_pde = (
        mse(r1_fgpm, torch.zeros_like(r1_fgpm))
      + mse(r2_fgpm, torch.zeros_like(r2_fgpm))
      + mse(r1_hydro, torch.zeros_like(r1_hydro))
      + mse(r2_hydro, torch.zeros_like(r2_hydro))
      + mse(r1_sub, torch.zeros_like(r1_sub))
      + mse(r2_sub, torch.zeros_like(r2_sub))
    )


    return loss_pde

    
def compute_top_surface_loss_open(net1, xt_top, params_fgpm):

    tau_xz, Dx = top_surface_open_bc(net1, xt_top, params_fgpm)

    loss_bc = (
        mse(tau_xz, torch.zeros_like(tau_xz))
      + mse(Dx, torch.zeros_like(Dx))
    )

    return loss_bc



def compute_top_surface_loss_short(net1, xt_top, params_fgpm):

    tau_xz, phi = top_surface_short_bc(net1, xt_top, params_fgpm)

    loss_bc = (
        mse(tau_xz, torch.zeros_like(tau_xz))
      + mse(phi, torch.zeros_like(phi))
    )

    return loss_bc



# def compute_interface_loss_fgpm_hydro(net1, net2, xyt_int):

#     dw, dphi, dgw, dgphi = interface_fgpm_hydro(net1, net2, xyt_int)

#     loss_int = (
#         mse(dw, torch.zeros_like(dw))
#       + mse(dphi, torch.zeros_like(dphi))
#       + mse(dgw, torch.zeros_like(dgw))
#       + mse(dgphi, torch.zeros_like(dgphi))
#     )

#     return loss_int


def compute_interface_loss_fgpm_hydro(net1, net2, xyt_int,
                                     w_w=10.0, w_phi=1.0,
                                     w_grad_w=10.0, w_grad_phi=1.0):

    dw, dphi, dgw, dgphi = interface_fgpm_hydro(net1, net2, xyt_int)

    loss_int = (
        w_w * mse(dw, torch.zeros_like(dw))
      + w_phi * mse(dphi, torch.zeros_like(dphi))
      + w_grad_w * mse(dgw, torch.zeros_like(dgw))
      + w_grad_phi * mse(dgphi, torch.zeros_like(dgphi))
    )

    return loss_int



# def compute_interface_loss_hydro_sub(net2, net3, xyt_int):

#     dw, dphi, dgw, dgphi = interface_hydro_substrate(net2, net3, xyt_int)

#     loss_int = (
#         mse(dw, torch.zeros_like(dw))
#       + mse(dphi, torch.zeros_like(dphi))
#       + mse(dgw, torch.zeros_like(dgw))
#       + mse(dgphi, torch.zeros_like(dgphi))
#     )

#     return loss_int


def compute_interface_loss_hydro_sub(net2, net3, xyt_int,
                                    w_w=10.0, w_phi=1.0,
                                    w_grad_w=10.0, w_grad_phi=1.0):

    dw, dphi, dgw, dgphi = interface_hydro_substrate(net2, net3, xyt_int)

    loss_int = (
        w_w * mse(dw, torch.zeros_like(dw))
      + w_phi * mse(dphi, torch.zeros_like(dphi))
      + w_grad_w * mse(dgw, torch.zeros_like(dgw))
      + w_grad_phi * mse(dgphi, torch.zeros_like(dgphi))
    )

    return loss_int



def compute_far_field_loss(net3, xyt_far):

    w, phi = substrate_far_field_bc(net3, xyt_far)

    loss_far = (
        mse(w, torch.zeros_like(w))
      + mse(phi, torch.zeros_like(phi))
    )

    return loss_far



def total_loss(
    net1, net2, net3,
    xyt_fgpm, xyt_hydro, xyt_sub,
    xt_top,
    xyt_int_fgpm_hydro,
    xyt_int_hydro_sub,
    xyt_far_sub, 
    params_fgpm,
    params_hydro,
    params_sub,
    electrically_open=True,
    w_pde=1.0,
    w_bc=1.0,
    w_int=50.0,
    w_far=20.0   
):

    loss_pde = compute_pde_loss(
        net1, net2, net3,
        xyt_fgpm, xyt_hydro, xyt_sub,
        params_fgpm, params_hydro, params_sub
    )

    if electrically_open:
        loss_top = compute_top_surface_loss_open(net1, xt_top, params_fgpm)
    else:
        loss_top = compute_top_surface_loss_short(net1, xt_top, params_fgpm)

    # loss_int1 = compute_interface_loss_fgpm_hydro(net1, net2, xyt_int_fgpm_hydro)
    # loss_int2 = compute_interface_loss_hydro_sub(net2, net3, xyt_int_hydro_sub)

    loss_int1 = compute_interface_loss_fgpm_hydro(
    net1, net2, xyt_int_fgpm_hydro,
    w_w=200.0,
    w_phi=50.0,
    w_grad_w=200.0,
    w_grad_phi=50.0
)

    loss_int2 = compute_interface_loss_hydro_sub(
        net2, net3, xyt_int_hydro_sub,
        w_w=50.0,
        w_phi=10.0,
        w_grad_w=50.0,
        w_grad_phi=10.0
    )



    loss_far = compute_far_field_loss(net3, xyt_far_sub)
    loss_total = (
        w_pde * loss_pde
      + w_bc * loss_top
      + w_int * (loss_int1 + loss_int2)
      + w_far * loss_far 
    )

    return loss_total, {
        "pde": loss_pde.item(),
        "bc_top": loss_top.item(),
        "interface_1": loss_int1.item(),
        "interface_2": loss_int2.item(),
        "far": loss_far.item()
    }
