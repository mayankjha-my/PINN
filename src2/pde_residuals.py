import torch

def gradients(u, x):
    """Compute du/dx with proper gradient tracking."""
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True        
    )[0]

# --------------------------------------------------
# Residual for FUNCTIONALLY GRADED LAYER (FIXED)
# --------------------------------------------------
def residual_layer_coupled(model, z, params, k,c):
    """
    PINN residual for:
    A1 * V''(z) + A2 * V'(z) + A3 * V(z) = 0
    k=kH
    c=c/
    """

    z.requires_grad_(True)

    # ---------------------------------
    # Network output
    # ---------------------------------
    V = model(z)          # complex-valued split assumed
    V_R = V[:, 0:1]       # real part
    V_I = V[:, 1:2]       # imaginary part

    # ---------------------------------
    # Derivatives
    # ---------------------------------
    V_R_z  = gradients(V_R, z)
    V_R_zz = gradients(V_R_z, z)
    V_I_z  = gradients(V_I, z)
    V_I_zz = gradients(V_I_z, z)

    # ---------------------------------
    # Parameters
    # ---------------------------------
    mu44_0 = params["mu44_0"]
    mu66_0 = params["mu66_0"]
    rho_0  = params["rho_0"]
    P_0    = params["P_0"]

    mu_e = params["mu_e"]
    H0   = params["H0"]
    phi  = params.get("phi_tensor", torch.tensor(params["phi"], device=z.device)
    )

    beta1 = params["beta1"]

    # ---------------------------------
    # Coefficients A1, A2, A3
    # ---------------------------------
    A1 = mu44_0 + mu_e * H0**2 * (torch.sin(phi))**2

    A2_real = beta1 * mu44_0
    A2_imag = -k * mu_e * H0**2 * (torch.sin(2.0 * phi))

    A3 = (
        rho_0 *(k*c)**2
        - k**2 * (
            mu66_0
            + mu_e * H0**2 * (torch.cos(phi))**2
            - P_0 / 2.0
        )
    )

    # ---------------------------------
    # REAL residual
    # ---------------------------------
    r_real = (
        A1 * V_R_zz
        + A2_real * V_R_z
        - A2_imag * V_I_z
        + A3 * V_R
    )/mu44_0

    # ---------------------------------
    # IMAG residual
    # ---------------------------------
    r_imag = (
        A1 * V_I_zz
        + A2_real * V_I_z
        + A2_imag * V_R_z
        + A3 * V_I
    )/mu44_0

    return r_real, r_imag

# --------------------------------------------------
# Residual for HALF-SPACE (FIXED with correct gravity terms)
# --------------------------------------------------
def residual_halfspace(model, z, params, k, c):
    """
    PINN residual for half-space equation with gravity and heterogeneity
    """

    z.requires_grad_(True)

   # Forward pass (no scaling, already non-dimensional)
    V = model(z)
    V_z = gradients(V, z)
    V_zz = gradients(V_z, z)

    # ---------------------------------
    # Parameters
    # ---------------------------------
    mu44_0 = params["mu44_0"]
    mu66_0 = params["mu66_0"]
    rho_0  = params["rho_0"]
    P_0    = params["P_0"]

    beta2 = params["beta2"]
    g     = params["g"]

    # ---------------------------------
    # z-dependent coefficients
    # ---------------------------------
    A1 = mu44_0 - (rho_0 * g * z) / 2.0

    A2 = (
        beta2 * mu44_0
        - (rho_0 * g) / 2.0
        - (beta2 * rho_0 * g * z) / 2.0
    )

    A3 = (
        rho_0 * (k*c)**2 - k**2 * (
            mu66_0
            - P_0 / 2.0
            - (rho_0 * g * z) / 2.0
        )
    )

    # ---------------------------------
    # residual
    # ---------------------------------
    r = (
        A1 * V_zz
        + A2 * V_z
        + A3 * V
    )/mu44_0

    
    return r

