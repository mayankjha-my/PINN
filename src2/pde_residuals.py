import torch

# --------------------------------------------------
# Gradient utility (included directly)
# --------------------------------------------------
def gradients(u, x):
    """
    Computes du/dx using torch.autograd
    u : (N,1)
    x : (N,1)
    """
    return torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]


# --------------------------------------------------
# Residual for FUNCTIONALLY GRADED LAYER
# --------------------------------------------------
def residual_layer_coupled(model, z, params, k, c):
    """
    Layer residual with FULL electromagnetic coupling
    (correct signs, directly from governing PDE)
    """

    z.requires_grad_(True)

    pred = model(z)
    V_R = pred[:, 0:1]   # real part
    V_I = pred[:, 1:2]   # imaginary part

    # derivatives
    V_R_z  = gradients(V_R, z)
    V_R_zz = gradients(V_R_z, z)

    V_I_z  = gradients(V_I, z)
    V_I_zz = gradients(V_I_z, z)

    # functional grading
    beta1 = params["beta1"]

    mu44 = params["mu44_0"] * (1.0 + torch.sin(beta1 * z))
    mu66 = params["mu66_0"] * (1.0 + torch.sin(beta1 * z))
    rho  = params["rho_0"]  * (1.0 + torch.sin(beta1 * z))
    P    = params["P_0"]    * (1.0 + torch.sin(beta1 * z))

    # EM parameters
    mu_e = params["mu_e"]
    H0   = params["H0"]
    phi = torch.tensor(params["phi"], device=z.device)

    # coefficients (✔ corrected)
    A = mu66 - P / 2.0 + mu_e * H0**2 * torch.cos(phi)**2
    B = mu_e * H0**2 * torch.sin(phi)**2
    C = k * mu_e * H0**2 * torch.sin(2.0 * phi)

    # ---------------- REAL residual ----------------
    r_real = (
        gradients(mu44 * V_R_z, z)
        - k**2 * A * V_R
        + B * V_R_zz
        - C * V_I_z
        + rho * k**2 * c**2 * V_R
    )

    # ---------------- IMAG residual ----------------
    r_imag = (
        gradients(mu44 * V_I_z, z)
        - k**2 * A * V_I
        + B * V_I_zz
        + C * V_R_z
        + rho * k**2 * c**2 * V_I
    )

    return r_real, r_imag



# --------------------------------------------------
# Residual for FUNCTIONALLY GRADED HALF-SPACE
# --------------------------------------------------
def residual_halfspace(model, z, params, k, c):
    """
    Lower half-space residual with FUNCTIONALLY GRADED rho(z)
    """

    z.requires_grad_(True)

    V = model(z)

    # derivatives of V
    V_z  = gradients(V, z)
    V_zz = gradients(V_z, z)

    # functional grading
    beta2 = params["beta2"]

    mu44 = params["mu44_0"] * (1.0 - torch.sin(beta2 * z))
    mu66 = params["mu66_0"] * (1.0 - torch.sin(beta2 * z))
    rho  = params["rho_0"]  * (1.0 - torch.sin(beta2 * z))
    P    = params["P_0"]    * (1.0 - torch.sin(beta2 * z))

    # derivative of rho(z)
    rho_z = gradients(rho, z)

    g = params["g"]

    # stiffness-like term
    A_h = mu66 - P / 2.0 - rho * g * z / 2.0

    # residual (✔ fully correct)
    r = (
        gradients(mu44 * V_z, z)
        - k**2 * A_h * V
        - (g / 2.0) * (rho + z * rho_z) * V_z
        - (g / 2.0) * (rho * z) * V_zz
        + rho * k**2 * c**2 * V
    )

    return r
