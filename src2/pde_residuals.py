import torch

def gradients(u, x):
    """Compute du/dx with proper gradient tracking."""
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

# --------------------------------------------------
# Residual for FUNCTIONALLY GRADED LAYER (FIXED)
# --------------------------------------------------
def residual_layer_coupled(model, z, params, k, c):
    """
    Layer residual - CORRECTED according to actual PDE.
    """
    z.requires_grad_(True)
    
    # Forward pass
    pred = model(z)
    scale = 1e-2  # Why this scaling?
    V_R = scale * pred[:, 0:1]
    V_I = scale * pred[:, 1:2]
    
    # Derivatives
    V_R_z = gradients(V_R, z)
    V_R_zz = gradients(V_R_z, z)
    
    V_I_z = gradients(V_I, z)
    V_I_zz = gradients(V_I_z, z)
    
    # Functionally graded properties
    beta1 = params["beta1"]
    z_sin = torch.sin(beta1 * z)
    
    mu44 = params["mu44_0"] * (1.0 + z_sin)
    mu66 = params["mu66_0"] * (1.0 + z_sin)
    rho = params["rho_0"] * (1.0 + z_sin)
    P = params["P_0"] * (1.0 + z_sin)
    
    # Gradient of mu44 (for product rule)
    mu44_z = gradients(mu44, z)
    
    # EM parameters
    mu_e = params["mu_e"]
    H0 = params["H0"]
    phi = params.get("phi_tensor", torch.tensor(params["phi"], device=z.device))
    
    # Phase velocity squared
    c2 = c**2
    
    # ----- REAL part residual -----
    # 1. d/dz(tau23) = d/dz(mu44 * V_R_z) = mu44 * V_R_zz + mu44_z * V_R_z
    term1 = mu44 * V_R_zz + mu44_z * V_R_z
    
    # 2. d/dx(tau12) = mu66 * (-k^2) V_R
    term2 = -k**2 * mu66 * V_R
    
    # 3. -P/2 * d²v/dx² = (P/2) * k^2 V_R
    term3 = (P/2.0) * k**2 * V_R
    
    # 4. EM force:
    #    cos²φ term: μe H0² cos²φ (-k² V_R) = -μe H0² cos²φ k² V_R
    #    sin(2φ) term: μe H0² sin(2φ) * (-k V_I_z) = -μe H0² sin(2φ) k V_I_z
    #    sin²φ term: μe H0² sin²φ * V_R_zz
    term4_em1 = -mu_e * H0**2 * torch.cos(phi)**2 * k**2 * V_R
    term4_em2 = -mu_e * H0**2 * torch.sin(2.0 * phi) * k * V_I_z
    term4_em3 = mu_e * H0**2 * torch.sin(phi)**2 * V_R_zz
    
    # 5. Right side: ρ ∂²v/∂t² = -ρ ω² V_R = -ρ k² c² V_R
    term5 = -rho * k**2 * c2 * V_R
    
    r_real = term1 + term2 + term3 + term4_em1 + term4_em2 + term4_em3 + term5
    
    # ----- IMAG part residual -----
    # Same structure but signs for mixed derivative term change
    term1_imag = mu44 * V_I_zz + mu44_z * V_I_z
    term2_imag = -k**2 * mu66 * V_I
    term3_imag = (P/2.0) * k**2 * V_I
    
    # EM force for imag part:
    # sin(2φ) term: μe H0² sin(2φ) * (k V_R_z)
    term4_em1_imag = -mu_e * H0**2 * torch.cos(phi)**2 * k**2 * V_I
    term4_em2_imag = mu_e * H0**2 * torch.sin(2.0 * phi) * k * V_R_z  # SIGN CHANGE!
    term4_em3_imag = mu_e * H0**2 * torch.sin(phi)**2 * V_I_zz
    
    term5_imag = -rho * k**2 * c2 * V_I
    
    r_imag = (term1_imag + term2_imag + term3_imag + 
              term4_em1_imag + term4_em2_imag + term4_em3_imag + term5_imag)
    
    return r_real, r_imag

# --------------------------------------------------
# Residual for HALF-SPACE (FIXED with correct gravity terms)
# --------------------------------------------------
def residual_halfspace(model, z, params, k, c):
    """
    Half-space residual - CORRECTED gravity terms.
    """
    z.requires_grad_(True)
    
    # Forward pass
    V = 1e-2 * model(z)  # Why scaling?
    V_z = gradients(V, z)
    V_zz = gradients(V_z, z)
    
    # Functionally graded properties
    beta2 = params["beta2"]
    z_sin = torch.sin(beta2 * z)
    
    mu44 = params["mu44_0"] * (1.0 - z_sin)
    mu66 = params["mu66_0"] * (1.0 - z_sin)
    rho = params["rho_0"] * (1.0 - z_sin)
    P = params["P_0"] * (1.0 - z_sin)
    
    # Gradients
    mu44_z = gradients(mu44, z)
    rho_z = gradients(rho, z)
    
    # Gravity
    g = params["g"]
    c2 = c**2
    
    # ----- CORRECT gravity terms from PDE -----
    # 1. d/dz(tau23) = d/dz(mu44 * V_z) = mu44 * V_zz + mu44_z * V_z
    term1 = mu44 * V_zz + mu44_z * V_z
    
    # 2. d/dx(tau12) = mu66 * (-k^2) V
    term2 = -k**2 * mu66 * V
    
    # 3. -P/2 * d²v/dx² = (P/2) * k^2 V
    term3 = (P/2.0) * k**2 * V
    
    # 4. -∂/∂x[(ρgz/2) ∂v/∂x] = (k² ρ g z / 2) V
    term4 = (k**2 * rho * g * z / 2.0) * V
    
    # 5. -∂/∂z[(ρgz/2) ∂v/∂z] 
    #    = -g/2 * [d/dz(ρz V_z)]
    #    = -g/2 * [(ρ + z ρ_z) V_z + ρ z V_zz]
    term5a = -(g / 2.0) * (rho + z * rho_z) * V_z
    term5b = -(g / 2.0) * (rho * z) * V_zz
    
    # 6. Right side: -ρ ω² V = -ρ k² c² V
    term6 = -rho * k**2 * c2 * V
    
    r = term1 + term2 + term3 + term4 + term5a + term5b + term6
    
    return r