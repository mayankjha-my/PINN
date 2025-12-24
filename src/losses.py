# src/losses.py
import torch


def shear_modulus(z):
    """
    Exponential heterogeneity (from your paper)
    μ(z) = μ0 * exp(α z)
    """
    mu0 = 1.0
    alpha = 0.5
    return mu0 * torch.exp(alpha * z)


def density(z):
    """
    Quadratic heterogeneity (from your paper)
    ρ(z) = ρ0 * (1 + β z^2)
    """
    rho0 = 1.0
    beta = 0.3
    return rho0 * (1.0 + beta * z**2)


def pde_loss(model, r, z, t):
    """
    Physics-informed loss for torsional wave equation
    """

    r.requires_grad_(True)
    z.requires_grad_(True)
    t.requires_grad_(True)

    X = torch.cat([r, z, t], dim=1)
    u = model(X)

    # First derivatives
    u_r = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

    # Second derivatives
    u_rr = torch.autograd.grad(u_r, r, torch.ones_like(u_r), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, torch.ones_like(u_z), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]

    mu = shear_modulus(z)
    rho = density(z)

    # PDE residual
    residual = mu * (u_rr + (1.0 / r) * u_r + u_zz) - rho * u_tt


    return torch.mean(residual**2)



# src/losses.py (ADD below existing code)

def surface_stress_free_loss(model, r, t):
    """
    τ_{zθ} = μ(z) * du/dz = 0 at z = 0
    """
    z = torch.zeros_like(r, requires_grad=True)

    X = torch.cat([r, z, t], dim=1)
    u = model(X)

    u_z = torch.autograd.grad(
        u, z, torch.ones_like(u), create_graph=True
    )[0]

    mu = shear_modulus(z)
    stress = mu * u_z

    return torch.mean(stress**2)


def interface_continuity_loss(model, r, t, h=0.5):
    """
    Enforces:
    u1 = u2
    μ1 du1/dz = μ2 du2/dz
    """
    z1 = torch.full_like(r, h - 1e-3, requires_grad=True)
    z2 = torch.full_like(r, h + 1e-3, requires_grad=True)

    X1 = torch.cat([r, z1, t], dim=1)
    X2 = torch.cat([r, z2, t], dim=1)

    u1 = model(X1)
    u2 = model(X2)

    # displacement continuity
    disp_loss = torch.mean((u1 - u2)**2)

    # stress continuity
    u1_z = torch.autograd.grad(u1, z1, torch.ones_like(u1), create_graph=True)[0]
    u2_z = torch.autograd.grad(u2, z2, torch.ones_like(u2), create_graph=True)[0]

    mu1 = shear_modulus(z1)
    mu2 = shear_modulus(z2)

    stress_loss = torch.mean((mu1 * u1_z - mu2 * u2_z)**2)

    return disp_loss + stress_loss


def decay_loss(model, r, t, z_max=2.0):
    """
    u → 0 as z → ∞ (approximated at z = z_max)
    """
    z = torch.full_like(r, z_max)

    X = torch.cat([r, z, t], dim=1)
    u = model(X)

    return torch.mean(u**2)
