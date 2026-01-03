# # src/pinn.py
# import torch
# from torch.optim import Adam

# from model import PINN
# from losses import pde_loss


# class PINNTrainer:
#     def __init__(self):
#         self.device = torch.device("cpu")

#         self.model = PINN(
#             layers=[3, 64, 64, 64, 1]  # (r, z, t) → u
#         ).to(self.device)

#         self.optimizer = Adam(self.model.parameters(), lr=1e-3)

#     def sample_collocation(self, N):
#         r = torch.rand(N, 1, device=self.device)
#         z = torch.rand(N, 1, device=self.device)
#         t = torch.rand(N, 1, device=self.device)
#         return r, z, t

#     def train(self, epochs=5000, N=2000):
#         for epoch in range(epochs):
#             self.optimizer.zero_grad()

#             r, z, t = self.sample_collocation(N)
#             loss = pde_loss(self.model, r, z, t)

#             loss.backward()
#             self.optimizer.step()

#             if epoch % 500 == 0:
#                 print(f"Epoch {epoch:5d} | PDE Loss: {loss.item():.6e}")

#         return self.model



# # src/pinn.py (MODIFY train loop)

# from losses import (
#     pde_loss,
#     surface_stress_free_loss,
#     interface_continuity_loss,
#     decay_loss
# )

# def train(self, epochs=5000, N=2000):
#     for epoch in range(epochs):
#         self.optimizer.zero_grad()

#         r, z, t = self.sample_collocation(N)

#         loss_pde = pde_loss(self.model, r, z, t)
#         loss_surface = surface_stress_free_loss(self.model, r, t)
#         loss_interface = interface_continuity_loss(self.model, r, t)
#         loss_decay = decay_loss(self.model, r, t)

#         loss = (
#             loss_pde
#             + 10.0 * loss_surface
#             + 10.0 * loss_interface
#             + 1.0 * loss_decay
#         )

#         loss.backward()
#         self.optimizer.step()

#         if epoch % 10 == 0:
#             print(
#                 f"Epoch {epoch:5d} | "
#                 f"PDE: {loss_pde.item():.2e} | "
#                 f"Surface: {loss_surface.item():.2e} | "
#                 f"Interface: {loss_interface.item():.2e}"
#             )


import torch
import numpy as np
from .model import MLP
from .losses import pde_residual, boundary_loss



def sample_domain(N=200):
    r = torch.rand(N, 1) * 2 - 1
    z = torch.rand(N, 1) * -2
    return r, z


def train_pinn(epochs=8000, omega=2.0, lr=1e-3):

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):

        r, z = sample_domain()

        loss_pde = pde_residual(model, r, z, omega)
        loss_bc = boundary_loss(model)

        loss = loss_pde + loss_bc

        opt.zero_grad()
        loss.backward()
        opt.step()

        if e % 500 == 0:
            print(f"Epoch {e}  PDE={loss_pde.item():.4e}  BC={loss_bc.item():.4e}")

    return model



import torch
import numpy as np
from scipy.optimize import curve_fit


def estimate_k_from_model(model, z_level=0.0):

    r = torch.linspace(-1, 1, 200).reshape(-1, 1)
    z = torch.full_like(r, z_level)

    with torch.no_grad():
        u = model(torch.cat([r, z], dim=1)).numpy().flatten()

    r = r.numpy().flatten()

    def sinusoid(x, A, k, phi):
        return A * np.sin(k * x + phi)

    params, _ = curve_fit(sinusoid, r, u, p0=[1.0, 2.0, 0.0])

    A, k, phi = params
    return abs(k)
