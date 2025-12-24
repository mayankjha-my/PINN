# src/pinn.py
import torch
from torch.optim import Adam

from model import PINN
from losses import pde_loss


class PINNTrainer:
    def __init__(self):
        self.device = torch.device("cpu")

        self.model = PINN(
            layers=[3, 64, 64, 64, 1]  # (r, z, t) → u
        ).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

    def sample_collocation(self, N):
        r = torch.rand(N, 1, device=self.device)
        z = torch.rand(N, 1, device=self.device)
        t = torch.rand(N, 1, device=self.device)
        return r, z, t

    def train(self, epochs=5000, N=2000):
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            r, z, t = self.sample_collocation(N)
            loss = pde_loss(self.model, r, z, t)

            loss.backward()
            self.optimizer.step()

            if epoch % 500 == 0:
                print(f"Epoch {epoch:5d} | PDE Loss: {loss.item():.6e}")

        return self.model



# src/pinn.py (MODIFY train loop)

from losses import (
    pde_loss,
    surface_stress_free_loss,
    interface_continuity_loss,
    decay_loss
)

def train(self, epochs=5000, N=2000):
    for epoch in range(epochs):
        self.optimizer.zero_grad()

        r, z, t = self.sample_collocation(N)

        loss_pde = pde_loss(self.model, r, z, t)
        loss_surface = surface_stress_free_loss(self.model, r, t)
        loss_interface = interface_continuity_loss(self.model, r, t)
        loss_decay = decay_loss(self.model, r, t)

        loss = (
            loss_pde
            + 10.0 * loss_surface
            + 10.0 * loss_interface
            + 1.0 * loss_decay
        )

        loss.backward()
        self.optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"PDE: {loss_pde.item():.2e} | "
                f"Surface: {loss_surface.item():.2e} | "
                f"Interface: {loss_interface.item():.2e}"
            )
