import torch
from .model import PINN
from .params import params
from .pde_fgpm import fgpm_residual
from .pde_hydrogel import hydrogel_residual
from .pde_substrate import substrate_residual
from .sampler import sample_fgpm, sample_hydrogel, sample_substrate
from .bc import top_surface_bc, interface_loss


def train(epochs=5000, N=500, lr=1e-3):

    model = PINN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    h1 = params["geom"]["h1"]
    h2 = params["geom"]["h2"]

    for e in range(epochs):

        x1, y1, t1 = sample_fgpm(N, h1)
        x2, y2, t2 = sample_hydrogel(N, h2)
        x3, y3, t3 = sample_substrate(N, h2)

        loss1m, loss1e = fgpm_residual(model, x1, y1, t1, params["fgpm"])
        loss2m, loss2e = hydrogel_residual(model, x2, y2, t2, params["hydrogel"])
        loss3m, loss3e = substrate_residual(model, x3, y3, t3, params["substrate"])

        # BC and interfaces
        xb = -h1 * torch.ones(N, 1)
        yb = torch.rand(N, 1)
        tb = torch.rand(N, 1)

        lossBC = top_surface_bc(model, params["fgpm"], xb, yb, tb)

        xi = torch.zeros(N, 1)
        yi = torch.rand(N, 1)
        ti = torch.rand(N, 1)

        lossIF = interface_loss(model, xi, yi, ti)

        loss = (
            loss1m + loss1e +
            loss2m + loss2e +
            loss3m + loss3e +
            5 * lossBC +
            5 * lossIF
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if e % 200 == 0:
            print(f"Epoch {e} loss={loss.item():.4e}")

    return model
