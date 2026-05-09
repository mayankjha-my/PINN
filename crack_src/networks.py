import torch
import torch.nn as nn


# --------------------------------------------------
# Generic PINN network (2D → 4 outputs)
# --------------------------------------------------
class PINN(nn.Module):
    """
    Fully-connected neural network for:
    inputs  : (x1, x3)
    outputs : (u1, u3, phi, T)
    """

    def __init__(self, in_dim=2, out_dim=4, width=128, depth=8):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------
# Single network (NEW)
# --------------------------------------------------
def get_all_networks():
    """
    Returns single PINN model for:
    (u1, u3, phi, T)
    """

    model = PINN(
        in_dim=2,    # (x1, x3)
        out_dim=4,   # (u1, u3, phi, T)
        width=128,
        depth=8
    )

    return [model]