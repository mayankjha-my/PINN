import torch
import torch.nn as nn


# --------------------------------------------------
# Generic PINN network
# --------------------------------------------------
class PINN(nn.Module):
    """
    Fully-connected neural network
    """

    def __init__(self, in_dim, out_dim, width=128, depth=8):
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
# Network factory for dispersion problem
# --------------------------------------------------
def get_all_networks():
    """
    Returns PINN models for:
    - Functionally graded layer (complex field: V_R, V_I)
    - Functionally graded half-space (real field: V)
    """

    # Layer: input z → outputs [V_R, V_I]
    net_layer = PINN(
        in_dim=1,
        out_dim=2,
        width=128,
        depth=8
    )

    # Half-space: input z → output [V]
    net_halfspace = PINN(
        in_dim=1,
        out_dim=1,
        width=128,
        depth=8
    )

    return net_layer, net_halfspace
