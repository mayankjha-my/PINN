import torch
import torch.nn as nn

class Arctan(nn.Module):
    def forward(self, x):
        return torch.atan(x)

# --------------------------------------------------
# Activation selector (NEW)
# --------------------------------------------------
def get_activation(name):
    name = name.lower()

    if name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "swish":
        return nn.SiLU()          # Swish
    elif name == "softplus":
        return nn.Softplus()
    elif name == "arctan":
        return Arctan()

    else:
        raise ValueError(f"Unknown activation: {name}")


# --------------------------------------------------
# Generic PINN network (MODIFIED)
# --------------------------------------------------
class PINN(nn.Module):
    """
    Fully-connected neural network
    """

    def __init__(self, in_dim, out_dim, width=128, depth=8, activation="tanh"):
        super().__init__()

        act = get_activation(activation)

        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(act)

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(act)

        # Output layer (NO activation)
        layers.append(nn.Linear(width, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# --------------------------------------------------
# Network factory for dispersion problem (MODIFIED)
# --------------------------------------------------
def get_all_networks(activation="tanh"):
    """
    Returns PINN models for:
    - Functionally graded layer (complex field: V_R, V_I)
    - Functionally graded half-space (real field: V)
    """

    net_layer = PINN(
        in_dim=1,
        out_dim=2,
        width=64,
        depth=5,
        activation=activation
    )

    net_halfspace = PINN(
        in_dim=1,
        out_dim=1,
        width=64,
        depth=5,
        activation=activation
    )

    return net_layer, net_halfspace
