import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Neural network model returning [w, phi]
    """

    def __init__(self, in_dim=3, out_dim=2, width=128, depth=8):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, xyt):
        return self.model(xyt)


def get_all_networks():
    """
    Returns PINN models for all three media
    """

    net1 = PINN()
    net2 = PINN()
    net3 = PINN()

    return net1, net2, net3
