import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, width=128, depth=8):
        super().__init__()

        layers = []
        layers.append(nn.Linear(3, width))
        layers.append(nn.Tanh())

        for _ in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 6))

        self.net = nn.Sequential(*layers)

    def forward(self, xyt):
        return self.net(xyt)
    

    # input (x,y,t)
    # output - w1, phi1, w2, phi2, w3, phi3

