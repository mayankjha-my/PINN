# # src/model.py
# import torch
# import torch.nn as nn


# class PINN(nn.Module):
#     def __init__(self, layers):
#         super().__init__()

#         self.activation = nn.Tanh()
#         self.layers = nn.ModuleList()

#         for i in range(len(layers) - 1):
#             self.layers.append(nn.Linear(layers[i], layers[i + 1]))

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.layers:
#             nn.init.xavier_normal_(m.weight)
#             nn.init.zeros_(m.bias)

#     def forward(self, x):
#         """
#         x = [r, z, t]
#         """
#         for layer in self.layers[:-1]:
#             x = self.activation(layer(x))
#         return self.layers[-1](x)


import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, width=64, depth=6):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, out_dim))

        self.model = nn.Sequential(*layers)

        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)
