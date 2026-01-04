import torch


def sample_fgpm(N, h1):
    x = -h1 + torch.rand(N, 1) * h1
    y = torch.rand(N, 1)
    t = torch.rand(N, 1)
    return x, y, t


def sample_hydrogel(N, h2):
    x = torch.rand(N, 1) * h2
    y = torch.rand(N, 1)
    t = torch.rand(N, 1)
    return x, y, t


def sample_substrate(N, h2):
    x = h2 + torch.rand(N, 1) * 2
    y = torch.rand(N, 1)
    t = torch.rand(N, 1)
    return x, y, t
