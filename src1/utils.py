import torch


def grad(u, x):
    return torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]


def grad2(u, x):
    return grad(grad(u, x), x)
