import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Utility: uniform sampling
# --------------------------------------------------
def sample_uniform(n, low, high):
    return low + (high - low) * torch.rand(
        n, 1, device=DEVICE, dtype=torch.float32
    )


# --------------------------------------------------
# DOMAIN SAMPLING (2D)
# --------------------------------------------------
def sample_domain(n, geom):
    """
    Sample (x1, x3, t) in domain
    """

    x1_min = geom["x1_min"]
    x1_max = geom["x1_max"]

    x3_min = geom["x3_min"]
    x3_max = geom["x3_max"]

    x1 = sample_uniform(n, x1_min, x1_max)
    x3 = sample_uniform(n, x3_min, x3_max)
    t  = sample_uniform(n, 0.0, 1.0)

    return x1, x3, t


# --------------------------------------------------
# CRACK SAMPLING (VERY IMPORTANT)
# x1 = 0, a < x3 < b
# --------------------------------------------------
def sample_crack(n, geom):
    """
    Crack lies along x1 = 0
    """

    a = geom["crack_start"]
    b = geom["crack_end"]

    x1 = torch.zeros((n, 1), device=DEVICE)
    x3 = sample_uniform(n, a, b)

    return x1, x3


# --------------------------------------------------
# UNCRACKED REGION (x1 = 0 but outside crack)
# --------------------------------------------------
def sample_uncracked(n, geom):
    """
    Regions where displacement is continuous
    """

    a = geom["crack_start"]
    b = geom["crack_end"]
    H = geom["x3_max"]

    # sample two regions
    x3_lower = sample_uniform(n // 2, 0.0, a)
    x3_upper = sample_uniform(n // 2, b, H)

    x3 = torch.cat([x3_lower, x3_upper], dim=0)
    x1 = torch.zeros_like(x3)

    return x1, x3


# --------------------------------------------------
# SURFACE SAMPLING (TOP & BOTTOM)
# --------------------------------------------------
def sample_surface(n, geom):
    """
    x3 = 0 and x3 = H
    """

    x1 = sample_uniform(n, geom["x1_min"], geom["x1_max"])

    x3_bottom = torch.zeros((n // 2, 1), device=DEVICE)
    x3_top = torch.full((n // 2, 1), geom["x3_max"], device=DEVICE)

    x3 = torch.cat([x3_bottom, x3_top], dim=0)

    return x1, x3


# --------------------------------------------------
# LEFT & RIGHT BOUNDARIES
# --------------------------------------------------
def sample_boundaries(n, geom):
    """
    x1 = left and x1 = right
    """

    x3 = sample_uniform(n, geom["x3_min"], geom["x3_max"])

    x1_left = torch.full((n // 2, 1), geom["x1_min"], device=DEVICE)
    x1_right = torch.full((n // 2, 1), geom["x1_max"], device=DEVICE)

    x1 = torch.cat([x1_left, x1_right], dim=0)

    return x1, x3


# --------------------------------------------------
# TEMPERATURE IC SAMPLING (t = 0)
# --------------------------------------------------
def sample_initial_time(n, geom):
    """
    t = 0
    """

    x3 = sample_uniform(n, geom["x3_min"], geom["x3_max"])
    t = torch.zeros_like(x3)

    return x3, t


# --------------------------------------------------
# TEMPERATURE BC (x3 = 0 and x3 = H)
# --------------------------------------------------
def sample_temp_boundary(n, geom):
    """
    thermal boundaries
    """

    t = sample_uniform(n, 0.0, 1.0)

    x3_left = torch.zeros((n // 2, 1), device=DEVICE)
    x3_right = torch.full((n // 2, 1), geom["x3_max"], device=DEVICE)

    x3 = torch.cat([x3_left, x3_right], dim=0)

    return x3, t