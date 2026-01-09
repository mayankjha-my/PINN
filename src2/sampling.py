import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sample_uniform(n, low, high):
    return low + (high - low) * torch.rand(n, 1)


def sample_domain_points(n_domain, geom):
    """
    Returns:
        xyt_fgpm
        xyt_hydro
        xyt_sub
    """

    h1 = geom["h1"]
    h2 = geom["h2"]
    h3 = geom.get("h3", h2 + h1)   # default substrate thickness

    y_min = geom.get("y_min", -6.0)
    y_max = geom.get("y_max", 6.0)

    t_min = geom.get("t_min", 0.0)
    t_max = geom.get("t_max", 6.0)

    n1 = n2 = n3 = n_domain

    # ---------------- FGPM ----------------
    x1 = sample_uniform(n1, -h1, 0.0)
    y1 = sample_uniform(n1, y_min, y_max)
    t1 = sample_uniform(n1, t_min, t_max)
    xyt_fgpm = torch.cat([x1, y1, t1], dim=1)

    # ---------------- HYDROGEL ----------------
    x2 = sample_uniform(n2, 0.0, h2)
    y2 = sample_uniform(n2, y_min, y_max)
    t2 = sample_uniform(n2, t_min, t_max)
    xyt_hydro = torch.cat([x2, y2, t2], dim=1)

    # ---------------- SUBSTRATE ----------------
    x3 = sample_uniform(n3, h2, h3)
    y3 = sample_uniform(n3, y_min, y_max)
    t3 = sample_uniform(n3, t_min, t_max)
    xyt_sub = torch.cat([x3, y3, t3], dim=1)

    return (
        xyt_fgpm.to(DEVICE),
        xyt_hydro.to(DEVICE),
        xyt_sub.to(DEVICE),
    )



def sample_boundary_points(n_boundary, geom):
    """
    Top free surface at x = -h1
    """

    h1 = geom["h1"]

    y_min = geom.get("y_min", -6.0)
    y_max = geom.get("y_max", 6.0)

    t_min = geom.get("t_min", 0.0)
    t_max = geom.get("t_max", 6.0)

    x = torch.full((n_boundary, 1), -h1)
    y = sample_uniform(n_boundary, y_min, y_max)
    t = sample_uniform(n_boundary, t_min, t_max)

    xt_top = torch.cat([x, y, t], dim=1)

    return xt_top.to(DEVICE)



def sample_interface_points(n_interface, geom):
    """
    Returns:
        FGPM-HYDRO interface points at x = 0
        HYDRO-SUB interface points at x = h2
    """

    h2 = geom["h2"]

    y_min = geom.get("y_min", -6.0)
    y_max = geom.get("y_max", 6.0)

    t_min = geom.get("t_min", 0.0)
    t_max = geom.get("t_max", 6.0)

    # ---------- FGPM / HYDRO ----------
    x1 = torch.zeros((n_interface, 1))
    y1 = sample_uniform(n_interface, y_min, y_max)
    t1 = sample_uniform(n_interface, t_min, t_max)
    xyt_int1 = torch.cat([x1, y1, t1], dim=1)

    # ---------- HYDRO / SUB ----------
    x2 = torch.full((n_interface, 1), h2)
    y2 = sample_uniform(n_interface, y_min, y_max)
    t2 = sample_uniform(n_interface, t_min, t_max)
    xyt_int2 = torch.cat([x2, y2, t2], dim=1)

    return (
        xyt_int1.to(DEVICE),
        xyt_int2.to(DEVICE),
    )
def sample_substrate_far_boundary(n_far, geom):
    """
    Far-field boundary for substrate (half-space)
    x = h3
    """

    h3 = geom.get("h3", geom["h1"] + geom["h2"])

    y_min = geom.get("y_min", -6.0)
    y_max = geom.get("y_max", 6.0)

    t_min = geom.get("t_min", 0.0)
    t_max = geom.get("t_max", 6.0)

    x = torch.full((n_far, 1), h3)
    y = sample_uniform(n_far, y_min, y_max)
    t = sample_uniform(n_far, t_min, t_max)

    return torch.cat([x, y, t], dim=1).to(DEVICE)
