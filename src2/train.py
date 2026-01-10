import torch
import torch.optim as optim

from networks import get_all_networks
from config import CONFIG
from sampling import (
    sample_domain_points,
    sample_top_surface,
    sample_interface,
    sample_far_field
)
from losses import total_loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_for_single_k(
    k,
    n_epochs=20000,
    n_domain=5000,
    n_bc=1000,
    n_int=1000,
    n_far=1000,
    lr=1e-3
):
    """
    Train PINN for a fixed wave number k
    Returns learned phase velocity c
    """

    print(f"\nTraining for k = {k:.3f} on {DEVICE}")

    # --------------------------------------------------
    # Initialize networks
    # --------------------------------------------------
    model_layer, model_half = get_all_networks()
    model_layer.to(DEVICE)
    model_half.to(DEVICE)

    # --------------------------------------------------
    # Trainable phase velocity
    # --------------------------------------------------
    c = torch.nn.Parameter(torch.tensor(0.8, device=DEVICE))

    # --------------------------------------------------
    # Parameters
    # --------------------------------------------------
    params_layer = CONFIG["LAYER"]
    params_half  = CONFIG["HALFSPACE"]
    geom         = CONFIG["GEOMETRY"]

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = optim.Adam(
        list(model_layer.parameters()) +
        list(model_half.parameters()) +
        [c],
        lr=lr
    )

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(1, n_epochs + 1):

        # --------- Sample points ---------
        z_layer, z_half = sample_domain_points(n_domain, geom)
        z_top  = sample_top_surface(n_bc, geom)
        z_int  = sample_interface(n_int)
        z_far  = sample_far_field(n_far, geom)

        optimizer.zero_grad()

        # --------- Loss ---------
        loss, logs = total_loss(
            model_layer,
            model_half,
            z_layer,
            z_half,
            z_top,
            z_int,
            z_far,
            params_layer,
            params_half,
            k,
            c,
            w_pde=1.0,
            w_bc=5.0,
            w_int=10.0,
            w_far=20.0
        )

        loss.backward()
        optimizer.step()

        # --------- Logging ---------
        if epoch % 500 == 0:
            print(
                f"Epoch {epoch:6d} | "
                f"Loss = {loss.item():.3e} | "
                f"c = {c.item():.5f} | "
                f"PDE = {logs['pde']:.2e} | "
                f"BC = {logs['bc_top']:.2e} | "
                f"INT = {logs['interface']:.2e} | "
                f"FAR = {logs['far']:.2e}"
            )

    return c.detach().cpu().item()


# --------------------------------------------------
# Dispersion curve extraction
# --------------------------------------------------
def train_dispersion():
    """
    Sweep over k and compute dispersion curve c(k)
    """

    k_vals = torch.linspace(
        CONFIG["GEOMETRY"]["k_min"],
        CONFIG["GEOMETRY"]["k_max"],
        CONFIG["GEOMETRY"]["num_k"]
    )

    dispersion = []

    for k in k_vals:
        c_val = train_for_single_k(k.item())
        dispersion.append([k.item(), c_val])

    return dispersion


if __name__ == "__main__":
    dispersion_data = train_dispersion()

    # Save results
    torch.save(dispersion_data, "dispersion_curve.pt")
    print("\nDispersion data saved to dispersion_curve.pt")
