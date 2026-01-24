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
    model_layer,
    model_half,
    c,
    n_epochs=5000,
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
    # Parameters FIRST (FIXED: moved before using them)
    # --------------------------------------------------
    params_layer = CONFIG["LAYER"]
    params_half  = CONFIG["HALFSPACE"]
    geom         = CONFIG["GEOMETRY"]
    H = geom["H"]

    # Convert to torch tensors on device
    params_layer = {
        k: torch.tensor(v, device=DEVICE, dtype=torch.float32)
        if isinstance(v, (int, float)) else v
        for k, v in params_layer.items()
    }

    params_half = {
        k: torch.tensor(v, device=DEVICE, dtype=torch.float32)
        if isinstance(v, (int, float)) else v
        for k, v in params_half.items()
    }

   

    

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = optim.Adam(
        [
            {"params": model_layer.parameters(), "lr": lr},
            {"params": model_half.parameters(), "lr": lr},
            {"params": [c], "lr": 1e-4},   # 🔧 CHANGE: smaller LR for eigenvalue
        ]
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

        def to_tensor(z):
            if not isinstance(z, torch.Tensor):
                z = torch.tensor(z, dtype=torch.float32, device=DEVICE)
            if z.ndim == 1:
                z = z.unsqueeze(1)
            return z

        z_layer = to_tensor(z_layer)
        z_half  = to_tensor(z_half)
        z_top   = to_tensor(z_top)
        z_int   = to_tensor(z_int)
        z_far   = to_tensor(z_far)

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
            k,  # Keep your original k*H if that's what your PDE expects
            c,
            w_pde=1.0,
            w_bc=0.05,   # FIXED: Changed from 0.1 to 10.0
            w_int=0.0001,  # FIXED: Changed from 0.01 to 50.0
            w_far=0.00001    # FIXED: Changed from 0.001 to 5.0
        )

        loss.backward()
        optimizer.step()
        # --------------------------------------------------
        # 🔧 CHANGE 3: CLAMP c (PREVENT NON-PHYSICAL VALUES)
        # --------------------------------------------------
       

        # --------- Logging ---------
        if epoch % 500 == 0:
            print(
                f"Epoch {epoch:6d} | "
                f"Loss = {loss.item():.3e} | "
                f"c = {c.item():.5f} | "
                f"PDE = {logs.get('pde', 0):.2e} | "
                f"BC = {logs.get('bc_top', 0):.2e} | "
                f"INT = {logs.get('interface', 0):.2e} | "
                f"FAR = {logs.get('far', 0):.2e}"
            )

    return c.detach().cpu().item()


# --------------------------------------------------
# Dispersion curve extraction
# --------------------------------------------------
def train_dispersion():
    """
    Sweep over k and compute dispersion curve c(k)
    """

    geom = CONFIG["GEOMETRY"]

    k_vals = torch.linspace(
        geom["k_min"],
        geom["k_max"],
        geom["num_k"]
    )

    # --------------------------------------------------
    # Initialize networks ONCE
    # --------------------------------------------------
    model_layer, model_half = get_all_networks()
    model_layer.to(DEVICE)
    model_half.to(DEVICE)

    # --------------------------------------------------
    # Initialize c ONCE
    # --------------------------------------------------
    c = torch.nn.Parameter(
        torch.tensor(5.0, device=DEVICE, dtype=torch.float32)
    )


    dispersion = []

    for idx, k in enumerate(k_vals):
        print(f"\n{'='*50}")
        print(f"Point {idx+1}/{len(k_vals)}: k = {k.item():.3f}")
        print(f"{'='*50}")

        c_val = train_for_single_k(
            k.item(),
            model_layer,
            model_half,
            c,
            n_epochs=5000 if idx == 0 else 2500
        )

        dispersion.append([k.item(), c_val])

    return dispersion


if __name__ == "__main__":

    print("Testing with single k value...")

    # --------------------------------------------------
    # 🔧 CHANGE 1: Initialize networks
    # --------------------------------------------------
    model_layer, model_half = get_all_networks()
    model_layer.to(DEVICE)
    model_half.to(DEVICE)

    # --------------------------------------------------
    # 🔧 CHANGE 2: Initialize eigenvalue c
    # --------------------------------------------------
    c = torch.nn.Parameter(
        torch.tensor(5.0, device=DEVICE, dtype=torch.float32)
     )

    # --------------------------------------------------
    # Single-k test
    # --------------------------------------------------
    test_k = 0.6

    test_c = train_for_single_k(
        k=test_k,
        model_layer=model_layer,
        model_half=model_half,
        c=c,
        n_epochs=1500,
        n_domain=200,
        n_bc=50,
        n_int=50,
        n_far=50,
        lr=1e-3
    )

    print(f"\nTest complete: c({test_k}) = {test_c:.5f} (non-dimensional)")

    # --------------------------------------------------
    # 🔧 CHANGE 3: Physical sanity check
    # --------------------------------------------------
   
    if 2.0 < test_c < 8.0:
        print("✓ Guided SH mode captured. Running full dispersion sweep...\n")

        dispersion = train_dispersion()
        torch.save(dispersion, "dispersion_curve.pt")

        print("Dispersion data saved to dispersion_curve.pt")

    else:
        print("✗ Bulk mode detected. Check loss constraints.")