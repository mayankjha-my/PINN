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
    # Parameters FIRST (FIXED: moved before using them)
    # --------------------------------------------------
    params_layer = CONFIG["LAYER"]
    params_half  = CONFIG["HALFSPACE"]
    geom         = CONFIG["GEOMETRY"]
    H = geom["H"]

    # Convert to torch tensors on device
    params_layer = {key: torch.tensor(value, device=DEVICE, dtype=torch.float32) 
                   if isinstance(value, (int, float)) else value
                   for key, value in params_layer.items()}
    
    params_half = {key: torch.tensor(value, device=DEVICE, dtype=torch.float32) 
                  if isinstance(value, (int, float)) else value
                  for key, value in params_half.items()}

    # --------------------------------------------------
    # Initialize networks
    # --------------------------------------------------
    model_layer, model_half = get_all_networks()
    model_layer.to(DEVICE)
    model_half.to(DEVICE)

    # --------------------------------------------------
    # Trainable phase velocity (FIXED: params_layer now exists)
    # --------------------------------------------------
    c = torch.nn.Parameter(
        torch.sqrt(
            params_layer["mu66_0"] / params_layer["rho_0"]
        )
    )

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

        # Convert to tensors if needed (FIXED)
        if not isinstance(z_layer, torch.Tensor):
            z_layer = torch.tensor(z_layer, dtype=torch.float32, device=DEVICE)
        if not isinstance(z_half, torch.Tensor):
            z_half = torch.tensor(z_half, dtype=torch.float32, device=DEVICE)
        if not isinstance(z_top, torch.Tensor):
            z_top = torch.tensor(z_top, dtype=torch.float32, device=DEVICE)
        if not isinstance(z_int, torch.Tensor):
            z_int = torch.tensor(z_int, dtype=torch.float32, device=DEVICE)
        if not isinstance(z_far, torch.Tensor):
            z_far = torch.tensor(z_far, dtype=torch.float32, device=DEVICE)

        # Ensure correct shape [n_points, 1] (FIXED)
        if len(z_layer.shape) == 1:
            z_layer = z_layer.unsqueeze(1)
        if len(z_half.shape) == 1:
            z_half = z_half.unsqueeze(1)
        if len(z_top.shape) == 1:
            z_top = z_top.unsqueeze(1)
        if len(z_int.shape) == 1:
            z_int = z_int.unsqueeze(1)
        if len(z_far.shape) == 1:
            z_far = z_far.unsqueeze(1)

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
            k*H,  # Keep your original k*H if that's what your PDE expects
            c,
            w_pde=1.0,
            w_bc=10.0,   # FIXED: Changed from 0.1 to 10.0
            w_int=50.0,  # FIXED: Changed from 0.01 to 50.0
            w_far=5.0    # FIXED: Changed from 0.001 to 5.0
        )

        loss.backward()
        optimizer.step()

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

    k_vals = torch.linspace(
        max(CONFIG["GEOMETRY"]["k_min"], 0.2),
        CONFIG["GEOMETRY"]["k_max"],
        CONFIG["GEOMETRY"]["num_k"]
    )

    dispersion = []

    for idx, k in enumerate(k_vals):
        print(f"\n{'='*50}")
        print(f"Point {idx+1}/{len(k_vals)}: k = {k.item():.3f}")
        print(f"{'='*50}")
        
        c_val = train_for_single_k(k.item())
        dispersion.append([k.item(), c_val])

    return dispersion


if __name__ == "__main__":
    # Test with single k first
    print("Testing with single k value...")
    test_k = 0.5
    test_c = train_for_single_k(
        k=test_k,
        n_epochs=1000,  # Test with fewer epochs
        n_domain=100,
        n_bc=20,
        n_int=20,
        n_far=20,
        lr=1e-3
    )
    
    print(f"\nTest complete: c({test_k}) = {test_c:.3f} m/s")
    
    # Check if result is reasonable
    params_layer = CONFIG["LAYER"]
    params_half = CONFIG["HALFSPACE"]
    
    c_min = min(
        torch.sqrt(torch.tensor(params_layer["mu44_0"] / params_layer["rho_0"])),
        torch.sqrt(torch.tensor(params_half["mu44_0"] / params_half["rho_0"]))
    )
    c_max = max(
        torch.sqrt(torch.tensor(params_layer["mu44_0"] / params_layer["rho_0"])),
        torch.sqrt(torch.tensor(params_half["mu44_0"] / params_half["rho_0"]))
    )
    
    if c_min * 0.8 <= test_c <= c_max * 1.2:
        print(f"✓ Result is reasonable! Running full analysis...")
        dispersion_data = train_dispersion()
        torch.save(dispersion_data, "dispersion_curve.pt")
        print("\nDispersion data saved to dispersion_curve.pt")
    else:
        print(f"✗ Result may be incorrect. Expected: {c_min:.0f}-{c_max:.0f} m/s")