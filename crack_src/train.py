import torch
import torch.optim as optim

from networks import get_all_networks
from config import CONFIG

from pde_residuals import residual_piezo, residual_temperature
from boundary_conditions import (
    crack_normal_stress_bc,
    displacement_bc,
    shear_bc,
    electric_bc_D1,
    surface_bc,
    temperature_ic,
    temperature_bc_left,
    temperature_bc_right
)

# --------------------------------------------------
# Device
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
# TRAIN FUNCTION
# ==================================================
def train(model, n_epochs=5000, lr=1e-3):

    print(f"\nTraining thermo–piezo PINN on {DEVICE}")

    # --------------------------------------------------
    # Parameters
    # --------------------------------------------------
    params = CONFIG["MATERIAL"].copy()
    params.update(CONFIG["THERMAL"])

    # Convert to tensors
    params = {
        k: torch.tensor(v, device=DEVICE, dtype=torch.float32)
        if isinstance(v, (int, float)) else v
        for k, v in params.items()
    }

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(1, n_epochs + 1):

        # --------------------------------------------------
        # SAMPLE DOMAIN POINTS
        # --------------------------------------------------
        x1 = torch.rand(3000, 1, device=DEVICE) * 2 - 1   # [-1,1]
        x3 = torch.rand(3000, 1, device=DEVICE) * 2 - 1
        t  = torch.rand(3000, 1, device=DEVICE)           # [0,1]

        optimizer.zero_grad()

        # --------------------------------------------------
        # PDE LOSS
        # --------------------------------------------------
        r1, r2, r3 = residual_piezo(model, x1, x3, params)
        rT = residual_temperature(model, x3, t, params)

        loss_pde = (
            (r1**2).mean() +
            (r2**2).mean() +
            (r3**2).mean() +
            (rT**2).mean()
        )

        # --------------------------------------------------
        # BOUNDARY LOSS
        # --------------------------------------------------
        loss_bc = (
            (displacement_bc(model, x1, x3)**2).mean() +
            (shear_bc(model, x1, x3, params)**2).mean() +
            (electric_bc_D1(model, x1, x3, params)**2).mean()
        )

        # --------------------------------------------------
        # CRACK LOSS
        # --------------------------------------------------
        tau0 = torch.tensor(1.0, device=DEVICE)

        def tauT(x):
            return 0.0 * x   # (modify later if needed)

        loss_crack = (
            crack_normal_stress_bc(model, x1, x3, params, tau0, tauT)**2
        ).mean()

        # --------------------------------------------------
        # SURFACE BC
        # --------------------------------------------------
        s1, s2, s3 = surface_bc(model, x1, x3, params)

        loss_surface = (
            (s1**2).mean() +
            (s2**2).mean() +
            (s3**2).mean()
        )

        # --------------------------------------------------
        # TEMPERATURE BC
        # --------------------------------------------------
        T0 = params["T0"]

        loss_temp_bc = (
            (temperature_ic(model, x3, torch.zeros_like(t))**2).mean() +
            (temperature_bc_left(model, torch.zeros_like(x3), t, T0)**2).mean() +
            (temperature_bc_right(model, x3, t)**2).mean()
        )

        # --------------------------------------------------
        # TOTAL LOSS
        # --------------------------------------------------
        loss = (
            1.0 * loss_pde +
            1.0 * loss_bc +
            5.0 * loss_crack +
            1.0 * loss_surface +
            1.0 * loss_temp_bc
        )

        loss.backward()
        optimizer.step()

        # --------------------------------------------------
        # LOGGING
        # --------------------------------------------------
        if epoch % 500 == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"Total = {loss.item():.3e} | "
                f"PDE = {loss_pde.item():.2e} | "
                f"BC = {loss_bc.item():.2e} | "
                f"Crack = {loss_crack.item():.2e}"
            )

    return model


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":

    print("\nRunning thermo–piezo PINN solver...\n")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = get_all_networks()[0]   # single network
    model.to(DEVICE)

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    trained_model = train(
        model,
        n_epochs=5000,
        lr=1e-3
    )

    print("\n✅ Training complete!")