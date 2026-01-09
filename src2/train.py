import torch
import torch.optim as optim
from networks import get_all_networks
from config import CONFIG
from sampling import (
    sample_domain_points,
    sample_boundary_points,
    sample_interface_points,
    sample_substrate_far_boundary
)
from losses import total_loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(
    n_epochs=20000,
    n_domain=5000,
    n_boundary=1000,
    n_interface=1000,
    lr=1e-3,
    electrically_open=True
):

    print(f"Training on device: {DEVICE}")

    # ---------------------------------------------
    # Initialize networks
    # ---------------------------------------------
    net1, net2, net3 = get_all_networks()
    net1.to(DEVICE)
    net2.to(DEVICE)
    net3.to(DEVICE)

    # ---------------------------------------------
    # Load material properties
    # ---------------------------------------------
    params_fgpm = CONFIG["FGPM"]
    params_hydro = CONFIG["HYDROGEL"]
    params_sub = CONFIG["SUBSTRATE"]
    geom = CONFIG["GEOMETRY"]

    # ---------------------------------------------
    # Optimizer
    # ---------------------------------------------
    optimizer = optim.Adam(
        list(net1.parameters())
        + list(net2.parameters())
        + list(net3.parameters()),
        lr=lr
    )

    # ---------------------------------------------
    # Training Loop
    # ---------------------------------------------
    for epoch in range(1, n_epochs + 1):

        # ================= SAMPLE POINTS =================
        xyt_fgpm, xyt_hydro, xyt_sub = sample_domain_points(
            n_domain, geom
        )

        xt_top = sample_boundary_points(
            n_boundary, geom
        )

        xyt_int_fgpm_hydro, xyt_int_hydro_sub = sample_interface_points(
            n_interface, geom
        )

        xyt_far_sub = sample_substrate_far_boundary(
            n_boundary, geom
        )

        # move to GPU if available
        xyt_fgpm = xyt_fgpm.to(DEVICE)
        xyt_hydro = xyt_hydro.to(DEVICE)
        xyt_sub = xyt_sub.to(DEVICE)

        xt_top = xt_top.to(DEVICE)

        xyt_int_fgpm_hydro = xyt_int_fgpm_hydro.to(DEVICE)
        xyt_int_hydro_sub = xyt_int_hydro_sub.to(DEVICE)

        # ================= ZERO GRAD =================
        optimizer.zero_grad()

        # ================= COMPUTE LOSS =================
        loss, logs = total_loss(
            net1, net2, net3,
            xyt_fgpm, xyt_hydro, xyt_sub,
            xt_top,
            xyt_int_fgpm_hydro,
            xyt_int_hydro_sub,
            params_fgpm,
            params_hydro,
            params_sub,
            electrically_open=electrically_open,
            w_pde=1.0,
            w_bc=5.0,
            w_int=10.0,
            w_far=20.0  
        )

        # ================= BACKPROP =================
        loss.backward()
        optimizer.step()

        # ================= LOGGING =================
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch} | "
                f"Total Loss = {loss.item():.6e} | "
                f"PDE = {logs['pde']:.2e} | "
                f"Top BC = {logs['bc_top']:.2e} | "
                f"Int1 = {logs['interface_1']:.2e} | "
                f"Int2 = {logs['interface_2']:.2e}"
                f"Far = {logs['far']:.2e}"

            )

        # ================= SAVE CHECKPOINT =================
        if epoch % 5000 == 0:
            torch.save({
                "net1": net1.state_dict(),
                "net2": net2.state_dict(),
                "net3": net3.state_dict(),
                "epoch": epoch
            }, f"pinn_checkpoint_epoch{epoch}.pth")


if __name__ == "__main__":
    train()
