# src/pipeline.py
import torch
import numpy as np
import pandas as pd

from pinn import PINNTrainer


class PINNPipeline:
    def __init__(
        self,
        epochs=5000,
        N=3000,
        model_path="trained_pinn.pt",
    ):
        self.epochs = epochs
        self.N = N
        self.model_path = model_path

    def run_training(self):
        print("🔹 Training PINN...")
        trainer = PINNTrainer()
        model = trainer.train(epochs=self.epochs, N=self.N)

        torch.save(model.state_dict(), self.model_path)
        print(f"✔ Model saved to {self.model_path}")

        self.model = model

    def extract_dispersion(self, z0=0.2):
        print("🔹 Extracting dispersion...")

        t = torch.linspace(0, 10, 1000).view(-1, 1)
        r = torch.full_like(t, 0.5)
        z = torch.full_like(t, z0)

        with torch.no_grad():
            X = torch.cat([r, z, t], dim=1)
            u_t = self.model(X).cpu().numpy().flatten()

        # Temporal FFT → omega
        u_fft = np.fft.fft(u_t)
        freqs = np.fft.fftfreq(len(u_t), d=(t[1] - t[0]).item())
        idx = np.argmax(np.abs(u_fft[1 : len(u_t) // 2])) + 1
        omega = 2 * np.pi * abs(freqs[idx])

        # Spatial FFT → k
        r = torch.linspace(0, 1, 1000).view(-1, 1)
        z = torch.full_like(r, z0)
        t = torch.zeros_like(r)

        with torch.no_grad():
            X = torch.cat([r, z, t], dim=1)
            u_r = self.model(X).cpu().numpy().flatten()

        u_fft = np.fft.fft(u_r)
        kfreqs = np.fft.fftfreq(len(u_r), d=(r[1] - r[0]).item())
        idx = np.argmax(np.abs(u_fft[1 : len(u_r) // 2])) + 1
        k = 2 * np.pi * abs(kfreqs[idx])

        c = omega / k

        return {
            "depth": z0,
            "omega": omega,
            "k": k,
            "phase_velocity": c,
        }

    def save_results(self, results, filename="dispersion_results.csv"):
        df = pd.DataFrame([results])
        df.to_csv(filename, index=False)
        print(f"✔ Results saved to {filename}")


if __name__ == "__main__":
    pipeline = PINNPipeline(epochs=500)
    pipeline.run_training()

