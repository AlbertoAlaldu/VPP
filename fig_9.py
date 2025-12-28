"""
2D Ising Model (GPU) — Minimal Proxy Robustness: Susceptibility vs Heat Capacity
===============================================================================
VPP Paper — Ising validation with two operational viability proxies:

- W_chi(rho) = susceptibility chi(rho)
- W_cv(rho)  = heat capacity Cv(rho) via energy variance

Minimal run: fixed N=40, single simulation sweep over temperatures on GPU.

Outputs:
- fig_ising_chi_cv_minimal.png/.pdf (publication-ready)
- ising_chi_cv_minimal.npz (data)

Author: Alberto Alejandro Duarte

Paradox Systems — Dec 2025
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# -----------------------------
# Device
# -----------------------------
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("CUDA not available, using CPU (slow)")

torch.set_grad_enabled(False)


# =============================================================================
# GPU simulation core
# =============================================================================

def checkerboard_update_batch(lattice: torch.Tensor, beta: torch.Tensor, color: int) -> torch.Tensor:
    """
    Vectorized checkerboard Metropolis update for a batch of lattices.
    lattice: (B, N, N) float32 with values +/-1
    beta:    (B,) float32
    color:   0 or 1
    """
    B, N, _ = lattice.shape
    dev = lattice.device

    i = torch.arange(N, device=dev).view(1, N, 1)
    j = torch.arange(N, device=dev).view(1, 1, N)
    mask = ((i + j) % 2 == color).expand(B, -1, -1)

    neighbors = (
        torch.roll(lattice,  1, dims=1) +
        torch.roll(lattice, -1, dims=1) +
        torch.roll(lattice,  1, dims=2) +
        torch.roll(lattice, -1, dims=2)
    )

    dE = 2.0 * lattice * neighbors  # J=1 absorbed
    beta3 = beta.view(-1, 1, 1)

    # Metropolis accept prob: exp(-beta * max(dE,0))
    prob = torch.exp(-beta3 * torch.clamp(dE, min=0.0))
    rand = torch.rand_like(lattice)

    flip = mask & ((dE <= 0.0) | (rand < prob))
    lattice = torch.where(flip, -lattice, lattice)
    return lattice


def full_sweep_batch(lattice: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    lattice = checkerboard_update_batch(lattice, beta, 0)
    lattice = checkerboard_update_batch(lattice, beta, 1)
    return lattice


def energy_batch(lattice: torch.Tensor, J: float = 1.0) -> torch.Tensor:
    """
    Energy per configuration with periodic boundary conditions.
    Counts each bond once: right + down neighbors.

    lattice: (B,N,N) +/-1
    returns: (B,) energy
    """
    right = torch.roll(lattice, -1, dims=2)
    down  = torch.roll(lattice, -1, dims=1)
    E = -J * (lattice * right + lattice * down).sum(dim=(1, 2))
    return E


def simulate_ising_gpu_two_proxies(
    N: int,
    betas: np.ndarray,
    n_equilib: int,
    n_measure: int,
    measure_every: int,
    n_realizations: int,
    device: str = "cuda",
    seed: int = 42
):
    """
    Runs a GPU batched simulation across:
      B = T * n_realizations configurations in parallel.

    Returns:
      m(T), chi(T), Cv(T)  (numpy arrays)
    """
    # Reproducibility (best-effort on GPU)
    torch.manual_seed(seed)
    np.random.seed(seed)

    T = len(betas)
    B = T * n_realizations

    betas_t = torch.tensor(betas, device=device, dtype=torch.float32)
    beta_per_config = betas_t.repeat_interleave(n_realizations)  # (B,)

    # Random +/-1 lattice
    lattice = (torch.randint(0, 2, (B, N, N), device=device, dtype=torch.int8) * 2 - 1).to(torch.float32)

    # Equilibration
    for _ in range(n_equilib):
        lattice = full_sweep_batch(lattice, beta_per_config)

    # Measurement
    M_samples = []
    E_samples = []

    for _ in range(n_measure):
        for _ in range(measure_every):
            lattice = full_sweep_batch(lattice, beta_per_config)

        M_samples.append(lattice.sum(dim=(1, 2)))      # (B,)
        E_samples.append(energy_batch(lattice, J=1.0)) # (B,)

    M = torch.stack(M_samples, dim=0)  # (n_measure, B)
    E = torch.stack(E_samples, dim=0)  # (n_measure, B)

    # Magnetization stats
    M_mean = M.mean(dim=0)
    M2_mean = (M * M).mean(dim=0)
    M_abs_mean = M.abs().mean(dim=0)

    # Energy stats
    E_mean = E.mean(dim=0)
    E2_mean = (E * E).mean(dim=0)

    # m(T): abs magnetization per spin
    m = (M_abs_mean / (N * N)).view(T, n_realizations).mean(dim=1)

    # chi(T): beta * ( <M^2> - <M>^2 ) / N^2
    chi = (beta_per_config * (M2_mean - M_mean * M_mean) / (N * N)).view(T, n_realizations).mean(dim=1)
    chi = torch.clamp(chi, min=0.0)

    # Cv(T): beta^2 * ( <E^2> - <E>^2 ) / N^2
    Cv = ((beta_per_config * beta_per_config) * (E2_mean - E_mean * E_mean) / (N * N)).view(T, n_realizations).mean(dim=1)
    Cv = torch.clamp(Cv, min=0.0)

    return m.detach().cpu().numpy(), chi.detach().cpu().numpy(), Cv.detach().cpu().numpy()


# =============================================================================
# Analysis + Plot (publication-ready)
# =============================================================================

def run_minimal_chi_cv(
    N=40,
    n_temps=70,
    T_range=(1.5, 3.5),
    n_realizations=25,
    n_equilib=3000,
    n_measure=500,
    measure_every=2,
    rho_plot_max=0.44,        # <-- recorte para quitar ruido en alta rho
    out_prefix="fig_ising_chi_cv_minimal"
):
    J = 1.0

    # Temperature grid (descending)
    T_min, T_max = T_range
    temps = np.linspace(T_max, T_min, n_temps)
    betas = 1.0 / temps

    # Mapping
    rhos = np.tanh(betas * J)

    # Onsager exact critical point
    Tc_exact = 2.0 * J / np.log(1.0 + np.sqrt(2.0))
    beta_c = 1.0 / Tc_exact
    rho_c = np.tanh(beta_c * J)

    print("=" * 70)
    print("ISING (GPU) — Minimal OM Robustness: chi vs Cv")
    print("=" * 70)
    print(f"N={N}, n_temps={n_temps}, realizations={n_realizations}")
    print(f"Equilib={n_equilib}, measure={n_measure}, every={measure_every}")
    print(f"Tc_exact={Tc_exact:.4f}, rho_c={rho_c:.4f}")
    print("-" * 70)

    m, chi, Cv = simulate_ising_gpu_two_proxies(
        N=N,
        betas=betas,
        n_equilib=n_equilib,
        n_measure=n_measure,
        measure_every=measure_every,
        n_realizations=n_realizations,
        device=device
    )

    # Normalize viability proxies
    W_chi = chi / (chi.max() if chi.max() > 0 else 1.0)
    W_cv  = Cv  / (Cv.max()  if Cv.max()  > 0 else 1.0)

    # Optima indices
    idx_chi = int(np.argmax(chi))
    idx_cv  = int(np.argmax(Cv))

    rho_star_chi = rhos[idx_chi]
    rho_star_cv  = rhos[idx_cv]
    T_star_chi   = temps[idx_chi]
    T_star_cv    = temps[idx_cv]

    print(f"chi peak: rho*={rho_star_chi:.4f}, T*={T_star_chi:.4f}")
    print(f"Cv  peak: rho*={rho_star_cv:.4f},  T*={T_star_cv:.4f}")

    # Crop mask for plotting
    mask = rhos <= rho_plot_max

    # ---------------- Figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=120)

    # Panel A: chi
    ax = axes[0]
    ax.plot(rhos[mask], W_chi[mask], linewidth=2.4, label=r"$W(\rho)=\chi(\rho)$ (norm.)")
    ax.scatter([rho_star_chi], [W_chi[idx_chi]], s=110, zorder=5, edgecolors="black", linewidths=1.2,
               label=rf"$\rho^*_\chi={rho_star_chi:.3f}$")
    ax.axvline(rho_c, linestyle="--", linewidth=2.0, label=rf"$\rho_c$ (Onsager)={rho_c:.3f}")

    ax.set_xlabel(r"Systemic Reduction $\rho=\tanh(\beta J)$", fontsize=11)
    ax.set_ylabel(r"Normalized viability $W(\rho)$", fontsize=11)
    ax.set_title("(A) Susceptibility OM", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.28)
    ax.set_xlim(rhos[mask].min(), rho_plot_max)
    ax.set_ylim(0, 1.06)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    # Panel B: Cv
    ax = axes[1]
    ax.plot(rhos[mask], W_cv[mask], linewidth=2.4, label=r"$W(\rho)=C_V(\rho)$ (norm.)")
    ax.scatter([rho_star_cv], [W_cv[idx_cv]], s=110, zorder=5, edgecolors="black", linewidths=1.2,
               label=rf"$\rho^*_{{C_V}}={rho_star_cv:.3f}$")
    ax.axvline(rho_c, linestyle="--", linewidth=2.0, label=rf"$\rho_c$ (Onsager)={rho_c:.3f}")

    ax.set_xlabel(r"Systemic Reduction $\rho=\tanh(\beta J)$", fontsize=11)
    ax.set_ylabel(r"Normalized viability $W(\rho)$", fontsize=11)
    ax.set_title("(B) Heat-capacity OM", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.28)
    ax.set_xlim(rhos[mask].min(), rho_plot_max)
    ax.set_ylim(0, 1.06)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    fig.suptitle(
        rf"2D Ising (N={N}) — OM robustness in mapped coordinate $\rho$ (GPU)",
        fontsize=13,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()

    # Save figure
    plt.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    print(f"\nSaved: {out_prefix}.png / {out_prefix}.pdf")

    # Save data
    np.savez(
        "ising_chi_cv_minimal.npz",
        N=N,
        temps=temps,
        betas=betas,
        rhos=rhos,
        m=m,
        chi=chi,
        Cv=Cv,
        W_chi=W_chi,
        W_cv=W_cv,
        Tc_exact=Tc_exact,
        rho_c=rho_c,
        rho_star_chi=rho_star_chi,
        rho_star_cv=rho_star_cv,
        T_star_chi=T_star_chi,
        T_star_cv=T_star_cv,
        rho_plot_max=rho_plot_max
    )
    print("Saved data: ising_chi_cv_minimal.npz")

    plt.show()

    return {
        "rho_star_chi": float(rho_star_chi),
        "rho_star_cv": float(rho_star_cv),
        "rho_c": float(rho_c),
        "T_star_chi": float(T_star_chi),
        "T_star_cv": float(T_star_cv),
        "Tc_exact": float(Tc_exact),
    }


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    summary = run_minimal_chi_cv(
        N=40,
        n_temps=70,           # más resolución que tu CPU
        T_range=(1.5, 3.5),
        n_realizations=25,
        n_equilib=3000,       # tu sugerencia
        n_measure=500,
        measure_every=2,
        rho_plot_max=0.44,    # recorte anti-ruido
        out_prefix="fig_ising_chi_cv_minimal"
    )
    print("\nSummary:", summary)
