"""
2D Ising Model - Finite-Size Scaling Analysis (GPU Accelerated)
================================================================
VPP Paper - Sensitivity Analysis B

Runs N = 20, 40, 60, 80 and generates publication-ready figure.

Author: Alberto Alejandro Duarte

Paradox Systems - December 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PLOT CONTROL (NEW)
# =========================
RHO_MAX_PLOT = 0.44   # plot window around critical region (readability)
RHO_MIN_PLOT = 0.28

# =============================================================================
# GPU SIMULATION CORE
# =============================================================================

def checkerboard_update_batch(lattice, beta, color):
    """
    Vectorized checkerboard Metropolis update for batch of lattices.
    lattice: (B, N, N)
    beta: (B,) or scalar
    """
    B, N, _ = lattice.shape
    device = lattice.device
    i = torch.arange(N, device=device).view(1, N, 1)
    j = torch.arange(N, device=device).view(1, 1, N)
    mask = ((i + j) % 2 == color).expand(B, -1, -1)

    neighbors = (
        torch.roll(lattice, 1, dims=1) +
        torch.roll(lattice, -1, dims=1) +
        torch.roll(lattice, 1, dims=2) +
        torch.roll(lattice, -1, dims=2)
    )

    dE = 2 * lattice * neighbors
    beta_expanded = beta.view(-1, 1, 1)
    prob = torch.exp(-beta_expanded * torch.clamp(dE, min=0))
    rand = torch.rand_like(lattice)
    flip = mask & ((dE <= 0) | (rand < prob))
    lattice = torch.where(flip, -lattice, lattice)

    return lattice


def full_sweep_batch(lattice, beta):
    """A full sweep = update blacks + update whites."""
    lattice = checkerboard_update_batch(lattice, beta, 0)
    lattice = checkerboard_update_batch(lattice, beta, 1)
    return lattice


def simulate_ising_gpu(
    N=40,
    betas=None,
    n_equilib=2000,
    n_measure=400,
    measure_every=2,
    n_realizations=25,
    device="cuda"
):
    """
    GPU-accelerated Ising simulation.

    Returns magnetization and susceptibility arrays.
    """
    T = len(betas)
    B = T * n_realizations

    betas_tensor = torch.tensor(betas, device=device, dtype=torch.float32)
    betas_expanded = betas_tensor.repeat_interleave(n_realizations)

    # Initialize random lattice
    lattice = torch.randint(0, 2, (B, N, N), device=device, dtype=torch.float32) * 2 - 1

    # Equilibration
    for _ in range(n_equilib):
        lattice = full_sweep_batch(lattice, betas_expanded)

    # Measurement
    M_samples = []
    for _ in range(n_measure):
        for _ in range(measure_every):
            lattice = full_sweep_batch(lattice, betas_expanded)
        M_samples.append(lattice.sum(dim=(1, 2)))

    M = torch.stack(M_samples)  # (n_measure, B)

    # Statistics
    M_mean = M.mean(dim=0)
    M2_mean = (M**2).mean(dim=0)
    M_abs_mean = M.abs().mean(dim=0)

    # Reshape and average over realizations
    m = (M_abs_mean / (N*N)).view(T, n_realizations).mean(dim=1)

    # NOTE: this matches your original approach
    chi = (betas_expanded[:T*n_realizations].view(T, n_realizations)[:, 0] *
           (M2_mean - M_mean**2).view(T, n_realizations).mean(dim=1) / (N*N))

    return m.cpu().numpy(), chi.cpu().numpy()


# =============================================================================
# FINITE-SIZE SCALING ANALYSIS
# =============================================================================

def run_finite_size_analysis(
    lattice_sizes=[20, 40, 60, 80],
    n_temps=60,
    n_realizations=25,
    device="cuda"
):
    """
    Run finite-size scaling analysis across multiple lattice sizes.
    """
    # Physical constants
    J = 1.0
    Tc_exact = 2.0 * J / np.log(1 + np.sqrt(2))  # Onsager
    beta_c = 1.0 / Tc_exact
    rho_c_theory = np.tanh(beta_c * J)

    # Temperature grid
    temps = np.linspace(3.5, 1.5, n_temps)
    betas = 1.0 / temps
    rhos = np.tanh(betas * J)

    results = {}

    print("="*70)
    print("FINITE-SIZE SCALING ANALYSIS (GPU) - v4")
    print("="*70)
    print(f"Lattice sizes: {lattice_sizes}")
    print(f"Temperatures: {n_temps} points in [{temps.min():.2f}, {temps.max():.2f}]")
    print(f"Realizations: {n_realizations}")
    print(f"Tc (Onsager) = {Tc_exact:.4f}")
    print(f"ρc (theory)  = {rho_c_theory:.4f}")
    print("-"*70)

    for N in lattice_sizes:
        # Your scaling choice (unchanged)
        n_eq = max(3000, (N * N) // 2)
        n_meas = max(400, N * 5)

        print(f"  N={N}×{N}: n_equilib={n_eq}, n_measure={n_meas}...", end=" ", flush=True)

        m, chi = simulate_ising_gpu(
            N=N,
            betas=betas,
            n_equilib=n_eq,
            n_measure=n_meas,
            measure_every=2,
            n_realizations=n_realizations,
            device=device
        )

        # Normalize viability
        W = chi / np.max(chi)

        # Find optimum
        idx_opt = np.argmax(chi)
        rho_opt = rhos[idx_opt]
        T_opt = temps[idx_opt]

        results[N] = {
            'temps': temps,
            'betas': betas,
            'rhos': rhos,
            'mag': m,
            'chi': chi,
            'W': W,
            'rho_opt': float(rho_opt),
            'T_opt': float(T_opt),
            'idx_opt': int(idx_opt)
        }

        rho_err = 100 * abs(rho_opt - rho_c_theory) / rho_c_theory
        T_err = 100 * abs(T_opt - Tc_exact) / Tc_exact
        print(f"✓  ρ*={rho_opt:.4f} (err={rho_err:.1f}%), T*={T_opt:.3f} (err={T_err:.1f}%)")

    print("-"*70)

    return results, Tc_exact, float(rho_c_theory)


# =============================================================================
# PUBLICATION FIGURE
# =============================================================================

def create_publication_figure(results, Tc_exact, rho_c_theory, save_prefix='fig_finite_size_v4'):
    """
    Create publication-ready figure for VPP paper.
    Two-panel layout: (A) Viability curves, (B) Convergence plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(results)))

    # ===== Panel A: Viability Curves =====
    ax1 = axes[0]

    for (N, r), color in zip(results.items(), colors):
        rhos = r['rhos'].copy()
        W = r['W'].copy()

        # -------------------------
        # NEW: kill post-0.44 region
        # -------------------------
        W = np.where(rhos <= RHO_MAX_PLOT, W, np.nan)

        ax1.plot(rhos, W, '-', color=color, linewidth=2.2,
                 label=f'$N={N}$ ($\\rho^*={r["rho_opt"]:.3f}$)')

        # Mark optimum only if it lies inside plot window
        if r['rho_opt'] <= RHO_MAX_PLOT:
            ax1.scatter([r['rho_opt']], [r['W'][r['idx_opt']]],
                        color=color, s=100, zorder=5,
                        edgecolors='black', linewidths=1.5)

    ax1.axvline(rho_c_theory, color='red', linestyle='--', linewidth=2,
                label=f'$\\rho_c$ (Onsager) = {rho_c_theory:.3f}')

    ax1.set_xlabel(r'Systemic Reduction $\rho = \tanh(\beta J)$', fontsize=12)
    ax1.set_ylabel(r'Normalized Viability $W(\rho)$', fontsize=12)
    ax1.set_title('(A) Viability Curves Across Lattice Sizes', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9, framealpha=0.95)
    ax1.grid(True, alpha=0.3)

    # NEW: crop x-range
    ax1.set_xlim(RHO_MIN_PLOT, RHO_MAX_PLOT)
    ax1.set_ylim(0, 1.08)

    # ===== Panel B: Finite-Size Scaling =====
    ax2 = axes[1]

    Ns = np.array(list(results.keys()))
    rho_stars = np.array([results[N]['rho_opt'] for N in Ns])

    ax2.errorbar(1/Ns, rho_stars, fmt='o-', color='steelblue', markersize=10,
                 linewidth=2, capsize=4, label=r'Measured $\rho^*$')

    ax2.axhline(rho_c_theory, color='coral', linestyle='--', linewidth=2.5,
                label=f'$\\rho_c$ (Onsager) = {rho_c_theory:.4f}')

    ax2.axhspan(rho_c_theory * 0.98, rho_c_theory * 1.02,
                color='coral', alpha=0.15, label='±2% band')

    from numpy.polynomial import polynomial as P
    coeffs = P.polyfit(1/Ns, rho_stars, 1)
    x_extrap = np.linspace(0, 1/Ns.min() + 0.005, 100)
    y_extrap = P.polyval(x_extrap, coeffs)
    ax2.plot(x_extrap, y_extrap, ':', color='steelblue', alpha=0.6, linewidth=1.5)

    rho_inf = P.polyval(0, coeffs)
    ax2.scatter([0], [rho_inf], color='steelblue', s=120, marker='*', zorder=5,
                edgecolors='black', linewidths=1,
                label=f'$\\rho^*_{{N\\to\\infty}}$ = {rho_inf:.4f}')

    ax2.set_xlabel(r'$1/N$', fontsize=12)
    ax2.set_ylabel(r'$\rho^*$', fontsize=12)
    ax2.set_title('(B) Finite-Size Scaling', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.005, 0.055)
    ax2.set_ylim(0.395, 0.430)

    plt.tight_layout()

    plt.savefig(f'{save_prefix}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_prefix}.pdf', bbox_inches='tight')
    print(f"\nFigure saved: {save_prefix}.png/pdf")

    return fig


def print_summary_table(results, Tc_exact, rho_c_theory):
    print("\n" + "="*70)
    print("SUMMARY TABLE (for paper)")
    print("="*70)
    print(f"{'N':<8} {'ρ*':<10} {'T*':<10} {'ρ error (%)':<12} {'T error (%)':<12}")
    print("-"*70)

    for N, r in results.items():
        rho_err = 100 * abs(r['rho_opt'] - rho_c_theory) / rho_c_theory
        T_err = 100 * abs(r['T_opt'] - Tc_exact) / Tc_exact
        print(f"{N:<8} {r['rho_opt']:<10.4f} {r['T_opt']:<10.4f} {rho_err:<12.2f} {T_err:<12.2f}")

    print("-"*70)
    print(f"Reference: ρc = {rho_c_theory:.4f}, Tc = {Tc_exact:.4f} (Onsager)")
    print("="*70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU (will be slower)")

    results, Tc_exact, rho_c_theory = run_finite_size_analysis(
        lattice_sizes=[20, 40, 60, 80],
        n_temps=60,
        n_realizations=25,
        device=device
    )

    fig = create_publication_figure(results, Tc_exact, rho_c_theory, save_prefix='fig_finite_size_v4')

    print_summary_table(results, Tc_exact, rho_c_theory)

    np.savez('finite_size_data_v4.npz',
             results=results,
             Tc_exact=Tc_exact,
             rho_c_theory=rho_c_theory,
             allow_pickle=True)
    print("\nData saved: finite_size_data_v4.npz")

    plt.show()
