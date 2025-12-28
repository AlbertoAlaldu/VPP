"""
2D Ising Model under the PVP Framework
Version with intrinsic normalization: ρ = tanh(βJ)

This version corrects the issue where ρ* depended on the simulated temperature
range. Now ρ is an intrinsic function of the system, independent of
arbitrary simulation parameters.
Alberto Alejandro Duarte
Paradox Systems - December 2025
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SIMULATION CORE
# =============================================================================

def checkerboard_update(lattice, beta, color):
    """
    Vectorized checkerboard update.

    Simultaneously updates all "black" (color=0) or "white" (color=1)
    sites of the board. Valid because sites of the same color are not
    neighbors of each other.
    """
    N = lattice.shape[0]

    # Checkerboard mask
    i, j = np.ogrid[:N, :N]
    mask = ((i + j) % 2 == color)

    # Sum of 4 neighbors (periodic boundary conditions)
    neighbors = (np.roll(lattice, 1, axis=0) +
                 np.roll(lattice, -1, axis=0) +
                 np.roll(lattice, 1, axis=1) +
                 np.roll(lattice, -1, axis=1))

    # ΔE = 2 * s * Σ(neighbors)
    dE = 2 * lattice * neighbors

    # Metropolis Criterion
    prob_accept = np.exp(-beta * np.clip(dE, 0, None))
    random_vals = np.random.random((N, N))

    # Flip where accepted, only on sites of the current color
    flip = mask & ((dE <= 0) | (random_vals < prob_accept))
    lattice[flip] *= -1

    return lattice


def full_sweep(lattice, beta):
    """A full sweep = update blacks + update whites."""
    lattice = checkerboard_update(lattice, beta, 0)
    lattice = checkerboard_update(lattice, beta, 1)
    return lattice


def simulate_temperature(N, beta, n_equilib, n_measure, measure_every=2):
    """
    Simulate the system at a fixed temperature.

    Returns:
        m: average magnetization per spin
        chi: magnetic susceptibility
    """
    lattice = np.random.choice([-1, 1], size=(N, N))

    # Equilibration
    for _ in range(n_equilib):
        lattice = full_sweep(lattice, beta)

    # Measurement
    M_samples = []
    for _ in range(n_measure):
        for _ in range(measure_every):
            lattice = full_sweep(lattice, beta)
        M_samples.append(np.sum(lattice))

    M = np.array(M_samples, dtype=np.float64)

    # Statistics
    M_abs_mean = np.mean(np.abs(M))
    M2_mean = np.mean(M**2)

    m = M_abs_mean / (N * N)
    chi = beta * (M2_mean - M_abs_mean**2) / (N * N)

    return m, max(chi, 0)


# =============================================================================
# ρ MAPPING FUNCTIONS
# =============================================================================

def rho_tanh(beta, J=1.0):
    """
    Intrinsic normalization: ρ = tanh(βJ)

    Properties:
    - ρ ∈ [0, 1] for all β ≥ 0
    - ρ → 0 when T → ∞ (β → 0)
    - ρ → 1 when T → 0 (β → ∞)
    - Independent of arbitrary parameters
    - Interpretation: effective correlation between spins
    """
    return np.tanh(beta * J)


def rho_exponential(beta, J=1.0):
    """
    Alternative normalization: ρ = 1 - exp(-βJ)

    Similar to tanh but grows faster near the origin.
    """
    return 1.0 - np.exp(-beta * J)


def rho_linear(beta, beta_min, beta_max):
    """
    Linear normalization (range dependent).
    """
    return (beta - beta_min) / (beta_max - beta_min)


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def ising_pvp_simulation(normalization='tanh'):
    """
    Full simulation of the Ising model under the PVP framework.

    Args:
        normalization: 'tanh', 'exp', or 'linear'
    """

    # --- PARAMETERS ---
    N = 40                  # Lattice size
    n_temps = 35            # Number of temperatures
    n_equilib = 2000        # Equilibration sweeps
    n_measure = 400         # Measurements per temperature
    measure_every = 2       # Sweeps between measurements
    n_realizations = 25     # Realizations for averaging

    J = 1.0                 # Coupling constant
    kB = 1.0                # Boltzmann constant

    # Exact Critical Temperature (Onsager, 1944)
    Tc_exact = 2.0 * J / (kB * np.log(1 + np.sqrt(2)))  # ≈ 2.269
    beta_c = 1.0 / Tc_exact

    # Theoretical critical ρ depending on normalization
    if normalization == 'tanh':
        rho_c_theory = np.tanh(beta_c * J)
    elif normalization == 'exp':
        rho_c_theory = 1.0 - np.exp(-beta_c * J)
    else:
        rho_c_theory = None  # Range dependent

    # Temperature Range
    T_min, T_max = 1.5, 3.5
    temps = np.linspace(T_max, T_min, n_temps)
    betas = 1.0 / temps

    # Calculate ρ based on chosen normalization
    if normalization == 'tanh':
        rhos = rho_tanh(betas, J)
        rho_label = r'$\rho = \tanh(\beta J)$'
    elif normalization == 'exp':
        rhos = rho_exponential(betas, J)
        rho_label = r'$\rho = 1 - e^{-\beta J}$'
    else:  # linear
        rhos = rho_linear(betas, betas.min(), betas.max())
        rho_label = r'$\rho$ linear (range dependent)'

    # Arrays for results
    mag_all = np.zeros((n_realizations, n_temps))
    chi_all = np.zeros((n_realizations, n_temps))

    print("=" * 65)
    print("2D ISING MODEL SIMULATION - PVP FRAMEWORK")
    print("=" * 65)
    print(f"Lattice: {N}×{N} = {N*N} spins")
    print(f"J = {J}, kB = {kB}")
    print(f"Exact Tc (Onsager): {Tc_exact:.4f}")
    print(f"βc = 1/Tc = {beta_c:.4f}")
    print(f"ρ Normalization: {normalization}")
    if rho_c_theory:
        print(f"Theoretical ρc: {rho_c_theory:.4f}")
    print(f"T Range: [{T_min}, {T_max}]")
    print(f"ρ Range: [{rhos.min():.3f}, {rhos.max():.3f}]")
    print(f"Temperatures: {n_temps}, Realizations: {n_realizations}")
    print("-" * 65)

    # --- MAIN LOOP ---
    for r in range(n_realizations):
        print(f"Realization {r+1}/{n_realizations}...", end=" ", flush=True)

        for i, beta in enumerate(betas):
            m, chi = simulate_temperature(N, beta, n_equilib, n_measure, measure_every)
            mag_all[r, i] = m
            chi_all[r, i] = chi

        print("✓")

    # Average over realizations
    mag_mean = np.mean(mag_all, axis=0)
    chi_mean = np.mean(chi_all, axis=0)
    mag_std = np.std(mag_all, axis=0) / np.sqrt(n_realizations)
    chi_std = np.std(chi_all, axis=0) / np.sqrt(n_realizations)

    # --- PVP CURVE CONSTRUCTION ---

    # G(ρ): Gain = Magnetization
    G = mag_mean / np.max(mag_mean)

    # W(ρ): Viability = Susceptibility
    W = chi_mean / np.max(chi_mean)

    # C(ρ): Cost = Convex increasing function
    C = rhos**2

    # Find ρ* (maximum susceptibility)
    idx_opt = np.argmax(chi_mean)
    rho_opt = rhos[idx_opt]
    T_opt = temps[idx_opt]
    beta_opt = betas[idx_opt]

    # Results
    print("-" * 65)
    print("RESULTS:")
    print(f"  Measured T* = {T_opt:.4f}")
    print(f"  Exact Tc    = {Tc_exact:.4f}")
    print(f"  T Error: {100*abs(T_opt - Tc_exact)/Tc_exact:.2f}%")
    print()
    print(f"  Measured ρ* = {rho_opt:.4f}")
    if rho_c_theory:
        print(f"  Theoretical ρc = {rho_c_theory:.4f}")
        print(f"  ρ Error: {100*abs(rho_opt - rho_c_theory)/rho_c_theory:.2f}%")
    print("=" * 65)

    # --- PLOTTING ---

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ===== Left Panel: PVP Space =====
    ax1 = axes[0]

    ax1.plot(rhos, G, '--', color='green', linewidth=2.5,
             label='Structural Coherence ($G$)')
    ax1.plot(rhos, W, '-', color='blue', linewidth=3,
             label='Adaptability Capacity ($W$)')
    ax1.plot(rhos, C, ':', color='red', linewidth=2,
             label='Rigidity Cost ($C$)')

    # Mark measured optimal point
    ax1.axvline(rho_opt, color='black', linestyle=':', alpha=0.6)
    ax1.scatter([rho_opt], [W[idx_opt]], color='blue', s=150, zorder=5,
                edgecolors='black', linewidths=2)

    # Mark theoretical ρc if it exists
    if rho_c_theory:
        ax1.axvline(rho_c_theory, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

    # Annotation
    annotation_text = f'$\\rho^* = {rho_opt:.3f}$\n$(T^* = {T_opt:.2f})$'
    if rho_c_theory:
        annotation_text += f'\n$\\rho_c^{{theory}} = {rho_c_theory:.3f}$'

    ax1.annotate(annotation_text,
                 xy=(rho_opt, W[idx_opt]),
                 xytext=(rho_opt + 0.08, W[idx_opt] - 0.2),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                 arrowprops=dict(arrowstyle='->', color='black'))

    # Colored zones
    ax1.axvspan(rhos.min(), rho_opt, color='yellow', alpha=0.12)
    ax1.axvspan(rho_opt, rhos.max(), color='gray', alpha=0.12)

    # Zone labels
    zone_y = 0.5
    ax1.text((rhos.min() + rho_opt) / 2, zone_y, "Noise Zone\n(Entropy)",
             fontsize=11, color='darkorange', fontweight='bold',
             ha='center', va='center')
    ax1.text((rhos.max() + rho_opt) / 2, zone_y, "SRP Zone\n(Freezing)",
             fontsize=11, color='dimgray', fontweight='bold',
             ha='center', va='center')

    ax1.set_xlabel(f'Systemic Reduction Degree {rho_label}', fontsize=12)
    ax1.set_ylabel('Normalized Magnitude', fontsize=12)
    ax1.set_title('PVP in 2D Ising Model\n(Intrinsic Normalization)',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(rhos.min() - 0.02, rhos.max() + 0.02)
    ax1.set_ylim(0, 1.15)

    # ===== Right Panel: Physical Variables =====
    ax2 = axes[1]

    ax2.errorbar(temps, mag_mean, yerr=mag_std, fmt='o-', color='green',
                 markersize=5, capsize=2, linewidth=1.5,
                 label='Magnetization $\\langle |m| \\rangle$')

    chi_norm = chi_mean / np.max(chi_mean)
    chi_std_norm = chi_std / np.max(chi_mean)
    ax2.errorbar(temps, chi_norm, yerr=chi_std_norm, fmt='s-', color='blue',
                 markersize=5, capsize=2, linewidth=1.5,
                 label='Susceptibility $\\chi / \\chi_{max}$')

    ax2.axvline(Tc_exact, color='red', linestyle='--', linewidth=2.5,
                label=f'Exact $T_c$ = {Tc_exact:.3f}')
    ax2.axvline(T_opt, color='black', linestyle=':', linewidth=2,
                label=f'Measured $T^*$ = {T_opt:.3f}')

    ax2.set_xlabel('Temperature $T$', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.set_title('Phase Transition: Physical Variables', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    plt.tight_layout()

    # Save
    filename_base = f'ising_pvp_{normalization}'
    plt.savefig(f'{filename_base}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{filename_base}.pdf', bbox_inches='tight')
    print(f"\nFigures saved: {filename_base}.png/pdf")

    # Save data
    np.savez(f'{filename_base}_data.npz',
             N=N, J=J, temps=temps, rhos=rhos, betas=betas,
             normalization=normalization,
             magnetization=mag_mean, magnetization_std=mag_std,
             susceptibility=chi_mean, susceptibility_std=chi_std,
             G=G, C=C, W=W,
             rho_opt=rho_opt, T_opt=T_opt,
             Tc_exact=Tc_exact, rho_c_theory=rho_c_theory)
    print(f"Data saved: {filename_base}_data.npz")

    plt.show()

    return {
        'temps': temps,
        'rhos': rhos,
        'mag': mag_mean,
        'chi': chi_mean,
        'G': G, 'C': C, 'W': W,
        'rho_opt': rho_opt,
        'T_opt': T_opt,
        'Tc_exact': Tc_exact,
        'rho_c_theory': rho_c_theory
    }


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys

    # Select normalization from command line or use tanh by default
    if len(sys.argv) > 1:
        norm = sys.argv[1]
        if norm not in ['tanh', 'exp', 'linear']:
            print(f"Normalization '{norm}' not recognized. Using 'tanh'.")
            norm = 'tanh'
    else:
        norm = 'tanh'

    print(f"\nUsing normalization: {norm}\n")
    results = ising_pvp_simulation(normalization=norm)

    # Final Summary
    print("\n" + "=" * 65)
    print("SUMMARY FOR PUBLICATION:")
    print("=" * 65)
    print(f"Model: 2D Ising, Algorithm: Checkerboard Metropolis")
    print(f"Normalization: ρ = tanh(βJ)" if norm == 'tanh' else f"Normalization: {norm}")
    print(f"Critical Temperature:")
    print(f"  Tc (Onsager) = {results['Tc_exact']:.4f}")
    print(f"  T* (measured) = {results['T_opt']:.4f}")
    print(f"PVP Optimal Point:")
    print(f"  ρc (theory)   = {results['rho_c_theory']:.4f}")
    print(f"  ρ* (measured) = {results['rho_opt']:.4f}")
    print("=" * 65)