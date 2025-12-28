"""
PVP Simulation: Topological Regime
Small-World Network Model (Watts-Strogatz)

In this model:
- ρ = 1 - p, where p is the rewiring probability
- ρ = 0: Random network (high efficiency, low structure)
- ρ = 1: Regular/Lattice network (high structure, low efficiency)

Viability combines:
- G(ρ): Clustering (local robustness, redundancy)
- C(ρ): Path length (communication cost)

The "sweet spot" for small-worldness is where there is high clustering AND low path length.

Paradox Systems - December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm  # Optional, for progress bar


def watts_strogatz_pvp():
    """
    PVP Simulation in Watts-Strogatz networks.
    """

    # =========================================================================
    # PARAMETERS
    # =========================================================================
    N = 200           # Number of nodes
    k = 6             # Initial neighbors (must be even)
    n_probs = 40      # Number of p values to test
    n_samples = 20    # Samples per p value (to reduce noise)

    # Range of rewiring probabilities
    # We use a logarithmic scale to capture the transition
    p_values = np.logspace(-4, 0, n_probs)  # From 0.0001 to 1

    # =========================================================================
    # REFERENCE VALUES
    # =========================================================================
    # Regular Lattice (p=0): max clustering, max path length
    G_lattice = nx.watts_strogatz_graph(N, k, 0)
    C_lattice = nx.average_clustering(G_lattice)
    L_lattice = nx.average_shortest_path_length(G_lattice)

    # Equivalent Random Network (p=1): low clustering, low path length
    G_random = nx.watts_strogatz_graph(N, k, 1)
    C_random = nx.average_clustering(G_random)
    L_random = nx.average_shortest_path_length(G_random)

    print("=" * 65)
    print("PVP SIMULATION IN WATTS-STROGATZ NETWORKS")
    print("=" * 65)
    print(f"Nodes: {N}, Neighbors: {k}")
    print(f"Values of p: {n_probs}, Samples per point: {n_samples}")
    print(f"\nReferences:")
    print(f"  Lattice (p=0): C = {C_lattice:.4f}, L = {L_lattice:.2f}")
    print(f"  Random (p=1):  C = {C_random:.4f}, L = {L_random:.2f}")
    print("-" * 65)

    # =========================================================================
    # SIMULATION
    # =========================================================================
    clustering_mean = np.zeros(n_probs)
    clustering_std = np.zeros(n_probs)
    path_length_mean = np.zeros(n_probs)
    path_length_std = np.zeros(n_probs)

    print("Simulating networks...")

    for i, p in enumerate(p_values):
        C_samples = []
        L_samples = []

        for _ in range(n_samples):
            G = nx.watts_strogatz_graph(N, k, p)
            C_samples.append(nx.average_clustering(G))

            # Check connectivity before calculating path length
            if nx.is_connected(G):
                L_samples.append(nx.average_shortest_path_length(G))
            else:
                # For disconnected graphs, use the largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                L_samples.append(nx.average_shortest_path_length(subgraph))

        clustering_mean[i] = np.mean(C_samples)
        clustering_std[i] = np.std(C_samples)
        path_length_mean[i] = np.mean(L_samples)
        path_length_std[i] = np.std(L_samples)

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_probs} completed")

    print("Simulation completed.\n")

    # =========================================================================
    # DEFINE ρ AND PVP METRICS
    # =========================================================================

    # ρ = Structural Rigidity = 1 - p
    # ρ = 0: Random network
    # ρ = 1: Regular lattice
    rho = 1 - p_values

    # G(ρ): Gain = Normalized Clustering
    # G = 1 when C = C_lattice, G ≈ 0 when C = C_random
    G_norm = (clustering_mean - C_random) / (C_lattice - C_random)
    G_norm = np.clip(G_norm, 0, 1)

    # C(ρ): Cost = Normalized Path length (inverted so high ρ = high cost)
    # We want C = 0 when L = L_random (efficient)
    # and C = 1 when L = L_lattice (inefficient)
    L_norm = (path_length_mean - L_random) / (L_lattice - L_random)
    L_norm = np.clip(L_norm, 0, 1)

    # =========================================================================
    # VIABILITY W(ρ)
    # =========================================================================
    # Option 1: W = G - λC (Standard PVP form)
    lambda_cost = 1.0
    W_additive = G_norm - lambda_cost * L_norm

    # Option 2: W = G / L (small-worldness ratio)
    # Avoid division by zero
    W_ratio = G_norm / (L_norm + 0.01)

    # Option 3: W = G * (1 - L) (product)
    W_product = G_norm * (1 - L_norm)

    # We use the additive form for consistency with the rest of the paper
    W = W_additive
    W_norm = (W - W.min()) / (W.max() - W.min())

    # =========================================================================
    # FIND OPTIMUM
    # =========================================================================
    idx_opt = np.argmax(W)
    rho_opt = rho[idx_opt]
    p_opt = p_values[idx_opt]

    print("=" * 65)
    print("RESULTS:")
    print(f"  Optimal ρ* = {rho_opt:.4f}")
    print(f"  Corresponding p* = {p_opt:.4f}")
    print(f"  Clustering at ρ* = {clustering_mean[idx_opt]:.4f}")
    print(f"  Path length at ρ* = {path_length_mean[idx_opt]:.2f}")
    print(f"  G(ρ*) = {G_norm[idx_opt]:.3f}")
    print(f"  C(ρ*) = {L_norm[idx_opt]:.3f}")
    print("=" * 65)

    # =========================================================================
    # PLOTTING
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ----- Left Panel: PVP Space -----
    ax1 = axes[0]

    # Sort by increasing ρ for plotting
    sort_idx = np.argsort(rho)
    rho_s = rho[sort_idx]
    G_s = G_norm[sort_idx]
    L_s = L_norm[sort_idx]
    W_s = W_norm[sort_idx]

    ax1.plot(rho_s, G_s, '--', color='green', linewidth=2.5,
             label='Local Robustness $G$ (Clustering)')
    ax1.plot(rho_s, L_s, ':', color='red', linewidth=2.5,
             label='Transport Inefficiency $C$ (Path Length)')
    ax1.plot(rho_s, W_s, '-', color='blue', linewidth=3,
             label='Net Viability $W = G - C$')

    # Optimal Point
    idx_opt_sorted = np.argmax(W_s)
    rho_opt_s = rho_s[idx_opt_sorted]

    ax1.axvline(rho_opt_s, color='black', linestyle=':', alpha=0.6)
    ax1.scatter([rho_opt_s], [W_s[idx_opt_sorted]], color='blue', s=150,
                zorder=5, edgecolors='black', linewidths=2)

    # Annotation
    ax1.annotate(f'$\\rho^* = {rho_opt_s:.3f}$\n(Small-World)',
                 xy=(rho_opt_s, W_s[idx_opt_sorted]),
                 xytext=(rho_opt_s - 0.15, W_s[idx_opt_sorted] - 0.2),
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                 arrowprops=dict(arrowstyle='->', color='black'))

    # Zones
    ax1.axvspan(0, 0.5, color='yellow', alpha=0.1)
    ax1.axvspan(rho_opt_s, 1, color='gray', alpha=0.1)

    ax1.text(0.15, 0.5, "Random Network\n(Efficient but\nno structure)",
             fontsize=10, color='darkorange', fontweight='bold',
             ha='center', va='center')
    ax1.text(0.92, 0.5, "SRP\n(Rigid\nLattice)",
             fontsize=10, color='dimgray', fontweight='bold',
             ha='center', va='center')

    ax1.set_xlabel('Structural Rigidity $\\rho = 1 - p$', fontsize=12)
    ax1.set_ylabel('Normalized Magnitude', fontsize=12)
    ax1.set_title('PVP in Small-World Networks\n(Watts-Strogatz Model)',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.1)

    # ----- Right Panel: Logarithmic p Scale -----
    ax2 = axes[1]

    # Plot vs p on log scale (classic way to present Watts-Strogatz)
    ax2.semilogx(p_values, clustering_mean / C_lattice, 'o-', color='green',
                 markersize=4, linewidth=1.5, label='$C(p) / C(0)$')
    ax2.semilogx(p_values, path_length_mean / L_lattice, 's-', color='red',
                 markersize=4, linewidth=1.5, label='$L(p) / L(0)$')

    # Mark small-world region
    ax2.axvline(p_opt, color='blue', linestyle='--', linewidth=2,
                label=f'$p^* = {p_opt:.4f}$')
    ax2.axvspan(0.001, 0.1, color='blue', alpha=0.1, label='Small-World Zone')

    ax2.set_xlabel('Rewiring Probability $p$', fontsize=12)
    ax2.set_ylabel('Normalized Metric', fontsize=12)
    ax2.set_title('Classic Small-World Transition\n(Watts & Strogatz, 1998)',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='right', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(1e-4, 1)
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('watts_strogatz_pvp.png', dpi=150, bbox_inches='tight')
    plt.savefig('watts_strogatz_pvp.pdf', bbox_inches='tight')
    print("\nFigures saved: watts_strogatz_pvp.png/pdf")

    plt.show()

    # Save data
    np.savez('watts_strogatz_pvp_data.npz',
             N=N, k=k,
             p_values=p_values, rho=rho,
             clustering=clustering_mean, clustering_std=clustering_std,
             path_length=path_length_mean, path_length_std=path_length_std,
             G=G_norm, C=L_norm, W=W,
             rho_opt=rho_opt, p_opt=p_opt,
             C_lattice=C_lattice, L_lattice=L_lattice,
             C_random=C_random, L_random=L_random)
    print("Data saved: watts_strogatz_pvp_data.npz")

    return {
        'rho': rho,
        'p': p_values,
        'G': G_norm,
        'C': L_norm,
        'W': W,
        'rho_opt': rho_opt,
        'p_opt': p_opt
    }


if __name__ == "__main__":
    results = watts_strogatz_pvp()

    print("\n" + "=" * 65)
    print("PUBLICATION SUMMARY:")
    print("=" * 65)
    print(f"Model: Watts-Strogatz (N=200, k=6)")
    print(f"Optimal ρ* = {results['rho_opt']:.4f}")
    print(f"Corresponding p* = {results['p_opt']:.4f}")
    print(f"Interpretation: The optimum is close to the lattice regime")
    print(f"                but requires ~{results['p_opt']*100:.2f}% of random shortcuts")
    print("=" * 65)