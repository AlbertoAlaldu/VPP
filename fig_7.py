# ============================================================
# COLAB - ISING (GPU Torch) + SENSITIVITY FIG.7 (full pipeline)
# ============================================================

# --- Colab tip: Runtime -> Change runtime type -> GPU ---

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ----------------------------
# 0) Global config (EDIT HERE)
# ----------------------------
FAST = False
SEED = 42

# Base simulation params (match your defaults)
N_BASE = 40
N_TEMPS = 35
T_RANGE = (1.5, 3.5)  # (T_min, T_max) but we will generate from T_max down to T_min (like your scripts)
MEASURE_EVERY = 2

# If FAST=True, downscale (useful to validate pipeline quickly)
if FAST:
    N_REALIZATIONS_BASE = 8
    N_EQUILIB = 600
    N_MEASURE = 150
    LATTICE_SIZES = [20, 40]          # smaller set
    N_REALIZATIONS_SIZE = 8
else:
    N_REALIZATIONS_BASE = 25
    N_EQUILIB = 2000
    N_MEASURE = 400
    LATTICE_SIZES = [20, 40, 60, 80]
    N_REALIZATIONS_SIZE = 20

# χ definition for consistency.
# - chi_def="mean" chi = β(<M^2> - <M>^2)/(N^2)
CHI_DEF = "mean"  
SAVE_PREFIX_A = "sensitivity_mapping_gpu"
SAVE_PREFIX_B = "sensitivity_size_gpu"
SAVE_PREFIX_SUMMARY = "sensitivity_summary_gpu"


# ----------------------------
# 1) GPU Torch Ising simulator
# ----------------------------
import torch

def _make_checkerboard_masks(N, device):
    i = torch.arange(N, device=device).view(N, 1)
    j = torch.arange(N, device=device).view(1, N)
    mask0 = ((i + j) % 2 == 0)
    mask1 = ~mask0
    return mask0.view(1,1,N,N), mask1.view(1,1,N,N)

@torch.no_grad()
def _checkerboard_update_batch(lattice, betas, mask):
    # lattice: (B,R,N,N) int8 in {-1,+1}
    # betas:   (B,1,1,1) float32
    # mask:    (1,1,N,N) bool

    nbrs = (torch.roll(lattice,  1, dims=2) +
            torch.roll(lattice, -1, dims=2) +
            torch.roll(lattice,  1, dims=3) +
            torch.roll(lattice, -1, dims=3))

    dE = 2 * lattice.to(torch.int16) * nbrs.to(torch.int16)     # values in {-8,-4,0,4,8}
    dEpos = torch.clamp(dE, min=0).to(torch.float32)

    prob = torch.exp(-betas * dEpos)                            # (B,R,N,N)
    rnd  = torch.rand_like(prob)

    accept = (dE <= 0) | (rnd < prob)
    flip = mask & accept
    return torch.where(flip, -lattice, lattice)

@torch.no_grad()
def _full_sweep_batch(lattice, betas, mask0, mask1):
    lattice = _checkerboard_update_batch(lattice, betas, mask0)
    lattice = _checkerboard_update_batch(lattice, betas, mask1)
    return lattice

@torch.no_grad()
def run_ising_simulation_gpu_torch(
    N=40,
    n_temps=35,
    n_equilib=2000,
    n_measure=400,
    measure_every=2,
    n_realizations=25,
    T_range=(1.5, 3.5),
    seed=42,
    chi_def="mean",     # "mean" or "abs"
    verbose=True,
):
    """
    Batched GPU simulation across all temperatures (B) and realizations (R).
    Returns dict: N, temps, betas, mag_mean/std, chi_mean/std (numpy arrays).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[GPU torch] device = {device}")
        if device.type != "cuda":
            print("WARNING: No CUDA detected. This will run on CPU.")

    # Seeds
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    T_min, T_max = T_range
    # like your code: temps go from T_max down to T_min
    temps = torch.linspace(T_max, T_min, n_temps, device=device, dtype=torch.float32)
    betas = (1.0 / temps).view(n_temps, 1, 1, 1)  # (B,1,1,1)

    # lattice: (B,R,N,N) spins in {-1,+1}
    lattice = torch.randint(0, 2, (n_temps, n_realizations, N, N),
                            device=device, dtype=torch.int8)
    lattice = lattice * 2 - 1

    mask0, mask1 = _make_checkerboard_masks(N, device)

    # Equilibration
    if verbose:
        print(f"Equilibrating: n_equilib={n_equilib} sweeps (batched over B={n_temps}, R={n_realizations})")
    for _ in range(n_equilib):
        lattice = _full_sweep_batch(lattice, betas, mask0, mask1)

    # Accumulators per (B,R)
    sum_abs = torch.zeros((n_temps, n_realizations), device=device, dtype=torch.float32)
    sum_sq  = torch.zeros((n_temps, n_realizations), device=device, dtype=torch.float32)
    sum_m   = torch.zeros((n_temps, n_realizations), device=device, dtype=torch.float32)

    # Measurement
    if verbose:
        print(f"Measuring: n_measure={n_measure}, measure_every={measure_every}")
    for _ in range(n_measure):
        for _ in range(measure_every):
            lattice = _full_sweep_batch(lattice, betas, mask0, mask1)

        M = lattice.sum(dim=(-1, -2)).to(torch.float32)  # (B,R)
        sum_sq  += M * M
        sum_abs += M.abs()
        sum_m   += M

    M_abs_mean = sum_abs / n_measure
    M2_mean    = sum_sq  / n_measure
    M_mean     = sum_m   / n_measure

    denom = float(N * N)
    m = M_abs_mean / denom  # magnetization per spin (abs)

    beta_BR = betas.view(n_temps, 1)  # (B,1)
    if chi_def == "abs":
        chi = beta_BR * (M2_mean - M_abs_mean * M_abs_mean) / denom
    else:
        chi = beta_BR * (M2_mean - M_mean * M_mean) / denom

    chi = torch.clamp(chi, min=0.0)

    # Means over realizations (axis=1)
    mag_mean = m.mean(dim=1)
    chi_mean = chi.mean(dim=1)
    mag_std  = m.std(dim=1, unbiased=True) / np.sqrt(n_realizations)
    chi_std  = chi.std(dim=1, unbiased=True) / np.sqrt(n_realizations)

    return {
        "N": int(N),
        "temps": temps.detach().cpu().numpy(),
        "betas": (1.0 / temps).detach().cpu().numpy(),
        "mag_mean": mag_mean.detach().cpu().numpy(),
        "mag_std":  mag_std.detach().cpu().numpy(),
        "chi_mean": chi_mean.detach().cpu().numpy(),
        "chi_std":  chi_std.detach().cpu().numpy(),
    }


# ----------------------------
# 2) ρ mapping functions (match your sensitivity script)
# ----------------------------
def rho_tanh(beta, J=1.0):
    return np.tanh(beta * J)

def rho_exponential(beta, J=1.0):
    return 1.0 - np.exp(-beta * J)

def rho_logistic(beta, J=1.0, k=3.0):
    x = beta * J
    return 1.0 / (1.0 + np.exp(-k * (x - 0.5)))

def rho_algebraic(beta, J=1.0, n=2):
    x = beta * J
    return (x**n) / (1.0 + x**n)

RHO_MAPPINGS = {
    "tanh": {
        "func": rho_tanh,
        "label": r"$\rho = \tanh(\beta J)$",
        "color": "blue",
        "description": "Hyperbolic tangent (baseline)",
    },
    "exp": {
        "func": rho_exponential,
        "label": r"$\rho = 1 - e^{-\beta J}$",
        "color": "red",
        "description": "Exponential saturation",
    },
    "logistic": {
        "func": lambda b, J=1.0: rho_logistic(b, J, k=3.0),
        "label": r"$\rho = \sigma(3(\beta J - 0.5))$",
        "color": "green",
        "description": "Logistic (k=3)",
    },
    "algebraic": {
        "func": lambda b, J=1.0: rho_algebraic(b, J, n=2),
        "label": r"$\rho = (\beta J)^2/(1+(\beta J)^2)$",
        "color": "purple",
        "description": "Hill function (n=2)",
    },
}


# ----------------------------
# 3) Analysis functions (ρ*, ρc, W(ρ))
# ----------------------------
def analyze_with_mapping(sim_data, mapping_name):
    """
    Returns dict with keys used by the plotting code:
    rhos, W, rho_opt, T_opt, rho_c_theory, Tc_exact
    """
    J = 1.0
    mapping = RHO_MAPPINGS[mapping_name]
    rho_func = mapping["func"]

    betas = np.asarray(sim_data["betas"], dtype=float)
    temps = np.asarray(sim_data["temps"], dtype=float)
    chi_mean = np.asarray(sim_data["chi_mean"], dtype=float)

    # Onsager critical temperature (J=kB=1)
    Tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0))
    beta_c = 1.0 / Tc_exact

    # rhos across simulated betas
    rhos = rho_func(betas, J)

    # W(ρ): normalized susceptibility (as in your script)
    W = chi_mean / np.max(chi_mean)

    # Optimum (max susceptibility)
    idx_opt = int(np.argmax(chi_mean))
    rho_opt = float(rhos[idx_opt])
    T_opt = float(temps[idx_opt])

    # theoretical rho_c for this mapping
    rho_c_theory = float(rho_func(np.array([beta_c]), J)[0])

    return {
        "mapping": mapping_name,
        "Tc_exact": float(Tc_exact),
        "rhos": rhos,
        "W": W,
        "rho_opt": rho_opt,
        "T_opt": T_opt,
        "rho_c_theory": rho_c_theory,
    }

def compute_mapping_results(sim_data):
    results = {}
    for name in RHO_MAPPINGS.keys():
        results[name] = analyze_with_mapping(sim_data, name)
    return results

def print_mapping_table(results):
    print("\n" + "="*70)
    print("SENSITIVITY A: ρ* across mappings (measured vs theoretical ρc)")
    print("="*70)
    print(f"{'Mapping':<12} {'rho*':<12} {'rho_c':<12} {'Error%':<10} {'T*':<8}")
    print("-"*70)
    for name, r in results.items():
        err = 100.0 * abs(r["rho_opt"] - r["rho_c_theory"]) / abs(r["rho_c_theory"])
        print(f"{name:<12} {r['rho_opt']:<12.4f} {r['rho_c_theory']:<12.4f} {err:<10.2f} {r['T_opt']:<8.3f}")
    print("-"*70)
    print(f"Exact Tc (Onsager): {results['tanh']['Tc_exact']:.4f}")
    print("="*70)


# ----------------------------
# 4) Plotting (Fig.7A style + SRP shading)
# ----------------------------
def plot_sensitivity_mappings(sim_data, results, save_prefix="sensitivity_mapping_gpu"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    Tc_exact = results["tanh"]["Tc_exact"]

    for ax, (name, mapping) in zip(axes.flat, RHO_MAPPINGS.items()):
        r = results[name]
        rhos = r["rhos"]
        W = r["W"]
        rho_opt = r["rho_opt"]
        rho_c = r["rho_c_theory"]

        ax.plot(rhos, W, "-", color=mapping["color"], linewidth=2.5,
                label=r"Viability $W(\rho)$")

        idx_opt = int(np.argmax(W))
        ax.scatter([rho_opt], [W[idx_opt]], color=mapping["color"], s=150,
                   zorder=5, edgecolors="black", linewidths=2,
                   label=fr"$\rho^* = {rho_opt:.3f}$")

        ax.axvline(rho_c, color="red", linestyle="--", linewidth=2, alpha=0.7,
                   label=fr"$\rho_c^{{theory}} = {rho_c:.3f}$")

        # SRP zone (shading to the right of rho*)
        ax.axvspan(rho_opt, np.max(rhos), color="gray", alpha=0.15, label="SRP Zone")

        err = 100.0 * abs(rho_opt - rho_c) / abs(rho_c)
        ax.text(0.95, 0.95, f"Error: {err:.1f}%", transform=ax.transAxes,
                fontsize=11, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        ax.set_xlabel(mapping["label"], fontsize=11)
        ax.set_ylabel(r"Normalized Viability $W(\rho)$", fontsize=11)
        ax.set_title(mapping["description"], fontsize=12, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.15)

    plt.suptitle(
        "Sensitivity Analysis A: Robustness Across ρ Mappings\n"
        f"(N = {sim_data['N']}, Tc = {Tc_exact:.3f}, χ-def = {CHI_DEF})",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved: {save_prefix}.png / {save_prefix}.pdf")
    return fig


# ----------------------------
# 5) Finite-size analysis (Fig.7B style)
# ----------------------------
def sensitivity_finite_size(lattice_sizes, n_realizations, seed_base=123):
    Ns = np.array(lattice_sizes, dtype=int)

    # exact Tc and mapped rho_c (tanh)
    Tc_exact = 2.0 / np.log(1.0 + np.sqrt(2.0))
    rho_c_theory = np.tanh(1.0 / Tc_exact)

    rho_stars = []
    T_stars = []

    for i, N in enumerate(Ns):
        simN = run_ising_simulation_gpu_torch(
            N=int(N),
            n_temps=N_TEMPS,
            n_equilib=N_EQUILIB,
            n_measure=N_MEASURE,
            measure_every=MEASURE_EVERY,
            n_realizations=n_realizations,
            T_range=T_RANGE,
            seed=seed_base + i,
            chi_def=CHI_DEF,
            verbose=True
        )
        res = analyze_with_mapping(simN, "tanh")
        rho_stars.append(res["rho_opt"])
        T_stars.append(res["T_opt"])

    rho_stars = np.array(rho_stars, dtype=float)
    T_stars = np.array(T_stars, dtype=float)

    return {
        "Ns": Ns,
        "rho_stars": rho_stars,
        "T_stars": T_stars,
        "Tc_exact": float(Tc_exact),
        "rho_c_theory": float(rho_c_theory),
    }

def plot_sensitivity_finite_size(size_results, save_prefix="sensitivity_size_gpu"):
    Ns = size_results["Ns"]
    rho_stars = size_results["rho_stars"]
    T_stars = size_results["T_stars"]
    Tc_exact = size_results["Tc_exact"]
    rho_c_theory = size_results["rho_c_theory"]

    invN = 1.0 / Ns

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.35)

    # Panel A: rho* vs N
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(Ns, rho_stars, "o-", linewidth=2)
    ax1.axhline(rho_c_theory, linestyle="--", linewidth=2, alpha=0.7)
    ax1.set_xlabel("Lattice Size N")
    ax1.set_ylabel(r"$\rho^*(N)$")
    ax1.set_title("A: Finite-size dependence of $\\rho^*$", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Panel B: rho* vs 1/N (extrapolation)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(invN, rho_stars, "o", markersize=8)
    coeffs = np.polyfit(invN, rho_stars, 1)
    x_ex = np.linspace(0, invN.max()*1.05, 200)
    ax2.plot(x_ex, np.polyval(coeffs, x_ex), ":", linewidth=2, alpha=0.8)
    rho_inf = np.polyval(coeffs, 0)
    ax2.scatter([0], [rho_inf], s=120, marker="*", zorder=5,
                label=fr"$\rho^*_{{N\to\infty}}={rho_inf:.4f}$")
    ax2.axhline(rho_c_theory, linestyle="--", linewidth=2, alpha=0.7,
                label=fr"$\rho_c={rho_c_theory:.4f}$")
    ax2.set_xlabel(r"$1/N$")
    ax2.set_ylabel(r"$\rho^*$")
    ax2.set_title("B: Linear extrapolation in $1/N$", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel C: T* vs 1/N (extrapolation)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(invN, T_stars, "s", markersize=8)
    coeffsT = np.polyfit(invN, T_stars, 1)
    ax3.plot(x_ex, np.polyval(coeffsT, x_ex), ":", linewidth=2, alpha=0.8)
    T_inf = np.polyval(coeffsT, 0)
    ax3.scatter([0], [T_inf], s=120, marker="*", zorder=5,
                label=fr"$T^*_{{N\to\infty}}={T_inf:.4f}$")
    ax3.axhline(Tc_exact, linestyle="--", linewidth=2, alpha=0.7,
                label=fr"$T_c={Tc_exact:.4f}$")
    ax3.set_xlabel(r"$1/N$")
    ax3.set_ylabel(r"$T^*$")
    ax3.set_title("C: Finite-size scaling of $T^*$", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel D: relative error vs N (log scale)
    ax4 = fig.add_subplot(gs[1, 1])
    rho_err = 100.0 * np.abs(rho_stars - rho_c_theory) / rho_c_theory
    T_err = 100.0 * np.abs(T_stars - Tc_exact) / Tc_exact
    ax4.semilogy(Ns, rho_err, "o-", linewidth=2, label=r"$\rho^*$ error")
    ax4.semilogy(Ns, T_err, "s-", linewidth=2, label=r"$T^*$ error")
    ax4.set_xlabel("Lattice Size N")
    ax4.set_ylabel("Relative Error (%)")
    ax4.set_title("D: Convergence to Onsager solution", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which="both")

    plt.suptitle(
        "Sensitivity Analysis B: Finite-Size Effects\n"
        f"(ρ mapping: tanh(βJ), χ-def = {CHI_DEF})",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    plt.savefig(f"{save_prefix}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved: {save_prefix}.png / {save_prefix}.pdf")
    return fig


# ----------------------------
# 6) Summary figure (A bars + B errors)
# ----------------------------
def create_summary_figure(mapping_results, size_results, save_prefix="sensitivity_summary_gpu"):
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    # Panel A: bars for rho* vs rho_c per mapping
    ax1 = fig.add_subplot(gs[0, 0])
    mapping_names = list(mapping_results.keys())
    rho_stars = [mapping_results[m]["rho_opt"] for m in mapping_names]
    rho_cs = [mapping_results[m]["rho_c_theory"] for m in mapping_names]
    errs = [100.0 * abs(r - c) / abs(c) for r, c in zip(rho_stars, rho_cs)]

    x = np.arange(len(mapping_names))
    width = 0.35
    ax1.bar(x - width/2, rho_stars, width, edgecolor="black", label=r"Measured $\rho^*$")
    ax1.bar(x + width/2, rho_cs, width, edgecolor="black", label=r"Theoretical $\rho_c$")

    for i, e in enumerate(errs):
        ax1.annotate(f"{e:.1f}%", xy=(x[i] - width/2, rho_stars[i] + 0.01),
                     ha="center", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([RHO_MAPPINGS[m]["label"] for m in mapping_names],
                        fontsize=9, rotation=15, ha="right")
    ax1.set_ylabel(r"$\rho^*$ / $\rho_c$")
    ax1.set_title("A: Robustness across ρ mappings", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel B: finite-size error curves
    ax2 = fig.add_subplot(gs[0, 1])
    Ns = size_results["Ns"]
    rho_starsN = size_results["rho_stars"]
    T_starsN = size_results["T_stars"]
    Tc_exact = size_results["Tc_exact"]
    rho_c = size_results["rho_c_theory"]

    rho_err = 100.0 * np.abs(rho_starsN - rho_c) / rho_c
    T_err = 100.0 * np.abs(T_starsN - Tc_exact) / Tc_exact

    ax2.semilogy(Ns, rho_err, "o-", linewidth=2, label=r"$\rho^*$ error")
    ax2.semilogy(Ns, T_err, "s-", linewidth=2, label=r"$T^*$ error")
    ax2.set_xlabel("Lattice Size N")
    ax2.set_ylabel("Relative Error (%)")
    ax2.set_title("B: Finite-size convergence", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")

    plt.suptitle(f"Combined Summary (χ-def = {CHI_DEF})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved: {save_prefix}.png / {save_prefix}.pdf")
    return fig


# ----------------------------
# 7) RUN EVERYTHING
# ----------------------------
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("\n=== STEP 1/3: Base simulation (N=40) on GPU ===")
sim_data = run_ising_simulation_gpu_torch(
    N=N_BASE,
    n_temps=N_TEMPS,
    n_equilib=N_EQUILIB,
    n_measure=N_MEASURE,
    measure_every=MEASURE_EVERY,
    n_realizations=N_REALIZATIONS_BASE,
    T_range=T_RANGE,
    seed=SEED,
    chi_def=CHI_DEF,
    verbose=True
)

print("\n=== STEP 2/3: Sensitivity A (ρ mappings) ===")
mapping_results = compute_mapping_results(sim_data)
print_mapping_table(mapping_results)
plot_sensitivity_mappings(sim_data, mapping_results, save_prefix=SAVE_PREFIX_A)

print("\n=== STEP 3/3: Sensitivity B (finite-size) ===")
size_results = sensitivity_finite_size(
    lattice_sizes=LATTICE_SIZES,
    n_realizations=N_REALIZATIONS_SIZE,
    seed_base=SEED + 100
)
plot_sensitivity_finite_size(size_results, save_prefix=SAVE_PREFIX_B)

print("\n=== SUMMARY FIGURE ===")
create_summary_figure(mapping_results, size_results, save_prefix=SAVE_PREFIX_SUMMARY)

print("\nDone. Files in current directory:")
print([f for f in os.listdir(".") if f.endswith(".png") or f.endswith(".pdf")])
