# -*- coding: utf-8 -*-
"""
VPP Biological Validation - CORRECTED VERSION
Parámetros biofísicos: Ea = 55 kJ/mol, Ed = 280 kJ/mol

Authors: Alberto A. Duarte, Carlos Paul Avalos Soto
Paradox Systems R&D - December 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CORRECTED Sharpe–Schoolfield parameters
# From literature: Ea ≈ 55 kJ/mol, Ed ≈ 280 kJ/mol
# Conversion: 1 eV = 96.485 kJ/mol
# ============================================================
k_B = 8.617333262145e-5  # eV/K

# CORRECTED parameters (biophysically grounded)
mu_params = {
    "B0": 1.0,
    "E": 0.5700,      # 55 kJ/mol / 96.485 = 0.5700 eV
    "Eh": 2.9020,     # 280 kJ/mol / 96.485 = 2.9020 eV
    "Th": 315.15,     # K (42°C)
    "Tref": 293.15    # K (20°C)
}

def mu_schoolfield_high_K(TK, B0=1.0, E=0.5700, Eh=2.9020, Th=315.15, Tref=293.15):
    """Sharpe–Schoolfield (high-temp inactivation). Temperature in Kelvin."""
    num = np.exp(-E / k_B * (1.0 / TK - 1.0 / Tref))
    den = 1.0 + np.exp(Eh / k_B * (1.0 / Th - 1.0 / TK))
    return B0 * num / den

# ============================================================
# VPP functional: W(T) = G(T) - C(ρ(T))
# ============================================================
def rho_from_T(TC, Tmin_C, Tmax_C):
    return (TC - Tmin_C) / (Tmax_C - Tmin_C)

def cost_edge_asym_from_rho(rho, kappa_left=0.10, kappa_right=0.90, m=5.0):
    """
    Asymmetric edge-accelerating cost:
      C = kappa_left*(1-rho)^m + kappa_right*rho^m
    With kappa_right >> kappa_left => stronger penalty near hot edge.
    """
    rho = np.clip(rho, 0.0, 1.0)
    return kappa_left * (1.0 - rho) ** m + kappa_right * (rho ** m)

def compute_optima_and_curves(
    Tmin_C, Tmax_C, mu_params,
    kappa_left=0.10, kappa_right=0.90, m=5.0,
    n_grid=20001
):
    TC = np.linspace(Tmin_C, Tmax_C, n_grid)
    TK = TC + 273.15

    mu = mu_schoolfield_high_K(TK, **mu_params)
    mu_max = float(np.max(mu))

    G = mu / mu_max
    rho = rho_from_T(TC, Tmin_C, Tmax_C)
    C = cost_edge_asym_from_rho(rho, kappa_left=kappa_left, kappa_right=kappa_right, m=m)
    W = G - C

    T_mu = float(TC[int(np.argmax(mu))])
    T_vpp = float(TC[int(np.argmax(W))])

    return T_mu, T_vpp, TC, mu, G, C, W

# ============================================================
# Survival simulation (vectorized) with damage h(t)
# ============================================================
def simulate_damage_survival_fast(
    T_set_C,
    Tmin_C, Tmax_C,
    # Environment OU process - CORRECTED values
    T_env_mean_C=38.0, sigma_env=1.6, theta_env=0.02,
    # Internal tracking - CORRECTED values
    theta_ctrl=0.18, sigma_internal=0.45, env_coupling=0.10,
    # Damage model - CORRECTED values
    T_damage_rho=0.75,
    a_damage=0.070,
    b_heal=0.015,
    h_death=0.7,
    # Rare stochastic shocks - CORRECTED values
    shock_prob=0.002,
    shock_scale=0.12,
    # Simulation
    dt=1.0, horizon=2000, n_sims=3000, seed=0
):
    """
    Vectorized simulation across n_sims agents.
    Returns integer survival times in [1..horizon] or horizon if survived.
    """
    rng = np.random.default_rng(seed)

    T_env = np.full(n_sims, T_env_mean_C, dtype=float)
    T_int = np.full(n_sims, T_set_C, dtype=float)
    h = np.zeros(n_sims, dtype=float)

    alive = np.ones(n_sims, dtype=bool)
    times = np.full(n_sims, horizon, dtype=int)

    env_noise = rng.normal(size=(horizon, n_sims))
    int_noise = rng.normal(size=(horizon, n_sims))
    shock_u = rng.random(size=(horizon, n_sims))
    shock_n = np.abs(rng.normal(size=(horizon, n_sims)))

    denom = max(1e-9, (Tmax_C - Tmin_C))
    hot_span = max(1e-9, (1.0 - T_damage_rho))

    for t in range(horizon):
        if not np.any(alive):
            break

        idx = alive

        T_env[idx] = (
            T_env[idx]
            + theta_env * (T_env_mean_C - T_env[idx]) * dt
            + sigma_env * np.sqrt(dt) * env_noise[t, idx]
        )

        T_int[idx] = (
            T_int[idx]
            + theta_ctrl * (T_set_C - T_int[idx]) * dt
            + env_coupling * (T_env[idx] - T_int[idx]) * dt
            + sigma_internal * np.sqrt(dt) * int_noise[t, idx]
        )

        T_int[idx] = np.clip(T_int[idx], Tmin_C - 3.0, Tmax_C + 3.0)

        rho = (T_int[idx] - Tmin_C) / denom
        rho = np.clip(rho, 0.0, 1.0)

        ramp = np.maximum(0.0, rho - T_damage_rho) / hot_span
        dh_plus = a_damage * (ramp ** 2)
        dh_minus = b_heal * (1.0 - ramp)

        shock = (shock_u[t, idx] < shock_prob) * (shock_scale * shock_n[t, idx])

        h[idx] = np.maximum(0.0, h[idx] + (dh_plus - dh_minus) * dt + shock)

        died_now = h[idx] >= h_death
        if np.any(died_now):
            alive_ids = np.where(idx)[0]
            dead_ids = alive_ids[died_now]
            times[dead_ids] = t + 1
            alive[dead_ids] = False

    return times

# ============================================================
# Kaplan–Meier utilities
# ============================================================
def km_curve_censored(times, horizon):
    times = np.asarray(times)
    event = (times < horizon).astype(int)

    deaths = np.bincount(times[event==1], minlength=horizon+1)
    cens = np.bincount(times[event==0], minlength=horizon+1)

    n = len(times)
    at_risk = n
    S = np.ones(horizon+1, dtype=float)
    for t in range(1, horizon+1):
        d_t = deaths[t]
        if at_risk > 0:
            S[t] = S[t-1] * (1.0 - d_t/at_risk)
        else:
            S[t] = S[t-1]
        at_risk -= (deaths[t] + cens[t])
    return S

# ============================================================
# MAIN - Run with CORRECTED parameters
# ============================================================
if __name__ == "__main__":
    # Temperature window
    Tmin_C, Tmax_C = 5.0, 50.0
    
    # VPP cost parameters
    kappa_left = 0.10
    kappa_right = 0.90
    m_cost = 5.0
    
    # Simulation
    horizon = 2000
    n_sims = 3000
    
    print("=" * 70)
    print("VPP BIOLOGICAL VALIDATION - CORRECTED PARAMETERS")
    print("=" * 70)
    print(f"\nBiophysical parameters (from literature):")
    print(f"  Ea = 55 kJ/mol  → E  = {mu_params['E']:.4f} eV")
    print(f"  Ed = 280 kJ/mol → Eh = {mu_params['Eh']:.4f} eV")
    print(f"  Ed/Ea = {280/55:.2f}")
    
    # Compute optima
    T_mu, T_vpp, TC_grid, mu, G, C, W = compute_optima_and_curves(
        Tmin_C, Tmax_C, mu_params,
        kappa_left=kappa_left, kappa_right=kappa_right, m=m_cost
    )
    
    rho_mu = (T_mu - Tmin_C) / (Tmax_C - Tmin_C)
    rho_vpp = (T_vpp - Tmin_C) / (Tmax_C - Tmin_C)
    
    print(f"\nOptima:")
    print(f"  T*_μ   = {T_mu:.2f}°C  (ρ*_μ   = {rho_mu:.3f})")
    print(f"  T*_VPP = {T_vpp:.2f}°C  (ρ*_VPP = {rho_vpp:.3f})")
    print(f"  ΔT     = {T_mu - T_vpp:.2f}°C")
    
    # Run survival simulations
    print(f"\nRunning survival simulations (n={n_sims} per policy)...")
    
    times_mu = simulate_damage_survival_fast(
        T_set_C=T_mu, Tmin_C=Tmin_C, Tmax_C=Tmax_C,
        horizon=horizon, n_sims=n_sims, seed=1
    )
    
    times_vpp = simulate_damage_survival_fast(
        T_set_C=T_vpp, Tmin_C=Tmin_C, Tmax_C=Tmax_C,
        horizon=horizon, n_sims=n_sims, seed=2
    )
    
    # Kaplan-Meier
    S_mu = km_curve_censored(times_mu, horizon)
    S_vpp = km_curve_censored(times_vpp, horizon)
    
    delta_S = S_vpp[horizon] - S_mu[horizon]
    
    print(f"\nSurvival results:")
    print(f"  S_μ({horizon})   = {S_mu[horizon]:.4f}")
    print(f"  S_VPP({horizon}) = {S_vpp[horizon]:.4f}")
    print(f"  ΔS({horizon})    = {delta_S:.4f}")
    
    # Bootstrap CI
    B = 3000
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(B):
        idx_mu = rng.integers(0, n_sims, n_sims)
        idx_vpp = rng.integers(0, n_sims, n_sims)
        S_mu_b = km_curve_censored(times_mu[idx_mu], horizon)[horizon]
        S_vpp_b = km_curve_censored(times_vpp[idx_vpp], horizon)[horizon]
        deltas.append(S_vpp_b - S_mu_b)
    
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
    
    print("\n" + "=" * 70)
    print("COMPARISON WITH PAPER")
    print("=" * 70)
    print(f"Paper target: ΔS ≈ 0.131, CI [0.114, 0.148]")
    print(f"Obtained:     ΔS = {delta_S:.3f}, CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: VPP functional
    ax = axes[0]
    ax.plot(TC_grid, G, label="G(T) = μ/μ_max", linewidth=2)
    ax.plot(TC_grid, C, label="C(T) edge cost", linewidth=2)
    ax.plot(TC_grid, W, label="W(T) = G - C", linewidth=2.5)
    ax.axvline(T_mu, color='blue', linestyle='--', alpha=0.7, label=f"T*_μ = {T_mu:.1f}°C")
    ax.axvline(T_vpp, color='green', linestyle='--', alpha=0.7, label=f"T*_VPP = {T_vpp:.1f}°C")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Value")
    ax.set_title("VPP Functional (Ea=55, Ed=280 kJ/mol)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Kaplan-Meier
    ax = axes[1]
    ax.plot(np.arange(horizon+1), S_mu, label=f"μ-optimal (T={T_mu:.1f}°C)", linewidth=2)
    ax.plot(np.arange(horizon+1), S_vpp, label=f"VPP-optimal (T={T_vpp:.1f}°C)", linewidth=2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("S(t)")
    ax.set_title(f"Kaplan-Meier Survival\nΔS({horizon}) = {delta_S:.3f} (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("vpp_biological_CORRECTED.png", dpi=200)
    print("\nFigure saved: vpp_biological_CORRECTED.png")
    plt.show()
