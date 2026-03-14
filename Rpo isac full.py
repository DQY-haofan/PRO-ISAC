"""
RPO-ISAC Level 1 FIM — Complete Verification + CRB Curves
===========================================================
Single-file script: run locally, paste terminal output back.

Contents:
  Phase 1: Analytical vs finite-difference derivative verification (small N,M)
  Phase 2: FIM with phi_0 nuisance parameter + Schur complement
  Phase 3: Full-parameter CRB curves (N=1024, M=781)
    - Fig 1: CRB vs distance (0.5–25 km)
    - Fig 2: CRB vs spin rate (0.1–10 deg/s)
    - Fig 3: CRB vs pilot ratio (sensing–comm tradeoff)
    - Fig 4: CRB vs number of scatterers K

Usage:
  python rpo_isac_full.py

Output:
  - Terminal log with all numerical checks
  - rpo_isac_crb_curves.png (4-panel figure)

Author: Haofan + Claude, March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

matplotlib.rcParams.update({
    'font.size': 11,
    'figure.dpi': 150,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})

# ============================================================
# Physical Constants & System Parameters (Table I)
# ============================================================
c = 3e8
k_B = 1.38e-23

# Baseline parameters
PARAMS = {
    'f_c': 26e9,  # Ka-band
    'B': 20e6,  # bandwidth
    'N': 1024,  # subcarriers (full)
    'M': 781,  # OFDM symbols per CPI (full)
    'P_t': 30,  # transmit power [W]
    'D_ant': 0.5,  # antenna diameter [m]
    'eta_ant': 0.6,  # antenna efficiency
    'T_sys': 500,  # system noise temp [K]
    'L_dB': 6.0,  # total system losses [dB]
    'CP_ratio': 0.25,  # CP / useful symbol duration → T_sym ≈ 64 μs
}


def derived_params(p, N_override=None, M_override=None):
    """Compute derived parameters from base params."""
    N = N_override or p['N']
    M = M_override or p['M']
    lam = c / p['f_c']
    Delta_f = p['B'] / N
    T = 1.0 / Delta_f
    T_cp = T * p['CP_ratio']
    T_sym = T + T_cp
    T_cpi = M * T_sym
    G_ant = p['eta_ant'] * (np.pi * p['D_ant'] / lam) ** 2
    L = 10 ** (p['L_dB'] / 10)
    sigma_w2 = k_B * p['T_sys'] * Delta_f
    return {
        'lam': lam, 'N': N, 'M': M, 'Delta_f': Delta_f,
        'T_sym': T_sym, 'T_cpi': T_cpi, 'G_ant': G_ant,
        'L': L, 'sigma_w2': sigma_w2, 'f_c': p['f_c'],
        'P_t': p['P_t'], 'B': p['B'],
    }


# ============================================================
# Rotation Matrix Utilities
# ============================================================
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def R_rodrigues(theta, K_mat, K2):
    return np.eye(3) + np.sin(theta) * K_mat + (1 - np.cos(theta)) * K2


def R_dot(theta, K_mat, K2):
    return np.cos(theta) * K_mat + np.sin(theta) * K2


# ============================================================
# Target Generation
# ============================================================
def generate_target(K_sc, seed=42):
    """Generate random scattering centers."""
    rng = np.random.RandomState(seed)
    p_k = rng.randn(K_sc, 3) * 2.0  # ~2m spread
    sigma_k = (rng.randn(K_sc) + 1j * rng.randn(K_sc)) / np.sqrt(2)
    omega_hat = np.array([0.0, 0.0, 1.0])
    n_hat = np.array([1.0, 0.0, 0.0])
    return p_k, sigma_k, omega_hat, n_hat


# ============================================================
# Core: Compute signal h(theta) and partial derivatives
# ============================================================
def compute_h_and_derivs(d, v_r, Omega, dp, p_k, sigma_k, omega_hat, n_hat,
                         compute_derivs=True):
    """
    Compute h(theta) and optionally dh/d[d, v_r, Omega, phi_0].

    Returns:
        h: complex array [N*M]
        derivs: dict with keys 'dd', 'dvr', 'dO', 'dphi0' (if compute_derivs)
    """
    N = dp['N']
    M = dp['M']
    f_c = dp['f_c']
    Delta_f = dp['Delta_f']
    T_sym = dp['T_sym']
    P_t = dp['P_t']
    G = dp['G_ant']
    L = dp['L']
    lam = dp['lam']

    K_sc = len(sigma_k)
    K_mat = skew(omega_hat)
    K2 = K_mat @ K_mat

    beta = np.sqrt(P_t * G * G * lam ** 2 / ((4 * np.pi) ** 3 * d ** 4 * L))
    tau_0 = 2.0 * d / c
    phi_0_val = 2 * np.pi * f_c * tau_0
    f_d = -2.0 * v_r * f_c / c

    NM = N * M
    h = np.zeros(NM, dtype=complex)

    if compute_derivs:
        dh_dd = np.zeros(NM, dtype=complex)
        dh_dvr = np.zeros(NM, dtype=complex)
        dh_dO = np.zeros(NM, dtype=complex)
        dh_dphi0 = np.zeros(NM, dtype=complex)

    # Precompute subcarrier frequencies (baseband index only)
    n_arr = np.arange(N)
    f_n_arr = f_c + n_arr * Delta_f  # absolute freq per subcarrier

    # Precompute baseband phase slope for range
    phase_range_arr = -2 * np.pi * n_arr * Delta_f * tau_0  # [N]

    for m in range(M):
        t_m = m * T_sym
        theta_m = Omega * t_m
        R_m = R_rodrigues(theta_m, K_mat, K2)

        if compute_derivs:
            Rd_m = R_dot(theta_m, K_mat, K2)

        phase_doppler = 2 * np.pi * f_d * t_m  # scalar

        idx_start = m * N
        idx_end = idx_start + N

        for k in range(K_sc):
            delta_tau_k = (2.0 / c) * (n_hat @ R_m @ p_k[k])

            # Phase: carrier + baseband_range + doppler + micro
            phase_carrier = -phi_0_val
            phase_micro_arr = -2 * np.pi * f_n_arr * delta_tau_k  # [N]

            total_phase = phase_carrier + phase_range_arr + phase_doppler + phase_micro_arr
            g_k_arr = beta * sigma_k[k] * np.exp(1j * total_phase)  # [N]

            h[idx_start:idx_end] += g_k_arr

            if compute_derivs:
                # dh/dd: baseband only (phi_0 separated)
                dh_dd[idx_start:idx_end] += (-1j * 4 * np.pi * n_arr * Delta_f / c) * g_k_arr

                # dh/dv_r
                dh_dvr[idx_start:idx_end] += (-1j * 4 * np.pi * f_c * t_m / c) * g_k_arr

                # dh/dOmega
                d_delta_tau_dO = (2.0 / c) * t_m * (n_hat @ Rd_m @ p_k[k])
                dh_dO[idx_start:idx_end] += (-1j * 2 * np.pi * f_n_arr * d_delta_tau_dO) * g_k_arr

                # dh/dphi_0
                dh_dphi0[idx_start:idx_end] += (-1j) * g_k_arr

    if compute_derivs:
        return h, {'dd': dh_dd, 'dvr': dh_dvr, 'dO': dh_dO, 'dphi0': dh_dphi0}
    return h


# ============================================================
# FIM Computation with Schur Complement
# ============================================================
def compute_fim_4x4(derivs, sigma_w2):
    """Compute 4x4 FIM for [d, v_r, Omega, phi_0]."""
    keys = ['dd', 'dvr', 'dO', 'dphi0']
    P = 4
    J = np.zeros((P, P))
    for i in range(P):
        for j in range(i, P):
            J[i, j] = (2.0 / sigma_w2) * np.real(np.conj(derivs[keys[i]]) @ derivs[keys[j]])
            J[j, i] = J[i, j]
    return J


def marginalize_phi0(J4):
    """Schur complement: marginalize phi_0 from 4x4 FIM -> 3x3."""
    J_theta = J4[:3, :3]
    J_cross = J4[:3, 3:4]
    J_phi = J4[3, 3]
    return J_theta - J_cross @ J_cross.T / J_phi


def crb_from_fim(J3):
    """CRB = sqrt(diag(J^{-1}))."""
    try:
        J_inv = np.linalg.inv(J3)
        return np.sqrt(np.diag(J_inv)), J_inv
    except np.linalg.LinAlgError:
        return np.array([np.inf, np.inf, np.inf]), None


# ============================================================
# PHASE 1: Derivative Verification (small N, M)
# ============================================================
def phase1_verify():
    print("\n" + "=" * 70)
    print("PHASE 1: DERIVATIVE VERIFICATION (N=128, M=64)")
    print("=" * 70)

    dp = derived_params(PARAMS, N_override=128, M_override=64)
    p_k, sigma_k, omega_hat, n_hat = generate_target(5)

    d0, vr0, Om0 = 5e3, -0.5, np.deg2rad(2.0)

    # Analytical
    _, derivs_a = compute_h_and_derivs(d0, vr0, Om0, dp, p_k, sigma_k, omega_hat, n_hat)

    # Numerical: for d, FD gives TOTAL derivative (phi_0 co-varies with d)
    # Our analytical gives PARTIAL derivatives with phi_0 separated.
    # So we verify: (1) v_r and Omega partials directly,
    #               (2) the MARGINALIZED 3x3 FIM and CRB (which must match regardless)
    eps = {'d': 1e-6, 'vr': 1e-6, 'Om': 1e-5}  # Om eps larger: small CPI → tiny rotation angle

    h_vp = compute_h_and_derivs(d0, vr0 + eps['vr'], Om0, dp, p_k, sigma_k, omega_hat, n_hat, False)
    h_vm = compute_h_and_derivs(d0, vr0 - eps['vr'], Om0, dp, p_k, sigma_k, omega_hat, n_hat, False)
    dh_dvr_n = (h_vp - h_vm) / (2 * eps['vr'])

    h_Op = compute_h_and_derivs(d0, vr0, Om0 + eps['Om'], dp, p_k, sigma_k, omega_hat, n_hat, False)
    h_Om = compute_h_and_derivs(d0, vr0, Om0 - eps['Om'], dp, p_k, sigma_k, omega_hat, n_hat, False)
    dh_dO_n = (h_Op - h_Om) / (2 * eps['Om'])

    print("\n  Derivative comparison (analytical vs finite-difference):")
    print("  Note: dh/dd not compared directly (our formulation separates phi_0;")
    print("        FD computes total derivative). Verified via FIM instead.\n")

    pairs = [('dh/dv_r', derivs_a['dvr'], dh_dvr_n),
             ('dh/dΩ', derivs_a['dO'], dh_dO_n)]

    all_pass = True
    for name, da, dn in pairs:
        rel = np.linalg.norm(da - dn) / (np.linalg.norm(dn) + 1e-30)
        status = "✅ PASS" if rel < 1e-3 else "⚠️ FAIL"
        if rel >= 1e-3:
            all_pass = False
        print(f"    {name:>8}: rel_err = {rel:.2e}  {status}")

    # FIM verification: compute "total-derivative" FIM numerically and compare CRB
    # The total derivative dh/dd_total = dh/dd_partial + (dphi_0/dd) * dh/dphi_0
    # where dphi_0/dd = 4*pi*f_c/c
    dphi0_dd = 4 * np.pi * dp['f_c'] / c
    dh_dd_total_from_analytical = derivs_a['dd'] + dphi0_dd * derivs_a['dphi0']

    h_dp = compute_h_and_derivs(d0 + eps['d'], vr0, Om0, dp, p_k, sigma_k, omega_hat, n_hat, False)
    h_dm = compute_h_and_derivs(d0 - eps['d'], vr0, Om0, dp, p_k, sigma_k, omega_hat, n_hat, False)
    dh_dd_total_numerical = (h_dp - h_dm) / (2 * eps['d'])

    rel_dd = np.linalg.norm(dh_dd_total_from_analytical - dh_dd_total_numerical) / \
             (np.linalg.norm(dh_dd_total_numerical) + 1e-30)
    status_dd = "✅ PASS" if rel_dd < 1e-3 else "⚠️ FAIL"
    if rel_dd >= 1e-3:
        all_pass = False
    print(f"    {'dh/dd':>8}: rel_err = {rel_dd:.2e}  {status_dd}  (total = partial + dphi0/dd * dh/dphi0)")

    # FIM comparison
    J4_a = compute_fim_4x4(derivs_a, dp['sigma_w2'])
    J3_a = marginalize_phi0(J4_a)

    print("\n  FIM element comparison (marginalized 3×3):")
    # Reconstruct numerical FIM using total derivatives
    derivs_n_total = {'dd': dh_dd_total_numerical, 'dvr': dh_dvr_n, 'dO': dh_dO_n, 'dphi0': derivs_a['dphi0']}
    J4_n = compute_fim_4x4(derivs_n_total, dp['sigma_w2'])
    J3_n = marginalize_phi0(J4_n)

    names_3 = ['d', 'v_r', 'Ω']
    for i in range(3):
        for j in range(i, 3):
            rel = abs(J3_a[i, j] - J3_n[i, j]) / (abs(J3_n[i, j]) + 1e-30)
            status = "✅" if rel < 2e-2 else "⚠️"  # 2% threshold for small N,M verification
            print(
                f"    J_{names_3[i]},{names_3[j]}: analytical={J3_a[i, j]:.4e}, numerical={J3_n[i, j]:.4e}, rel_err={rel:.2e} {status}")
            if rel >= 2e-2:
                all_pass = False

    crb_vals, _ = crb_from_fim(J3_a)
    print(f"\n  CRB (N=128, M=64, d=5km):")
    print(f"    CRB(d)   = {crb_vals[0] * 1e3:.4f} mm")
    print(f"    CRB(v_r) = {crb_vals[1] * 1e3:.4f} mm/s")
    print(f"    CRB(Ω)   = {np.rad2deg(crb_vals[2]):.6f} deg/s")

    # Sanity: compare with Gaudio
    alpha_eff = np.sqrt(PARAMS['P_t'] * dp['G_ant'] ** 2 * dp['lam'] ** 2 /
                        ((4 * np.pi) ** 3 * d0 ** 4 * dp['L'])) * np.sqrt(np.sum(np.abs(sigma_k) ** 2))
    crb_d_gaudio = (c / 2) * np.sqrt(6 * dp['sigma_w2'] /
                                     (alpha_eff ** 2 * (2 * np.pi * dp['Delta_f']) ** 2 * dp['N'] * dp['M'] * (
                                                 dp['N'] ** 2 - 1)))
    crb_vr_gaudio = (dp['lam'] / 2) * np.sqrt(6 * dp['sigma_w2'] /
                                              (alpha_eff ** 2 * (2 * np.pi * dp['T_sym']) ** 2 * dp['N'] * dp['M'] * (
                                                          dp['M'] ** 2 - 1)))

    print(f"\n  Gaudio et al. (point target) reference:")
    print(f"    CRB(d)   = {crb_d_gaudio * 1e3:.4f} mm  (ours/Gaudio = {crb_vals[0] / crb_d_gaudio:.3f}x)")
    print(f"    CRB(v_r) = {crb_vr_gaudio * 1e3:.4f} mm/s  (ours/Gaudio = {crb_vals[1] / crb_vr_gaudio:.3f}x)")

    eigvals = np.linalg.eigvalsh(J3_a)
    print(f"\n  FIM eigenvalues: {eigvals}")
    print(f"  Condition number: {eigvals[-1] / eigvals[0]:.2e}")

    if all_pass:
        print("\n  ✅✅✅ PHASE 1 ALL CHECKS PASSED ✅✅✅")
    else:
        print("\n  ⚠️⚠️⚠️ PHASE 1 HAS FAILURES ⚠️⚠️⚠️")

    return all_pass


# ============================================================
# PHASE 2: Full-Parameter CRB Curves
# ============================================================
def phase2_crb_curves():
    print("\n" + "=" * 70)
    print("PHASE 2: FULL-PARAMETER CRB CURVES (N=1024, M=781)")
    print("=" * 70)

    dp_full = derived_params(PARAMS)
    p_k, sigma_k, omega_hat, n_hat = generate_target(5)

    print(f"  N={dp_full['N']}, M={dp_full['M']}, T_cpi={dp_full['T_cpi'] * 1e3:.1f} ms")
    print(f"  Total SNR integration: N*M = {dp_full['N'] * dp_full['M']}")

    # ---- Fig 1: CRB vs Distance ----
    print("\n  [Fig 1] CRB vs Distance...")
    d_arr = np.linspace(0.5e3, 25e3, 30)
    vr_fix = -0.5
    Om_fix = np.deg2rad(2.0)

    crb_d_vs_dist = []
    crb_vr_vs_dist = []
    crb_Om_vs_dist = []

    t0 = time.time()
    for i, d_val in enumerate(d_arr):
        _, derivs = compute_h_and_derivs(d_val, vr_fix, Om_fix, dp_full, p_k, sigma_k, omega_hat, n_hat)
        J4 = compute_fim_4x4(derivs, dp_full['sigma_w2'])
        J3 = marginalize_phi0(J4)
        crb, _ = crb_from_fim(J3)
        crb_d_vs_dist.append(crb[0])
        crb_vr_vs_dist.append(crb[1])
        crb_Om_vs_dist.append(crb[2])
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(d_arr)} done ({time.time() - t0:.1f}s)")

    crb_d_vs_dist = np.array(crb_d_vs_dist)
    crb_vr_vs_dist = np.array(crb_vr_vs_dist)
    crb_Om_vs_dist = np.array(crb_Om_vs_dist)

    print(f"    Completed in {time.time() - t0:.1f}s")
    print(f"    CRB(d)  @ 1km: {crb_d_vs_dist[0] * 1e6:.2f} μm, @ 25km: {crb_d_vs_dist[-1] * 1e3:.4f} mm")
    print(f"    CRB(v_r) @ 1km: {crb_vr_vs_dist[0] * 1e6:.2f} μm/s, @ 25km: {crb_vr_vs_dist[-1] * 1e3:.4f} mm/s")
    print(
        f"    CRB(Ω)  @ 1km: {np.rad2deg(crb_Om_vs_dist[0]):.6f} deg/s, @ 25km: {np.rad2deg(crb_Om_vs_dist[-1]):.6f} deg/s")

    # ---- Fig 2: CRB vs Spin Rate ----
    print("\n  [Fig 2] CRB vs Spin Rate...")
    Om_arr = np.deg2rad(np.linspace(0.1, 10, 25))
    d_fix = 5e3

    crb_Om_vs_spin = []
    crb_d_vs_spin = []

    t0 = time.time()
    for i, Om_val in enumerate(Om_arr):
        _, derivs = compute_h_and_derivs(d_fix, vr_fix, Om_val, dp_full, p_k, sigma_k, omega_hat, n_hat)
        J4 = compute_fim_4x4(derivs, dp_full['sigma_w2'])
        J3 = marginalize_phi0(J4)
        crb, _ = crb_from_fim(J3)
        crb_Om_vs_spin.append(crb[2])
        crb_d_vs_spin.append(crb[0])
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(Om_arr)} done ({time.time() - t0:.1f}s)")

    crb_Om_vs_spin = np.array(crb_Om_vs_spin)
    crb_d_vs_spin = np.array(crb_d_vs_spin)
    print(f"    Completed in {time.time() - t0:.1f}s")

    # ---- Fig 3: CRB vs Pilot Ratio (Sensing-Comm Tradeoff) ----
    print("\n  [Fig 3] CRB vs Pilot Ratio (Sensing-Comm Tradeoff)...")
    rho_arr = np.linspace(0.05, 0.95, 20)

    crb_d_vs_rho = []
    crb_Om_vs_rho = []
    rate_vs_rho = []

    t0 = time.time()
    for i, rho in enumerate(rho_arr):
        N_p = max(1, int(rho * dp_full['N']))
        N_d = dp_full['N'] - N_p

        # For sensing: with pilot-only mode, N_eff = N_p subcarriers
        # With monostatic ISAC: all N subcarriers contribute (data is known)
        # The pilot ratio affects ONLY communication rate, not sensing CRB
        # But if we consider a bistatic scenario or partial knowledge,
        # only pilots contribute deterministically.
        #
        # For THIS analysis (monostatic, data known at Tx):
        # CRB is independent of rho! All subcarriers sense.
        # The tradeoff is: rho affects ONLY comm rate.
        #
        # For a fair comparison, we also compute "pilot-only sensing CRB"
        # where only N_p subcarriers are used.

        # Full monostatic CRB (all N subcarriers, rho-independent)
        _, derivs = compute_h_and_derivs(d_fix, vr_fix, Om_fix, dp_full, p_k, sigma_k, omega_hat, n_hat)
        J4 = compute_fim_4x4(derivs, dp_full['sigma_w2'])
        J3 = marginalize_phi0(J4)
        crb_full, _ = crb_from_fim(J3)

        # Pilot-only CRB (use only N_p subcarriers)
        dp_pilot = derived_params(PARAMS, N_override=N_p, M_override=dp_full['M'])
        if N_p >= 2:
            _, derivs_p = compute_h_and_derivs(d_fix, vr_fix, Om_fix, dp_pilot, p_k, sigma_k, omega_hat, n_hat)
            J4_p = compute_fim_4x4(derivs_p, dp_pilot['sigma_w2'])
            J3_p = marginalize_phi0(J4_p)
            crb_pilot, _ = crb_from_fim(J3_p)
        else:
            crb_pilot = np.array([np.inf, np.inf, np.inf])

        crb_d_vs_rho.append((crb_full[0], crb_pilot[0]))
        crb_Om_vs_rho.append((crb_full[2], crb_pilot[2]))

        # Comm rate (Shannon, AWGN, using data subcarriers only)
        H_c_power = dp_full['P_t'] * dp_full['G_ant'] * 10 ** (35 / 10) * dp_full['lam'] ** 2 / \
                    ((4 * np.pi * 2000e3) ** 2 * dp_full['L'])  # direct-to-ground
        snr_c = H_c_power / (k_B * 300 * dp_full['Delta_f'])
        rate = N_d / dp_full['N'] * np.log2(1 + snr_c) * dp_full['B'] / 1e6  # Mbps
        rate_vs_rho.append(rate)

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(rho_arr)} done ({time.time() - t0:.1f}s)")

    crb_d_vs_rho = np.array(crb_d_vs_rho)
    crb_Om_vs_rho = np.array(crb_Om_vs_rho)
    rate_vs_rho = np.array(rate_vs_rho)
    print(f"    Completed in {time.time() - t0:.1f}s")

    # ---- Fig 4: CRB vs K (number of scatterers) ----
    print("\n  [Fig 4] CRB vs K (number of scatterers)...")
    K_arr = [1, 2, 3, 5, 8, 10, 15]
    crb_Om_vs_K = []
    crb_d_vs_K = []

    t0 = time.time()
    for K_val in K_arr:
        pk, sk, wh, nh = generate_target(K_val, seed=42)
        _, derivs = compute_h_and_derivs(d_fix, vr_fix, Om_fix, dp_full, pk, sk, wh, nh)
        J4 = compute_fim_4x4(derivs, dp_full['sigma_w2'])
        J3 = marginalize_phi0(J4)
        crb, _ = crb_from_fim(J3)
        crb_Om_vs_K.append(crb[2])
        crb_d_vs_K.append(crb[0])
        print(f"    K={K_val:>2}: CRB(Ω)={np.rad2deg(crb[2]):.6f} deg/s, CRB(d)={crb[0] * 1e3:.4f} mm")

    crb_Om_vs_K = np.array(crb_Om_vs_K)
    crb_d_vs_K = np.array(crb_d_vs_K)
    print(f"    Completed in {time.time() - t0:.1f}s")

    # ============================================================
    # PLOTTING
    # ============================================================
    print("\n  Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RPO-ISAC Level 1 CRB Analysis (Ka-band, 26 GHz, 20 MHz BW)', fontsize=14, fontweight='bold')

    # Fig 1: CRB vs Distance
    ax = axes[0, 0]
    ax.semilogy(d_arr / 1e3, crb_d_vs_dist * 1e3, 'b-o', markersize=3, label='CRB($d$) [mm]')
    ax.semilogy(d_arr / 1e3, crb_vr_vs_dist * 1e3, 'r-s', markersize=3, label='CRB($v_r$) [mm/s]')
    ax.semilogy(d_arr / 1e3, np.rad2deg(crb_Om_vs_dist) * 1e3, 'g-^', markersize=3, label=r'CRB($\Omega$) [mdeg/s]')
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel('CRB')
    ax.set_title(f'CRB vs Distance ($v_r$={vr_fix} m/s, $\\Omega$={np.rad2deg(Om_fix):.0f}°/s, K={len(sigma_k)})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 25])

    # Fig 2: CRB(Omega) vs Spin Rate
    ax = axes[0, 1]
    ax.semilogy(np.rad2deg(Om_arr), np.rad2deg(crb_Om_vs_spin), 'g-^', markersize=4, label=r'CRB($\Omega$)')
    ax.set_xlabel('Spin Rate [deg/s]')
    ax.set_ylabel(r'CRB($\Omega$) [deg/s]')
    ax.set_title(f'CRB($\\Omega$) vs Spin Rate ($d$={d_fix / 1e3:.0f} km)')
    ax.grid(True, alpha=0.3)
    # Add relative accuracy on right y-axis
    ax2 = ax.twinx()
    rel_acc = crb_Om_vs_spin / Om_arr * 100
    ax2.plot(np.rad2deg(Om_arr), rel_acc, 'k--', alpha=0.5, label='Relative [%]')
    ax2.set_ylabel('Relative CRB [%]', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax.legend(loc='upper left', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)

    # Fig 3: Sensing-Comm Tradeoff
    ax = axes[1, 0]
    ax_rate = ax.twinx()

    l1, = ax.semilogy(rho_arr * 100, np.rad2deg(crb_Om_vs_rho[:, 1]) * 1e3, 'g-^', markersize=4,
                      label=r'CRB($\Omega$) pilot-only [mdeg/s]')
    l1b, = ax.semilogy(rho_arr * 100, np.ones(len(rho_arr)) * np.rad2deg(crb_Om_vs_rho[0, 0]) * 1e3,
                       'g--', alpha=0.5, label=r'CRB($\Omega$) monostatic [mdeg/s]')
    l2, = ax_rate.plot(rho_arr * 100, rate_vs_rho, 'b-o', markersize=3, label='Comm Rate [Mbps]')

    ax.set_xlabel('Pilot Ratio $\\rho = N_p/N$ [%]')
    ax.set_ylabel(r'CRB($\Omega$) [mdeg/s]', color='green')
    ax_rate.set_ylabel('Data Rate [Mbps]', color='blue')
    ax.set_title('Sensing–Communication Tradeoff')
    ax.tick_params(axis='y', labelcolor='green')
    ax_rate.tick_params(axis='y', labelcolor='blue')
    lines = [l1, l1b, l2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='center right')
    ax.grid(True, alpha=0.3)

    # Fig 4: CRB vs K
    ax = axes[1, 1]
    ax.semilogy(K_arr, np.rad2deg(crb_Om_vs_K), 'g-^', markersize=6, label=r'CRB($\Omega$)')
    ax.semilogy(K_arr, crb_d_vs_K * 1e3, 'b-o', markersize=6, label='CRB($d$) [mm]')
    ax.set_xlabel('Number of Scatterers $K$')
    ax.set_ylabel('CRB')
    ax.set_title(f'CRB vs $K$ ($d$={d_fix / 1e3:.0f} km, $\\Omega$={np.rad2deg(Om_fix):.0f}°/s)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_arr)

    plt.tight_layout()
    plt.savefig('rpo_isac_crb_curves.png', bbox_inches='tight')
    plt.close()
    print("  Saved: rpo_isac_crb_curves.png")


# ============================================================
# PHASE 3: Key Numerical Results Summary
# ============================================================
def phase3_summary():
    print("\n" + "=" * 70)
    print("PHASE 3: KEY RESULTS SUMMARY (Table for Paper)")
    print("=" * 70)

    dp = derived_params(PARAMS)
    p_k, sigma_k, omega_hat, n_hat = generate_target(5)

    print(
        f"\n  {'Distance [km]':>15} | {'CRB(d) [mm]':>12} | {'CRB(v_r) [mm/s]':>16} | {'CRB(Ω) [deg/s]':>15} | {'CRB(Ω)/Ω [%]':>13}")
    print(f"  {'-' * 80}")

    Om_fix = np.deg2rad(2.0)
    vr_fix = -0.5

    for d_km in [0.5, 1, 2, 5, 10, 15, 25]:
        d_val = d_km * 1e3
        _, derivs = compute_h_and_derivs(d_val, vr_fix, Om_fix, dp, p_k, sigma_k, omega_hat, n_hat)
        J4 = compute_fim_4x4(derivs, dp['sigma_w2'])
        J3 = marginalize_phi0(J4)
        crb, _ = crb_from_fim(J3)
        rel_Om = crb[2] / Om_fix * 100
        print(
            f"  {d_km:>15.1f} | {crb[0] * 1e3:>12.4f} | {crb[1] * 1e3:>16.4f} | {np.rad2deg(crb[2]):>15.6f} | {rel_Om:>13.4f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("RPO-ISAC FIM Analysis — Full Pipeline")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parameters: f_c={PARAMS['f_c'] / 1e9:.0f} GHz, B={PARAMS['B'] / 1e6:.0f} MHz, "
          f"P_t={PARAMS['P_t']} W, D={PARAMS['D_ant']} m")

    total_t0 = time.time()

    # Phase 1: Verify
    ok = phase1_verify()
    if not ok:
        print("\n⚠️  Phase 1 failed. Fix before proceeding.")

    # Phase 2: CRB curves (this takes a while)
    phase2_crb_curves()

    # Phase 3: Summary table
    phase3_summary()

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"TOTAL RUNTIME: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 70}")