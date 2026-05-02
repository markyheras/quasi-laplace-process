"""
Re-run the Monte Carlo using the corrected (no-prefactor) simulator.

Estimators:
- sigma^2: truncated realised variation
- (mu, lam_+, lam_-, kappa): method-of-moments using corrected cumulant system

Cumulant system (corrected):
  Delta^{-1} kappa_1 = mu
  Delta^{-1} kappa_2 = sigma^2 + kappa(1/lp^2 + 1/lm^2)
  Delta^{-1} kappa_3 = 2 * kappa * (1/lp^3 - 1/lm^3)
  Delta^{-1} kappa_4 = 6 * kappa * (1/lp^4 + 1/lm^4)

Compared to the previous implementation, the variance equation has no
factor of 2 in front of kappa.
"""

import numpy as np
import json
from numpy.random import default_rng
from scipy.optimize import least_squares
import sys
sys.path.insert(0, '/home/claude')
from mc_study_v2 import simulate_quasi_laplace_v2


def truncated_realised_var(dX, T, alpha=1.0, beta=0.49):
    n = len(dX)
    dt = T / n
    u_n = alpha * dt ** beta
    return np.sum(dX[np.abs(dX) <= u_n] ** 2) / T


def sample_cumulants(dX):
    m1 = np.mean(dX)
    c = dX - m1
    m2 = np.mean(c ** 2); m3 = np.mean(c ** 3); m4 = np.mean(c ** 4)
    return m1, m2, m3, m4 - 3 * m2 ** 2


def mom_estimator_v2(dX, T, sigma2_hat):
    """Corrected MoM: A = kappa(rp^2+rm^2) directly (no factor 2)."""
    n = len(dX)
    Delta = T / n
    k1, k2, k3, k4 = sample_cumulants(dX)
    A = max(k2 / Delta - sigma2_hat, 1e-12)        # Note: NO division by 2
    B = k3 / (2.0 * Delta)
    C = max(k4 / (6.0 * Delta), 1e-12)

    # Initial guess
    r_avg2 = max(C / A, 1e-8)
    r_avg = np.sqrt(r_avg2)
    d_init = (2.0 / 3.0) * B / A if A > 1e-12 else 0.0
    d_init = np.clip(d_init, -0.5 * r_avg, 0.5 * r_avg)
    r_p0 = max(r_avg + 0.5 * d_init, 1e-3)
    r_m0 = max(r_avg - 0.5 * d_init, 1e-3)
    kappa0 = max(A / (r_p0 ** 2 + r_m0 ** 2), 1e-3)

    log_init = np.log([kappa0, r_p0, r_m0])

    def residuals(log_params):
        kp, rp, rm = np.exp(log_params)
        return np.array([
            (kp * (rp ** 2 + rm ** 2) - A) / max(A, 1e-6),
            (kp * (rp ** 3 - rm ** 3) - B) / max(abs(B), 1e-2),
            (kp * (rp ** 4 + rm ** 4) - C) / max(C, 1e-6),
        ])

    lb = np.log([1e-3, 1e-2, 1e-2])
    ub = np.log([1e3, 50, 50])
    try:
        sol = least_squares(residuals, log_init, bounds=(lb, ub),
                            method='trf', max_nfev=2000)
        kp, rp, rm = np.exp(sol.x)
        return k1 / Delta, 1.0 / rp, 1.0 / rm, kp
    except Exception:
        return None


def run_mc(name, params, n_grid, R, gamma=0.5, Delta0=0.05, eps=5e-4,
           seed0=2024, alpha=1.0, beta=0.49):
    sigma_t, mu_t = params['sigma'], params['mu']
    lp_t, lm_t, kp_t = params['lam_p'], params['lam_m'], params['kappa']
    truth = {'sigma2': sigma_t ** 2, 'mu': mu_t,
             'lam_p': lp_t, 'lam_m': lm_t, 'kappa': kp_t}
    names = list(truth.keys())

    T_for_n = {n: float(n ** gamma * Delta0) for n in n_grid}
    estimates = {n: {nm: [] for nm in names} for n in n_grid}
    n_failed = {n: 0 for n in n_grid}

    rng = default_rng(seed0)
    for r in range(R):
        for n in n_grid:
            T = T_for_n[n]
            sub = default_rng(rng.integers(2 ** 32 - 1))
            try:
                _, dX = simulate_quasi_laplace_v2(T, n, sigma_t, mu_t,
                                                  lp_t, lm_t, kp_t, eps, sub)
                s2 = truncated_realised_var(dX, T, alpha, beta)
                m = mom_estimator_v2(dX, T, s2)
                if m is None:
                    n_failed[n] += 1
                    continue
                mu_h, lp_h, lm_h, kp_h = m
                vals = {'sigma2': s2, 'mu': mu_h,
                        'lam_p': lp_h, 'lam_m': lm_h, 'kappa': kp_h}
                for nm in names:
                    estimates[n][nm].append(vals[nm])
            except Exception:
                n_failed[n] += 1

    summary = {'name': name, 'truth': truth,
               'n_grid': list(n_grid), 'T': T_for_n,
               'R': R, 'failed': n_failed,
               'median_bias': {}, 'iqr': {}, 'rmse': {}}
    for n in n_grid:
        summary['median_bias'][n] = {}
        summary['iqr'][n] = {}
        summary['rmse'][n] = {}
        for nm in names:
            arr = np.array(estimates[n][nm])
            if arr.size == 0:
                continue
            med = float(np.median(arr))
            q1, q3 = np.percentile(arr, [25, 75])
            arr_t = arr[(arr > np.percentile(arr, 5)) &
                        (arr < np.percentile(arr, 95))]
            rmse = float(np.sqrt(np.mean((arr_t - truth[nm]) ** 2))) \
                if arr_t.size > 0 else float('nan')
            summary['median_bias'][n][nm] = med - truth[nm]
            summary['iqr'][n][nm] = float(q3 - q1)
            summary['rmse'][n][nm] = rmse
    return summary


if __name__ == '__main__':
    n_grid = [400, 1600, 6400]
    R = 500
    pA = dict(sigma=0.5, mu=0.1, lam_p=2.0, lam_m=2.0, kappa=2.0)
    pB = dict(sigma=0.3, mu=0.0, lam_p=2.5, lam_m=1.8, kappa=3.0)

    print("Config A...")
    SA = run_mc('A', pA, n_grid, R)
    print("Config B...")
    SB = run_mc('B', pB, n_grid, R)

    out = {'A': SA, 'B': SB}
    with open('/home/claude/mc_v2_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)

    names = ['sigma2', 'mu', 'lam_p', 'lam_m', 'kappa']
    for cfg, S in out.items():
        print(f"\n=== Config {cfg} === truth: {S['truth']}")
        print(f"  T values: {S['T']}")
        print(f"  failed:   {S['failed']}")
        for n in n_grid:
            print(f"\n  n={n}, T_n={S['T'][n]}:")
            for nm in names:
                print(f"    {nm:>7}: bias={S['median_bias'][n][nm]:+8.4f}  "
                      f"IQR={S['iqr'][n][nm]:8.4f}  "
                      f"RMSE5={S['rmse'][n][nm]:8.4f}")
