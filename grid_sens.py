"""
Sensitivity grid for the truncation estimator in (alpha, beta) space.
Report median bias and IQR over R replications at fixed (n, T).
"""

import numpy as np
import json
from numpy.random import default_rng
import sys
sys.path.insert(0, '/home/claude')
from mc_study import simulate_quasi_laplace, truncated_realised_var
from sensitivity import small_jump_var_in_threshold


def grid_sensitivity(params, n, T, R, alphas, betas, eps=5e-4, seed=2024):
    sigma_t = params['sigma']
    rng = default_rng(seed)
    # Pre-generate paths for fair comparison
    paths = []
    for r in range(R):
        sub = default_rng(rng.integers(2**32 - 1))
        _, dX = simulate_quasi_laplace(T, n, sigma_t, params['mu'],
                                       params['lam_p'], params['lam_m'],
                                       params['kappa'], eps, sub)
        paths.append(dX)

    out = {}
    for alpha in alphas:
        for beta in betas:
            vals = np.array([truncated_realised_var(dX, T, alpha, beta)
                             for dX in paths])
            med = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])
            out[(alpha, beta)] = {
                'median': med,
                'bias': med - sigma_t**2,
                'iqr': float(q3 - q1),
                'rmse': float(np.sqrt(np.mean((vals - sigma_t**2)**2))),
            }
    return out


if __name__ == '__main__':
    params_B = dict(sigma=0.3, mu=0.0, lam_p=2.5, lam_m=1.8, kappa=3.0)
    n, T, R = 1600, 2.0, 500

    alphas = [0.5, 1.0, 2.0, 3.0]
    betas = [0.30, 0.40, 0.49]

    s = grid_sensitivity(params_B, n, T, R, alphas, betas)

    print(f"\nSensitivity grid for sigma^2 estimator (Config B,"
          f" n={n}, T={T}, R={R})")
    print(f"True sigma^2 = {0.3**2}\n")
    print(f"  {'alpha':>6} {'beta':>6}  {'med bias':>10} {'IQR':>10} {'RMSE':>10}")
    print('-'*55)
    for alpha in alphas:
        for beta in betas:
            v = s[(alpha, beta)]
            print(f"  {alpha:>6.1f} {beta:>6.2f}  "
                  f"{v['bias']:>+10.4f} {v['iqr']:>10.4f} {v['rmse']:>10.4f}")
        print()

    # Save in JSON
    out = {f"a={a}_b={b}": v for (a, b), v in s.items()}
    with open('/home/claude/grid_sensitivity.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
