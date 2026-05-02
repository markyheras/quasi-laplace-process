"""
Hybrid simulator for the quasi-Laplace process (corrected v2).

Density (no lam prefactor):
    nu_QL(dx) = kappa * exp(-lam_+ x)/x dx        for x > 0
              + kappa * exp(-lam_- |x|)/|x| dx    for x < 0

Asmussen-Rosinski hybrid scheme:
  1. Gaussian increment with variance sigma^2 + small-jump variance
  2. Compound Poisson jumps with |x| > eps, rate kappa * E1(lam * eps)
  3. Rejection sampling for jump sizes from exp(-lam x)/x on (eps, infty)
  4. Subtract mean of |x|>eps jumps so E[X_t] = mu*t
"""

import numpy as np
from scipy.special import exp1


def small_jump_variance_v2(eps, lam_p, lam_m, kappa):
    """
    int_{|x|<eps} x^2 nu(dx) under the no-prefactor density.
    int_0^eps x^2 (kappa exp(-lam x)/x) dx
        = kappa * int_0^eps x exp(-lam x) dx
        = kappa * [1 - (1 + lam*eps) exp(-lam*eps)] / lam^2
    """
    def half(lam):
        return kappa * (1.0 - (1.0 + lam * eps) * np.exp(-lam * eps)) / lam ** 2
    return half(lam_p) + half(lam_m)


def simulate_quasi_laplace_v2(T, n, sigma, mu, lam_p, lam_m, kappa,
                              eps=5e-4, rng=None):
    """Simulate the quasi-Laplace process under the corrected density."""
    rng = rng if rng is not None else np.random.default_rng()
    dt = T / n

    # Rate of |x| > eps jumps under no-prefactor density:
    # nu({|x|>eps}) = kappa * int_eps^inf exp(-lam x)/x dx = kappa * E1(lam*eps)
    rate_p = kappa * exp1(lam_p * eps)
    rate_m = kappa * exp1(lam_m * eps)

    # Mean of |x| > eps jumps:
    # int_eps^inf x * (kappa exp(-lam x)/x) dx = kappa * exp(-lam eps)/lam
    mean_pos = (kappa / lam_p) * np.exp(-lam_p * eps)
    mean_neg = -(kappa / lam_m) * np.exp(-lam_m * eps)
    drift_comp = mean_pos + mean_neg

    # Small-jump variance correction
    var_small = small_jump_variance_v2(eps, lam_p, lam_m, kappa)
    sigma_total = np.sqrt(sigma ** 2 + var_small)

    # Gaussian + small-jump increments
    dX = sigma_total * np.sqrt(dt) * rng.standard_normal(n)

    # Sample large jumps on each side
    for rate, lam, sign in [(rate_p, lam_p, +1.0), (rate_m, lam_m, -1.0)]:
        N = rng.poisson(rate * T)
        if N == 0:
            continue
        # Rejection sampling for size: target density propto exp(-lam x)/x on (eps, inf)
        # Proposal: x = eps + Exp(lam), accept with prob eps/x
        sizes = np.empty(N)
        k = 0
        while k < N:
            need = N - k
            xs = eps + rng.exponential(1.0 / lam, size=need)
            us = rng.uniform(size=need)
            ok = us < (eps / xs)
            n_ok = ok.sum()
            sizes[k:k + n_ok] = xs[ok]
            k += n_ok
        times = rng.uniform(0, T, N)
        for tt, ss in zip(times, sizes):
            i = int(tt / dt)
            if i < n:
                dX[i] += sign * ss

    # Drift correction: subtract mean of |x|>eps jumps, add mu drift
    dX += (mu - drift_comp) * dt

    X = np.concatenate([[0.0], np.cumsum(dX)])
    return X, dX


if __name__ == '__main__':
    # Verify variance matches Corollary 5 (corrected): sigma^2 + kappa*(1/lp^2 + 1/lm^2)
    from numpy.random import default_rng

    print("Verification of Var(X_1) under corrected density:")
    for cfg, p in [('A', dict(sigma=0.5, mu=0.1, lam_p=2.0, lam_m=2.0, kappa=2.0)),
                   ('B', dict(sigma=0.3, mu=0.0, lam_p=2.5, lam_m=1.8, kappa=3.0))]:
        T, n, R = 1.0, 50000, 500
        rng = default_rng(0)
        finals = []
        for r in range(R):
            sub = default_rng(rng.integers(2 ** 32 - 1))
            X, _ = simulate_quasi_laplace_v2(T, n, **p, eps=5e-4, rng=sub)
            finals.append(X[-1])
        finals = np.array(finals)
        truth = p['sigma'] ** 2 + p['kappa'] * (1 / p['lam_p'] ** 2 + 1 / p['lam_m'] ** 2)
        print(f"  Config {cfg}: empirical Var = {finals.var():.4f}  truth = {truth:.4f}")
