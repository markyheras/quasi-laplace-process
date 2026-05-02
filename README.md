# Quasi-Laplace Process: Reproducibility Package

This repository contains the reference implementation accompanying the paper

> **Quasi-Laplace Processes: Hybrid Lévy Models with Continuous Quadratic Variation**
> Kenneth P. Perez, University of Science and Technology of Southern Philippines, Cagayan de Oro City, Philippines.

## Contents

- `simulator.py` — hybrid simulator for the quasi-Laplace process
  using the Asmussen–Rosiński small-jump variance correction and
  rejection sampling for jumps with `|x| > eps`.
- `mc_runner.py` — driver that runs the Monte Carlo study reported in
  Table 2 of the paper. Performs 500 replications across three
  `(n, T_n)` cells under the mixed-asymptotic regime
  `T_n = n^{1/2} * 0.05`.
- `mc_results_v2.json` — raw Monte Carlo output (median bias, IQR,
  trimmed RMSE) for both parameter configurations.
- `grid_sens.py` — runs the sensitivity grid for the truncation
  parameters `(alpha, beta)` reported in Table 3.
- `grid_sensitivity.json` — raw sensitivity grid output.

## Reproducing the Tables

### Table 2 (Monte Carlo results)

```bash
python mc_runner.py
```

This produces `mc_v2_results.json` with the bias, IQR, and trimmed
RMSE for each of the five parameters across the three sample sizes
`n in {400, 1600, 6400}` and the two configurations.

### Table 3 (sensitivity to truncation parameters)

```bash
python grid_sens.py
```

This produces `grid_sensitivity.json` with bias, IQR, and RMSE of
`sigma^2_hat` for `alpha in {0.5, 1, 2, 3}` and
`beta in {0.30, 0.40, 0.49}` at `n=1600`, `T_n=2`, with `R=500`
replications per cell.

## Configurations Used

- **Configuration A (symmetric):** `sigma=0.5, mu=0.1, lam_+=lam_-=2.0, kappa=2.0`
- **Configuration B (asymmetric):** `sigma=0.3, mu=0.0, lam_+=2.5, lam_-=1.8, kappa=3.0`

## Levy Density

The simulator uses the corrected (no-prefactor) form of the
quasi-Laplace Levy density:

    nu_QL(dx) = kappa * exp(-lam_+ * x) / x  dx,    for x > 0
              = kappa * exp(-lam_- * |x|) / |x| dx, for x < 0

This corresponds to the asymmetric variance-gamma (CGMY) parametrisation
with `C_+ = C_- = kappa`, `M = lam_+`, `G = lam_-`.

## Requirements

- Python 3.9+
- numpy >= 1.23
- scipy >= 1.10

```bash
pip install numpy scipy
```

## License

MIT License.

## Citation

If you use this code, please cite:

```
Perez, K. P. (2026). Quasi-Laplace Processes: Hybrid Levy Models
with Continuous Quadratic Variation.
Statistics & Probability Letters (forthcoming).
```

## Contact

Kenneth P. Perez — markyheras@gmail.com
