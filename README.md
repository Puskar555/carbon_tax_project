## Overview

Carbon taxation reduces greenhouse gas emissions but places a disproportionate burden on lower-income households, who spend a larger share of their income on energy. Once the government collects tax revenue, the design of how that revenue is returned to households matters just as much as the tax rate itself.

This project builds a **static heterogeneous-agent general equilibrium model** in Python and compares two recycling schemes:

- **Scheme A:** Universal lump-sum rebate — every household receives the same transfer T = R / N
- **Scheme B:** Targeted compensation — bottom 40% of households receive λ times the mean rebate; top 60% receive a reduced transfer determined by the budget constraint

Both schemes are evaluated on four outcome variables: welfare change by income decile, tax burden share, share of households that gain, and aggregate labour supply.

---

## Repository Structure

```
carbon_tax_project/
│
├── main.ipynb                  ← Complete working notebook (run this)
├── README.md                   ← This file
│
└── figures/                    ← All output charts (auto-generated)
    ├── fig1_wage_distribution.png
    ├── fig2_tax_burden_baseline.png
    ├── fig3_welfare_change.png
    ├── fig4_share_of_winners.png
    ├── fig5_aggregate_labour_supply.png
    └── fig6_equity_efficiency_frontier.png
```

---

## Model

### Households

The economy has N = 1,000 households indexed by their wage w_i, drawn from a simulated log-normal distribution calibrated to Gini ≈ 0.35 (consistent with European OECD economies such as Germany or France).

Each household solves:

```
max  U(c, l) = log(c) + α · log(1 − l)
s.t. c · (1 + τ·β) = w_i · l + T_i
```

where:
- `c`   — consumption of the numeraire good
- `l`   — labour supply on [0, 1]
- `τ`   — carbon tax rate (ad valorem on energy)
- `β`   — energy expenditure share of consumption (e = β·c)
- `w_i` — household wage (exogenous, drawn from simulated distribution)
- `T_i` — government transfer received (varies by recycling scheme)
- `α`   — leisure preference parameter

### Analytical Solution (First-Order Condition)

Substituting the energy demand rule `e = β·c` and the budget constraint into the FOC:

```
l* = [w_i · (1 + τ·β) − α · T_i] / [w_i · (1 + τ·β + α)]
```

This gives a closed-form optimal labour supply for each household. It is verified numerically using `scipy.optimize.minimize_scalar`.

### Government Budget Constraint

The government runs a balanced budget — all tax revenue is returned to households:

```
Σ τ · e*_i  =  Σ T_i
```

The equilibrium transfer is found by **fixed-point iteration**: start with T = 0, solve household decisions, compute revenue, update T, repeat until convergence (typically 4–10 iterations).

### Redistribution Schemes

| | Scheme A | Scheme B |
|---|---|---|
| Who receives | All households equally | Bottom 40% receive λ · T̄; top 60% receive a reduced transfer |
| Transfer formula | T = R / N | T_low = λ · T̄; T_high from budget balance |
| Baseline λ | — | 1.50 |

---

## Calibration Parameters

| Parameter | Symbol | Baseline | Sensitivity Range |
|-----------|--------|----------|-------------------|
| Carbon tax rate | τ | 0.10 | 0.05, 0.15, 0.20 |
| Energy expenditure share | β | 0.15 | 0.10, 0.20 |
| Leisure preference | α | 0.50 | 0.30, 0.70 |
| Targeting multiplier | λ | 1.50 | 1.25, 1.75, 2.00 |
| Number of households | N | 1,000 | 5,000 (robustness) |
| Target Gini coefficient | G | 0.35 | 0.25, 0.45 |

---

## Notebook Structure (`main.ipynb`)

The notebook is divided into 10 clearly separated sections. Run them top to bottom in order.

| Section | Content | Course Week |
|---------|---------|-------------|
| 1 | Imports and parameters dict | Week 1 |
| 2 | Wage simulation and Gini calibration (`np.trapezoid`) | Week 5, 6 |
| 3 | Household optimisation — analytical FOC and numerical verification (`scipy.optimize`) | Week 4 |
| 4 | Baseline: no redistribution — tax burden by decile | Week 7–8 |
| 5 | Scheme A fixed-point iteration loop | Week 2–3 |
| 6 | Scheme B fixed-point iteration loop (two-tier transfers) | Week 2–3 |
| 7 | Newton's method robustness check (`scipy.optimize.newton`) | Week 3 |
| 8 | Outcome variables by decile — welfare, burden, winners, labour | Week 7–9 |
| 9 | Sensitivity analysis over τ and λ grids | Week 9 |
| 10 | All six output figures (Matplotlib) | Week 1 |

---

## Output Figures

| Figure | Title | Key Message |
|--------|-------|-------------|
| 1 | Simulated Wage Distribution + Lorenz Curve | Confirms Gini calibration to target |
| 2 | Carbon Tax Burden by Decile (No Redistribution) | Shows raw regressivity of the tax |
| 3 | Net Welfare Change by Decile: Scheme A vs Scheme B | Central result of the project |
| 4 | Share of Winners by Decile | Breadth of benefit across the income distribution |
| 5 | Aggregate Labour Supply: Baseline vs A vs B | Efficiency cost comparison |
| 6 | Equity-Efficiency Frontier (λ sensitivity) | Policy tradeoff across targeting intensity |

All figures are saved automatically to the `figures/` folder when Section 10 is run.

---

## Setup and Requirements

### Python Version

Python 3.10 or higher is recommended. The notebook was developed and tested on Python 3.12.

### Dependencies

All libraries are from the standard EBA 3650 course toolkit. No external packages are needed beyond these:

```
numpy >= 2.0
scipy >= 1.11
matplotlib >= 3.8
pandas >= 2.1
```

Install all dependencies with:

```bash
pip install numpy scipy matplotlib pandas
```

Or using the standard course environment:

```bash
pip install -r requirements.txt
```

### NumPy Compatibility Note

The notebook uses `np.trapezoid` (correct for NumPy 2.x). If you are running NumPy 1.x, replace all occurrences of `np.trapezoid` with `np.trapz`. You can check your version with:

```bash
python -c "import numpy as np; print(np.__version__)"
```

---

## How to Run

1. Clone or download this repository to your local machine.

2. Open a terminal in the project folder and launch Jupyter:

```bash
jupyter notebook main.ipynb
```

3. Run all cells from top to bottom using **Kernel > Restart & Run All**.

4. Results are printed inline. All six figures are saved to the `figures/` folder automatically.

To change any model parameter (tax rate, targeting multiplier, sample size), edit the `params` dictionary in **Section 1** only — all downstream cells read from it.

---

## Key Results (Baseline Parameters)

The following results are produced at baseline (τ = 0.10, λ = 1.50, N = 1,000, Gini = 0.35):

- **Gini of simulated wages:** ~0.35 (calibrated to target)
- **Scheme A equilibrium transfer:** T = 0.0126 per household
- **Scheme B transfers:** T_low = 0.0189 (bottom 40%), T_high = 0.0084 (top 60%)
- **Newton's method check:** matches fixed-point result to within 1.2 × 10⁻¹⁰

**Aggregate labour supply:**

| Scenario | Labour Supply | Change vs Baseline |
|----------|--------------|-------------------|
| Baseline (no redistribution) | 669.97 | — |
| Scheme A (lump-sum) | 664.93 | −5.04 |
| Scheme B (targeted) | 663.85 | −6.12 |

### Qualitative Findings

1. **Without redistribution**, the carbon tax is regressive: the burden falls disproportionately on lower-income deciles who spend a larger share of income on energy.

2. **Scheme A (universal rebate)** reduces regressivity because all households receive the same nominal rebate while lower-income households paid less tax, improving the net position of lower deciles.

3. **Scheme B (targeted compensation)** produces stronger redistribution for the bottom 40%: welfare gains are higher and the share of winners in lower deciles is larger. However, middle-income deciles may experience a smaller net gain because they fund the higher payments to the bottom group.

4. **Efficiency cost:** Scheme B has a slightly larger negative effect on aggregate labour supply than Scheme A, consistent with the income effect from higher transfers reducing labour supply at the margin for lower-income households.

5. **Policy frontier (Figure 6):** Higher λ (more aggressive targeting) improves equity at a growing efficiency cost, with diminishing marginal equity gains. This mirrors the Ramsey optimal tax tradeoff studied in Week 9 of the course.

---

## Limitations

This project is a stylised analytical exercise within the scope of EBA 3650. The following limitations are acknowledged:

- **Static model only.** There are no dynamic savings decisions. Long-run effects on capital accumulation are outside the scope of a one-period model.
- **Exogenous wages.** A richer general equilibrium would allow carbon taxation to affect labour demand and the pre-tax wage distribution. This extension is beyond the course scope.
- **Uniform energy share.** Setting β constant across all households ignores the reality that rural or lower-income households may have higher energy needs due to heating and transport.
- **Binary targeting rule.** Scheme B uses a simple two-tier transfer (bottom 40% vs top 60%). A continuous income-dependent transfer function would be more realistic but would complicate the budget-balance iteration.
- **No energy price feedback.** The carbon tax rate τ is exogenous. In a full model, the tax would affect the producer price of energy and feed back into household decisions.

---

## References

Goulder, L. H., Hafstead, M. A. C., and Williams, R. C. (2016). General equilibrium impacts of a federal clean energy standard. *American Economic Journal: Economic Policy*, 8(2), 186–218.

Metcalf, G. E. (2019). On the economics of a carbon tax for the United States. *Brookings Papers on Economic Activity*, Spring 2019, 405–458.

Ohlendorf, N., Jakob, M., Minx, J. C., Schroeder, C., and Steckel, J. C. (2021). Distributional impacts of carbon pricing: A meta-analysis. *Environmental and Resource Economics*, 78(1), 1–42.

Fremstad, A., and Paul, M. (2019). The impact of a carbon tax on inequality. *Ecological Economics*, 163, 88–97.

Ramsey, F. P. (1927). A contribution to the theory of taxation. *The Economic Journal*, 37(145), 47–61.

McKinney, W. (2022). *Python for Data Analysis: Data Wrangling with pandas, NumPy, and Jupyter* (3rd ed.). O'Reilly Media.

---

## Course Alignment

This project directly covers techniques from EBA 3650 Weeks 2 through 9:

- **Weeks 2–3:** Fixed-point iteration and Newton's method for solving the government budget equilibrium
- **Week 4:** Consumer theory and utility maximisation via the analytical FOC and `scipy.optimize`
- **Week 5:** Simulation of income distributions with `np.random.lognormal`; endogenous labour supply
- **Week 6:** Numerical integration (`np.trapezoid`) for the Lorenz curve and Gini coefficient
- **Weeks 7–8:** General equilibrium framework; welfare analysis by income decile
- **Week 9:** Optimal taxation and the Ramsey equity-efficiency tradeoff

---

*EBA 3650 Quantitative Economics | Project Period: March 27 – April 27, 2026*

