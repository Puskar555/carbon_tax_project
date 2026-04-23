import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# PARAMETERS
N = 1000
alpha = 3.0   # leisure preference

gamma_min = 0.20
gamma_max = 0.65

tau = 0.20
lambda_target = 1.75

phi = 1.0

tol = 1e-6
max_iter = 200

# HOUSEHOLDS
wages = np.random.lognormal(0, 0.3, N)
wages = wages / np.mean(wages)

df0 = pd.DataFrame({"w": wages})
df0 = df0.sort_values("w").reset_index(drop=True)
df0["rank"] = np.linspace(0, 1, N)

gamma_base = gamma_max - (gamma_max - gamma_min) * df0["rank"]
noise = np.random.normal(0, 0.02, N)
df0["gamma"] = np.clip(gamma_base + noise, 0.05, 0.95)


# =========================================
# HOUSEHOLD PROBLEM IN REDUCED FORM
#
# Original problem:
#   max_{x,y,l} gamma log(x) + (1-gamma) log(y) + alpha log(1-l)
#   s.t. p_x x + y = w l + T
#
# Reduced problem:
#   max_{x,l} gamma log(x) + (1-gamma) log(wl + T - p_x x) + alpha log(1-l)
#
# Conditional on l, the x-FOC gives:
#   x*(l) = gamma (w l + T) / p_x
#   y*(l) = (1-gamma) (w l + T)
#
# Then we solve over l.
# =========================================
def utility_from_choices(x, l, w, T, gamma, p_x):
    if l <= 0 or l >= 1:
        return -1e12

    y = w * l + T - p_x * x
    if x <= 0 or y <= 0:
        return -1e12

    return (
        gamma * np.log(x)
        + (1 - gamma) * np.log(y)
        + alpha * np.log(1 - l)
    )


def x_star_given_l(l, w, T, gamma, p_x):
    m = w * l + T
    if m <= 0:
        return 1e-12
    return gamma * m / p_x


def y_star_given_l(l, w, T, gamma):
    m = w * l + T
    if m <= 0:
        return 1e-12
    return (1 - gamma) * m


def reduced_utility(l, w, T, gamma, p_x):
    if l <= 0 or l >= 1:
        return -1e12

    x = x_star_given_l(l, w, T, gamma, p_x)
    y = w * l + T - p_x * x

    if x <= 0 or y <= 0:
        return -1e12

    return (
        gamma * np.log(x)
        + (1 - gamma) * np.log(y)
        + alpha * np.log(1 - l)
    )


# =========================================
# SOLVE HOUSEHOLDS
#
# Since the reduced problem is 1D in l after substituting x*(l),
# we can still use the implied closed form:
#
#   dU/dl = 1/(w l + T) * w - alpha/(1-l) = 0
#   => w/(w l + T) = alpha/(1-l)
#   => l* = (w - alpha T) / (w(1+alpha))
#
# This is the solution IMPLIED by the reduced problem.
# =========================================
def solve_households(df, T_array, w_eq, p_x):
    l_list, x_list, y_list, U_list = [], [], [], []

    for i in range(len(df)):
        w_i = w_eq * df.loc[i, "w"]
        gamma_i = df.loc[i, "gamma"]
        T_i = T_array[i]

        # labor choice from reduced problem
        l_star = (w_i - alpha * T_i) / (w_i * (1 + alpha))
        l_star = np.clip(l_star, 1e-6, 0.95)

        # consumption choices from x-FOC and budget
        x_star = x_star_given_l(l_star, w_i, T_i, gamma_i, p_x)
        y_star = w_i * l_star + T_i - p_x * x_star

        # safety
        x_star = max(x_star, 1e-12)
        y_star = max(y_star, 1e-12)

        U_star = utility_from_choices(x_star, l_star, w_i, T_i, gamma_i, p_x)

        l_list.append(l_star)
        x_list.append(x_star)
        y_list.append(y_star)
        U_list.append(U_star)

    df_out = df.copy()
    df_out["l"] = l_list
    df_out["x"] = x_list
    df_out["y"] = y_list
    df_out["U"] = U_list

    return df_out


# =========================================
# FIRM SIDE
# =========================================
def labor_demand(w):
    return 0.8 / (w + 0.5)


# =========================================
# GE SOLVER
# =========================================
def solve_GE(df, scheme, tau_val):
    p_x = 1 + tau_val
    T_array = np.zeros(len(df))
    w_eq = 1.0

    for _ in range(max_iter):
        df_hh = solve_households(df, T_array, w_eq, p_x)

        L_supply = df_hh["l"].mean()
        L_demand = labor_demand(w_eq)

        excess = L_supply - L_demand
        w_new = max(0.05, w_eq - 0.2 * excess)

        X = df_hh["x"].sum()
        R = tau_val * phi * X

        if scheme == "lump":
            T_new = np.ones(len(df)) * (R / len(df))

        elif scheme == "target":
            T_bar = R / len(df)
            T_new = T_bar * (1 + lambda_target * (1 - df["rank"]))

            total = T_new.sum()
            if total > 1e-10:
                T_new *= R / total
            else:
                T_new = np.zeros(len(df))

        else:   # "none"
            T_new = np.zeros(len(df))

        if np.abs(w_new - w_eq) < tol and np.max(np.abs(T_new - T_array)) < tol:
            w_eq = w_new
            T_array = T_new.copy()
            break

        w_eq = w_new
        T_array = T_new.copy()

    # final household solution using converged values
    df_hh = solve_households(df, T_array, w_eq, p_x)
    df_hh["T"] = T_array
    df_hh.attrs["w_eq"] = w_eq
    return df_hh


# =========================================
# HELPER: consistent utility evaluation
# =========================================
def add_utility(df, tau_used):
    # utility already computed in solve_households,
    # but we keep this helper so the rest of your script still works
    return df.copy()


# =========================================
# HELPER: add deciles
# =========================================
def add_deciles(df):
    df = df.copy()
    df["decile"] = pd.qcut(df["w"], 10, labels=False)
    return df


# =========================================
# RUN SCENARIOS FOR CURRENT TAU
# =========================================
df_tax_only = solve_GE(df0.copy(), "none", tau)
df_lump = solve_GE(df0.copy(), "lump", tau)
df_target = solve_GE(df0.copy(), "target", tau)

df_tax_only = add_utility(df_tax_only, tau)
df_lump = add_utility(df_lump, tau)
df_target = add_utility(df_target, tau)

df_tax_only = add_deciles(df_tax_only)
df_lump = add_deciles(df_lump)
df_target = add_deciles(df_target)

# Baseline: no carbon tax
df_notax = solve_GE(df0.copy(), "none", 0.0)
df_notax = add_utility(df_notax, 0.0)
df_notax = add_deciles(df_notax)

# Welfare relative to NO-TAX baseline
df_lump["dU"] = df_lump["U"] - df_notax["U"]
df_target["dU"] = df_target["U"] - df_notax["U"]

# =========================================
# FAST GRAPH: EQUITY–EMISSIONS TRADE-OFF
# =========================================
tau_grid = np.array([0.00, 0.10, 0.20, 0.30, 0.40])

emissions_lump = []
emissions_target = []
equity_lump = []
equity_target = []

cutoff = df0["w"].quantile(0.4)

for tau_val in tau_grid:
    print(f"Running tau = {tau_val:.2f}")

    df_tax = solve_GE(df0.copy(), "none", tau_val)
    df_l = solve_GE(df0.copy(), "lump", tau_val)
    df_t = solve_GE(df0.copy(), "target", tau_val)

    df_tax = add_utility(df_tax, tau_val)
    df_l = add_utility(df_l, tau_val)
    df_t = add_utility(df_t, tau_val)

    emissions_lump.append(df_l["x"].sum())
    emissions_target.append(df_t["x"].sum())

    bottom = df0["w"] <= cutoff
    equity_lump.append((df_l.loc[bottom, "U"] - df_tax.loc[bottom, "U"]).mean())
    equity_target.append((df_t.loc[bottom, "U"] - df_tax.loc[bottom, "U"]).mean())

plt.figure(figsize=(7, 4))
plt.plot(emissions_lump, equity_lump, marker="o", linewidth=2, label="Lump-sum")
plt.plot(emissions_target, equity_target, marker="o", linewidth=2, label="Targeted")

for i, tau_val in enumerate(tau_grid):
    plt.annotate(f"{tau_val:.1f}", (emissions_lump[i], equity_lump[i]),
                 xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.annotate(f"{tau_val:.1f}", (emissions_target[i], equity_target[i]),
                 xytext=(4, 4), textcoords="offset points", fontsize=8)

plt.xlabel("Emissions (sum of x)")
plt.ylabel("Bottom 40% Welfare Gain")
plt.title("Equity–Emissions Trade-off")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================
# CHANGE IN CLEAN GOOD CONSUMPTION
# BIN INTO 50 GROUPS
# relative to no-tax baseline
# =========================================
bins = 50

dy_lump = df_lump["y"] - df_notax["y"]
dy_target = df_target["y"] - df_notax["y"]

df_plot = pd.DataFrame({
    "rank": df_tax_only["rank"],
    "dy_lump": dy_lump,
    "dy_target": dy_target
})

df_plot["bin"] = pd.qcut(df_plot["rank"], bins, labels=False)
grouped = df_plot.groupby("bin").mean()

plt.figure(figsize=(7, 4))
plt.plot(grouped["rank"], grouped["dy_lump"], marker="o", label="Lump-sum")
plt.plot(grouped["rank"], grouped["dy_target"], marker="o", label="Targeted")

plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Income Rank")
plt.ylabel("Change in clean-good consumption")
plt.title("Change in Clean-Good Consumption (Binned)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================
# GRAPH: EFFICIENCY COST
# =========================================
tau_grid = np.linspace(0.0, 0.4, 8)

eff_lump = []
eff_target = []

for tau_val in tau_grid:
    df_tax = solve_GE(df0.copy(), "none", tau_val)
    df_l = solve_GE(df0.copy(), "lump", tau_val)
    df_t = solve_GE(df0.copy(), "target", tau_val)

    df_tax = add_utility(df_tax, tau_val)
    df_l = add_utility(df_l, tau_val)
    df_t = add_utility(df_t, tau_val)

    eff_lump.append((df_l["U"] - df_tax["U"]).mean())
    eff_target.append((df_t["U"] - df_tax["U"]).mean())

plt.figure(figsize=(7, 4))
plt.plot(tau_grid, eff_lump, marker="o", label="Lump-sum")
plt.plot(tau_grid, eff_target, marker="o", label="Targeted")
plt.axhline(0, linestyle="--")

plt.xlabel("Carbon Tax")
plt.ylabel("Average Welfare Change")
plt.title("Efficiency Cost of Recycling Schemes")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================
# GRAPH: CARBON TAX BURDEN BY DECILE
# burden = tax paid / labor income, using tax-only scenario
# =========================================
burden = df_tax_only.groupby("decile").apply(
    lambda g: (tau * g["x"]).sum() /
              ((df_tax_only.attrs["w_eq"] * g["w"] * g["l"]).sum())
)

plt.figure(figsize=(7, 4))
plt.bar(burden.index + 1, burden.values)
plt.title("Carbon Tax Burden by Decile")
plt.xlabel("Decile")
plt.ylabel("Tax Burden / Labor Income")
plt.tight_layout()
plt.show()


# =========================================
# GRAPH: WELFARE CHANGE BY DECILE
# =========================================
welfare_lump = df_lump.groupby("decile")["dU"].mean()
welfare_target = df_target.groupby("decile")["dU"].mean()

plt.figure(figsize=(7, 4))
plt.plot(welfare_lump.index + 1, welfare_lump.values, marker="o", label="Lump-sum")
plt.plot(welfare_target.index + 1, welfare_target.values, marker="o", label="Targeted")
plt.axhline(0, linestyle="--")
plt.title("Welfare Change by Decile")
plt.xlabel("Decile")
plt.ylabel("Average Δ Utility")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================
# FULL CODE: ILLUSTRATIVE LABOR MARKET
# UNDER LUMP-SUM RECYCLING
# =========================================

# assumes these already exist from your model:
# - alpha
# - tau
# - df0
# - solve_GE()
# - labor_demand()

# -----------------------------------------
# Solve lump-sum scenario
# -----------------------------------------
df_lump = solve_GE(df0.copy(), "lump", tau)

# in lump-sum, everyone gets the same transfer
T_eq_lump = df_lump["T"].iloc[0]

print(f"T_eq_lump = {T_eq_lump:.4f}")

# -----------------------------------------
# Helper: labor supply curve
# infeasible values are set to NaN
# -----------------------------------------
def labor_supply_curve(w_grid, T, alpha):
    L = (w_grid - alpha * T) / (w_grid * (1 + alpha))
    L[(L <= 0) | (L >= 1)] = np.nan
    return L

# wage grid
w_grid = np.linspace(0.05, 3.5, 500)

# supply and demand
L_supply_lump = labor_supply_curve(w_grid, T_eq_lump, alpha)
L_demand = labor_demand(w_grid)

# approximate intersection
diff = np.abs(L_supply_lump - L_demand)
idx_star = np.nanargmin(diff)
w_star = w_grid[idx_star]
L_star = L_supply_lump[idx_star]

# -----------------------------------------
# Plot
# -----------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(w_grid, L_supply_lump, linewidth=2,
         label=f"Labor supply (lump-sum, T = {T_eq_lump:.3f})")
plt.plot(w_grid, L_demand, linewidth=2, label="Labor demand")
plt.scatter(w_star, L_star, s=70, zorder=5,
            label=f"Intersection ≈ ({w_star:.2f}, {L_star:.2f})")

plt.xlabel("Wage")
plt.ylabel("Labor")
plt.title("Illustrative Labor Market under Lump-Sum Recycling")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================
# ILLUSTRATIVE HOUSEHOLD LABOR SUPPLY
# UNDER TARGETED RECYCLING
# ========================================

# -----------------------------------------
# Solve targeted scenario
# -----------------------------------------
df_target = solve_GE(df0.copy(), "target", tau)

# pick representative households by income rank
idx_low = int(0.1 * len(df_target))
idx_mid = int(0.5 * len(df_target))
idx_high = int(0.9 * len(df_target))

T_low = df_target.loc[idx_low, "T"]
T_mid = df_target.loc[idx_mid, "T"]
T_high = df_target.loc[idx_high, "T"]

print(f"T_low  = {T_low:.4f}")
print(f"T_mid  = {T_mid:.4f}")
print(f"T_high = {T_high:.4f}")

# -----------------------------------------
# Helper: labor supply curve 
# infeasible values are set to NaN
# -----------------------------------------
def labor_supply_curve(w_grid, T, alpha):
    L = (w_grid - alpha * T) / (w_grid * (1 + alpha))
    L[(L <= 0) | (L >= 1)] = np.nan
    return L

# wage grid
w_grid = np.linspace(0.05, 3.5, 500)

# supply curves
L_low = labor_supply_curve(w_grid, T_low, alpha)
L_mid = labor_supply_curve(w_grid, T_mid, alpha)
L_high = labor_supply_curve(w_grid, T_high, alpha)

# demand curve
L_demand = labor_demand(w_grid)

# -----------------------------------------
# Plot
# -----------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(w_grid, L_low, linewidth=2, label=f"Low-income supply (T = {T_low:.3f})")
plt.plot(w_grid, L_mid, linewidth=2, label=f"Middle-income supply (T = {T_mid:.3f})")
plt.plot(w_grid, L_high, linewidth=2, label=f"High-income supply (T = {T_high:.3f})")
plt.plot(w_grid, L_demand, linewidth=2, label="Labor demand")

plt.xlabel("Wage")
plt.ylabel("Labor")
plt.title("Illustrative Household Labor Supply under Targeted Recycling")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
