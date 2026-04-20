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
# UTILITY: new function from the board
# U = gamma log(x) + (1-gamma) log(y) + alpha log(1-l)
# =========================================
def utility(l, w, T, gamma, p_x):
    if l <= 0 or l >= 1:
        return -1e12

    m = w * l + T
    if m <= 0:
        return -1e12

    x = gamma * m / p_x
    y = (1 - gamma) * m

    if x <= 0 or y <= 0:
        return -1e12

    return (
        gamma * np.log(x)
        + (1 - gamma) * np.log(y)
        + alpha * np.log(1 - l)
    )


# =========================================
# CLOSED-FORM HOUSEHOLD SOLVER
# FOC:
#   w/(w l + T) = alpha/(1-l)
# =>
#   l* = (w - alpha T) / (w(1+alpha))
#
# Then:
#   m = w l + T
#   x = gamma m / p_x
#   y = (1-gamma) m
# =========================================
def solve_households(df, T_array, w_eq, p_x):
    l_list, x_list, y_list = [], [], []

    for i in range(len(df)):
        w_i = w_eq * df.loc[i, "w"]
        gamma = df.loc[i, "gamma"]
        T = T_array[i]

        # closed-form labor supply
        l_star = (w_i - alpha * T) / (w_i * (1 + alpha))

        # keep l inside feasible interval
        l_star = np.clip(l_star, 1e-6, 0.95)

        m = w_i * l_star + T

        # extra safety
        if m <= 1e-12:
            m = 1e-12

        x = gamma * m / p_x
        y = (1 - gamma) * m

        l_list.append(l_star)
        x_list.append(x)
        y_list.append(y)

    df_out = df.copy()
    df_out["l"] = l_list
    df_out["x"] = x_list
    df_out["y"] = y_list

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
# GRAPH: LABOR SUPPLY AS A FUNCTION OF NET WAGE
# model-consistent version
#
# If T = 0:
#   l*(w_net) = 1/(1+alpha)   --> flat line
#
# If you want a non-flat line, you need T > 0 or a different utility.
# =========================================
w_net = np.linspace(0.1, 1.2, 300)
l_supply = np.ones_like(w_net) * (1 / (1 + alpha))

w_net_example = 1.0
l_example = 1 / (1 + alpha)

plt.figure(figsize=(7, 4.5))
plt.plot(w_net, l_supply, linewidth=2, label="Labor supply")
plt.scatter(w_net_example, l_example, zorder=5)
plt.annotate(
    f"l* = 1/(1+alpha) = {l_example:.2f}",
    (w_net_example, l_example),
    xytext=(10, 10),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", lw=0.8)
)

plt.xlabel("Net wage")
plt.ylabel("Labor supply")
plt.title("Labor Supply and Net Wage")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================
# HELPER: consistent utility evaluation
# =========================================
def add_utility(df, tau_used):
    df = df.copy()
    w_eq = df.attrs["w_eq"]
    p_x = 1 + tau_used

    U_list = []
    for i in range(len(df)):
        w_i = w_eq * df.loc[i, "w"]
        l = df.loc[i, "l"]
        T = df.loc[i, "T"]
        gamma = df.loc[i, "gamma"]

        U = utility(l, w_i, T, gamma, p_x)
        U_list.append(U)

    df["U"] = U_list
    return df


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

    # emissions
    emissions_lump.append(df_l["x"].sum())
    emissions_target.append(df_t["x"].sum())

    # bottom 40%
    bottom = df0["w"] <= cutoff

    equity_lump.append((df_l.loc[bottom, "U"] - df_tax.loc[bottom, "U"]).mean())
    equity_target.append((df_t.loc[bottom, "U"] - df_tax.loc[bottom, "U"]).mean())

plt.figure(figsize=(7,4))
plt.plot(emissions_lump, equity_lump, marker="o", linewidth=2, label="Lump-sum")
plt.plot(emissions_target, equity_target, marker="o", linewidth=2, label="Targeted")

for i, tau_val in enumerate(tau_grid):
    plt.annotate(f"{tau_val:.1f}", (emissions_lump[i], equity_lump[i]), xytext=(4,4), textcoords="offset points", fontsize=8)
    plt.annotate(f"{tau_val:.1f}", (emissions_target[i], equity_target[i]), xytext=(4,4), textcoords="offset points", fontsize=8)

plt.xlabel("Emissions (sum of x)")
plt.ylabel("Bottom 40% Welfare Gain")
plt.title("Equity–Emissions Trade-off")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================
# CHAGNE IN CLEAN GOOD PRODUCTION
# BIN INTO 50 GROUPS
# =========================================
bins = 50

df_plot = pd.DataFrame({
    "rank": df_tax_only["rank"],
    "dy_lump": dy_lump,
    "dy_target": dy_target
})

df_plot["bin"] = pd.qcut(df_plot["rank"], bins, labels=False)

grouped = df_plot.groupby("bin").mean()

plt.figure(figsize=(7,4))
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

plt.figure(figsize=(7,4))
plt.plot(tau_grid, eff_lump, marker="o", label="Lump-sum")
plt.plot(tau_grid, eff_target, marker="o", label="Targeted")

plt.axhline(0, linestyle="--")

plt.xlabel("Carbon Tax")
plt.ylabel("Average Welfare Change")
plt.title("Efficiency Cost of Recycling Schemes")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

## GRAPH LABOR MARKET DEMAND AND SUPPLY

w_grid = np.linspace(0.1, 2.0, 100)

L_d = 0.8 / (w_grid + 0.5)  # your demand

# supply from model (constant)
L_s = np.ones_like(w_grid) * (1 / (1 + alpha))

plt.figure(figsize=(7,4))
plt.plot(w_grid, L_d, label="Labor demand")
plt.plot(w_grid, L_s, label="Labor supply")

plt.xlabel("Wage")
plt.ylabel("Labor")
plt.title("Labor Market Equilibrium")
plt.legend()
plt.grid(alpha=0.3)
plt.show()