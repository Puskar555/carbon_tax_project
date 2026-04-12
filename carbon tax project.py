import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

np.random.seed(42)

# PARAMETERS
N = 1000
alpha = 3.0
eta = 2.0

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

# UTILITY
def utility(l, w, T, gamma, p_x):
    if l <= 0 or l >= 1:
        return -1e12
    
    m = w * l + T
    if m <= 0:
        return -1e12
    
    x = gamma * m / p_x
    y = (1 - gamma) * m

    return (
        gamma * np.log(x)
        + (1 - gamma) * np.log(y)
        - alpha * (l ** (1 + eta)) / (1 + eta)
    )

# HOUSEHOLD SOLVER
def solve_households(df, T_array, w_eq, p_x):
    l_list, x_list, y_list = [], [], []

    for i in range(len(df)):
        w_i = w_eq * df.loc[i, "w"]
        gamma = df.loc[i, "gamma"]
        T = T_array[i]

        res = minimize_scalar(
            lambda l: -utility(l, w_i, T, gamma, p_x),
            bounds=(1e-6, 0.95),
            method="bounded"
        )

        l_star = res.x
        m = w_i * l_star + T

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

# FIRM SIDE
def labor_demand(w):
    return 0.8 / (w + 0.5)

# GE SOLVER
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

        else:  # "none"
            T_new = np.zeros(len(df))

        if np.abs(w_new - w_eq) < tol and np.max(np.abs(T_new - T_array)) < tol:
            break

        w_eq = w_new
        T_array = T_new.copy()

    df_hh["T"] = T_array
    df_hh.attrs["w_eq"] = w_eq
    return df_hh

import matplotlib.pyplot as plt

# =========================================
# GRAPH: LABOR SUPPLY AS A FUNCTION OF NET WAGE
# =========================================

# Net wage grid
w_net = np.linspace(0.1, 1.2, 300)

# Simple upward-sloping labor supply function
# You can adjust the shape if you want it flatter or steeper
def labor_supply(w_net):
    return 0.15 + 0.85 * np.sqrt(w_net)

l_supply = labor_supply(w_net)

# Optional: mark your current tau = 0.20 example
tau = 0.20
w_gross_example = 1.0
w_net_example = (1 - tau) * w_gross_example
l_example = labor_supply(w_net_example)

# Plot
plt.figure(figsize=(7, 4.5))
plt.plot(w_net, l_supply, linewidth=2, label="Labor supply")

plt.scatter(w_net_example, l_example, zorder=5)
plt.annotate(
    f"Example: τ={tau:.2f}\nnet wage={(w_net_example):.2f}",
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

# Welfare relative to TAX-ONLY baseline
df_lump["dU"] = df_lump["U"] - df_tax_only["U"]
df_target["dU"] = df_target["U"] - df_tax_only["U"]


# =========================================
# GRAPH: CARBON TAX BURDEN BY DECILE
# =========================================
# burden = tax paid / labor income, using tax-only scenario
burden = df_tax_only.groupby("decile").apply(
    lambda g: (tau * g["x"]).sum() /
              ((df_tax_only.attrs["w_eq"] * g["w"] * g["l"]).sum())
)

plt.figure(figsize=(7, 4))
plt.bar(burden.index + 1, burden.values)
plt.title("Carbon Tax Burden by Decile")
plt.xlabel("Decile")
plt.ylabel("Tax Burden / Labor Income")
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
plt.show()