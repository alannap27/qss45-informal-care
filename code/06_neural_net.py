"""06_neural_net.py

Takes in: data/processed/hrs2022_caregiving.csv
Does: fits a multilayer perceptron against the linear and gradient-boosted
baselines under the household-clustered CV design adopted in script 04, and
checks whether the extra capacity buys anything.
Outputs: output/figures/f10_neural_net.png,
output/tables/t06_architecture_search.csv, t06_model_family_comparison.csv

Gradient-boosted trees are the strong default on small, heterogeneous, 
mostly-tabular data, and this sample has 2,142 rows, 33 features, 
and an outcome with 85% of its mass on one value.
Neural networks work best on high-dimensional structured inputs, text,
images, sequences, none of which is present here.

A comparison is only informative if it could have come out either way, 
and reporting the result against a stated expectation is what separates 
a test from a demonstration.

Run:  python3 code/06_neural_net.py
"""

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import r2_score

from utils import (suptitle, paths, build_features, savefig, savetable, style_axis,
                   caption, RANDOM_STATE, BLUE, ORANGE, LIGHT, GREY)

P = paths()
df = pd.read_csv(P["processed"] / "hrs2022_caregiving.csv", dtype={"HHID": str, "PN": str})
X = build_features(df)
y = df["informal_share"].values
groups = df["HHID"].values
cv = GroupKFold(5)
print(f"n = {len(df):,}, features = {X.shape[1]}, households = {df['HHID'].nunique():,}")

# Functions

def make_mlp(hidden, alpha=1e-3, lr=1e-3, max_iter=600):
    """Standardize then fit an MLP. Scaling is inside the pipeline so it is
    refitted within each CV fold and cannot leak across the split."""
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                     alpha=alpha, learning_rate_init=lr, max_iter=max_iter,
                     early_stopping=True, n_iter_no_change=25, validation_fraction=0.15,
                     random_state=RANDOM_STATE))

def score(model, label):
    t0 = time.time()
    s = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="r2", n_jobs=-1)
    print(f"  {label:38} R2 = {s.mean():+.4f}  (sd {s.std():.4f})  [{time.time()-t0:.1f}s]")
    return s.mean(), s.std()

# Architecture search
# Depth and width are varied over a small grid. Everything is scored with the
# same clustered CV, so the numbers are comparable with scripts 04 and 07.

ARCHS = [(8,), (32,), (64,), (32, 16), (64, 32), (128, 64), (64, 32, 16)]
rows = []
for h in ARCHS:
    m, s = score(make_mlp(h), f"MLP {h}")
    rows.append({"architecture": str(h), "n_hidden_layers": len(h),
                 "total_units": sum(h), "cv_r2": round(m, 4), "sd": round(s, 4)})
arch = pd.DataFrame(rows).sort_values("cv_r2", ascending=False)
print()
print(arch.to_string(index=False))
savetable(arch, "t06_architecture_search.csv")
best_arch = eval(arch.iloc[0]["architecture"])
print(f"\nbest architecture: {best_arch}")

# Regularization sweep on the best architecture

rows = []
for a in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
    m, s = score(make_mlp(best_arch, alpha=a), f"MLP {best_arch} alpha={a:g}")
    rows.append({"alpha": a, "cv_r2": round(m, 4), "sd": round(s, 4)})
reg = pd.DataFrame(rows)
print()
print(reg.to_string(index=False))
best_alpha = float(reg.loc[reg["cv_r2"].idxmax(), "alpha"])
print(f"best alpha: {best_alpha:g}")

# Model family comparison

families = {
    "Ridge (linear)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    f"MLP {best_arch}, alpha={best_alpha:g}": make_mlp(best_arch, alpha=best_alpha),
    "XGBoost (tuned in script 02)": xgb.XGBRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.02, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE),
}
print("household-clustered 5-fold CV:")
res = []
for label, mdl in families.items():
    m, s = score(mdl, label)
    res.append({"model_family": label, "cv_r2": round(m, 4), "sd_across_folds": round(s, 4)})

baseline = cross_val_score(
    make_pipeline(StandardScaler(), Ridge(alpha=1e12)), X, y,
    cv=cv, groups=groups, scoring="r2", n_jobs=-1)
res.insert(0, {"model_family": "Intercept only", "cv_r2": round(baseline.mean(), 4),
               "sd_across_folds": round(baseline.std(), 4)})

comp = pd.DataFrame(res).sort_values("cv_r2", ascending=False)
print()
print(comp.to_string(index=False))
savetable(comp, "t06_model_family_comparison.csv")

# Figure 10

fig, axes = plt.subplots(1, 3, figsize=(16.4, 6.7),
                         gridspec_kw={"wspace": 0.40,
                                      "width_ratios": [1.15, 1.0, 0.95]})

ax = axes[0]
a = arch.sort_values("cv_r2")
ax.barh(a["architecture"], a["cv_r2"],
        color=[ORANGE if i == len(a) - 1 else LIGHT for i in range(len(a))], zorder=3)
ax.axvline(0, color="black", lw=1)
for i, v in enumerate(a["cv_r2"]):
    ax.text(v + 0.001, i, f"{v:+.3f}", va="center", fontsize=10)
ax.set_xlabel("Clustered CV $R^2$")
ax.set_ylabel("Hidden layer sizes")
ax.set_title("A. Architecture search\nmore capacity does not help", fontsize=12)
style_axis(ax, axis="x")

ax = axes[1]
ax.semilogx(reg["alpha"], reg["cv_r2"], marker="o", color=BLUE, lw=2, zorder=3)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel(r"L2 penalty $\alpha$ (log scale)")
ax.set_ylabel("Clustered CV $R^2$")
ax.set_title("B. Heavier regularization helps,\nwhich says the net is overfitting", fontsize=12)
style_axis(ax)

ax = axes[2]
c = comp.sort_values("cv_r2")
cols = []
for name in c["model_family"]:
    cols.append(ORANGE if name.startswith("MLP") else (GREY if "Intercept" in name else BLUE))
ax.barh(range(len(c)), c["cv_r2"], xerr=c["sd_across_folds"], capsize=4,
        color=cols, zorder=3)
ax.set_yticks(range(len(c)))
SHORT_NAME = {"XGBoost (tuned in script 02)": "XGBoost",
              "Ridge (linear)": "Ridge",
              "Intercept only": "Intercept only",
              "MLP (8,), alpha=1": "MLP (8,)"}
ax.set_yticklabels([SHORT_NAME.get(n, n) for n in c["model_family"]], fontsize=10.5)
ax.axvline(0, color="black", lw=1)
for i, (v, e) in enumerate(zip(c["cv_r2"], c["sd_across_folds"])):
    ax.text(v + e + 0.007, i, f"{v:+.3f}", va="center", fontsize=10)
ax.set_xlim(c["cv_r2"].min() - c["sd_across_folds"].max() - 0.02,
            c["cv_r2"].max() + c["sd_across_folds"].max() + 0.045)
ax.set_xlabel("Clustered CV $R^2$")
ax.set_title("C. The MLP does not beat the booster", fontsize=12)
style_axis(ax, axis="x")

suptitle(fig, "A neural network buys nothing on 2,142 rows of tabular survey data")
caption(fig, f"HRS 2022, n = {len(df):,}, {X.shape[1]} features, scored by 5-fold GroupKFold on household so no household spans the train/test boundary. Panels A and B search over architecture and L2 penalty; the flat, "
             "noisy\nresponse to added capacity in Panel A and the improvement under heavier penalty in Panel B are both symptoms of a model with far more parameters than the data can identify. Error bars in Panel C are\n"
             "the standard deviation across the five folds and are wide enough to overlap, so the ordering between the top families should be read as 'indistinguishable', not as a ranking.")
savefig(fig, "f10_neural_net.png")

# Verdict:
# The MLP is not better than gradient boosting here; the fold-to-fold
# spread is wide enough that most of the ordering is noise,
# and the regularization sweep shows the network overfitting rather 
# than finding structure the tree missed.
