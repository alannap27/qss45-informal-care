"""09_interactions.py

Takes in: data/processed/hrs2022_caregiving_networks.csv
Does: the three theories in the background section make different
predictions about interactions, not main effects, and nothing so far has
tested one. Andersen predicts need should dominate; Cantor predicts household
structure should dominate; Litwak predicts the two should interact: that
the effect of rising need on who provides care depends on whether there is
anyone in the household to absorb it.
Outputs: output/figures/f15_interaction_grid.png,
f16_shap_interactions.png, output/tables/t09_interactions.csv

For someone living with a partner, an extra ADL limitation can be absorbed by
the person already there. For someone living alone, the same limitation has to
be met by someone traveling in: a child, a neighbor, or a paid worker.
Litwak's task-specific model therefore predicts a steeper need gradient
among people who live alone.

Scripts 02–07 fit only main effects, so they cannot see this. Both a
formal interaction term and SHAP interaction values are used below, because
they can disagree, and the disagreement is helpful.

Run:  python3 code/09_interactions.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import xgboost as xgb
import shap
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from utils import (suptitle, paths, build_features, savefig, savetable, style_axis,
                   caption, use_paper_style, annotate_bars, panel_label,
                   RANDOM_STATE, BLUE, ORANGE, LIGHT, GREY)

use_paper_style()
P = paths()
df = pd.read_csv(P["processed"] / "hrs2022_caregiving_networks.csv",
                 dtype={"HHID": str, "PN": str})
y = df["informal_share"].values
groups = df["HHID"].values
print(f"n = {len(df):,}")
print(f"living alone: {df['lives_alone'].mean():.1%}   "
      f"with partner: {df['lives_with_partner'].mean():.1%}   "
      f"with relative: {df['lives_with_relative'].mean():.1%}")

# Functions

SAFE_NET = ["net_size", "net_hhi_hours", "net_top_helper_share"]

def stratified_gradient(frame, by, over, min_n=25):
    """Mean outcome across `over`, computed separately within each `by` group."""
    out = []
    for g, sub in frame.groupby(by):
        cell = sub.groupby(over)["informal_share"].agg(["mean", "size", "sem"])
        cell = cell[cell["size"] >= min_n]
        for lvl, r in cell.iterrows():
            out.append({by: g, over: lvl, "mean": r["mean"],
                        "n": int(r["size"]), "sem": r["sem"]})
    return pd.DataFrame(out)

def fit_interaction(frame, moderator, focal, controls):
    """OLS with a focal x moderator product term, standardized."""
    cols = [focal, moderator] + [c for c in controls if c not in (focal, moderator)]
    X = frame[cols].copy().fillna(frame[cols].median())
    sc = StandardScaler().fit(X)
    Xs = pd.DataFrame(sc.transform(X), columns=cols, index=X.index)
    Xs[f"{focal}_x_{moderator}"] = Xs[focal] * Xs[moderator]
    m = sm.OLS(frame["informal_share"].values, sm.add_constant(Xs)).fit(
        cov_type="cluster", cov_kwds={"groups": frame["HHID"]})
    term = f"{focal}_x_{moderator}"
    return {"focal": focal, "moderator": moderator,
            "interaction_beta": m.params[term], "p_value": m.pvalues[term],
            "ci_low": m.conf_int().loc[term, 0], "ci_high": m.conf_int().loc[term, 1],
            "n": int(m.nobs)}, m

# Descriptive

df["arrangement"] = np.select(
    [df["lives_alone"] == 1, df["lives_with_partner"] == 1, df["lives_with_relative"] == 1],
    ["Lives alone", "Lives with partner", "Lives with relative"], default="Other")
print(df["arrangement"].value_counts().to_string())
print()

grad = stratified_gradient(df[df["arrangement"] != "Other"], "arrangement", "adl_count")
print(grad.pivot(index="adl_count", columns="arrangement", values="mean").round(3).to_string())

# slope of the need gradient within each arrangement
print("\nlinear slope of informal share on ADL count, within arrangement:")
slopes = {}
for a, sub in df[df["arrangement"] != "Other"].groupby("arrangement"):
    m = sm.OLS(sub["informal_share"].values,
               sm.add_constant(sub["adl_count"].fillna(sub["adl_count"].median()).values)).fit()
    slopes[a] = (m.params[1], m.pvalues[1], len(sub))
    print(f"  {a:22} slope {m.params[1]:+.4f}  p = {m.pvalues[1]:.4f}  n = {len(sub):,}")

# Formal interaction tests
# Standard errors are clustered on household throughout, matching the CV design
# adopted in script 04.

CONTROLS = ["age", "female", "years_school", "iadl_count", "self_rated_health",
            "stroke", "dementia", "word_recall", "n_children", "total_care_hours"]

tests = []
for focal in ["adl_count", "iadl_count", "stroke", "dementia", "total_care_hours",
              "word_recall", "n_children"]:
    for moderator in ["lives_alone", "lives_with_partner"]:
        row, _ = fit_interaction(df, moderator, focal, CONTROLS)
        tests.append(row)
inter = pd.DataFrame(tests)
inter["significant_5pct"] = inter["p_value"] < 0.05
inter = inter.sort_values("p_value")
print(inter.round(4).to_string(index=False))
savetable(inter, "t09_interactions.csv")
print()
print(f"interactions significant at 5%: {int(inter['significant_5pct'].sum())} of {len(inter)}")
print("(with 14 tests, roughly 0.7 would be expected by chance alone)")

# Does adding interactions improve prediction?

X_main = build_features(df, extra=SAFE_NET)
X_int = X_main.copy()
for f in ["adl_count", "iadl_count", "total_care_hours", "n_children", "stroke"]:
    for m in ["lives_alone", "lives_with_partner"]:
        X_int[f"{f}_x_{m}"] = X_main[f] * X_main[m]

model = lambda: xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.02,
                                 min_child_weight=10, subsample=0.8,
                                 colsample_bytree=0.8, random_state=RANDOM_STATE)
cv = GroupKFold(5)
r2_main = cross_val_score(model(), X_main, y, cv=cv, groups=groups, scoring="r2", n_jobs=-1)
r2_int = cross_val_score(model(), X_int, y, cv=cv, groups=groups, scoring="r2", n_jobs=-1)
print(f"main effects only        : R2 = {r2_main.mean():+.4f} (sd {r2_main.std():.4f})")
print(f"with explicit interactions: R2 = {r2_int.mean():+.4f} (sd {r2_int.std():.4f})")
print(f"difference: {r2_int.mean() - r2_main.mean():+.4f}")
print()
print("A tree can already represent interactions by stacking splits, so a gain")
print("here would mean the explicit terms help it find them, not that the tree")
print("was previously incapable of expressing them.")

# SHAP interaction values
# These make each prediction into main effects plus pairwise interaction
# terms, so they find interactions the tree used rather than only the
# ones specified in advance.

booster = model().fit(X_main, y)
expl = shap.TreeExplainer(booster)
sub_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_main), size=600, replace=False)
X_sub = X_main.iloc[sub_idx]
siv = expl.shap_interaction_values(X_sub)
print(f"interaction tensor: {siv.shape}")

names = list(X_main.columns)
inter_strength = np.abs(siv).mean(axis=0)
np.fill_diagonal(inter_strength, 0)
pairs = [(names[i], names[j], inter_strength[i, j])
         for i in range(len(names)) for j in range(i + 1, len(names))]
top_pairs = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "mean_abs_interaction"]) \
    .sort_values("mean_abs_interaction", ascending=False).head(12)
print()
print(top_pairs.round(5).to_string(index=False))
savetable(top_pairs, "t09_shap_interaction_pairs.csv")

alone_rank = top_pairs[(top_pairs["feature_a"] == "lives_alone") |
                       (top_pairs["feature_b"] == "lives_alone")]
print()
print(f"pairs involving lives_alone in the top 12: {len(alone_rank)}")

# Figure 15: the need gradient conditional on living arrangement

fig, axes = plt.subplots(1, 3, figsize=(16.4, 6.7),
                         gridspec_kw={"width_ratios": [1.05, 1.15, 1.0],
                                      "wspace": 0.62})

ax = axes[0]
COLS = {"Lives alone": ORANGE, "Lives with partner": BLUE, "Lives with relative": GREY}
for a, sub in grad.groupby("arrangement"):
    sub = sub.sort_values("adl_count")
    ax.errorbar(sub["adl_count"], sub["mean"], yerr=sub["sem"], marker="o",
                capsize=3, color=COLS[a], label=f"{a} (slope {slopes[a][0]:+.3f})", zorder=3)
ax.set_xlabel("Number of ADL limitations")
ax.set_ylabel("Mean informal care share")
ax.legend(loc="lower left", fontsize=7.5, ncol=1, framealpha=0.9,
          facecolor="white", edgecolor="none")
ax.set_ylim(0.58, 1.03)
ax.set_title("Need gradient by living arrangement")
panel_label(ax, "A")
style_axis(ax)

ax = axes[1]
top = inter.head(10).iloc[::-1]
labels = [f"{r.focal}\n× {r.moderator}".replace("_", " ") for r in top.itertuples()]
ax.errorbar(top["interaction_beta"], range(len(top)),
            xerr=[top["interaction_beta"] - top["ci_low"],
                  top["ci_high"] - top["interaction_beta"]],
            fmt="o", capsize=3, markersize=5,
            color=BLUE, ecolor=GREY, zorder=3)
for i, sig in enumerate(top["significant_5pct"]):
    if sig:
        ax.scatter([top["interaction_beta"].iloc[i]], [i], s=90, color=ORANGE, zorder=4)
ax.axvline(0, ls="--", color="#333333", lw=1)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(labels, fontsize=7.5, linespacing=1.25)
ax.set_xlabel("Interaction coefficient (95% CI, household-clustered)")
ax.set_title("Formal interaction tests")
panel_label(ax, "B")
style_axis(ax, axis="x")

ax = axes[2]
tp = top_pairs.head(8).iloc[::-1]
lab = [f"{a}\n× {b}".replace("_", " ") for a, b in zip(tp["feature_a"], tp["feature_b"])]
bars = ax.barh(range(len(tp)), tp["mean_abs_interaction"],
               color=[ORANGE if "lives alone" in l else LIGHT for l in lab], zorder=3)
ax.set_yticks(range(len(tp)))
ax.set_yticklabels(lab, fontsize=7.5, linespacing=1.25)
ax.set_xlabel("Mean |SHAP interaction|")
ax.set_title("Interactions the tree actually used")
panel_label(ax, "C")
style_axis(ax, axis="x")

suptitle(fig, "Figure 15. Testing Litwak's prediction that need and household structure interact")
caption(fig, f"HRS 2022, n = {len(df):,}. Panel A: cells with fewer than 25 respondents are suppressed; error bars are standard errors of the mean; the slope in each legend entry is from a within-group regression of the informal\n"
             "share on ADL count. Panel B: coefficients on the product of a standardized focal variable and a standardized living-arrangement indicator, with standard errors clustered on household; orange marks p < 0.05.\n"
             "Panel C: mean absolute SHAP interaction values from the tuned booster, computed on a random 600-respondent subsample; these find interactions the model used rather than only those specified in advance.")
savefig(fig, "f15_interaction_grid.png")

# Figure 16: the strongest interaction

best = inter.iloc[0]
f_, m_ = best["focal"], best["moderator"]
fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.3))

ax = axes[0]
q = pd.qcut(df[f_].fillna(df[f_].median()), 4, labels=False, duplicates="drop")
tmp = df.assign(_q=q)
for val, lab, c in [(0, f"{m_.replace('_',' ')} = no", BLUE), (1, f"{m_.replace('_',' ')} = yes", ORANGE)]:
    sub = tmp[tmp[m_] == val].groupby("_q")["informal_share"].agg(["mean", "sem", "size"])
    sub = sub[sub["size"] >= 20]
    ax.errorbar(sub.index, sub["mean"], yerr=sub["sem"], marker="o", capsize=3,
                color=c, label=lab, zorder=3)
ax.set_xticks(range(4))
ax.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"])
ax.set_xlabel(f"Quartile of {f_.replace('_',' ')}")
ax.set_ylabel("Mean informal care share")
ax.legend()
ax.set_title(f"Observed: {f_.replace('_',' ')} by {m_.replace('_',' ')}")
panel_label(ax, "A")
style_axis(ax)

ax = axes[1]
i_f = names.index(f_) if f_ in names else 0
i_m = names.index(m_) if m_ in names else 1
ax.scatter(X_sub[f_], siv[:, i_f, i_m] * 2, s=18, alpha=0.55,
           c=[ORANGE if v == 1 else BLUE for v in X_sub[m_]], zorder=3)
ax.axhline(0, ls="--", color="#333333", lw=1)
ax.set_xlabel(f_.replace("_", " "))
ax.set_ylabel("SHAP interaction value")
handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label=f"{m_.replace('_',' ')} = no"),
           plt.Line2D([], [], marker="o", ls="", color=ORANGE, label=f"{m_.replace('_',' ')} = yes")]
ax.legend(handles=handles)
ax.set_title("Model's view of the same interaction")
panel_label(ax, "B")
style_axis(ax)

suptitle(fig, f"Figure 16. Strongest interaction: {f_.replace('_',' ')} × {m_.replace('_',' ')} "
             f"(β = {best['interaction_beta']:+.3f}, p = {best['p_value']:.3f})")
caption(fig, f"HRS 2022, n = {len(df):,}. Panel A splits the sample by {m_.replace('_',' ')} and plots the mean informal share across quartiles of {f_.replace('_',' ')}; non-parallel lines are the interaction. Error bars are standard errors;\n"
             "quartile cells with fewer than 20 respondents are suppressed. Panel B shows the SHAP interaction values for the same pair from the tuned booster, doubled because SHAP splits each pairwise interaction\n"
             "symmetrically between the two features. Agreement between a pre-specified test and a model-derived decomposition is weak evidence that the interaction is real rather than an artifact of either method.")
savefig(fig, "f16_shap_interactions.png")
