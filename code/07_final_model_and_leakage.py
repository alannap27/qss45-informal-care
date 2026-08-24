"""07_final_model_and_leakage.py

Takes in: data/processed/hrs2022_caregiving_networks.csv (from script 05)
Does: First, documents a target-leakage trap in the network features 
and shows what it looks like when you fall into it. Second, runs a clean 
ablation to test whether care-network structure adds anything once the
existing covariates are present. Third, fits the final model and interprets it
with SHAP.
Outputs: output/figures/f11_leakage.png, f12_ablation_and_shap.png,
output/tables/t07_ablation.csv, t07_final_shap.csv

The trap:
Script 05 built net_any_formal: whether any helper in the network is a
paid or institutional provider; that feature is the outcome. The informal
share is below 1 when some hours come from a formal helper.

net_n_informal is a count of the numerator's contributors.

Run:  python3 code/07_final_model_and_leakage.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import r2_score

from utils import (suptitle, paths, build_features, savefig, savetable, style_axis,
                   caption, RANDOM_STATE, BLUE, ORANGE, LIGHT, GREY)

P = paths()
df = pd.read_csv(P["processed"] / "hrs2022_caregiving_networks.csv",
                 dtype={"HHID": str, "PN": str})
y = df["informal_share"].values
groups = df["HHID"].values
cv = GroupKFold(5)

# Tier 1: outright leakage. These are the outcomes, re-encoded.
LEAKY = ["net_any_formal", "net_n_informal"]

# Tier 2: partly endogenous. Each of these is 0 or undefined when the
# respondent has no informal helper at all, so each carries a piece of the
# outcome even after the tier-1 features are removed. Excluded from the final
# model; quantified below so the exclusion is justified.
ENDOGENOUS = ["net_has_spouse", "net_n_children", "net_n_kin_types",
              "net_generation_span"]

# Tier 3: describe the shape of the roster without encoding who paid.
SAFE_NET = ["net_size", "net_hhi_hours", "net_top_helper_share"]
print(f"n = {len(df):,}")
print("tier 1, leakage         :", LEAKY)
print("tier 2, endogenous      :", ENDOGENOUS)
print("tier 3, usable          :", SAFE_NET)

# Functions

def model():
    return xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.02,
                            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                            random_state=RANDOM_STATE)

def clustered_r2(X, label):
    s = cross_val_score(model(), X, y, cv=cv, groups=groups, scoring="r2", n_jobs=-1)
    print(f"  {label:44} R2 = {s.mean():+.4f}  (sd {s.std():.4f})")
    return s.mean(), s.std()

def corr_with_outcome(col):
    return float(np.corrcoef(df[col].fillna(df[col].median()), y)[0, 1])

# Springing the trap

for c in LEAKY:
    print(f"{c:20} correlation with outcome: {corr_with_outcome(c):+.3f}")
print()
X_base = build_features(df)
X_leaky = build_features(df, extra=LEAKY)

r2_base, sd_base = clustered_r2(X_base, "base covariates only")
r2_leaky, sd_leaky = clustered_r2(X_leaky, "base + LEAKY network features")
print()
print(f"apparent gain from leakage: {r2_leaky - r2_base:+.4f}")
print("A jump of this size on a near-degenerate outcome is not a discovery.")
print("It is the model reading the answer off the feature.")

# The kin-composition features are zero when nobody in the roster is family,
# which is the same event as the informal share being zero. 

for c in ENDOGENOUS:
    zero_rows = df[c] == 0
    if zero_rows.sum() > 0:
        print(f"{c:24} = 0 for {int(zero_rows.sum()):4d} respondents; "
              f"their mean informal share is {y[zero_rows.values].mean():.3f} "
              f"vs {y[~zero_rows.values].mean():.3f} otherwise")

X_endo = build_features(df, extra=SAFE_NET + ENDOGENOUS)
r2_endo, sd_endo = clustered_r2(X_endo, "base + tier 3 + tier 2 (endogenous)")

# Clean ablation:
# Tier 3 only: features that describe the shape of the roster without encoding
# who paid for it.

X_net = build_features(df, extra=SAFE_NET)
r2_net, sd_net = clustered_r2(X_net, "base + tier 3 network shape (FINAL)")

blocks = {
    "Demographics only": ["age", "female", "hispanic", "years_school"],
    "+ household structure": ["age", "female", "hispanic", "years_school",
                              "lives_alone", "lives_with_partner", "lives_with_relative",
                              "widowed", "never_married", "sep_divorced", "n_children",
                              "kids_within_10mi"],
    "+ clinical need": ["age", "female", "hispanic", "years_school",
                        "lives_alone", "lives_with_partner", "lives_with_relative",
                        "widowed", "never_married", "sep_divorced", "n_children",
                        "kids_within_10mi", "adl_count", "iadl_count",
                        "self_rated_health", "stroke", "dementia", "alzheimers",
                        "word_recall", "rate_memory"],
}
rows = []
for label, cols in blocks.items():
    keep = [c for c in cols if c in df.columns]
    Xb = build_features(df[keep + ["race"]].assign(**{c: df[c] for c in keep}))
    m, s = clustered_r2(Xb, label)
    rows.append({"feature_block": label, "n_features": Xb.shape[1],
                 "cv_r2": round(m, 4), "sd": round(s, 4)})

rows.append({"feature_block": "+ care intensity (full base set)", "n_features": X_base.shape[1],
             "cv_r2": round(r2_base, 4), "sd": round(sd_base, 4)})
rows.append({"feature_block": "+ network shape, tier 3 (FINAL)", "n_features": X_net.shape[1],
             "cv_r2": round(r2_net, 4), "sd": round(sd_net, 4)})
rows.append({"feature_block": "+ tier 2 endogenous (excluded)", "n_features": X_endo.shape[1],
             "cv_r2": round(r2_endo, 4), "sd": round(sd_endo, 4)})
rows.append({"feature_block": "+ tier 1 leakage (excluded)", "n_features": X_leaky.shape[1],
             "cv_r2": round(r2_leaky, 4), "sd": round(sd_leaky, 4)})
abl = pd.DataFrame(rows)
print()
print(abl.to_string(index=False))
savetable(abl, "t07_ablation.csv")

# Figure 11: what leakage looks like

fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.3))

ax = axes[0]
names = ["Base\ncovariates", "+ tier 3\nshape (final)", "+ tier 2\nendogenous", "+ tier 1\nleakage"]
vals = [r2_base, r2_net, r2_endo, r2_leaky]
errs = [sd_base, sd_net, sd_endo, sd_leaky]
cols = [BLUE, BLUE, "#dd9a4e", ORANGE]
ax.bar(names, vals, yerr=errs, capsize=6, color=cols, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + errs[i] + 0.012, f"{v:+.3f}", ha="center", fontsize=10, fontweight="bold")
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("Household-clustered CV $R^2$")
ax.set_title("A. Two tiers of contamination,\neach found by asking what a feature encodes", fontsize=12)
style_axis(ax)

ax = axes[1]
sub = df.groupby("net_any_formal")["informal_share"].agg(["mean", "min", "max", "size"])
ax.scatter(df["net_any_formal"] + np.random.default_rng(1).uniform(-0.06, 0.06, len(df)),
           df["informal_share"], s=8, alpha=0.18, color=BLUE, zorder=3)
ax.set_xticks([0, 1])
ax.set_xticklabels([f"No formal helper\n(n = {int(sub.loc[0,'size'])})",
                    f"Has a formal helper\n(n = {int(sub.loc[1,'size'])})"])
ax.set_ylabel("Informal share (the outcome)")
ax.set_title(f"B. net_any_formal is the outcome\nr = {corr_with_outcome('net_any_formal'):+.2f}", fontsize=12)
ax.annotate(f"all {int((df.loc[df['net_any_formal']==0,'informal_share']==1).sum()):,} of these\nsit at exactly 1.0",
            xy=(0, 1.0), xytext=(0.42, 0.72), fontsize=9, color=ORANGE, ha="center",
            arrowprops=dict(arrowstyle="->", lw=1.0, color=ORANGE))
ax.set_ylim(-0.06, 1.12)
style_axis(ax)

suptitle(fig, "A target-leakage trap, sprung deliberately and then removed")
caption(fig, f"HRS 2022, n = {len(df):,}, 5-fold GroupKFold on household. Panel A: error bars are the standard deviation across folds. The apparent improvement from the leaky features is {r2_leaky - r2_base:+.3f}, which on an outcome\n"
             "this degenerate would be an implausible finding rather than a good one. Panel B: the informal share falls below 1 exactly when a formal helper is present, so the feature and the outcome are two\n"
             "encodings of the same fact. Points are jittered horizontally for visibility. Neither leaky feature is used in the final model.")
savefig(fig, "f11_leakage.png")

# Final model and SHAP

final = model().fit(X_net, y)
explainer = shap.TreeExplainer(final)
sv = explainer.shap_values(X_net)
mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X_net.columns).sort_values(ascending=False)
print("top 15 features by mean |SHAP| in the final model:")
print(mean_abs.head(15).round(4).to_string())
savetable(mean_abs.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"}),
          "t07_final_shap.csv")

net_share = mean_abs[[c for c in mean_abs.index if c.startswith("net_")]].sum() / mean_abs.sum()
print()
print(f"share of total attribution going to network-structure features: {net_share:.1%}")

# Figure 12

# Composited for the same reason as Figure 4: shap.summary_plot resizes the
# active figure, which would push the title and caption out of the saved frame.
import figstyle

tmp_abl = paths()["figures"] / "_panel_ablation.png"
tmp_shap2 = paths()["figures"] / "_panel_final_shap.png"

fig_a, ax = plt.subplots(figsize=(7.4, 6.4))
a = abl[~abl["feature_block"].str.contains("excluded")]
ax.plot(range(len(a)), a["cv_r2"], marker="o", lw=2, color=BLUE, zorder=3)
ax.fill_between(range(len(a)), a["cv_r2"] - a["sd"], a["cv_r2"] + a["sd"],
                color=BLUE, alpha=0.15, zorder=2)
ax.set_xticks(range(len(a)))
ax.set_xticklabels(a["feature_block"], rotation=20, ha="right", fontsize=8.4)
ax.axhline(0, color="#333333", lw=1)
for i, (v, n) in enumerate(zip(a["cv_r2"], a["n_features"])):
    ax.annotate(f"{v:+.3f}\n({n}f)", (i, v), textcoords="offset points",
                xytext=(0, 13), ha="center", fontsize=8.2)
ax.set_ylabel("Household-clustered CV $R^2$")
style_axis(ax)
fig_a.savefig(tmp_abl, format="png")
plt.close(fig_a)

fig_b = plt.figure(figsize=(7.4, 6.4))
shap.summary_plot(sv, X_net, show=False, max_display=14, plot_size=None)
plt.gcf().savefig(tmp_shap2, format="png")
plt.close("all")

figstyle.compose(
    [tmp_abl, tmp_shap2],
    ["A.  Where the predictive power comes from", "B.  Final model attribution"],
    paths()["figures"] / "f12_ablation_and_shap.png",
    "Household structure carries the signal; network shape adds almost nothing",
    f"HRS 2022, n = {len(df):,}. Panel A adds feature blocks cumulatively under identical five-fold GroupKFold folds; the shaded band is one standard deviation across folds and is wide enough that the last two\n"
    f"steps overlap, so the apparent gain from network structure is not distinguishable from noise. Panel B: each dot is one respondent, horizontal position is that feature's contribution to their prediction, and color\n"
    f"is the feature value. Network-structure features receive {net_share:.0%} of total SHAP attribution, so they are real but clearly secondary to living arrangement and care intensity.")
print("  wrote output/figures/f12_ablation_and_shap.png")


# 1. Leakage came in two tiers, and the second was only visible after the first
#    was removed. Two features were the outcome re-encoded; four more were zero
#    when the outcome was zero. Finding the second tier required asking
#    what each feature encodes, not scanning for large correlations: the
#    endogenous features have unremarkable correlations individually.
# 2. Once both tiers are removed, care-network structure adds little over the
#    existing covariates. 
# 3. Household structure and care intensity dominate; clinical need is detectable
#    Adding network features, survey weights, clustered CV, a bounded likelihood
#    and a neural network has not overturned it.
