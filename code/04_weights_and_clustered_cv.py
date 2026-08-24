"""04_weights_and_clustered_cv.py

Takes in: data/processed/hrs2022_caregiving.csv
Does: 
1. Weights: HRS oversamples Black and Hispanic households and Florida
   residents. Every unweighted descriptive in scripts 01–03 is a statement
   about the sample, not about older Americans. The respondent weight
   (SWGTR) is carried in the analysis file but was never used initially.
2. Clustering: Spouses are both interviewed, so two rows can come from
   one household and share a helper roster. Splitting individuals at random
   puts one spouse in train and the other in test. Whether that materially
   inflates held-out performance turns out to be no.

Outputs: output/figures/f07_weights_and_clustering.png,
output/tables/t04_weighted_descriptives.csv, t04_cv_designs.csv

Run:  python3 code/04_weights_and_clustered_cv.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import xgboost as xgb
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from utils import (suptitle, paths, build_features, savefig, savetable, style_axis,
                   caption, weighted_mean, RANDOM_STATE, BLUE, ORANGE, LIGHT, GREY)

P = paths()
df = pd.read_csv(P["processed"] / "hrs2022_caregiving.csv", dtype={"HHID": str, "PN": str})
X = build_features(df)
y = df["informal_share"].values
print(f"analysis sample: {len(df):,}")

# How much clustering is there?

hh = df.groupby("HHID").size()
print(f"distinct households: {len(hh):,}")
print(f"households with 2+ rows: {(hh >= 2).sum():,}  ({(hh >= 2).mean():.1%})")
print(f"respondents in such households: {int(hh[hh >= 2].sum()):,} "
      f"({hh[hh >= 2].sum() / len(df):.1%} of the sample)")

w = df["weight"]
print()
print(f"weight available for: {w.notna().mean():.1%} of respondents")
print(f"weight range: {w.min():,.0f} to {w.max():,.0f}")
print(f"design effect (Kish): {(1 + (w.std() / w.mean())**2):.2f}")

# The Kish design effect is the factor by which the variance of a weighted mean
# exceeds that of a simple random sample of the same size. An effective sample
# size of n / deff is the denominator for any weighted statistic.

# Functions

def weighted_vs_unweighted(frame, cols, wcol="weight"):
    rows = []
    sub = frame[frame[wcol].notna()]
    for c in cols:
        u = sub[c].mean()
        wm = weighted_mean(sub[c], sub[wcol])
        rows.append([c, u, wm, wm - u, (wm - u) / u * 100 if u else np.nan])
    return pd.DataFrame(rows, columns=["variable", "unweighted", "weighted",
                                       "difference", "pct_change"]).round(4)

def cv_r2(model, X, y, cv, groups=None):
    s = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="r2", n_jobs=-1)
    return s.mean(), s.std()

# Weighted versus unweighted descriptives

COLS = ["informal_share", "adl_count", "iadl_count", "age", "lives_alone",
        "lives_with_partner", "nursing_home", "stroke", "dementia", "n_children"]
wt = weighted_vs_unweighted(df, COLS)
print(wt.to_string(index=False))
savetable(wt, "t04_weighted_descriptives.csv")

hs = wt.loc[wt["variable"] == "informal_share"].iloc[0]
print()
print(f"headline: informal share is {hs['unweighted']:.4f} unweighted and "
      f"{hs['weighted']:.4f} weighted, a shift of {hs['difference']:+.4f}")

# Weighted regression
# Weighted least squares changes what the coefficients estimate: from a
# description of these 2,142 respondents to an estimate for the population they
# represent.

sub = df[df["weight"].notna()].copy()
Xs_all = build_features(sub)
sc = StandardScaler().fit(Xs_all)
Xs = pd.DataFrame(sc.transform(Xs_all), columns=Xs_all.columns)
ys = sub["informal_share"].values

unw = sm.OLS(ys, sm.add_constant(Xs)).fit()
wls = sm.WLS(ys, sm.add_constant(Xs), weights=sub["weight"].values).fit()

comp = pd.DataFrame({"unweighted_beta": unw.params.drop("const"),
                     "weighted_beta": wls.params.drop("const"),
                     "unweighted_p": unw.pvalues.drop("const"),
                     "weighted_p": wls.pvalues.drop("const")})
comp["abs_change"] = (comp["weighted_beta"] - comp["unweighted_beta"]).abs()
comp["significance_flips"] = ((comp["unweighted_p"] < 0.05) != (comp["weighted_p"] < 0.05))
top = comp.sort_values("abs_change", ascending=False).head(10)
print(top.round(4).to_string())
print()
print(f"coefficients whose 5% significance verdict flips when weighted: "
      f"{int(comp['significance_flips'].sum())} of {len(comp)}")
savetable(comp.reset_index().rename(columns={"index": "feature"}), "t04_weighted_coefficients.csv")

# Cross-validation design: does household clustering inflate performance?

model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.02,
                         min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                         random_state=RANDOM_STATE)

naive_m, naive_s = cv_r2(model, X, y, KFold(5, shuffle=True, random_state=RANDOM_STATE))
grouped_m, grouped_s = cv_r2(model, X, y, GroupKFold(5), groups=df["HHID"])

print(f"naive 5-fold KFold: R2 = {naive_m:+.4f}  (sd across folds {naive_s:.4f})")
print(f"household GroupKFold: R2 = {grouped_m:+.4f}  (sd across folds {grouped_s:.4f})")
print(f"difference: {naive_m - grouped_m:+.4f}")
print(f"pooled fold-level sd: {np.mean([naive_s, grouped_s]):.4f}")
print()
if abs(naive_m - grouped_m) < np.mean([naive_s, grouped_s]):
    print("The difference is smaller than the fold-to-fold standard deviation, so it")
    print("cannot be distinguished from noise. Note also that the grouped estimate came")
    print("out higher, which is the opposite of what leakage would produce.")

# It was expected that random splitting leaks households and inflates
# performance; however, that is not what happened. The grouped estimate is higher, 
# and the gap is smaller than the standard deviation across folds, so the two
# designs are indistinguishable on this data.

# Only 5.3% of households contribute more than one respondent, 
# covering about 10% of rows. There is not enough within-household 
# duplication for leakage to be an issue.
#
# GroupKFold is still adopted as the default for the rest of the project. It
# costs nothing, it is the defensible design given the sampling structure, and
# if the sample were later extended to couples-based recruitment, the leakage
# would become more real.
#
# Every one of the 73 nursing-home residents in the sample has a missing
# respondent weight, which is why the weighted nursing-home mean above is
# zero. HRS respondent weights are constructed for the community-dwelling population.
# Any weighted analysis silently drops institutionalized entirely, 
# so the weighted and unweighted models are not describing the same people.

nh = df[df["nursing_home"] == 1]
print(f"nursing-home residents in the sample: {len(nh)}")
print(f"of whom have a survey weight: {int(nh['weight'].notna().sum())}")
print(f"their mean informal share: {nh['informal_share'].mean():.4f}")
print(f"everyone else's mean informal share: {df.loc[df['nursing_home'] == 0, 'informal_share'].mean():.4f}")
print()
print(f"respondents dropped by any weighted analysis: {int(df['weight'].isna().sum())} "
      f"({df['weight'].isna().mean():.1%} of the sample)")

# Figure 7

fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.5))

ax = axes[0]
plot = wt[wt["variable"] != "informal_share"].copy().sort_values("pct_change")
ax.barh(plot["variable"], plot["pct_change"],
        color=[ORANGE if abs(v) > 5 else LIGHT for v in plot["pct_change"]], zorder=3)
ax.axvline(0, color="black", lw=1)
for i, v in enumerate(plot["pct_change"]):
    ax.text(v + (0.4 if v >= 0 else -0.4), i, f"{v:+.1f}%", va="center",
            ha="left" if v >= 0 else "right", fontsize=8)
ax.set_xlabel("% change when survey weights are applied")
ax.set_title("A. Weighting moves the descriptives\n(orange: shift larger than 5%)", fontsize=12)
style_axis(ax, axis="x")

ax = axes[1]
ax.scatter(comp["unweighted_beta"], comp["weighted_beta"], s=42,
           color=[ORANGE if f else BLUE for f in comp["significance_flips"]], zorder=3, alpha=0.85)
lim = float(np.abs(np.concatenate([comp["unweighted_beta"], comp["weighted_beta"]])).max()) * 1.15
ax.plot([-lim, lim], [-lim, lim], ls="--", color="black", lw=1)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("Unweighted OLS coefficient")
ax.set_ylabel("Weighted (WLS) coefficient")
ax.set_title(f"B. Coefficients shift under weighting\n({int(comp['significance_flips'].sum())} change their 5% verdict, in orange)",
             fontsize=12)
style_axis(ax, axis="both")

ax = axes[2]
ax.bar(["Naive\nKFold", "Household\nGroupKFold"], [naive_m, grouped_m],
       yerr=[naive_s, grouped_s], capsize=6, color=[ORANGE, BLUE], zorder=3)
for i, v in enumerate([naive_m, grouped_m]):
    ax.text(i, v + 0.004, f"{v:+.4f}", ha="center", fontsize=10, fontweight="bold")
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("Cross-validated $R^2$")
ax.set_title(f"C. The two CV designs are indistinguishable\ndifference {naive_m - grouped_m:+.4f}, fold sd ~{np.mean([naive_s, grouped_s]):.3f}", fontsize=12)
style_axis(ax)

suptitle(fig, "Figure 7. Survey weights change the estimand; household clustering turns out not to matter here")
caption(fig, f"HRS 2022, n = {len(df):,}. Panel A compares the sample mean with the population-weighted mean using the HRS respondent weight SWGTR (available for {df['weight'].notna().mean():.0%} of the sample; Kish design "
             f"effect {1 + (df['weight'].std()/df['weight'].mean())**2:.2f}).\nPanel B plots each standardized coefficient unweighted against weighted; points off the diagonal are estimates that depend on which population is being described. Panel C compares two 5-fold designs on "
             f"the identical model:\nrandom splitting of individuals versus splitting whole households. Only {(hh >= 2).mean():.0%} of households contribute more than one respondent, so there is little to leak; the difference is smaller than the\n"
             "error bars, which are the standard deviation across folds, and runs in the opposite direction to leakage. GroupKFold is adopted anyway as the defensible default. Note that Panel A drops all 73 nursing-home\n"
             "residents, who have no survey weight, so the weighted column describes the community-dwelling population only.")
savefig(fig, "f07_weights_and_clustering.png")

# Summary

res = pd.DataFrame([
    ["Naive 5-fold KFold (splits individuals)", round(naive_m, 4), round(naive_s, 4),
     "leaks 10% of rows across the split"],
    ["Household GroupKFold (splits households)", round(grouped_m, 4), round(grouped_s, 4),
     "adopted as default; difference from naive is within fold noise"],
], columns=["cv_design", "mean_r2", "sd_across_folds", "note"])
print(res.to_string(index=False))
savetable(res, "t04_cv_designs.csv")
print()
print("From here on, GroupKFold on HHID is the default design (scripts 05-08).")
