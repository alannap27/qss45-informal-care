"""03_two_part_model.py

Takes in: data/processed/hrs2022_caregiving.csv
Does: A linear model on a bounded, spike-at-one outcome is the wrong likelihood. 
Two alternatives are fitted and compared against the script-02 baselines:
a fractional logit (GLM, binomial family, logit link, quasi-likelihood)
and an explicit two-part model: a logit for whether any formal care is
used, then a conditional model for the share among those who use some.
Outputs: output/figures/f06_two_part.png, output/tables/t03_two_part.csv

The outcome is a proportion bounded on [0, 1] with 85% of its mass at
1.0. OLS can predict outside the unit interval, assumes constant variance,
and treats the spike as noise. The data-generating process is better described 
as two decisions: does any paid or institutional care enter this household at all,
and conditional on entering, how much of the load does it take. 
Those may have different drivers, and a single equation has them share one.

Run:  python3 code/03_two_part_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from utils import (suptitle, paths, build_features, savefig, savetable, style_axis,
                   caption, brier, RANDOM_STATE, BLUE, ORANGE, LIGHT)

P = paths()
df = pd.read_csv(P["processed"] / "hrs2022_caregiving.csv", dtype={"HHID": str, "PN": str})
X = build_features(df)
y = df["informal_share"].values

idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, random_state=RANDOM_STATE)
Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
sc = StandardScaler().fit(Xtr)
Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=X.columns, index=Xtr.index)
Xte_s = pd.DataFrame(sc.transform(Xte), columns=X.columns, index=Xte.index)
print(f"train {len(Xtr):,} / test {len(Xte):,}")

# Functions

def fit_fractional_logit(Xs, yv):
    """GLM with a binomial family on a continuous [0,1] response.
    Papke and Wooldridge's fractional response estimator: the binomial
    likelihood is used as a quasi-likelihood, so the point estimates stay
    consistent even though the outcome is not a count of successes. Predictions
    are guaranteed to land inside the unit interval.
    """
    return sm.GLM(yv, sm.add_constant(Xs), family=sm.families.Binomial()).fit()

def fit_two_part(Xs, yv):
    """Part 1: logit for 'any formal care'. Part 2: conditional share."""
    any_formal = (yv < 1).astype(int)
    part1 = LogisticRegression(max_iter=3000).fit(Xs, any_formal)

    mask = yv < 1
    # squeeze away from the boundary so the logit transform is finite
    y2 = np.clip(yv[mask], 1e-3, 1 - 1e-3)
    part2 = sm.GLM(y2, sm.add_constant(Xs[mask]), family=sm.families.Binomial()).fit()
    print(f"  part 1 fitted on {len(yv):,} rows; part 2 on {int(mask.sum()):,} "
          f"({mask.mean():.1%} of the sample use some formal care)")
    return part1, part2

def predict_two_part(part1, part2, Xs):
    """E[share] = P(no formal care) * 1 + P(any formal care) * E[share | some]."""
    p_any = part1.predict_proba(Xs)[:, 1]
    cond = part2.predict(sm.add_constant(Xs))
    return (1 - p_any) * 1.0 + p_any * cond, p_any, cond

# Fit all three specifications

ols = sm.OLS(ytr, sm.add_constant(Xtr_s)).fit()
pred_ols = np.asarray(ols.predict(sm.add_constant(Xte_s)))

frac = fit_fractional_logit(Xtr_s, ytr)
pred_frac = np.asarray(frac.predict(sm.add_constant(Xte_s)))

part1, part2 = fit_two_part(Xtr_s, ytr)
pred_2p, p_any_te, cond_te = predict_two_part(part1, part2, Xte_s)

print()
for name, pred in [("OLS", pred_ols), ("Fractional logit", pred_frac), ("Two-part", pred_2p)]:
    out_of_range = int(((pred < 0) | (pred > 1)).sum())
    print(f"{name:18} test R2 {r2_score(yte, pred):+.4f}   MAE {mean_absolute_error(yte, pred):.4f}   "
          f"predictions outside [0,1]: {out_of_range}")

# OLS produces predictions outside the unit interval, which are
# not interpretable as shares; both alternatives are constrained by
# construction. Whether they predict better is a separate question.

# What the two parts are doing

any_formal_te = (yte < 1).astype(int)
auc_part1 = roc_auc_score(any_formal_te, p_any_te)
print(f"part 1 (any formal care) test AUC: {auc_part1:.4f}")
print(f"part 1 Brier score: {brier(any_formal_te, p_any_te):.4f}")
mask_te = yte < 1
print(f"part 2 (conditional share) test R2: {r2_score(yte[mask_te], cond_te[mask_te]):+.4f} "
      f"on {int(mask_te.sum())} test rows")

# Part 1 does the majority while Part 2 is modeling how much of the load formal
# care takes once it is present, and is close to unpredictable from these
# covariates, which is itself a finding.

# Which coefficients differ between the two parts

p1_coef = pd.Series(part1.coef_[0], index=X.columns)
p2_coef = part2.params.drop("const")
both = pd.DataFrame({"part1_any_formal": p1_coef, "part2_conditional_share": p2_coef})
both["sign_flip"] = np.sign(both["part1_any_formal"]) != np.sign(both["part2_conditional_share"])
top = both.reindex(both["part1_any_formal"].abs().sort_values(ascending=False).index).head(12)
print(top.round(3).to_string())
print()
print(f"features whose sign flips between the two parts: {int(both['sign_flip'].sum())} of {len(both)}")
savetable(both.reset_index().rename(columns={"index": "feature"}), "t03_two_part_coefficients.csv")

# Figure 6

fig, axes = plt.subplots(1, 3, figsize=(16.4, 6.7),
                         gridspec_kw={"wspace": 0.36})

ax = axes[0]
for pred, lab, c in [(pred_ols, "OLS", ORANGE), (pred_2p, "Two-part", BLUE)]:
    ax.hist(pred, bins=40, alpha=0.6, label=lab, color=c, zorder=3)
ax.axvline(1.0, ls="--", color="black", lw=1.2, zorder=4)
ax.axvspan(ax.get_xlim()[0], 0, color="red", alpha=0.07, zorder=1)
ax.axvspan(1, ax.get_xlim()[1], color="red", alpha=0.07, zorder=1)
ax.set_xlabel("Predicted informal share")
ax.set_ylabel("Test respondents")
ax.set_title(f"A. OLS predicts outside [0,1]\n({int(((pred_ols<0)|(pred_ols>1)).sum())} predictions; two-part: 0)",
             fontsize=12)
ax.legend(fontsize=8.5, frameon=False)
style_axis(ax)

ax = axes[1]
bins = np.linspace(0, 1, 11)
who = np.digitize(p_any_te, bins) - 1
obs, exp, ns = [], [], []
for b in range(10):
    m = who == b
    if m.sum() >= 15:
        obs.append(any_formal_te[m].mean()); exp.append(p_any_te[m].mean()); ns.append(m.sum())
ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1.1)
ax.scatter(exp, obs, s=[n * 1.6 for n in ns], color=BLUE, zorder=3, alpha=0.85)
ax.set_xlabel("Predicted P(any formal care)")
ax.set_ylabel("Observed frequency")
ax.set_title(f"B. Part 1 is well calibrated\nAUC {auc_part1:.3f}, Brier {brier(any_formal_te, p_any_te):.3f}",
             fontsize=12)
style_axis(ax)

ax = axes[2]
names = ["Intercept\nonly", "OLS", "Fractional\nlogit", "Two-part"]
vals = [r2_score(yte, np.full_like(yte, ytr.mean())), r2_score(yte, pred_ols),
        r2_score(yte, pred_frac), r2_score(yte, pred_2p)]
cols = [LIGHT, ORANGE, BLUE, BLUE]
ax.bar(names, vals, color=cols, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + 0.002, f"{v:+.3f}", ha="center", fontsize=9.5, fontweight="bold")
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("Held-out $R^2$")
ax.set_title("C. Respecification buys coherence,\nnot much predictive power", fontsize=12)
style_axis(ax)

suptitle(fig, "Figure 6. A bounded outcome with a spike at one needs a bounded likelihood")
caption(fig, f"HRS 2022, n = {len(Xte):,} held-out respondents. Panel A: red bands mark the impossible region: a share below 0 or above 1. Panel B: each point is a decile of predicted probability, sized by the "
             "number of\nrespondents in it; points on the diagonal mean predicted and observed frequencies agree. Panel C: all four models on the same held-out split. The two-part model is the only specification that is "
             "both\ncoherent and competitive, which is why it is carried forward as the preferred specification in scripts 04 and 08.")
savefig(fig, "f06_two_part.png")

# Summary table

res = pd.DataFrame([
    ["Intercept only", r2_score(yte, np.full_like(yte, ytr.mean())), np.nan, 0],
    ["OLS", r2_score(yte, pred_ols), mean_absolute_error(yte, pred_ols),
     int(((pred_ols < 0) | (pred_ols > 1)).sum())],
    ["Fractional logit (GLM binomial)", r2_score(yte, pred_frac),
     mean_absolute_error(yte, pred_frac), int(((pred_frac < 0) | (pred_frac > 1)).sum())],
    ["Two-part (logit + conditional GLM)", r2_score(yte, pred_2p),
     mean_absolute_error(yte, pred_2p), int(((pred_2p < 0) | (pred_2p > 1)).sum())],
], columns=["specification", "test_r2", "test_mae", "predictions_outside_unit_interval"]).round(4)
print(res.to_string(index=False))
savetable(res, "t03_two_part.csv")
