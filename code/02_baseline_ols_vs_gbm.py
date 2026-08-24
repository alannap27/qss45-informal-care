"""02_baseline_ols_vs_gbm.py

Takes in: data/processed/hrs2022_caregiving.csv
Does: reproduces the Initial Data Analysis comparison, a linear model
against a cross-validated XGBoost regressor, and adds baselines
Outputs: output/figures/f04_forest_vs_shap.png, f05_importance_ranks.png,
output/tables/t02_model_comparison.csv, t02_importance_ranks.csv

Run:  python3 code/02_baseline_ols_vs_gbm.py
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
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyRegressor

from utils import (suptitle, paths, build_features, savefig, savetable, style_axis,
                   caption, RANDOM_STATE, BLUE, ORANGE)

P = paths()
df = pd.read_csv(P["processed"] / "hrs2022_caregiving.csv", dtype={"HHID": str, "PN": str})
X = build_features(df)
y = df["informal_share"].values
print(f"feature matrix: {X.shape[0]:,} rows x {X.shape[1]} columns")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
print(f"train {len(Xtr):,} / test {len(Xte):,}")

# Functions

def fit_ols(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=X.columns)
    Xte_s = pd.DataFrame(sc.transform(Xte), columns=X.columns)
    m = sm.OLS(ytr, sm.add_constant(Xtr_s)).fit()
    pred = m.predict(sm.add_constant(Xte_s))
    return m, r2_score(yte, pred), pred

def tune_booster(Xtr, ytr):
    grid = {"max_depth": [2, 3, 4], "n_estimators": [150, 300, 600],
            "learning_rate": [0.02, 0.05], "min_child_weight": [1, 10]}
    s = GridSearchCV(
        xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         random_state=RANDOM_STATE),
        grid, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="r2", n_jobs=-1)
    s.fit(Xtr, ytr)
    return s

# Fit

dummy = DummyRegressor(strategy="mean").fit(Xtr, ytr)
r2_dummy = r2_score(yte, dummy.predict(Xte))
print(f"intercept-only baseline test R2 : {r2_dummy:.4f}  (predicts {ytr.mean():.3f} for everyone)")

ols, r2_ols, _ = fit_ols(Xtr, ytr, Xte, yte)
print(f"OLS                  test R2 : {r2_ols:.4f}   (in-sample {ols.rsquared:.4f})")

untuned = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                           random_state=RANDOM_STATE).fit(Xtr, ytr)
r2_untuned = r2_score(yte, untuned.predict(Xte))
print(f"XGBoost, untuned     test R2 : {r2_untuned:.4f}")

search = tune_booster(Xtr, ytr)
booster = search.best_estimator_
r2_xgb = r2_score(yte, booster.predict(Xte))
print(f"XGBoost, 5-fold CV   test R2 : {r2_xgb:.4f}")
print(f"  best params: {search.best_params_}  (CV R2 {search.best_score_:.4f})")

# The untuned booster loses to OLS; the tuned one wins. Reporting only the
# tuned number would misrepresent gradient boosting as reliably better here;
# reporting only the untuned number would misrepresent it as worse. Both go in
# the table.

# Binary framing
# Since the outcome is nearly binary, the meaningful contrast is
# whether the household gets any formal help at all.

yb = (df["informal_share"] < 1).astype(int).values
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, yb, test_size=0.25, random_state=RANDOM_STATE)
sc2 = StandardScaler().fit(Xtr2)

logit = LogisticRegression(max_iter=3000).fit(sc2.transform(Xtr2), ytr2)
auc_logit = roc_auc_score(yte2, logit.predict_proba(sc2.transform(Xte2))[:, 1])

clf = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=RANDOM_STATE, eval_metric="logloss").fit(Xtr2, ytr2)
auc_clf = roc_auc_score(yte2, clf.predict_proba(Xte2)[:, 1])

print(f"base rate of any formal care : {yb.mean():.3f}")
print(f"logistic regression test AUC: {auc_logit:.4f}")
print(f"XGBoost classifier test AUC: {auc_clf:.4f}")

# Importance: coefficients versus SHAP

explainer = shap.TreeExplainer(booster)
sv = explainer.shap_values(Xte)
mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)

coefs = ols.params.drop("const")
ci = ols.conf_int().drop("const")

comp = pd.DataFrame({"ols_abs_coef": coefs.abs(), "shap_mean_abs": mean_abs})
comp["ols_rank"] = comp["ols_abs_coef"].rank(ascending=False)
comp["shap_rank"] = comp["shap_mean_abs"].rank(ascending=False)
comp["rank_gap"] = comp["ols_rank"] - comp["shap_rank"]
rho = comp["ols_rank"].corr(comp["shap_rank"], method="spearman")
print(f"Spearman correlation of the two importance rankings: {rho:.3f}")
print()
print(comp.sort_values("shap_rank").head(12).round(3).to_string())
savetable(comp.sort_values("shap_rank").reset_index().rename(columns={"index": "feature"}),
          "t02_importance_ranks.csv")

# Figure 4: forest plot beside the SHAP beeswarm

order = coefs.abs().sort_values(ascending=False).head(15).index[::-1]

# Each panel is rendered to its own file first, then composited. 
import figstyle

tmp_forest = paths()["figures"] / "_panel_forest.png"
tmp_shap = paths()["figures"] / "_panel_shap.png"

fig_a, ax = plt.subplots(figsize=(7.4, 6.6))
ax.errorbar(coefs[order], range(len(order)),
            xerr=[coefs[order] - ci.loc[order, 0], ci.loc[order, 1] - coefs[order]],
            fmt="o", capsize=3, markersize=4.5, color=BLUE, ecolor="#9aa0a6")
ax.axvline(0, ls="--", lw=1.1, color="#333333")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=8.6)
ax.set_xlabel("Standardized OLS coefficient (95% CI)")
style_axis(ax, axis="x")
fig_a.savefig(tmp_forest, format="png")
plt.close(fig_a)

fig_b = plt.figure(figsize=(7.4, 6.6))
shap.summary_plot(sv, Xte, show=False, max_display=15, plot_size=None)
plt.gcf().savefig(tmp_shap, format="png")
plt.close("all")

figstyle.compose(
    [tmp_forest, tmp_shap],
    ["A.  OLS: one partial effect for everyone",
     "B.  SHAP: a contribution per respondent"],
    paths()["figures"] / "f04_forest_vs_shap.png",
    f"The two models rank the same predictors differently (Spearman rho = {rho:.2f})",
    "Panel A: standardized OLS coefficients, so 0.06 means a one-standard-deviation increase is associated with a 6-percentage-point higher informal share, holding every other predictor fixed; bars are 95%\n"
    "confidence intervals. Panel B: each dot is one held-out respondent, horizontal position is how far that feature moved that person's prediction away from the average, and color is the feature value, red high\n"
    "and blue low. A coefficient and a mean absolute SHAP value are not the same quantity, which is why the rankings diverge. The two features that move furthest between the rankings are lives_alone, which\n"
    "is first by SHAP and 24th by coefficient because the living-arrangement indicators are nearly collinear, and total_care_hours, whose relationship to the outcome is non-monotone.")
print("  wrote output/figures/f04_forest_vs_shap.png")

# Figure 5: where the rankings disagree

top = comp.sort_values("shap_rank").head(14).copy()
fig, ax = plt.subplots(figsize=(10, 6))
ypos = np.arange(len(top))
ax.hlines(ypos, top["ols_rank"], top["shap_rank"], color="#cccccc", lw=2, zorder=2)
ax.scatter(top["ols_rank"], ypos, s=55, color=BLUE, zorder=3, label="OLS rank")
ax.scatter(top["shap_rank"], ypos, s=55, color=ORANGE, zorder=3, label="SHAP rank")
ax.set_yticks(ypos)
ax.set_yticklabels(top.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Importance rank (1 = most important)")
ax.legend(fontsize=9, frameon=False, loc="lower right")
ax.set_title("Two features move a long way between the two rankings", fontsize=12)
for i, (f, r) in enumerate(zip(top.index, top["rank_gap"])):
    if abs(r) >= 10:
        ax.annotate(f"gap {int(abs(r))}", (max(top['ols_rank'].iloc[i], top['shap_rank'].iloc[i]) + 0.7, i),
                    fontsize=8, va="center", color=ORANGE)
style_axis(ax, axis="x")
caption(fig, "Features ordered by SHAP rank. A long grey bar means the two methods disagree. total_care_hours and lives_alone are the two large movers: the first because its relationship to the outcome\n"
             "is non-monotone (very low-intensity help is nearly always informal, very high-intensity help skews formal, the middle is mixed), the second because the three living-arrangement indicators are\n"
             "nearly collinear, so OLS splits the signal between them while the tree assigns it to whichever single indicator splits best.")
savefig(fig, "f05_importance_ranks.png")

# Model comparison table

res = pd.DataFrame([
    ["Intercept only", "regression on informal share", "test R2", round(r2_dummy, 4)],
    ["OLS", "regression on informal share", "test R2", round(r2_ols, 4)],
    ["XGBoost (untuned defaults)", "regression on informal share", "test R2", round(r2_untuned, 4)],
    ["XGBoost (5-fold CV tuned)", "regression on informal share", "test R2", round(r2_xgb, 4)],
    ["Logistic regression", "any formal care", "test AUC", round(auc_logit, 4)],
    ["XGBoost classifier", "any formal care", "test AUC", round(auc_clf, 4)],
], columns=["model", "task", "metric", "value"])
print(res.to_string(index=False))
savetable(res, "t02_model_comparison.csv")
