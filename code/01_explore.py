"""01_explore.py

Takes in: data/processed/hrs2022_caregiving.csv
Does: distributions of the outcome and key predictors, the helper
composition, and the need gradient; records the observations that drive every
modeling decision downstream.
Outputs: output/figures/f01_*.png, output/tables/t01_descriptives.csv

Run:  python3 code/01_explore.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import suptitle, paths, savefig, savetable, style_axis, caption, BLUE, ORANGE, LIGHT

P = paths()
df = pd.read_csv(P["processed"] / "hrs2022_caregiving.csv", dtype={"HHID": str, "PN": str})
print(f"analysis sample: {len(df):,} respondents")

# Functions

REL_LABELS = {2: "Spouse/partner", 3: "Son", 5: "Son-in-law", 6: "Daughter",
              8: "Daughter-in-law", 9: "Grandchild", 15: "Brother", 17: "Sister",
              19: "Other relative", 20: "Other individual", 21: "Organization",
              22: "Institution employee", 23: "Paid helper", 37: "Online service"}
FORMAL_CODES = [21, 22, 23, 24, 25, 37]

def describe_numeric(frame, cols):
    d = frame[cols].describe().T
    d["missing_%"] = (frame[cols].isna().mean() * 100).round(1)
    return d.round(3)

# The outcome is close to degenerate; this is very consequential.

share = df["informal_share"]
print(f"share receiving 100% informal care: {(share == 1).mean():.1%}")
print(f"share receiving 0% informal care: {(share == 0).mean():.1%}")
print(f"share strictly between 0 and 1: {((share > 0) & (share < 1)).mean():.1%}")
print(f"median monthly care hours {df['total_care_hours'].median():.0f}")
print(f"mean helpers per respondent: {df['n_helpers'].mean():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.3))
ax = axes[0]
ax.hist(share, bins=20, color=BLUE, zorder=3)
ax.set_xlabel("Share of monthly care hours from informal helpers")
ax.set_ylabel("Respondents")
ax.set_title("A. The outcome piles up at 1", fontsize=11)
ax.annotate(f"{(share == 1).mean():.1%} of respondents\nreceive ALL care informally",
            xy=(1.0, (share == 1).sum()), xytext=(0.30, (share == 1).sum() * 0.72),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=1.0))
style_axis(ax)

ax = axes[1]
ax.hist(df["total_care_hours"], bins=45, color=BLUE, zorder=3)
ax.axvline(df["total_care_hours"].median(), color=ORANGE, lw=2, zorder=4)
ax.text(df["total_care_hours"].median() * 1.15, ax.get_ylim()[1] * 0.8,
        f"median {df['total_care_hours'].median():.0f} h/month", fontsize=9, color=ORANGE)
ax.set_xlabel("Total care hours received per month")
ax.set_ylabel("Respondents")
ax.set_title("B. Care intensity is heavily right-skewed", fontsize=11)
style_axis(ax)

suptitle(fig, "Figure 1. Distribution of the outcome and of care intensity")
caption(fig, f"HRS 2022, n = {len(df):,} respondents receiving any help with ADLs or IADLs. Panel A: informal share = informal care hours / total care hours per month. "
             "Panel B is truncated\nat the physical ceiling of 24 hours x 31 days. The mass at 1.0 in Panel A is why a plain linear model is a poor match and why script 03 fits a two-part specification.")
savefig(fig, "f01_outcome_and_intensity.png")

# Who actually provides the care

hp = pd.read_csv(P["raw"] / "h22g_hp.csv", dtype={"HHID": str, "PN": str}, low_memory=False)
rel = hp["SG069"].map(REL_LABELS).dropna()
counts = rel.value_counts().sort_values()

formal_names = {REL_LABELS[c] for c in FORMAL_CODES if c in REL_LABELS}
colors = [ORANGE if n in formal_names else BLUE for n in counts.index]

fig, ax = plt.subplots(figsize=(9.5, 6))
ax.barh(counts.index, counts.values, color=colors, zorder=3)
for i, v in enumerate(counts.values):
    ax.text(v + 8, i, str(v), va="center", fontsize=10)
ax.set_xlabel("Number of helper records")
ax.set_xlim(0, counts.max() * 1.14)
ax.set_title("Figure 2. Daughters are the modal caregiver, ahead of spouses", fontsize=12)
handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
ax.legend(handles, ["Informal (family, friends, neighbors)", "Formal (paid or institutional)"],
          fontsize=8.5, loc="lower right", frameon=False)
style_axis(ax, axis="x")
caption(fig, f"All {len(hp):,} helper records in HRS 2022 Section G, before the out-of-pocket-payment reclassification applied in script 00. Daughters ({counts.get('Daughter',0):,}) outnumber sons "
             f"({counts.get('Son',0):,}) roughly two to one,\na gendered pattern the hierarchical compensatory model does not itself predict. Formal providers of all kinds together account for a small minority of records.")
savefig(fig, "f02_helper_composition.png")

# The need gradient
# Andersen's equity criterion predicts that formal care should enter as need
# rises. It does, but weakly and not monotonically.

grp = df.groupby("adl_count")["informal_share"].agg(["mean", "size"])
grp = grp[grp["size"] >= 30]

fig, ax = plt.subplots(figsize=(9, 5.6))
ax.bar(grp.index.astype(int).astype(str), grp["mean"], color=BLUE, zorder=3, width=0.68)
for i, (m, n) in enumerate(zip(grp["mean"], grp["size"])):
    ax.text(i, m + 0.012, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i, 0.03, f"n = {int(n)}", ha="center", fontsize=10, color="white")
ax.set_ylim(0, 1.06)
ax.set_xlabel("Number of ADL limitations")
ax.set_ylabel("Mean informal care share")
ax.set_title("Figure 3. Formal care enters as need rises, but weakly and non-monotonically", fontsize=12)
style_axis(ax)
caption(fig, "Mean informal care share by count of the six HRS activities of daily living (dressing, walking, bathing, eating, transferring, toileting). Cells with fewer than 30 respondents are suppressed.\n"
             f"The share falls from {grp['mean'].iloc[0]:.3f} at zero limitations to {grp['mean'].min():.3f} at four, then partially reverses: the most severely limited respondents are not the most likely to have formal help.")
savefig(fig, "f03_need_gradient.png")

print(grp.round(3).to_string())
print()
print("informal share by neurological diagnosis (stroke, dementia or Alzheimer's):")
print(df.groupby("neuro_dx")["informal_share"].agg(["mean", "size"]).round(3).to_string())

# Table

CONT = ["informal_share", "total_care_hours", "n_helpers", "adl_count", "iadl_count",
        "age", "years_school", "word_recall", "rate_memory", "n_children"]
desc = describe_numeric(df, CONT)
print(desc.to_string())
savetable(desc.reset_index().rename(columns={"index": "variable"}), "t01_descriptives.csv")

BIN = ["female", "lives_alone", "lives_with_partner", "lives_with_relative",
       "nursing_home", "proxy", "stroke", "dementia", "alzheimers", "neuro_dx"]
binary = pd.DataFrame({"variable": BIN,
                       "share_1": [df[c].mean() for c in BIN],
                       "n_nonmissing": [df[c].notna().sum() for c in BIN]}).round(3)
print()
print(binary.to_string(index=False))
savetable(binary, "t01_binary_shares.csv")
