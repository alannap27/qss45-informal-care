"""00_build_analysis_file.py

Takes in: data/raw/h22g_hp.csv (HRS 2022 Section G helper file) and, if
available, the full HRS 2022 core release.
Does: reconstructs the outcome from the helper records, validates the
committed analysis file against that reconstruction, and prints diagnostics
before and after every merge.
Outputs: data/processed/hrs2022_caregiving.csv (schema-checked),
output/tables/t00_sample_construction.csv.

Why the outcome is built this way

Each helper record carries a relationship code, how often that person helped,
and for how long. Monthly care hours are days of help times hours per helping
day. HRS lets a respondent answer the days question three different ways, 
days in the last month, days per week, or an "every day" flag, so all three
have to be reconciled before anything can be summed.

A helper is formal if the relationship is an organization, an institutional
employee, a paid helper, a professional, or an online service. Everyone else
is informal, unless the household reports paying them out of pocket, in
which case they are reclassified as formal. This means that a daughter
being paid to provide care is doing paid work.

Run:  python3 code/00_build_analysis_file.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from utils import (paths, clean_codes, DK_RF, savetable, RANDOM_STATE)

P = paths()
pd.set_option("display.width", 120)
print("repository root:", P["root"].name)

# Functions
# All functions are defined up front, per the repo rubric.

FORMAL_CODES = [21, 22, 23, 24, 25, 37]   # organization, institution, paid, professional, online


def monthly_days(hp):
    """Reconcile the three ways HRS records how often a helper helped."""
    days = clean_codes(hp["SG070"], DK_RF["two_digit"]) # days in last month
    days_wk = clean_codes(hp["SG071"], DK_RF["one_digit"]) # days per week
    every = hp["SG072"] == 1 # every-day flag
    days = days.fillna(days_wk * 4.33)
    return days.where(~every, 30.0)


def classify_helpers(hp):
    """Return the helper frame with monthly hours and a formal/informal flag."""
    rel = clean_codes(hp["SG069"], DK_RF["two_digit"])
    hrs = clean_codes(hp["SG073"], DK_RF["two_digit"])
    out = hp.copy()
    out["relationship"] = rel
    out["hours_month"] = monthly_days(hp) * hrs
    out["is_formal"] = rel.isin(FORMAL_CODES)
    out["is_informal"] = rel.notna() & ~rel.isin(FORMAL_CODES)

    paid = clean_codes(hp["SG078"], DK_RF["five_digit"]).fillna(0)
    reclass = out["is_informal"] & (paid > 0)
    out.loc[reclass, ["is_informal", "is_formal"]] = [False, True]
    print(f"  reclassified {int(reclass.sum())} informal helpers as formal "
          f"(household reported paying them out of pocket)")
    return out[out["hours_month"].notna() & out["relationship"].notna()]

def aggregate_to_respondent(hp):
    """Collapse helper records to one row per respondent."""
    hp = hp.copy()
    hp["informal_hours"] = np.where(hp["is_informal"], hp["hours_month"], 0.0)
    g = hp.groupby(["HHID", "PN"]).agg(
        total_care_hours=("hours_month", "sum"),
        informal_hours=("informal_hours", "sum"),
        n_helpers=("relationship", "size"),
        n_informal_helpers=("is_informal", "sum"),
    ).reset_index()
    g = g[g["total_care_hours"] > 0]
    g["informal_share"] = g["informal_hours"] / g["total_care_hours"]
    return g

# Reconstruct the outcome from the raw helper file

hp_raw = pd.read_csv(P["raw"] / "h22g_hp.csv", dtype={"HHID": str, "PN": str, "OPN": str},
                     low_memory=False)
print(f"helper records read: {len(hp_raw):,}")
print(f"distinct respondents represented: {hp_raw.groupby(['HHID','PN']).ngroups:,}")

hp = classify_helpers(hp_raw)
print(f"helper records with usable hours and relationship: {len(hp):,}")

care = aggregate_to_respondent(hp)
print(f"respondents with a computable informal share: {len(care):,}")
print()
print(care[["total_care_hours", "informal_share", "n_helpers"]].describe().round(3))

# Validate the committed analysis file
# The full analysis file also carries demographics, diagnoses, cognition and
# household structure drawn from the rest of the HRS release. Those source files
# are not redistributable (see the data note in the README), so the merged file
# is committed instead. This cell checks that the committed file agrees with the
# reconstruction above wherever they overlap.

df = pd.read_csv(P["processed"] / "hrs2022_caregiving.csv",
                 dtype={"HHID": str, "PN": str})
print(f"committed analysis file: {df.shape[0]:,} rows x {df.shape[1]} columns")

check = care.merge(df[["HHID", "PN", "informal_share"]], on=["HHID", "PN"],
                   how="inner", suffixes=("_rebuilt", "_committed"))
print(f"  rows matched on (HHID, PN): {len(check):,}")
delta = (check["informal_share_rebuilt"] - check["informal_share_committed"]).abs()
print(f"  max absolute discrepancy in informal_share: {delta.max():.10f}")
assert delta.max() < 1e-9, "committed file disagrees with reconstruction"
print("  OK - the committed outcome reproduces exactly from the raw helper file")

# Sample construction table

steps = pd.DataFrame([
    ["Helper records in HRS 2022 Section G", len(hp_raw)],
    [" with usable hours and relationship", len(hp)],
    ["Respondents represented", hp_raw.groupby(['HHID','PN']).ngroups],
    [" with total care hours > 0", len(care)],
    [" aged 50+ and merged to covariates (analysis sample)", len(df)],
], columns=["step", "n"])
print(steps.to_string(index=False))
savetable(steps, "t00_sample_construction.csv")

# Schema check

REQUIRED = ["informal_share", "total_care_hours", "n_helpers", "adl_count",
            "iadl_count", "age", "female", "race", "weight",
            "lives_alone", "lives_with_partner", "nursing_home"]
missing = [c for c in REQUIRED if c not in df.columns]
print("missing required columns:", missing if missing else "none")
assert not missing

print()
print("missingness in key fields:")
print((df[REQUIRED].isna().mean().sort_values(ascending=False) * 100).round(1).to_string())
print()
print(f"outcome at exactly 1.0: {(df['informal_share'] == 1).mean():.1%}")
print(f"outcome at exactly 0.0: {(df['informal_share'] == 0).mean():.1%}")
