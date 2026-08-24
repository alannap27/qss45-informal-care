"""05_care_networks.py

Takes in: data/raw/h22g_hp.csv, data/processed/hrs2022_caregiving.csv
Does: treats each respondent's set of helpers as an ego network and builds
structural features from it: size, kin composition, concentration of hours on
a single caregiver, generational span. These describe the shape of a care
arrangement, which none of the covariates in scripts 02–04 capture.
Outputs: output/figures/f08_care_networks.png, f09_network_features.png,
output/tables/t05_network_features.csv

Why a network framing

Cantor's hierarchical compensatory model: care is drawn from a ranked 
set of relations, and formal help enters when that set is exhausted. 
A model with n_children and lives_with_partner as separate
scalars cannot express "this person's care rests entirely on one daughter"
versus "four relatives split it".

The most useful quantity turns out to be the Herfindahl concentration of
care hours: the probability that two randomly drawn care hours came from
the same person; a value of 1 means one caregiver does everything.

Warning: The outcome is built from this same helper roster. Any feature derived from
the roster is therefore downstream of the outcome and has to be checked for
target leakage. Two of the features constructed here are leakage 
and two others are partly endogenous; script 07 identifies them, quantifies the damage, 
and drops them. 

Run:  python3 code/05_care_networks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from utils import (suptitle, paths, clean_codes, DK_RF, savefig, savetable, style_axis,
                   caption, BLUE, ORANGE, LIGHT, GREY)

P = paths()
df = pd.read_csv(P["processed"] / "hrs2022_caregiving.csv", dtype={"HHID": str, "PN": str})
hp = pd.read_csv(P["raw"] / "h22g_hp.csv", dtype={"HHID": str, "PN": str, "OPN": str},
                 low_memory=False)
print(f"helper records: {len(hp):,}")

# Functions

FORMAL = [21, 22, 23, 24, 25, 37]
SPOUSE = [2]
CHILD = [3, 4, 6, 7, 28, 30, 90]
CHILD_IN_LAW = [5, 8, 31, 91]
GRANDCHILD = [9, 33]
SIBLING = [15, 16, 17, 18]
PARENT = [10, 11, 12, 13, 14, 35, 36]
OTHER_KIN = [19]
NON_KIN = [20, 26, 27, 32]

GENERATION = {}
for c in PARENT: GENERATION[c] = -1
for c in SPOUSE + SIBLING: GENERATION[c] = 0
for c in CHILD + CHILD_IN_LAW: GENERATION[c] = 1
for c in GRANDCHILD: GENERATION[c] = 2

def prepare_helpers(hp):
    days = clean_codes(hp["SG070"], DK_RF["two_digit"])
    days_wk = clean_codes(hp["SG071"], DK_RF["one_digit"])
    every = hp["SG072"] == 1
    days = days.fillna(days_wk * 4.33).where(~every, 30.0)
    hrs = clean_codes(hp["SG073"], DK_RF["two_digit"])
    out = hp.copy()
    out["rel"] = clean_codes(hp["SG069"], DK_RF["two_digit"])
    out["hours"] = days * hrs
    return out[out["hours"].notna() & out["rel"].notna()].copy()

def herfindahl(hours):
    """Concentration of care hours on a single helper. 1 = one person does all."""
    h = np.asarray(hours, float)
    if h.sum() <= 0:
        return np.nan
    s = h / h.sum()
    return float((s ** 2).sum())

def ego_features(g):
    """Structural summary of one respondent's care network."""
    rel = g["rel"].values
    hours = g["hours"].values
    informal = ~np.isin(rel, FORMAL)
    gens = [GENERATION.get(int(r)) for r in rel]
    gens = [x for x in gens if x is not None]
    return pd.Series({
        "net_size": len(g),
        "net_n_informal": int(informal.sum()),
        "net_hhi_hours": herfindahl(hours),
        "net_top_helper_share": float(hours.max() / hours.sum()) if hours.sum() > 0 else np.nan,
        "net_has_spouse": int(np.isin(rel, SPOUSE).any()),
        "net_n_children": int(np.isin(rel, CHILD).sum()),
        # counted over informal helpers only. Counting "formal" as a kin type
        # would make this feature partly an encoding of the outcome.
        "net_n_kin_types": int(len({("spouse" if r in SPOUSE else
                                     "child" if r in CHILD + CHILD_IN_LAW else
                                     "grandchild" if r in GRANDCHILD else
                                     "sibling" if r in SIBLING else
                                     "parent" if r in PARENT else
                                     "otherkin" if r in OTHER_KIN else "nonkin")
                                    for r in rel if r not in FORMAL})),
        # 0 rather than NaN when no kin helper is present: a NaN here would be
        # missing exactly when every helper is formal, which is the outcome.
        "net_generation_span": (max(gens) - min(gens)) if len(gens) >= 1 else 0,
        "net_any_formal": int(np.isin(rel, FORMAL).any()),
    })

def build_bipartite(hp_sub):
    """Respondent-helper bipartite graph, for the structural illustration."""
    G = nx.Graph()
    for _, r in hp_sub.iterrows():
        ego = f"R:{r['HHID']}-{r['PN']}"
        alter = f"H:{r['HHID']}-{r['PN']}-{r['OPN']}"
        G.add_node(ego, kind="respondent")
        G.add_node(alter, kind="formal" if r["rel"] in FORMAL else "informal")
        G.add_edge(ego, alter, weight=float(r["hours"]))
    return G

# Build the ego-network features

hpc = prepare_helpers(hp)
print(f"helper records with usable hours: {len(hpc):,}")

feat = hpc.groupby(["HHID", "PN"]).apply(ego_features, include_groups=False).reset_index()
print(f"ego networks built: {len(feat):,}")
print()
print(feat[["net_size", "net_hhi_hours", "net_top_helper_share",
            "net_n_kin_types", "net_generation_span"]].describe().round(3).to_string())

before = len(df)
merged = df.merge(feat, on=["HHID", "PN"], how="left")
print()
print(f"  merge [network features -> analysis file]: {before} rows in -> {len(merged)} rows out, "
      f"{int(merged['net_size'].notna().sum())} matched (left join)")
assert len(merged) == before

# What the structure looks like

print("concentration of care hours:")
print(f" networks where one helper supplies all hours: {(merged['net_hhi_hours'] == 1).mean():.1%}")
print(f" median top-helper share of hours: {merged['net_top_helper_share'].median():.3f}")
print(f" median network size: {merged['net_size'].median():.0f}")
print()
print("informal share by concentration:")
bins = pd.cut(merged["net_hhi_hours"], [0, 0.4, 0.6, 0.8, 0.999, 1.0],
              labels=["<0.4 (spread)", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0 (sole carer)"])
print(merged.groupby(bins, observed=True)["informal_share"].agg(["mean", "size"]).round(3).to_string())

# Figure 8: four illustrative care networks

merged["_key"] = merged["HHID"] + "-" + merged["PN"]
hpc["_key"] = hpc["HHID"] + "-" + hpc["PN"]

picks = []
cand = merged[merged["net_size"].notna()]
picks.append(("Sole informal carer", cand[(cand["net_size"] == 1) & (cand["net_any_formal"] == 0)].iloc[0]))
picks.append(("Several kin sharing", cand[(cand["net_size"] >= 4) & (cand["net_any_formal"] == 0)].nlargest(1, "net_n_kin_types").iloc[0]))
picks.append(("Mixed kin and paid", cand[(cand["net_size"] >= 3) & (cand["net_any_formal"] == 1)].iloc[0]))
picks.append(("Formal only", cand[(cand["net_any_formal"] == 1) & (cand["net_n_informal"] == 0)].iloc[0]))

fig, axes = plt.subplots(1, 4, figsize=(16.0, 5.4))
for ax, (label, row) in zip(axes, picks):
    sub = hpc[hpc["_key"] == row["_key"]]
    G = build_bipartite(sub)
    ego = [n for n, d in G.nodes(data=True) if d["kind"] == "respondent"]
    pos = nx.spring_layout(G, seed=7, k=1.4)
    cols = ["black" if G.nodes[n]["kind"] == "respondent"
            else (ORANGE if G.nodes[n]["kind"] == "formal" else BLUE) for n in G.nodes]
    sizes = [340 if G.nodes[n]["kind"] == "respondent" else 190 for n in G.nodes]
    widths = [0.6 + 3.4 * (G[u][v]["weight"] / max(d["weight"] for *_, d in G.edges(data=True)))
              for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#bbbbbb")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cols, node_size=sizes)
    ax.set_title(f"{label}\nHHI = {row['net_hhi_hours']:.2f}, "
                 f"informal share = {row['informal_share']:.2f}", fontsize=9.5)
    ax.axis("off")

handles = [plt.Line2D([], [], marker="o", ls="", color="black", label="Respondent"),
           plt.Line2D([], [], marker="o", ls="", color=BLUE, label="Informal helper"),
           plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="Formal helper")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, -0.04))
suptitle(fig, "Four real care networks from HRS 2022")
caption(fig, "Each panel is one respondent's ego network, drawn from their Section G helper records. Edge thickness is monthly care hours. HHI is the Herfindahl concentration of those hours: 1.00 means a single\n"
             "person supplies every hour of care. These four are selected to span the structural range, not sampled at random. The scalar covariates used in scripts 02-04 cannot distinguish the first panel from\n"
             "the second, both are 100% informal, even though one rests on a single person and the other spreads across four.")
savefig(fig, "f08_care_networks.png")

# Figure 9: do the structural features carry signal?

fig, axes = plt.subplots(1, 3, figsize=(16.4, 6.7),
                         gridspec_kw={"wspace": 0.42})

ax = axes[0]
ax.hist(merged["net_hhi_hours"].dropna(), bins=30, color=BLUE, zorder=3)
ax.axvline(1.0, ls="--", color=ORANGE, lw=1.6)
# Anchored to the left of the panel, not beside the spike at 1.0, where it sat
# on top of the tallest bar in the histogram.
ax.set_ylim(0, ax.get_ylim()[1] * 1.16)
ax.text(0.02, 0.95, f"{(merged['net_hhi_hours'] == 1).mean():.0%} rely on\na single carer",
        transform=ax.transAxes, ha="left", va="top", fontsize=10.5, color=ORANGE)
ax.set_xlabel("Herfindahl concentration of care hours")
ax.set_ylabel("Respondents")
ax.set_title("A. Most care rests on one person", fontsize=12)
style_axis(ax)

ax = axes[1]
g = merged.groupby("net_size")["informal_share"].agg(["mean", "size"])
g = g[g["size"] >= 25]
ax.bar(g.index.astype(int).astype(str), g["mean"], color=BLUE, zorder=3, width=0.66)
for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
    ax.text(i, m + 0.014, f"{m:.2f}", ha="center", fontsize=10)
    ax.text(i, 0.045, f"n={int(n)}", ha="center", fontsize=9.5, color="white")
ax.set_ylim(0, 1.08)
ax.set_xlabel("Number of helpers in the network")
ax.set_ylabel("Mean informal share")
ax.set_title("B. Larger networks are less purely informal", fontsize=12)
style_axis(ax)

ax = axes[2]
corr = merged[["informal_share", "net_size", "net_hhi_hours", "net_top_helper_share",
               "net_n_kin_types", "net_n_children", "net_has_spouse",
               "net_generation_span"]].corr()["informal_share"].drop("informal_share")
corr = corr.sort_values()
ax.barh(corr.index, corr.values,
        color=[ORANGE if v < 0 else BLUE for v in corr.values], zorder=3)
ax.axvline(0, color="black", lw=1)
for i, v in enumerate(corr.values):
    ax.text(v + (0.008 if v >= 0 else -0.008), i, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=10)
ax.set_xlabel("Correlation with informal share")
ax.set_title("C. Concentration is the strongest structural signal", fontsize=12)
# Widen past the extremes so the value labels have somewhere to sit, and shorten
# the tick labels, which are what was reaching back into panel B.
ax.set_yticklabels([t.get_text().replace("net_", "").replace("_", " ")
                    for t in ax.get_yticklabels()], fontsize=10)
lo, hi = ax.get_xlim()
ax.set_xlim(lo - 0.05, hi + 0.05)
style_axis(ax, axis="x")

suptitle(fig, "Network structure carries information the scalar covariates do not")
caption(fig, f"HRS 2022, n = {int(merged['net_size'].notna().sum()):,} respondents with a reconstructable care network. Panel B suppresses network sizes with fewer than 25 respondents. Panel C shows simple Pearson correlations, "
             "not partial\neffects; they establish that the features move with the outcome, while script 07 tests whether they add anything once the existing covariates are already in the model.")
savefig(fig, "f09_network_features.png")

# Save the augmented file

NETCOLS = [c for c in merged.columns if c.startswith("net_")]
print("network features created:", NETCOLS)
merged.drop(columns=["_key"]).to_csv(P["processed"] / "hrs2022_caregiving_networks.csv", index=False)
print(f"wrote data/processed/hrs2022_caregiving_networks.csv  ({len(merged):,} rows)")

summary = merged[NETCOLS].describe().T.round(3)
summary["corr_with_outcome"] = merged[NETCOLS].corrwith(merged["informal_share"]).round(3)
print()
print(summary.to_string())
savetable(summary.reset_index().rename(columns={"index": "feature"}), "t05_network_features.csv")
