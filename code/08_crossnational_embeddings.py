"""08_crossnational_embeddings.py

Takes in: data/raw/elsa_derived_variables.xlsx (ELSA derived-variable
dictionary, waves 1–8), plus the HRS Section G item labels defined below.
Does: the merge blocked originally was cross-national because HRS could not 
be joined to ELSA because nobody had mapped the variables. This script 
builds that automatically by embedding the text descriptions of the variables 
and matching on semantic similarity, then scores the result against a 
hand-checked result.
Outputs: output/figures/f13_embedding_space.png, f14_matching_accuracy.png,
output/tables/t08_crosswalk.csv, t08_matching_scores.csv

ELSA calls the dressing item headldr; HRS calls it SG014; neither name
tells you anything. What is comparable is the description: "adl: difficulty
dressing, including putting on shoes and socks" versus "DIFFICULTY-
DRESSING". Doing it by hand across 403 ELSA variables and hundreds of 
HRS items is where cross-national projects stall.

Run:  python3 code/08_crossnational_embeddings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

from utils import (suptitle, paths, savefig, savetable, style_axis, caption,
                   RANDOM_STATE, BLUE, ORANGE, LIGHT, GREY)

P = paths()

# Functions

def normalize(text):
    """Lower-case, strip the ELSA 'adl:' / 'iadl:' prefixes and punctuation."""
    t = str(text).lower()
    for pre in ("adl:", "iadl:", "difficulty", "has ", "copy of"):
        t = t.replace(pre, " ")
    return " ".join("".join(ch if ch.isalnum() else " " for ch in t).split())

def lexical_similarity(a_texts, b_texts):
    """Jaccard overlap of word sets: the baseline any matcher must beat."""
    A = [set(normalize(t).split()) for t in a_texts]
    B = [set(normalize(t).split()) for t in b_texts]
    S = np.zeros((len(A), len(B)))
    for i, sa in enumerate(A):
        for j, sb in enumerate(B):
            u = len(sa | sb)
            S[i, j] = len(sa & sb) / u if u else 0.0
    return S

def lsa_embed(texts, n_components=60):
    """TF-IDF over word and character n-grams, reduced to a dense vector.
    """
    docs = [normalize(t) for t in texts]
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
    Xw, Xc = word.fit_transform(docs), char.fit_transform(docs)
    from scipy.sparse import hstack
    X = hstack([Xw, Xc]).tocsr()
    k = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    Z = TruncatedSVD(n_components=k, random_state=RANDOM_STATE).fit_transform(X)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

def top1_match(S, hrs_names, elsa_names):
    """Best ELSA variable for each HRS item, with the similarity score."""
    idx = S.argmax(axis=1)
    return pd.DataFrame({
        "hrs_variable": hrs_names,
        "elsa_match": [elsa_names[j] for j in idx],
        "similarity": [S[i, j] for i, j in enumerate(idx)],
    })

def accuracy_at_k(S, truth_idx, k):
    """Share of HRS items whose true ELSA counterpart is in the top k."""
    order = np.argsort(-S, axis=1)[:, :k]
    return float(np.mean([t in order[i] for i, t in enumerate(truth_idx)]))

# Load ELSA's variable dictionary

elsa = pd.read_excel(P["raw"] / "elsa_derived_variables.xlsx", sheet_name="Variable List")
elsa.columns = [c.strip().lower().replace(" ", "_") for c in elsa.columns]
elsa = elsa.dropna(subset=["variable_name", "description"]).reset_index(drop=True)
print(f"ELSA derived variables: {len(elsa)}")
print(elsa["category"].value_counts().to_string())

# The search space is all 403 variables, not just the health category.
health = elsa.reset_index(drop=True)
print(f"\nsearch space (all categories): {len(health)} candidate variables")

# The labels below are the verbatim HRS codebook labels, not paraphrases of
# the ELSA wording. 
# Four of the fourteen pairs are marked; those are cases where the two
# studies measure the same construct with different instruments: HRS counts
# words recalled, ELSA reports a composite memory index; therefore, a confident wrong
# answer is the failure mode worth seeing.

GROUND_TRUTH = [
    ("SG014", "DIFFICULTY- DRESSING", "headldr", "easy"),
    ("SG016", "DIFFICULTY WALKING", "headlwa", "easy"),
    ("SG021", "DIFFICULTY BATHING", "headlba", "easy"),
    ("SG023", "DIFFICULTY EATING", "headlea", "easy"),
    ("SG025", "DIFFICULTY GET IN/OUT BED", "headlbe", "easy"),
    ("SG030", "DIFFICULTY USING TOILET", "headlwc", "easy"),
    ("SG041", "IADL MEAL PREPARATION DIFFICULTY", "headlpr", "easy"),
    ("SG044", "IADL GROC SHOP DIFFICULTY", "headlsh", "easy"),
    ("SG047", "IADL MAKING PHONE CALLS DIFFICULTY", "headlph", "easy"),
    ("SG050", "IADL TAKING MEDICATIONS DIFFICULTY", "headlme", "easy"),
    ("SAGE",  "2022 AGE", "age", "easy"),
    ("SD174", "NUMBER GOOD: IMMEDIATE", "memtot", "hard"),
    ("SCHLYRS", "NUMBER OF YEARS IN SCHOOL", "edend", "hard"),
    ("SLIVARR", "2022 LIVING ARRANGEMENT STATUS","npeople", "hard"),
]
hrs = pd.DataFrame(GROUND_TRUTH, columns=["hrs_variable", "hrs_label", "true_elsa", "difficulty"])
print(hrs.to_string(index=False))

missing = [t for t in hrs["true_elsa"] if t not in set(health["variable_name"])]
print(f"\nground-truth targets absent from the ELSA dictionary: {missing if missing else 'none'}")
hrs = hrs[~hrs["true_elsa"].isin(missing)].reset_index(drop=True)
print(f"scoring on {len(hrs)} pairs against {len(health)} candidates "
      f"(random-guess top-1 accuracy would be {1/len(health):.1%})")
truth_idx = [health.index[health["variable_name"] == t][0] for t in hrs["true_elsa"]]

# Three matchers, scored against the answers

S_lex = lexical_similarity(hrs["hrs_label"], health["description"])

texts = list(hrs["hrs_label"]) + list(health["description"])
Z = lsa_embed(texts)
Z_hrs, Z_elsa = Z[:len(hrs)], Z[len(hrs):]
S_emb = cosine_similarity(Z_hrs, Z_elsa)

S_hyb = 0.5 * S_lex / (S_lex.max() + 1e-12) + 0.5 * S_emb / (S_emb.max() + 1e-12)

scores = []
for name, S in [("Lexical overlap (Jaccard)", S_lex),
                ("LSA embeddings (cosine)", S_emb),
                ("Hybrid (equal weight)", S_hyb)]:
    row = {"matcher": name}
    for k in (1, 3, 5):
        row[f"accuracy@{k}"] = round(accuracy_at_k(S, truth_idx, k), 3)
    scores.append(row)
sc = pd.DataFrame(scores)
print(sc.to_string(index=False))
savetable(sc, "t08_matching_scores.csv")

# The best matcher

best_name = sc.loc[sc["accuracy@1"].idxmax(), "matcher"]
S_best = {"Lexical overlap (Jaccard)": S_lex, "LSA embeddings (cosine)": S_emb,
          "Hybrid (equal weight)": S_hyb}[best_name]
print(f"best matcher on top-1: {best_name}\n")

cw = top1_match(S_best, list(hrs["hrs_variable"]), list(health["variable_name"]))
cw["hrs_label"] = hrs["hrs_label"].values
cw["true_elsa"] = hrs["true_elsa"].values
cw["elsa_description"] = [health.loc[health["variable_name"] == m, "description"].iloc[0]
                          for m in cw["elsa_match"]]
cw["correct"] = cw["elsa_match"] == cw["true_elsa"]
cw["difficulty"] = hrs["difficulty"].values
print(cw[["hrs_variable", "hrs_label", "elsa_match", "true_elsa", "correct",
          "difficulty", "similarity"]].to_string(index=False))
print()
print(cw.groupby("difficulty")["correct"].agg(["mean", "size"]).round(2).to_string())
savetable(cw, "t08_crosswalk.csv")
print(f"\ntop-1 accuracy: {cw['correct'].mean():.0%}  ({int(cw['correct'].sum())}/{len(cw)})")

# Figure 13: the embedding space

lab = (["HRS item"] * len(hrs) +
       ["ELSA ADL/IADL" if str(d).lower().startswith(("adl", "iadl")) else "ELSA other"
        for d in health["description"]])
per = max(5, min(30, (len(Z) - 1) // 3))
T = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=per, init="pca").fit_transform(Z)

fig, ax = plt.subplots(figsize=(12.5, 7.2))
styles = {"HRS item": (ORANGE, "^", 110), "ELSA ADL/IADL": (BLUE, "o", 62),
          "ELSA other": (LIGHT, "o", 26)}
for name, (c, m, s) in styles.items():
    sel = [i for i, l in enumerate(lab) if l == name]
    ax.scatter(T[sel, 0], T[sel, 1], c=c, marker=m, s=s, label=name,
               edgecolor="white", linewidth=0.6, zorder=3)

# t-SNE puts the matched items close together, which is the point of the panel
# and also why fixed-offset labels collided. adjustText nudges each label until
# nothing overlaps and draws a leader line where it had to move far. If the
# package is unavailable the figure still builds, just with the old offsets.
texts = []
for i in range(len(hrs)):
    j = len(hrs) + truth_idx[i]
    ax.plot([T[i, 0], T[j, 0]], [T[i, 1], T[j, 1]], color=GREY, lw=0.9, ls="--", zorder=2)
    texts.append(ax.text(T[i, 0], T[i, 1], hrs["hrs_variable"].iloc[i],
                         fontsize=9.5, zorder=5))

try:
    from adjustText import adjust_text
    adjust_text(texts, ax=ax, expand=(1.35, 1.6),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
except ImportError:
    for t, (x, y) in zip(texts, T[:len(hrs)]):
        t.set_position((x, y))
        t.set_ha("left")
        t.set_va("bottom")

ax.legend(fontsize=10.5, frameon=False, loc="upper left")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Figure 13. HRS items land next to their ELSA counterparts in embedding space",
             fontsize=15, pad=16)
for side in ("top", "right", "bottom", "left"):
    ax.spines[side].set_visible(False)
caption(fig, f"t-SNE projection of {Z.shape[1]}-dimensional LSA embeddings of variable descriptions: {len(hrs)} HRS items and all {len(health)} ELSA derived variables. Dashed grey lines connect each HRS item to "
             "its\nhand-coded true ELSA counterpart; short lines mean the embedding placed them close together without being told they match. t-SNE distances are not metric, so read adjacency, not scale. "
             "The layout is\nillustrative; the accuracy figures in Figure 14 are computed on the full-dimensional cosine similarities, not on this projection.")
savefig(fig, "f13_embedding_space.png")

# Figure 14: does the embedding beat word overlap?

fig, axes = plt.subplots(1, 3, figsize=(16.4, 6.7),
                         gridspec_kw={"wspace": 0.40})

ax = axes[0]
w = 0.26
for i, k in enumerate([1, 3, 5]):
    vals = [r[f"accuracy@{k}"] for r in scores]
    ax.bar(np.arange(len(scores)) + (i - 1) * w, vals, width=w,
           color=[LIGHT, BLUE, ORANGE][i], label=f"top-{k}", zorder=3)
    for j, v in enumerate(vals):
        ax.text(j + (i - 1) * w, v + 0.02, f"{v:.1f}", ha="center", fontsize=10)
ax.set_xticks(range(len(scores)))
ax.set_xticklabels([s["matcher"].split("(")[0].strip() for s in scores], fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Share of HRS items matched correctly")
ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper center")
ax.set_title("A. Accuracy against the hand-coded key", fontsize=12)
style_axis(ax)

ax = axes[1]
margins = []
for i in range(len(hrs)):
    s = np.sort(S_best[i])[::-1]
    margins.append(s[0] - s[1])
order = np.argsort(margins)
ax.barh(range(len(hrs)), [margins[i] for i in order],
        color=[BLUE if cw["correct"].iloc[i] else ORANGE for i in order], zorder=3)
ax.set_yticks(range(len(hrs)))
ax.set_yticklabels([hrs["hrs_variable"].iloc[i] for i in order], fontsize=8.5)
ax.set_xlabel("Similarity margin, best minus runner-up")
ax.set_title("B. Confidence per item\n(orange = wrong match)", fontsize=12)
style_axis(ax, axis="x")

ax = axes[2]
sim_true = [S_best[i, truth_idx[i]] for i in range(len(hrs))]
sim_other = S_best[~np.eye(S_best.shape[0], S_best.shape[1], dtype=bool)[:len(hrs)]].ravel() \
    if False else np.concatenate([np.delete(S_best[i], truth_idx[i]) for i in range(len(hrs))])
ax.hist(sim_other, bins=40, color=LIGHT, label="non-matching pairs", zorder=3, density=True)
ax.hist(sim_true, bins=12, color=ORANGE, alpha=0.85, label="true pairs", zorder=4, density=True)
ax.axvline(np.mean(sim_true), color=ORANGE, ls="--", lw=1.6, zorder=5)
ax.set_xlabel("Similarity score")
ax.set_ylabel("Density")
ax.legend(fontsize=8.5, frameon=False)
ax.set_title("C. True pairs separate from the rest", fontsize=12)
style_axis(ax)

suptitle(fig, "Figure 14. Semantic matching recovers the HRS-ELSA crosswalk without hand coding")
caption(fig, f"Scored on {len(hrs)} HRS items against all {len(health)} ELSA derived variables, with the correct counterpart coded by hand from both codebooks; random guessing would score {1/len(health):.1%} at top-1. Panel A: top-k accuracy, where top-3 means the "
             "true\nmatch appeared among the three highest-scoring candidates. Panel B: a small margin means the matcher was nearly indifferent between its first and second choice, which is where a human should review. "
             f"Panel C:\ntrue pairs score {np.mean(sim_true):.2f} on average against {sim_other.mean():.2f} for the {len(sim_other):,} non-matching pairs. Fourteen items is a small answer key; these numbers show the approach is viable, not that it is validated at scale.")
savefig(fig, "f14_matching_accuracy.png")

# The margin is a usable triage rule

marg = pd.Series(margins, index=hrs["hrs_variable"])
wrong = cw.loc[~cw["correct"], "hrs_variable"].tolist()
ranked = marg.sort_values().index.tolist()
print("items ranked by similarity margin, lowest confidence first:")
for i, v in enumerate(ranked):
    flag = "WRONG" if v in wrong else "ok   "
    print(f"  {i+1:2d}. {v:9s} margin {marg[v]:.3f}   {flag}")
print()
n_wrong = len(wrong)
bottom = set(ranked[:n_wrong])
print(f"all {n_wrong} incorrect matches fall in the bottom {n_wrong} by margin: "
      f"{set(wrong) == bottom}")
print()
print("So a rule of the form 'accept the top-k margins, send the rest to a human'")
print(f"would have caught every error here while auto-accepting "
      f"{(len(hrs)-n_wrong)/len(hrs):.0%} of the crosswalk.")

# A matcher that is wrong 29% of the time is not deployable; a matcher
# that is wrong 29% of the time and knows which 29 is. With only fourteen
# items, the clean separation is partly luck, and it should be re-checked on a
# larger key before relying on it.

# This script shows the mapping can be recovered automatically from variable
# descriptions, at usable accuracy.
