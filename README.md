# Who carries the care? Predicting reliance on informal caregiving

QSS 45 final project; Alanna Polyak, Dartmouth College.

Most long-term care in the United States is not delivered by the health system;
it is delivered by spouses, adult children, and other relatives, unpaid and
usually untrained. Among older adults who need help with daily activities, what
predicts whether their care comes entirely from family rather than from paid or
institutional providers?

Household structure beats clinical need; whom a person lives with dominates
every model fitted here, and stroke and functional limitation are detectable but
an order of magnitude smaller. Read through Andersen's behavioral model that is
an equity result rather than a null one: enabling structure, not need, is
allocating care.

---

## Three research questions

### RQ1. Is the outcome continuous enough to model as continuous?
No, **84.7%** of respondents receiving any help receive every hour of it from
unpaid family, and only 5.9% receive none informally. The middle is almost
empty, which is why a plain linear model is a poor match and why a two-part
specification is fitted alongside it.
→ `output/figures/f01_outcome_and_intensity.png`

### RQ2. What predicts reliance on family care?
Living arrangement, not diagnosis; living with a partner (β = +0.061,
*P* = 0.006) and living with a relative move the informal share far more than
stroke (β = −0.015, *P* = 0.03) or functional limitation (β = −0.026,
*P* = 0.001). No model class explains more than about a tenth of held-out
variance, and the neural network performs worse than predicting the mean.
→ `output/figures/f04_forest_vs_shap.png`, `f10_neural_net.png`

### RQ3. Do need and household structure interact?
Yes, **5 of 14** pre-specified interaction tests are significant at 5% against
roughly 0.7 expected by chance. Additional functional limitation reliably brings
formal care in for people living with a partner (slope −0.011, *P* = 0.0006) or
a relative (−0.014, *P* = 0.031), but for people living alone the slope is flat
and insignificant (−0.005, *P* = 0.52). A SHAP interaction decomposition ranks
the same pair first out of 630, which is two methods with different assumptions
selecting the same interaction.
→ `output/figures/f15_interaction_grid.png`, `f16_shap_interactions.png`

---

## Held-out performance

Two evaluation schemes are reported because they disagree, and the disagreement
is informative.

| Model | Task | Single 25% split | Grouped 5-fold CV |
|---|---|---|---|
| Intercept only | informal share | −0.007 | −0.000 |
| OLS / ridge | informal share | 0.011 | 0.061 |
| XGBoost, untuned defaults | informal share | −0.053 | |
| XGBoost, CV-tuned | informal share | 0.039 | 0.087 |
| Fractional logit | informal share | 0.032 | |
| Two-part model | informal share | 0.035 | |
| Multilayer perceptron | informal share | | −0.001 |
| Logistic regression | any formal care | AUC 0.710 | |
| XGBoost classifier | any formal care | AUC 0.728 | |

Cross-validation is grouped by household so that a married couple never
straddles the train and test sets. An untuned booster with sensible-looking 
defaults scores worse than predicting the mean, so reporting only the tuned 
figure would misrepresent boosting as reliably superior on data of this size.

Five scripts of added sophistication do not overturn the main finding, 
and that stability is the most useful thing this much robustness work can report.

---

## The outcome

```
    (monthly care hours supplied by unpaid family and friends)
    ---------------------------------------------------------
              (total monthly care hours received)
```

- **1.0**: every hour of care comes from unpaid family, 84.7% of the sample
- **0.5**: half the hours are paid or institutional
- **0.0**: no informal care at all, 5.9% of the sample

Built from 4,481 individual helper records rather than taken from a survey item,
which is what makes the leakage audit in script 07 necessary: every feature
derived from the helper roster is downstream of the outcome by construction.

---

## Scripts usefulness

### 1. Target leakage in two tiers

Script 05 builds nine care-network features; two of them are the outcome
re-encoded: `net_any_formal` is true when the informal share is below 1,
and including them lifts CV R² from **0.087 to 0.644**.

Removing those two is not enough; four more, the kin-composition features, are
zero exactly when nobody in the roster is family, which is the same event as the
informal share being zero. Ninety-four respondents have `net_n_kin_types = 0`
and their mean informal share is exactly 0.000. That second tier lifts R² to
**0.553** and is invisible to a correlation scan, because each feature looks
unremarkable alone.

Once both tiers are removed, the usable network features move CV R²
from 0.0874 to 0.0864, which is basically not at all. The test is not "is this
correlated with the outcome" but "could this have been known without knowing the
outcome?"
→ `output/figures/f11_leakage.png`

### 2. An automated crosswalk

Script 08 does the cross-national merge; the obstacle was that ELSA calls
the dressing item `headldr` and HRS calls it `SG014`.

Embedding the variable descriptions and matching on cosine similarity recovers
the crosswalk at **71% top-1 and 79% top-5** against all 403 ELSA derived
variables, where random guessing scores 0.2%.

It fails in an instructive way; on the ten ADL/IADL items it gets 9 of 10; on
the four items where the two studies measure the same construct with different
instruments, it gets 0 of 4, and does so confidently, matching HRS "NUMBER OF
YEARS IN SCHOOL" to ELSA `numbuad` (number of adults in the benefit unit) at
0.82 similarity. High similarity is not high accuracy, which is why the script
reports a per-item confidence margin to route uncertain matches to a human
rather than a headline accuracy number. All four errors fall in the bottom four
by margin.

A first version scored 100% on everything; that was a broken test: the HRS
labels had been written as paraphrases of the ELSA wording, so the matcher was
scoring on words that had been put there for it. The current version uses
verbatim HRS codebook labels.
→ `output/figures/f13_embedding_space.png`, `f14_matching_accuracy.png`

### 3. A hypothesis that did not survive

Script 04 expected household clustering to inflate performance, since spouses
are both interviewed. It does not: GroupKFold scores higher than naive KFold,
and the gap is smaller than the standard deviation across folds. Only 5.3% of
households contribute more than one respondent, so there is too little
duplication to leak. 

The same script found that all 73 nursing-home residents have no survey
weight, because HRS weights are built for the community-dwelling population.
Any weighted analysis silently drops them, so weighted and unweighted models are
not describing the same people.
→ `output/figures/f07_weights_and_clustering.png`

---

## Scripts

`code/utils.py` holds every shared function: path resolution, HRS code cleaning,
the feature builder, the network feature constructor, and the metric helpers.
`code/figstyle.py` holds the figure layout rules, including the two-pass
compositing that SHAP figures require.

| Script | Takes in | Does | Outputs |
|---|---|---|---|
| [`00_build_analysis_file.py`](code/00_build_analysis_file.py) | `data/raw/h22g_hp.csv` (not committed, needs registration) | reconstructs the outcome from 4,481 helper records, validates the analysis file against that reconstruction, schema-checks | `data/processed/*.csv`, `t00` |
| [`01_explore.py`](code/01_explore.py) | analysis file | outcome and intensity distributions, helper composition, need gradient | `f01`–`f03`, `t01` |
| [`02_baseline_ols_vs_gbm.py`](code/02_baseline_ols_vs_gbm.py) | analysis file | OLS against tuned and untuned XGBoost; forest plot against SHAP; rank comparison | `f04`–`f05`, `t02` |
| [`03_two_part_model.py`](code/03_two_part_model.py) | analysis file | fractional logit and two-part model for a bounded, spiked outcome | `f06`, `t03` |
| [`04_weights_and_clustered_cv.py`](code/04_weights_and_clustered_cv.py) | analysis file | survey weights, weighted regression, GroupKFold against KFold | `f07`, `t04` |
| [`05_care_networks.py`](code/05_care_networks.py) | helper file + analysis file | ego-network features via NetworkX; concentration and kin composition | `f08`–`f09`, `t05` |
| [`06_neural_net.py`](code/06_neural_net.py) | analysis file | MLP architecture and regularization search against linear and boosted baselines | `f10`, `t06` |
| [`07_final_model_and_leakage.py`](code/07_final_model_and_leakage.py) | networked analysis file | two-tier leakage audit, feature-block ablation, final model with SHAP | `f11`–`f12`, `t07` |
| [`08_crossnational_embeddings.py`](code/08_crossnational_embeddings.py) | ELSA variable dictionary | embeds variable descriptions to build the HRS→ELSA crosswalk, scored against a hand-coded key | `f13`–`f14`, `t08` |
| [`09_interactions.py`](code/09_interactions.py) | networked analysis file | pre-specified interaction tests and SHAP interaction values | `f15`–`f16`, `t09` |

```bash
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` runs scripts 01 through 09. Script 00 is excluded because its input
is the raw HRS Section G helper file, which cannot be redistributed under the
data use agreement; run it once you have registered and downloaded the
data. Every script states its inputs, what it does, and its outputs in the module
docstring, defines functions before use, and prints row counts around every merge.

### Checks the pipeline performs on itself

| Check | Where | Result |
|---|---|---|
| Reconstructed outcome against the committed analysis file | `00` | agrees on all 2,142 rows |
| Naive KFold against household-grouped KFold | `04` | difference within fold noise |
| Correlation scan against the "known without the outcome" test | `07` | scan misses tier 2 entirely |
| Pre-specified interaction tests against SHAP interaction values | `09` | both rank the same pair first |

---

## Data

**The HRS microdata are not committed and must not be redistributed.** Access is
free but requires registration at [hrs.isr.umich.edu](https://hrs.isr.umich.edu).
`data/raw/` and `data/processed/` are gitignored in full.

| Source | Gives | Coverage | Year |
|---|---|---|---|
| HRS 2022 Core, Section G (`h22g_hp.csv`) | one record per helper: relationship, frequency, hours, payment | 4,481 helper records | 2022 |
| HRS 2022 Core, respondent file | demographics, diagnoses, functional limitation, living arrangement | 2,142 respondents receiving any help | 2022 |
| HRS 2022 tracker | survey weights, household identifiers | same | 2022 |
| ELSA derived-variable dictionary | variable names and descriptions for the crosswalk | 403 candidates | various |

Produced and distributed by the University of Michigan with funding from the
National Institute on Aging (U01AG009740). 
---

## Limitations

1. **Cross-sectional:** A single 2022 wave, so nothing here identifies how care
   arrangements change as need progresses; every result is a description of
   who is receiving what, not of what caused it.
2. **The outcome is nearly degenerate:** With 85% of respondents at exactly 1.0
   there is very little variance for any model to explain, which caps every R²
   reported here and is the main reason the neural network fails.
3. **Care hours are self-reported** and reported by proxy for 15.4% of
   respondents, who are systematically the most impaired.
4. **Nursing-home residents have no survey weight**, so weighted and unweighted
   analyses do not describe the same population; both are reported.
5. **Roster-derived features are downstream of the outcome** by construction.
   Two tiers of leakage were found and removed, and there is no guarantee a third
   does not exist; the operative test is documented so it can be reapplied.
6. **The crosswalk is scored on 14 hand-coded pairs**, which is enough to
   demonstrate the method, but it is not enough to certify it. The confidence
   margin is reported for this reason.

---

## The full figure set

All sixteen are in the paper: four in the body, twelve in Appendix A.
**Paper** is the number LaTeX assigns by order of appearance, which is what a
reader cites; **Site** is what the website calls the same image.

| | Figure | Script | Paper | Site |
|---|---|---|---|---|
| f01 | Informal care share and care intensity | 01 | Fig. 1 | Figure 5 |
| f02 | Who the helpers are | 01 | Fig. 5 | gallery |
| f03 | Need gradient in the informal share | 01 | Fig. 6 | gallery |
| f04 | OLS coefficients against SHAP values | 02 | Fig. 2 | Figure 6 |
| f05 | Importance rank agreement across methods | 02 | Fig. 7 | gallery |
| f06 | Two-part and fractional-response models | 03 | Fig. 3 | gallery |
| f07 | Survey weights and household clustering | 04 | Fig. 9 | gallery |
| f08 | Care networks as ego graphs | 05 | Fig. 10 | gallery |
| f09 | Network-derived features | 05 | Fig. 11 | gallery |
| f10 | Neural network training and performance | 06 | Fig. 8 | gallery |
| f11 | Target leakage, sprung and removed | 07 | Fig. 12 | Figure 9 |
| f12 | Feature-block ablation with SHAP | 07 | Fig. 13 | gallery |
| f13 | Variable-description embedding space | 08 | Fig. 14 | gallery |
| f14 | Semantic variable-matching accuracy | 08 | Fig. 15 | Figure 8 |
| f15 | Need by living arrangement, and interaction tests | 09 | Fig. 4 | Figure 7 |
| f16 | SHAP interaction decomposition | 09 | Fig. 16 | gallery |

The columns disagree because each document orders figures for its own reader:
the file index is production order, the paper puts body figures first and holds
the rest for the appendix, and the site follows its argument. 

Every figure carries a self-contained caption naming its source and sample size,
so it can be read without the surrounding text (as required).
---

## Paper and website

- `paper/qss45_paper.tex` — the manuscript, written against the PNAS
  `pnasmathematics` template and 8 pages maximum.
- Project website source lives in a different `website/` folder, not in this repo.
