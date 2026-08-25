# Transcript 02: A narrative written before the number existed

**Script:** `code/04_weights_and_clustered_cv.py`
**Paper reference:** Agentic Analysis, "Rejected: a narrative written before the
number existed"

Edited excerpt from the session log.

---

## The hypothesis

HRS interviews both members of a married couple, so a household can appear
twice. If a couple straddles a train/test split, the model may be scored partly
on people it has effectively already seen. I expected this to inflate
cross-validated performance and asked for the comparison.

> **Alanna:** Spouses are both in the sample, so households can span the split.
> Compare normal kfold against grouping on household

## What the AI wrote before running it

Asked to explain the comparison, the AI produced the explanation first:

> **AI:** Household clustering inflates naive cross-validation here
> because related respondents share unobserved household-level characteristics —
> income, housing, local care markets — so a model that has seen one spouse
> carries information about the other. GroupKFold corrects this, and the gap
> between the two designs is the size of the leak.

Fluent, mechanistically plausible, and consistent with what I expected; it was
also written before the number existed.

## What the number turned out to be

```
cv_design              mean_r2    sd
naive KFold             0.0831   0.0186
GroupKFold (household)  0.0874   0.0203
```

GroupKFold scores higher, not lower: the opposite of the predicted
direction, and the gap is smaller than the standard deviation across folds.

The reason is in the sampling: only 5.3% of households contribute more than one
respondent. There is simply not enough within-household duplication for leakage
to bite.

The note now in the script:

```python
# There is simply not enough within-household duplication for leakage to bite.
# Reporting this as "clustering inflated our R2 by X" would have been a
# fabricated finding.
#
# GroupKFold is still adopted as the default for the rest of the project. It
# costs nothing, it is the defensible design given the sampling structure, and
# if the sample were different it would matter.
```

## The other thing this script found

Separately, and not part of the hypothesis: all 73 nursing-home residents carry
no survey weight, because HRS weights are constructed for the
community-dwelling population; any weighted analysis silently drops them. Thus, the
weighted and unweighted models are not describing the same people, and the paper
says so.

---

## Reflection

What I asked for: a comparison of two cross-validation designs.

What I accepted: GroupKFold as the default, on design grounds rather than
results grounds.

What I rejected: the explanation, which was written to fit an expected result.

Where the AI went wrong: it supplied a mechanism for a finding that had
not been measured. Had the number come out as expected, the paragraph would have
been kept and would have looked like analysis. Confident responses about a result are
not evidence for it, and the order in which the two arrive is the only thing
that distinguishes explanation from rationalization.

I report the null, and the fact that it went the wrong way, rather than what I expected.
