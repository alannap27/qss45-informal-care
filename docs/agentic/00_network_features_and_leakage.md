# Transcript 00: Care-network features and the two-tier leakage

**Scripts:** `code/05_care_networks.py`, `code/07_final_model_and_leakage.py`
**Paper reference:** Agentic Analysis, "Failure: two-tier target leakage,
introduced by the AI"

Edited excerpt from the session log. Prompts and replies are as written; tool
output and code blocks are abridged, with cuts marked `[...]`.

---

## The task

> **Alanna:** Each person has a roster of helpers. Can we treat that as a
> network and build features off the structure, like how concentrated the care
> is and who is in it.

The AI built nine features over the helper roster: network size, hours
Herfindahl, top-helper share, spouse present, number of children, number of
distinct kin types, generation span, and `net_any_formal`.

---

## Tier one

Running the descriptives, there was a flag:

> 57% of respondents rely on a single caregiver. Also caught a
> leakage trap: `net_any_formal` is almost the outcome by definition. Handling
> that explicitly in the modeling notebooks.

```
feature_block                        n_features   cv_r2
+ care intensity (full base set)             33  0.0874
+ tier 1 leakage (excluded)                  42  0.6440
```

An $R^2$ jumping from 0.087 to 0.644 on an outcome where 85% of the mass sits at
a single value is not a breakthrough. 

`net_any_formal` is true when the informal share is below 1. It is the
outcome, re-encoded as a binary.

---

## Tier two, which the first fix missed

> **Alanna:** Ok so remove net_any_formal and whatever else is the outcome in
> disguise.

> **AI:** Removed `net_any_formal` and `net_n_informal`. The remaining
> seven features are structural and safe to use.

They were not; with the two obvious ones gone, the ablation still returned:

```
+ tier 2 endogenous (excluded)               40  0.5530
```

Four kin-composition features are zero exactly when nobody in the roster is
family: which is the same event as the informal share being zero.

```
respondents with net_n_kin_types == 0 : 94
their mean informal_share            : 0.000
```

Ninety-four people, all at 0.000. The feature and the outcome are not
correlated quantities; they are the same event described twice.

This tier was invisible to a correlation scan, because each of the four features
looks unremarkable alone. The AI had described the remaining set as
"safe" without running the check that would have tested the claim.

---

## What settles it

The operative test is not is this correlated with the outcome but could this
have been known without knowing the outcome. The outcome is constructed from
the helper roster, so every roster-derived feature is downstream of it by
construction: a property of the study design.

With both tiers removed:

```
+ care intensity (full base set)             33  0.0874
+ network shape, tier 3 (FINAL)              36  0.0864
```

The exogenous network features move $R^2$ by −0.001, which is nothing.

---

## Reflection

What I asked for: network features.

What I accepted: the nine features, and the first removal.

What I rejected: the claim that the remaining seven were safe.

Where the AI went wrong: it introduced the leakage, caught the obvious
tier with my prompting, then certified the rest without testing. It was a partial fix that reads like a complete one. Had I stopped at 0.553, I
would have reported a six-fold improvement in held-out $R^2$ as a finding.

I kept the whole progression in the paper rather than deleting the features,
because the issue transfers further than the result does.
