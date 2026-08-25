# Transcript 01: The harmonization test that scored 100%

**Script:** `code/08_crossnational_embeddings.py`
**Paper reference:** Agentic Analysis, "Rejected: a harmonization test that
scored 100%"

Edited excerpt from the session log, abridged.

---

## The task

> **Alanna:** If I wanted to run this in England too, ELSA names everything
> differently. Can the variable descriptions be matched automatically instead of
> me doing it by hand?

The AI built three matchers over the variable descriptions: a lexical
TF-IDF baseline, an LSA embedding, and a combination; scored them against a
hand-coded key.

## The result that was inaccurate because it was too "good"

All three matchers scored 100% at top-1, which I pointed out.

> **AI:** All three matchers hit 100% — the test is too easy because I
> paraphrased the HRS labels. Let me use the real codebook labels and search all
> 403 variables.

**Why it was broken:** The AI had written the HRS-side labels itself,
paraphrasing the ELSA wording. The matcher was scoring on words that had been
put there for it to find. The lexical baseline in particular was matching shared
vocabulary that existed only because one side had been written after reading the
other.

Nothing errored; every line ran, and 100% is a real number computed correctly
from the inputs it was given. The failure is entirely in the construction of the
test, and the code cannot see it.

## What the script does now

The HRS side uses verbatim codebook labels, and the search runs against all
403 ELSA-derived variables rather than a shortlist. The comment in the script
records why:

```python
# ## The HRS side, with hand-coded ground truth
#
# The labels below are the **verbatim HRS codebook labels**, not paraphrases of
# the ELSA wording. That distinction decides whether this is a real test: an
# earlier version used tidied-up labels and all three matchers scored 100%,
# because the lexical baseline could match on shared phrasing I had introduced.
```

Honest scores against the real test:

```
matcher                 acc@1   acc@3   acc@5
lexical TF-IDF          0.500   0.643   0.643
LSA embedding           0.714   0.786   0.786
combined                0.714   0.786   0.786
```

71.4% top-1 against a 0.2% random baseline, and the drop from 100% is the
difference between a test and a demonstration.

## The failure pattern worth keeping

On the ten ADL and IADL items, the matcher gets nine right. On four items where
the two studies measure the same construct with different instruments, it gets
zero right: matching HRS "NUMBER OF YEARS IN SCHOOL"
to ELSA `numbuad`, the count of adults in the benefit unit, at 0.82 similarity.

High similarity is not high accuracy, so the script reports a per-item
confidence margin, the gap between best and runner-up, rather than a
headline accuracy. All four errors fall in the bottom four by margin, which
makes the margin usable as a triage rule.

---

## Reflection

What I asked for: an automated crosswalk.

What I accepted: the three-matcher design, and the margin-based determination rule.

What I rejected: the 100% result, and the version of the notebook that produced
it.

Where the AI went wrong: it built the answer key and the thing being
tested from the same source. It did flag the problem after being prompted by me, which matters,
but only after presenting the number first. A result that good on a task should have been checked before it was reported, not after.
