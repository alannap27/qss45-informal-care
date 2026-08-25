# AI transcripts

The Agentic Analysis section of the paper refers to this folder.

## What is here

Four transcripts, one for each of the longer forms of explicit the paper cites.
Each is an edited excerpt from the session logs: prompts and the AI's 
replies are reproduced as written, while tool output, file diffs, and long code 
blocks are abridged to the lines that matter, with cuts marked `[...]`.

```
docs/agentic/
├── 00_network_features_and_leakage.md   the two-tier leakage, introduced and removed
├── 01_crosswalk_test.md                 the harmonization test that scored 100%
├── 02_clustering_hypothesis.md          a narrative written before the number existed
└── 03_website.md                        the project website, its design passes and deploy failures
```

| Claim in the paper | Transcript |
|---|---|
| Rejected: a harmonization test that scored 100% | `01_crosswalk_test.md` |
| Rejected: a narrative written before the number existed | `02_clustering_hypothesis.md` |
| Failure: two-tier target leakage, introduced by the AI | `00_network_features_and_leakage.md` |
| Website | `04_website.md` |

## Scope

The session ran from August 4th, 2026 to August 24th, 2026, across several sittings, and some would not download. The excerpts here are the exchanges that changed what the analysis does, which is what the reflection in the paper is about. The large remainder is routine: checking the comments structure, correcting a column name, re-running a script.
