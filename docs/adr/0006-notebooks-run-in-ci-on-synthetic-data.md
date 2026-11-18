# 0006. CI runs every notebook on a synthetic file with the DataCo schema

**Status:** Accepted

## Context
The analysis lives in five notebooks. Notebooks that nobody re-runs rot: a library upgrade or an edit to `src/` can break them without anyone noticing. The DataCo dataset needs a Kaggle login, so CI cannot download it.

## Decision
`tests/make_sample_dataco.py` writes about 12,000 orders with the same columns, formats and value ranges the notebooks use: a fixed promise per mode, the 2017-10-02 volume break, and enough Second and Standard Class orders for notebook 05's sample. `notebooks.yml` installs the pinned `requirements.txt`, writes the file and executes all five notebooks with nbconvert.

## Alternatives considered
- **Store a Kaggle API key as a secret.** It would run on the real data, but it depends on a personal credential and the dataset's terms.
- **Commit a slice of the real data.** Redistributing the dataset is not ours to do.

## Consequences
- Every change to `src/`, the notebooks or the requirements proves that the notebooks still run end to end (about 30 seconds).
- The synthetic numbers are meaningless. The results in the README come from the real dataset and are saved in the committed notebook outputs.
