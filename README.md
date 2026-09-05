# Protein function prediction: do language model embeddings survive an honest split?

## The question

Pretrained protein language models (PLMs) produce embeddings that beat simple
sequence-derived features on protein function prediction. That result is
everywhere. It is almost always measured on a **random** train/test split.

Protein databases are full of homologs — sequences that are near-copies of each
other because evolution works by copying an ancestor and mutating it. A random
split scatters members of the same family across train and test, so a model can
score well by recognising a near-duplicate rather than by learning anything.
That is data leakage.

This project measures what happens when the split is done honestly: sequences
are clustered by similarity and **whole clusters** are assigned to train or
test, so the test set contains families the classifier has never seen.

> For each feature set, how much of its random-split performance was real, and
> how much was homology memorisation?

The gap between the two split regimes — per method — is the result. Not the
accuracy.

## Vocabulary, for readers without a biology background

- **Amino acid / residue** — proteins are chains built from 20 possible
  building blocks. One link in the chain is one residue. Write one letter per
  residue and a protein becomes a string: `MKTAYIAKQRQISFVKSHFSRQ...`
- **Homolog** — two proteins descended from a common ancestor, which in
  practice means their strings are similar. Human and chimp haemoglobin are
  homologs.
- **Protein language model** — a transformer trained on protein sequences the
  way BERT was trained on text: mask a residue, predict it from context. It is
  shown no function labels, ever. We use it only as a feature extractor.

## Status

**Milestone 0 — complete.** Toy pipeline, synthetic data, no downloads.

```bash
python src/main.py
```

The 20 sequences in `src/main.py` are generated, not real. Class 1 is drawn
from a hydrophobic residue pool and class 0 from a polar/charged pool, which
caricatures the real signal separating membrane proteins from soluble ones.

**The 100% accuracy this prints is not a result.** The classes were built to be
separable by composition, the test set is six sequences, and the model has 21
features to fit 14 training points. It is a plumbing check: features have the
right shape, the model trains, metrics print. That is the entire bar for
Milestone 0.

## Roadmap

| Milestone | What |
|---|---|
| 0 | Toy end-to-end pipeline — **done** |
| 1 | Real dataset; homology-aware splitting with MMseqs2; measure the leakage |
| 2 | ESM-2 embeddings, small checkpoint, CPU, cached to disk |
| 3 | Baseline vs ESM under both split regimes; per-class breakdown |
| 4 | Write-up |

## Guardrails this project holds itself to

- Every split and model is seeded; results reproduce within a pinned environment.
- The composition baseline is computed before any embedding is, and is the
  number ESM has to beat.
- Class support is reported next to every metric. The primary metric is chosen
  before results are seen, not after.
- Embeddings are cached to disk and never recomputed in a loop.

## A limitation stated up front

Homology-aware splitting removes *label* leakage: it stops a test protein's
near-twin from sitting in the training set with the answer attached. It does
not remove **pretraining contamination**. ESM-2 was pretrained on UniRef, so it
has almost certainly already seen these test sequences, or close relatives, in
its unsupervised phase. Nothing done here can undo that.

So the defensible claim is not "PLM embeddings generalise to unseen protein
families." It is "PLM embeddings generalise to families unseen *by the
downstream classifier*." The narrower claim is the honest one.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
pytest
```
