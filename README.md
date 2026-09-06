# Do protein language models still win when you test them honestly?

A small, self-contained machine learning experiment. Written so that someone
with no biology background can follow it — because that is who wrote it.

## First, what a protein is

Living things run on proteins. They digest your food, carry oxygen in your
blood, copy your DNA, fight off infections. Each protein is a chain built from
20 possible building blocks called amino acids. Write one letter per building
block and a protein turns into a piece of text:

```
MKTAYIAKQRQISFVKSHFSRQEDLQ...
```

A typical one runs a few hundred letters. That is the entire input to this
project. **A protein is a string.** That is why a computer scientist can work
on this at all.

## The problem

The string does not tell you the job. The chain folds up into a 3D shape, and
the shape decides what the protein does. Cheap DNA sequencing has handed us
hundreds of millions of protein sequences, but only a tiny sliver have ever
been studied in a lab to find out what they actually do. The gap is enormous
and it grows every year: mountains of strings, almost no labels.

So the question is whether we can predict what a protein does from its string
alone. (Which specific job we predict — where in the cell the protein works, or
which parts of it grab onto DNA — is settled in Milestone 1.)

## Two ways to turn a string into numbers

A model cannot eat text. It needs numbers, and it needs the *same count* of
numbers for every protein, even though proteins come in different lengths.
There are two ways to do that, and comparing them is the point of this project.

**The cheap way: count the letters.** What fraction of this protein is each of
the 20 building blocks? That is 20 numbers, and it works for a protein of any
length. It discards all the ordering, which sounds fatal but is not — a cell
membrane is greasy, so proteins that live inside a membrane are built out of
greasy building blocks, and plain letter-counting notices that.

**The expensive way: ask a protein language model.** ESM-2 is a transformer,
trained the same way a language model is trained on English, except its
language is protein sequences and it read roughly 250 million of them. Hide a
letter, guess it from context, repeat billions of times. It is never told what
a single protein *does*. But to get good at guessing, it has to absorb a great
deal about how proteins are built. Hand it a sequence and it returns a few
hundred numbers summarising that protein — an *embedding*.

Everybody already knows the expensive way wins. That is not the project.

## The catch, which *is* the project

Evolution works by copying. Every species inherited its proteins from an
ancestor and picked up a few mutations along the way, so protein databases are
stuffed with families of near-identical strings. Human, chimp, mouse and whale
haemoglobin are largely the same sequence with edits. Proteins related this way
are called **homologs**.

Now do what every machine learning tutorial says: shuffle the data, train on
80%, test on the other 20%. Human haemoglobin lands in the training pile. Chimp
haemoglobin lands in the test pile. Your model gets the chimp one right — not
because it learned any biology, but because it saw a near-duplicate during
training and effectively looked up the answer.

**That is data leakage, and it quietly inflates every number you report.** It
is an exam where half the questions are ones the student already saw with the
numbers changed. The score is real. The conclusion you draw from it is not.

Almost every published "language model beats simple features" result is
measured this way.

## How we solve it

Before splitting the data, group all the sequences into families by similarity
— that is what [MMseqs2](https://github.com/soedinglab/MMseqs2) does, a fast
clustering tool for sequences. Then hand out **whole families** to training or
to testing, never splitting a family across both. Now the test set contains
protein families the classifier has genuinely never encountered.

Then run the whole experiment twice, once with the sloppy random split and once
with the honest one, and compare.

> For each method, how much of its random-split score was real understanding,
> and how much was memorising near-duplicates?

The answer is the *drop* between the two splits, measured separately for the
cheap features and the expensive ones. Not the accuracy. The drop.

That is worth doing because there is no boring outcome. If the language model
holds its lead, that is evidence it learned something general. If it collapses
while letter-counting barely moves, that says the published numbers are partly
an artifact of how the data was split. Either result is a finding.

## Where the project is now

**Milestone 0 — done.** A toy pipeline on invented data, to prove the plumbing
works before any real data arrives.

```bash
source .venv/bin/activate
python src/main.py
```

```
20 sequences -> feature matrix (20, 21)
  class 0 (soluble-like): 10 sequences
  class 1 (membrane-like): 10 sequences

train: 14 sequences   test: 6 sequences
accuracy: 1.000
confusion matrix (rows = true, columns = predicted):
[[3 0]
 [0 3]]
```

### What that 100% is worth: nothing

Three independent reasons, all of them disqualifying:

1. The 20 sequences are **made up**. One class was generated from a greasy pool
   of building blocks, the other from a charged pool. They were built to be
   separable by letter-counting, and then separated by letter-counting.
2. The test set is **six sequences**. The only possible scores are 0%, 17%,
   33%, 50%, 67%, 83% and 100%.
3. There are 21 features fitted to 14 training examples. A model with that much
   freedom can separate almost anything.

Milestone 0 asked one question — does a sequence make it all the way through to
a printed metric without crashing — and the answer is yes. That is the whole
bar. Any number printed here is a plumbing check, not a result.

`pytest` covers the one piece of real logic: that a feature vector has the
right shape and that the 20 letter-frequencies actually sum to 1.

## Roadmap

| Milestone | What happens |
|---|---|
| 0 | Toy pipeline end to end — **done** |
| 1 | Real dataset; group sequences into families with MMseqs2; measure how much leakage the random split was hiding |
| 2 | ESM-2 embeddings, smallest checkpoint, runs on a laptop CPU, cached to disk |
| 3 | Cheap features vs. embeddings, under both splits, with a per-class breakdown |
| 4 | Write-up |

## Rules this project holds itself to

- Every split and every model is seeded, so results reproduce exactly within a
  pinned environment.
- The letter-counting baseline is computed *before* any embedding is. It is the
  number the language model has to beat, and it is not allowed to be a
  strawman.
- How many examples are in each class gets printed next to every metric. "90%
  accurate" means something completely different when the classes are 50/50
  than when they are 90/10.
- The headline metric is chosen before the results are seen, not after.
- Embeddings are computed once and cached to disk, never recomputed in a loop.

## One limitation, stated up front

Splitting by family removes *label* leakage: it stops a test protein's near-twin
from sitting in the training set with the answer attached.

It does **not** undo the fact that ESM-2 already read these proteins. ESM-2 was
trained on a huge public database of sequences, so it has almost certainly seen
our test proteins, or close relatives of them, during its own training. Nothing
in this project can remove that.

So the honest claim is not "language model embeddings generalise to protein
families nobody has ever seen." It is "they generalise to families unseen *by
the classifier we trained on top*." The narrower claim is the true one, and it
is stated here rather than buried.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
pytest
```
