"""Step 4: the experiment.

Four runs: two feature sets (amino acid composition, ESM-2 embeddings) crossed
with two splits (random, homology-aware). The model, the preprocessing and the
seed are identical across all four -- only the features and the split change.
That is what makes it a controlled comparison rather than four separate results.

The number of interest is not any single accuracy. It is the DROP each feature
set suffers when the split stops leaking.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from main import featurize

DATA = Path("data")
SEED = 0

# Both feature sets go through the identical pipeline. class_weight="balanced"
# is chosen because the pre-registered primary metric is macro-F1, which weights
# every compartment equally; it is applied to both arms, so it cannot favour one.
def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000, random_state=SEED, class_weight="balanced"
        ),
    )


def load_records():
    with open(DATA / "proteins.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_split(name, accessions):
    with open(DATA / "splits" / f"{name}.csv", encoding="utf-8") as f:
        assignment = {row["accession"]: row["split"] for row in csv.DictReader(f)}
    return np.array([assignment[a] == "test" for a in accessions])


def evaluate(X, y, is_test, feature_name, split_name):
    model = make_model()
    model.fit(X[~is_test], y[~is_test])
    predicted = model.predict(X[is_test])
    truth = y[is_test]
    return {
        "features": feature_name,
        "split": split_name,
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "n_train": int((~is_test).sum()),
        "n_test": int(is_test.sum()),
        "_truth": truth,
        "_predicted": predicted,
    }


def main():
    records = load_records()
    accessions = [r["accession"] for r in records]
    y = np.array([r["label"] for r in records])
    print(f"{len(records)} proteins, {len(set(y))} compartments")

    cached = np.load(DATA / "embeddings" / "esm2_t12_35M.npz", allow_pickle=True)
    assert list(cached["accessions"]) == accessions, "embedding order does not match"

    features = {
        "composition (21-d)": np.vstack([featurize(r["sequence"]) for r in records]),
        "ESM-2 35M (480-d)": cached["embeddings"],
    }
    splits = {name: load_split(name, accessions) for name in ("random", "clustered")}

    results = []
    for feature_name, X in features.items():
        for split_name, is_test in splits.items():
            print(f"  fitting {feature_name} on {split_name} split...", flush=True)
            results.append(evaluate(X, y, is_test, feature_name, split_name))

    # A model that always guesses the largest class, for scale.
    is_test = splits["clustered"]
    dummy = DummyClassifier(strategy="most_frequent").fit(y[~is_test], y[~is_test])
    dummy_acc = accuracy_score(y[is_test], dummy.predict(y[is_test].reshape(-1, 1)))

    print("\n" + "=" * 72)
    print(f"{'features':<22}{'split':<12}{'accuracy':>10}{'macro-F1':>11}")
    print("=" * 72)
    for r in results:
        print(f"{r['features']:<22}{r['split']:<12}"
              f"{r['accuracy']:>10.3f}{r['macro_f1']:>11.3f}")
    print("-" * 72)
    print(f"{'majority-class guess':<34}{dummy_acc:>10.3f}{'--':>11}")

    print("\n" + "=" * 72)
    print("THE RESULT: what each feature set loses when the split stops leaking")
    print("=" * 72)
    drops = {}
    for feature_name in features:
        rnd = next(r for r in results if r["features"] == feature_name and r["split"] == "random")
        clu = next(r for r in results if r["features"] == feature_name and r["split"] == "clustered")
        d_f1 = clu["macro_f1"] - rnd["macro_f1"]
        d_acc = clu["accuracy"] - rnd["accuracy"]
        drops[feature_name] = {"macro_f1": d_f1, "accuracy": d_acc}
        print(f"  {feature_name:<22} macro-F1 {rnd['macro_f1']:.3f} -> {clu['macro_f1']:.3f}"
              f"   ({d_f1:+.3f}, {100 * d_f1 / rnd['macro_f1']:+.1f}%)")

    def macro(feature_name, split_name):
        return next(r["macro_f1"] for r in results
                    if r["features"] == feature_name and r["split"] == split_name)

    comp, esm = list(features)
    gap_random = macro(esm, "random") - macro(comp, "random")
    gap_clustered = macro(esm, "clustered") - macro(comp, "clustered")
    print(f"\n  ESM advantage over composition:")
    print(f"    random split     {gap_random:+.3f} macro-F1")
    print(f"    clustered split  {gap_clustered:+.3f} macro-F1")

    print("\n" + "=" * 72)
    print("PER-CLASS, clustered split (the honest one)")
    print("=" * 72)
    for r in results:
        if r["split"] != "clustered":
            continue
        print(f"\n{r['features']}:")
        print(classification_report(r["_truth"], r["_predicted"], digits=3, zero_division=0))

    (DATA / "results.json").write_text(
        json.dumps(
            {
                "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
                "drops": drops,
                "esm_advantage": {"random": gap_random, "clustered": gap_clustered},
                "majority_class_accuracy": dummy_acc,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
