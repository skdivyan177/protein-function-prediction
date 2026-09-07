"""Step 2: build a homology-aware split and a random split, and measure the gap.

A random split scatters homologs -- near-identical sequences produced by
evolution copying an ancestor -- across train and test, so a model can score
well by recognising a near-duplicate. This script builds both kinds of split
and then measures, for each, how many test proteins still have a close relative
sitting in the training set.

That measurement is the point of the whole project. Everything downstream just
runs the same models over these two splits.
"""

import csv
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("data")
PROTEINS = DATA / "proteins.csv"
WORK = DATA / "work"
SPLITS = DATA / "splits"
MMSEQS = Path("tools/mmseqs/bin/mmseqs")

SEED = 0
TEST_FRACTION = 0.3

# Two sequences are treated as related if they align over at least 80% of their
# length at 30% or better identity. 30% is the conventional stringent threshold
# in this literature -- below it, sequence similarity stops being reliable
# evidence of a shared ancestor.
MIN_SEQ_ID = 0.3
COVERAGE = 0.8


def load():
    with open(PROTEINS, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_fasta(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f">{r['accession']}\n{r['sequence']}\n")


def run(args):
    result = subprocess.run(
        [str(MMSEQS)] + args, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"mmseqs failed:\n{result.stdout[-2000:]}{result.stderr[-2000:]}")
    return result


def cluster(records):
    """Group sequences into families with MMseqs2. Returns {accession: cluster_id}."""
    write_fasta(WORK / "all.fasta", records)
    run([
        "easy-cluster", str(WORK / "all.fasta"), str(WORK / "clust"), str(WORK / "tmp"),
        "--min-seq-id", str(MIN_SEQ_ID), "-c", str(COVERAGE), "--cov-mode", "0", "-v", "1",
    ])
    membership = {}
    with open(WORK / "clust_cluster.tsv", encoding="utf-8") as f:
        for line in f:
            representative, member = line.rstrip("\n").split("\t")
            membership[member] = representative
    return membership


def clustered_split(records, membership, rng):
    """Assign whole families to train or test, never splitting one across both.

    Families are grouped by their dominant label first so that every compartment
    still shows up on both sides in roughly the right proportion.
    """
    families = defaultdict(list)
    for r in records:
        families[membership[r["accession"]]].append(r)

    by_label = defaultdict(list)
    for members in families.values():
        dominant = Counter(m["label"] for m in members).most_common(1)[0][0]
        by_label[dominant].append(members)

    test = set()
    for label, fams in by_label.items():
        fams.sort(key=lambda f: f[0]["accession"])
        rng.shuffle(fams)
        target = TEST_FRACTION * sum(len(f) for f in fams)
        taken = 0
        for fam in fams:
            if taken >= target:
                break
            test.update(m["accession"] for m in fam)
            taken += len(fam)
    return test


def random_split(records, rng):
    """The naive split: shuffle within each class and take 30%. This is the one
    that leaks, and reproducing it faithfully is the point of including it."""
    test = set()
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)
    for label, rows in by_label.items():
        rows = sorted(rows, key=lambda r: r["accession"])
        rng.shuffle(rows)
        n_test = round(TEST_FRACTION * len(rows))
        test.update(r["accession"] for r in rows[:n_test])
    return test


def measure_leakage(records, test_ids, name):
    """Fraction of test proteins that have a >=30%-identity relative in train."""
    train = [r for r in records if r["accession"] not in test_ids]
    test = [r for r in records if r["accession"] in test_ids]
    write_fasta(WORK / f"{name}_train.fasta", train)
    write_fasta(WORK / f"{name}_test.fasta", test)
    hits = WORK / f"{name}_hits.m8"
    run([
        "easy-search", str(WORK / f"{name}_test.fasta"), str(WORK / f"{name}_train.fasta"),
        str(hits), str(WORK / "tmp"),
        "--min-seq-id", str(MIN_SEQ_ID), "-c", str(COVERAGE), "--cov-mode", "0",
        "-s", "7.5", "-e", "1e-3", "--format-output", "query,target,fident", "-v", "1",
    ])
    leaked = set()
    with open(hits, encoding="utf-8") as f:
        for line in f:
            query, target, fident = line.rstrip("\n").split("\t")
            if float(fident) >= MIN_SEQ_ID:
                leaked.add(query)
    return len(leaked), len(test)


def save(records, test_ids, name):
    SPLITS.mkdir(parents=True, exist_ok=True)
    with open(SPLITS / f"{name}.csv", "w", encoding="utf-8") as f:
        f.write("accession,split\n")
        for r in records:
            f.write(f"{r['accession']},{'test' if r['accession'] in test_ids else 'train'}\n")


def report(records, test_ids, name):
    counts = Counter(r["label"] for r in records if r["accession"] in test_ids)
    total = Counter(r["label"] for r in records)
    n_test = len(test_ids)
    print(f"\n{name} split: {len(records) - n_test} train / {n_test} test")
    for label, n in total.most_common():
        print(f"    {label:<24} test {counts[label]:>4} / {n:>4}")


def main():
    records = load()
    print(f"{len(records)} proteins loaded")

    print(f"\nclustering at {MIN_SEQ_ID:.0%} identity, {COVERAGE:.0%} coverage...")
    membership = cluster(records)
    sizes = Counter(membership.values())
    print(f"  {len(sizes)} families from {len(records)} proteins")
    print(f"  largest family: {max(sizes.values())} proteins")
    print(f"  singletons: {sum(1 for v in sizes.values() if v == 1)}")

    splits = {
        "clustered": clustered_split(records, membership, random.Random(SEED)),
        "random": random_split(records, random.Random(SEED)),
    }

    summary = {}
    for name, test_ids in splits.items():
        save(records, test_ids, name)
        report(records, test_ids, name)
        leaked, n_test = measure_leakage(records, test_ids, name)
        summary[name] = {
            "n_train": len(records) - n_test,
            "n_test": n_test,
            "test_with_relative_in_train": leaked,
            "leakage_rate": round(leaked / n_test, 4),
        }

    print("\n" + "=" * 62)
    print("LEAKAGE: test proteins with a >=30% identity relative in train")
    print("=" * 62)
    for name, s in summary.items():
        print(f"  {name:<10} {s['test_with_relative_in_train']:>5} / {s['n_test']:<5} "
              f"= {s['leakage_rate']:6.1%}")

    (DATA / "splits_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
