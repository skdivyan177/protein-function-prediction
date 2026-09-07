"""Step 1: build a subcellular localization dataset from UniProt.

Downloads manually-reviewed (Swiss-Prot) proteins for human, mouse and rat,
keeps only those whose subcellular location is backed by experimental evidence
and is unambiguous, and writes a clean CSV.

Nothing here is committed to git. The script is the reproducible artifact; the
data it produces is not. Re-running reproduces the dataset, and
data/provenance.json records exactly what was pulled and when.
"""

import gzip
import json
import re
import ssl
import urllib.parse
import urllib.request
import certifi

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATA = Path("data")
RAW = DATA / "raw" / "uniprot.tsv.gz"
OUT = DATA / "proteins.csv"
PROVENANCE = DATA / "provenance.json"

# Human, mouse, rat. Orthologs across these three are highly similar, which is
# precisely the homology we want the clustered split to have to deal with.
ORGANISMS = {"9606": "human", "10090": "mouse", "10116": "rat"}

QUERY = (
    "reviewed:true AND ("
    + " OR ".join(f"organism_id:{t}" for t in ORGANISMS)
    + ") AND cc_scl_term:*"
)
FIELDS = "accession,organism_id,length,sequence,cc_subcellular_location"
URL = (
    "https://rest.uniprot.org/uniprotkb/stream?"
    + urllib.parse.urlencode(
        {"query": QUERY, "fields": FIELDS, "format": "tsv", "compressed": "true"}
    )
)

# UniProt writes locations as free text with a controlled vocabulary buried in
# it. We collapse the fine-grained terms into a handful of top-level
# compartments -- the "rooms of the cell" a protein can work in.
COMPARTMENTS = {
    "Nucleus": ("nucleus", "nucleoplasm", "nucleolus", "nuclear"),
    "Cytoplasm": ("cytoplasm", "cytosol"),
    "Cell membrane": ("cell membrane", "plasma membrane"),
    "Mitochondrion": ("mitochondri",),
    "Secreted": ("secreted",),
    "Endoplasmic reticulum": ("endoplasmic reticulum",),
    "Golgi apparatus": ("golgi",),
    "Lysosome": ("lysosom",),
    "Peroxisome": ("peroxisom",),
}

# ECO:0000269 is UniProt's code for "experimentally determined, with a paper to
# cite". Anything else is inferred or predicted, which we do not want as a label.
EXPERIMENTAL = "ECO:0000269"

# The python.org macOS build ships without CA certificates wired up, so we
# point at certifi explicitly rather than depend on a machine-local fix.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 50, 1000


def download():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if RAW.exists():
        print(f"using cached {RAW} ({RAW.stat().st_size / 1e6:.1f} MB)")
        return
    print("downloading from UniProt (this takes a minute)...")
    with urllib.request.urlopen(URL, timeout=300, context=SSL_CONTEXT) as r, open(RAW, "wb") as f:
        f.write(r.read())
    print(f"saved {RAW} ({RAW.stat().st_size / 1e6:.1f} MB)")


def compartments_with_evidence(cc_text):
    """Top-level compartments that this entry claims with experimental evidence.

    Returns a set. A protein is only usable as a single-label training example
    if that set has exactly one member.
    """
    body = cc_text.split("SUBCELLULAR LOCATION:", 1)[-1]
    body = re.split(r"Note=", body)[0]  # free-form notes mention other places
    found = set()
    for statement in re.split(r"\.\s+", body):
        if EXPERIMENTAL not in statement:
            continue
        low = statement.lower()
        for name, keys in COMPARTMENTS.items():
            if any(k in low for k in keys):
                found.add(name)
    return found


def main():
    download()

    rows, reasons = [], Counter()
    with gzip.open(RAW, "rt", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                reasons["malformed row"] += 1
                continue
            acc, org, _length, seq, cc = parts[0], parts[1], parts[2], parts[3], parts[4]

            if not (MIN_LEN <= len(seq) <= MAX_LEN):
                reasons["length outside 50-1000"] += 1
                continue
            if not set(seq) <= STANDARD_AA:
                reasons["non-standard residues"] += 1
                continue

            found = compartments_with_evidence(cc)
            if not found:
                reasons["no experimentally-backed compartment"] += 1
                continue
            if len(found) > 1:
                reasons["multiple compartments (ambiguous)"] += 1
                continue

            rows.append((acc, ORGANISMS.get(org, org), len(seq), found.pop(), seq))

    counts = Counter(r[3] for r in rows)
    by_organism = Counter(r[1] for r in rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("accession,organism,length,label,sequence\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    PROVENANCE.write_text(
        json.dumps(
            {
                "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "query": QUERY,
                "fields": FIELDS,
                "kept": len(rows),
                "class_counts": dict(counts.most_common()),
                "by_organism": dict(by_organism.most_common()),
                "discarded": dict(reasons.most_common()),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nkept {len(rows)} proteins -> {OUT}")
    print("\ndiscarded:")
    for reason, n in reasons.most_common():
        print(f"  {n:>7,}  {reason}")
    print("\nclass counts:")
    total = sum(counts.values())
    for name, n in counts.most_common():
        print(f"  {n:>7,}  {100 * n / total:5.1f}%  {name}")
    print("\nby organism:")
    for name, n in by_organism.most_common():
        print(f"  {n:>7,}  {name}")


if __name__ == "__main__":
    main()
