"""Step 3: turn each protein into an ESM-2 embedding, once, and cache it.

ESM-2 is a transformer trained on ~250 million protein sequences by masking a
residue and predicting it from context. It never sees a function label. We use
it purely as a feature extractor: run a sequence through it, average the
per-residue outputs into one vector, and hand that vector to the same logistic
regression the letter-counting baseline uses.

The 35M-parameter checkpoint is the smallest ESM-2 and runs on a laptop CPU.
Embeddings are written to disk and never recomputed.
"""

import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
PROTEINS = Path("data/proteins.csv")
OUT = Path("data/embeddings/esm2_t12_35M.npz")

# Batches are built to a token budget rather than a fixed sequence count, so a
# batch of short proteins holds more of them than a batch of long ones.
MAX_TOKENS_PER_BATCH = 8000


def load():
    with open(PROTEINS, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def batches(indices, sequences):
    """Group sequence indices into batches under the token budget.

    Indices arrive sorted by length, so each batch holds similarly-sized
    sequences and little compute is wasted on padding.
    """
    batch, longest = [], 0
    for i in indices:
        candidate = max(longest, len(sequences[i]) + 2)  # +2 for <cls> and <eos>
        if batch and candidate * (len(batch) + 1) > MAX_TOKENS_PER_BATCH:
            yield batch
            batch, longest = [i], len(sequences[i]) + 2
        else:
            batch.append(i)
            longest = candidate
    if batch:
        yield batch


def main():
    limit = int(os.environ.get("LIMIT", "0"))
    records = load()
    if limit:
        records = records[:limit]
    elif OUT.exists():
        print(f"{OUT} already exists -- delete it to recompute")
        return

    sequences = [r["sequence"] for r in records]
    print(f"{len(records)} proteins, {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    dim = model.config.hidden_size
    out = np.zeros((len(records), dim), dtype=np.float32)
    print(f"embedding dimension: {dim}")

    order = sorted(range(len(records)), key=lambda i: len(sequences[i]))
    todo = list(batches(order, sequences))
    started = time.time()
    done = 0

    with torch.no_grad():
        for n, batch in enumerate(todo, 1):
            encoded = tokenizer(
                [sequences[i] for i in batch],
                padding=True,
                return_tensors="pt",
            )
            hidden = model(**encoded).last_hidden_state

            # Average over real residues only. The attention mask covers padding;
            # positions 0 and the final token are <cls> and <eos>, which are
            # bookkeeping rather than residues and would skew the average.
            mask = encoded["attention_mask"].clone()
            lengths = mask.sum(1)
            mask[:, 0] = 0
            mask[torch.arange(mask.size(0)), lengths - 1] = 0
            mask = mask.unsqueeze(-1).float()

            pooled = (hidden * mask).sum(1) / mask.sum(1)
            out[batch] = pooled.numpy()

            done += len(batch)
            if n % 20 == 0 or n == len(todo):
                elapsed = time.time() - started
                rate = done / elapsed
                remaining = (len(records) - done) / rate if rate else 0
                print(
                    f"  batch {n}/{len(todo)}  {done}/{len(records)} proteins"
                    f"  {rate:.0f}/s  eta {remaining/60:.1f} min",
                    flush=True,
                )

    if limit:
        print(f"\nsmoke test OK: {out.shape}, no NaNs: {not np.isnan(out).any()}")
        print(f"first vector, first 6 dims: {np.round(out[0][:6], 4)}")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        accessions=np.array([r["accession"] for r in records]),
        embeddings=out,
    )
    print(f"\nsaved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"total time: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
