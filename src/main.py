"""Milestone 0: a toy, end-to-end protein function classifier.

The data and the labels in this file are synthetic. The purpose is to prove the
pipeline runs from one end to the other -- sequence -> features -> model ->
metrics -- not to produce a result. The accuracy printed here has no scientific
meaning; see the README for why.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

RANDOM_STATE = 0

# The 20 standard amino acids in a fixed order. Every composition vector below
# uses this order, so index 0 is always alanine, index 1 always cysteine, etc.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

CLASS_NAMES = ["soluble-like", "membrane-like"]

# 20 synthetic sequences. Class 1 is drawn from a hydrophobic ("greasy") residue
# pool and class 0 from a polar/charged pool -- a caricature of the real signal
# that separates membrane proteins from soluble ones, since a cell membrane is
# greasy and the proteins embedded in it are built to match. Every sequence
# starts with M (methionine), as real proteins do; it is shared across both
# classes so it carries no class information.
SEQUENCES = [
    (  # class 1
        "MAMVFLLACWLGAVPMLGLGIALLICTALFVVVPLATMTSTFASSGVAWFVSGMYFFLSATL"
        "FFIAVFSCWVIILAFSIM"
    ),
    (  # class 1
        "MVLFMVFLSYMGYGLVSSLCFLVVCAGIYWTAMAMGLTGVMSILVMTFAVFIIALFFFYICG"
        "ICLIAVYVGYAVTAAVSILSLCAVC"
    ),
    (  # class 0
        "MDQNSKCDTTCTGRTCETSRTCRYQTNTESYKNDCTSGNTSDEQDQCERKGTTHKRCEHERP"
        "TTD"
    ),
    (  # class 0
        "MQTTRDRKTEPHTTTGDCCYETNDRNEGYSQTKYKRSPTSDNDEEDRQDGYHSYGENTGCEQ"
        "RDRRRQRTSY"
    ),
    (  # class 1
        "MVTLGMAYSFTWICMILFFWTCMVFIFTMAAAAVSVAFMLVWAMYIIGCASAVIF"
    ),
    (  # class 0
        "MDKRRQQQSSSQEKPSSRTYTERESRHNTPTDRDPPQQKKRKSSDYTSSERKCRRQPSEEER"
        "SDTTNSDTKNGEETPGETKHNRPCKKGKTD"
    ),
    (  # class 1
        "MGWLFCFAIFAPMILIPMAAPVLGLVMFFTLYASIMYLVFFFVFVIWFGIAIAAFCFFMTSC"
        "AFLALLFM"
    ),
    (  # class 1
        "MMLLMGLGVFSIICPMLLMALAAIPLVITGCAMVLPILVLFLTAY"
    ),
    (  # class 1
        "MLWIAVAIWMFILWAPFWSSATFMLPGFMFVCYTIIIFPIFGAAVTMPAGLLGLFAIWIVIV"
        "A"
    ),
    (  # class 1
        "MFITGWCTFLAVIAPIILGFYGIFWGLYVLVPMAYTICFMLLYAAPMVIGY"
    ),
    (  # class 0
        "MQHRKNGGGQCQHTTGKRKRDQEYKGEDPYDTTTRCEEKRTNYETRKSQKDDTSSTQRRKYD"
        "SHDCTTCQTER"
    ),
    (  # class 0
        "MSDESRSTQHSEEQDGTCKEHRKSKTTYSPNKETDTGPKYQRRQRTCKGEPDTYTHNNRYDD"
        "RRTTKCSSHGSYQD"
    ),
    (  # class 0
        "MREQGERTRSKDKDCSGKNSSETSKECKSNNGNEQGNKTTTPDNRYYTDNRKDRNNQT"
    ),
    (  # class 0
        "MGTDTQPEKKRRQKKHDSQDTDKKTQDEEESRRDNKTYRERCSQTNTKQQNQGKQRTTRKDQ"
        "DKDQDRETTS"
    ),
    (  # class 0
        "MEGCKDKTNRRRDTPRERTYQDSNDGSNTNDEHSKEPKNRKRDKTGNGERKHRSQCGYSRDN"
        "THSTKTSRDEKK"
    ),
    (  # class 1
        "MYGVWVSYVVPLGVWILFGILLFIISLLAIGFCVWILSILVIMAGGPGIGWMAAWAGPSMIL"
        "VVVIFFGTLFIACILFLVTI"
    ),
    (  # class 1
        "MALGIFGITIGMGGFCIFSAMVWTIFGAPGVVVMFVMVVVGVAISGFAIFMAWLGPSPTLAF"
        "MLVGGLWWALFTFVGSFMGMASAVIMIL"
    ),
    (  # class 0
        "MGTSRDPEDPESERGDEERESKDKRYTSEHTTYSSCKSRCSQDSSTGTREDRDYREYDCTKT"
        "TDRKGTCTRDPNREEKRHSK"
    ),
    (  # class 1
        "MFFIVVFTTFGTFVVWLYAVIYFVAALATLLAACYGFIAVFFMGSPLVTT"
    ),
    (  # class 0
        "MEKTQNQQNQPHPKCEKCTCDKSQQEQENERRHGCGQRQQRKRHCCDHTTSGNRSGQSETRY"
        "PTGCHTPEDQEKTPTCQNTPSDRKEESKKCD"
    ),
]

LABELS = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0])


def composition(sequence):
    """Fraction of the sequence made up by each amino acid.

    Returns 20 numbers that sum to 1. All ordering information is discarded --
    this is a bag-of-residues, which is exactly why it makes a hard-to-cheat
    baseline later on.
    """
    counts = np.array([sequence.count(aa) for aa in AMINO_ACIDS], dtype=float)
    return counts / counts.sum()


def featurize(sequence):
    """Composition (20 values) with log10 of the sequence length appended.

    Length is logged so that it lands on roughly the same scale as the
    frequencies. Raw length, in the hundreds, would dominate the logistic
    regression's coefficients through sheer magnitude rather than relevance.
    """
    return np.append(composition(sequence), np.log10(len(sequence)))


def build_matrix(sequences):
    """Stack per-sequence feature vectors into an (n_sequences, 21) matrix."""
    return np.vstack([featurize(s) for s in sequences])


def main():
    X = build_matrix(SEQUENCES)
    y = LABELS

    print(f"{len(SEQUENCES)} sequences -> feature matrix {X.shape}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  class {i} ({name}): {(y == i).sum()} sequences")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\ntrain: {len(y_train)} sequences   test: {len(y_test)} sequences")
    print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


if __name__ == "__main__":
    main()
