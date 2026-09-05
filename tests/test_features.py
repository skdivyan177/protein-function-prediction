import numpy as np

from main import SEQUENCES, composition, featurize


def test_composition_is_a_probability_distribution():
    for sequence in SEQUENCES:
        vector = composition(sequence)
        assert vector.shape == (20,)
        assert (vector >= 0).all()
        assert np.isclose(vector.sum(), 1.0)


def test_featurize_appends_log_length():
    sequence = SEQUENCES[0]
    vector = featurize(sequence)
    assert vector.shape == (21,)
    # The composition block still sums to 1. The appended length sits outside
    # that constraint, which is why the assertion is on the first 20 entries.
    assert np.isclose(vector[:20].sum(), 1.0)
    assert np.isclose(vector[20], np.log10(len(sequence)))
