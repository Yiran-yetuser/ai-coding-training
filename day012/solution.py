"""Day 012 starter: vectorized confusion-matrix counts."""

import numpy as np


def confusion_counts(y_true, y_pred, num_classes):
    """Return counts with rows=true classes and columns=predicted classes."""
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    # TODO: Add one count for every paired (true, predicted) label.
    np.add.at(counts, (y_true, y_pred), 1)
    # TODO: Make sure repeated pairs are all counted.
    return counts
    raise NotImplementedError


# Sample 1: the pair (1, 2) appears twice; class 3 is absent.
y_true = np.array([0, 1, 1, 2, 2])
y_pred = np.array([0, 2, 2, 1, 2])
y_true_before = y_true.copy()
y_pred_before = y_pred.copy()
expected = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 2, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ],
    dtype=np.int64,
)
result = confusion_counts(y_true, y_pred, num_classes=4)
assert result.shape == (4, 4)
assert result.dtype == np.int64
np.testing.assert_array_equal(result, expected)
np.testing.assert_array_equal(y_true, y_true_before)
np.testing.assert_array_equal(y_pred, y_pred_before)

# Sample 2: a single sample still returns the full class matrix.
one = confusion_counts(np.array([1]), np.array([0]), num_classes=3)
assert one.shape == (3, 3)
assert one[1, 0] == 1 and one.sum() == 1
