"""Day 007 starter: debug row-wise softmax stability."""

import numpy as np


def softmax_rows(logits):
    # TODO: 修复数值稳定性问题，保留输入且返回每行概率。
    max_val = np.max(logits, axis=1, keepdims=True)

    exp_logits = np.exp(logits - max_val)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


# 样例 1：普通 logits。
ordinary = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
ordinary_before = ordinary.copy()
ordinary_probs = softmax_rows(ordinary)
assert ordinary_probs.shape == ordinary.shape
assert np.all(np.isfinite(ordinary_probs))
assert np.all((ordinary_probs >= 0.0) & (ordinary_probs <= 1.0))
np.testing.assert_allclose(ordinary_probs.sum(axis=1), [1.0, 1.0])
assert np.array_equal(ordinary_probs.argmax(axis=1), ordinary.argmax(axis=1))
assert np.array_equal(ordinary, ordinary_before)

# 样例 2：有限但幅度很大的 logits。
extreme = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
    extreme_probs = softmax_rows(extreme)
assert np.all(np.isfinite(extreme_probs))
np.testing.assert_allclose(extreme_probs.sum(axis=1), [1.0, 1.0])
assert np.array_equal(extreme_probs.argmax(axis=1), extreme.argmax(axis=1))
