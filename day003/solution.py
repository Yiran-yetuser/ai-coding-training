import numpy as np


def center_rows(features):
    """Subtract each row's mean and return a new array."""
    # TODO: 计算每行均值，并保持可用于 broadcasting 的 shape。
    feature_mean = np.mean(features, axis=1, keepdims=True)
    # TODO: 返回中心化后的新数组。
    return features - feature_mean
    raise NotImplementedError


# 样例 1：按行中心化，而不是使用整个矩阵的均值。
features = np.array([[1.0, 3.0, 5.0], [10.0, 14.0, 18.0]])
original = features.copy()
centered = center_rows(features)
np.testing.assert_allclose(centered, [[-2.0, 0.0, 2.0], [-4.0, 0.0, 4.0]])
np.testing.assert_allclose(centered.mean(axis=1), [0.0, 0.0])
assert np.array_equal(features, original)
assert not np.shares_memory(centered, features)

# 样例 2：空 batch 仍保留二维 shape。
empty = np.empty((0, 3))
assert center_rows(empty).shape == (0, 3)
