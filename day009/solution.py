"""Day 009 starter: preserve feature-label pairs while shuffling."""

import numpy as np


def shuffle_training_pairs(features, labels, seed=0):
    # TODO: 生成一次样本随机顺序，并保持特征和标签配对。
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(features))
    shuffled_features = features[perm]
    shuffled_labels = labels[perm]
    return shuffled_features, shuffled_labels


# 样例 1：每行首列是样本 ID，标签也使用同一个 ID。
features = np.array([[10.0, 0.1], [20.0, 0.2], [30.0, 0.3], [40.0, 0.4]])
labels = np.array([10, 20, 30, 40])
features_before = features.copy()
labels_before = labels.copy()
shuffled_features, shuffled_labels = shuffle_training_pairs(features, labels, seed=0)
assert shuffled_features.shape == features.shape
assert shuffled_labels.shape == labels.shape
np.testing.assert_array_equal(shuffled_features[:, 0], shuffled_labels)
np.testing.assert_array_equal(np.sort(shuffled_labels), labels)
features_again, labels_again = shuffle_training_pairs(features, labels, seed=0)
np.testing.assert_array_equal(shuffled_features, features_again)
np.testing.assert_array_equal(shuffled_labels, labels_again)
assert np.array_equal(features, features_before)
assert np.array_equal(labels, labels_before)


# 样例 2：单样本 batch 保持原 shape 和配对关系。
single_features = np.array([[7.0, 8.0]])
single_labels = np.array([3])
single_x, single_y = shuffle_training_pairs(single_features, single_labels, seed=5)
assert single_x.shape == (1, 2)
assert single_y.shape == (1,)
np.testing.assert_array_equal(single_x, single_features)
np.testing.assert_array_equal(single_y, single_labels)
