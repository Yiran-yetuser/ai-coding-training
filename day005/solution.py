"""Day 005 starter: accuracy over non-padding token labels."""

import numpy as np


def masked_token_accuracy(predictions, targets, ignore_index=-100):
    # predictions 和 targets 都是二维整数 NumPy 数组。
    # TODO: 只在有效标签位置计算准确率。
    acc = 0.0
    pred_masked = predictions[targets != ignore_index]
    tar_masked = targets[targets != ignore_index]
    # TODO: 没有有效标签时返回题目约定的结果。
    if tar_masked.size == 0:
        return 0.0
    acc = (pred_masked == tar_masked).mean()
    return acc
    raise NotImplementedError


# 样例 1：padding 不计入 accuracy，预期为 0.75。
predictions = np.array([[2, 1, 7], [1, 2, 7]])
targets = np.array([[2, 0, -100], [1, 2, -100]])
assert masked_token_accuracy(predictions, targets) == 0.75

# 样例 2：没有有效 token，预期为 0.0。
predictions = np.array([[0, 1], [2, 1]])
targets = np.full((2, 2), -100)
assert masked_token_accuracy(predictions, targets) == 0.0
