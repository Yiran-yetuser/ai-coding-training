"""Day 004 starter: debug batch accuracy for classifier logits."""

import numpy as np


def batch_accuracy(logits, targets):
    # TODO: 修复预测类别与 batch 标签的轴对齐问题。
    predicted = np.argmax(logits, axis=1)
    # logits.shape = (B,C)
    # predicted.shape = (B,)
    # targets.shape = (B,)
    return np.mean(predicted == targets)


# 样例 1：batch 大小等于类别数，检查是否静默算错。
logits = np.array(
    [
        [0.9, 0.1, 0.0],
        [0.0, 0.8, 0.2],
        [0.1, 0.8, 0.1],
    ]
)
targets = np.array([0, 1, 1])
assert batch_accuracy(logits, targets) == 1.0

# 样例 2：单样本 batch，检查 broadcasting 是否掩盖错误。
logits = np.array([[0.1, 0.8, 0.1]])
targets = np.array([1])
assert batch_accuracy(logits, targets) == 1.0
