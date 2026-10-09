# Day 009 · 修复训练数据打乱后的特征标签错位 · 2026-10-09

日期：2026-10-09（北京时间） · 难度：入门进阶

预计用时：20 分钟，读题约 4 分钟，Debug 与修改约 11 分钟，运行验收约 5 分钟。

## 1. 今日 Coding Skill

**NumPy 配对数据索引与可复现随机排列。**

## 2. 必要背景

训练模型前常会打乱样本顺序，让每个 batch 不总是按数据原有顺序出现。每条特征和它的标签必须保持配对；如果两者被独立打乱，程序通常仍能运行，但模型会拿错标签训练。函数接收特征矩阵 `features`（shape `(N, F)`）、标签数组 `labels`（shape `(N,)`）和随机种子，返回同一随机顺序下的两份数组。这里仅练习训练数据的随机重排，不实现优化器或完整训练循环。

## 3. 小例子

```text
样本 ID：       10   20   30
特征首列：      10   20   30
对应标签：       0    1    2

若随机顺序是 [2, 0, 1]：
特征首列变为 [30, 10, 20]
标签也要按该顺序变为 [ 2,  0,  1]
```

## 4. Coding 任务

阅读并修复这个训练数据打乱函数：

```python
import numpy as np


def shuffle_training_pairs(features, labels, seed=0):
    rng = np.random.default_rng(seed)
    shuffled_features = features[rng.permutation(len(features))]
    shuffled_labels = labels[rng.permutation(len(labels))]
    return shuffled_features, shuffled_labels
```

修复后保留函数名和参数。要求同一 `seed` 可重复得到相同顺序，不修改输入数组，不写逐样本的 Python 循环。

输入约定：`features` 是 shape `(N, F)` 的 NumPy 数组，`labels` 是 shape `(N,)` 的 NumPy 数组，且 `N >= 1`、`F >= 1`、两者样本数相同；`seed` 是整数。本题不要求检查非法 shape 或类型。

### 语法提示

- `np.random.default_rng(seed)` 会创建由 `seed` 控制的随机数生成器。
- `rng.permutation(n)` 会返回 `0` 到 `n-1` 的随机排列索引。
- NumPy 数组可以用整数索引数组选择并重排行，例如 `array[indices]`。

在仓库根目录运行：

```bash
uv run python day009/solution.py
```

## 5. 验收标准

- 特征和标签使用同一随机顺序，所有样本恰好出现一次。
- 相同 seed 的两次调用结果相同；输入数组保持不变。
- 两个样例均通过：一个检查多样本配对与可复现性，一个检查单样本 batch 的 shape。

完成后告诉我实际用时。卡住时说 `hint`，我每次只给一个小提示。
