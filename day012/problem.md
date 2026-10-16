# Day 012 · 为分类评估构造混淆矩阵 · 2026-10-12

日期：2026-10-12（北京时间） · 难度：入门进阶

预计用时：20 分钟，读题与标注输入约 4 分钟，编码约 11 分钟，运行检查约 5 分钟。

## 1. 今日 Coding Skill

**NumPy 配对索引与重复位置的向量化累加。**

## 2. 必要背景

分类模型验证时，accuracy 只能给出总体正确比例，混淆矩阵还能显示哪些类别经常被互相错分。矩阵第 `i` 行、第 `j` 列记录真实类别 `i` 被预测成类别 `j` 的样本数，这是 scikit-learn 等分类评估流程使用的常见表示法。[scikit-learn 混淆矩阵文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) 输入是等长的一维真实标签和预测标签，以及类别总数；输出是固定 shape 的整数矩阵。本题只负责统计标签对，不从 logits 生成预测，也不计算 precision 或 recall。

## 3. 小例子

```text
真实标签： [0, 1, 1]
预测标签： [0, 0, 1]

矩阵的行是真实类别、列是预测类别：
[[1, 0],
 [1, 1]]
```

## 4. Coding 任务

在 `solution.py` 中完成 `confusion_counts(y_true, y_pred, num_classes)`：

- `y_true` 和 `y_pred` 是 shape 相同的一维 NumPy 整数数组；标签范围为 `0` 到 `num_classes - 1`，且 `num_classes >= 1`。
- 返回 shape 为 `(num_classes, num_classes)`、dtype 为 `np.int64` 的新数组；不修改输入。
- 不写 Python 循环，不调用 scikit-learn。注意一个真实/预测类别对可能出现多次，每个样本都必须累加。

### 语法提示

- `np.zeros((C, C), dtype=np.int64)` 创建固定大小的计数矩阵。
- NumPy 的 `np.add.at(array, indices, 1)` 会逐次累加；多个样本落到同一索引时，每次都会计数。[NumPy `ufunc.at` 文档](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html)
- 二维配对索引可以用 `(row_indices, column_indices)` 表示：两个数组按位置配对，不构成笛卡尔积。

在仓库根目录运行：

```bash
uv run python day012/solution.py
```

## 5. 验收标准

- 行表示真实类别，列表示预测类别；每个样本对对应位置贡献 1 次。
- 固定输出 shape 与整数 dtype；未出现的类别仍保留全零行/列；输入内容保持不变。
- 两个样例均通过，尤其检查重复类别对不能漏计。

```python
# 样例 1：类别对 (1, 2) 重复两次；类别 3 没有出现在标签中。
y_true = np.array([0, 1, 1, 2, 2])
y_pred = np.array([0, 2, 2, 1, 2])
expected = np.array([
    [1, 0, 0, 0],
    [0, 0, 2, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
], dtype=np.int64)
result = confusion_counts(y_true, y_pred, num_classes=4)
np.testing.assert_array_equal(result, expected)

# 样例 2：单个样本也保持完整的类别矩阵 shape。
one = confusion_counts(np.array([1]), np.array([0]), num_classes=3)
assert one.shape == (3, 3)
assert one[1, 0] == 1 and one.sum() == 1
```

完成后告诉我实际用时。卡住时说 `hint`，我会先给一个很小的提示。