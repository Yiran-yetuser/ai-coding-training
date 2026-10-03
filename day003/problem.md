# Day 003 · 用 NumPy 按样本做中心化 · 2026-10-03

日期：2026-10-03（北京时间） · 难度：入门进阶

预计用时：25 分钟，读题约 4 分钟，编写代码约 16 分钟，运行验证约 5 分钟。

## 1. 今日 Coding Skill

**NumPy shape reasoning、axis、`keepdims` 和 broadcasting。**

前两题使用 Python 列表处理单条结果和变长 batch。今天把输入换成 NumPy 的二维数组，练习按照 batch 中的每一行计算统计量，并把结果广播回原数组。今天只新增“按行操作与形状对齐”这一项能力，不要求实现模型或复杂的数值算法。

## 2. 必要背景

AI 模型通常把一批样本表示为形状 `(batch_size, feature_dim)` 的矩阵，每一行是一条样本的特征。预处理或模型内部有时会先让每条样本的特征围绕 0 分布；本题把这个步骤简化为“减去每行自己的平均值”。这和 Transformer 中 LayerNorm 的中心化步骤有关，但完整的 LayerNorm 还会计算方差并做缩放、平移，本题不实现那些部分。你的函数接收一个二维 NumPy 数组，返回同样 shape 的新数组，并且不修改输入。

**这一步在哪里使用？**

- **Transformer / LayerNorm：** 对每个 token 的 hidden-state 向量计算均值，是归一化的一部分；今天只练习其中的按行中心化步骤。[Layer Normalization 原始论文](https://arxiv.org/abs/1607.06450)
- **特征预处理：** batch 中每行代表一个样本时，错误地计算全 batch 的一个均值，会让不同样本互相影响；`axis` 的选择直接决定预处理是否正确。[NumPy `mean` 文档](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)

## 3. 小例子

```text
输入 shape：(2, 3)
[[ 1,  3,  5],
 [10, 14, 18]]

每行均值 shape：(2, 1)
[3, 14]

输出：每行分别减去自己的均值，shape 仍为 (2, 3)
```

注意：这里两行使用不同的均值；不能把整个矩阵只算成一个标量均值。

## 4. Coding 任务

自己创建 `day003/solution.py`，完成下面的函数：

```python
import numpy as np


def center_rows(features):
    """Subtract each row's mean and return a new array."""
    # TODO: 计算每行均值，并保持可用于 broadcasting 的 shape。
    # TODO: 返回中心化后的新数组。
    raise NotImplementedError
```

要求：

1. `features` 是形状 `(B, F)` 的二维 NumPy 数组，其中 `F >= 1`；不需要处理非法类型或三维输入。
2. 每一行减去该行自己的均值，返回 shape 仍为 `(B, F)` 的 NumPy 数组。
3. 使用 NumPy 向量化操作完成，不写逐元素的 Python 双重循环；不要调用现成的标准化或 LayerNorm 实现。
4. 不修改输入数组；返回结果应当是独立数组。整数输入可以产生浮点结果，不需要强制指定某个浮点 dtype。
5. 用给定样例验证后，再用断点或打印检查 `features.shape`、均值数组的 shape 和结果的 shape。

在仓库根目录运行：

```bash
uv run python day003/solution.py
```

## 5. 验收标准

- 输出与输入 shape 完全相同。
- 每一行输出的均值接近 `0`；不同输入行必须使用各自的均值。
- 输入内容保持不变，输出不与输入共享底层内存。
- 边界：空 batch（例如 shape `(0, 3)`）应返回 shape `(0, 3)` 的结果；本题不要求处理特征数为 `0` 的数组。

```python
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
```

完成后提交代码，并告诉我实际用时。卡住时说 `hint`，我每次只给一个小提示。
