# Day 004 · Debug 分类模型的 batch accuracy · 2026-10-04

日期：2026-10-04（北京时间） · 难度：入门进阶

预计用时：20 分钟，读题约 4 分钟，追踪并修复代码约 11 分钟，补充与运行检查约 5 分钟。

## 1. 今日 Coding Skill

**Debug NumPy 分类评估中的轴选择与 batch 对齐。**

延续 Day 003 的 `axis` 推理，今天阅读一段已有代码，定位并修复分类准确率计算中的错误。新增的重点只有一个：分清 logits 的样本轴和类别轴。

## 2. 必要背景

图像分类器或文本分类模型在验证阶段通常输出 logits 矩阵，形状为 `(batch_size, num_classes)`。每行对应一个样本，每列对应一个类别；评估代码要为每个样本选出分数最高的类别，再和真实标签比较并统计准确率。函数输入是 logits 和一维整数标签，输出是 0 到 1 之间的准确率比例。`argmax` 的轴选错时，代码可能报 shape 错误，也可能因 broadcasting 仍能运行却算错指标。这里练习的是分类评估步骤，不需要实现模型或 loss。

## 3. 小例子

```text
某个样本的 logits：[0.1, 0.8, 0.1]
类别编号：            0    1    2
真实标签：1

这个样本预测正确。
```

检查代码时要确认每个样本最终只对应一个预测类别，并且预测与同一位置的标签比较。

## 4. Coding 任务

阅读并 Debug 下面的评估函数：

```python
import numpy as np


def batch_accuracy(logits, targets):
    predicted = np.argmax(logits, axis=0)
    return np.mean(predicted == targets)
```

1. 用注释写下 `logits`、`predicted`、`targets` 各自的 shape。
2. 修复函数，让它对每个样本产生一个类别预测，再计算 batch accuracy；尽量只改必要的代码。
3. 运行下面两个验收样例，留意其中一个 shape 会让错误实现报错，另一个可能静默地产生错误结果。

输入约定：`logits` 是 shape `(B, C)` 的 NumPy 数组，`targets` 是 shape `(B,)` 的类别编号数组，`B >= 1` 且 `C >= 2`。不要求处理非法输入或空 batch。

在仓库根目录运行：

```bash
uv run python day004/solution.py
```

## 5. 验收标准

- 返回一个 0 到 1 之间的数值，表示预测正确的样本比例。
- 每个样本的预测只从它自己的类别分数中产生；输出 shape 错误或因 broadcasting 静默错算都不通过。
- 两个样例都应返回 `1.0`：

```python
# 样例 1：batch 大小恰好等于类别数，错误轴可能不报错但会算错。
logits = np.array([
    [0.9, 0.1, 0.0],
    [0.0, 0.8, 0.2],
    [0.1, 0.8, 0.1],
])
targets = np.array([0, 1, 1])
assert batch_accuracy(logits, targets) == 1.0

# 样例 2：单样本 batch，检查一维 broadcasting 是否掩盖轴错误。
logits = np.array([[0.1, 0.8, 0.1]])
targets = np.array([1])
assert batch_accuracy(logits, targets) == 1.0
```

完成后告诉我实际用时。卡住时说 `hint`，我每次只给一个小提示。
