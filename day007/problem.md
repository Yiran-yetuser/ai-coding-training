# Day 007 · Debug 分类器的 softmax 数值稳定性 · 2026-10-07

日期：2026-10-07（北京时间） · 难度：入门进阶

预计用时：25 分钟，读题约 4 分钟，Debug 约 15 分钟，验收约 6 分钟。

## 1. 今日 Coding Skill

**Debug 数值稳定性：处理指数运算的溢出与下溢。**

## 2. 必要背景

分类模型通常输出 shape 为 `(B, C)` 的 logits：每行对应一个样本，每列对应一个类别。推理阶段可以把 logits 转成每行总和为 1 的概率，方便后续阈值判断或展示置信度；训练时通常直接把 logits 交给交叉熵 loss。直接对很大的有限 logits 做指数运算，仍可能得到无穷大或全零，最终概率会出现 `NaN`。今天只修复这个 NumPy 后处理函数的数值稳定性，不实现模型或 loss。

## 3. 小例子

```text
普通 logits：[1.0, 2.0, 3.0] → 三个有限概率，且总和为 1
极端 logits：[1000, 1001, 1002] → 也必须得到有限概率
```

每行概率最大的类别应与原 logits 中最大值所在类别相同。

## 4. Coding 任务

阅读并 Debug 下面的分类后处理代码，使它能稳定处理本题约定的二维 logits：

```python
import numpy as np


def softmax_rows(logits):
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)
```

输入是非空、二维、有限的浮点 NumPy 数组，shape 为 `(B, C)`；不要求处理非法类型、空维度或 `NaN/inf` 输入。保留函数名和参数，不修改输入，不使用 Python 逐元素循环。

在仓库根目录运行：

```bash
uv run python day007/solution.py
```

## 5. 验收标准

- 输出 shape 与输入相同，所有值有限且在 0～1 之间，每行之和约为 1。
- 每行最大概率对应的类别与 logits 的最大类别一致，输入保持不变。
- 两个样例均通过：普通 logits；包含大正数和大负数的 logits。

完成后告诉我实际用时。卡住时说 `hint`，我每次只给一个小提示。
