# Day 013 · 重构训练数据的 mini-batch 划分 · 2026-10-13

日期：2026-10-13（北京时间） · 难度：入门进阶

预计用时：25 分钟，读题约 4 分钟，重构约 16 分钟，运行检查约 5 分钟。

## 1. 今日 Coding Skill

**用 NumPy 切片重构逐样本代码，处理 batch 边界。** 延续特征／标签配对，今天新增的重点是最后不足一批时的处理。

## 2. 必要背景

使用 mini-batch SGD 训练分类器时，数据准备流程需要把特征和对应标签分成小批，交给模型计算预测和 loss。PyTorch 的 `DataLoader` 用 `batch_size` 控制每批数量，用 `drop_last` 决定是否丢弃最后不足一批的样本。[官方说明](https://docs.pytorch.org/docs/2.14/data.html#torch.utils.data.DataLoader) 输入是已排好顺序的特征矩阵 `(N, F)` 和标签数组 `(N,)`，输出是按原顺序排列的一组 `(batch_features, batch_labels)`。若尾批被无意丢弃，这些样本就不会参与本轮训练。本题只用 NumPy 完成内存中数据的分批，不涉及随机打乱、梯度或完整训练循环。

## 3. 小例子

```text
样本 ID： [10, 20, 30, 40, 50]，batch_size = 2

drop_last=False：[[10, 20], [30, 40], [50]]
drop_last=True： [[10, 20], [30, 40]]

每个 ID 对应的特征和标签必须进入同一批。
若 F=3，单个样本组成的尾批 shape 仍是 (1, 3) 和 (1,)。
```

## 4. Coding 任务

下面的旧代码可以工作，但逐条收集样本，并重复了组装 batch 的代码。读懂它，再在 `solution.py` 中完成 `make_training_batches`，改为**按批遍历、直接切片**。

```python
import numpy as np


def legacy_batches(features, labels, batch_size, drop_last=False):
    batches = []
    x_rows = []
    y_rows = []
    for index in range(len(features)):
        x_rows.append(features[index])
        y_rows.append(labels[index])
        if len(x_rows) == batch_size:
            batches.append((np.array(x_rows), np.array(y_rows)))
            x_rows = []
            y_rows = []
    if x_rows and not drop_last:
        batches.append((np.array(x_rows), np.array(y_rows)))
    return batches
```

- 输入：NumPy 数组 `features` 为 `(N, F)`、`labels` 为 `(N,)`，`N >= 0`、`F >= 1`；`batch_size` 是正整数，`drop_last` 是布尔值。输入保证合法，不用写参数检查。
- 返回 Python 列表，每个元素为二元组 `(batch_features, batch_labels)`；沿 **axis 0（样本轴）** 分批，保留全部特征列。
- 只用一个按批推进的循环，不逐条收集样本，也不调用 `np.split`、`np.array_split` 或 `DataLoader`。
- 保持样本顺序、特征／标签对应关系和 dtype；函数不得修改输入。输出允许是切片视图，不要求复制成独立数组。

### 语法提示

- `range(start, stop, step)` 按 `step` 推进，右边界 `stop` 不包含在内。
- `array[start:stop]` 选取连续行，保留其他维度；`stop` 超过数组长度时，切片会在末尾停止。
- `array.shape[0]` 是样本数；`batches.append((x, y))` 向列表加入一对数组。

在仓库根目录运行：

```bash
uv run python day013/solution.py
```

骨架尚未实现时会抛出 `NotImplementedError`，补全函数后再运行验收。

## 5. 验收标准

- `drop_last=False` 时每个样本恰好出现一次；`True` 时只丢弃不足 `batch_size` 的尾批，完整批不能丢。
- 每批 shape 为 `(当前批大小, F)` 和 `(当前批大小,)`，dtype 与相应输入一致，输入内容不变。
- 空数据返回 `[]`；`batch_size > N > 0` 时，保留尾批得到一批，丢弃尾批得到 `[]`；恰好整除时不产生空批。
- `solution.py` 有两个验收样例：同一份 5 条样本按每批 2 条划分，分别检查 `drop_last=False` 的批大小 `[2, 2, 1]` 和 `True` 的 `[2, 2]`，以及配对、shape 和 dtype。

完成后告诉我实际用时。卡住时说 `hint`，我每次只给一个小提示。
