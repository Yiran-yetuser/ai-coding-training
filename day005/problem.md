# Day 005 · 忽略 padding 位置计算 token accuracy · 2026-10-05

日期：2026-10-05（北京时间） · 难度：入门进阶

预计用时：25 分钟，读题约 4 分钟，编写代码约 16 分钟，验收约 5 分钟。

## 1. 今日 Coding Skill

**NumPy boolean mask 与布尔索引：只在有效 token 上计算指标。**

延续 Day 004 的分类准确率，今天把 batch 项从单个样本扩展到序列中的 token，并处理 padding 标签。主要新难点是根据标签构造有效位置 mask，再用它筛出真正参与评估的位置。

## 2. 必要背景

NER 等序列标注模型会为每个 token 预测一个类别，评估时通常把多条序列组成 `(batch_size, sequence_length)` 的矩阵。为了让不同长度的序列能组成 batch，较短的序列会补上 padding，但这些位置不代表真实 token，不能计入 accuracy。PyTorch 的 `CrossEntropyLoss` 支持 `ignore_index`，让指定标签不参与 loss；评估指标也需要跳过这些标签。[PyTorch 官方说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) 本题输入是预测类别和目标标签两个同 shape 的整数数组，输出只基于非忽略位置的准确率；不需要处理 logits、模型或 loss。

## 3. 小例子

```text
预测：[2, 1, 7]
标签：[2, 0, -100]

-100 表示 padding，不参与统计；前两个位置一对一错，accuracy = 0.5。
```

被忽略位置上的预测值可以是任意类别，都不应改变结果。

## 4. Coding 任务

完成下面的函数，在 token 分类评估时忽略目标值等于 `ignore_index` 的位置：

```python
def masked_token_accuracy(predictions, targets, ignore_index=-100):
    # predictions 和 targets 都是二维整数 NumPy 数组。
    # TODO: 只在有效标签位置计算准确率。
    # TODO: 按题目约定处理没有有效位置的 batch。
    raise NotImplementedError
```

输入约定：`predictions` 和 `targets` shape 相同，都是 `(B, T)`，其中 `B >= 1`、`T >= 1`。`targets` 可在任意位置包含 `ignore_index`；本题不要求检查非法类型或 shape。

使用 NumPy mask / 布尔索引完成，不写逐 token 的 Python 双重循环，不修改输入。没有任何有效标签时，函数约定返回 `0.0`。

在仓库根目录运行：

```bash
uv run python day005/solution.py
```

## 5. 验收标准

- 返回 0 到 1 之间的准确率比例，只统计 `targets != ignore_index` 的位置。
- 被忽略位置的预测不影响结果；输入数组保持不变。
- 全部标签均为 `ignore_index` 时返回 `0.0`。
- 两个样例都应通过：

```python
# 样例 1：padding 不计入 accuracy，预期为 0.75。
predictions = np.array([[2, 1, 7], [1, 2, 7]])
targets = np.array([[2, 0, -100], [1, 2, -100]])
assert masked_token_accuracy(predictions, targets) == 0.75

# 样例 2：没有有效 token，预期为 0.0。
predictions = np.array([[0, 1], [2, 1]])
targets = np.full((2, 2), -100)
assert masked_token_accuracy(predictions, targets) == 0.0
```

完成后告诉我实际用时。卡住时说 `hint`，我每次只给一个小提示。
