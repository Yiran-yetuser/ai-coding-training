# Day 011 · Debug 分类模型的 batch top-k 预测 · 2026-10-11

日期：2026-10-11（北京时间） · 难度：入门进阶

预计用时：20 分钟，读题和 shape 标注约 4 分钟，Debug 与修改约 11 分钟，运行检查约 5 分钟。

## 1. 今日 Coding Skill

**PyTorch batch 维度推理，以及 top-k 分数与类别索引的对应。**

## 2. 必要背景

分类模型对一批输入输出分数矩阵，shape 为 `(B, C)`：每行是一个样本，每列是一个类别。图像分类推理常展示每张图最可能的前几类，而不只展示第一名；`torch.topk` 会同时给出选中的分数和这些分数在原类别轴上的索引。[PyTorch 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.topk.html) 今天要 Debug 的函数把 `topk` 用在了错误的轴上，所以它会跨样本挑选，而不是为每个样本挑类别。你只需修复这一步，输入和输出都保持为 PyTorch tensor。

## 3. 小例子

若两张图的分数矩阵 shape 是 `(2, 4)`，取 `k=2` 后，结果分数和类别编号都应是 `(2, 2)`：每张图各有两个候选类别，而不是让两张图互相竞争。

## 4. Coding 任务

在 `solution.py` 中 Debug `topk_predictions(scores, k)`，保留函数签名和返回顺序 `(top_scores, class_ids)`。输入 `scores` 是非空二维浮点 tensor，shape 为 `(B, C)`；`k` 满足 `1 <= k <= C`。不写 Python 循环，不修改输入。

### 语法提示

- `torch.topk(tensor, k, dim=...)` 返回 `(values, indices)`；`indices` 是被选元素在输入中的位置。
- 对 `(B, C)`，先写出 `dim=0` 和 `dim=1` 分别代表什么；本题需要每行独立选择类别。
- `k=1` 时结果仍应保留二维 shape `(B, 1)`。

运行：

```bash
uv run python day011/solution.py
```

## 5. 验收标准

- 每个样本分别选出最大的 `k` 个类别分数，分数按降序排列；类别索引与原分数位置对应。
- 两个返回 tensor 的 shape 都是 `(B, k)`，输入不变；两个样例均通过。分数并列的情况不要求固定索引顺序。

```python
# 样例 1：两条样本分别选 top-2。
scores = torch.tensor([[0.1, 0.9, 0.4, 0.3], [0.8, 0.2, 0.7, 0.1]])
top_scores, class_ids = topk_predictions(scores, k=2)
assert top_scores.shape == class_ids.shape == (2, 2)
torch.testing.assert_close(top_scores, torch.tensor([[0.9, 0.4], [0.8, 0.7]]))
assert class_ids.tolist() == [[1, 2], [0, 2]]

# 样例 2：k=1 也保留 batch 维和 k 维。
one_score, one_id = topk_predictions(torch.tensor([[0.2, 0.5, 0.3]]), k=1)
assert one_score.shape == one_id.shape == (1, 1)
assert one_id.item() == 1
```

完成后告诉我实际用时。卡住时说 `hint`，我会先给一个很小的提示。