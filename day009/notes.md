# Day 009 · Review 与总结

日期：2026-10-09（北京时间） · 实际用时：约 7 分钟（用户自报）

## Coding Skill

NumPy 配对数据索引与可复现随机排列，用于训练前打乱特征和标签。

## 出现的问题

Review 未发现功能错误。多样本配对、固定 seed 可复现、输入不变和单样本 shape 检查均通过。

## 值得记住的 Pattern

只生成一次排列索引，并将同一索引应用于特征和标签。固定 seed 让随机顺序可复现；NumPy 整数数组索引会返回重排结果，不会原地改动输入。实现核心：

```python
rng = np.random.default_rng(seed)
perm = rng.permutation(len(features))
shuffled_features = features[perm]
shuffled_labels = labels[perm]
```

## 后续可以加强

本题约 7 分钟完成，明显低于预计用时；后续可小幅提高难度，并继续提供简短的 Python / NumPy 语法提示。
