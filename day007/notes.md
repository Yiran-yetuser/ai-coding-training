# Day 007 · Review 与总结

日期：2026-10-07（北京时间） · 实际用时：30 分钟以上（用户自报）

## Coding Skill

NumPy 数值稳定性与按类别轴计算 softmax；理解 logits 的 batch 与类别维度。

## 出现的问题

第一次读题把 softmax 理解为对整个矩阵求均值，导致行与行之间混在一起。改成按行取均值后，虽然能处理给定的相近 logits，但遇到 `[1000, 0, -1000]` 仍会溢出并产生 `NaN`。最终改为每行减去该行最大值；普通样例、极端幅度样例和大跨度样例均通过。

## 值得记住的 Pattern

对 `(B, C)` logits，先确认每行是一个独立样本、每列是一个类别，归一化沿类别轴进行。Softmax 对同一行减去一个常数保持不变；减去行最大值可让指数输入不大于 0。实现核心：

```python
max_val = np.max(logits, axis=1, keepdims=True)
exp_logits = np.exp(logits - max_val)
return exp_logits / exp_logits.sum(axis=1, keepdims=True)
```

## 后续可以加强

读题后先把 `(B, C)` 的轴含义写清楚，再动手实现；涉及 `axis` 的题目附简短语法与 shape 提示。
