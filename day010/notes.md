# Day 010 · Review 与总结

题目日期：2026-10-10（北京时间） · 完成与 Review：2026-10-14

实际用时：约 30 分钟（用户自报）。状态：通过 Review，用户已接受并授权提交、推送。

## Coding Skill

PyTorch mask broadcasting、沿 token 轴求均值，以及 dtype、device 和梯度的保持。

## 出现的问题

- 主要时间花在理解题意：不清楚 mask、hidden_dim 的含义，以及几个张量分别需要如何处理。明确后，核心池化计算正确。
- 初版只对 mask 做 `unsqueeze(-1)`；当隐藏状态为 `float32`、0/1 mask 为 `float64` 时，输出会提升为 `float64`。用户补上 `.to(hidden_states)`，使参与计算的 mask 与隐藏状态保持相同 dtype 和 device，混合 dtype 检查通过。
- 两个给定样例、bool/int/浮点 mask、全 padding、输入保护和 CPU 梯度检查通过；检查覆盖 `float16`、`float32` 和 `float64` 隐藏状态。当前进程的 MPS 不可用，未进行 MPS 实机验证。

## 值得记住的 Pattern

先解释每个轴：`B` 是样本数，`T` 是 token 位置数（包含 padding），`H` 是每个 token 的特征数量。mask 中的每个值控制一整条 `H` 维向量。

```text
hidden_states：(B, T, H)
mask：         (B, T) → (B, T, 1)
有效向量求和： (B, H)
有效 token 数：(B, 1)
池化输出：     (B, H)
```

沿 `T` 轴对有效 token 的同一特征坐标求平均，保留 `B` 和 `H`；分母限制为至少 1，使全 padding 行返回零向量。代码保持张量运算，有效位置正常传播梯度，padding 位置的梯度为零。时间复杂度为 `O(BTH)`，相乘产生的中间张量占用 `O(BTH)` 空间。

## 后续可以加强

后续题目先给出“每个轴代表什么 → 操作哪一维 → 输出保留哪些维度”，配一个小数值例子和中间 shape，再附必要语法提示。继续练习从数据含义推导张量操作，暂不因这题完成就跳到复杂模型或完整训练循环。
