# Day 008 · Review 与总结

日期：2026-10-08（北京时间） · 实际用时：约 15 分钟（用户自报）

## Coding Skill

用 NumPy `reshape` 和 `transpose` 将图像拆成按顺序排列的 patch 序列。

## 出现的问题

实现正确，两个验收样例通过。主要困难是读懂 `transpose` 的轴参数如何改变维度顺序。

## 值得记住的 Pattern

先把图像 shape 拆成 batch、patch 网格坐标和 patch 内坐标，再调整轴顺序，最后分别展平网格和 patch 内维度。核心实现：

```python
x = images.reshape(B, n_h, patch_size, n_w, patch_size)
x = x.transpose(0, 1, 3, 2, 4)
patches = x.reshape(B, n_h * n_w, patch_size * patch_size)
```

## 后续可以加强

涉及 `transpose` 的题目附简短语法提示，并把转置前后的轴名称与 shape 对照写清楚。
