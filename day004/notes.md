# Day 004 · Review 与总结

日期：2026-10-04（北京时间）

状态：完成，通过 Review，用户已确认。

实际用时：约 5 分钟（用户自报）。

- **Coding Skill：** Debug 分类模型评估代码中的 NumPy axis 选择与 batch 对齐。
- **出现的问题：** 初始实现沿 `axis=0` 取最大值，得到的是每个类别对应的样本位置，而不是每个样本的预测类别；改为按行计算后，`predicted` 与 `targets` 都是 `(B,)`。用户补充了 `logits`、`predicted`、`targets` 的 shape 注释。两个验收样例均通过。
- **值得记住的 Pattern：** 对 `(B, C)` logits 按类别维取 `argmax`，应为每行产生一个类别编号；比较前核对预测与标签的 shape，避免相等维度时静默算错，或 broadcasting 改变预期。
- **后续可以加强：** 本题约 5 分钟完成，难度偏低。下一题提高一小步，并放进更真实的 AI 评估流程，例如使用 mask 忽略 padding 标签；一次只新增一个主要 Coding 难点。

Review 验证：`uv run --locked python day004/solution.py` 通过。实现对 batch 的时间复杂度为 `O(B * C)`，预测数组占用 `O(B)` 空间。
