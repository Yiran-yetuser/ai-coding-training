# Day 003 · Review 与总结

日期：2026-10-03（北京时间）

状态：完成，通过 Review。

实际用时：约 10 分钟（用户自报）。

- **Coding Skill：** NumPy 二维数组的 `axis`、`keepdims`、broadcasting 和输入保护。
- **出现的问题：** 核心逻辑正确；运行题目中的样例和空 batch 检查均通过。没有影响结果的功能错误。
- **值得记住的 Pattern：** 按行计算均值并保留 `(B, 1)` shape，就能广播到 `(B, F)`；数组相减返回新结果，输入不被修改。时间和输出空间复杂度均为 `O(BF)`。
- **后续可以加强：** 后续题目尽量放进具体的 AI 开发步骤，例如模型输入预处理、loss/指标计算或推理后处理；保持实战背景，同时每题只新增一个主要 Coding 难点并循序渐进。

Review 验证：`uv run --locked python day003/solution.py` 通过。
