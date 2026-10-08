# Day 006 · Review 与总结

日期：2026-10-06（北京时间） · 实际用时：约 5 分钟（用户自报）

## Coding Skill

NumPy dtype 与图像预处理：将 `uint8` 像素安全转换为 `float32` 并缩放到 0～1。

## 出现的问题

Review 未发现功能错误。RGB 和灰度样例都通过；shape、输出 dtype 和输入保护均符合要求。

## 值得记住的 Pattern

先通过 `astype(np.float32)` 得到浮点数组，再做缩放，避免对原始 `uint8` 图像原地运算。实现核心：

```python
image_float = image.astype(np.float32)
image_float /= 255.0
return image_float
```

## 后续可以加强

本题约 5 分钟完成，明显低于预计用时；后续可小幅提高难度，并附上简短的 Python / NumPy 语法提示。
