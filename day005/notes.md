# Day 005 · Review 与总结

日期：2026-10-05（北京时间） · 实际用时：约 20 分钟（用户自报）

## Coding Skill

NumPy boolean mask 与布尔索引：在 token 分类 accuracy 中跳过 padding 标签。

## 出现的问题

初版用 `targets > -100`，默认标签能处理，但没有遵循可配置的 `ignore_index` 参数；全 padding 时先对空数组求均值，也会产生运行警告。修正后，自定义忽略值和全忽略输入均通过检查。用户反馈时间主要花在理解题意和查 Python 语法。

## 值得记住的 Pattern

同一个布尔 mask 同步筛选预测和目标，避免二者位置错位；统计前先处理有效元素数量为 0 的情况。提交实现的核心片段：

```python
pred_masked = predictions[targets != ignore_index]
tar_masked = targets[targets != ignore_index]
if tar_masked.size == 0:
    return 0.0
acc = (pred_masked == tar_masked).mean()
return acc
```

## 后续可以加强

题目附简短的 Python / NumPy 语法提示，减少查语法的时间；继续练习 mask 与参数边界的对应关系。
