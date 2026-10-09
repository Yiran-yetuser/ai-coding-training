# Day 008 · 把图像切成 ViT patch 序列 · 2026-10-08

日期：2026-10-08（北京时间） · 难度：入门进阶

预计用时：25 分钟，读题约 4 分钟，编写代码约 16 分钟，运行验证约 5 分钟。

## 1. 今日 Coding Skill

**NumPy shape reasoning：用 reshape 与轴顺序整理图像 patch。**

## 2. 必要背景

Vision Transformer（ViT）会把图像分成不重叠的小块，再把每块展平成一个向量，作为后续 patch embedding 的输入。真实模型还会处理颜色通道、线性投影和位置编码；本题只实现前面的切块与展平。输入是单通道图像 batch，shape 为 `(B, H, W)`，以及 patch 边长 `P`；输出是 shape 为 `(B, patch_count, P*P)` 的数组。关键是保持 patch 按从左到右、从上到下排列，且每个 patch 内部按行优先展平。

## 3. 小例子

一张 4×4 图像切成边长为 2 的 patch，顺序应为左上、右上、左下、右下：

```text
[[ 0,  1,  2,  3],
 [ 4,  5,  6,  7],
 [ 8,  9, 10, 11],
 [12, 13, 14, 15]]

对应前两个 patch：
[0, 1, 4, 5], [2, 3, 6, 7]
```

## 4. Coding 任务

完成 `patchify(images, patch_size)`，把不重叠图像块整理成 patch 序列：

```python
def patchify(images, patch_size):
    # images.shape == (B, H, W)
    # TODO: 将高度和宽度拆成 patch 网格与 patch 内坐标。
    # TODO: 按行优先排列 patch，并展平每个 patch。
    raise NotImplementedError
```

输入为非空 NumPy 数组，shape 为 `(B, H, W)`；`patch_size` 是正整数，且能整除 `H` 和 `W`。输出保留输入 dtype，不修改输入。使用 NumPy 的向量化形状操作完成，不写 Python 循环；本题不要求检查非法 shape、非连续数组或多通道图像。

在仓库根目录运行：

```bash
uv run python day008/solution.py
```

## 5. 验收标准

- 输出 shape 为 `(B, (H // P) * (W // P), P * P)`。
- patch 顺序为从左到右、从上到下；patch 内部按行优先展开。
- dtype 保持不变，输入内容不变；两个样例均通过，其中一个检查 patch 的具体内容，另一个检查 batch 和单个 patch 的边界形状。

完成后告诉我实际用时。卡住时说 `hint`，我每次只给一个小提示。
