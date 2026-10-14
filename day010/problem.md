# Day 010 · 实现 Transformer token 的带 mask 均值池化 · 2026-10-10

日期：2026-10-10（北京时间） · 难度：入门进阶

预计用时：20 分钟，读题约 4 分钟，编码约 11 分钟，运行与检查约 5 分钟。

## 1. 今日 Coding Skill

**PyTorch tensor shape 推理、mask broadcasting 与按轴归约。**

今天把已练过的 mask 和按轴操作带到 PyTorch 的真实模型步骤中；主要新挑战是让 `(B, T)` 的 token mask 对齐 `(B, T, H)` 的隐藏状态。

## 2. 必要背景

Transformer encoder 会为每个 token 输出一个隐藏向量。Sentence Transformers 等句向量模型会把这些 token 向量池化成固定长度的句向量，mean pooling 是常用方式之一。[Sentence Transformers 官方文档](https://www.sbert.net/docs/package_reference/sentence_transformer/modules.html) Batch 中的 padding token 不能参与平均，因此要用 `attention_mask` 只统计有效 token。本题输入隐藏状态 `(B, T, H)` 和 mask `(B, T)`，输出每条序列一个 `(H,)` 向量；全 padding 的行按本题约定输出零向量。

## 3. 小例子

若 `hidden_states.shape == (2, 3, 4)`、`attention_mask.shape == (2, 3)`，输出 shape 应为 `(2, 4)`。第一行 mask `[1, 1, 0]` 表示只把前两个 token 向量纳入这一行的池化结果，第三个向量是 padding。

## 4. Coding 任务

在 `solution.py` 中完成 `masked_mean_pool(hidden_states, attention_mask)`：

- 输入 `hidden_states` 是浮点 PyTorch tensor，shape 为 `(B, T, H)`；`attention_mask` 是同设备上的 bool 或 0/1 tensor，shape 为 `(B, T)`。
- 返回 shape 为 `(B, H)` 的 tensor；每一行只对 mask 为 1 的 token 求均值。
- 某一行的 mask 全为 0 时，该行返回全零；不修改输入，不写逐样本或逐 token 的 Python 循环。
- 保持 PyTorch tensor 运算，不转成 NumPy。

### 语法提示

- `tensor.unsqueeze(-1)` 在最后增加一个长度为 1 的轴；想一想它如何让 `(B, T)` 的 mask 与 `(B, T, H)` 对齐。
- `tensor.sum(dim=1)` 沿 token 轴求和并移除这一轴。
- `tensor.clamp_min(1)` 可把张量中的值限制为至少 `1`，可用于处理除数为零的边界。

在仓库根目录运行：

```bash
uv run python day010/solution.py
```

## 5. 验收标准

- 有效 token 向量均匀参与平均，padding 向量不影响结果；输出 shape、dtype 和 device 与约定一致，输入保持不变。
- 全零 mask 返回对应的全零句向量；两个样例均通过。

```python
# 样例 1：有 padding，第二条序列只有一个有效 token。
hidden_states = torch.tensor([
    [[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]],
    [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
])
mask = torch.tensor([[1, 1, 0], [0, 1, 0]])
assert masked_mean_pool(hidden_states, mask).shape == (2, 2)
torch.testing.assert_close(masked_mean_pool(hidden_states, mask), torch.tensor([[2.0, 4.0], [6.0, 8.0]]))

# 样例 2：整行都是 padding。
all_padding = torch.zeros((1, 3), dtype=torch.bool)
torch.testing.assert_close(
    masked_mean_pool(torch.ones((1, 3, 2)), all_padding),
    torch.zeros((1, 2)),
)
```

完成后告诉我实际用时。卡住时说 `hint`，我会先给一个很小的提示。