# Day 002 · 补全文本 batch 的 padding 与 mask · 2026-10-02

日期：2026-10-02（北京时间） · 难度：入门进阶

预计用时：25 分钟，读题约 4 分钟，补全代码约 16 分钟，运行验证约 5 分钟。

## 1. 今日 Coding Skill

**Batch processing：把不同长度的一维列表，整理成行长度一致的二维列表。**

继续使用昨天的循环与列表收集。今天新增的主要难点是：同时维护每一行的数据和对应的位置标记，并保持原输入不变。只使用 Python 列表即可完成。

## 2. 必要背景

NLP 中，一段文本可以先被切成 token，再转换为整数编号列表。不同文本的列表长度往往不同，批量送入模型前常需要把它们补齐成相同长度。Padding 就是给短序列添加占位编号，本题统一在右侧补齐。对应的 mask 用 `1` 标记原始位置，用 `0` 标记后来补上的位置，帮助模型区分有效输入与占位。输入是一批整数列表，输出是补齐后的二维列表和同样大小的 mask。你的代码只负责准备 batch，不需要实现分词器或运行模型。

**这一步在哪里使用？**

- **BERT 文本分类：** 模型接收形如 `(batch_size, sequence_length)` 的 token 编号和 attention mask；今天练习的是送入模型前的数据整理步骤。[BERT 官方接口说明](https://huggingface.co/docs/transformers/model_doc/bert)
- **Hugging Face 的批量预处理：** Padding 可以对齐到当前 batch 的最长序列，避免所有 batch 都补到一个固定的大长度。本题采用同样的长度约定，用 Python 列表练习这个过程。[Padding 官方说明](https://huggingface.co/docs/transformers/main/pad_truncation)

本题中的整数是简化的编号，不验证真实词表；mask 专指原始位置与 padding 位置的标记。

## 3. 小例子

```text
输入：[[8, 9], [3]]
pad_id：0

补齐后的数据：         对应 mask：
[[8, 9],              [[1, 1],
 [3, 0]]               [1, 0]]
```

两份输出都是 2 行、每行 2 个元素。第二行的数据和 mask 必须保持位置对应。

## 4. Coding 任务

自己创建 `day002/solution.py`，复制下面的骨架并补全 TODO：

```python
def pad_batch(sequences, pad_id=0):
    padded = []
    masks = []
    width = max((len(seq) for seq in sequences), default=0)

    for seq in sequences:
        # TODO：构造补齐后的新行，以及对应的 mask 行。
        # TODO：把两行分别加入 padded 和 masks。
        pass

    return padded, masks
```

`width` 已经计算好了：它是输入中最长一行的长度；没有任何行时取 `0`。你只需补全每一行的处理，不必重写这段长度计算。

操作顺序：

1. 完成右侧 padding，并支持传入不同的 `pad_id`。
2. 同时生成 mask，确保每个位置与数据对应。
3. 运行两个给定样例，再自己选择一个验收标准中的边界情况验证。
4. 用断点观察一个短序列处理前后的长度，确认原始列表没有被补长。

使用普通循环和列表即可；本次不调用库里现成的 padding 函数。保留函数名和参数，不把样例答案写死。

在仓库根目录运行：

```bash
uv run python day002/solution.py
```

使用 VS Code 时，打开你创建的 `.py` 文件，选择“调试当前 Python 文件”，通过“运行 → 开始调试”启动。

## 5. 验收标准

- 输入：`sequences` 是整数列表组成的列表，`pad_id` 是整数，默认为 `0`。不要求处理非法类型。
- 输出：`(padded, masks)`，其中两项都是 Python 二维列表。保持输入行顺序，每行长度等于当前 batch 的最长长度，只在右侧补齐，不截断。
- Mask：原始位置为整数 `1`，补齐位置为整数 `0`。**即使原始编号恰好等于 `pad_id`，它仍是有效位置。**
- 输入保护：不修改原始列表；每一行输出都是新列表，修改输出不能影响输入或其他行。
- 边界：空 batch 返回 `([], [])`；batch 中的空序列保留为一行；所有序列都为空时，输出保留原行数，每行仍为空。

```python
# 样例 1：不同长度、空序列，以及原始编号 0。
sequences = [[8, 0], [3], []]
padded, masks = pad_batch(sequences)
assert padded == [[8, 0], [3, 0], [0, 0]]
assert masks == [[1, 1], [1, 0], [0, 0]]
assert sequences == [[8, 0], [3], []]

# 样例 2：没有任何序列。
assert pad_batch([]) == ([], [])
```

完成后提交代码，并告诉我实际用时。卡住时说 `hint`，我每次只给一个小提示。
