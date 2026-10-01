# Day 001 · 修复预测结果筛选函数

日期：2026-10-01（北京时间） · 难度：入门

预计用时：25 分钟，读题约 3 分钟，修改代码约 17 分钟，运行验证约 5 分钟。

## 1. 今日 Coding Skill

**Python 控制流 Debug：列表遍历、条件判断、函数返回。**

今天主要练习：读一小段代码，观察它实际执行了哪些步骤，再修复行为。只用 Python 标准语法即可完成。

## 2. 必要背景

目标检测模型会在图片中提出多个候选目标，并给每个候选结果打分。今天把这些分数简化为一个 Python 列表，每个分数都在 0～1 之间。输入还有一个阈值，分数大于或等于阈值的结果应该保留。输出是这些结果在原列表中的索引，索引从 0 开始。你的代码只负责筛选，不需要加载图片或运行模型。

**这类操作在哪里使用？**

- **Faster R-CNN 的推理后处理：** 模型生成候选检测结果后，先按分数过滤，再用非极大值抑制（NMS）处理重叠框。本题把其中的分数筛选步骤缩小为列表操作。[Torchvision 官方实现](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py)
- **逻辑回归的二分类决策：** 模型给出正类概率后，用阈值转成类别判断；例如判断一条消息是否属于某一类。本题保留多个结果的索引，与它共享“比较分数并作出选择”的编程模式。[scikit-learn 官方示例](https://scikit-learn.org/stable/auto_examples/model_selection/plot_tuned_decision_threshold.html)

不同实现对等于阈值的处理可能不同，**本题以 `>=` 为准**。这些背景帮助理解代码的用途，不要求学习或实现完整模型。

## 3. 小例子

```text
索引：        0     1
分数：      0.2   0.7
阈值：0.5

保留的索引：[1]
```

注意输出是索引，列表里的 `1` 表示保留第二个结果。

## 4. Coding 任务

下面是已有的筛选函数。它能执行，但结果不符合预期。请阅读并修复它。

自己创建 `day001/solution.py`，复制下面这段有问题的代码，再进行修改：

```python
def select_predictions(scores, threshold=0.5):
    selected = []
    for index, score in enumerate(scores):
        if score > threshold:
            selected.append(index)
        return selected
```

`enumerate(scores)` 每次提供两个值：当前位置的索引 `index`，以及该位置的分数 `score`。

操作顺序：

1. 在函数下方调用它，先观察第一个验收样例的实际输出。
2. 用 `print` 或 VS Code 断点观察 `index`、`score`、`selected`，追踪程序如何执行到返回。
3. 修改函数，使它满足下面的验收标准。保留函数名和参数，不修改输入列表，不把样例答案写死。
4. 运行两个验收样例，再自己试一下所有分数都低于阈值的情况。

在仓库根目录运行你的代码：

```bash
uv run python day001/solution.py
```

使用 VS Code 时，打开你创建的 Python 文件，选择“调试当前 Python 文件”，通过“运行 → 开始调试”启动。

## 5. 验收标准

- 输入：`scores` 是由 0～1 数值组成的列表；`threshold` 是 0～1 的数值，默认值为 `0.5`。本题不要求处理非法输入。
- 输出：一个 Python 列表，包含所有分数 **大于或等于** 阈值的位置索引，顺序与原列表一致。
- 原始 `scores` 保持不变。
- 边界：等于阈值也要保留；空列表或没有符合条件的结果时返回 `[]`。

给定验收样例：

```python
# 样例 1：同时检查遍历、顺序和等于阈值的行为。
assert select_predictions([0.2, 0.8, 0.5, 0.9], 0.5) == [1, 2, 3]

# 样例 2：空输入，也使用默认阈值。
assert select_predictions([]) == []
```

完成后提交你的代码，并告诉我大约用了多久。卡住时说 `hint`，我会只给一个小提示。
