# AI Coding Training

长期 AI Coding 训练记录。每天一个 20～30 分钟的小训练，Coding 占至少 70%；AI 知识是背景，编程能力是目标。

目前只配置环境，尚未开始出题。

## 环境

使用 Python 3.12、uv 和项目独立的 `.venv`。基础依赖包括 NumPy、PyTorch、scikit-learn、Matplotlib，开发工具包括 pytest 和 Ruff。版本由 `uv.lock` 锁定。

在仓库根目录运行：

```bash
uv sync --locked
uv run python scripts/check_env.py
```

`uv run` 自动使用项目虚拟环境，不需要手动激活。Apple Silicon 的 MPS 可用时，检查脚本会验证 GPU tensor 运算和反向传播；CPU 也可用于日常训练。

本机已通过 CPU 和 Apple MPS 的运算、反向传播检查。在受限执行环境中 MPS 可能显示不可用，可以在本机终端重新运行上述检查。

如果在编辑器中运行 Python，选择本仓库的 `.venv/bin/python` 作为解释器。

## VS Code 调试

在 VS Code 中打开整个仓库文件夹，安装微软的 Python 和 Python Debugger 扩展。仓库的 `.vscode/launch.json` 已固定使用 `.venv/bin/python`。

1. 按 `⌘⇧D` 打开“运行和调试”。
2. 选择“检查环境（启动即暂停）”，点击菜单“运行 → 开始调试”。程序会先停在脚本开头。
3. 在 `scripts/check_env.py` 第 15 行设置断点，点击“继续”，查看 `x`、`weights` 和 `output` 的值与 shape。
4. 后续训练时，打开当天的 Python 文件，选择“调试当前 Python 文件”启动。

普通运行会直接执行程序；用“开始调试”启动才能使用 VS Code 断点。详细操作见 [VS Code 官方调试文档](https://code.visualstudio.com/docs/python/debugging)。

## 训练记录

开始训练后，按顺序使用以下结构：

```text
day001/
  problem.md    # 教练生成的题目
  solution.py   # 我自己编写的代码
  notes.md      # 完成并接受 Review 后整理
day002/
  ...
```

后续运行某一天的代码，例如：

```bash
uv run python day001/solution.py
```

需要测试或检查时：

```bash
uv run pytest day001/ -q
uv run ruff check day001/solution.py
```

pytest 命令只适用于当天已有测试文件的情况。未创建训练目录时，先运行环境检查即可。

## 与教练协作

- 说“开始今天的训练”后，教练先读历史代码和笔记，再只生成当天的 `problem.md`。
- 题目从简单到难，初期从 Python、NumPy 和基础 tensor 操作开始；每题只新增一个主要难点，根据实际完成情况逐步升级。
- 说 `hint` 时，每次只增加一个小提示。
- 提交代码后，教练先判断正确性，再定位问题并给修改提示。
- 明确说“给我参考答案”时，才提供完整实现。
- 完成并接受 Review 后，再整理 `notes.md`。

完整规则保存在 [AGENTS.md](AGENTS.md)，用于后续 Codex 工作。[OpenAI 的 AGENTS.md 文档](https://learn.chatgpt.com/docs/agent-configuration/agents-md)说明了项目指令的读取方式。

## GitHub 同步

训练记录完成后，先查看改动，再提交和推送：

```bash
git status
git diff
git add day001/
git commit -m "Complete day001 training"
git push
```

同步后续训练时，将 `day001/` 替换为实际目录。环境、缓存、凭据、下载的数据集和模型权重由 `.gitignore` 排除；不设置自动提交或推送。
