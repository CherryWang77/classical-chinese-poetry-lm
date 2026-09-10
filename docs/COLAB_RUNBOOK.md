# Colab T4 操作说明书

适用项目：**Every Character Counts: Tokenization-Aware Hard Constraints for Chinese Song-Ci Generation**

这份说明按“首次设置 → 强制 smoke test → 正式训练 → 断线恢复 → 验收”的顺序编写。第一次操作只执行到 smoke test 通过；不要直接运行正式训练。

## 0. 开始前检查

准备以下条件：

- Colab 能分配 T4 GPU；
- Google Drive 建议至少保留 20 GB 空间；
- 一个 Hugging Face token，至少能读取 Qwen 模型；发布到私有 staging repo 时需要写权限；
- GitHub 分支 `research/every-character-counts` 已推送并可访问。

安全要求：

- 不要把 HF token 粘贴进代码单元、聊天、截图或 GitHub；
- 不要改正式 JSON 配置中的 seed、split、超参数或模型 revision；
- 不要在 rhyme lambda 冻结前运行 test grid；
- 每次只跑一个正式 seed。

## 1. 打开 notebook

分支发布后，可用以下地址打开：

```text
https://colab.research.google.com/github/CherryWang77/classical-chinese-poetry-lm/blob/research/every-character-counts/notebooks/colab_research.ipynb
```

也可以在 Colab 选择 `File → Upload notebook`，上传仓库中的 `notebooks/colab_research.ipynb`。

## 2. 选择 T4 GPU

1. 打开 Colab 顶部菜单 `Runtime`。
2. 选择 `Change runtime type`。
3. `Hardware accelerator` 选择 `T4 GPU`。
4. Runtime version 保持最新默认版本。
5. 点击保存，然后等待右上角显示已连接。

GPU 类型和运行时限额由 Colab 动态分配，不是永久保证。不要依靠一个 session 连续跑完全部实验。

## 3. 配置 HF_TOKEN Secret

1. 在 Colab 左侧栏点击钥匙形状的 `Secrets`。
2. 点击 `Add new secret`。
3. Name 填写：`HF_TOKEN`。
4. Value 粘贴 Hugging Face token。
5. 打开该 Secret 的 `Notebook access` 开关。
6. 不要把 token 写入任何代码单元。

若第一个 notebook 单元提示 `Add an HF_TOKEN secret`，通常是名称不对或 Notebook access 没打开。

## 4. 首次运行：只执行单元 1–9

逐个点击单元左侧的运行按钮，不要使用 `Run all`。

### 单元 1：挂载 Drive 与读取 Secret

首次会弹出 Google Drive 授权。选择自己的账号并允许挂载。

预期输出包含：

```text
Persistent root: /content/drive/MyDrive/every-character-counts
```

模型缓存、数据、checkpoint 和结果都会进入这个目录。Colab 虚拟机被回收后，这些文件仍保留。

### 单元 2：下载代码

该单元从固定研究分支创建干净 checkout，并打印 Git commit。

如果出现 `Remote branch ... not found`，停止操作：说明研究分支尚未推送，不能自行改用 `main`。

### 单元 3：连接持久化目录

预期输出：

```text
Drive persistence links are ready.
```

它只把 `data/research`、`artifacts/research`、`results/research` 和 `output/pdf` 指向 Drive，不改历史 educational baseline。

### 单元 4：安装依赖

第一次通常需要数分钟。黄色依赖提示不一定是错误；只有红色 traceback 或 `CalledProcessError` 才算失败。

### 单元 5：GPU 自检

必须看到：

- GPU 名称含 `Tesla T4`；
- `torch.cuda.is_available()` 通过；
- 显存约为 15 GiB。

若显示 `No CUDA GPU`，返回第 2 节重新选择 T4，然后从单元 1 重跑。

### 单元 6：数据与离线测试

该单元会：

1. 下载两个锁定 commit；
2. 清洗、去重和 group split；
3. 重建模板审计；
4. 运行 pytest 和 ruff。

关键期望值：

```text
all = 20135
selected = 4419
selected train/validation/test = 3541/441/437
```

最后必须看到所有测试通过和 `All checks passed!`。数量或 hash 不同就停止，不要训练。

### 单元 7：20 条数据、2 optimizer steps 的 QLoRA smoke training

首次运行会下载固定 revision 的 Qwen3-1.7B，所需时间主要取决于网络。这个步骤不是正式 seed 训练。

如果 T4 首次发生 CUDA OOM，训练器会自动从 micro batch 2 降为 1，并把 accumulation 调整为 16，保持 effective batch 为 16。

### 单元 8：20 个在线硬约束样本

它运行 10 个词牌 × 2 个主题，并生成 smoke evaluation。

### 单元 9：强制发布门槛

最后一行必须严格显示：

```text
SMOKE TEST PASSED
```

其含义是：20/20 structural exact match，且 0 dead ends。任何 assertion error 都要停止，把完整报错和单元 5 的 GPU 信息发回项目维护者。

## 5. Smoke 通过后如何跑正式 QLoRA

只有收到“可以开始正式训练”的确认后，才执行单元 10。

每个 Colab session 只设置一个 seed：

```python
FORMAL_SEED = 42
```

三个允许值依次为：

```text
42
1729
2026
```

一个 seed 完成后，确认以下文件位于 Drive：

```text
MyDrive/every-character-counts/artifacts/research/qwen3_1.7b_lora/seed_<SEED>/best_adapter/
MyDrive/every-character-counts/artifacts/research/qwen3_1.7b_lora/seed_<SEED>/run_manifest.json
```

然后结束 runtime，下一次把 `FORMAL_SEED` 改为另一个值。不要同时开多个 Colab session 写同一 seed。

## 6. 断线恢复

Colab runtime 被回收或浏览器断线后：

1. 重新选择 T4；
2. 从单元 1 依次执行到单元 6；
3. 把 `FORMAL_SEED` 设置为中断的 seed；
4. 重新执行单元 10。

命令中的 `--resume auto` 会在 Drive 中寻找最大的 `checkpoint-<step>` 并恢复 model、optimizer、scheduler 和 global step。不要手工复制 checkpoint，也不要删除已有 checkpoint。

如果中断发生在第一个 checkpoint 保存前，只能从该 seed 的开头重跑。

## 7. Validation lambda sweep

某个 seed 的 adapter 完成后，可执行单元 11。它只生成 validation，不读取 test 内容。

必须等三个训练 seed 的五个 lambda 值全部完成，再执行单元 12。单元 12 生成：

```text
results/research/selected_rhyme_lambda.json
```

没有这个文件，rhyme test 系统会主动拒绝运行。

## 8. 正式 test 与自动评测

lambda 冻结后：

1. 对当前 seed 执行单元 13；
2. 对三个 seed 分别完成；
3. 只执行一次单元 14 的 zero-shot；
4. 全部系统完成后执行单元 15。

生成是 append-only，并按 system/template/theme/training seed/decoding seed/lambda 跳过已完成项。因此断线后可安全重跑同一单元。

最终核心文件：

```text
results/research/generations.jsonl
results/research/evaluation/summary.json
results/research/evaluation/main_results_test.csv
```

简历数字只能引用 `summary.json`，不能手工抄取 notebook 中间输出。

## 9. 常见故障

### `CUDA out of memory`

- 等待自动 batch fallback；
- 若仍失败，选择 `Runtime → Disconnect and delete runtime`，重新连接 T4；
- 不要在同一 runtime 中同时保留 judge 和训练模型；
- 仍失败时保存完整 traceback，不要自行改 LoRA rank、max length 或 effective batch。

### `401`、`403` 或 Hugging Face 权限错误

- 检查 Secret 名称是否严格为 `HF_TOKEN`；
- 检查 Notebook access 是否打开；
- 检查 token 是否过期及是否有目标仓库权限；
- 不要在输出里打印 token。

### Drive 空间不足

- 停止当前单元；
- 不要随意删除 checkpoint；
- 先确认哪些 seed 已有 `best_adapter` 和 manifest，再决定归档策略。

### Colab GPU 配额暂时不可用

Colab 的 GPU 类型、空闲超时和额度会动态变化。不要改成 CPU 跑 QLoRA；结束 GPU runtime，等待配额恢复。数据构建、测试、统计和报告可以使用 CPU runtime。

### 浏览器断线但单元可能仍在运行

先重新连接，检查输出和 Drive checkpoint。不要立刻启动第二个相同 seed，以免两个进程同时写同一目录。

## 10. 每个阶段的停止条件

- 数据阶段：hash 或计数不同——停止；
- smoke 阶段：不是 20/20 exact 或出现 dead end——停止；
- 正式训练：没有 `best_adapter` 或 manifest——该 seed 不算完成；
- lambda 阶段：五个候选 × 三 seed 不完整——不得运行 test；
- test 阶段：七系统 grid 不完整——不得生成简历数字；
- 人工评测：未确认两名中文母语评测者——只报告自动指标和 descriptive LLM judge。

