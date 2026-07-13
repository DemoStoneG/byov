# Training Output Standardization Plan

本文档用于整理 BYOV 项目训练输出的规范化方案。目标是让每一次训练都能清楚回答：

- 这次实验是谁跑的、什么时候跑的、用的什么参数？
- checkpoint、TensorBoard、日志、评估结果分别在哪里？
- 某个指标来自哪个 epoch、哪个 checkpoint、哪个配置？
- 多次实验之间是否互相覆盖？

## 1. 当前主要问题

### 1.1 实验输出目录边界不清晰

当前训练脚本会把输出目录拼成：

```text
./logs/exp_<dataset>/<output_dir>
```

例如：

```text
./logs/exp_break_eggs/bestconfig
```

但 PyTorch Lightning 又会在该目录下自动创建：

```text
lightning_logs/version_*
```

于是一次实验的产物会分散在多个位置：

```text
logs/exp_break_eggs/bestconfig/
  args.json
  train_embeds.npy
  train_label.npy
  val_embeds.npy
  val_label.npy
  lightning_logs/
    version_0/
      checkpoints/
      events.out.tfevents...
```

问题是：`bestconfig`、`version_0`、`args.json`、`train_embeds.npy` 之间的关系不够直观。实验多了以后，很难判断某个文件到底属于哪次训练。

### 1.2 训练恢复 checkpoint 被硬编码

当前 `train.py` 中存在硬编码 checkpoint：

```python
trainer.fit(task, ckpt_path="/mnt/data/wzh/projects/byov-main/logs/exp_break_eggs/bestconfig/lightning_logs/version_1/checkpoints/epoch=89.ckpt")
```

这会导致：

- 不传任何参数时，训练也可能从旧 checkpoint 恢复。
- 换数据集时也可能错误地加载 `break_eggs` 的旧 checkpoint。
- 复现实验时，很难从命令行看出这次训练到底从哪里开始。

### 1.3 配置文件可能被覆盖

当前 `args.json` 写在：

```text
logs/exp_<dataset>/<output_dir>/args.json
```

如果重复使用同一个 `output_dir`，旧的 `args.json` 会被覆盖。科研实验中，参数文件应该和一次具体 run 绑定，不能被后续实验覆盖。

### 1.4 下游评估 embedding 会被覆盖

训练期间每隔 `ds_every_n_epoch` 会提取 downstream embedding。当前保存路径类似：

```text
train_embeds.npy
train_label.npy
val_embeds.npy
val_label.npy
```

这些文件直接放在实验根目录，没有 epoch 信息。第 10、20、30 epoch 的评估会反复覆盖同一批文件。

### 1.5 TensorBoard run 名称不可读

Lightning 默认生成：

```text
lightning_logs/version_0
lightning_logs/version_1
lightning_logs/version_2
```

`version_57` 这类名称不能表达数据集、关键参数、时间、实验目的。后续对比实验时可读性较差。

### 1.6 控制台输出和文件日志没有统一

当前项目中混用了：

- `print()`
- `tqdm`
- Lightning 的 `self.log()`
- 未充分使用的 `utils/logger.py`

这导致控制台信息、TensorBoard 指标和磁盘日志不能完整对应。

### 1.7 验证 loss 没有记录

`video_tasks.py` 中计算了 `val_loss`，但记录语句被注释：

```python
# self.log('val_loss', loss, on_step=True, on_epoch=True)
```

训练时应该至少记录：

- training loss
- validation loss
- downstream evaluation metrics

否则很难判断模型训练目标本身是否稳定。

### 1.8 大量训练产物放在代码仓库下

当前输出默认在项目目录下的 `logs/`。如果 checkpoint、embedding、TensorBoard 文件很多，会带来：

- Git 状态扫描变慢。
- 容易误提交大文件。
- 代码和实验产物混在一起。
- 项目目录越来越乱。

更推荐将训练产物放到数据盘或专门的实验目录。

## 2. 建议修改内容

### 2.1 增加统一实验输出根目录

建议新增参数：

```bash
--output_root /mnt/data/wzh/experiments/byov
```

含义：

- `output_root` 是所有实验输出的总目录。
- 不建议默认把大文件写进代码仓库。
- 如果用户想临时写到项目目录，也可以显式传入 `--output_root ./outputs`。

推荐默认结构：

```text
/mnt/data/wzh/experiments/byov/
```

或项目内软链接：

```text
/home/wzh/projects/byov/outputs -> /mnt/data/wzh/experiments/byov
```

### 2.2 每次训练创建独立 run 目录

建议新增参数：

```bash
--run_name baseline
```

每次训练自动生成唯一 run 目录：

```text
<output_root>/<dataset>/<timestamp>_<run_name>_<backbone>_bs<batch_size>_lr<lr>_seed<seed>/
```

例如：

```text
/mnt/data/wzh/experiments/byov/break_eggs/20260710_153012_baseline_clip_bs4_lr1e-5_seed42/
```

这里：

- `output_root`：所有实验总目录。
- `dataset`：数据集或任务名。
- `timestamp_run_name_backbone_bs_lr_seed`：一次具体训练 run。

### 2.3 用命令行参数控制断点续训

建议新增参数：

```bash
--resume /path/to/existing/run_dir
```

规则：

- 不传 `--resume`：创建新的 run 目录，从头训练。
- 传 `--resume`：继续已有 run，自动加载 `<run_dir>/checkpoints/last.ckpt`。
- 断点续训继续写入原 run 目录，不创建新目录。
- 删除 `train.py` 中硬编码 checkpoint。

示例：

```bash
python train.py \
  --resume /mnt/data/wzh/experiments/byov/break_eggs/20260710_153012_baseline_clip_bs4_lr1e-5_seed42
```

### 2.4 统一 run 目录结构

建议每次训练固定生成如下结构：

```text
<run_dir>/
  config/
    args.json
    command.txt
    git.txt
    env.txt

  logs/
    train.log

  tensorboard/
    events.out.tfevents...

  checkpoints/
    epoch=009.ckpt
    epoch=019.ckpt
    epoch=029.ckpt
    last.ckpt
    best.ckpt

  metrics/
    metrics.csv
    metrics.jsonl
    downstream_epoch_009.json
    downstream_epoch_019.json

  artifacts/
    embeddings/
      epoch_009/
        train_embeds.npy
        train_label.npy
        val_embeds.npy
        val_label.npy
```

### 2.5 配置和环境信息保存到 config/

建议保存：

#### args.json

保存本次训练的所有命令行参数。

#### command.txt

保存原始训练命令，例如：

```text
python train.py --dataset break_eggs --run_name baseline --batch_size 4 --lr 1e-5
```

#### git.txt

保存代码版本信息，例如：

```text
commit: <git_commit_hash>
branch: <branch_name>
status: clean/dirty
diff summary: ...
```

#### env.txt

保存环境信息，例如：

```text
python version
torch version
pytorch lightning version
cuda version
hostname
working directory
```

### 2.6 统一日志系统

建议使用 Python logging 统一输出：

- 控制台输出一份。
- 同步写入 `<run_dir>/logs/train.log`。
- 普通信息用 `logger.info()`。
- 警告信息用 `logger.warning()`。
- 错误信息用 `logger.error()`。

尽量减少直接 `print()`。需要进度条时保留 `tqdm`，但关键结果仍应写入 logger。

### 2.7 统一 TensorBoard 输出

建议使用 Lightning 的 `TensorBoardLogger`，并固定写入：

```text
<run_dir>/tensorboard/
```

不要再依赖：

```text
lightning_logs/version_*
```

这样查看 TensorBoard 时可以直接运行：

```bash
tensorboard --logdir /mnt/data/wzh/experiments/byov
```

或查看单次 run：

```bash
tensorboard --logdir /mnt/data/wzh/experiments/byov/break_eggs/20260710_153012_baseline_clip_bs4_lr1e-5_seed42/tensorboard
```

### 2.8 规范 TensorBoard 指标命名

建议使用分组命名：

```text
train/loss_step
train/loss_epoch
val/loss

classification/regular_f1
classification/ego2exo_val_f1
classification/exo2ego_val_f1

retrieval/regular_map10
retrieval/ego2exo_val_map10
retrieval/exo2ego_val_map10

progression/train_score
progression/val_score

kendall/train_tau
kendall/val_tau
```

好处：

- TensorBoard 左侧会自动按 `train`、`val`、`classification` 等分组。
- 指标含义更清楚。
- 论文画图和结果整理更方便。

### 2.9 downstream embedding 按 epoch 保存

训练中第 N 个 epoch 做 downstream evaluation 时，embedding 应保存到：

```text
<run_dir>/artifacts/embeddings/epoch_009/
```

或如果使用 1-based epoch 显示，也可以保存为：

```text
<run_dir>/artifacts/embeddings/epoch_010/
```

建议全项目统一一种规则。由于 Lightning checkpoint 常见文件名是 `epoch=009.ckpt`，可以使用 0-based 的 `epoch_009`，但日志中显示为第 10 个 epoch。

### 2.10 downstream 指标保存为 JSON

每次 downstream evaluation 除了写 TensorBoard，还应保存 JSON：

```text
<run_dir>/metrics/downstream_epoch_009.json
```

示例：

```json
{
  "epoch": 9,
  "display_epoch": 10,
  "classification": {
    "regular_f1": 0.4312,
    "ego2exo_val_f1": 0.4021,
    "exo2ego_val_f1": 0.4158
  },
  "retrieval": {
    "regular_map10": 0.2875,
    "ego2exo_val_map10": 0.2664,
    "exo2ego_val_map10": 0.2719
  },
  "progression": {
    "train_score": 0.6112,
    "val_score": 0.5021
  },
  "kendall": {
    "train_tau": 0.4520,
    "val_tau": 0.3844
  }
}
```

### 2.11 checkpoint 保存到 checkpoints/

建议 checkpoint 固定保存到：

```text
<run_dir>/checkpoints/
```

保留：

- 周期性 checkpoint，例如每 10 epoch 一个。
- `last.ckpt`，用于恢复最新训练。
- 可选 `best.ckpt`，根据指定指标保存最佳模型。

示例：

```text
checkpoints/
  epoch=009.ckpt
  epoch=019.ckpt
  epoch=029.ckpt
  last.ckpt
  best.ckpt
```

### 2.12 记录 val_loss

建议恢复验证 loss 记录：

```python
self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
```

训练 loss 建议拆成：

```python
self.log("train/loss_step", loss, on_step=True, on_epoch=False)
self.log("train/loss_epoch", loss, on_step=False, on_epoch=True)
```

也可以只记录一个统一的：

```python
self.log("train/loss", loss, on_step=True, on_epoch=True)
```

但 TensorBoard 里会出现 step 和 epoch 两种序列，需要命名清楚。

### 2.13 .gitignore 忽略实验产物

如果仍然允许输出到项目目录，建议 `.gitignore` 至少包含：

```gitignore
logs/
outputs/
lightning_logs/
*.ckpt
*.npy
*.pt
*.pth
*.tfevents*
```

更推荐将大文件写到项目目录外，例如：

```text
/mnt/data/wzh/experiments/byov/
```

### 2.14 使用软链接在项目中查看实验文件

实验产物建议真实存放在项目目录外，例如：

```text
/mnt/data/wzh/experiments/byov/
```

同时在当前代码项目中创建软链接：

```text
/home/wzh/projects/byov/outputs -> /mnt/data/wzh/experiments/byov
```

这样有两个好处：

- 大文件实际在数据盘，不会拖慢 Git，也不容易误提交。
- 在 VS Code 打开 `byov` 项目时，可以直接通过 `outputs/` 查看所有实验文件。

推荐命令：

```bash
ln -s /mnt/data/wzh/experiments/byov outputs
```

`outputs/` 需要加入 `.gitignore`。

### 2.15 断点续训继续写入同一个 run 目录

当前阶段只考虑真正的断点续训：训练中断后，从同一次实验的 `last.ckpt` 继续训练。

建议新增参数：

```bash
--resume /path/to/existing/run_dir
```

当传入 `--resume` 时，训练脚本应自动读取：

```text
<run_dir>/checkpoints/last.ckpt
```

并继续写入原来的 run 目录：

```text
<run_dir>/
  logs/
  tensorboard/
  checkpoints/
  metrics/
  artifacts/
```

也就是说，中断前后的这些内容都放在一起：

- `logs/train.log`
- `tensorboard/`
- `checkpoints/`
- `metrics/`
- `artifacts/`

这样 TensorBoard 曲线是连续的，checkpoint 和评估结果也不会被拆到多个目录。

建议额外记录一次恢复事件，例如写入：

```text
logs/resume.log
```

示例内容：

```text
2026-07-10 18:20:11 resume from checkpoints/last.ckpt
2026-07-10 18:20:11 continue in run_dir: /mnt/data/wzh/experiments/byov/break_eggs/20260710_153012_baseline_clip_bs4_lr1e-5_seed42
```

注意：当前阶段暂不设计“用旧 checkpoint 开新实验”的逻辑，先避免参数过多。

### 2.16 run 目录名包含关键超参数

仅有时间戳不方便查找实验。建议 run 目录名包含：

- 时间戳
- 人工命名的 `run_name`
- backbone
- batch size
- learning rate
- seed

推荐格式：

```text
<timestamp>_<run_name>_<backbone>_bs<batch_size>_lr<lr>_seed<seed>
```

示例：

```text
20260710_153012_baseline_clip_bs4_lr1e-5_seed42
```

完整路径：

```text
/mnt/data/wzh/experiments/byov/break_eggs/20260710_153012_baseline_clip_bs4_lr1e-5_seed42/
```

目录名用于人工快速查找，完整且权威的配置仍然以：

```text
config/args.json
```

为准。

### 2.17 best checkpoint 按 val/loss 记录 epoch

当前阶段 best checkpoint 先固定按照训练目标选择，也就是：

```text
val/loss
```

规则：

- `val/loss` 越小越好。
- `best.ckpt` 作为稳定入口，方便脚本加载。
- 额外保留带 epoch 和指标值的 checkpoint，方便人工查看。
- 保存 `metrics/best.json` 记录最佳模型信息。

推荐 checkpoint 结构：

```text
checkpoints/
  epoch=009.ckpt
  epoch=019.ckpt
  epoch=029.ckpt
  last.ckpt
  best.ckpt
  best-epoch=119-val_loss=0.4321.ckpt
```

推荐 `metrics/best.json`：

```json
{
  "monitor": "val/loss",
  "mode": "min",
  "best_epoch": 119,
  "best_score": 0.4321,
  "checkpoint": "checkpoints/best-epoch=119-val_loss=0.4321.ckpt"
}
```

### 2.18 scripts/run.sh 与命令行参数的关系

项目中已有：

```text
scripts/run.sh
```

建议保留它，但让它成为“标准训练命令模板”，不要隐藏复杂逻辑。

推荐原则：

- `train.py` 是权威入口，所有关键配置都通过命令行参数表达。
- `scripts/run.sh` 负责保存常用数据集的推荐参数，减少手动输入错误。
- 脚本内部不要写死 checkpoint。
- 脚本展开后的完整命令仍然写入 `config/command.txt`。

推荐使用方式：

```bash
bash scripts/run.sh break_eggs baseline 42
```

脚本内部展开为：

```bash
python train.py \
  --dataset break_eggs \
  --output_root /mnt/data/wzh/experiments/byov \
  --run_name baseline \
  --seed 42 \
  --batch_size 4 \
  --lr 1e-5 \
  --freeze_base
```

如果临时调试，也可以直接命令行运行：

```bash
python train.py --dataset break_eggs --run_name debug --epochs 3
```

## 3. 修改后的模拟训练输出

假设运行：

```bash
python train.py \
  --dataset break_eggs \
  --output_root /mnt/data/wzh/experiments/byov \
  --run_name baseline \
  --seed 42 \
  --batch_size 4 \
  --lr 1e-5 \
  --epochs 300 \
  --save_every 10 \
  --ds_every_n_epoch 10 \
  --freeze_base
```

### 3.1 自动创建的 run 目录

```text
/mnt/data/wzh/experiments/byov/break_eggs/20260710_153012_baseline_clip_bs4_lr1e-5_seed42/
```

### 3.2 最终目录结构

```text
20260710_153012_baseline_clip_bs4_lr1e-5_seed42/
  config/
    args.json
    command.txt
    git.txt
    env.txt

  logs/
    train.log

  tensorboard/
    events.out.tfevents.20260710...

  checkpoints/
    epoch=009.ckpt
    epoch=019.ckpt
    epoch=029.ckpt
    ...
    epoch=299.ckpt
    last.ckpt
    best.ckpt
    best-epoch=119-val_loss=0.4321.ckpt

  metrics/
    metrics.csv
    metrics.jsonl
    best.json
    downstream_epoch_009.json
    downstream_epoch_019.json
    downstream_epoch_029.json
    ...
    downstream_epoch_299.json

  artifacts/
    embeddings/
      epoch_009/
        train_embeds.npy
        train_label.npy
        val_embeds.npy
        val_label.npy
      epoch_019/
        train_embeds.npy
        train_label.npy
        val_embeds.npy
        val_label.npy
      epoch_029/
        train_embeds.npy
        train_label.npy
        val_embeds.npy
        val_label.npy
```

### 3.3 控制台输出示例

```text
[2026-07-10 15:30:12] Run directory:
  /mnt/data/wzh/experiments/byov/break_eggs/20260710_153012_baseline_clip_bs4_lr1e-5_seed42

[2026-07-10 15:30:12] Saved config:
  config/args.json
  config/command.txt
  config/git.txt
  config/env.txt

[2026-07-10 15:30:13] Dataset: break_eggs
[2026-07-10 15:30:13] Train videos: xx
[2026-07-10 15:30:13] Val videos: xx
[2026-07-10 15:30:14] Base model frozen

[2026-07-10 15:30:15] Start training from scratch

Epoch 1/300
  train/loss_step: 0.8421
  train/loss_epoch: 0.8014
  val/loss: 0.7742

Epoch 10/300
  train/loss_epoch: 0.6128
  val/loss: 0.5901
  saved checkpoint:
    checkpoints/epoch=009.ckpt

[Downstream eval: epoch 10]
  saved embeddings:
    artifacts/embeddings/epoch_009/

  classification/regular_f1: 0.4312
  classification/ego2exo_val_f1: 0.4021
  classification/exo2ego_val_f1: 0.4158

  retrieval/regular_map10: 0.2875
  retrieval/ego2exo_val_map10: 0.2664
  retrieval/exo2ego_val_map10: 0.2719

  progression/train_score: 0.6112
  progression/val_score: 0.5021

  kendall/train_tau: 0.4520
  kendall/val_tau: 0.3844

  saved metrics:
    metrics/downstream_epoch_009.json
```

### 3.4 train.log 内容

`logs/train.log` 应保存与控制台一致或更完整的信息。它用于训练结束后追溯细节。

示例：

```text
2026-07-10 15:30:12 INFO train.py: Run directory: /mnt/data/wzh/experiments/byov/break_eggs/20260710_153012_baseline_clip_bs4_lr1e-5_seed42
2026-07-10 15:30:12 INFO train.py: Command: python train.py --dataset break_eggs --run_name baseline ...
2026-07-10 15:30:13 INFO dataset: Train videos: xx
2026-07-10 15:30:13 INFO dataset: Val videos: xx
2026-07-10 15:30:14 INFO model: Base model frozen
2026-07-10 15:42:30 INFO train: Epoch 10 downstream metrics saved to metrics/downstream_epoch_009.json
```

### 3.5 TensorBoard 指标

TensorBoard 中应看到：

```text
train/loss_step
train/loss_epoch
val/loss

classification/regular_f1
classification/ego2exo_val_f1
classification/exo2ego_val_f1

retrieval/regular_map10
retrieval/ego2exo_val_map10
retrieval/exo2ego_val_map10

progression/train_score
progression/val_score

kendall/train_tau
kendall/val_tau
```

查看全部实验：

```bash
tensorboard --logdir /mnt/data/wzh/experiments/byov
```

查看单次实验：

```bash
tensorboard --logdir /mnt/data/wzh/experiments/byov/break_eggs/20260710_153012_baseline_clip_bs4_lr1e-5_seed42/tensorboard
```

## 4. 推荐执行顺序

建议按以下顺序修改代码：

1. 增加 `--output_root`、`--run_name`、`--seed`、`--resume` 参数。
2. 删除 `train.py` 中硬编码 checkpoint。
3. 实现唯一 run 目录创建逻辑，目录名包含 backbone、bs、lr、seed。
4. 实现真正断点续训：`--resume <run_dir>` 自动读取 `<run_dir>/checkpoints/last.ckpt`，并继续写入原 run 目录。
5. 将 `args.json`、命令、git 信息、环境信息写入 `config/`。
6. 配置 logging，输出到 `logs/train.log`。
7. 配置 TensorBoardLogger，输出到 `tensorboard/`。
8. 配置 checkpoint，输出到 `checkpoints/`。
9. best checkpoint 固定按 `val/loss` 选择，并记录 `metrics/best.json`。
10. 恢复 `val/loss` 记录。
11. 将 downstream embedding 改为按 epoch 写入 `artifacts/embeddings/epoch_xxx/`。
12. 将 downstream metrics 写入 `metrics/downstream_epoch_xxx.json`。
13. 更新 `scripts/run.sh`，让它成为标准训练命令模板。
14. 更新 `.gitignore`，避免提交训练产物。
15. 可选：在项目目录创建 `outputs` 软链接，指向真实实验目录。

## 5. 核心原则

规范后的训练输出应遵守以下原则：

- 一次训练一个 run 目录。
- 一个 run 目录包含复现实验需要的全部信息。
- 不同 run 之间不能互相覆盖。
- TensorBoard、checkpoint、metrics、embedding 都能对应到同一个 epoch。
- 大体积产物尽量不放在代码仓库中。
- 命令行能明确表达是否从 checkpoint 恢复。
