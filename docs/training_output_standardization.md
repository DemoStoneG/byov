# Training Output Standardization

本文档记录 BYOV 训练输出规范和当前实现状态。目标是让每次实验都有独立目录，能追溯配置、日志、checkpoint、TensorBoard 和下游评估结果。

## 1. Run 目录

一次训练对应一个 run 目录：

```text
<output_root>/<dataset>/<timestamp>_<run_name>_<backbone>_bs<batch_size>_lr<lr>_seed<seed>/
```

示例：

```text
/mnt/data/wzh/experiments/byov/break_eggs/20260713_205138_baseline_clip_bs4_lr1e-5_seed42/
```

推荐真实实验输出放在数据盘：

```text
/mnt/data/wzh/experiments/byov/
```

正式训练时，代码会尝试在项目内创建软链接：

```text
outputs -> /mnt/data/wzh/experiments/byov
```

这样 VS Code 可以直接浏览实验文件，同时 Git 不追踪大文件。

## 2. 目录结构

每个 run 固定包含：

```text
<run_dir>/
  config/
    args.json
    command.txt
    git.txt
    env.txt
  logs/
    train.log
    resume.log          # only when using --resume
  tensorboard/
  checkpoints/
    epoch=009.ckpt
    last.ckpt
    best.ckpt
    best-epoch=119-val_loss=0.4321.ckpt
  metrics/
    metrics.csv
    best.json
    downstream_epoch_009.json
  artifacts/
    embeddings/
      epoch_009/
        train_embeds.npy
        train_label.npy
        val_embeds.npy
        val_label.npy
```

## 3. 常用命令

正式训练：

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

脚本模板：

```bash
bash scripts/run.sh break_eggs baseline 42
```

断点续训：

```bash
python train.py --resume /mnt/data/wzh/experiments/byov/break_eggs/<run_dir>
```

`--resume` 会自动读取：

```text
<run_dir>/checkpoints/last.ckpt
```

并继续写入同一个 run 目录。

## 4. Dry Run

用于检查配置和输出目录，不加载 PyTorch Lightning、backbone 或数据集：

```bash
python train.py \
  --dry_run config \
  --output_root /tmp/byov_dry_run \
  --dataset break_eggs \
  --run_name drytest \
  --batch_size 4 \
  --lr 1e-5 \
  --seed 42
```

它会创建 run 目录、`config/`、`logs/train.log`，但不会启动训练，也不会创建 `outputs` 软链接。

这个模式适合排查：

- 参数是否能解析。
- run 目录命名是否正确。
- 日志和配置文件是否能写入。
- 环境中 Lightning / matplotlib / libstdc++ 有问题时，训练入口的轻量部分是否正常。

## 5. TensorBoard 和指标

TensorBoard 写入：

```text
<run_dir>/tensorboard/
```

主要指标命名：

```text
train/loss_step
train/loss_epoch
val/loss
classification/regular_f1
retrieval/regular_map10
progression/val_score
kendall/val_tau
```

best checkpoint 当前固定按训练目标选择：

```text
monitor: val/loss
mode: min
```

代码内部额外记录 `val_loss`，用于 Lightning checkpoint 文件名和 monitor。

## 6. Git Ignore

实验产物不进入 Git：

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

## 7. 当前限制

- 当前真实 backbone 只有 `clip`。
- 真实训练仍需要可用的 PyTorch Lightning、GPU、backbone 权重和 AE2 数据集。
- `--dry_run config` 只能检查配置与输出目录，不能证明模型 forward、dataset 或训练 loop 正常。

