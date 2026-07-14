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
    epoch=010.ckpt
    last-epoch=083.ckpt
    best-val_loss-epoch=037-val_loss=0.4321.ckpt
    best-classification-epoch=080-score=0.9000.ckpt
    best-retrieval-epoch=100-score=0.8000.ckpt
    best-progression-epoch=120-score=0.7000.ckpt
    best-kendall-epoch=110-score=0.9500.ckpt
  metrics/
    metrics.csv
    best.json
    downstream_epoch_010.json
  artifacts/
    embeddings/
      epoch_010/
        train_embeds.npy
        train_label.npy
        val_embeds.npy
        val_label.npy
```

## 3. 常用命令

正式训练：

```bash
bash scripts/run.sh \
  --dataset break_eggs \
  --dataset-root /root/autodl-tmp/datasets/AE2/AE2_data \
  --output-root /root/autodl-tmp/experiments/byov_training \
  --vision-encoder-path /root/autodl-tmp/ai_models/transformers-clip-vit-b16 \
  --run-name baseline \
  --seed 42 \
  --ds-every 10
```

断点续训：

```bash
bash scripts/run.sh --resume /root/autodl-tmp/experiments/byov_training/break_eggs/<run_dir>
```

`--resume` 会自动读取：

```text
<run_dir>/checkpoints/last-epoch=NNN.ckpt
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

真正启动完整训练前还应运行 `--smoke-test`。它执行一个 train batch、一个 val batch、
backward、optimizer step 和 checkpoint/log 落盘。Smoke test 会自动设置
`ds_every_n_epoch=0`，避免为了入口检查而提取全量下游 embedding。

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

此外，启用周期下游验证时，会分别按 classification regular F1、retrieval regular
mAP@10、progression validation score 和 Kendall validation tau 保存对应 best checkpoint。
所有 checkpoint 文件名都使用从 1 开始的“已完成 epoch 数”。不使用 test 指标选择模型。

`ds_every_n_epoch=0` 只关闭周期下游验证。BYOV 主训练是自监督的，不读取
`label.pickle`；设置为正数时，代码才读取标签并每隔指定 epoch 跑四项下游验证。


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

## 8. 官方 Backbone + BYOV Probe 直接测试

默认运行论文 Table 1 的四项评估。整个过程中 backbone 和 BYOV probe/encoder 都保持冻结；任务 1 会在 train embedding 上拟合 SVM，任务 3 会拟合线性回归器，这是论文下游评估协议的一部分：

```bash
bash scripts/eval.sh \
  --dataset break_eggs \
  --checkpoint /path/to/official_probe.ckpt \
  --dataset-root /path/to/AE2_data \
  --output-root /path/to/experiments/byov \
  --run-name official_probe_test
```

默认按论文 Table 1 的 CLIP ViT-B/16 + 256 维 BYOV probe 执行。CLIP ViT-L/14 + 512 维 probe 使用：

```bash
bash scripts/eval.sh \
  --dataset break_eggs \
  --checkpoint /path/to/large_probe.ckpt \
  --dataset-root /path/to/AE2_data \
  --output-root /path/to/experiments/byov \
  --run-name official_large_test \
  --backbone large \
  --vision-encoder-path /path/to/openai-clip-vit-large-patch14
```

`eval.sh` 命令行负责本次运行会变化的参数：

```text
--dataset
--checkpoint
--dataset-root
--output-root
--run-name
--embedding-dir
--embedding-file-split
--eval-mode
--eval-tasks
--backbone
--vision-encoder-path
--device
--num-workers
```

代码/脚本内部固定管理与论文配置绑定的参数：Base/Large 对应的 hidden dimension、patch token 数、probe dimension，Tennis Forehand 的 20 帧与其他数据集的 32 帧，CLIP 冻结，以及默认的 STM/MSM/MCM ratio。底层 Python 参数仍完整保存在 `config/args.json`，正常评估不建议绕过 `eval.sh` 单独修改这些绑定项。

官方发布的 `*_eval` 目录如果包含 `train_embeds.npy`、`train_label.npy`、`val_embeds.npy`、`val_label.npy`，可跳过 backbone forward，直接运行四项评估：

```bash
bash scripts/eval.sh \
  --dataset break_eggs \
  --checkpoint /root/autodl-tmp/datasets/AE2/AE2_ckpts/break_eggs.ckpt \
  --dataset-root /root/autodl-tmp/datasets/AE2/AE2_data \
  --output-root /root/autodl-tmp/experiments/byov \
  --run-name official_precomputed_eval \
  --eval-mode test \
  --embedding-file-split val \
  --embedding-dir /root/autodl-tmp/datasets/AE2/AE2_ckpts/break_eggs_eval
```

官方原始评估代码可能把 test split 固定保存为 `val_embeds.npy`/`val_label.npy`，因此示例将数据 split 设为 `EVAL_MODE=test`，同时用 `EMBEDDING_FILE_SPLIT=val` 读取文件。正式运行前必须核对 NPY 行数是否等于 test split 的总帧数；若实际等于 val split，则两者都设为 `val`。

即使使用预计算 embedding，仍需 `DATASET_ROOT` 指向完整 AE2 数据目录，因为评估器需要读取视频划分、每段视频长度及 ego/exo 身份。此模式不会加载 checkpoint 做 forward；checkpoint 路径只作为实验来源记录。

每次评估创建独立 run，保存：

```text
<run_dir>/
  config/
    args.json
    command.txt
    git.txt
    env.txt
    evaluation_plan.json
    checkpoint_load.json
  logs/eval.log
  metrics/test.json
  artifacts/embeddings/
    test_embeds.npy
    test_label.npy
```

任务 2/4 不加载 train split。只有显式运行任务 1/3 时才额外生成 `train_embeds.npy` 和 `train_label.npy`，并拟合对应的 SVM/线性回归器。

如果要求连下游 SVM/线性回归也完全不拟合，只运行 training-free 的 frame retrieval 和 Kendall's tau：

```bash
bash scripts/eval.sh \
  --dataset break_eggs \
  --checkpoint /path/to/official_probe.ckpt \
  --dataset-root /path/to/AE2_data \
  --output-root /path/to/experiments/byov \
  --run-name retrieval_tau_only \
  --eval-tasks 24 \
  --no-downstream-fit
```

`--no_downstream_fit` 会拒绝任务 1 和任务 3。配置和输出目录可先做无依赖检查：

```bash
python evaluation/evaluate_features.py \
  --dry_run config \
  --output_root /tmp/byov_eval_check \
  --dataset break_eggs \
  --run_name direct_test \
  --eval_mode test \
  --eval_task 24 \
  --no_downstream_fit
```

## 9. 四数据集与多 Backbone 对比目录

切换数据集时只需要改变：

```text
--dataset
--checkpoint
--embedding-dir
```

数据目录读取规则、Tennis Forehand 的 20 帧以及其他数据集的 32 帧由代码自动选择。四个数据集可以用统一入口批量运行：

```bash
bash scripts/eval_all.sh \
  --checkpoint-root /root/autodl-tmp/datasets/AE2/AE2_ckpts \
  --dataset-root /root/autodl-tmp/datasets/AE2/AE2_data \
  --output-root /root/autodl-tmp/experiments/byov_comparisons \
  --backbone-label clip_vit_b16 \
  --backbone base \
  --eval-mode test \
  --embedding-file-split val
```

严格复现论文而不是复算现成 NPY 时，使用正确的 BYOV checkpoint 和 CLIP ViT-B/16 重新提取 embedding：

```bash
bash scripts/eval_all.sh \
  --checkpoint-root /root/autodl-tmp/datasets/BYOV/BYOV_ckpts \
  --dataset-root /root/autodl-tmp/datasets/AE2/AE2_data \
  --output-root /root/autodl-tmp/experiments/byov_comparisons \
  --backbone-label byov_clip_vit_b16 \
  --backbone base \
  --vision-encoder-path /root/autodl-tmp/ai_models/openai-clip-vit-base-patch16 \
  --eval-mode test \
  --extract-embedding
```

`--extract-embedding` 会忽略 `<checkpoint-root>/<dataset>_eval`，从原始视频执行 CLIP + BYOV encoder forward。Table 1 的 Base checkpoint 应对应 768 维 CLIP 输入和 256 维 BYOV latent。

输出结构：

```text
byov_comparisons/
  clip_vit_b16/
    break_eggs/<run_dir>/...
    pour_milk/<run_dir>/...
    pour_liquid/<run_dir>/...
    tennis_forehand/<run_dir>/...
    summary/
      break_eggs_test.json
      pour_milk_test.json
      pour_liquid_test.json
      tennis_forehand_test.json
      all_results.json
```

实现其他 backbone 后，只需使用新的 `--backbone-label`，例如 `clip_vit_l14` 或 `resnet50`，使其成为 `byov_comparisons` 下的兄弟目录。`summary/all_results.json` 汇总该 backbone 的四数据集结果，便于生成对比表。
