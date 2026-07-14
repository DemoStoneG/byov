# BYOV 论文复现笔记

> 论文：Jungin Park, Jiyoung Lee, Kwanghoon Sohn, **Bootstrap Your Own Views: Masked Ego-Exo Modeling for Fine-grained View-invariant Video Representations**, CVPR 2025，arXiv:2503.19706v2。  
> 本文档依据 `docs/(CVPR 2025)Bootstrap Your Own Views Masked Ego-Exo Modeling for Fine-grained View-invariant Video Representations.pdf` 的正文与补充材料整理，并在“当前仓库对照”部分标出代码实现细节和风险。  
> 首要复现目标：可靠保存实验结果；明确视频帧如何送入 CLIP；明确 CLIP 输出如何变成 BYOV 的训练目标和下游表示。

## 1. 一句话概括方法

BYOV 冻结预训练图像编码器，逐帧取得 patch token；根据相邻帧同位置 patch token 的变化选出 top-K token 并求均值，得到每帧一个向量；随后用共享 Transformer encoder 将帧向量映射到跨视角潜空间，并以同视角掩码重建（MSM）学习时间因果性、以跨视角掩码重建（MCM）学习 ego/exo 视角不变性。训练后丢弃 decoder，只保留图像编码器和 BYOV encoder 产生逐帧表示。

## 2. 完整数据流与张量形状（CLIP ViT-B/16）

以下以论文主实验配置为准。`B` 是 batch size，`T=32`（Tennis Forehand 为 20），输入分辨率为 `224×224`，CLIP ViT-B/16 的 patch size 为 16。

```text
ego/exo 视频
  -> 从整段视频随机采样 T 帧，按时间排序
  -> resize、RGB、[0,1]、CLIP mean/std normalize
  -> frames: [B, T, 3, 224, 224]
  -> 合并 batch 和时间维: [B*T, 3, 224, 224]
  -> 冻结的 CLIPVisionModel
  -> last_hidden_state: [B*T, 197, 768]
  -> 删除第 0 个 CLS token
  -> patch tokens: [B*T, 196, 768]
  -> reshape: [B, T, 196, 768]
  -> 相邻帧差分 + top 30% token（约 59 个）
  -> 对选中 token 求均值
  -> frame targets X_bar: [B, T, 768]
  -> BYOV encoder（12 blocks，768 -> 256）
  -> frame latents Z: [B, T, 256]
  -> 训练：decoder 重建 [B, T, 768] 的 CLIP 帧目标
  -> 下游：丢弃 decoder，使用逐帧 Z
```

### 2.1 送入 CLIP 之前

论文明确给出的信息：

- Break Eggs、Pour Milk、Pour Liquid：训练时从覆盖整段视频的范围内随机采样 32 帧。
- Tennis Forehand：视频更短，训练采样 20 帧。
- Charades-Ego：训练采样 32 帧。
- AE2 下游评估使用全部帧，而不是只用训练时的 32/20 帧。
- ego/exo 即使本身同步，训练也不使用帧级时间对应关系。

论文没有明确写出 resize、颜色通道和归一化细节。当前官方仓库实现为：

1. OpenCV 读取/缓存帧，`cv2.resize(frame, (input_size, input_size), INTER_CUBIC)`，即直接拉伸成正方形，没有保持宽高比或 center crop。
2. BGR 转 RGB。
3. 转 `float32` 并除以 255。
4. 使用 CLIP 归一化：

   ```text
   mean = [0.48145466, 0.45782750, 0.40821073]
   std  = [0.26862954, 0.26130258, 0.27577711]
   ```

5. dataset 先产生 `[B, 2, T, H, W, 3]`，训练步骤分别取两段视频并 permute 为 `[B, T, 3, H, W]`。
6. embedder 再 reshape 为 `[B*T, 3, H, W]`，一次性逐帧送入 `CLIPVisionModel`。

这里使用的是 Hugging Face `CLIPVisionModel`，不是 `CLIPModel.get_image_features()`。因此需要的是视觉 Transformer 的全部 hidden tokens，而不是经过视觉投影层后的单个全局图像向量。

### 2.2 CLIP 输出的处理：Selective Token Merging（STM）

论文公式：对第 `t` 和 `t+1` 帧中空间位置相同的 token，先在通道维取绝对差的均值：

```text
s_t[n] = mean_d(|x_t[n, d] - x_{t+1}[n, d]|),  s_t ∈ R^N
n_t = topK(s_t)
```

然后取当前帧在这些位置的 K 个 token 并求平均，形成一个 `d` 维帧向量：

```text
x_bar_t = mean_{n in n_t}(x_t[n])
```

ViT-B/16 的 `N=196, d=768`，token selection ratio 为 0.3。论文将该过程称为 selective token merging，但实际操作是“按时间差选择 token，再平均”，没有可训练的 merging 模块。目的在于保留手-物交互等发生变化的局部区域并排除静态背景。

边界处理是复现要点：当前代码用 `t` 与 `t+1` 的差分选择第 `t` 帧 token；最后一帧没有后继帧，复用倒数第二帧与最后一帧的差分排名。

### 2.3 BYOV encoder 如何处理 CLIP 帧向量

每个视角独立经过同一个共享 encoder `g_phi`：

- 输入完整序列 `X_bar: [B,T,768]`，供 MCM 的另一个视角作为上下文，并产生下游表示。
- 输入 MSM 保留序列：mask ratio 0.4，即保留约 60% 帧。
- 输入 MCM 保留序列：mask ratio 0.8，即保留约 20% 帧。
- 先加入固定一维 sin-cos 时间位置编码，再经过 `LayerNorm + Linear(768,256)`。
- 经过 12 个 Transformer blocks 和最终 LayerNorm，得到 `[B,T,256]` 的逐帧 latent。

训练结束后：

- 冻结/丢弃 decoder。
- 冻结 `g_phi` 后训练 SVM 或线性回归器做下游评估。
- 帧分类、检索、phase progression 和 Kendall's tau 均直接使用逐帧 latent。
- Charades-Ego 的视频级表示是所有逐帧 latent 的平均值，再训练线性分类器。

不要把下列三种向量混为一谈：

| 名称 | 形状（B/16） | 用途 |
|---|---:|---|
| CLIP patch tokens | `[B,T,196,768]` | STM 的输入 |
| STM frame target `X_bar` | `[B,T,768]` | encoder 输入，也是 decoder 的 MSE 重建目标 |
| BYOV latent `Z` | `[B,T,256]` | 训练时给 decoder；训练后作为最终下游表示 |

## 3. 两个训练目标

### 3.1 MSM：Masked Self-view Modeling

对 ego 和 exo 各自随机删除 40% 的帧 token，只将保留的约 60% 帧送入 encoder。decoder 将 encoder 输出与可学习 mask token 恢复到完整时间顺序，并重建本视角原始 STM 帧向量。

MSM decoder attention 使用 causal mask：重建第 `t` 帧时只能关注此前 token。这是用来学习动作的因果时间进程，而不只是双向插值。

论文给出的损失是两个视角完整序列上的归一化 MSE 之和：

```text
L_MSM = ||X_bar_ego - Y_MSM_ego||^2 / T_ego
      + ||X_bar_exo - Y_MSM_exo||^2 / T_exo
```

当前仓库代码只在被 mask 的位置汇总 MSE，这一点比论文公式更具体，应在实验元数据中记录。

### 3.2 MCM：Masked Cross-view Modeling

对目标视角删除 80% 的帧 token，只保留约 20%；decoder 同时接收：

- 目标视角恢复位置后的少量可见 latent + mask tokens；
- 另一视角的完整 latent 序列。

```text
Y_MCM_ego = decoder(mask-filled Z_MCM_ego || Z_exo)
Y_MCM_exo = decoder(mask-filled Z_MCM_exo || Z_ego)
```

重建目标仍是各自的原始 STM/CLIP 帧向量 `[B,T,768]`。高达 80% 的目标视角 masking 迫使模型主要从另一视角恢复内容，从而把共享 encoder 推向视角不变的潜空间。

### 3.3 联合训练

论文目标：

```text
L_BYOV = L_MSM + L_MCM
```

论文说明两个 loss 的 forward/backward 分开执行但共享参数；当前仓库则在同一 `training_step` 内得到三个 masked MSE 并相加后一次 backward。若严格论文复现，应确认作者最终发布代码是否就是论文所说的“separately performed”，并把选择写入 run 配置。

## 4. 主实验配置

| 项目 | 论文主设置 |
|---|---|
| 图像编码器 | LAION-400M 预训练 CLIP ViT-B/16 |
| 图像编码器状态 | 冻结 |
| CLIP 输出 | 最后一层 hidden states，删除 CLS，保留 196×768 patch tokens |
| STM token ratio | 0.3 |
| encoder | 12 Transformer blocks，latent dim 256 |
| decoder | 4 Transformer blocks，dim 256，输出 dim 768 |
| encoder 参数量 | 9.7M |
| decoder 参数量 | 2.6M |
| MSM mask ratio | 0.4 |
| MCM mask ratio | 0.8 |
| 帧数 | AE2 前三个数据集 32；Tennis Forehand 20；Charades-Ego 32 |
| 训练配对 | 同一动作类别内的 unpaired、asynchronous ego/exo 视频，不用时间对应 |
| 推理 | decoder 丢弃；逐帧 CLIP→STM→encoder |

论文未报告 optimizer、learning rate、weight decay、batch size、epoch、scheduler、warmup、随机种子和硬件/训练时长。这些不能从论文擅自补全，必须来自代码、脚本或作者配置，并完整保存到每次 run。

当前仓库默认值为 Adam、`lr=1e-5`、`weight_decay=5e-6`、300 epochs、batch size 1、seed 42；这些属于代码默认值，不属于论文明确报告值。

## 5. 数据集与划分

AE2 benchmark 的每个数据集本身就是一个动作类别，内部标签是更细的 action phase。

| 数据集 | Train ego/exo | Val ego/exo | Test ego/exo | 固定 exo 机位 | ego-exo 同步 |
|---|---:|---:|---:|:---:|:---:|
| Break Eggs | 61 / 57 | 5 / 5 | 10 / 10 | 是 | 是（训练不用同步关系） |
| Pour Milk | 29 / 48 | 4 / 8 | 7 / 16 | 是 | 否/部分异步 |
| Pour Liquid | 70 / 67 | 10 / 9 | 19 / 18 | 否 | 否 |
| Tennis Forehand | 94 / 79 | 25 / 24 | 50 / 50 | 否 | 否 |

来源与 phase：

- Break Eggs：CMU-MMAC，43 位用户、5 类菜谱；关键事件定义 4 个 phase。
- Pour Milk：H2O，10 位用户，每场景 1 个 ego 和 4 个静态 exo；3 个 phase。
- Pour Liquid：ego 来自 EPIC-Kitchens 的 “pour water”，exo 来自 HMDB51 的 “pour”；环境和数据源均不同；3 个 phase。
- Tennis Forehand：exo 来自 Penn Action，ego 为 12 位球员使用 GoPro HERO8 采集；2 个 phase。

## 6. 下游评估协议

| 任务 | 评估方式 | 指标 |
|---|---|---|
| Action phase classification | 冻结表示，训练 SVM | F1；Regular、Ego2Exo、Exo2Ego |
| Frame retrieval | latent 上 nearest-neighbor，不训练 | mAP@5/10/15；正文主报 mAP@10 |
| Phase progression | 冻结表示，训练线性回归器预测归一化 phase 时间进度 | 平均 R² |
| Temporal alignment | 从视频 A 取帧对，在视频 B 中各取 NN，比较前后顺序一致性 | Kendall's tau |
| Charades-Ego action recognition | 逐帧 latent 平均成视频向量，linear probing | mAP；Regular、Ego2Exo、Exo2Ego |

其中 Regular 使用两个视角训练/检索；Ego2Exo 表示用 ego 训练或查询、在 exo 测试或检索；Exo2Ego 反之。

## 7. 论文目标结果（复现对照值）

### 7.1 AE2 主结果：BYOV ViT-B/16

| 数据集 | 分类 F1 Reg/E2X/X2E | 检索 mAP@10 Reg/E2X/X2E | Progression R² | Kendall tau |
|---|---:|---:|---:|---:|
| Break Eggs | 74.30 / 75.01 / 71.28 | 67.17 / 70.65 / 69.02 | 0.8533 | 0.9451 |
| Pour Milk | 86.46 / 85.09 / 86.61 | 89.42 / 87.73 / 85.06 | 0.8992 | 0.9466 |
| Pour Liquid | 79.48 / 71.83 / 76.23 | 71.06 / 75.03 / 70.03 | 0.4483 | 0.3052 |
| Tennis Forehand | 89.12 / 94.47 / 85.73 | 90.61 / 88.34 / 88.94 | 0.7881 | 0.7852 |

### 7.2 仅使用原始 CLIP 特征的基线

| 数据集 | 分类 F1 Reg/E2X/X2E | 检索 mAP@10 Reg/E2X/X2E | Progression R² | Kendall tau |
|---|---:|---:|---:|---:|
| Break Eggs | 51.66 / 27.97 / 26.24 | 44.46 / 35.85 / 35.70 | 0.0402 | 0.0168 |
| Pour Milk | 43.24 / 49.21 / 30.94 | 52.16 / 46.39 / 40.34 | -4.0754 | 0.0046 |
| Pour Liquid | 60.60 / 36.97 / 48.43 | 43.63 / 47.58 / 37.02 | -0.3139 | -0.0048 |
| Tennis Forehand | 67.81 / 43.41 / 44.22 | 74.54 / 59.57 / 52.02 | -0.4996 | 0.0618 |

这组基线很重要：它验证数据进入 CLIP 和下游 evaluator 的链路。如果复现出的 CLIP-only 数值已经严重偏离，应先排查预处理、数据划分、标签和评估代码，不能直接归因于 BYOV 训练。

### 7.3 Charades-Ego

| 方法 | Regular | Ego2Exo | Exo2Ego |
|---|---:|---:|---:|
| CLIP ViT-B/16 | 13.7 | 8.3 | 10.7 |
| BYOV ViT-B/16 | 31.8 | 26.5 | 27.3 |

### 7.4 核心消融（Break Eggs）

论文表 3 的 F1 与 mAP 是三个视角设置的汇总口径，因此不应和表 1 的单列直接比较。

| 方法 | F1 | mAP@10 | Progression | tau |
|---|---:|---:|---:|---:|
| BYOV | 73.53 | 68.95 | 0.8533 | 0.9451 |
| 去掉 token selection | 71.84 | 67.55 | 0.8224 | 0.9016 |
| 去掉 causal mask | 72.29 | 68.10 | 0.6420 | 0.7091 |
| 去掉 MSM | 62.12 | 60.34 | 0.4362 | 0.4906 |
| 去掉 MCM | 57.43 | 55.83 | 0.6724 | 0.7086 |

补充材料的最佳默认超参为 STM/MSM/MCM=`30%/40%/80%`。latent size 在 Break Eggs 上以 256 综合最佳；512 虽提高部分检索项，却因训练数据有限导致整体下降。

## 8. 实验结果保存规范（本项目必须执行）

论文没有规定磁盘格式；本仓库已在 `docs/training_output_standardization.md` 定义 run 目录。为了能与上述目标表逐项核对，每个正式 run 至少保存：

```text
run_dir/
  config/
    args.json                 # 全部参数，不能只保存非默认参数
    command.txt
    git.txt                   # commit、branch、dirty diff 状态
    env.txt                   # Python/CUDA/PyTorch/transformers/timm/CLIP 权重标识
  logs/train.log
  tensorboard/
  checkpoints/{last,best,...}.ckpt
  metrics/
    metrics.csv               # 每 epoch 的 train/MSM/MCM/total/val loss
    downstream_epoch_XXX.json
    test_best.json            # 最终 test 指标，不以 val 指标冒充
  artifacts/embeddings/
    epoch_XXX/{train,val,test}_embeds.npy
    epoch_XXX/{train,val,test}_label.npy
```

`downstream_epoch_XXX.json` 建议固定 schema：

```json
{
  "dataset": "break_eggs",
  "split": "val",
  "checkpoint": "epoch=XXX.ckpt",
  "classification": {"regular_f1": 0.0, "ego2exo_f1": 0.0, "exo2ego_f1": 0.0},
  "retrieval": {"regular_map10": 0.0, "ego2exo_map10": 0.0, "exo2ego_map10": 0.0},
  "progression": {"r2": 0.0},
  "kendall": {"tau": 0.0}
}
```

还必须保存这些容易造成结果漂移的信息：

- 数据集文件清单或 hash、train/val/test 划分、ego/exo 判定方式。
- CLIP 权重的精确目录/模型 revision，而不只是字符串 “clip”。
- resize/crop/interpolation、RGB 顺序、mean/std、输入 dtype。
- 帧采样的实际索引、随机种子、每段视频帧数；至少在 debug run 中保存。
- `num_tokens`、STM/MSM/MCM ratio、实际 `topk` 和实际保留帧数（整数取整规则）。
- best checkpoint 的选择依据。训练 loss 最低不保证四个下游指标最佳，建议同时保存 `best-val_loss` 和每个下游主指标的 best epoch。
- 三个独立 loss：当前复现分支已分别记录 `train/val loss_msm`、`loss_mcm_view1`、`loss_mcm_view2` 以及 total loss，用于诊断某一预测分支是否失效。

## 9. 当前仓库与论文对照时发现的高风险点

这些是代码审阅结果，不是论文声称的结论。正式训练前应逐项做单元测试或最小 forward 验证。

1. `utils/config.py` 中 `topk_ratio`、`mask_ratio` 和 `dp_rate` 的类型问题已经在当前复现分支修正为 `float`；正式运行仍应从 `config/args.json` 核对其实际值为 0.3、0.4 和 0.1。
2. 论文要求冻结 CLIP；当前代码只有显式传 `--freeze_base` 才设置 `requires_grad=False`，虽然训练步骤对输出调用 `.detach()`，仍建议显式冻结并记录。
3. `Embedder.token_selection()` 的 `topk_idx` 只返回最后一个 batch/时间位置的索引，当前下游未使用，但如果要可视化 STM，不能把它当成所有帧的索引。
4. `byov_decoder.forward()` 原发布代码把 `x2_r1` 的预测写成了 `decoder_pred(x2_r2)`；当前复现分支已修正为 `decoder_pred(x2_r1)`，使第二个 MSM 分支使用其自身的小掩码表示。正式实验必须记录这一修正。
5. 当前发布代码使用 `merge_all=True`，将 ego/exo 路径合并为一个集合，再独立随机抽两段视频。因此训练明确允许 ego-ego、exo-exo 和 ego-exo 组合；代码复现应保留该行为，不应擅自改为强制一 ego 一 exo。这里的 unpaired 表示不依赖视频或帧级对应关系。
6. 训练采样在视频帧数不足 `num_frames` 时生成 `arange(0,num_frames)` 再 clamp，会重复最后一帧；必须记录短视频行为。
7. 论文公式的 MSM/MCM 是视角两项之和；当前代码把 MSM 合成一项、两个方向 MCM 各一项，且只在 masked 位置算损失。复现报告要明确采用“论文公式口径”还是“发布代码口径”。
8. 论文说两个 loss 分别 forward/backward；当前实现相加后共同 backward。两者可能影响优化轨迹。
9. 当前配置默认 `n_heads=16`，256 维 latent 可以整除；论文只报告层数与 latent dim，没有报告 head 数。该参数必须随结果保存。
10. 当前代码按 `val/loss` 选 best checkpoint，而论文最终指标全是下游指标。建议同时报告 last、best-val-loss 和 best-downstream，避免 checkpoint 选择造成不可比。

## 10. 推荐复现顺序与验收点

1. **CLIP-only 基线**：不训练 BYOV，按论文 preprocessing/STM/evaluator 跑四个数据集；至少先复现表 1 的 CLIP features 数量级。
2. **单 batch 张量验收**：保存一批视频的帧索引、输入图像、CLIP `[B,T,196,768]`、STM top-K 索引、`X_bar [B,T,768]`、`Z [B,T,256]`。
3. **STM 可视化**：把 top-K patch 叠加回图像，检查是否覆盖手、容器、球拍等动态区域。
4. **loss 分支测试**：分别只开 MSM、只开 MCM，检查 loss 非 NaN、mask 数量正确、所有预期参数有梯度。
5. **Break Eggs 完整训练**：它是论文消融使用的数据集，先对齐默认结果与消融趋势。
6. **四数据集主表**：固定相同代码版本与模型选择协议，写出论文值、复现值、绝对差和多 seed 均值/标准差。
7. **最终 test**：只在超参和 checkpoint 策略用 val 固定之后运行，test 结果单独落盘。

建议最终汇总表：

| Dataset | Metric | Paper | Reproduced mean | Std | Delta | Seeds | Checkpoint rule |
|---|---|---:|---:|---:|---:|---|---|
| Break Eggs | Kendall tau | 0.9451 |  |  |  |  |  |

## 11. 论文未说明、复现中必须补证的信息

- optimizer 之外是否有 scheduler、warmup、gradient clipping、AMP。
- batch size、epoch、数据增强、训练硬件与时长。
- 发布代码不强制一 ego 一 exo，而是从合并视角池独立采样；复现实验应记录并保留该采样策略。
- 相同动作类别内视频对的 epoch 长度和采样分布。
- SVM 的 kernel/C/标准化、线性回归超参和 nearest-neighbor 距离是否对 latent 做 L2 normalize。
- 使用哪个 epoch/checkpoint 生成论文表格。
- 论文公式计算所有位置 MSE，而代码计算 masked-only MSE 的最终口径。

这些项目应通过官方代码、发布 checkpoint 和最小实验逐一确定；未确认前应在实验报告中标为“未知/当前实现选择”，不能伪装成论文设定。

## 12. 当前项目的正式测试参数

### 12.1 使用官方预计算 embedding 的前提

每个数据集的官方 checkpoint 目录应包含：

```text
AE2_ckpts/
  break_eggs.ckpt
  break_eggs_eval/
    train_embeds.npy
    train_label.npy
    val_embeds.npy
    val_label.npy
  pour_milk.ckpt
  pour_milk_eval/{train,val}_{embeds,label}.npy
  pour_liquid.ckpt
  pour_liquid_eval/{train,val}_{embeds,label}.npy
  tennis_forehand.ckpt
  tennis_forehand_eval/{train,val}_{embeds,label}.npy
```

预计算 embedding 模式不会重新执行 CLIP/BYOV forward，但仍需要完整 `AE2_data`，以恢复视频顺序、每段视频帧数以及 ego/exo 身份。程序会检查：

```text
embedding 行数 == label 数量 == 对应 split 的视频总帧数
train 与 val/test embedding 的维度相同
```

预计算模式会记录 NPY 的实际 embedding 维度，但不会仅凭 `--backbone base/large` 强制它必须是 256/512，因为此时并未加载 backbone。若实际维度为 128，则下游四项仍可计算，但这不符合论文 Table 1 的 BYOV ViT-B/16 256 维主配置；必须先确认该 NPY 是 BYOV 结果还是 AE2/其他基线资产。

官方原始评估代码可能把 test split 固定保存成 `val_embeds.npy` 和 `val_label.npy`。因此需要区分：

- `--eval-mode test`：使用原始数据的 test 视频清单和边界。
- `--embedding-file-split val`：读取文件名以 `val_` 开头的官方 NPY。

若帧数校验失败，先确认 NPY 实际对应 test 还是 val，不可直接绕过校验。

### 12.2 单数据集命令

```bash
bash scripts/eval.sh \
  --dataset break_eggs \
  --checkpoint /root/autodl-tmp/datasets/AE2/AE2_ckpts/break_eggs.ckpt \
  --dataset-root /root/autodl-tmp/datasets/AE2/AE2_data \
  --output-root /root/autodl-tmp/experiments/byov_comparisons/clip_vit_b16 \
  --run-name official_precomputed_eval \
  --eval-mode test \
  --embedding-file-split val \
  --embedding-dir /root/autodl-tmp/datasets/AE2/AE2_ckpts/break_eggs_eval \
  --backbone base \
  --eval-tasks 1234
```

切换数据集只需要同步改变：

```text
--dataset
--checkpoint
--embedding-dir
```

例如 Pour Milk 对应：

```text
--dataset pour_milk
--checkpoint .../AE2_ckpts/pour_milk.ckpt
--embedding-dir .../AE2_ckpts/pour_milk_eval
```

### 12.3 四数据集批量命令

推荐用统一入口，保证四个数据集进入同一个 backbone 结果目录：

```bash
bash scripts/eval_all.sh \
  --checkpoint-root /root/autodl-tmp/datasets/AE2/AE2_ckpts \
  --dataset-root /root/autodl-tmp/datasets/AE2/AE2_data \
  --output-root /root/autodl-tmp/experiments/byov_comparisons \
  --backbone-label clip_vit_b16 \
  --backbone base \
  --eval-mode test \
  --embedding-file-split val \
  --eval-tasks 1234
```

上面的命令读取官方预计算 NPY，只适合在已经确认 NPY 来源和维度时复算下游指标。严格复现论文 Table 1 应重新执行 CLIP + BYOV encoder forward：

```bash
bash scripts/eval_all.sh \
  --checkpoint-root /root/autodl-tmp/datasets/BYOV/BYOV_ckpts \
  --dataset-root /root/autodl-tmp/datasets/AE2/AE2_data \
  --output-root /root/autodl-tmp/experiments/byov_comparisons \
  --backbone-label byov_clip_vit_b16 \
  --backbone base \
  --vision-encoder-path /root/autodl-tmp/ai_models/openai-clip-vit-base-patch16 \
  --eval-mode test \
  --eval-tasks 1234 \
  --extract-embedding
```

此模式忽略 `<checkpoint-root>/<dataset>_eval/*.npy`，从原始视频重新生成 embedding。每个 checkpoint 必须是 CLIP ViT-B/16 对应的 BYOV 256 维 encoder；仅有 128 维 AE2 checkpoint 不能复现 Table 1。

输出目录：

```text
byov_comparisons/
  clip_vit_b16/
    break_eggs/<run_dir>/
      config/
        args.json
        command.txt
        git.txt
        env.txt
        evaluation_plan.json
      logs/eval.log
      metrics/test.json
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

后续 backbone 使用同级目录，例如：

```text
byov_comparisons/
  clip_vit_b16/
  clip_vit_l14/
  resnet50/
  new_backbone/
```

### 12.4 哪些参数应由命令传入

这些参数描述本次实验的数据、文件来源和输出位置，应在 `eval.sh`/`eval_all.sh` 命令中显式传入：

| 参数 | 含义 |
|---|---|
| `--dataset` | 单数据集名称 |
| `--checkpoint` / `--checkpoint-root` | BYOV probe checkpoint 或四数据集 checkpoint 根目录 |
| `--dataset-root` | 完整 AE2 数据根目录 |
| `--output-root` | 结果根目录 |
| `--run-name` | 单次运行名称 |
| `--backbone-label` | 批量结果的 backbone 目录名 |
| `--embedding-dir` | 单数据集预计算 embedding 目录 |
| `--embedding-file-split` | NPY 文件前缀 `val` 或 `test` |
| `--eval-mode` | 实际数据元信息 split：`val` 或 `test` |
| `--eval-tasks` | `1234` 表示执行全部四项 |
| `--backbone` | 当前为 `base` 或 `large` |
| `--vision-encoder-path` | 重新提取 embedding 时使用的 CLIP 路径 |
| `--device` | `auto/cpu/cuda/cuda:N` |
| `--num-workers` | DataLoader worker 数量 |

### 12.5 哪些参数由代码自动绑定

| 配置 | CLIP ViT-B/16 | CLIP ViT-L/14 |
|---|---:|---:|
| `hidden_dim` | 768 | 1024 |
| `num_tokens` | 196 | 256 |
| `embedding_size` | 256 | 512 |
| `decoder_embedding_size` | 256 | 512 |
| backbone | 冻结 | 冻结 |

数据集帧数也由脚本自动选择：

```text
Break Eggs / Pour Milk / Pour Liquid: 32
Tennis Forehand: 20
```

论文默认 STM/MSM/MCM ratio 由底层配置保持为 `0.3/0.4/0.8`。所有最终解析值必须写入 `<run_dir>/config/args.json`。

## 13. 四项测试的最终模拟输出

> 本节只模拟输出结构。下面使用 Break Eggs 的论文目标结果作为示例；除论文明确报告的 test 指标外，帧数和 `train_r2` 等均是示意值，不能当作真实运行结果。

四个任务分别产生：

1. Classification：Regular、Ego2Exo、Exo2Ego F1；会在冻结 embedding 上拟合 SVM。
2. Frame Retrieval：Regular、Ego2Exo、Exo2Ego mAP@10；nearest-neighbor，无训练。
3. Phase Progression：R²；会在冻结 embedding 上拟合线性回归器。
4. Kendall's tau：时间顺序对齐分数；无训练。

`metrics/test.json` 固定先记录数据/参数，后记录四项结果：

```json
{
  "dataset": "break_eggs",
  "split": "test",
  "checkpoint": "/root/autodl-tmp/datasets/AE2/AE2_ckpts/break_eggs.ckpt",
  "embedding_source": "/root/autodl-tmp/datasets/AE2/AE2_ckpts/break_eggs_eval",
  "parameters": {
    "tasks": ["1", "2", "3", "4"],
    "embedding_file_split": "val",
    "uses_precomputed_embeddings": true,
    "fits_downstream_svm": true,
    "fits_downstream_linear_regressor": true,
    "device": null,
    "base_model_name": "clip",
    "backbone_frozen": true,
    "vision_encoder_path": "/mnt/data/wzh/ai_model/openai-clip-vit-base-patch16",
    "input_size": 224,
    "num_frames": 32,
    "num_tokens": 196,
    "backbone_hidden_dim": 768,
    "configured_probe_embedding_size": 256,
    "evaluated_embedding_size": 256,
    "token_selection_ratio": 0.3,
    "msm_mask_ratio": 0.4,
    "mcm_mask_ratio": 0.8
  },
  "data": {
    "precomputed_validation": {
      "val": {
        "embedding_shape": [1797, 256],
        "label_shape": [1797],
        "expected_frames": 1797
      },
      "train": {
        "embedding_shape": [18979, 256],
        "label_shape": [18979],
        "expected_frames": 18979
      }
    }
  },
  "classification": {
    "regular_f1": 0.7430,
    "ego2exo_f1": 0.7501,
    "exo2ego_f1": 0.7128,
    "fits_svm": true
  },
  "retrieval": {
    "regular_map10": 0.6717,
    "ego2exo_map10": 0.7065,
    "exo2ego_map10": 0.6902
  },
  "progression": {
    "train_r2": 0.9124,
    "eval_r2": 0.8533,
    "fits_linear_regressor": true
  },
  "kendall": {
    "eval_tau": 0.9451
  }
}
```

换算到论文表格显示形式：F1 和 mAP 乘以 100，R² 和 tau 保持原值。

```text
Dataset: Break Eggs

Classification F1
  Regular: 74.30
  Ego2Exo: 75.01
  Exo2Ego: 71.28

Frame Retrieval mAP@10
  Regular: 67.17
  Ego2Exo: 70.65
  Exo2Ego: 69.02

Phase Progression R²: 0.8533
Kendall's tau:       0.9451
```

批量执行结束后，`<output-root>/<backbone-label>/summary/all_results.json` 以数据集为一级 key：

```json
{
  "break_eggs": {
    "dataset": "break_eggs",
    "parameters": {},
    "classification": {},
    "retrieval": {},
    "progression": {},
    "kendall": {}
  },
  "pour_milk": {},
  "pour_liquid": {},
  "tennis_forehand": {}
}
```

每个数据集的完整参数和结果保留在对应对象中；空对象仅用于上述结构示意。
