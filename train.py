import logging
import json
import os
import sys

from utils.config import argparser
from utils.experiment import (
    maybe_create_outputs_symlink,
    prepare_experiment,
    save_run_metadata,
    setup_logging,
    update_best_checkpoint,
)


def build_trainer(args):
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

    class BestCheckpointMetadata(Callback):
        def __init__(self, checkpoint_callback, monitor="val/loss", mode="min",
                     stable_filename=None, metadata_filename="best.json"):
            self.checkpoint_callback = checkpoint_callback
            self.monitor = monitor
            self.mode = mode
            self.stable_filename = stable_filename
            self.metadata_filename = metadata_filename

        def on_validation_end(self, trainer, pl_module):
            update_best_checkpoint(
                self.checkpoint_callback.best_model_path,
                self.checkpoint_callback.best_model_score,
                pl_module.args.metrics_dir,
                pl_module.args.checkpoints_dir,
                monitor=self.monitor,
                mode=self.mode,
                stable_filename=self.stable_filename,
                metadata_filename=self.metadata_filename,
            )

        def on_train_end(self, trainer, pl_module):
            self.on_validation_end(trainer, pl_module)

    periodic_checkpoint = ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        filename="epoch={completed_epochs:03.0f}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        save_last=False,
        every_n_epochs=args.save_every,
    )
    best_checkpoint = ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        filename="best-val_loss-epoch={completed_epochs:03.0f}-val_loss={val_loss:.4f}",
        auto_insert_metric_name=False,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    tensorboard_logger = TensorBoardLogger(
        save_dir=args.run_dir,
        name="tensorboard",
        version="",
    )
    csv_logger = CSVLogger(
        save_dir=args.run_dir,
        name="metrics",
        version="",
    )

    latest_checkpoint = ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        filename="last-epoch={completed_epochs:03.0f}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=False,
        every_n_epochs=1,
    )
    callbacks = [
        periodic_checkpoint,
        latest_checkpoint,
        best_checkpoint,
        BestCheckpointMetadata(best_checkpoint),
    ]

    if args.ds_every_n_epoch > 0:
        downstream_monitors = {
            '1': ('classification', 'checkpoint_classification'),
            '2': ('retrieval', 'checkpoint_retrieval'),
            '3': ('progression', 'checkpoint_progression'),
            '4': ('kendall', 'checkpoint_kendall'),
        }
        for task_id, (metric_name, monitor_name) in downstream_monitors.items():
            if task_id not in args.eval_task:
                continue
            checkpoint = ModelCheckpoint(
                dirpath=args.checkpoints_dir,
                filename=(
                    f'best-{metric_name}-epoch={{completed_epochs:03.0f}}'
                    f'-score={{{monitor_name}:.4f}}'
                ),
                auto_insert_metric_name=False,
                monitor=monitor_name,
                mode='max',
                save_top_k=1,
                every_n_epochs=args.ds_every_n_epoch,
            )
            callbacks.extend([
                checkpoint,
                BestCheckpointMetadata(
                    checkpoint,
                    monitor=monitor_name,
                    mode='max',
                    metadata_filename=f'best_{metric_name}.json',
                ),
            ])

    trainer_options = {}
    if args.smoke_test:
        trainer_options.update(
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
        )

    return Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        callbacks=callbacks,
        max_epochs=args.epochs if not args.smoke_test else 1,
        default_root_dir=args.run_dir,
        logger=[tensorboard_logger, csv_logger],
        log_every_n_steps=4,
        **trainer_options,
    )


def validate_training_inputs(args):
    import cv2
    import torch

    if args.num_gpus != 1:
        raise ValueError('The current BYOV training path is validated for one GPU; use --num_gpus 1')
    if not torch.cuda.is_available():
        raise RuntimeError('BYOV training requires an available CUDA GPU')
    if not args.freeze_base:
        raise ValueError('Paper reproduction requires the CLIP backbone to be frozen; pass --freeze_base')
    if args.save_every < 1:
        raise ValueError('--save_every must be at least 1')
    if args.ds_every_n_epoch < 0:
        raise ValueError('--ds_every_n_epoch cannot be negative')
    unknown_tasks = set(args.eval_task) - set('1234')
    if unknown_tasks:
        raise ValueError(f'Unknown downstream task identifiers: {sorted(unknown_tasks)}')
    if args.ds_every_n_epoch > 0 and not args.eval_task:
        raise ValueError('--eval_task cannot be empty when periodic downstream evaluation is enabled')
    if args.num_frames < 2:
        raise ValueError('--num_frames must be at least 2 for STM')
    if not (0 < args.topk_ratio <= 1):
        raise ValueError('--topk_ratio must be in (0, 1]')
    if not (0 < args.mask_ratio < 0.5):
        raise ValueError('--mask_ratio must be in (0, 0.5) because BYOV also uses 2x masking')

    dataset_dir = os.path.join(args.dataset_root, args.dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f'Dataset directory not found: {dataset_dir}')
    required = []
    if args.dataset in ('break_eggs', 'tennis_forehand'):
        for view in (args.view1, args.view2):
            for split in ('train', 'val'):
                required.append(os.path.join(dataset_dir, view, f'{split}.csv'))
    else:
        for split in ('train', 'val'):
            for view in (args.view1, args.view2):
                required.append(os.path.join(dataset_dir, split, view))
    if args.ds_every_n_epoch > 0:
        required.append(os.path.join(dataset_dir, 'label.pickle'))
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError('Training inputs are missing: ' + ', '.join(missing))

    split_stats = {}
    for split in ('train', 'val'):
        video_paths = []
        if args.dataset in ('break_eggs', 'tennis_forehand'):
            for view in (args.view1, args.view2):
                csv_path = os.path.join(dataset_dir, view, f'{split}.csv')
                with open(csv_path, 'r') as f:
                    video_paths.extend(
                        os.path.join(dataset_dir, view, line.strip())
                        for line in f if line.strip()
                    )
        else:
            for view in (args.view1, args.view2):
                view_dir = os.path.join(dataset_dir, split, view)
                video_paths.extend(
                    os.path.join(view_dir, name)
                    for name in os.listdir(view_dir) if name.endswith('.mp4')
                )
        video_paths = sorted(set(video_paths))
        missing_videos = [path for path in video_paths if not os.path.isfile(path)]
        if missing_videos:
            raise FileNotFoundError(
                f'{split} split references missing videos: ' + ', '.join(missing_videos[:10])
            )
        frame_counts = []
        for path in video_paths:
            capture = cv2.VideoCapture(path)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if frame_count < 2:
                raise ValueError(f'Video has fewer than two readable frames: {path}')
            frame_counts.append((frame_count, path))
        if not frame_counts:
            raise ValueError(f'No videos found in the {split} split')
        max_frames, max_video = max(frame_counts)
        split_stats[split] = {
            'videos': len(frame_counts),
            'total_frames': sum(count for count, _path in frame_counts),
            'min_frames': min(count for count, _path in frame_counts),
            'max_frames': max_frames,
            'max_video': os.path.relpath(max_video, dataset_dir),
        }

    stats_path = os.path.join(args.config_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({'dataset': args.dataset, 'splits': split_stats}, f, indent=2)
    logging.getLogger(__name__).info('Dataset frame statistics: %s', split_stats)

    if not os.path.isdir(args.vision_encoder_path):
        raise FileNotFoundError(f'Vision encoder directory not found: {args.vision_encoder_path}')
    weights = (
        'model.safetensors', 'pytorch_model.bin',
        'model.safetensors.index.json', 'pytorch_model.bin.index.json',
    )
    missing_model_files = []
    if not os.path.isfile(os.path.join(args.vision_encoder_path, 'config.json')):
        missing_model_files.append('config.json')
    if not any(os.path.isfile(os.path.join(args.vision_encoder_path, name)) for name in weights):
        missing_model_files.append('Transformers PyTorch/safetensors weights (single or sharded)')
    if missing_model_files:
        raise FileNotFoundError(
            f'Incompatible Transformers CLIP directory {args.vision_encoder_path}; missing: '
            + ', '.join(missing_model_files)
        )

    logger = logging.getLogger(__name__)
    if args.dp_rate != 0:
        logger.warning(
            'dp_rate=%s is recorded but is not wired into the released BYOV blocks; '
            'effective BYOV dropout remains 0 to preserve released-code behavior',
            args.dp_rate,
        )


def main(args):
    from pytorch_lightning import seed_everything

    from video_tasks import VideoAlignment

    logger = logging.getLogger(__name__)
    seed_everything(args.seed, workers=True)
    validate_training_inputs(args)
    task = VideoAlignment(args)
    trainer = build_trainer(args)

    logger.info("Run directory: %s", args.run_dir)
    if args.resume_ckpt:
        if not os.path.isfile(args.resume_ckpt):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_ckpt}")
        logger.info("Resume training from %s", args.resume_ckpt)
    else:
        logger.info("Start training from scratch")

    if args.eval_only:
        trainer.test(task, ckpt_path=args.ckpt)
    else:
        trainer.fit(task, ckpt_path=args.resume_ckpt or None)


if __name__ == "__main__":
    cli_args = argparser.parse_args()
    args, is_resume = prepare_experiment(cli_args)
    logger = setup_logging(args.logs_dir)
    save_run_metadata(args, sys.argv, is_resume)
    logger.info("Prepared run directory: %s", args.run_dir)
    if args.dry_run == "config":
        logger.info("Dry run config complete. Training dependencies, backbone, and dataset were not loaded.")
        sys.exit(0)
    maybe_create_outputs_symlink(args.output_root)
    main(args)
