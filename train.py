import logging
import os
import sys

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from utils.config import argparser
from utils.experiment import (
    maybe_create_outputs_symlink,
    prepare_experiment,
    save_run_metadata,
    setup_logging,
    update_best_checkpoint,
)
from utils.util import CustomModelCheckpoint
from video_tasks import VideoAlignment


class BestCheckpointMetadata(Callback):
    def __init__(self, checkpoint_callback):
        self.checkpoint_callback = checkpoint_callback

    def on_validation_end(self, trainer, pl_module):
        update_best_checkpoint(
            self.checkpoint_callback.best_model_path,
            self.checkpoint_callback.best_model_score,
            pl_module.args.metrics_dir,
            pl_module.args.checkpoints_dir,
        )

    def on_train_end(self, trainer, pl_module):
        self.on_validation_end(trainer, pl_module)


def build_trainer(args):
    periodic_checkpoint = CustomModelCheckpoint(
        every_n_epochs=args.save_every,
        dirpath=args.checkpoints_dir,
        filename="epoch={epoch:03d}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        save_last=True,
    )
    best_checkpoint = ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        filename="best-epoch={epoch:03d}-val_loss={val_loss:.4f}",
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

    return Trainer(
        devices=[0],
        accelerator="gpu",
        callbacks=[periodic_checkpoint, best_checkpoint, BestCheckpointMetadata(best_checkpoint)],
        max_epochs=args.epochs,
        default_root_dir=args.run_dir,
        logger=[tensorboard_logger, csv_logger],
        log_every_n_steps=4,
    )


def main(args):
    logger = logging.getLogger(__name__)
    seed_everything(args.seed, workers=True)
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
    maybe_create_outputs_symlink(args.output_root)
    logger.info("Prepared run directory: %s", args.run_dir)
    main(args)
