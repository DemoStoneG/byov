import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
from argparse import Namespace
from datetime import datetime


def format_lr(lr):
    if lr >= 1e-3:
        return f"{lr:g}"
    mantissa, exponent = f"{lr:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def build_run_dir(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.replace("/", "_").replace(" ", "_")
    backbone = args.base_model_name.replace("/", "_").replace(" ", "_")
    lr = format_lr(args.lr)
    dirname = f"{timestamp}_{run_name}_{backbone}_bs{args.batch_size}_lr{lr}_seed{args.seed}"
    return os.path.join(args.output_root, args.dataset, dirname)


def ensure_run_subdirs(run_dir):
    paths = {
        "config_dir": os.path.join(run_dir, "config"),
        "logs_dir": os.path.join(run_dir, "logs"),
        "tensorboard_dir": os.path.join(run_dir, "tensorboard"),
        "checkpoints_dir": os.path.join(run_dir, "checkpoints"),
        "metrics_dir": os.path.join(run_dir, "metrics"),
        "artifacts_dir": os.path.join(run_dir, "artifacts"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def load_resume_args(cli_args):
    args_path = os.path.join(cli_args.resume, "config", "args.json")
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"Resume config not found: {args_path}")
    with open(args_path, "r") as f:
        saved_args = json.load(f)
    saved_args["resume"] = cli_args.resume
    saved_args["eval_only"] = cli_args.eval_only
    return Namespace(**saved_args)


def prepare_experiment(cli_args):
    if cli_args.resume:
        args = load_resume_args(cli_args)
        run_dir = os.path.abspath(cli_args.resume)
        is_resume = True
    else:
        args = cli_args
        run_dir = os.path.abspath(build_run_dir(args))
        is_resume = False

    paths = ensure_run_subdirs(run_dir)
    args.run_dir = run_dir
    args.output_dir = run_dir
    args.config_dir = paths["config_dir"]
    args.logs_dir = paths["logs_dir"]
    args.tensorboard_dir = paths["tensorboard_dir"]
    args.checkpoints_dir = paths["checkpoints_dir"]
    args.metrics_dir = paths["metrics_dir"]
    args.artifacts_dir = paths["artifacts_dir"]
    args.resume_ckpt = os.path.join(run_dir, "checkpoints", "last.ckpt") if is_resume else ""

    return args, is_resume


def setup_logging(logs_dir):
    log_path = os.path.join(logs_dir, "train.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logging.getLogger(__name__)


def _run_git_command(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def save_run_metadata(args, argv, is_resume):
    if is_resume:
        resume_log = os.path.join(args.logs_dir, "resume.log")
        with open(resume_log, "a") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} resume from {args.resume_ckpt}\n")
            f.write(f"{datetime.now().isoformat(timespec='seconds')} continue in run_dir: {args.run_dir}\n")
        return

    args_path = os.path.join(args.config_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    command_path = os.path.join(args.config_dir, "command.txt")
    with open(command_path, "w") as f:
        f.write(" ".join(argv) + "\n")

    git_path = os.path.join(args.config_dir, "git.txt")
    with open(git_path, "w") as f:
        f.write(f"commit: {_run_git_command(['git', 'rev-parse', 'HEAD'])}\n")
        f.write(f"branch: {_run_git_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])}\n")
        f.write(f"status:\n{_run_git_command(['git', 'status', '--short'])}\n")

    env_path = os.path.join(args.config_dir, "env.txt")
    with open(env_path, "w") as f:
        f.write(f"python: {sys.version}\n")
        f.write(f"platform: {platform.platform()}\n")
        f.write(f"hostname: {socket.gethostname()}\n")
        f.write(f"cwd: {os.getcwd()}\n")
        try:
            import torch
            f.write(f"torch: {torch.__version__}\n")
            f.write(f"cuda_available: {torch.cuda.is_available()}\n")
            f.write(f"cuda: {torch.version.cuda}\n")
        except Exception as exc:
            f.write(f"torch: unavailable: {exc}\n")
        try:
            import pytorch_lightning as pl
            f.write(f"pytorch_lightning: {pl.__version__}\n")
        except Exception as exc:
            f.write(f"pytorch_lightning: unavailable: {exc}\n")


def maybe_create_outputs_symlink(output_root, link_path="outputs"):
    if os.path.exists(link_path) or os.path.islink(link_path):
        return
    try:
        os.symlink(output_root, link_path)
    except OSError:
        pass


def update_best_checkpoint(best_model_path, best_score, metrics_dir, checkpoints_dir):
    if not best_model_path:
        return
    stable_path = os.path.join(checkpoints_dir, "best.ckpt")
    if os.path.abspath(best_model_path) != os.path.abspath(stable_path):
        shutil.copy2(best_model_path, stable_path)

    epoch = None
    basename = os.path.basename(best_model_path)
    if basename.startswith("best-epoch="):
        epoch_part = basename.split("-", 2)[1]
        if epoch_part.startswith("epoch="):
            try:
                epoch = int(epoch_part.split("=", 1)[1])
            except ValueError:
                epoch = None

    best_path = os.path.join(metrics_dir, "best.json")
    with open(best_path, "w") as f:
        json.dump(
            {
                "monitor": "val/loss",
                "mode": "min",
                "best_epoch": epoch,
                "best_score": float(best_score) if best_score is not None else None,
                "checkpoint": os.path.relpath(best_model_path, os.path.dirname(metrics_dir)),
                "stable_checkpoint": "checkpoints/best.ckpt",
            },
            f,
            indent=4,
        )
