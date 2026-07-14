#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run.sh \
    --dataset <name> \
    --dataset-root <path> \
    --output-root <path> \
    --vision-encoder-path <path> [options]

Required for a new run:
  --dataset <name>              break_eggs, pour_milk, pour_liquid, or tennis_forehand
  --dataset-root <path>         Root containing the four AE2 dataset directories
  --output-root <path>          Root for training run directories
  --vision-encoder-path <path>  Transformers-format CLIP vision model directory

Options:
  --run-name <name>             Default: baseline
  --seed <int>                  Default: 42
  --epochs <int>                Default: 300
  --lr <float>                  Default: 1e-5
  --weight-decay <float>        Default: 5e-6
  --batch-size <int>            Default: 4 for break_eggs, otherwise 1
  --num-workers <int>           Default: 0
  --ds-every <int>              Downstream validation interval; 0 disables it (default: 10)
  --save-every <int>            Periodic checkpoint interval (default: 10)
  --eval-tasks <ids>            Downstream tasks used during training (default: 1234)
  --smoke-test                  Run one train batch and one validation batch for one epoch
  --dry-run                     Only validate arguments and output configuration
  --resume <run-dir>            Resume from the newest last-epoch=NNN.ckpt
  -h, --help                    Show this help

BYOV training itself is self-supervised. Labels are required only when --ds-every is greater
than zero, because periodic downstream validation fits/evaluates the four labelled tasks.
EOF
}

DATASET=""
DATASET_ROOT=""
OUTPUT_ROOT=""
VISION_ENCODER_PATH=""
RUN_NAME="baseline"
SEED=42
EPOCHS=300
LR="1e-5"
WEIGHT_DECAY="5e-6"
BATCH_SIZE=""
NUM_WORKERS=0
DS_EVERY=10
SAVE_EVERY=10
EVAL_TASKS="1234"
SMOKE_TEST=0
DRY_RUN=0
RESUME=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --vision-encoder-path) VISION_ENCODER_PATH="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --ds-every) DS_EVERY="$2"; shift 2 ;;
    --save-every) SAVE_EVERY="$2"; shift 2 ;;
    --eval-tasks) EVAL_TASKS="$2"; shift 2 ;;
    --smoke-test) SMOKE_TEST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --resume) RESUME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "${OMP_NUM_THREADS:-1}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Python executable not found (tried python and python3)" >&2
    exit 2
  fi
fi

if [ -n "$RESUME" ]; then
  if ! compgen -G "$RESUME/checkpoints/last-epoch=*.ckpt" >/dev/null && \
      [ ! -f "$RESUME/checkpoints/last.ckpt" ]; then
    echo "Resume checkpoint not found under: $RESUME/checkpoints" >&2
    exit 2
  fi
  exec "$PYTHON_BIN" train.py --resume "$RESUME"
fi

missing=()
[ -n "$DATASET" ] || missing+=(--dataset)
[ -n "$DATASET_ROOT" ] || missing+=(--dataset-root)
[ -n "$OUTPUT_ROOT" ] || missing+=(--output-root)
[ -n "$VISION_ENCODER_PATH" ] || missing+=(--vision-encoder-path)
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing required arguments: ${missing[*]}" >&2
  usage >&2
  exit 2
fi

case "$DATASET" in
  break_eggs) DEFAULT_BATCH_SIZE=4; NUM_FRAMES=32 ;;
  pour_milk|pour_liquid) DEFAULT_BATCH_SIZE=1; NUM_FRAMES=32 ;;
  tennis_forehand) DEFAULT_BATCH_SIZE=1; NUM_FRAMES=20 ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
if [ "$SMOKE_TEST" -eq 1 ]; then
  DS_EVERY=0
fi

if [ "$DRY_RUN" -eq 0 ] && [ ! -d "$DATASET_ROOT/$DATASET" ]; then
  echo "Dataset directory not found: $DATASET_ROOT/$DATASET" >&2
  exit 2
fi
if [ "$DRY_RUN" -eq 0 ] && [ ! -d "$VISION_ENCODER_PATH" ]; then
  echo "Vision encoder directory not found: $VISION_ENCODER_PATH" >&2
  exit 2
fi

ARGS=(
  --dataset "$DATASET"
  --dataset_root "$DATASET_ROOT"
  --output_root "$OUTPUT_ROOT"
  --vision_encoder_path "$VISION_ENCODER_PATH"
  --run_name "$RUN_NAME"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --lr "$LR"
  --wd "$WEIGHT_DECAY"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --ds_every_n_epoch "$DS_EVERY"
  --save_every "$SAVE_EVERY"
  --eval_task "$EVAL_TASKS"
  --num_frames "$NUM_FRAMES"
  --freeze_base
)

if [ "$SMOKE_TEST" -eq 1 ]; then
  ARGS+=(--smoke_test)
fi
if [ "$DRY_RUN" -eq 1 ]; then
  ARGS+=(--dry_run config)
fi

exec "$PYTHON_BIN" train.py "${ARGS[@]}"
