#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_all.sh \
    (--training-root <path> | --checkpoint-root <path>) \
    --dataset-root <path> \
    --output-root <path> \
    --backbone-label <name> \
    [options]

Recommended training-root layout (created by scripts/run.sh):
  <root>/<dataset>/<timestamp_run_name>/config/args.json
  <root>/<dataset>/<timestamp_run_name>/metrics/best.json
  <root>/<dataset>/<timestamp_run_name>/checkpoints/*.ckpt

Legacy checkpoint-root layout:
  <root>/break_eggs.ckpt
  <root>/break_eggs_eval/{train,val}_*.npy
  <root>/pour_milk.ckpt
  <root>/pour_milk_eval/{train,val}_*.npy
  <root>/pour_liquid.ckpt
  <root>/pour_liquid_eval/{train,val}_*.npy
  <root>/tennis_forehand.ckpt
  <root>/tennis_forehand_eval/{train,val}_*.npy

Options:
  --checkpoint-selection <name> Select val_loss (default), classification, retrieval,
                                progression, kendall, or last from each training run
  --run-name-filter <text>      Only consider training run directory names containing text
  --extract-embedding           Ignore *_eval NPY files and run CLIP + BYOV checkpoint
  --backbone <base|large>       Default: base
  --eval-mode <val|test>        Default: test
  --embedding-file-split <name> Precomputed eval-file prefix only; defaults to --eval-mode
  --eval-tasks <ids>            Default: 1234
  --device <device>             Default: auto
  --num-workers <n>             Default: 0
  --vision-encoder-path <path>  Optional Transformers-format CLIP model path
  -h, --help                    Show this help
EOF
}

CHECKPOINT_ROOT=""
TRAINING_ROOT=""
CHECKPOINT_SELECTION="val_loss"
RUN_NAME_FILTER=""
DATASET_ROOT=""
OUTPUT_ROOT=""
BACKBONE_LABEL=""
BACKBONE="base"
EVAL_MODE="test"
EMBEDDING_FILE_SPLIT=""
EVAL_TASKS="1234"
DEVICE="auto"
NUM_WORKERS="0"
VISION_ENCODER_PATH=""
EXTRACT_EMBEDDING="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --training-root) TRAINING_ROOT="$2"; shift 2 ;;
    --checkpoint-selection) CHECKPOINT_SELECTION="$2"; shift 2 ;;
    --run-name-filter) RUN_NAME_FILTER="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --backbone-label) BACKBONE_LABEL="$2"; shift 2 ;;
    --backbone) BACKBONE="$2"; shift 2 ;;
    --eval-mode) EVAL_MODE="$2"; shift 2 ;;
    --embedding-file-split) EMBEDDING_FILE_SPLIT="$2"; shift 2 ;;
    --eval-tasks) EVAL_TASKS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --vision-encoder-path) VISION_ENCODER_PATH="$2"; shift 2 ;;
    --extract-embedding) EXTRACT_EMBEDDING="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

for required_name in DATASET_ROOT OUTPUT_ROOT BACKBONE_LABEL; do
  if [ -z "${!required_name}" ]; then
    echo "Missing required argument: ${required_name,,}"
    usage
    exit 1
  fi
done
if { [ -z "$CHECKPOINT_ROOT" ] && [ -z "$TRAINING_ROOT" ]; } || \
   { [ -n "$CHECKPOINT_ROOT" ] && [ -n "$TRAINING_ROOT" ]; }; then
  echo "Provide exactly one of --training-root or --checkpoint-root" >&2
  exit 2
fi
if [ -n "$EMBEDDING_FILE_SPLIT" ] && \
   { [ -n "$TRAINING_ROOT" ] || [ "$EXTRACT_EMBEDDING" = "1" ]; }; then
  echo "--embedding-file-split is only valid when reading precomputed embeddings" >&2
  exit 2
fi

COMPARISON_ROOT="$OUTPUT_ROOT/$BACKBONE_LABEL"
COMMON_ARGS=(
  --dataset-root "$DATASET_ROOT"
  --output-root "$COMPARISON_ROOT"
  --eval-mode "$EVAL_MODE"
  --eval-tasks "$EVAL_TASKS"
  --backbone "$BACKBONE"
  --device "$DEVICE"
  --num-workers "$NUM_WORKERS"
)
if [ -n "$VISION_ENCODER_PATH" ]; then
  COMMON_ARGS+=(--vision-encoder-path "$VISION_ENCODER_PATH")
fi

for dataset in break_eggs pour_milk pour_liquid tennis_forehand; do
  DATASET_ARGS=(
    --dataset "$dataset"
    --run-name trained_checkpoint_test
    "${COMMON_ARGS[@]}"
  )
  if [ -n "$TRAINING_ROOT" ]; then
    RESOLVER_ARGS=(
      --training-root "$TRAINING_ROOT"
      --dataset "$dataset"
      --selection "$CHECKPOINT_SELECTION"
    )
    if [ -n "$RUN_NAME_FILTER" ]; then
      RESOLVER_ARGS+=(--run-name-filter "$RUN_NAME_FILTER")
    fi
    TRAINING_RUN="$(python utils/training_checkpoint.py "${RESOLVER_ARGS[@]}")"
    echo "[$dataset] training run: $TRAINING_RUN"
    echo "[$dataset] checkpoint selection: $CHECKPOINT_SELECTION"
    DATASET_ARGS+=(
      --training-run "$TRAINING_RUN"
      --checkpoint-selection "$CHECKPOINT_SELECTION"
    )
  else
    DATASET_ARGS+=(--checkpoint "$CHECKPOINT_ROOT/$dataset.ckpt")
  fi
  if [ -z "$TRAINING_ROOT" ] && [ "$EXTRACT_EMBEDDING" = "0" ]; then
    DATASET_ARGS+=(--embedding-dir "$CHECKPOINT_ROOT/${dataset}_eval")
    if [ -n "$EMBEDDING_FILE_SPLIT" ]; then
      DATASET_ARGS+=(--embedding-file-split "$EMBEDDING_FILE_SPLIT")
    fi
  fi
  bash scripts/eval.sh "${DATASET_ARGS[@]}"
done

echo "All four datasets completed."
echo "Comparison root: $COMPARISON_ROOT"
echo "Combined results: $COMPARISON_ROOT/summary/all_results.json"
