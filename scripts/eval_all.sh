#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_all.sh \
    --checkpoint-root <path> \
    --dataset-root <path> \
    --output-root <path> \
    --backbone-label <name> \
    [options]

Expected checkpoint-root layout:
  <root>/break_eggs.ckpt
  <root>/break_eggs_eval/{train,val}_*.npy
  <root>/pour_milk.ckpt
  <root>/pour_milk_eval/{train,val}_*.npy
  <root>/pour_liquid.ckpt
  <root>/pour_liquid_eval/{train,val}_*.npy
  <root>/tennis_forehand.ckpt
  <root>/tennis_forehand_eval/{train,val}_*.npy

Options:
  --extract-embedding           Ignore *_eval NPY files and run CLIP + BYOV checkpoint
  --backbone <base|large>       Default: base
  --eval-mode <val|test>        Default: test
  --embedding-file-split <name> Default: val
  --eval-tasks <ids>            Default: 1234
  --device <device>             Default: auto
  --num-workers <n>             Default: 0
  --vision-encoder-path <path>  Optional CLIP model path
  -h, --help                    Show this help
EOF
}

CHECKPOINT_ROOT=""
DATASET_ROOT=""
OUTPUT_ROOT=""
BACKBONE_LABEL=""
BACKBONE="base"
EVAL_MODE="test"
EMBEDDING_FILE_SPLIT="val"
EVAL_TASKS="1234"
DEVICE="auto"
NUM_WORKERS="0"
VISION_ENCODER_PATH=""
EXTRACT_EMBEDDING="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
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

for required_name in CHECKPOINT_ROOT DATASET_ROOT OUTPUT_ROOT BACKBONE_LABEL; do
  if [ -z "${!required_name}" ]; then
    echo "Missing required argument: ${required_name,,}"
    usage
    exit 1
  fi
done

COMPARISON_ROOT="$OUTPUT_ROOT/$BACKBONE_LABEL"
COMMON_ARGS=(
  --dataset-root "$DATASET_ROOT"
  --output-root "$COMPARISON_ROOT"
  --eval-mode "$EVAL_MODE"
  --embedding-file-split "$EMBEDDING_FILE_SPLIT"
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
    --checkpoint "$CHECKPOINT_ROOT/$dataset.ckpt"
    --run-name official_eval
    "${COMMON_ARGS[@]}"
  )
  if [ "$EXTRACT_EMBEDDING" = "0" ]; then
    DATASET_ARGS+=(--embedding-dir "$CHECKPOINT_ROOT/${dataset}_eval")
  fi
  bash scripts/eval.sh "${DATASET_ARGS[@]}"
done

echo "All four datasets completed."
echo "Comparison root: $COMPARISON_ROOT"
echo "Combined results: $COMPARISON_ROOT/summary/all_results.json"
