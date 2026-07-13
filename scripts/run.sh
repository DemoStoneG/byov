#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <dataset-name> [run-name] [seed]"
    exit 1
fi

DATASET="$1"
RUN_NAME="${2:-baseline}"
SEED="${3:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/wzh/experiments/byov}"

COMMON_ARGS=(
  --dataset "$DATASET"
  --output_root "$OUTPUT_ROOT"
  --run_name "$RUN_NAME"
  --seed "$SEED"
  --lr 1e-5
  --freeze_base
)

if [ "$DATASET" = "break_eggs" ]; then
  python train.py "${COMMON_ARGS[@]}" --batch_size 4
elif [ "$DATASET" = "pour_milk" ]; then
  # pour milk is missing one det_bounding_box.pickle file; use the basic encoder setup.
  python train.py "${COMMON_ARGS[@]}" --batch_size 1
elif [ "$DATASET" = "pour_liquid" ]; then
  python train.py "${COMMON_ARGS[@]}" --batch_size 1
elif [ "$DATASET" = "tennis_forehand" ]; then
  python train.py "${COMMON_ARGS[@]}" --num_frames 20 --batch_size 1
else
    echo "Unknown dataset: $DATASET, select among [break_eggs, pour_milk, pour_liquid, tennis_forehand]"
    exit 2
fi
