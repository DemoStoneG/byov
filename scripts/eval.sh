#!/usr/bin/env bash
set -euo pipefail

# libgomp requires a positive integer. Some container images export an empty
# or otherwise invalid value, which produces a warning before Python starts.
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval.sh \
    --dataset <name> \
    (--checkpoint <path> | --training-run <path>) \
    --dataset-root <path> \
    --output-root <path> \
    [options]

Required:
  --dataset <name>              break_eggs | pour_milk | pour_liquid | tennis_forehand
  --checkpoint <path>           Explicit BYOV encoder checkpoint
  --training-run <path>         Training run containing config/, metrics/, checkpoints/
  --dataset-root <path>         Root containing the four AE2 dataset directories
  --output-root <path>          Root for logs, metrics, and generated artifacts

Input mode (choose by providing or omitting --embedding-dir):
  --embedding-dir <path>        Use precomputed train/val or train/test NPY files
  --embedding-file-split <name> Eval NPY prefix: val or test; defaults to --eval-mode.
                                Tasks 1/3 still fit from train_*.npy independently.
  --vision-encoder-path <path>  Transformers-format CLIP directory used for extraction
  --checkpoint-selection <name> val_loss (default), classification, retrieval,
                                progression, kendall, or last; requires --training-run

Evaluation options:
  --run-name <name>             Default: official_probe_eval
  --eval-mode <name>            val or test; default: test
  --eval-tasks <ids>            Default: 1234
  --backbone <name>             base or large; default: base
  --device <device>             auto, cpu, cuda, cuda:N; default: auto
  --num-workers <n>             Default: 0
  --no-downstream-fit           Only allow training-free tasks 2 and 4
  -h, --help                    Show this help
EOF
}

DATASET=""
CHECKPOINT=""
TRAINING_RUN=""
CHECKPOINT_SELECTION="val_loss"
DATASET_ROOT=""
OUTPUT_ROOT=""
RUN_NAME="official_probe_eval"
EVAL_MODE="test"
EVAL_TASKS="1234"
BACKBONE_VARIANT="base"
DEVICE="auto"
NUM_WORKERS="0"
EMBEDDING_DIR=""
EMBEDDING_FILE_SPLIT=""
VISION_ENCODER_PATH=""
NO_DOWNSTREAM_FIT="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --training-run) TRAINING_RUN="$2"; shift 2 ;;
    --checkpoint-selection) CHECKPOINT_SELECTION="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --eval-mode) EVAL_MODE="$2"; shift 2 ;;
    --eval-tasks) EVAL_TASKS="$2"; shift 2 ;;
    --backbone) BACKBONE_VARIANT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --embedding-dir) EMBEDDING_DIR="$2"; shift 2 ;;
    --embedding-file-split) EMBEDDING_FILE_SPLIT="$2"; shift 2 ;;
    --vision-encoder-path) VISION_ENCODER_PATH="$2"; shift 2 ;;
    --no-downstream-fit) NO_DOWNSTREAM_FIT="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

for required_name in DATASET DATASET_ROOT OUTPUT_ROOT; do
  if [ -z "${!required_name}" ]; then
    echo "Missing required argument: ${required_name,,}"
    usage
    exit 1
  fi
done
if { [ -z "$CHECKPOINT" ] && [ -z "$TRAINING_RUN" ]; } || \
   { [ -n "$CHECKPOINT" ] && [ -n "$TRAINING_RUN" ]; }; then
  echo "Provide exactly one of --checkpoint or --training-run" >&2
  exit 2
fi
if [ -n "$TRAINING_RUN" ] && [ -n "$EMBEDDING_DIR" ]; then
  echo "--training-run always re-extracts embeddings; do not pass --embedding-dir" >&2
  exit 2
fi
if [ -n "$EMBEDDING_FILE_SPLIT" ] && [ -z "$EMBEDDING_DIR" ]; then
  echo "--embedding-file-split is only valid together with --embedding-dir" >&2
  exit 2
fi

case "$DATASET" in
  break_eggs|pour_milk|pour_liquid) NUM_FRAMES="32" ;;
  tennis_forehand) NUM_FRAMES="20" ;;
  *) echo "Unknown dataset: $DATASET"; exit 2 ;;
esac

MODEL_ARGS=()
if [ "$BACKBONE_VARIANT" = "base" ]; then
  if [ -z "$VISION_ENCODER_PATH" ] && [ -z "$TRAINING_RUN" ]; then
    VISION_ENCODER_PATH="/mnt/data/wzh/ai_model/openai-clip-vit-base-patch16"
  fi
  MODEL_ARGS+=(--hidden_dim 768 --num_tokens 196 --embedding_size 256 --decoder_embedding_size 256)
elif [ "$BACKBONE_VARIANT" = "large" ]; then
  if [ -z "$VISION_ENCODER_PATH" ] && [ -z "$TRAINING_RUN" ]; then
    VISION_ENCODER_PATH="/mnt/data/wzh/ai_model/openai-clip-vit-large-patch14"
  fi
  MODEL_ARGS+=(--hidden_dim 1024 --num_tokens 256 --embedding_size 512 --decoder_embedding_size 512)
else
  echo "Unknown backbone: $BACKBONE_VARIANT; use base or large"
  exit 2
fi

MODE_ARGS=()
if [ -n "$EMBEDDING_DIR" ]; then
  MODE_ARGS+=(--embedding_dir "$EMBEDDING_DIR")
  if [ -n "$EMBEDDING_FILE_SPLIT" ]; then
    MODE_ARGS+=(--embedding_file_split "$EMBEDDING_FILE_SPLIT")
  fi
else
  MODE_ARGS+=(--extract_embedding)
fi

CHECKPOINT_ARGS=()
if [ -n "$TRAINING_RUN" ]; then
  CHECKPOINT_ARGS+=(--training_run "$TRAINING_RUN" --checkpoint_selection "$CHECKPOINT_SELECTION")
else
  CHECKPOINT_ARGS+=(--ckpt "$CHECKPOINT")
fi

VISION_ARGS=()
if [ -n "$VISION_ENCODER_PATH" ]; then
  VISION_ARGS+=(--vision_encoder_path "$VISION_ENCODER_PATH")
fi

if [ "$NO_DOWNSTREAM_FIT" = "1" ]; then
  if [[ "$EVAL_TASKS" == *1* ]] || [[ "$EVAL_TASKS" == *3* ]]; then
    echo "--no-downstream-fit only supports tasks 2 and 4; pass --eval-tasks 24"
    exit 3
  fi
  MODE_ARGS+=(--no_downstream_fit)
fi

python evaluation/evaluate_features.py \
  --dataset "$DATASET" \
  --dataset_root "$DATASET_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --eval_mode "$EVAL_MODE" \
  --eval_task "$EVAL_TASKS" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS" \
  --num_frames "$NUM_FRAMES" \
  --freeze_base \
  "${VISION_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${MODE_ARGS[@]}"
