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
    --checkpoint <path> \
    --dataset-root <path> \
    --output-root <path> \
    [options]

Required:
  --dataset <name>              break_eggs | pour_milk | pour_liquid | tennis_forehand
  --checkpoint <path>           Official BYOV probe/encoder checkpoint
  --dataset-root <path>         Root containing the four AE2 dataset directories
  --output-root <path>          Root for logs, metrics, and generated artifacts

Input mode (choose by providing or omitting --embedding-dir):
  --embedding-dir <path>        Use precomputed train/val or train/test NPY files
  --embedding-file-split <name> NPY prefix: val or test; defaults to --eval-mode
  --vision-encoder-path <path>  CLIP path used when extracting embeddings

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

for required_name in DATASET CHECKPOINT DATASET_ROOT OUTPUT_ROOT; do
  if [ -z "${!required_name}" ]; then
    echo "Missing required argument: ${required_name,,}"
    usage
    exit 1
  fi
done

case "$DATASET" in
  break_eggs|pour_milk|pour_liquid) NUM_FRAMES="32" ;;
  tennis_forehand) NUM_FRAMES="20" ;;
  *) echo "Unknown dataset: $DATASET"; exit 2 ;;
esac

MODEL_ARGS=()
if [ "$BACKBONE_VARIANT" = "base" ]; then
  VISION_ENCODER_PATH="${VISION_ENCODER_PATH:-/mnt/data/wzh/ai_model/openai-clip-vit-base-patch16}"
  MODEL_ARGS+=(--hidden_dim 768 --num_tokens 196 --embedding_size 256 --decoder_embedding_size 256)
elif [ "$BACKBONE_VARIANT" = "large" ]; then
  VISION_ENCODER_PATH="${VISION_ENCODER_PATH:-/mnt/data/wzh/ai_model/openai-clip-vit-large-patch14}"
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
  --vision_encoder_path "$VISION_ENCODER_PATH" \
  --ckpt "$CHECKPOINT" \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --eval_mode "$EVAL_MODE" \
  --eval_task "$EVAL_TASKS" \
  --device "$DEVICE" \
  --num_workers "$NUM_WORKERS" \
  --num_frames "$NUM_FRAMES" \
  --freeze_base \
  "${MODEL_ARGS[@]}" \
  "${MODE_ARGS[@]}"
