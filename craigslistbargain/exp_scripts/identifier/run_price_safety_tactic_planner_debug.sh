#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${CB_DIR}/.." && pwd)"

cd "${CB_DIR}"

EXP_SUFFIX="${1:-_price_safety_tactic_planner}"
USE_GPU="${2:-0}"
SEED="${3:-0}"
TEMP="${4:-${ROLLOUT_TEMPERATURE:-1.0}}"
LOAD_SAMPLE_CACHE="${5:-}"

EXP_NAME="switch_aware_tom${EXP_SUFFIX}"
MODEL_NAME="switch_aware"

AGENT0="${ROLLOUT_AGENT0:-tom}"
AGENT1="${ROLLOUT_AGENT1:-pt-neural-r}"

PRINT_DIALOGUES="${PRINT_DIALOGUES:-1}"
MAX_TURNS="${MAX_TURNS:-20}"
MAX_LENGTH="${MAX_LENGTH:-20}"
REWARD_NAME="${REWARD_NAME:-margin}"
TRACE_PATH="${TURN_TRACE_PATH:-results/turn_trace_price_safety_tactic_planner.csv}"
NUM_DIALOGUES="${NUM_DIALOGUES:-2}"

GPU_ARGS=""
if [ -n "${USE_GPU}" ]; then
  GPU_ARGS="--gpuid ${USE_GPU}"
fi

LOAD_SAMPLE_ARGS=""
if [ -n "${LOAD_SAMPLE_CACHE}" ]; then
  LOAD_SAMPLE_ARGS="--load-sample cache/${LOAD_SAMPLE_CACHE}/data.pkl"
fi

mkdir -p "checkpoint/${EXP_NAME}" "$(dirname "${TRACE_PATH}")"

echo "[rollout] exp_name=${EXP_NAME}"
echo "[rollout] cwd=$(pwd)"
echo "[rollout] project_root=${PROJECT_ROOT}"
echo "[rollout] agents=${AGENT0} ${AGENT1}"
echo "[rollout] temperature=${TEMP}"
echo "[rollout] turn_trace=${TRACE_PATH}"

PYTHONPATH="${PROJECT_ROOT}" python -m craigslistbargain.multi_rl \
  --schema-path data/craigslist-schema.json \
  --scenarios-path data/train-scenarios.json \
  --valid-scenarios-path data/dev-scenarios.json \
  --price-tracker data/price_tracker.pkl \
  --agent-checkpoints checkpoint/language/model_best.pt checkpoint/language/model_best.pt \
  --model-path checkpoint/${EXP_NAME} \
  --mappings mappings/language \
  --optim adam \
  --rnn-type RNN \
  --rnn-size 300 \
  --max-grad-norm -1 \
  --agents "${AGENT0}" "${AGENT1}" \
  --report-every 50 \
  --max-turns "${MAX_TURNS}" \
  --num-dialogues "${NUM_DIALOGUES}" \
  --sample \
  --temperature "${TEMP}" \
  --max-length "${MAX_LENGTH}" \
  --reward "${REWARD_NAME}" \
  --dia-num 1 \
  --state-length 4 \
  --epochs 1 \
  --use-utterance \
  --model lf2lf \
  --model-type a2c \
  --tom-test \
  ${LOAD_SAMPLE_ARGS} \
  --learning-rate 0.001 \
  --name "${EXP_NAME}" \
  --seed "${SEED}" \
  --tom-hidden-size 128 \
  --tom-hidden-depth 2 \
  --id-hidden-size 128 \
  --id-hidden-depth 2 \
  --strategy-in-words \
  --tom-model "${MODEL_NAME}" \
  --print-dialogues "${PRINT_DIALOGUES}" \
  --print-dev-detail \
  --enable-buyer-price-safety \
  --enable-seller-tactic-tracker \
  --enable-rule-offer-planner \
  --price-safety-debug \
  --tactic-tracker-debug \
  --offer-planner-debug \
  --turn-trace-path "${TRACE_PATH}" \
  --verbose \
  ${GPU_ARGS}
