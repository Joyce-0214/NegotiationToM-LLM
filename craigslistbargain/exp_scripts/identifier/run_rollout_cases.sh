#!/usr/bin/env bash
set -u -o pipefail

# ===== 可改参数 =====
SPLIT="${SPLIT:-dev}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-49}"          # 跑 50 个例子；想跑 100 个就改成 99
INTERVENE_AGENT="${INTERVENE_AGENT:-0}"
INTERVENE_TURN="${INTERVENE_TURN:-2}"
TEMP="${TEMP:-1.0}"

# rollout 用哪个 checkpoint 目录
# 对应 checkpoint/${EXP_NAME}
EXP_SUFFIX="${EXP_SUFFIX:-}"      # 例如 "_debug"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
LOAD_SAMPLE_CACHE="${LOAD_SAMPLE_CACHE:-}"

OUTDIR="logs/rollout_search"
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTDIR}/rollout_${SPLIT}_${TS}"
SUMMARY_JSONL="${RUN_DIR}/summary.jsonl"

mkdir -p "${RUN_DIR}"
: > "${SUMMARY_JSONL}"

echo "Run dir: ${RUN_DIR}"
echo "Summary: ${SUMMARY_JSONL}"

for i in $(seq "${START_IDX}" "${END_IDX}"); do
  LOGFILE="${RUN_DIR}/scenario_${i}.log"
  echo "===== Running scenario ${i} ====="

  exit_code=0

  if ! DEBUG_TOM_ROLLOUT=1 \
      ROLLOUT_SPLIT="${SPLIT}" \
      ROLLOUT_SCENARIO_IDX="${i}" \
      ROLLOUT_INTERVENE_AGENT="${INTERVENE_AGENT}" \
      ROLLOUT_INTERVENE_TURN="${INTERVENE_TURN}" \
      ROLLOUT_TEMPERATURE="${TEMP}" \
      ROLLOUT_AGENT0="tom" \
      ROLLOUT_AGENT1="pt-neural-r" \
      PRINT_DIALOGUES=0 \
      bash exp_scripts/identifier/run_switch_aware_rollout_once.sh \
        "${EXP_SUFFIX}" "${GPU_ID}" "${SEED}" "${TEMP}" "${LOAD_SAMPLE_CACHE}" \
      > "${LOGFILE}" 2>&1
  then
    exit_code=$?
  fi

  python - <<'PY' "${LOGFILE}" "${SUMMARY_JSONL}" "${i}" "${exit_code}"
import json, re, sys, pathlib

logfile = pathlib.Path(sys.argv[1])
summary = pathlib.Path(sys.argv[2])
scenario_idx = int(sys.argv[3])
exit_code = int(sys.argv[4])

text = logfile.read_text(encoding="utf-8", errors="ignore")

m = re.search(r'FULL RESULT JSON\s*-+\s*(\{.*\})\s*=+', text, re.S)
obj = None
if m:
    try:
        obj = json.loads(m.group(1))
    except Exception:
        obj = None

if obj is None:
    row = {
        "scenario_idx": scenario_idx,
        "logfile": str(logfile),
        "exit_code": exit_code,
        "parse_ok": False,
    }
else:
    cmp = obj.get("comparison", {})
    row = {
        "scenario_idx": scenario_idx,
        "logfile": str(logfile),
        "exit_code": exit_code,
        "parse_ok": True,

        "normal_is_agreed": cmp.get("normal_is_agreed"),
        "force_off_is_agreed": cmp.get("force_off_is_agreed"),
        "force_on_is_agreed": cmp.get("force_on_is_agreed"),

        "normal_final_price": cmp.get("normal_final_price"),
        "force_off_final_price": cmp.get("force_off_final_price"),
        "force_on_final_price": cmp.get("force_on_final_price"),

        "normal_num_turns": cmp.get("normal_num_turns"),
        "force_off_num_turns": cmp.get("force_off_num_turns"),
        "force_on_num_turns": cmp.get("force_on_num_turns"),

        "normal_buyer_reward": cmp.get("normal_buyer_reward"),
        "force_off_buyer_reward": cmp.get("force_off_buyer_reward"),
        "force_on_buyer_reward": cmp.get("force_on_buyer_reward"),

        "normal_avg_switch_prob_used": cmp.get("normal_avg_switch_prob_used"),
        "force_off_avg_switch_prob_used": cmp.get("force_off_avg_switch_prob_used"),
        "force_on_avg_switch_prob_used": cmp.get("force_on_avg_switch_prob_used"),

        "force_off_buyer_last_price": cmp.get("force_off_buyer_last_price"),
        "force_on_buyer_last_price": cmp.get("force_on_buyer_last_price"),
        "force_off_buyer_last_intent": cmp.get("force_off_buyer_last_intent"),
        "force_on_buyer_last_intent": cmp.get("force_on_buyer_last_intent"),

        "force_off_vs_on_first_diverge_turn": cmp.get("force_off_vs_on_first_diverge_turn"),
        "force_off_vs_on_last_price_diff": cmp.get("force_off_vs_on_last_price_diff"),
        "force_off_vs_on_last_intent_diff": cmp.get("force_off_vs_on_last_intent_diff"),
    }

with summary.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

done

echo "Done. Logs saved in: ${RUN_DIR}"
echo "Summary saved in: ${SUMMARY_JSONL}"