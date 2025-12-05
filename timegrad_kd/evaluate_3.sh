#!/bin/sh
set -eu

SEED=2025
NUM_MAIN=100
NUM_CALIB=200
BWIN=8
PIT_BINS=20
COV_LEVELS="0.5,0.8,0.9,0.95"

# TEACHER files (단일 파일 경로!)
T_ROOT="/root/Storage/hanyang_paper_study/Models/checkpoints/final_model/teacher_model"
T_EXC="$T_ROOT/exchange_rate_teacher.pt"
T_ELE="$T_ROOT/electricity_nips_teacher.pt"
T_SOL="$T_ROOT/solar_nips_teacher.pt"
T_TRF="$T_ROOT/traffic_nips_teacher.pt"
T_TAX="$T_ROOT/taxi_30min_teacher.pt"
T_WIK="$T_ROOT/wiki-rolling_nips_teacher.pt"

# STUDENT files (확정 최종)
S_ROOT="/root/Storage/hanyang_paper_study/Models/checkpoints/final_model/student_model"
S_EXC="$S_ROOT/exchange_rate_student.pt"
S_ELE="$S_ROOT/electricity_nips_student.pt"
S_SOL="$S_ROOT/solar_nips_student.pt"
S_TRF="$S_ROOT/traffic_nips_student.pt"
S_TAX="$S_ROOT/taxi_30min_student.pt"
S_WIK="$S_ROOT/wiki-rolling_nips_student.pt"

GPU_EXC=${GPU_EXC:-0}; GPU_ELE=${GPU_ELE:-1}; GPU_SOL=${GPU_SOL:-2}
GPU_TRF=${GPU_TRF:-3}; GPU_TAX=${GPU_TAX:-4}; GPU_WIK=${GPU_WIK:-5}

BASE="/root/Storage/hanyang_paper_study/Models/checkpoints"
DAY="$(date +%Y%m%d)"; RUN="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$BASE/$DAY/$RUN/eval_H1H2H3"
echo "[OUT] $OUT_DIR"
for role in teacher student; do
  for ds in exchange_rate electricity_nips solar_nips traffic_nips taxi_30min wiki-rolling_nips; do
    mkdir -p "$OUT_DIR/$role/$ds"
  done
done

run_ds () {
  DS="$1"; GPU="$2"; TCKPT="$3"; SCKPT="$4"
  echo "[RUN] $DS on GPU $GPU"

  # Teacher: 본평가 + per-window CRPS (H1/H2)
  CUDA_VISIBLE_DEVICES="$GPU" \
  python evaluate_3.py --dataset "$DS" \
    --ckpt "$TCKPT" --device cuda --rolling \
    --num_samples "$NUM_MAIN" --batch_windows "$BWIN" --seed "$SEED" \
    --save_per_window_crps \
    --save_json "$OUT_DIR/teacher/$DS/${DS}_teacher.json" \
    > "$OUT_DIR/teacher/$DS/eval_main.log" 2>&1

  # Student: 본평가 + per-window CRPS (H1/H2)
  CUDA_VISIBLE_DEVICES="$GPU" \
  python evaluate_3.py --dataset "$DS" \
    --ckpt "$SCKPT" --device cuda --rolling \
    --num_samples "$NUM_MAIN" --batch_windows "$BWIN" --seed "$SEED" \
    --save_per_window_crps \
    --save_json "$OUT_DIR/student/$DS/${DS}_student.json" \
    > "$OUT_DIR/student/$DS/eval_main.log" 2>&1

  # Student: 캘리브레이션 보조(200) (H3)
  CUDA_VISIBLE_DEVICES="$GPU" \
  python evaluate_3.py --dataset "$DS" \
    --ckpt "$SCKPT" --device cuda --rolling \
    --num_samples "$NUM_CALIB" --batch_windows "$BWIN" --seed "$SEED" \
    --save_pit_cov --pit_bins "$PIT_BINS" --coverage_levels "$COV_LEVELS" \
    --save_json "$OUT_DIR/student/$DS/${DS}_student_calib200.json" \
    > "$OUT_DIR/student/$DS/eval_calib200.log" 2>&1

  # (선택) 구조 요약: traffic/wiki에만 — 필요하면 다른 셋에도 복붙
  case "$DS" in
    (traffic_nips|wiki-rolling_nips|solar_nips|taxi_30min|exchange_rate|electricity_nips)
      CUDA_VISIBLE_DEVICES="$GPU" \
      python evaluate_3.py --dataset "$DS" \
        --ckpt "$TCKPT" --device cuda --rolling \
        --num_samples "$NUM_MAIN" --batch_windows "$BWIN" --seed "$SEED" \
        --save_struct --struct_sample_dims 64 --struct_sample_windows 50 \
        --save_json "$OUT_DIR/teacher/$DS/${DS}_teacher_struct.json" \
        > "$OUT_DIR/teacher/$DS/eval_struct.log" 2>&1
      CUDA_VISIBLE_DEVICES="$GPU" \
      python evaluate_final.py --dataset "$DS" \
        --ckpt "$SCKPT" --device cuda --rolling \
        --num_samples "$NUM_MAIN" --batch_windows "$BWIN" --seed "$SEED" \
        --save_struct --struct_sample_dims 64 --struct_sample_windows 50 \
        --save_json "$OUT_DIR/student/$DS/${DS}_student_struct.json" \
        > "$OUT_DIR/student/$DS/eval_struct.log" 2>&1
      ;;
  esac

  echo "[OK] $DS done"
}

(run_ds exchange_rate     "$GPU_EXC" "$T_EXC" "$S_EXC") &
(run_ds electricity_nips  "$GPU_ELE" "$T_ELE" "$S_ELE") &
(run_ds solar_nips        "$GPU_SOL" "$T_SOL" "$S_SOL") &
(run_ds traffic_nips      "$GPU_TRF" "$T_TRF" "$S_TRF") &
(run_ds taxi_30min        "$GPU_TAX" "$T_TAX" "$S_TAX") &
(run_ds wiki-rolling_nips "$GPU_WIK" "$T_WIK" "$S_WIK") &
wait

echo "[DONE] $OUT_DIR"
