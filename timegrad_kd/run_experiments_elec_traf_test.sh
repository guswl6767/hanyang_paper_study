#!/usr/bin/env bash
set -eu  # bash로 실행: chmod +x run_eval_best.sh && ./run_eval_best.sh

# ===== 고정 평가 설정 (논문 재현) =====
SEED=2025
NUM=100
BWIN=8

# ===== 이전 student ckpt 경로 (네가 준 절대경로) =====
ELEC_CKPT="/root/Storage/hanyang_paper_study/Models/checkpoints/20250929/20250929-164409/student/electricity_nips/student.pt"
TRAF_CKPT="/root/Storage/hanyang_paper_study/Models/checkpoints/20250929/20250929-172641/student/traffic_nips/student.pt"

# (필요 시 바꿔) 사용할 GPU
GPU_ELEC=${GPU_ELEC:-4}
GPU_TRAFFIC=${GPU_TRAFFIC:-5}

# ===== 결과 버킷 =====
BASE="/root/Storage/hanyang_paper_study/Models/checkpoints"
DAY="$(date +%Y%m%d)"
RUN="$(date +%Y%m%d-%H%M%S)"
TEST="elec_traf_TEST"
OUT_DIR="$BASE/$DAY/$RUN/$TEST"
echo "[OUT] $OUT_DIR"
mkdir -p "$OUT_DIR/student/electricity_nips" "$OUT_DIR/student/traffic_nips"

# ===== 평가 (병렬) =====

# electricity_nips
(
  CUDA_VISIBLE_DEVICES="$GPU_ELEC" \
  python evaluate_final.py --dataset electricity_nips \
    --ckpt "$ELEC_CKPT" \
    --device cuda --rolling --num_samples "$NUM" --batch_windows "$BWIN" --seed "$SEED" \
    --save_json "$OUT_DIR/student/electricity_nips/electricity_nips_student.json" \
    > "$OUT_DIR/student/electricity_nips/eval.log" 2>&1

  echo "[OK] electricity_nips -> $OUT_DIR/student/electricity_nips/electricity_nips_student.json"
) &

# traffic_nips
(
  CUDA_VISIBLE_DEVICES="$GPU_TRAFFIC" \
  python evaluate_final.py --dataset traffic_nips \
    --ckpt "$TRAF_CKPT" \
    --device cuda --rolling --num_samples "$NUM" --batch_windows "$BWIN" --seed "$SEED" \
    --save_json "$OUT_DIR/student/traffic_nips/traffic_nips_student.json" \
    > "$OUT_DIR/student/traffic_nips/eval.log" 2>&1

  echo "[OK] traffic_nips -> $OUT_DIR/student/traffic_nips/traffic_nips_student.json"
) &

wait

echo
echo "[DONE] All saved to: $OUT_DIR"
echo "  - $OUT_DIR/student/electricity_nips/electricity_nips_student.json"
echo "  - $OUT_DIR/student/traffic_nips/traffic_nips_student.json"
