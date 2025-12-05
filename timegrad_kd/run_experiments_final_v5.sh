#!/usr/bin/env bash
set -eu  # bash로 실행: chmod +x run_three.sh && ./run_three.sh

# 재현 고정값
SEED=2025
NUM=100
BWIN=8

# 결과 버킷
BASE="/root/Storage/hanyang_paper_study/Models/checkpoints"
DAY="$(date +%Y%m%d)"
RUN="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$BASE/$DAY/$RUN"
echo "[OUT] $OUT_DIR"
mkdir -p "$OUT_DIR"

# ★ 기존 teacher 버킷 (환경에 맞게 유지)
TEACHER_DIR="/root/Storage/hanyang_paper_study/Models/checkpoints/20250929/20250929-182349/teacher"

# 각 데이터셋 폴더 미리 생성
for ds in traffic_nips; do
  mkdir -p "$OUT_DIR/student/$ds"
done

########################################
# ---------- exchange (GPU 0) ----------
# (
#   CUDA_VISIBLE_DEVICES=0 python train_student_final.py --dataset exchange_rate \
#     --teacher_ckpt "$TEACHER_DIR/exchange_rate/teacher.pt" \
#     --epochs 200 --device cuda \
#     --n_steps_s 45 --res_channels 8 --res_blocks 4 \
#     --kd_warmup 10 --lambda_kd 0.5 --mu_kd 0.05 --gamma_x0 0.15 --grad_clip 1.0 \
#     --save_dir "$OUT_DIR/student/exchange_rate" \
#     > "$OUT_DIR/student/exchange_rate/train.log" 2>&1

#   CUDA_VISIBLE_DEVICES=0 python evaluate_final.py --dataset exchange_rate \
#     --ckpt "$OUT_DIR/student/exchange_rate/student.pt" \
#     --device cuda --rolling --num_samples "$NUM" --batch_windows "$BWIN" --seed "$SEED" \
#     --save_json "$OUT_DIR/student/exchange_rate/exchange_rate_student.json" \
#     > "$OUT_DIR/student/exchange_rate/eval.log" 2>&1
# ) &

########################################
# ---------- traffic (GPU 1) ----------
(
  CUDA_VISIBLE_DEVICES=1 python train_student_final.py --dataset traffic_nips \
    --teacher_ckpt "$TEACHER_DIR/traffic_nips/teacher.pt" \
    --epochs 200 --device cuda \
    --n_steps_s 60 --res_channels 14 --res_blocks 8 \
    --kd_warmup 20 --lambda_kd 0.5 --mu_kd 0.05 --gamma_x0 0.20 --grad_clip 1.0 \
    --save_dir "$OUT_DIR/student/traffic_nips" \
    > "$OUT_DIR/student/traffic_nips/train.log" 2>&1

  CUDA_VISIBLE_DEVICES=1 python evaluate_final.py --dataset traffic_nips \
    --ckpt "$OUT_DIR/student/traffic_nips/student.pt" \
    --device cuda --rolling --num_samples "$NUM" --batch_windows "$BWIN" --seed "$SEED" \
    --save_json "$OUT_DIR/student/traffic_nips/traffic_nips_student.json" \
    > "$OUT_DIR/student/traffic_nips/eval.log" 2>&1
) &

########################################
# ---------- solar (GPU 2) ----------
# (
#   CUDA_VISIBLE_DEVICES=2 python train_student_final.py --dataset solar_nips \
#     --teacher_ckpt "$TEACHER_DIR/solar_nips/teacher.pt" \
#     --epochs 200 --device cuda \
#     --n_steps_s 70 --res_channels 12 --res_blocks 6 \
#     --kd_warmup 20 --lambda_kd 0.5 --mu_kd 0.05 --gamma_x0 0.25 --grad_clip 1.0 \
#     --save_dir "$OUT_DIR/student/solar_nips" \
#     > "$OUT_DIR/student/solar_nips/train.log" 2>&1

#   CUDA_VISIBLE_DEVICES=2 python evaluate_final.py --dataset solar_nips \
#     --ckpt "$OUT_DIR/student/solar_nips/student.pt" \
#     --device cuda --rolling --num_samples "$NUM" --batch_windows "$BWIN" --seed "$SEED" \
#     --save_json "$OUT_DIR/student/solar_nips/solar_nips_student.json" \
#     > "$OUT_DIR/student/solar_nips/eval.log" 2>&1
# ) &

wait
echo "[DONE] saved to: $OUT_DIR"
