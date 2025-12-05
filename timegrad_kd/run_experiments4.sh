#!/usr/bin/env bash
set -eu


# 공통 재현 설정
SEED=2025
NUM=100
BWIN=8

# 날짜/타임스탬프 버킷
BASE="/root/Storage/hanyang_paper_study/Models/checkpoints"
DAY="$(date +%Y%m%d)"
RUN="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$BASE/$DAY/$RUN"
echo "[OUT] $OUT_DIR"
mkdir -p "$OUT_DIR"

# ★ 추가: student/<dataset> 디렉토리 미리 생성
for ds in electricity_nips traffic_nips taxi_30min wiki-rolling_nips solar_nips; do
  mkdir -p "$OUT_DIR/student/$ds"
done

########################################
# ---------- electricity ---------- (GPU 0)
(
  CUDA_VISIBLE_DEVICES=1 \
  python train_student_2.py --dataset electricity_nips \
    --teacher_ckpt "$BASE/20250928-222915/teacher/electricity_nips/teacher.pt" \
    --epochs 200 --device cuda \
    --n_steps_s 40 --res_channels 8 --res_blocks 6 \
    --kd_warmup 20 --lambda_kd 0.5 --mu_kd 0.05 --gamma_x0 0.10 --grad_clip 1.0 \
    --save_dir "$OUT_DIR/student/electricity_nips" \
    > "$OUT_DIR/student/electricity_nips/train.log" 2>&1

  CUDA_VISIBLE_DEVICES=1 \
  python evaluate_2.py --dataset electricity_nips \
    --ckpt "$OUT_DIR/student/electricity_nips/student.pt" \
    --device cuda --rolling --num_samples "$NUM" --batch_windows "$BWIN" --seed "$SEED" \
    --save_json "$OUT_DIR/student/electricity_nips/electricity_nips_student.json" \
    > "$OUT_DIR/student/electricity_nips/eval.log" 2>&1
) &

########################################
# ---------- traffic ---------- (GPU 1)
(
  CUDA_VISIBLE_DEVICES=2 \
  python train_student_2.py --dataset traffic_nips \
    --teacher_ckpt "$BASE/20250928-222915/teacher/traffic_nips/teacher.pt" \
    --epochs 200 --device cuda \
    --n_steps_s 40 --res_channels 12 --res_blocks 8 \
    --kd_warmup 20 --lambda_kd 0.5 --mu_kd 0.05 --gamma_x0 0.15 --grad_clip 1.0 \
    --save_dir "$OUT_DIR/student/traffic_nips" \
    > "$OUT_DIR/student/traffic_nips/train.log" 2>&1

  CUDA_VISIBLE_DEVICES=2 \
  python evaluate_2.py --dataset traffic_nips \
    --ckpt "$OUT_DIR/student/traffic_nips/student.pt" \
    --device cuda --rolling --num_samples "$NUM" --batch_windows "$BWIN" --seed "$SEED" \
    --save_json "$OUT_DIR/student/traffic_nips/traffic_nips_student.json" \
    > "$OUT_DIR/student/traffic_nips/eval.log" 2>&1
) &

########################################
# ---------- taxi ---------- (GPU 2)
(
  CUDA_VISIBLE_DEVICES=3 \
  python train_student_2.py --dataset taxi_30min \
    --teacher_ckpt "$BASE/20250928-222915/teacher/taxi_30min/teacher.pt" \
    --epochs 200 --device cuda \
    --n_steps_s 40 --res_channels 12 --res_blocks 8 \
    --kd_warmup 20 --lambda_kd 0.5 --mu_kd 0.05 --gamma_x0 0.15 --grad_clip 1.0 \
    --save_dir "$OUT_DIR/student/taxi_30min" \
    > "$OUT_DIR/student/taxi_30min/train.log" 2>&1

  CUDA_VISIBLE_DEVICES=3 \
  python evaluate_2.py --dataset taxi_30min \
    --ckpt "$OUT_DIR/student/taxi_30min/student.pt" \
    --device cuda --rolling --num_samples "$NUM" --batch_windows "$BWIN" --seed "$SEED" \
    --save_json "$OUT_DIR/student/taxi_30min/taxi_30min_student.json" \
    > "$OUT_DIR/student/taxi_30min/eval.log" 2>&1
) &

########################################
# ---------- wiki ---------- (GPU 3)
(
  CUDA_VISIBLE_DEVICES=5 \
  python train_student_2.py --dataset wiki-rolling_nips \
    --teacher_ckpt "$BASE/20250928-222915/teacher/wiki-rolling_nips/teacher.pt" \
    --epochs 200 --device cuda \
    --n_steps_s 50 --res_channels 16 --res_blocks 8 \
    --kd_warmup 20 --lambda_kd 0.5 --mu_kd 0.05 --gamma_x0 0.20 --grad_clip 1.0 \
    --save_dir "$OUT_DIR/student/wiki-rolling_nips" \
    > "$OUT_DIR/student/wiki-rolling_nips/train.log" 2>&1

  CUDA_VISIBLE_DEVICES=5 \
  python evaluate_2.py --dataset wiki-rolling_nips \
    --ckpt "$OUT_DIR/student/wiki-rolling_nips/student.pt" \
    --device cuda --rolling --num_samples "$NUM" --batch_windows "$BWIN" --seed "$SEED" \
    --save_json "$OUT_DIR/student/wiki-rolling_nips/wiki-rolling_nips_student.json" \
    > "$OUT_DIR/student/wiki-rolling_nips/eval.log" 2>&1
) &

########################################
# ---------- solar (권장) ---------- (GPU 4)
(
  CUDA_VISIBLE_DEVICES=6 \
  python train_student_2.py --dataset solar_nips \
    --teacher_ckpt "$BASE/20250928-222915/teacher/solar_nips/teacher.pt" \
    --epochs 200 --device cuda \
    --n_steps_s 40 --res_channels 8 --res_blocks 6 \
    --kd_warmup 20 --lambda_kd 0.5 --mu_kd 0.05 --gamma_x0 0.10 --grad_clip 1.0 \
    --save_dir "$OUT_DIR/student/solar_nips" \
    > "$OUT_DIR/student/solar_nips/train.log" 2>&1

  CUDA_VISIBLE_DEVICES=6 \
  python evaluate_2.py --dataset solar_nips \
    --ckpt "$OUT_DIR/student/solar_nips/student.pt" \
    --device cuda --rolling --num_samples "$NUM" --batch_windows "$BWIN" --seed "$SEED" \
    --save_json "$OUT_DIR/student/solar_nips/solar_nips_student.json" \
    > "$OUT_DIR/student/solar_nips/eval.log" 2>&1
) &

wait
echo "[DONE] saved to: $OUT_DIR"
