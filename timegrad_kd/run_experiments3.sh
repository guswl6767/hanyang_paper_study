# electricity
# ---------- electricity ----------
python train_student_2.py --dataset electricity_nips \
  --teacher_ckpt /root/Storage/hanyang_paper_study/Models/checkpoints/20250928-222915/teacher/electricity_nips/teacher.pt \
  --epochs 200 --device cuda \
  --n_steps_s 30 --res_channels 8 --res_blocks 6 \
  --kd_warmup 20 --lambda_kd 0.5 --mu_kd 0.05 --gamma_x0 0.10 --grad_clip 1.0

python evaluate_2.py --dataset electricity_nips \
  --ckpt /root/Storage/hanyang_paper_study/Models/checkpoints/v2/student/electricity_nips/student.pt \
  --device cuda --rolling --num_samples 100 --batch_windows 8 --seed 2025 --save_json /root/Storage/hanyang_paper_study/Models/checkpoints/v2/student/electricity_nips/electricity_student.json

# ---------- traffic ----------
python train_student_2.py --dataset traffic_nips \
  --teacher_ckpt /root/Storage/hanyang_paper_study/Models/checkpoints/20250928-222915/teacher/traffic_nips/teacher.pt \
  --epochs 200 --device cuda \
  --n_steps_s 40 --res_channels 12 --res_blocks 8 \
  --kd_warmup 20 --lambda_kd 0.5 --mu_kd 0.05 --gamma_x0 0.15 --grad_clip 1.0

python evaluate_2.py --dataset traffic_nips \
  --ckpt /root/Storage/hanyang_paper_study/Models/checkpoints/v2/student/traffic_nips/student.pt \
  --device cuda --rolling --num_samples 100 --batch_windows 8 --seed 2025 --save_json /root/Storage/hanyang_paper_study/Models/checkpoints/v2/student/traffic_nips/traffic_nips_student.json

# ---------- taxi ----------
python train_student_2.py --dataset taxi_30min \
  --teacher_ckpt /root/Storage/hanyang_paper_study/Models/checkpoints/20250928-222915/teacher/taxi_30min/teacher.pt \
  --epochs 200 --device cuda \
  --n_steps_s 40 --res_channels 12 --res_blocks 8 \
  --kd_warmup 20 --lambda_kd 0.5 --mu_kd 0.05 --gamma_x0 0.15 --grad_clip 1.0

python evaluate_2.py --dataset taxi_30min \
  --ckpt /root/Storage/hanyang_paper_study/Models/checkpoints/v2/student/taxi_30min/student.pt \
  --device cuda --rolling --num_samples 100 --batch_windows 8 --seed 2025 --save_json /root/Storage/hanyang_paper_study/Models/checkpoints/v2/student/taxi_30min/taxi_30min_student.json
 
# ---------- wiki ----------
python train_student_2.py --dataset wiki-rolling_nips \
  --teacher_ckpt /root/Storage/hanyang_paper_study/Models/checkpoints/20250928-222915/teacher/wiki-rolling_nips/teacher.pt \
  --epochs 200 --device cuda \
  --n_steps_s 50 --res_channels 16 --res_blocks 8 \
  --kd_warmup 20 --lambda_kd 0.5 --mu_kd 0.05 --gamma_x0 0.20 --grad_clip 1.0

python evaluate_2.py --dataset wiki-rolling_nips \
  --ckpt /root/Storage/hanyang_paper_study/Models/checkpoints/v2/student/wiki-rolling_nips/student.pt \
  --device cuda --rolling --num_samples 100 --batch_windows 8 --seed 2025 --save_json /root/Storage/hanyang_paper_study/Models/checkpoints/v2/student/wiki-rolling_nips/wiki-rolling_nips_student.json

# ---------- solar (권장) ----------
python train_student_2.py --dataset solar_nips \
  --teacher_ckpt /root/Storage/hanyang_paper_study/Models/checkpoints/20250928-222915/teacher/solar_nips/teacher.pt \
  --epochs 200 --device cuda \
  --n_steps_s 30 --res_channels 8 --res_blocks 6 \
  --kd_warmup 20 --lambda_kd 0.5 --mu_kd 0.05 --gamma_x0 0.10 --grad_clip 1.0

python evaluate_2.py --dataset solar_nips \
  --ckpt /root/Storage/hanyang_paper_study/Models/checkpoints/v2/student/solar_nips/student.pt \
  --device cuda --rolling --num_samples 100 --batch_windows 8 --seed 2025 --save_json /root/Storage/hanyang_paper_study/Models/checkpoints/v2/student/solar_nips/solar_nips_student.json