# 예: Electricity (D=370, H, P=24)
python train_teacher.py --dataset electricity_nips --epochs 50 --device cuda
# => checkpoints/teacher/teacher.pt

python train_student.py --dataset electricity_nips \
  --teacher_ckpt checkpoints/teacher/teacher.pt \
  --epochs 50 --device cuda
# => checkpoints/student/student.pt

# 평가(교사/학생 각각)
python evaluate.py --dataset electricity_nips --ckpt checkpoints/teacher/teacher.pt
python evaluate.py --dataset electricity_nips --ckpt checkpoints/student/student.pt
