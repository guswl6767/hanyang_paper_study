python run_experiments.py \
  --device cuda --epochs_teacher 200 --epochs_student 200 --num_samples 100 \
  --gpus 0,1,2,3,4,5,6,7 --batch_windows 8 \
  --results_dir /root/Storage/hanyang_paper_study/Models/checkpoints

# 이미 학습/증류/평가 결과가 들어있는 기존 폴더로 재평가만 수행
# python run_experiments.py \
#   --skip_train \
#   --device cuda --num_samples 100 \
#   --gpus 0,1,2,3,4,5,6,7 --batch_windows 8 \
#   --results_dir /root/Storage/hanyang_paper_study/Models/checkpoints/20250928-190640
