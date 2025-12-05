# 6개 데이터셋(논문 동일) 순차 학습→증류→롤링평가→CSV/MD 결과 저장
python run_experiments.py --device cuda --epochs_teacher 200 --epochs_student 200 --num_samples 100 --results_dir /root/Storage/hanyang_paper_study/Models/checkpoints
# # 이미 학습된 체크포인트로 평가만 다시 돌리고 싶다면:
# python run_experiments.py --skip_train --num_samples 100
