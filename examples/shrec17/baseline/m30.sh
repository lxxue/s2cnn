source activate cuda9
CUDA_VISIBLE_DEVICES=0 \
    python ../scripts/baseline_train.py \
    --log_dir log_30 \
    --model_path ../models/baseline.py \
    --augmentation 12 \
    --dataset ModelNet30 \
    --batch_size 32 \
    --learning_rate 0.5
