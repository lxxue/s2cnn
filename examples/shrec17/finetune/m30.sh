CUDA_VISIBLE_DEVICES=0 \
    python /home/lixin/Documents/s2cnn/examples/shrec17/scripts/baseline_train.py \
    --log_dir log_30 \
    --model_path /home/lixin/Documents/s2cnn/examples/shrec17/models/baseline_model.py \
    --augmentation 12 \
    --dataset ModelNet30 \
    --num_cls 30 \
    --batch_size 32 \
    --num_worker 12 \
    --learning_rate 0.5
