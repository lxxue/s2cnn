CUDA_VISIBLE_DEVICES=1 \
    python /home/lixin/Documents/s2cnn/examples/shrec17/scripts/baseline_train.py \
    --log_dir log_10 \
    --model_path /home/lixin/Documents/s2cnn/examples/shrec17/models/baseline_model.py \
    --augmentation 12 \
    --dataset ModelNet10 \
    --num_cls 10 \
    --batch_size 32 \
    --num_worker 12 \
    --learning_rate 0.5
