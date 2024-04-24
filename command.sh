python3 train.py \
    "/home/py95/training_data.binpack" \
    "/home/py95/training_data.binpack" \
    --gpus "0,1,2,3" \
    --depth-list 512 256 256 128 \
    --dff-list 256 256 128 \
    --num-heads 16 \
    --threads 6 \
    --num-workers 1 \
    --batch-size 1024 \
    --random-fen-skipping 3 \
    --features Transformer \
    --lr 0.00009 \
    --lambda 1.0 \
    --gamma 0.998 \
    --validation-size 8192 \
    --max_epochs 4000 \
    --smolgen-hidden 64 \
    --eval-hidden-depth 128 \
    --activation-function relu \
    --policy-classification-weight 0.000001 \
    --return-policy-index yes \
    --epoch-size 8192000 \
    --default_root_dir ./training/runs/run_0 \