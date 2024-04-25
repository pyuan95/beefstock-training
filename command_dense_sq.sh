python3 train.py \
    "/home/py95/data/training_data.binpack" \
    "/home/py95/data/training_data.binpack" \
    --gpus "0,1,2,3" \
    --threads 6 \
    --depth 256 \
    --num-workers 1 \
    --batch-size 8192 \
    --random-fen-skipping 3 \
    --features DenseSquares \
    --lr 0.0001 \
    --lambda 1.0 \
    --gamma 0.999 \
    --validation-size 8192 \
    --max_epochs 4000 \
    --eval-hidden-depth 256 \
    --activation-function silu \
    --policy-classification-weight 0.000002 \
    --return-policy-index yes \
    --epoch-size 8192000 \
    --default_root_dir ./training/runs/run_0 \

# "/media/patrick/New Volume/chess_project/test80-2023-10-oct-2tb7p.binpack"
# "/home/py95/data/training_data.binpack"