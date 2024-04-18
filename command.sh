python3 train.py \
    "/media/patrick/New Volume/chess_project/test80-2023-10-oct-2tb7p.binpack" \
    "/media/patrick/New Volume/chess_project/test80-2023-10-oct-2tb7p.binpack" \
    --gpus "0," \
    --depth 256 \
    --dff 384 \
    --n_heads 8 \
    --n_layers 2 \
    --threads 6 \
    --num-workers 6 \
    --batch-size 2304 \
    --random-fen-skipping 3 \
    --features Transformer \
    --lr 0.000075 \
    --lambda 1.0 \
    --gamma 0.9 \
    --max_epochs 400 \
    --default_root_dir ./training/runs/run_0 \

    #     --resume-from-model ./training/runs/run_0/lightning_logs/version_0/checkpoints/last.ckpt \
