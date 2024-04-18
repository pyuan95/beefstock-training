python3 train.py \
    "/media/patrick/New Volume/chess_project/test80-2023-10-oct-2tb7p.binpack" \
    "/media/patrick/New Volume/chess_project/test80-2023-10-oct-2tb7p.binpack" \
    --gpus "0," \
    --threads 24 \
    --num-workers 24 \
    --batch-size 16384 \
    --random-fen-skipping 3 \
    --features HalfKA \
    --lambda 1.0 \
    --max_epochs 400 \
    --default_root_dir ./training/runs/run_0
