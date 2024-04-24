python3 train.py \
    "/media/patrick/New Volume/chess_project/test80-2023-10-oct-2tb7p.binpack" \
    "/media/patrick/New Volume/chess_project/test80-2023-10-oct-2tb7p.binpack" \
    --gpus "0," \
    --threads 2 \
    --num-workers 2 \
    --batch-size 16384 \
    --random-fen-skipping 3 \
    --features HalfKAv2_hm^ \
    --lambda 1.0 \
    --max_epochs 400 \
    --default_root_dir ./training/runs/run_0
