python3 train.py \
    "/home/py95/data/2022.binpack" \
    "/home/py95/data/2023.binpack" \
    --gpus "0,1,2,3" \
    --depth-list 1024,512,512,512,128 \
    --dff-list 768,512,512,256 \
    --residual-connection-list yes,yes,yes,no \
    --num-heads 32 \
    --threads 6 \
    --num-workers 1 \
    --batch-size 1024 \
    --random-fen-skipping 16 \
    --features Transformer \
    --lr 0.00009 \
    --lambda 1.0 \
    --gamma 0.999 \
    --validation-size 8192 \
    --max_epochs 4000 \
    --smolgen-initial 64 \
    --smolgen-hidden 64 \
    --eval-hidden-depth 128 \
    --activation-function relu \
    --policy-classification-weight 0.000002 \
    --return-policy-index yes \
    --epoch-size 8192000 \
    --default_root_dir ./training/runs/transformer \
