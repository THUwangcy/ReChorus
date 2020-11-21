#!/bin/sh -x

# Electronics
python main.py --model_name KDA --emb_size 64 --include_attr 1 --include_val 1 --freq_rand 1 --lr 1e-4 --l2 1e-6 --num_heads 4 --num_layers 5 --gamma -1 --history_max 20 --dataset Electronics

# RecSys2017
python main.py --model_name KDA --emb_size 64 --include_attr 1 --include_val 1 --freq_rand 0 --lr 1e-4 --l2 1e-6 --num_heads 4 --num_layers 5 --gamma -1 --history_max 20 --dataset RecSys2017