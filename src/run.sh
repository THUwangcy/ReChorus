#!/bin/sh -x

# Beauty
python main.py --model_name TiMiRec --lr 1e-4 --l2 1e-6 --history_max 20 --K 4 --add_pos 1 --add_trm 1 --stage pretrain --dataset Beauty
python main.py --model_name TiMiRec --lr 1e-4 --l2 1e-6 --history_max 20 --K 4 --add_pos 1 --add_trm 1 --stage finetune --check_epoch 10 --temp 0.1 --n_layers 1 --dataset Beauty

# ml-1m
python main.py --model_name TiMiRec --lr 1e-4 --l2 1e-6 --history_max 20 --K 2 --add_pos 1 --add_trm 1 --stage pretrain --dataset ml-1m
python main.py --model_name TiMiRec --lr 1e-4 --l2 1e-6 --history_max 20 --K 2 --add_pos 1 --add_trm 1 --stage finetune --check_epoch 10 --temp 0.5 --n_layers 2 --dataset ml-1m