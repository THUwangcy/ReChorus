#!/bin/sh -x

# Beauty
python main.py --model_name ContraRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Beauty --encoder BERT4Rec --num_neg 1 --ctc_temp 1 --ccc_temp 0.2 --batch_size 4096 --gamma 1
python main.py --model_name ContraRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Beauty --encoder BERT4Rec --num_neg 64 --ctc_temp 0.5 --ccc_temp 0.2 --batch_size 4096 --gamma 1

# Yelp2018
python main.py --model_name ContraRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Yelp2018 --encoder BERT4Rec --num_neg 1 --ctc_temp 1 --ccc_temp 0.2 --batch_size 4096 --gamma 5
python main.py --model_name ContraRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Yelp2018 --encoder BERT4Rec --num_neg 64 --ctc_temp 0.5 --ccc_temp 0.2 --batch_size 4096 --gamma 5

# Gowalla
python main.py --model_name ContraRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Gowalla --encoder BERT4Rec --num_neg 1 --ctc_temp 1 --ccc_temp 0.2 --batch_size 4096 --gamma 0.01
python main.py --model_name ContraRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Gowalla --encoder BERT4Rec --num_neg 64 --ctc_temp 0.5 --ccc_temp 0.2 --batch_size 4096 --gamma 0.1
