#!/bin/sh -x

# Grocery_and_Gourmet_Food
python main.py --model_name BPR --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name NCF --emb_size 64 --layers '[64]' --lr 5e-4 --l2 1e-7 --dropout 0.2 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name Tensor --emb_size 64 --lr 1e-4 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name GRU4Rec --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name NARM --emb_size 64 --hidden_size 100 --attention_size 16 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name TiSASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name CFKG --emb_size 64 --margin 1 --lr 1e-4 --l2 1e-8 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name SLRC --emb_size 64 --lr 5e-4 --l2 1e-5 --check_epoch 10 --dataset 'Grocery_and_Gourmet_Food'

python main.py --model_name Chorus --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --check_epoch 10 --epoch 50 --early_stop 0 --batch_size 512 --dataset 'Grocery_and_Gourmet_Food' --stage 1
python main.py --model_name Chorus --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 --check_epoch 10 --dataset 'Grocery_and_Gourmet_Food' --base_method 'BPR' --stage 2
