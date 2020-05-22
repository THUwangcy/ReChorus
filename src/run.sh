#!/bin/sh -x

# Grocery_and_Gourmet_Food
 python main.py --model_name Chorus --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --check_epoch 10 --epoch 50 --early_stop 0 --batch_size 512 --dataset 'Grocery_and_Gourmet_Food' --stage 1

 python main.py --model_name Chorus --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 --check_epoch 10 --dataset 'Grocery_and_Gourmet_Food' --base_method 'BPR' --stage 2

 python main.py --model_name Chorus --emb_size 64 --margin 1 --lr_scale 0.1 --lr 5e-4 --l2 1e-5 --check_epoch 10 --dataset 'Grocery_and_Gourmet_Food' --base_method 'GMF' --stage 2

 python main.py --model_name SLRC --emb_size 64 --lr 5e-4 --l2 1e-5 --check_epoch 10 --dataset 'Grocery_and_Gourmet_Food'

 python main.py --model_name CFKG --emb_size 64 --margin 1. --lr 5e-4 --l2 1e-5 --check_epoch 10 --batch_size 512 --dataset 'Grocery_and_Gourmet_Food'

 python main.py --model_name NARM --emb_size 64 --hidden_size 200 --attention_size 50 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

 python main.py --model_name GRU4Rec --emb_size 64 --hidden_size 200 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'

 python main.py --model_name Tensor --emb_size 64 --lr 5e-4 --l2 0 --dataset 'Grocery_and_Gourmet_Food'

 python main.py --model_name GMF --layers '[64]' --lr 5e-4 --l2 0 --dataset 'Grocery_and_Gourmet_Food'

 python main.py --model_name BPR --emb_size 64 --lr 1e-3 --l2 0 --dataset 'Grocery_and_Gourmet_Food'


# Cell_Phones_and_Accessories
 python main.py --model_name Chorus --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --check_epoch 10 --epoch 50 --early_stop 0 --batch_size 512 --dataset 'Cell_Phones_and_Accessories' --stage 1

 python main.py --model_name Chorus --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 --check_epoch 10 --dataset 'Cell_Phones_and_Accessories' --base_method 'BPR' --stage 2

 python main.py --model_name Chorus --emb_size 64 --margin 1 --lr_scale 0.1 --lr 5e-4 --l2 1e-5 --check_epoch 10 --dataset 'Cell_Phones_and_Accessories' --base_method 'GMF' --stage 2

 python main.py --model_name SLRC --emb_size 64 --lr 5e-4 --l2 1e-5 --check_epoch 10 --dataset 'Cell_Phones_and_Accessories'

 python main.py --model_name CFKG --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --check_epoch 10 --batch_size 512 --dataset 'Cell_Phones_and_Accessories'

 python main.py --model_name NARM --emb_size 64 --hidden_size 200 --attention_size 50 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Cell_Phones_and_Accessories'

 python main.py --model_name GRU4Rec --emb_size 64 --hidden_size 200 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Cell_Phones_and_Accessories'

 python main.py --model_name Tensor --emb_size 64 --lr 5e-4 --l2 0 --dataset 'Cell_Phones_and_Accessories'

 python main.py --model_name GMF --layers '[64]' --lr 5e-4 --l2 0 --dataset 'Cell_Phones_and_Accessories'

 python main.py --model_name BPR --emb_size 64 --lr 1e-3 --l2 0 --dataset 'Cell_Phones_and_Accessories'


# Home_and_Kitchen
 python main.py --model_name Chorus --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --check_epoch 10 --epoch 50 --early_stop 0 --batch_size 512 --dataset 'Home_and_Kitchen' --stage 1

 python main.py --model_name Chorus --emb_size 64 --margin 1 --lr_scale 0.1 --lr 2e-4 --l2 0 --check_epoch 10 --dataset 'Home_and_Kitchen' --base_method 'BPR' --stage 2

 python main.py --model_name Chorus --emb_size 64 --margin 1 --lr_scale 0.1 --lr 2e-4 --l2 1e-5 --check_epoch 10 --dataset 'Home_and_Kitchen' --base_method 'GMF' --stage 2

 python main.py --model_name SLRC --emb_size 64 --lr 5e-4 --l2 1e-5 --check_epoch 10 --dataset 'Home_and_Kitchen'

 python main.py --model_name CFKG --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --check_epoch 10 --batch_size 512 --dataset 'Home_and_Kitchen'

 python main.py --model_name NARM --emb_size 64 --hidden_size 200 --attention_size 50 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Home_and_Kitchen'

 python main.py --model_name GRU4Rec --emb_size 64 --hidden_size 200 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Home_and_Kitchen'

 python main.py --model_name Tensor --emb_size 64 --lr 5e-4 --l2 0 --dataset 'Home_and_Kitchen'

 python main.py --model_name GMF --lr 1e-4 --l2 0 --dataset 'Home_and_Kitchen'

 python main.py --model_name BPR --emb_size 64 --lr 1e-3 --l2 0 --dataset 'Home_and_Kitchen'

