#!/bin/sh -x

# CTR prediction on MovieLens-1M dataset
python main.py --model_name FM --lr 1e-3 --l2 1e-4 --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name WideDeep --lr 5e-3 --l2 0 --dropout 0.5 --layers "[64,64,64]" --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name DeepFM --lr 1e-3 --l2 1e-4 --dropout 0.2 --layers "[512,128]" --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name AFM --lr 5e-4 --l2 1e-4 --dropout 0.8 --attention_size 128 --reg_weight 0.5 --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name DCN --lr 5e-4 --l2 1e-4 --layers "[512,128]" --cross_layer_num 1 --reg_weight 0.5 --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name xDeepFM --lr 1e-3 --l2 1e-4 --layers "[512,512,512]" --cin_layers "[8,8]" --direct 0 --reg_weight 0 --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name AutoInt --lr 2e-3 --l2 1e-6 --dropout 0.2 --attention_size 64 --num_heads 2 --num_layers 2 --layers "[64,64,64]" --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name DCNv2 --lr 1e-3 --l2 1e-4 --layers "[256,256,256]" --cross_layer_num 3 --mixed 0 --structure parallel --low_rank 64 --expert_num 1 --reg_weight 2.0 --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name FinalMLP --mlp1_dropout 0.2 --mlp2_dropout 0.5 --mlp1_batch_norm 1 --mlp2_batch_norm 1 --use_fs 1 --lr 5e-3 --l2 1e-6 --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE --fs1_context c_hour_c,c_weekday_c,c_period_c,c_day_f --fs2_context i_genre_c,i_title_c --mlp1_hidden_units "[64]" --mlp2_hidden_units "[64,64]" --fs_hidden_units "[256,64]"

python main.py --model_name SAM --lr 1e-3 --l2 1e-4 --interaction_type SAM3A --aggregation mean_pooling --num_layers 1 --use_residual 0 --dropout 0.5 --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name DIN --history_max 20 --lr 5e-4 --l2 1e-4 --dnn_layers "[512,64]" --att_layers "[64]" --dropout 0.5 --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name DIEN --lr 5e-3 --l2 1e-6 --history_max 20 --alpha_aux 0.5 --aux_hidden_layers "[64,64,64]" --fcn_hidden_layers "[256]" --evolving_gru_type AIGRU --dropout 0.2 --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name CAN --lr 2e-3 --l2 1e-4 --co_action_layers "[4,4,4]" --orders 1 --induce_vec_size 1024 --history_max 30 --alpha_aux 0.1 --aux_hidden_layers "[64,64,64]" --fcn_hidden_layers "[256,128]" --evolving_gru_type AIGRU --dropout 0.2 --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

