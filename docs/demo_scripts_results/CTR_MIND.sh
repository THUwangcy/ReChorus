#!/bin/sh -x

# CTR prediction on MIND dataset
python main.py --model_name FM --lr 5e-4 --l2 0 --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name WideDeep --lr 5e-4 --l2 0 --dropout 0.5 --layers "[64,64,64]" --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name DeepFM --lr 5e-4 --l2 0 --dropout 0.2 --layers "[512,64]" --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name AFM --lr 5e-4 --l2 0 --dropout 0 --attention_size 64 --reg_weight 1.0 --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name DCN --lr 5e-4 --l2 0 --layers "[128,64]" --cross_layer_num 4 --reg_weight 1.0 --dropout 0.8 --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name xDeepFM --lr 1e-4 --l2 0 --layers "[128,64]" --cin_layers "[8,8]" --direct 0 --reg_weight 0 --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name AutoInt --lr 5e-4 --l2 0 --dropout 0.5 --attention_size 64 --num_heads 2 --num_layers 2 --layers "[64,64,64]" --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name DCNv2 --lr 2e-3 --l2 0 --layers "[256,256,256]" --cross_layer_num 3 --mixed 1 --structure stacked --low_rank 64 --expert_num 2 --reg_weight 2.0 --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name FinalMLP --mlp1_dropout 0 --mlp2_dropout 0 --mlp1_batch_norm 0 --mlp2_batch_norm 0 --use_fs 1 --lr 5e-3 --l2 0 --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE --fs1_context c_hour_c,c_weekday_c,c_period_c,c_day_f --fs2_context i_category_c,i_subcategory_c --mlp1_hidden_units "[64]" --mlp2_hidden_units "[256]" --fs_hidden_units "[64]"

python main.py --model_name SAM --lr 5e-4 --l2 0 --interaction_type SAM3A --aggregation mean_pooling --num_layers 5 --use_residual 1 --dropout 0 --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name DIN --dropout 0 --lr 5e-3 --l2 0 --history_max 40 --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --eval_batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --att_layers "[64,64]" --dnn_layers "[64]" --loss_n BCE

python main.py --model_name DIEN --lr 5e-4 --l2 0 --history_max 30 --alpha_aux 0.1 --aux_hidden_layers "[64]" --fcn_hidden_layers "[64]" --evolving_gru_type AIGRU --dropout 0 --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE

python main.py --model_name CAN --lr 5e-4 --l2 0 --co_action_layers "[4,4]" --orders 1 --induce_vec_size 512 --history_max 20 --alpha_aux 0.1 --aux_hidden_layers "[64]" --fcn_hidden_layers "[64]" --evolving_gru_type AGRU --dropout 0 --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE