#!/bin/sh -x

# Top-k recommendation on MovieLens-1M dataset
python main.py --model_name FM --lr 1e-3 --l2 0 --dataset ML_1MTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name WideDeep --lr 1e-3 --l2 0 --dropout 0.5 --layers "[64,64,64]" --dataset ML_1MTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name DeepFM --lr 5e-4 --l2 1e-6 --dropout 0.5 --layers "[512,128]" --dataset ML_1MTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name AFM --lr 5e-3 --l2 0 --dropout 0.5 --attention_size 64 --reg_weight 2.0 --dataset ML_1MTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name DCN --lr 5e-4 --l2 1e-4 --layers "[64,64,64]" --cross_layer_num 2 --reg_weight 0.5 --dataset ML_1MTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name xDeepFM --lr 5e-4 --l2 0 --dropout 0.8 --layers "[512,512,512]" --cin_layers "[8,8]" --direct 0 --reg_weight 1.0 --dataset ML_1MTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name AutoInt --lr 2e-3 --l2 0 --dropout 0 --attention_size 64 --num_heads 2 --num_layers 2 --layers "[256]" --dataset ML_1MTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name DCNv2 --dropout 0 --lr 1e-3 --l2 1e-4 --layers "[256,64]" --cross_layer_num 2 --mixed 0 --structure stacked --low_rank 64 --expert_num 2 --reg_weight 2.0 --dataset ML_1MTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name FinalMLP --mlp1_hidden_units "[64]" --mlp2_hidden_units "[64,64,64]" --mlp1_dropout 0.5 --mlp2_dropout 0.2 --use_fs 1 --mlp1_batch_norm 0 --mlp2_batch_norm 0 --lr 1e-3 --l2 0 --fs1_context c_hour_c,c_weekday_c,c_period_c,c_day_f --fs2_context i_genre_c,i_title_c --dataset ML_1MTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name SAM --lr 1e-3 --l2 1e-4 --interaction_type SAM3A --aggregation mean_pooling --num_layers 1 --use_residual 1 --dropout 0.2 --dataset ML_1MTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name DIN --lr 2e-3 --l2 1e-6 --history_max 10 --dataset ML_1MTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK --att_layers "[64,64,64]" --dnn_layers "[128,64]" --dropout 0.5

python main.py --model_name DIEN --lr 5e-4 --l2 1e-6 --history_max 20 --alpha_aux 0.1 --aux_hidden_layers "[64]" --fcn_hidden_layers "[64]" --evolving_gru_type AIGRU --dropout 0 --dataset ML_1MTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 32 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name CAN --lr 5e-4 --l2 1e-4 --co_action_layers "[4,4]" --orders 2 --induce_vec_size 1024 --history_max 10 --alpha_aux 0.1 --aux_hidden_layers "[64]" --fcn_hidden_layers "[64,64]" --evolving_gru_type AIGRU --dropout 0.2 --dataset ML_1MTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 32 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK