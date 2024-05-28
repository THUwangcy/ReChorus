#!/bin/sh -x

# Top-k recommendation on MIND dataset
python main.py --model_name FM --lr 1e-3 --l2 1e-4 --dataset MINDTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name WideDeep --lr 5e-4 --l2 0 --dropout 0.5 --layers "[64,64,64]" --dataset MINDTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name DeepFM --lr 2e-3 --l2 1e-4 --dropout 0.5 --layers "[64]" --dataset MINDTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name AFM --lr 2e-3 --l2 1e-6 --dropout 0.8 --attention_size 64 --reg_weight 0 --dataset MINDTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name DCN --lr 2e-3 --l2 1e-4 --layers "[512,128]" --cross_layer_num 5 --reg_weight 2.0 --dataset MINDTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name xDeepFM --lr 2e-3 --l2 1e-4 --layers "[64,64,64]" --cin_layers "[8,8]" --direct 0 --reg_weight 2.0 --dataset MINDTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name AutoInt --lr 2e-3 --l2 1e-4 --dropout 0.8 --attention_size 64 --num_heads 1 --num_layers 1 --layers "[512]" --dataset MINDTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name DCNv2 --lr 2e-3 --l2 1e-4 --layers "[64,64,64]" --cross_layer_num 4 --mixed 1 --structure parallel --low_rank 64 --expert_num 1 --reg_weight 2.0 --dataset MINDTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name FinalMLP --mlp1_hidden_units "[256,256,256]" --mlp2_hidden_units "[256,128,64]" --mlp1_dropout 0.5 --mlp2_dropout 0.5 --use_fs 0 --mlp1_batch_norm 0 --mlp2_batch_norm 1 --lr 5e-3 --l2 0 --fs1_context c_hour_c,c_weekday_c,c_period_c,c_day_f --fs2_context i_category_c,i_subcategory_c --dataset MINDTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name SAM --lr 2e-4 --l2 1e-4 --interaction_type SAM3A --aggregation mean_pooling --num_layers 1 --use_residual 0 --dropout 0.0 --dataset MINDTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name DIN --lr 2e-3 --l2 0 --history_max 10 --dataset MINDTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 128 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK --att_layers "[256]" --dnn_layers "[64,64]" --dropout 0.8

python main.py --model_name DIEN --lr 5e-4 --l2 1e-4 --history_max 30 --alpha_aux 0.1 --aux_hidden_layers "[64]" --fcn_hidden_layers "[64]" --evolving_gru_type AUGRU --dropout 0 --dataset MINDTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 32 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK

python main.py --model_name CAN --lr 2e-3 --l2 0 --co_action_layers "[4,4]" --orders 1 --induce_vec_size 512 --history_max 30 --alpha_aux 0.1 --aux_hidden_layers "[64]" --fcn_hidden_layers "[64]" --evolving_gru_type AUGRU --dropout 0 --dataset MINDTOPK --path ../data/ --num_neg 1 --batch_size 256 --eval_batch_size 32 --metric NDCG,HR --topk 3,5,10,20 --include_item_features 1 --include_situation_features 1 --model_mode TopK