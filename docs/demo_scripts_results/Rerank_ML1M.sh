#!/bin/sh -x

# Impression-based Ranking and Reranking on MovieLens-1M dataset
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 0 --loss_n BPR --dataset ML_1MCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --model_mode Impression

python main.py --model_name LightGCN --n_layers 1 --emb_size 64 --lr 1e-3 --l2 0 --loss_n BPR --dataset ML_1MCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --model_mode Impression

python main.py --model_name GRU4Rec --hidden_size 32 --history_max 30 --emb_size 64 --lr 1e-3 --l2 1e-6 --loss_n BPR --dataset ML_1MCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --model_mode Impression

python main.py --model_name SASRec --num_layers 3 --num_heads 2 --history_max 20 --emb_size 64 --lr 2e-4 --l2 1e-6 --loss_n BPR --dataset ML_1MCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --model_mode Impression

# reranking model: the corresponding .yaml config file and .pt backbone checkpoint should be prepared in ../model/{ranker_name}/ in advance
random_seed=0
# PRM+LightGCN
python main.py --model_name PRM --positionafter 1 --num_hidden_unit 256 --emb_size 64 --n_blocks 4 --num_heads 2 --lr 1e-3 --l2 1e-6 --loss_n BPR --dataset ML_1MCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10,20 --main_metric NDCG@2 --num_workers 0 --ranker_name LightGCN --ranker_config_file ml_LightGCN_best.yaml --ranker_model_file LightGCN__ML_1MCTR__${random_seed}__best.pt --model_mode General

# SetRank+LightGCN
python main.py --model_name SetRank --emb_size 64 --n_blocks 4 --num_heads 4 --num_hidden_unit 32 --setrank_type IMSAB --positionafter 1 --lr 2e-4 --l2 1e-4 --loss_n BPR --dataset ML_1MCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --num_workers 0 --ranker_name LightGCN --ranker_config_file ml_LightGCN_best.yaml --ranker_model_file LightGCN__ML_1MCTR__${random_seed}__best.pt --model_mode General

# MIR+LightGCN
python main.py --model_name MIR --emb_size 64 --history_max 10 --num_heads 2 --num_hidden_unit 32 --lr 2e-3 --l2 0 --loss_n BPR --dataset ML_1MCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --num_workers 0 --ranker_name LightGCN --ranker_config_file ml_LightGCN_best.yaml --ranker_model_file LightGCN__ML_1MCTR__${random_seed}__best.pt --model_mode General

# PRM+SASRec
python main.py --model_name PRM --emb_size 64 --history_max 20 --n_blocks 1 --num_heads 1 --lr 1e-3 --l2 0 --loss_n BPR --dataset ML_1MCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10,20 --main_metric NDCG@2 --num_workers 0 --ranker_name SASRec --ranker_config_file ml_SASRec_best.yaml --ranker_model_file SASRec__ML_1MCTR__${random_seed}__best.pt --model_mode Sequential

# SetRank+SASRec
python main.py --model_name SetRank --emb_size 64 --history_max 20 --n_blocks 4 --num_heads 1 --num_hidden_unit 64 --setrank_type MSAB --positionafter 1 --lr 2e-4 --l2 1e-6 --loss_n BPR --dataset ML_1MCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --num_workers 0 --ranker_name SASRec_test --ranker_config_file ml_SASRec_best.yaml --ranker_model_file SASRec_test__ML_1MCTR__${random_seed}__best.pt --model_mode Sequential

# MIR+SASRec
python main.py --model_name MIR --history_max 20 --emb_size 64 --num_heads 4 --num_hidden_unit 128 --lr 2e-4 --l2 1e-6 --loss_n BPR --dataset ML_1MCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --num_workers 0 --ranker_name SASRec_test --ranker_config_file ml_SASRec_best.yaml --ranker_model_file SASRec_test__ML_1MCTR__${random_seed}__best.pt --model_mode Sequential
