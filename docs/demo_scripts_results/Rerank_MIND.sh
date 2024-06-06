#!/bin/sh -x

# Impression-based Ranking and Reranking on MIND dataset
python main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 0 --loss_n BPR --dataset MINDCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --model_mode Impression

python main.py --model_name GRU4Rec --hidden_size 64 --history_max 20 --emb_size 64 --lr 2e-3 --l2 0 --loss_n BPR --dataset MINDCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --model_mode Impression

python main.py --model_name SASRec --num_layers 3 --num_heads 1 --history_max 20 --emb_size 64 --lr 5e-4 --l2 0 --loss_n BPR --dataset MINDCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --model_mode Impression

# reranking model: the corresponding .yaml config file and .pt backbone checkpoint should be prepared in ../model/{ranker_name}/ in advance
random_seed=0
# PRM+BPRMF
python main.py --model_name PRM --num_hidden_unit 64 --positionafter 0 --emb_size 64 --n_blocks 4 --num_heads 4 --lr 5e-4 --l2 0 --loss_n BPR --dataset MINDCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10,20 --main_metric NDCG@2 --num_workers 0 --ranker_name BPRMF --ranker_config_file BPRMF_best.yaml --ranker_model_file BPRMF__MINDCTR__${random_seed}__best.pt --model_mode General

# SetRank+BPRMF
python main.py --model_name SetRank --emb_size 64 --n_blocks 4 --num_heads 4 --num_hidden_unit 64 --setrank_type IMSAB --positionafter 1 --lr 1e-3 --l2 0 --loss_n BPR --dataset MINDCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --num_workers 0 --ranker_name BPRMF --ranker_config_file BPRMF_best.yaml --ranker_model_file BPRMF__MINDCTR__${random_seed}__best.pt --model_mode General

# MIR+BPRMF
python main.py --model_name MIR --emb_size 64 --num_heads 4 --num_hidden_unit 64 --lr 5e-4 --l2 0 --loss_n BPR --dataset MINDCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --num_workers 0 --ranker_name BPRMF --ranker_config_file BPRMF_best.yaml --ranker_model_file BPRMF__MINDCTR__${random_seed}__best.pt --model_mode General

# PRM+GRU4Rec
python main.py --model_name PRM --num_hidden_unit 64 --history_max 10 --positionafter 1 --emb_size 64 --n_blocks 6 --num_heads 4 --lr 2e-3 --l2 0 --loss_n BPR --dataset MINDCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10,20 --main_metric NDCG@2 --num_workers 0 --ranker_name GRU4Rec --ranker_config_file GRU4Rec_MINDCTR_best.yaml --ranker_model_file GRU4Rec__MINDCTR__${random_seed}__best.pt --model_mode Sequential

# SetRank+GRU4Rec
python main.py --model_name SetRank --emb_size 64 --history_max 20 --n_blocks 8 --num_heads 4 --num_hidden_unit 32 --setrank_type MSAB --positionafter 1 --lr 1e-3 --l2 0 --loss_n BPR --dataset MINDCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --num_workers 0 --ranker_name GRU4Rec --ranker_config_file GRU4Rec_MINDCTR_best.yaml --ranker_model_file GRU4Rec__MINDCTR__${random_seed}__best.pt --model_mode Sequential

# MIR+GRU4Rec
python main.py --model_name MIR --history_max 10 --emb_size 64 --num_heads 2 --num_hidden_unit 64 --lr 1e-3 --l2 0 --loss_n BPR --dataset MINDCTR --path ../data/ --metric NDCG,HR --topk 1,2,3,5,10 --main_metric NDCG@2 --num_workers 0 --ranker_name GRU4Rec --ranker_config_file GRU4Rec_MINDCTR_best.yaml --ranker_model_file GRU4Rec__MINDCTR__${random_seed}__best.pt --model_mode Sequential