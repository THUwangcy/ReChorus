# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" TiMiRec
Reference:
    "Target Interest Distillation for Multi-Interest Recommendation"
    Wang et al., CIKM'2022.
CMD example:
    python main.py --model_name TiMiRec --dataset Grocery_and_Gourmet_Food \
                   --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --K 6 \
                   --add_pos 1 --add_trm 1 --stage pretrain
    python main.py --model_name TiMiRec --dataset Grocery_and_Gourmet_Food \
                   --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 20 --K 6 \
                   --add_pos 1 --add_trm 1 --stage finetune --temp 1 --n_layers 1
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.BaseModel import SequentialModel
from utils import layers


class TiMiRec(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'attn_size', 'K', 'temp', 'add_pos', 'add_trm', 'n_layers']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--attn_size', type=int, default=8,
                            help='Size of attention vectors.')
        parser.add_argument('--K', type=int, default=2,
                            help='Number of hidden interests.')
        parser.add_argument('--add_pos', type=int, default=1,
                            help='Whether add position embedding in extractor.')
        parser.add_argument('--add_trm', type=int, default=1,
                            help='Whether add the transformer layer in extractor.')
        parser.add_argument('--temp', type=float, default=1,
                            help='Temperature in knowledge distillation loss.')
        parser.add_argument('--n_layers', type=int, default=1,
                            help='Number of the projection layer.')
        parser.add_argument('--stage', type=str, default='finetune',
                            help='Training stage: pretrain / finetune.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.attn_size = args.attn_size
        self.K = args.K
        self.add_pos = args.add_pos
        self.add_trm = args.add_trm
        self.temp = args.temp
        self.n_layers = args.n_layers
        self.stage = args.stage
        self.max_his = args.history_max
        self._define_params()
        self.apply(self.init_weights)

        self.extractor_path = '../model/TiMiRec/Extractor__{}__{}__emb_size={}__K={}__add_pos={}__add_trm={}.pt' \
            .format(corpus.dataset, args.random_seed, self.emb_size, self.K, self.add_pos, self.add_trm)
        if self.stage == 'pretrain':
            self.model_path = self.extractor_path
        elif self.stage == 'finetune':
            if os.path.exists(self.extractor_path):
                self.load_model(self.extractor_path)
            else:
                logging.info('Train from scratch!')
        else:
            raise ValueError('Invalid stage: ' + self.stage)

    def _define_params(self):
        self.interest_extractor = MultiInterestExtractor(
            self.K, self.item_num, self.emb_size, self.attn_size, self.max_his, self.add_pos, self.add_trm)
        if self.stage == 'finetune':
            self.interest_predictor = InterestPredictor(self.item_num, self.emb_size)
            self.proj = nn.Sequential()
            for i, _ in enumerate(range(self.n_layers - 1)):
                self.proj.add_module('proj_' + str(i), nn.Linear(self.emb_size, self.emb_size))
                self.proj.add_module('dropout_' + str(i), nn.Dropout(p=0.5))
                self.proj.add_module('relu_' + str(i), nn.ReLU(inplace=True))
            self.proj.add_module('proj_final', nn.Linear(self.emb_size, self.K))

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        model_dict = self.state_dict()
        state_dict = torch.load(model_path)
        exist_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(exist_state_dict)
        self.load_state_dict(model_dict)
        logging.info('Load model from ' + model_path)

    @staticmethod
    def similarity(a, b):  # cosine similarity
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return (a * b).sum(dim=-1)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # bsz, -1
        history = feed_dict['history_items']  # bsz, max_his
        lengths = feed_dict['lengths']  # bsz
        batch_size, seq_len = history.shape

        out_dict = dict()
        if self.stage == 'pretrain':  # pretrain extractor
            interest_vectors = self.interest_extractor(history, lengths)  # bsz, K, emb
            i_vectors = self.interest_extractor.i_embeddings(i_ids)
            if feed_dict['phase'] == 'train':
                target_vector = i_vectors[:, 0]  # bsz, emb
                target_intent = (interest_vectors * target_vector[:, None, :]).sum(-1)  # bsz, K
                idx_select = target_intent.max(-1)[1]  # bsz
                user_vector = interest_vectors[torch.arange(batch_size), idx_select, :]  # bsz, emb
                prediction = (user_vector[:, None, :] * i_vectors).sum(-1)
            else:
                prediction = (interest_vectors[:, None, :, :] * i_vectors[:, :, None, :]).sum(-1)  # bsz, -1, K
                prediction = prediction.max(-1)[0]  # bsz, -1
        else:  # finetune
            interest_vectors = self.interest_extractor(history, lengths)  # bsz, K, emb
            i_vectors = self.interest_extractor.i_embeddings(i_ids)
            his_vector = self.interest_predictor(history, lengths)
            pred_intent = self.proj(his_vector)  # bsz, K
            if feed_dict['phase'] == 'train':
                target_vector = i_vectors[:, 0]  # bsz, emb
                target_intent = self.similarity(interest_vectors, target_vector.unsqueeze(1))  # bsz, K
                out_dict['pred_intent'] = pred_intent
                out_dict['target_intent'] = target_intent
                self.check_list.append(('pred_intent', pred_intent.softmax(-1)))
                self.check_list.append(('target_intent', target_intent.softmax(-1)))
            user_vector = (interest_vectors * pred_intent.softmax(-1)[:, :, None]).sum(-2)  # bsz, emb
            prediction = (user_vector[:, None, :] * i_vectors).sum(-1)
        out_dict['prediction'] = prediction.view(batch_size, -1)

        return out_dict

    def loss(self, out_dict: dict):
        if self.stage == 'pretrain':  # pretrain
            loss = super().loss(out_dict)
        else:  # finetune
            pred_intent = out_dict['pred_intent'] / self.temp
            target_intent = out_dict['target_intent'].detach() / self.temp
            kl_criterion = nn.KLDivLoss(reduction='batchmean')
            loss = kl_criterion(F.log_softmax(pred_intent, dim=1), F.softmax(target_intent, dim=1))
            loss = super().loss(out_dict) + self.temp * self.temp * loss
        return loss


class MultiInterestExtractor(nn.Module):
    def __init__(self, k, item_num, emb_size, attn_size, max_his, add_pos, add_trm):
        super(MultiInterestExtractor, self).__init__()
        self.max_his = max_his
        self.add_pos = add_pos
        self.add_trm = add_trm

        self.i_embeddings = nn.Embedding(item_num, emb_size)
        if self.add_pos:
            self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.W1 = nn.Linear(emb_size, attn_size)
        self.W2 = nn.Linear(attn_size, k)
        if self.add_trm:
            self.transformer = layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=1, kq_same=False)

    def forward(self, history, lengths):
        batch_size, seq_len = history.shape
        valid_his = (history > 0).long()

        his_vectors = self.i_embeddings(history)

        if self.add_pos:
            len_range = torch.from_numpy(np.arange(self.max_his)).to(history.device)
            position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his
            pos_vectors = self.p_embeddings(position)
            his_vectors = his_vectors + pos_vectors

        if self.add_trm:
            attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
            his_vectors = self.transformer(his_vectors, attn_mask)
            his_vectors = his_vectors * valid_his[:, :, None].float()

        # Multi-Interest Extraction
        attn_score = self.W2(self.W1(his_vectors).tanh())  # bsz, his_max, K
        attn_score = attn_score.masked_fill(valid_his.unsqueeze(-1) == 0, -np.inf)
        attn_score = attn_score.transpose(-1, -2)  # bsz, K, his_max
        attn_score = (attn_score - attn_score.max()).softmax(dim=-1)
        attn_score = attn_score.masked_fill(torch.isnan(attn_score), 0)
        interest_vectors = (his_vectors[:, None, :, :] * attn_score[:, :, :, None]).sum(-2)  # bsz, K, emb

        return interest_vectors


class InterestPredictor(nn.Module):
    def __init__(self, item_num, emb_size):
        super(InterestPredictor, self).__init__()
        self.i_embeddings = nn.Embedding(item_num + 1, emb_size)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=emb_size, batch_first=True)

    def forward(self, history, lengths):
        his_vectors = self.i_embeddings(history)
        sort_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
        sort_seq = his_vectors.index_select(dim=0, index=sort_idx)
        seq_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_seq, sort_lengths.cpu(), batch_first=True)
        output, hidden = self.rnn(seq_packed, None)
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
        his_vector = hidden[-1].index_select(dim=0, index=unsort_idx)
        return his_vector
