# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import os
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.BaseModel import SequentialModel
from utils import layers


class TiMiRec(SequentialModel):
    runner = 'TiMiRunner'
    extra_log_args = ['emb_size', 'attn_size', 'K', 'temp', 'stage']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--attn_size', type=int, default=8,
                            help='Size of attention vectors.')
        parser.add_argument('--K', type=int, default=2,
                            help='Number of hidden intent.')
        parser.add_argument('--temp', type=float, default=1,
                            help='Temperature in knowledge distillation loss.')
        parser.add_argument('--stage', type=int, default=1,
                            help='Stage of training: 1-pretrain, 2-joint_finetune, 3-detach_finetune, 4-new_idea.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.attn_size = args.attn_size
        self.K = args.K
        self.temp = args.temp
        self.stage = args.stage
        self.max_his = args.history_max
        super().__init__(args, corpus)

        self.pretrain_path = '../model/TiMiRec/Pre__{}__emb_size={}__K={}.pt' \
            .format(corpus.dataset, self.emb_size, self.K)
        if self.stage == 1:
            self.model_path = self.pretrain_path

    def _define_params(self):
        self.interest_extractor = MultiInterestExtractor(
            self.K, self.item_num, self.emb_size, self.attn_size, self.max_his)
        if self.stage in [2, 3]:
            self.intent_predictor = IntentPredictor(
                self.K, self.item_num, self.emb_size, self.max_his)

    def actions_before_train(self):
        if self.stage > 1 and os.path.exists(self.pretrain_path):
            model_dict = self.state_dict()
            pretrained_dict = torch.load(self.pretrain_path)
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.load_state_dict(model_dict)
            return
        logging.info('Train from scratch!')

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # bsz, -1
        history = feed_dict['history_items']  # bsz, max_his
        lengths = feed_dict['lengths']  # bsz
        batch_size, seq_len = history.shape

        interest_vectors = self.interest_extractor(history, lengths)  # bsz, K, emb
        i_vectors = self.interest_extractor.i_embeddings(i_ids)
        target_vector = i_vectors[:, 0]  # bsz, emb
        target_intent = (interest_vectors * target_vector[:, None, :]).sum(-1)  # bsz, K

        out_dict = dict()
        out_dict['target_intent'] = target_intent
        if self.stage == 1:  # pretrain
            if feed_dict['phase'] == 'train':
                idx_select = target_intent.max(-1)[1]  # bsz
                user_vector = interest_vectors[torch.arange(batch_size), idx_select, :]  # bsz, emb
                prediction = (user_vector[:, None, :] * i_vectors).sum(-1)
            else:
                prediction = (interest_vectors[:, None, :, :] * i_vectors[:, :, None, :]).sum(-1)  # bsz, -1, K
                prediction = prediction.max(-1)[0]  # bsz, -1
        elif self.stage in [2, 3]:  # finetune
            pred_intent = self.intent_predictor(history, lengths)  # bsz, K
            if feed_dict['phase'] == 'train':
                idx_select = pred_intent.max(-1)[1]  # bsz
                user_vector = interest_vectors[torch.arange(batch_size), idx_select, :]  # bsz, emb
                # user_vector = (interest_vectors * pred_intent.detach().softmax(-1)[:, :, None]).sum(-2)
                out_dict['pred_intent'] = pred_intent
                out_dict['epoch'] = feed_dict['epoch']
                self.check_list.append(('intent', pred_intent.softmax(-1)))
                self.check_list.append(('target', target_intent.softmax(-1)))
            else:
                user_vector = (interest_vectors * pred_intent.softmax(-1)[:, :, None]).sum(-2)  # bsz, emb
            prediction = (user_vector[:, None, :] * i_vectors).sum(-1)
        else:
            user_vector = interest_vectors.mean(-2)
            prediction = (user_vector[:, None, :] * i_vectors).sum(-1)
        out_dict['prediction'] = prediction.view(batch_size, -1)

        return out_dict

    def loss(self, out_dict: dict):
        if self.stage == 1:
            loss = super().loss(out_dict)
        elif self.stage == 2:
            pred_intent = out_dict['pred_intent'] / self.temp
            target_intent = out_dict['target_intent'].detach() / self.temp
            kl_criterion = nn.KLDivLoss(reduction='batchmean')
            loss = kl_criterion(F.log_softmax(pred_intent, dim=1), F.softmax(target_intent, dim=1))
            loss = super().loss(out_dict) + self.temp * self.temp * loss
        elif self.stage == 3:
            if out_dict['epoch'] % 2:  # the first epoch is 1
                pred_intent = out_dict['pred_intent'] / self.temp
                target_intent = out_dict['target_intent'].detach() / self.temp
                kl_criterion = nn.KLDivLoss(reduction='batchmean')
                loss = kl_criterion(F.log_softmax(pred_intent, dim=1), F.softmax(target_intent, dim=1))
                loss = self.temp * self.temp * loss
            else:
                loss = super().loss(out_dict)
        else:
            pred_intent = out_dict['target_intent']
            target_intent = out_dict['target_intent'].detach() / 1e6
            kl_criterion = nn.KLDivLoss(reduction='batchmean')
            loss = kl_criterion(F.log_softmax(pred_intent, dim=1), F.softmax(target_intent, dim=1))
            loss = super().loss(out_dict) + loss
        return loss


class MultiInterestExtractor(nn.Module):
    def __init__(self, k, item_num, emb_size, attn_size, max_his):
        super(MultiInterestExtractor, self).__init__()
        self.max_his = max_his
        self.i_embeddings = nn.Embedding(item_num, emb_size)
        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.W1 = nn.Linear(emb_size, attn_size)
        self.W2 = nn.Linear(attn_size, k)
        self.transformer = layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=1, kq_same=False)

    def forward(self, history, lengths):
        batch_size, seq_len = history.shape
        valid_his = (history > 0).long()
        len_range = torch.from_numpy(np.arange(self.max_his)).to(history.device)
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his

        his_vectors = self.i_embeddings(history)
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors

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


# class IntentPredictor(nn.Module):
#     def __init__(self, k, item_num, emb_size, max_his):
#         super(IntentPredictor, self).__init__()
#         self.max_his = max_his
#         self.i_embeddings = nn.Embedding(item_num, emb_size)
#         self.rnn = nn.GRU(input_size=emb_size, hidden_size=emb_size, batch_first=True)
#         self.proj = nn.Linear(emb_size, k)
#
#     def forward(self, history, lengths):
#         his_vectors = self.i_embeddings(history)
#
#         # RNN
#         sort_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
#         sort_seq = his_vectors.index_select(dim=0, index=sort_idx)
#         seq_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_seq, sort_lengths, batch_first=True)
#         output, hidden = self.rnn(seq_packed, None)
#         unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
#         his_vector = hidden[-1].index_select(dim=0, index=unsort_idx)
#
#         intent_pred = self.proj(his_vector)  # bsz, K
#         return intent_pred


class IntentPredictor(nn.Module):
    def __init__(self, k, item_num, emb_size, max_his):
        super(IntentPredictor, self).__init__()
        self.max_his = max_his
        self.i_embeddings = nn.Embedding(item_num, emb_size)

        self.p_embeddings = nn.Embedding(max_his + 1, emb_size)
        self.transformer = layers.TransformerLayer(d_model=emb_size, d_ff=emb_size, n_heads=1, kq_same=False)

        # TODO: 预测层结构
        self.proj = nn.Linear(emb_size, k)

    def forward(self, history, lengths):
        batch_size, seq_len = history.shape
        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)

        # Self-attention
        # TODO: 其他模型结构
        len_range = torch.from_numpy(np.arange(self.max_his)).to(history.device)
        position = (lengths[:, None] - len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors
        attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        his_vectors = self.transformer(his_vectors, attn_mask)
        his_vectors = his_vectors * valid_his[:, :, None].float()
        his_vector = his_vectors.sum(1) / lengths[:, None].float()
        # his_vector = his_vectors[torch.arange(batch_size), lengths - 1, :]

        intent_pred = self.proj(his_vector)  # bsz, K
        return intent_pred



