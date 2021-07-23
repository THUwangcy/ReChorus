# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.BaseModel import SequentialModel
from utils import layers


class TiMiRec(SequentialModel):
    extra_log_args = ['emb_size', 'attn_size', 'K', 'temp']

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
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.attn_size = args.attn_size
        self.K = args.K
        self.temp = args.temp
        self.num_layers, self.num_heads = 1, 2
        self.max_his = args.history_max
        super().__init__(args, corpus)

        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.i_embeddings_ = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings_ = nn.Embedding(self.max_his + 1, self.emb_size)
        self.W1 = nn.Linear(self.emb_size, self.attn_size)
        self.W2 = nn.Linear(self.attn_size, self.K)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                                    dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])
        # TODO: 预测层结构
        self.proj = nn.Linear(self.emb_size, self.K)
        # self.proj = nn.Sequential(
        #     nn.Linear(self.emb_size, 32, bias=False),
        #     nn.ReLU(inplace=True), nn.Linear(32, self.K, bias=True)
        # )

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long()
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his

        his_vectors = self.i_embeddings(history)
        pos_vectors = self.p_embeddings(position)
        his_vectors = his_vectors + pos_vectors

        # TODO: 是否共享底层参数
        his_vectors_intent = self.i_embeddings_(history)
        pos_vectors_intent = self.p_embeddings_(position)
        his_vectors_intent = his_vectors_intent + pos_vectors_intent

        # Self-attention
        # TODO: 多兴趣聚合前是否先加一层transformer
        # attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        # for block in self.transformer_block:
        #     his_vectors = block(his_vectors, attn_mask)
        # his_vectors = his_vectors * valid_his[:, :, None].float()

        # Multi-Interest Extraction
        attn_score = self.W2(self.W1(his_vectors).tanh())  # bsz, his_max, K
        attn_score = attn_score.masked_fill(valid_his.unsqueeze(-1) == 0, -np.inf)
        attn_score = attn_score.transpose(-1, -2)  # bsz, K, his_max
        attn_score = (attn_score - attn_score.max()).softmax(dim=-1)
        attn_score = attn_score.masked_fill(torch.isnan(attn_score), 0)
        interest_vectors = (his_vectors[:, None, :, :] * attn_score[:, :, :, None]).sum(-2)  # bsz, K, emb

        # Intent Prediction
        # TODO: 其他模型结构
        attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            his_vectors_intent = block(his_vectors_intent, attn_mask)
        his_vectors_intent = his_vectors_intent * valid_his[:, :, None].float()
        his_vector_intent = his_vectors_intent.sum(1) / lengths[:, None].float()
        intent_pred = self.proj(his_vector_intent)  # bsz, K
        # intent_pred = (interest_vectors * his_vector_intent[:, None, :]).sum(-1)  # bsz, K

        i_vectors = self.i_embeddings(i_ids)
        out_dict = dict()
        if feed_dict['phase'] == 'train':
            target_vector = i_vectors[:, 0]  # bsz, emb
            target_pred = (interest_vectors * target_vector[:, None, :]).sum(-1)  # bsz, K
            # TODO: 训练时取max还是softmax加权
            idx_select = target_pred.max(-1)[1]  # bsz
            user_vector = interest_vectors[torch.arange(batch_size), idx_select, :]  # bsz, emb
            prediction = (user_vector[:, None, :] * i_vectors).sum(-1)
            out_dict['intent_pred'] = intent_pred
            out_dict['target_pred'] = target_pred
            self.check_list.append(('intent', intent_pred.softmax(-1)))
            self.check_list.append(('target', target_pred.softmax(-1)))
        else:
            # ComiRec
            # prediction = (interest_vectors[:, None, :, :] * i_vectors[:, :, None, :]).sum(-1)  # bsz, -1, K
            # prediction = prediction.max(-1)[0]  # bsz, -1

            # TiMiRec-Oracle
            # target_vector = i_vectors[:, 0]  # bsz, emb
            # intent_pred = (interest_vectors * target_vector[:, None, :]).sum(-1)  # bsz, K

            # TiMiRec
            user_vector = (interest_vectors * intent_pred.softmax(-1)[:, :, None]).sum(-2)  # bsz, emb
            prediction = (user_vector[:, None, :] * i_vectors).sum(-1)
        out_dict['prediction'] = prediction.view(batch_size, -1)

        return out_dict

    def loss(self, out_dict: dict):
        rec_loss = super().loss(out_dict)
        intent_pred = out_dict['intent_pred'] / self.temp
        target_pred = out_dict['target_pred'] / self.temp
        kl_criterion = nn.KLDivLoss(reduction='batchmean')
        kd_loss = kl_criterion(F.log_softmax(intent_pred, dim=1), F.softmax(target_pred, dim=1))
        # TODO: two-stage训练
        return rec_loss + self.temp * self.temp * kd_loss