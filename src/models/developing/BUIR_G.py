# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

from models.BaseModel import GeneralModel
from models.general.LightGCN import LightGCN
from models.general.LightGCN import LGCNEncoder


class BUIR_G(GeneralModel):
    runner = 'MoRunner'
    extra_log_args = ['emb_size', 'momentum']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--momentum', type=float, default=0.995,
                            help='Momentum update.')
        parser.add_argument('--n_layers', type=int, default=3,
                            help='Number of LightGCN layers.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.momentum = args.momentum
        self.n_layers = args.n_layers
        self.norm_adj = LightGCN.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
        super().__init__(args, corpus)

    def _define_params(self):
        self.online_encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers)
        self.target_encoder = LGCNEncoder(self.user_num, self.item_num, self.emb_size, self.norm_adj, self.n_layers)
        self.predictor = nn.Linear(self.emb_size, self.emb_size)

    def actions_before_train(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
        super().actions_before_train()

    # will be called by MoRunner
    def _update_target(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']

        u_online, i_online = self.online_encoder(user, items)
        u_target, i_target = self.target_encoder(user, items)
        prediction = (self.predictor(i_online) * u_online[:, None, :]).sum(dim=-1) + \
                     (self.predictor(u_online)[:, None, :] * i_online).sum(dim=-1)
        out_dict = {'prediction': prediction}

        if feed_dict['phase'] == 'train':
            out_dict.update({
                'u_online': self.predictor(u_online),
                'u_target': u_target,
                'i_online': self.predictor(i_online).squeeze(1),
                'i_target': i_target.squeeze(1)
            })
        return out_dict

    def loss(self, output):
        u_online, u_target = output['u_online'], output['u_target']
        i_online, i_target = output['i_online'], output['i_target']

        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)

        # Euclidean distance between normalized vectors can be replaced with their negative inner product
        loss_ui = 2 - 2 * (u_online * i_target.detach()).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target.detach()).sum(dim=-1)

        return (loss_ui + loss_iu).mean()

    class Dataset(GeneralModel.Dataset):
        # No need to sample negative items
        def actions_before_epoch(self):
            self.data['neg_items'] = [[] for _ in range(len(self))]
