# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" BUIR
Reference:
    "Bootstrapping User and Item Representations for One-Class Collaborative Filtering"
    Lee et al., SIGIR'2021.
CMD example:
    python main.py --model_name BUIR --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import GeneralModel


class BUIR(GeneralModel):
    reader = 'BaseReader'
    runner = 'BUIRRunner'
    extra_log_args = ['emb_size', 'momentum']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--momentum', type=float, default=0.995,
                            help='Momentum update.')
        return GeneralModel.parse_model_args(parser)

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif 'Embedding' in str(type(m)):
            nn.init.xavier_normal_(m.weight.data)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.momentum = args.momentum
        self._define_params()
        self.apply(self.init_weights)

        for param_o, param_t in zip(self.user_online.parameters(), self.user_target.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
        for param_o, param_t in zip(self.item_online.parameters(), self.item_target.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False


    def _define_params(self):
        self.user_online = nn.Embedding(self.user_num, self.emb_size)
        self.user_target = nn.Embedding(self.user_num, self.emb_size)
        self.item_online = nn.Embedding(self.item_num, self.emb_size)
        self.item_target = nn.Embedding(self.item_num, self.emb_size)
        self.predictor = nn.Linear(self.emb_size, self.emb_size)
        self.bn = nn.BatchNorm1d(self.emb_size, eps=0, affine=False, track_running_stats=False)

    # will be called by BUIRRunner
    def _update_target(self):
        for param_o, param_t in zip(self.user_online.parameters(), self.user_target.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

        for param_o, param_t in zip(self.item_online.parameters(), self.item_target.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def forward(self, feed_dict):
        self.check_list = []
        user, items = feed_dict['user_id'], feed_dict['item_id']

        # prediction = (self.item_online(items) * self.user_online(user)[:, None, :]).sum(-1)
        prediction = (self.predictor(self.item_online(items)) * self.user_online(user)[:, None, :]).sum(dim=-1) + \
                     (self.predictor(self.user_online(user))[:, None, :] * self.item_online(items)).sum(dim=-1)
        out_dict = {'prediction': prediction}

        if feed_dict['phase'] == 'train':
            u_online = self.user_online(user)
            u_online = self.predictor(u_online)
            u_target = self.user_target(user)
            i_online = self.item_online(items).squeeze(1)
            i_online = self.predictor(i_online)
            i_target = self.item_target(items).squeeze(1)
            out_dict.update({
                'u_online': u_online,
                'u_target': u_target,
                'i_online': i_online,
                'i_target': i_target
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
